#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from message_filters import Subscriber, TimeSynchronizer
from dre.msg import RadarInfo, LocalMapInfo
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
import yaml
import copy
from dro import Dro, kDefaultDroOpts
import os
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
import pandas as pd
import cv2
import time

class DroNode(Node):
    def __init__(self):
        super().__init__('dro_node')
        self.get_logger().info("DroNode has been started.")

        # Subscribe to the imu topic
        self.imu_subscription = self.create_subscription(
            Imu,
            '/boreas/imu',
            self.imuCallback,
            1000)

        # Subscribe synchronously to the image topic and radar info topic
        self.image_subscription = Subscriber(self, Image, '/boreas/radar_image')
        self.radar_info_subscription = Subscriber(self, RadarInfo, '/boreas/radar_info')
        self.ts = TimeSynchronizer([self.image_subscription, self.radar_info_subscription], 20)
        self.ts.registerCallback(self.radarCallback)

        # Set the publisher for the odometry
        self.odometry_publisher = self.create_publisher(Odometry, 'dro_odometry', 10)
        self.local_map_odometry_publisher = self.create_publisher(Odometry, 'dro_local_map_odometry', 10)

        # Set the local map publishers
        self.local_map_image_publisher = self.create_publisher(Image, 'dro_local_map_image', 10)
        self.cumulated_returns_image_publisher = self.create_publisher(Image, 'dro_cumulated_returns_image', 10)
        self.local_map_info_publisher = self.create_publisher(LocalMapInfo, 'dro_local_map_info', 10)

        # Needed for rviz's "radar"-following camera view (XYOrbit Target Frame):
        # view controllers only track TF frames, not topics.
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publish an identity odom->radar transform right away so the "odom" frame
        # exists in TF from startup, instead of rviz spamming "frame does not exist"
        # warnings until the first radar+IMU pair actually arrives. Static (not the
        # regular tf_broadcaster above) because it's a one-shot message: /tf_static
        # is transient-local, so a late-joining subscriber (e.g. rviz starting up
        # concurrently) still gets it, unlike a single message on plain /tf.
        # Kept as self.* (not a local var): the transient-local cached message
        # only survives as long as the publisher itself does.
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        start_transform = TransformStamped()
        # Stamp zero (not wall-clock now()): real data replayed from a dataset carries
        # its own recorded timestamps (e.g. Boreas playback is stamped ~2024), which are
        # earlier than "now". A "now"-stamped placeholder would make every subsequent
        # real transform look like it's from the past, and tf2 would reject it as
        # TF_OLD_DATA. Zero is guaranteed earlier than any real timestamp.
        start_transform.header.stamp.sec = 0
        start_transform.header.stamp.nanosec = 0
        start_transform.header.frame_id = "odom"
        start_transform.child_frame_id = "radar"
        start_transform.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(start_transform)

        # Get the output path from the parameters
        self.declare_parameter('output_path', 'output')
        self.output_path = self.get_parameter('output_path').get_parameter_value().string_value

        self.radar_data_buffer = []
        self.imu_data_buffer = []

        self.last_imu_time = None

        self.initialized = False
        self.first = True

        # Frame processing stats
        self.frame_count = 0
        self.sum_runtime = 0.0
        self.stats_start_time = None

        # Load the config file and populate the DRO options
        config_file_path = "config/config_dro.yaml"
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_file_path = os.path.join(base_path, "share/dre", config_file_path)
        print(f"Loading DRO configuration from {config_file_path}")
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        dro_opts = copy.deepcopy(kDefaultDroOpts)
        if 'estimation' not in config:
            self.get_logger().error("DRO configuration file is missing the 'estimation' section.")
            return
        for section_name, defaults in dro_opts.items():
            if section_name not in config:
                self.get_logger().warn(f"DRO configuration missing section '{section_name}', using defaults.")
                continue
            for key in defaults.keys():
                if key in config[section_name]:
                    if section_name == 'estimation' and key == 'T_axle_radar':
                        dro_opts[section_name][key] = np.array(config[section_name][key], dtype=np.float64)
                    else:
                        dro_opts[section_name][key] = config[section_name][key]


        for section_name, section_cfg in config.items():
            if section_name not in dro_opts:
                self.get_logger().warn(
                    f"DRO configuration section '{section_name}' is not used by defaults and will be ignored."
                )
                continue
            unknown_keys = set(section_cfg.keys()) - set(dro_opts[section_name].keys())
            if unknown_keys:
                self.get_logger().warn(
                    f"DRO configuration section '{section_name}' has unused keys: {sorted(unknown_keys)}"
                )


        self.dro = Dro(dro_opts, self)
        self.dro_opts = dro_opts
        self.bridge = CvBridge()
        self.get_logger().info("DRO ready")


    def initialize(self, radar_data):
        # Create the output folders for the sequence
        seq_ID = radar_data['sequence_id']
        self.seq_output_folder = os.path.join(self.output_path, seq_ID)
        print(f"Creating output folder: {self.seq_output_folder}")
        os.makedirs(self.seq_output_folder, exist_ok=True)

        # Create the odometry output file
        self.odometry_output_path = os.path.join(self.seq_output_folder, "odometry_result")
        if os.path.exists(self.odometry_output_path):
            os.system('rm -r ' + self.odometry_output_path)
        os.makedirs(self.odometry_output_path)
        self.odometry_output_path = os.path.join(self.odometry_output_path, seq_ID + '.txt')

        # If saving local maps, create the folders
        if self.dro_opts['log']['save_local_maps']:
            self.local_map_output_path = os.path.join(self.seq_output_folder, "local_maps")
            if os.path.exists(self.local_map_output_path):
                os.system('rm -r ' + self.local_map_output_path)
            os.makedirs(self.local_map_output_path)

            self.odometry_2d_output_path = os.path.join(self.seq_output_folder, "odometry_2d")
            if os.path.exists(self.odometry_2d_output_path):
                os.system('rm -r ' + self.odometry_2d_output_path)
            os.makedirs(self.odometry_2d_output_path)
            self.odometry_2d_output_path = os.path.join(self.odometry_2d_output_path, seq_ID + '.txt')

            if self.dro_opts['estimation']['use_gyro']:
                self.odometry_3d_output_path = os.path.join(self.seq_output_folder, "odometry_3d")
                if os.path.exists(self.odometry_3d_output_path):
                    os.system('rm -r ' + self.odometry_3d_output_path)
                os.makedirs(self.odometry_3d_output_path)
                self.odometry_3d_output_path = os.path.join(self.odometry_3d_output_path, seq_ID + '.txt')

        # Independent of save_local_maps: create the cumulative-returns folder only
        # if that's separately enabled.
        if self.dro_opts['log']['save_cumulative_image']:
            self.cumulative_return_output_path = os.path.join(self.seq_output_folder, "cumulated_returns")
            if os.path.exists(self.cumulative_return_output_path):
                os.system('rm -r ' + self.cumulative_return_output_path)
            os.makedirs(self.cumulative_return_output_path)

        self.initialized = True


    def radarCallback(self, image_msg, radar_info_msg):
        if self.initialized == False:
            self.initialize({'sequence_id': radar_info_msg.sequence_id})
        polar_image = np.frombuffer(image_msg.data, dtype=np.float32).reshape((image_msg.height, image_msg.width))
        azimuths = np.asarray(radar_info_msg.azimuth, dtype=np.float32)
        timestamps = np.asarray(radar_info_msg.timestamps, dtype=np.int64)
        resolution = radar_info_msg.resolution
        chirps = np.asarray(radar_info_msg.chirps, dtype=np.uint8)

        self.radar_data_buffer.append({
            'polar': polar_image,
            'azimuths': azimuths,
            'timestamps': timestamps,
            'resolution': resolution,
            'chirps': chirps,
            'timestamp': np.int64(image_msg.header.stamp.sec * 1e6) + np.int64(image_msg.header.stamp.nanosec / 1e3)
        })

        self.odometryStepIfReady()

    def imuCallback(self, msg):
        time = np.int64(msg.header.stamp.sec * 1e6) + np.int64(msg.header.stamp.nanosec / 1e3)
        if self.last_imu_time is not None and time <= self.last_imu_time:
            self.get_logger().warn(f"Received out-of-order IMU message. Current time: {time}, Last time: {self.last_imu_time}")
            return
        self.last_imu_time = time
        imu_data = {
            'timestamp': time,
            'angular_velocity': np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]),
            'linear_acceleration': np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        }
        self.imu_data_buffer.append(imu_data)

        self.odometryStepIfReady()

    def odometryStepIfReady(self):
        if len(self.radar_data_buffer) > 10:
            self.radar_data_buffer.pop(0)
            self.get_logger().warn("Radar buffer > 10, likely a problem in with IMU data, dropping oldest radar.")

        # Check if the first IMU is before the first radar timestamps and the last IMU is after the last radar timestamp
        if len(self.radar_data_buffer) == 0 or len(self.imu_data_buffer) == 0:
            return
        first_radar_time = self.radar_data_buffer[0]['timestamps'][0]
        if self.first and self.imu_data_buffer[0]['timestamp'] > self.radar_data_buffer[0]['timestamps'][0]:
            self.imu_data_buffer[0]['timestamp'] = self.radar_data_buffer[0]['timestamps'][0] - 1000
            self.first = False

        last_radar_time = self.radar_data_buffer[0]['timestamps'][-1] + 2000  # Add 2ms to ensure we cover the radar timestamps
        if self.imu_data_buffer[0]['timestamp'] > first_radar_time or self.imu_data_buffer[-1]['timestamp'] < last_radar_time:
            return

        # Get the minimum number of IMU measurements that cover the radar timestamps
        imu_times = np.array([imu['timestamp'] for imu in self.imu_data_buffer])
        start_idx = np.searchsorted(imu_times, first_radar_time, side='left')
        start_idx = max(0, start_idx - 1)  # Ensure at least one IMU before the radar timestamps
        end_idx = np.searchsorted(imu_times, last_radar_time, side='right')
        end_idx = min(len(imu_times), end_idx + 1)  # Ensure at least one IMU after the radar timestamps
        relevant_imus = self.imu_data_buffer[start_idx:end_idx]

        #self.get_logger().info(f"Processing radar scan (from {round(first_radar_time*1e-6, 3)} to {round(last_radar_time*1e-6, 3)}) with {len(relevant_imus)} IMU measurements from {round(imu_times[start_idx]*1e-6, 3)} to {round(imu_times[end_idx-1]*1e-6, 3)}")

        t1 = time.time()
        self.dro.odometryStep(self.radar_data_buffer[0], relevant_imus)
        t2 = time.time()

        if self.frame_count == 0:
            self.stats_start_time = t1
        self.frame_count += 1
        self.sum_runtime += (t2 - t1)

        if self.frame_count % 50 == 0:
            elapsed = t2 - self.stats_start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0.0
            avg_runtime = self.sum_runtime / self.frame_count
            self.get_logger().info(
                f"Frames processed: {self.frame_count} | avg FPS: {avg_fps:.2f} | avg runtime/frame: {avg_runtime:.3f} seconds")


        # Get the odometry results
        current_odometry = self.dro.getPose(self.radar_data_buffer[0]['timestamp'])
        self.publishOdometry(current_odometry, self.radar_data_buffer[0]['timestamp'])
        self.logOdometry(current_odometry, self.radar_data_buffer[0]['timestamp'])


        # Clear the first radar
        temp_last_time = self.radar_data_buffer[0]['timestamps'][-1]
        # Remove the processed radar from the buffer
        self.radar_data_buffer.pop(0)

        # Remove the IMU measurements to keep at least one IMU before the next radar timestamps
        if len(self.radar_data_buffer) > 0:
            next_radar_time = self.radar_data_buffer[0]['timestamps'][0]
        else:
            next_radar_time = temp_last_time
        next_start_idx = np.searchsorted(imu_times, next_radar_time, side='left')
        self.imu_data_buffer = self.imu_data_buffer[next_start_idx - 1:]  # Keep one IMU before the next radar



    def publishOdometry(self, pose, timestamp):
        odom_msg = Odometry()
        odom_msg.header.stamp.sec = int(timestamp // 1e6)
        odom_msg.header.stamp.nanosec = int((timestamp % 1e6) * 1e3)
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "radar"

        # Set position
        odom_msg.pose.pose.position.x = pose[0, 3]
        odom_msg.pose.pose.position.y = pose[1, 3]
        odom_msg.pose.pose.position.z = 0.0  # Assuming planar motion

        # Set orientation (yaw only)
        quaternion = R.from_matrix(pose[:3, :3]).as_quat()  # Convert rotation matrix to quaternion
        odom_msg.pose.pose.orientation.x = quaternion[0]
        odom_msg.pose.pose.orientation.y = quaternion[1]
        odom_msg.pose.pose.orientation.z = quaternion[2]
        odom_msg.pose.pose.orientation.w = quaternion[3]
        self.odometry_publisher.publish(odom_msg)

        # Broadcast the same pose as a TF transform (see the comment on tf_broadcaster above)
        transform = TransformStamped()
        transform.header.stamp = odom_msg.header.stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "radar"
        transform.transform.translation.x = pose[0, 3]
        transform.transform.translation.y = pose[1, 3]
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]
        self.tf_broadcaster.sendTransform(transform)


    def logOdometry(self, pose, timestamp):
        inv_pose = np.linalg.inv(pose)
        data = np.concatenate([timestamp.reshape(1, 1), inv_pose[:3, :].reshape(1, -1)], axis=1)
        df_odom = pd.DataFrame(data)
        df_odom[0] = df_odom[0].astype(np.int64)
        if not os.path.exists(self.odometry_output_path):
            df_odom.to_csv(self.odometry_output_path, header=None, index=None, sep=' ')
        else:
            df_odom.to_csv(self.odometry_output_path, mode='a', header=None, index=None, sep=' ')


    def logOdometry3D(self, poses, timestamps):
        if poses.shape[0] != timestamps.shape[0]:
            self.get_logger().error(f"Number of poses {poses.shape[0]} does not match number of timestamps {timestamps.shape[0]}")
            return
        poses_inv = np.linalg.inv(poses)
        data = np.concatenate([timestamps.reshape(-1, 1), poses_inv[:, :3, :].reshape(poses.shape[0], -1)], axis=1)
        df_odom = pd.DataFrame(data)
        df_odom[0] = df_odom[0].astype(np.int64)
        if not os.path.exists(self.odometry_3d_output_path):
            df_odom.to_csv(self.odometry_3d_output_path, header=None, index=None, sep=' ')
        else:
            df_odom.to_csv(self.odometry_3d_output_path, mode='a', header=None, index=None, sep=' ')


    # True when a subscriber (e.g. mapping_node) wants the cumulative returns image
    def cumulativeReturnsNeeded(self):
        return self.cumulated_returns_image_publisher.get_subscription_count() > 0

    # Input local map needs to be in [0, 255] uint8 format np array
    def publishLocalMap(self, local_map, xy_theta, timestamp):
        if local_map.dtype != np.uint8:
            self.get_logger().error(f"Local map numpy array has dtype {local_map.dtype}, expected uint8.")
            return

        local_map_image_msg = self.bridge.cv2_to_imgmsg(local_map, encoding="mono8")
        local_map_image_msg.header.stamp.sec = int(timestamp // 1e6)
        local_map_image_msg.header.stamp.nanosec = int((timestamp % 1e6) * 1e3)
        local_map_image_msg.header.frame_id = "radar"
        self.local_map_image_publisher.publish(local_map_image_msg)

        map_info_msg = LocalMapInfo()
        map_info_msg.header = local_map_image_msg.header
        map_info_msg.x = xy_theta[0]
        map_info_msg.y = xy_theta[1]
        map_info_msg.theta = xy_theta[2]
        map_info_msg.resolution = self.dro_opts['direct']['local_map_res']
        self.local_map_info_publisher.publish(map_info_msg)

        # Publish the local map odometry
        local_map_odom_msg = Odometry()
        local_map_odom_msg.header = local_map_image_msg.header
        local_map_odom_msg.header.frame_id = "odom"
        local_map_odom_msg.child_frame_id = "radar"
        local_map_odom_msg.pose.pose.position.x = xy_theta[0]
        local_map_odom_msg.pose.pose.position.y = xy_theta[1]
        local_map_odom_msg.pose.pose.position.z = 0.0  # Assuming planar motion
        quaternion = R.from_euler('z', xy_theta[2]).as_quat()  # Convert yaw to quaternion
        local_map_odom_msg.pose.pose.orientation.x = quaternion[0]
        local_map_odom_msg.pose.pose.orientation.y = quaternion[1]
        local_map_odom_msg.pose.pose.orientation.z = quaternion[2]
        local_map_odom_msg.pose.pose.orientation.w = quaternion[3]
        self.local_map_odometry_publisher.publish(local_map_odom_msg)


    # Input cumulated_returns needs to be in [0, 255] uint8 format np array
    def publishCumulativeReturns(self, cumulated_returns, timestamp):
        if cumulated_returns.dtype != np.uint8:
            self.get_logger().error(f"Cumulated returns numpy array has dtype {cumulated_returns.dtype}, expected uint8.")
            return

        cumulated_returns_image_msg = self.bridge.cv2_to_imgmsg(cumulated_returns, encoding="mono8")
        cumulated_returns_image_msg.header.stamp.sec = int(timestamp // 1e6)
        cumulated_returns_image_msg.header.stamp.nanosec = int((timestamp % 1e6) * 1e3)
        cumulated_returns_image_msg.header.frame_id = "radar"
        self.cumulated_returns_image_publisher.publish(cumulated_returns_image_msg)


    # Input local map needs to be in [0, 255] uint8 format np array
    def writeLocalMap(self, local_map, xy_theta, timestamp):
        cv2.imwrite(os.path.join(self.local_map_output_path, str(timestamp) + '.png'), local_map)

        df_data_2d = pd.DataFrame(np.array([timestamp, xy_theta[0], xy_theta[1], xy_theta[2]]).reshape(1, -1))
        df_data_2d[0] = df_data_2d[0].astype(np.int64)
        if not os.path.exists(self.odometry_2d_output_path):
            df_data_2d.to_csv(self.odometry_2d_output_path, header=None, index=None, sep=' ')
        else:
            df_data_2d.to_csv(self.odometry_2d_output_path, mode='a', header=None, index=None, sep=' ')


    # Input cumulated_returns needs to be in [0, 255] uint8 format np array
    def writeCumulativeImage(self, cumulated_returns, timestamp):
        cv2.imwrite(os.path.join(self.cumulative_return_output_path, str(timestamp) + '.png'), cumulated_returns)



if __name__ == '__main__':
    rclpy.init()
    dro_node = DroNode()
    rclpy.spin(dro_node)
    dro_node.destroy_node()
    rclpy.shutdown()
