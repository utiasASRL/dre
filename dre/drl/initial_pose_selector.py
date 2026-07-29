#!/usr/bin/env python3
"""Interactive pose selector for initialization. Click on the map to set initial pose."""

import struct
import numpy as np
import matplotlib.pyplot as plt
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as R

from dre.msg import DRLEstimate


def load_voxel_map(file_path):
    """Load voxel map and return resolution and voxels."""
    voxels = {}
    with open(file_path, "rb") as f:
        res, = struct.unpack("<d", f.read(8))
        num_poses, = struct.unpack("<I", f.read(4))
        num_voxels, = struct.unpack("<I", f.read(4))
        pose_fmt = struct.Struct("<idddd")
        for _ in range(num_poses):
            pose_fmt.unpack(f.read(pose_fmt.size))
        vox_fmt = struct.Struct("<iid")
        for _ in range(num_voxels):
            data = f.read(vox_fmt.size)
            if len(data) < vox_fmt.size:
                break
            x, y, intensity = vox_fmt.unpack(data)
            voxels[(x, y)] = intensity
    return res, voxels


class InitialPoseSelectorNode(Node):
    def __init__(self):
        super().__init__("initial_pose_selector")

        # Load config
        self.load_config()

        # Publisher for initial pose (latched so late subscribers get the message)
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pose_pub_ = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", qos
        )

        self.fig_ = None
        self.shutdown_requested_ = False

        # This node's purpose is served once loc_node has consumed the initial
        # pose and started producing estimates, so shut down as soon as that
        # happens rather than guessing about subscriber timing.
        self.estimate_sub_ = self.create_subscription(
            DRLEstimate, "/drl_estimate", self.on_drl_estimate_received, 10
        )

        # Check if interactive selector should be skipped
        if self.skip_selector_:
            self.publish_default_pose()
            return

        # Subscribe so the interactive window closes as soon as a pose is
        # published, whether via a click here or from another node.
        self.pose_sub_ = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.on_initial_pose_received, qos
        )

        # Load map
        self.get_logger().info(f"Loading voxel map: {self.map_path_}")
        self.res_, self.voxels_ = load_voxel_map(self.map_path_)
        self.get_logger().info(f"Loaded {len(self.voxels_)} voxels, res={self.res_:.3f} m")

        # Setup map extent for coordinate conversion
        # Flip y for correct orientation, matching map_viz_node/loc_viz_node
        keys = np.asarray(list(self.voxels_.keys()), dtype=np.int32)
        self.ix_min_ = keys[:, 0].min()
        self.ix_max_ = keys[:, 0].max()
        self.iy_min_ = -keys[:, 1].max()
        self.iy_max_ = -keys[:, 1].min()

        # Create interactive plot
        self.fig_, self.ax_ = plt.subplots(figsize=(10, 10))
        self.fig_.suptitle("Click on map to set initial pose (x, y)\nClose window when done")

        # Rasterize map (y index flipped to match the flipped extent above)
        vals = np.asarray([self.voxels_.get((x, -y), 0.0) for x in range(self.ix_min_, self.ix_max_ + 1)
                          for y in range(self.iy_min_, self.iy_max_ + 1)], dtype=np.float32)
        vals = np.clip(vals, 0.0, 0.6) / 0.3
        img = vals.reshape((self.ix_max_ - self.ix_min_ + 1, self.iy_max_ - self.iy_min_ + 1))

        extent = [self.ix_min_ * self.res_, (self.ix_max_ + 1) * self.res_,
                  self.iy_min_ * self.res_, (self.iy_max_ + 1) * self.res_]

        self.ax_.imshow(img.T, origin="lower", cmap="magma_r", vmin=0.0, vmax=1.0,
                       extent=extent, interpolation="nearest")
        self.ax_.set_xlim(extent[0], extent[1])
        self.ax_.set_ylim(extent[2], extent[3])
        self.ax_.set_aspect("equal")
        self.ax_.set_xlabel("X (m)")
        self.ax_.set_ylabel("Y (m)")

        # Store extent for coordinate conversion
        self.extent_ = extent

        # Connect click and motion events
        self.fig_.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig_.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.selected_pose_ = None
        self.selecting_orientation_ = False
        self.orientation_arrow_ = None

        self.get_logger().info("Interactive pose selector ready. Click on the map to select position, then drag to set orientation.")

        # plt.show() blocks the main thread, so pump ROS callbacks on a timer
        # tied to the figure's event loop while the window is open.
        self.ros_timer_ = self.fig_.canvas.new_timer(interval=100)
        self.ros_timer_.add_callback(lambda: rclpy.spin_once(self, timeout_sec=0))
        self.ros_timer_.start()

        plt.show()

    def on_initial_pose_received(self, msg):
        """Close the interactive window once an initial pose has been published."""
        if self.fig_ is not None:
            plt.close(self.fig_)
            self.fig_ = None

    def on_drl_estimate_received(self, msg):
        """Shut down once loc_node has produced an estimate from the initial pose."""
        if self.shutdown_requested_:
            return
        self.shutdown_requested_ = True
        self.get_logger().info("Received /drl_estimate, loc initialized. Shutting down.")
        if self.fig_ is not None:
            plt.close(self.fig_)
            self.fig_ = None

    def publish_default_pose(self):
        """Publish default initial pose without interactive selection."""
        self.get_logger().info(
            f"Skipping interactive selector. Publishing default pose: "
            f"x={self.initial_x_:.2f}, y={self.initial_y_:.2f}, theta={self.initial_theta_:.2f}"
        )

        # Convert yaw to quaternion
        quat = R.from_euler("z", self.initial_theta_).as_quat()

        # Publish pose
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.pose.position = Point(x=self.initial_x_, y=self.initial_y_, z=0.0)
        pose_msg.pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

        # Set covariance
        pose_msg.pose.covariance[0] = 0.25    # x variance
        pose_msg.pose.covariance[7] = 0.25    # y variance
        pose_msg.pose.covariance[35] = 0.1    # theta variance

        self.pose_pub_.publish(pose_msg)
        self.get_logger().info("Default pose published to /initialpose")

    def load_config(self):
        """Load configuration from config_loc.yaml in the installed package share dir."""
        config_file = get_package_share_directory("dre") + "/config/config_loc.yaml"

        self.get_logger().info(f"Reading from config file: {config_file}")
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        self.map_path_ = config["map_path"]

        # Load initial pose configuration
        initial_pose_cfg = config.get("initial_pose", {})
        self.initial_x_ = initial_pose_cfg.get("x", 0.0)
        self.initial_y_ = initial_pose_cfg.get("y", 0.0)
        self.initial_theta_ = initial_pose_cfg.get("theta", 0.0)
        self.skip_selector_ = initial_pose_cfg.get("skip_selector", False)

    def on_click(self, event):
        """Handle mouse click on the map."""
        if event.xdata is None or event.ydata is None:
            return

        x_world = event.xdata
        y_world = event.ydata

        if not self.selecting_orientation_:
            # First click: set position (in flipped plot space) and enter orientation selection mode
            self.get_logger().info(f"Position selected: x={x_world:.2f}, y={-y_world:.2f}")
            self.get_logger().info("Now drag mouse to set orientation, then click to confirm")
            self.selected_pose_ = (x_world, y_world)
            self.selecting_orientation_ = True

            # Draw marker at selected location
            self.ax_.plot(x_world, y_world, "r*", markersize=20, markeredgecolor="white", markeredgewidth=1.5)
            self.fig_.canvas.draw_idle()
        else:
            # Second click: confirm orientation and publish
            x_plot, y_plot = self.selected_pose_

            # Calculate yaw from mouse position relative to selected point (plot space)
            dx = x_world - x_plot
            dy = y_world - y_plot
            yaw_plot = np.arctan2(dy, dx)

            # Map is displayed y-flipped, so convert back to the true world frame
            x_pos = x_plot
            y_pos = -y_plot
            yaw = -yaw_plot

            self.get_logger().info(f"Orientation confirmed: yaw={np.degrees(yaw):.1f}°")

            # Convert yaw to quaternion
            quat = R.from_euler("z", yaw).as_quat()

            # Publish pose
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "map"

            pose_msg.pose.pose.position = Point(x=x_pos, y=y_pos, z=0.0)
            pose_msg.pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

            # Set covariance (rough estimate for initialization)
            pose_msg.pose.covariance[0] = 0.25    # x variance
            pose_msg.pose.covariance[7] = 0.25    # y variance
            pose_msg.pose.covariance[35] = 0.1    # theta variance

            self.pose_pub_.publish(pose_msg)

            # Reset state
            self.selecting_orientation_ = False
            if self.orientation_arrow_ is not None:
                self.orientation_arrow_.remove()
                self.orientation_arrow_ = None
            self.fig_.canvas.draw_idle()

            self.get_logger().info("Pose published to /initialpose")

    def on_motion(self, event):
        """Handle mouse motion to update orientation arrow."""
        if not self.selecting_orientation_ or event.xdata is None or event.ydata is None:
            return

        x_pos, y_pos = self.selected_pose_
        x_mouse = event.xdata
        y_mouse = event.ydata

        # Remove previous arrow if it exists
        if self.orientation_arrow_ is not None:
            self.orientation_arrow_.remove()

        # Draw arrow from selected position to mouse position
        arrow_length = 0.5  # length in meters
        dx = x_mouse - x_pos
        dy = y_mouse - y_pos
        dist = np.sqrt(dx**2 + dy**2)

        if dist > 0.01:  # Only draw if mouse is far enough from position
            # Normalize and scale arrow
            dx_norm = (dx / dist) * arrow_length
            dy_norm = (dy / dist) * arrow_length

            self.orientation_arrow_ = self.ax_.arrow(
                x_pos, y_pos, dx_norm, dy_norm,
                head_width=0.2, head_length=0.1, fc="cyan", ec="cyan", alpha=0.7
            )

            # Calculate and display yaw angle (negate to show the true world-frame yaw)
            yaw = -np.degrees(np.arctan2(dy, dx))
            self.fig_.suptitle(f"Click on map to set initial pose (x, y)\nDrag to set orientation | Yaw: {yaw:.1f}°")

        self.fig_.canvas.draw_idle()


def main(args=None):
    rclpy.init(args=args)
    node = InitialPoseSelectorNode()
    while rclpy.ok() and not node.shutdown_requested_:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
