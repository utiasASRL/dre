#!/usr/bin/env python3
"""Interactive pose selector for initialization. Click on the map to set initial pose."""

import os
import struct
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R

from dre.msg import DRLEstimate

# init_mode="auto": pause between chunk publishes so they're visible one-by-one in
# rviz instead of flashing by faster than the eye can follow.
AUTO_CHUNK_PUBLISH_DELAY_S = 0.05

# init_mode="auto": skip a mapping-node pose if a chunk already exists within this
# Euclidean distance (meters), so chunks aren't generated redundantly along a slow
# or looping trajectory.
AUTO_CHUNK_MIN_SPACING_M = 40.0

# init_mode="auto" fake-local-map generation. Hard-coded to match config_dro.yaml's
# `direct:` block (local_map_res / max_local_map_range) for now — TODO: read these
# from config_dro.yaml dynamically once the auto-init pipeline is wired up end to end.
AUTO_LOCAL_MAP_RES = 0.1          # m/pixel, matches DRO's local_map_res
AUTO_MAX_LOCAL_MAP_RANGE = 100.0  # meters (half-width), matches DRO's max_local_map_range


def load_voxel_map(file_path):
    """Load voxel map and return resolution, voxels, and mapping-node poses."""
    voxels = {}
    poses = []
    with open(file_path, "rb") as f:
        res, = struct.unpack("<d", f.read(8))
        num_poses, = struct.unpack("<I", f.read(4))
        num_voxels, = struct.unpack("<I", f.read(4))
        pose_fmt = struct.Struct("<idddd")
        for _ in range(num_poses):
            pose_id, x, y, yaw, ate = pose_fmt.unpack(f.read(pose_fmt.size))
            poses.append((pose_id, x, y, yaw, ate))
        vox_fmt = struct.Struct("<iid")
        for _ in range(num_voxels):
            data = f.read(vox_fmt.size)
            if len(data) < vox_fmt.size:
                break
            x, y, intensity = vox_fmt.unpack(data)
            voxels[(x, y)] = intensity
    return res, voxels, poses


def rasterize_voxel_grid(voxels, res):
    """Dense north-up raster of the whole voxel map: grid[row, col], row 0 = max y, col 0 = min x.

    Returns (grid, ix_min, iy_max) so world (x, y) -> (row, col) via:
        col = floor(x / res) - ix_min
        row = iy_max - floor(y / res)
    """
    keys = np.asarray(list(voxels.keys()), dtype=np.int64)
    ix_min, ix_max = int(keys[:, 0].min()), int(keys[:, 0].max())
    iy_min, iy_max = int(keys[:, 1].min()), int(keys[:, 1].max())

    grid = np.zeros((iy_max - iy_min + 1, ix_max - ix_min + 1), dtype=np.float32)
    for (ix, iy), val in voxels.items():
        grid[iy_max - iy, ix - ix_min] = val
    return grid, ix_min, iy_max


def render_local_map_chunk(grid, ix_min, iy_max, map_res, cx, cy, out_res, half_range_m):
    """Crop a (2*half_range_m/out_res + 1) square, north-up, uint8 chunk centered on (cx, cy).

    The source `grid` is at the voxel map's native `map_res` (typically coarser than
    `out_res`); the crop is upsampled with nearest-neighbor so the output matches the
    pixel footprint of a real DRO local map without inventing detail.
    """
    half_idx_src = int(round(half_range_m / map_res))
    center_col = int(np.floor(cx / map_res)) - ix_min
    center_row = iy_max - int(np.floor(cy / map_res))

    src_size = 2 * half_idx_src + 1
    patch = np.zeros((src_size, src_size), dtype=np.float32)

    row0, row1 = center_row - half_idx_src, center_row + half_idx_src + 1
    col0, col1 = center_col - half_idx_src, center_col + half_idx_src + 1
    grid_h, grid_w = grid.shape
    src_row0, src_row1 = max(row0, 0), min(row1, grid_h)
    src_col0, src_col1 = max(col0, 0), min(col1, grid_w)
    if src_row0 < src_row1 and src_col0 < src_col1:
        patch[src_row0 - row0:src_row1 - row0, src_col0 - col0:src_col1 - col0] = \
            grid[src_row0:src_row1, src_col0:src_col1]

    out_size = 2 * int(round(half_range_m / out_res)) + 1
    resized = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return (np.clip(resized, 0.0, 1.0) * 255.0).astype(np.uint8)


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

        # init_mode="auto": publishes each fake local_map chunk as it's generated, for rviz
        # viewing and for raplace_node to ingest. KEEP_ALL so the burst of chunks (published
        # much faster than raplace's per-image Radon transform can consume them) isn't
        # silently truncated by a shallow KEEP_LAST queue.
        chunk_qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_ALL)
        self.chunk_pub_ = self.create_publisher(Image, "place_candidate", chunk_qos)
        self.bridge_ = CvBridge()

        self.fig_ = None
        self.shutdown_requested_ = False

        # init_mode="auto": maps each published place_candidate chunk's timestamp
        # (microseconds, same key raplace_node/registration_node reference as
        # candidate_time / registration_relative_pose's header.frame_id) to the
        # chunk's known world-frame (map) pose — populated in run_auto_init().
        # Lives entirely in this process; no need to publish it anywhere.
        self.chunk_world_poses_ = {}

        # This node's purpose is served once loc_node has consumed the initial
        # pose and started producing estimates, so shut down as soon as that
        # happens rather than guessing about subscriber timing.
        self.estimate_sub_ = self.create_subscription(
            DRLEstimate, "/drl_estimate", self.on_drl_estimate_received, 10
        )

        # Dispatch on init_mode
        if self.init_mode_ == "start":
            self.publish_start_pose()
            return
        elif self.init_mode_ == "auto":
            self.run_auto_init()
            return
        elif self.init_mode_ != "selector":
            self.get_logger().error(
                f"init_mode '{self.init_mode_}' is not implemented yet "
                "(supported: start, selector, auto). Not publishing an initial pose."
            )
            return

        # Subscribe so the interactive window closes as soon as a pose is
        # published, whether via a click here or from another node.
        self.pose_sub_ = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.on_initial_pose_received, qos
        )

        # Load map
        self.get_logger().info(f"Loading voxel map: {self.map_path_}")
        self.res_, self.voxels_, _ = load_voxel_map(self.map_path_)
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

    def publish_start_pose(self):
        """Publish x=0, y=0, theta=0 in the map frame without interactive selection."""
        self.get_logger().info("init_mode='start'. Publishing x=0, y=0, theta=0 in the map frame.")

        # Convert yaw to quaternion
        quat = R.from_euler("z", 0.0).as_quat()

        # Publish pose
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.pose.position = Point(x=0.0, y=0.0, z=0.0)
        pose_msg.pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

        # Set covariance
        pose_msg.pose.covariance[0] = 0.25    # x variance
        pose_msg.pose.covariance[7] = 0.25    # y variance
        pose_msg.pose.covariance[35] = 0.1    # theta variance

        self.pose_pub_.publish(pose_msg)
        self.get_logger().info("Start pose published to /initialpose")

    def run_auto_init(self):
        """init_mode='auto': render a fake DRO-style local_map image around each mapping-node
        pose in the prior map. This is just the chunking step for now — nothing is fed into
        raplace yet, chunks are written to disk so they can be inspected directly."""
        self.get_logger().info(f"Loading voxel map: {self.map_path_}")
        self.res_, self.voxels_, poses = load_voxel_map(self.map_path_)
        self.get_logger().info(
            f"Loaded {len(self.voxels_)} voxels, {len(poses)} mapping-node poses, res={self.res_:.3f} m"
        )

        grid, ix_min, iy_max = rasterize_voxel_grid(self.voxels_, self.res_)

        out_dir = "auto_init_local_maps"
        if os.path.exists(out_dir):
            for filename in os.listdir(out_dir):
                file_path = os.path.join(out_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        else:
            os.makedirs(out_dir, exist_ok=True)

        # Give rviz (launched alongside this node) time to come up and subscribe
        # to place_candidate before the first chunk goes out, so nothing is missed.
        time.sleep(2.0)

        chunk_centers = np.empty((0, 2), dtype=np.float64)
        num_written = 0
        for pose_id, x, y, yaw, ate in poses:
            if chunk_centers.shape[0] > 0:
                dists = np.hypot(chunk_centers[:, 0] - x, chunk_centers[:, 1] - y)
                if dists.min() < AUTO_CHUNK_MIN_SPACING_M:
                    continue
            chunk_centers = np.vstack([chunk_centers, [x, y]])

            chunk = render_local_map_chunk(
                grid, ix_min, iy_max, self.res_, x, y, AUTO_LOCAL_MAP_RES, AUTO_MAX_LOCAL_MAP_RANGE
            )
            cv2.imwrite(os.path.join(out_dir, f"{pose_id}.png"), chunk)

            stamp = self.get_clock().now().to_msg()
            chunk_msg = self.bridge_.cv2_to_imgmsg(chunk, encoding="mono8")
            chunk_msg.header.stamp = stamp
            chunk_msg.header.frame_id = "map"
            self.chunk_pub_.publish(chunk_msg)

            timestamp_us = int(stamp.sec) * 1_000_000 + int(stamp.nanosec) // 1_000
            self.chunk_world_poses_[timestamp_us] = (x, y, yaw)

            num_written += 1
            time.sleep(AUTO_CHUNK_PUBLISH_DELAY_S)

        out_size = 2 * int(round(AUTO_MAX_LOCAL_MAP_RANGE / AUTO_LOCAL_MAP_RES)) + 1
        self.get_logger().info(
            f"Wrote {num_written}/{len(poses)} fake local_map chunks to '{out_dir}/' "
            f"(skipped poses within {AUTO_CHUNK_MIN_SPACING_M} m of an existing chunk), "
            f"{out_size}x{out_size} px @ {AUTO_LOCAL_MAP_RES} m/px."
        )

    def load_config(self):
        """Load configuration from config_loc.yaml in the installed package share dir."""
        config_file = get_package_share_directory("dre") + "/config/config_loc.yaml"

        self.get_logger().info(f"Reading from config file: {config_file}")
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        self.map_path_ = config["map_path"]
        self.init_mode_ = config.get("init_mode", "start")

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
