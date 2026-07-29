#!/usr/bin/env python3
"""ROS2 node: visualizes localization with scan overlay and pose estimate."""

import array
import struct
from collections import deque
from pathlib import Path
from typing import Union

import yaml
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from message_filters import Subscriber, TimeSynchronizer

from dre.msg import DRLEstimate


def _get_config_path() -> Path:
    """Get the path to config_loc.yaml in the installed package share dir."""
    pkg_share_dir = get_package_share_directory("dre")
    return Path(pkg_share_dir) / "config" / "config_loc.yaml"


def _load_config(path: Union[str, Path]) -> dict:
    """Load config from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)




def _load_voxel_map(path: str):
    """Load voxel_map.bin and return (img, extent) ready for imshow.

    Replicates the rasterization from render_localization_video_frames:
      - intensities clipped to [0, 0.6] and rescaled so 0.3 → 1.0
      - y-axis flipped to match visualization convention
    Returns:
        img    : 2-D float array (NaN where no voxel), shape (W, H) for imshow .T
        extent : [xmin, xmax, ymin, ymax] in metres
        res    : voxel resolution in metres
    """
    voxels = {}
    with open(path, "rb") as f:
        res, = struct.unpack("<d", f.read(8))
        num_poses, = struct.unpack("<I", f.read(4))
        num_voxels, = struct.unpack("<I", f.read(4))

        pose_struct = struct.Struct("<idddd")
        for _ in range(num_poses):
            f.read(pose_struct.size)  # skip poses — not needed for background

        voxel_struct = struct.Struct("<iid")
        for _ in range(num_voxels):
            data = f.read(voxel_struct.size)
            if len(data) < voxel_struct.size:
                break
            x, y, intensity = voxel_struct.unpack(data)
            voxels[(x, y)] = intensity

    keys = np.asarray(list(voxels.keys()), dtype=np.int32)
    vals = np.asarray(list(voxels.values()), dtype=np.float32)

    vals = np.clip(vals, 0.0, 0.6)
    vals = vals / 0.3  # matches render_localization_video_frames normalisation

    ix, iy = keys[:, 0], keys[:, 1]
    iy = -iy  # flip y for correct orientation

    ix_min, ix_max = ix.min(), ix.max()
    iy_min, iy_max = iy.min(), iy.max()

    img = np.full((ix_max - ix_min + 1, iy_max - iy_min + 1), np.nan)
    img[ix - ix_min, iy - iy_min] = vals

    extent = [
        ix_min * res,
        (ix_max + 1) * res,
        iy_min * res,
        (iy_max + 1) * res,
    ]
    return img, extent


def _quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def _triangle_pts(x, y, yaw, veh_len, veh_wid):
    pts = np.array([
        [ veh_len / 2,  0.0],
        [-veh_len / 2,  veh_wid / 2],
        [-veh_len / 2, -veh_wid / 2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    pts = pts @ np.array([[c, -s], [s, c]]).T
    pts[:, 0] += x
    pts[:, 1] += y
    return pts


def _ros_image_to_gray(msg: Image) -> np.ndarray:
    """Convert a sensor_msgs/Image to a 2-D uint8 array.

    Handles row padding (msg.step may be wider than msg.width * channels).
    """
    enc = msg.encoding.lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)

    if enc in ("mono8", "8uc1"):
        return raw[:, : msg.width].copy()

    n_ch = msg.step // msg.width
    arr = raw[:, : msg.width * n_ch].reshape(msg.height, msg.width, n_ch)
    if enc == "bgr8":
        arr = arr[:, :, ::-1]
    return (arr[:, :, :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.uint8)


class LocVizNode(Node):
    def __init__(self):
        super().__init__("loc_viz_node")
        self.pub_ = self.create_publisher(Image, "/localization_visualization", 10)

        # Load config
        try:
            config_path = _get_config_path()
            self.get_logger().info(f"Reading from config file: {config_path}")
            config = _load_config(config_path)
            num_triangles = config.get("num_triangles", 10)
            loc_viz_cfg = config.get("loc_viz", {})
        except Exception as e:
            self.get_logger().warn(f"Failed to load config: {e}. Using defaults.")
            num_triangles = 10
            loc_viz_cfg = {}
            config = {}

        self.num_triangles_ = num_triangles
        self.map_path = config.get("map_path", "/home/asrl/ros2_ws/dr_ba/output/maps/glen/small_test/voxel_map.bin")
        self.scan_res = loc_viz_cfg.get("scan_res", 0.1)
        self.scan_display_radius_m = loc_viz_cfg.get("scan_display_radius_m", 60.0)
        self.veh_len = loc_viz_cfg.get("veh_len", 6.0)
        self.veh_wid = loc_viz_cfg.get("veh_wid", 3.0)
        self.zoom_range = loc_viz_cfg.get("zoom_range", 80.0)
        self.fig_size_px = loc_viz_cfg.get("fig_size_px", 400)
        self.dpi = loc_viz_cfg.get("dpi", 100)

        # Pose history (deque with max length, or no max if -1)
        max_len = None if num_triangles == -1 else num_triangles
        self.pose_history_ = deque(maxlen=max_len)
        self.trail_polygons_ = []

        image_sub = Subscriber(self, Image, "/dro_local_map_image")
        estimate_sub = Subscriber(self, DRLEstimate, "/drl_estimate")
        sync = TimeSynchronizer([image_sub, estimate_sub], 10)
        sync.registerCallback(self._callback)

        self.odom_sub_ = self.create_subscription(
            Odometry, "/dro_odometry", self._odom_callback, 10)

        # DRO odometry is dead-reckoned from wherever the node started (its own
        # local frame). Anchor it to the map frame using /initialpose so the
        # odometry triangle tracks a real global pose instead of drifting from
        # an arbitrary origin.
        self._odom_init_pose_ = None  # (x, y, yaw) anchor, set by /initialpose
        self.initialpose_sub_ = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self._initialpose_callback, 10)

        # Persistent figure — artists updated in-place each frame
        fig_in = self.fig_size_px / self.dpi
        self._fig, self._ax = plt.subplots(figsize=(fig_in, fig_in), facecolor="white")
        self._fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self._ax.set_facecolor("white")
        self._ax.set_aspect("auto")  # figure is square + equal x/y ranges → naturally 1:1
        self._ax.axis("off")

        # Static background map (loaded once, never updated)
        map_img, map_extent = _load_voxel_map(self.map_path)
        self._ax.imshow(
            map_img.T,
            origin="lower",
            cmap="magma_r",
            vmin=0.0,
            vmax=1.0,
            extent=map_extent,
            interpolation="nearest",
            zorder=1,
        )
        self.get_logger().info(f"Loaded voxel map: extent={map_extent}")

        # Scan artist and transform — initialised on first message so we know image size
        self._scan_im = None
        self._scan_tf = None
        self._circle_mask = None
        self._scan_radius_m = None

        # Raw odometry triangle (dark red), drawn below the localization triangle
        self._odom_tri = Polygon(
            np.zeros((3, 2)),
            closed=True,
            facecolor="darkred",
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )
        self._ax.add_patch(self._odom_tri)

        # Estimated pose triangle (current, brightest) — kept above the odometry triangle
        self._est_tri = Polygon(
            np.zeros((3, 2)),
            closed=True,
            facecolor="#00FF00",
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
        )
        self._ax.add_patch(self._est_tri)

        # --- Legend (proxy artists, static) ---
        legend_est = Polygon(
            [[0, 0], [1, 0], [0.5, 1]],
            closed=True,
            facecolor="#00FF00",
            edgecolor="black",
            linewidth=1.2,
        )
        legend_odom = Polygon(
            [[0, 0], [1, 0], [0.5, 1]],
            closed=True,
            facecolor="darkred",
            edgecolor="black",
            linewidth=1.2,
        )

        legend_fontsize = loc_viz_cfg.get("legend_fontsize", 8)
        legend = self._ax.legend(
            handles=[legend_est, legend_odom],
            labels=["localization", "odometry"],
            loc="upper right",
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.8",
            fontsize=legend_fontsize,
        )
        legend.set_zorder(10)
        legend.set_bbox_to_anchor((1.0, 1.0), transform=self._ax.transAxes)

    def _update_trail(self):
        """Update trail polygon positions and opacity based on pose history."""
        num_history = len(self.pose_history_)
        if num_history <= 1:
            return

        # Create or remove trail polygons to match history (excluding current pose)
        num_trail = num_history - 1
        while len(self.trail_polygons_) < num_trail:
            tri = Polygon(
                np.zeros((3, 2)),
                closed=True,
                facecolor="#00FF00",
                edgecolor="black",
                linewidth=1.2,
                zorder=5,  # Below current triangle
                alpha=0.5,
            )
            self._ax.add_patch(tri)
            self.trail_polygons_.append(tri)

        while len(self.trail_polygons_) > num_trail:
            tri = self.trail_polygons_.pop()
            tri.remove()

        # Update positions and opacity (oldest→newest fades in)
        history_list = list(self.pose_history_)[:-1]
        for i, (x, y, yaw) in enumerate(history_list):
            alpha = 0.3 + 0.5 * (i / max(num_trail, 1))
            self.trail_polygons_[i].set_xy(_triangle_pts(x, y, yaw, self.veh_len, self.veh_wid))
            self.trail_polygons_[i].set_alpha(alpha)

    def _initialpose_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        self._odom_init_pose_ = (x, y, yaw)
        self.get_logger().info(f"DRO odometry re-initialized to x={x:.2f} y={y:.2f} yaw={yaw:.2f}")

    def _odom_callback(self, msg: Odometry):
        # Raw odometry pose, in DRO's own local (dead-reckoning) frame.
        rx = msg.pose.pose.position.x
        ry = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        ryaw = _quat_to_yaw(q.x, q.y, q.z, q.w)

        # Propagate the raw pose from the /initialpose anchor into the map frame.
        if self._odom_init_pose_ is not None:
            ix, iy, iyaw = self._odom_init_pose_
            c, s = np.cos(iyaw), np.sin(iyaw)
            mx = ix + c * rx - s * ry
            my = iy + s * rx + c * ry
            myaw = iyaw + ryaw
        else:
            mx, my, myaw = rx, ry, ryaw

        # Same y-flip convention as the estimate: visualization frame vs. map frame.
        ox = mx
        oy = -my
        oyaw = -myaw

        self._odom_tri.set_xy(_triangle_pts(ox, oy, oyaw, self.veh_len, self.veh_wid))

    def _init_scan_artist(self, H: int, W: int):
        """Set up scan imshow artist using the actual image dimensions."""
        # Cap displayed radius to scan_display_radius_m regardless of image size
        r_px = min(min(H, W) // 2, int(self.scan_display_radius_m / self.scan_res))
        r_m = r_px * self.scan_res

        yy, xx = np.ogrid[-r_px:r_px + 1, -r_px:r_px + 1]
        self._circle_mask = (xx ** 2 + yy ** 2) <= r_px ** 2
        self._scan_radius_m = r_m

        # Persistent RGBA buffer, reused every frame. RGB channels are filled
        # via cv2.cvtColor (SIMD channel replication, writes straight into the
        # buffer) instead of a numpy broadcast, which profiled much slower.
        # cvtColor always writes alpha=255, so the static circular-mask alpha
        # is reapplied after each conversion.
        self._rgba_buf = np.zeros((2 * r_px + 1, 2 * r_px + 1, 4), dtype=np.uint8)
        self._alpha_static = np.where(self._circle_mask, 204, 0).astype(np.uint8)
        self._rgba_buf[..., 3] = self._alpha_static

        self._scan_im = self._ax.imshow(
            self._rgba_buf,
            origin="lower",
            zorder=2,
            extent=[-r_m, r_m, -r_m, r_m],
        )
        self._scan_tf = Affine2D()
        self._scan_im.set_transform(self._scan_tf + self._ax.transData)

    def _callback(self, image_msg: Image, estimate_msg: DRLEstimate):
        scan_gray = _ros_image_to_gray(image_msg)
        H, W = scan_gray.shape

        if self._scan_im is None:
            self._init_scan_artist(H, W)

        # The visualization frame has y flipped relative to the map frame,
        # matching voxel_map.py convention (est_ys = -est_ys).
        ex = estimate_msg.x
        ey = -estimate_msg.y
        eyaw = -estimate_msg.theta

        # --- Add pose to history and update trail ---
        self.pose_history_.append((ex, ey, eyaw))
        self._update_trail()

        # --- Update current triangle ---
        self._est_tri.set_xy(_triangle_pts(ex, ey, eyaw, self.veh_len, self.veh_wid))

        # --- Crop to display radius; alpha masking is static (set once) ---
        cy, cx = H // 2, W // 2
        r_px = min(min(H, W) // 2, int(self.scan_display_radius_m / self.scan_res))
        crop = scan_gray[cy - r_px : cy + r_px + 1, cx - r_px : cx + r_px + 1]

        cv2.cvtColor(crop, cv2.COLOR_GRAY2BGRA, dst=self._rgba_buf)
        self._rgba_buf[..., 3] = self._alpha_static
        self._scan_im.set_data(self._rgba_buf)

        # --- Reposition scan in map frame ---
        self._scan_tf.clear()
        self._scan_tf.scale(1.0, -1.0)
        self._scan_tf.rotate(eyaw - np.pi / 2.0)
        self._scan_tf.translate(ex, ey)

        # --- View centred on estimate ---
        self._ax.set_xlim(ex - self.zoom_range, ex + self.zoom_range)
        self._ax.set_ylim(ey - self.zoom_range, ey + self.zoom_range)

        # --- Render ---
        self._fig.canvas.draw()

        w, h = self._fig.canvas.get_width_height()
        # array.array (not bytes/list) hits the fast path in the generated
        # msg setter, skipping its slow per-element Python validation loop.
        # buffer_rgba() is multi-dimensional, so flatten via bytes() first.
        data = array.array("B", bytes(self._fig.canvas.buffer_rgba()))

        # --- Publish ---
        out = Image()
        out.header.stamp = image_msg.header.stamp
        out.header.frame_id = "map"
        out.height, out.width = h, w
        out.encoding = "rgba8"
        out.step = w * 4
        out.data = data
        self.pub_.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LocVizNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
