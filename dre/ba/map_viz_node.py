#!/usr/bin/env python3
"""ROS2 node: subscribes to /map_path, (re)loads the voxel map it points to,
publishes it as a plot_paper-style image, and overlays the current
/dro_local_map_info position as a green dot."""

import os
import struct
import time
import warnings

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from dre.msg import DRLEstimate


def load_config():
    """Load parameters from config_mapping.yaml."""
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_path, "share/dre/config", "config_mapping.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config, config_path


VOXEL_DTYPE = np.dtype([("x", "<i4"), ("y", "<i4"), ("intensity", "<f8")])


def load_voxel_map(file_path):
    """Returns (res, poses, voxels) with voxels as a structured numpy array
    (fields x, y, intensity), parsed in one vectorized read."""
    poses = []
    with open(file_path, "rb") as f:
        res, num_poses, num_voxels = struct.unpack("<dII", f.read(16))
        pose_fmt = struct.Struct("<idddd")
        for _ in range(num_poses):
            pose_id, x, y, yaw, ate = pose_fmt.unpack(f.read(pose_fmt.size))
            poses.append((pose_id, x, y, yaw, ate))
        voxels = np.fromfile(f, dtype=VOXEL_DTYPE, count=num_voxels)
    return res, poses, voxels


class MapVizNode(Node):
    def __init__(self):
        super().__init__("map_viz_node")

        config, config_path = load_config()
        self.get_logger().info(f"Reading from config file: {config_path}")
        map_viz_config = config.get("map_viz", {})
        self.publish_hz = map_viz_config.get("publish_hz", 4)
        self.dpi = map_viz_config.get("dpi", 100)
        self.dot_radius = map_viz_config.get("dot_radius", 4)
        self.plot_poses = map_viz_config.get("plot_poses", False)
        self.veh_len = map_viz_config.get("veh_len", 6.0)
        self.veh_wid = map_viz_config.get("veh_wid", 3.0)
        self.fig_size = map_viz_config.get("fig_size", 4)

        self.pub_ = self.create_publisher(Image, "/voxel_map_image", 10)

        self._fig = None
        self._ax = None
        self._dot = None
        self._bg = None
        self._last_dot_xy = (None, None)

        # Lifetime timing sums for the periodic (every kReportInterval reloads)
        # INFO summary; per-reload detail goes to DEBUG instead (enable with
        # --ros-args --log-level map_viz_node:=debug) so normal operation isn't
        # spammed with one reload every time the map changes.
        self._report_interval = 50
        self._reload_count = 0
        self._report_load_sum = 0.0
        self._report_raster_sum = 0.0
        self._report_render_sum = 0.0
        self._report_total_sum = 0.0

        self.est_sub_ = self.create_subscription(
            DRLEstimate, "/drl_estimate", self._estimate_callback, 10)
        # Depth 1: reloads can take longer than the save period, and only the
        # newest map is worth loading — never replay a backlog of stale paths
        # Must match loc_node's transient_local publisher so the latched
        # /map_path message is delivered even if this node starts late.
        map_path_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_path_sub_ = self.create_subscription(
            String, "/map_path", self._map_path_callback, map_path_qos)

        self.create_timer(1.0 / self.publish_hz, self._publish_frame)

    def _map_path_callback(self, msg: String):
        map_path = msg.data
        self.get_logger().debug(f"Loading voxel map: {map_path}")

        # Build the new figure on local variables only; self._fig/_ax/_dot/_bg
        # are not touched until everything below succeeds, so a malformed or
        # empty map file (a race with a save, a truncated write, ...) just
        # logs a warning and leaves the currently-displayed map untouched
        # instead of taking the whole node down.
        try:
            t0 = time.perf_counter()
            res, poses, voxels = load_voxel_map(map_path)
            if len(voxels) == 0:
                raise ValueError("voxel map has no voxels")
            t_load = time.perf_counter() - t0
            self.get_logger().debug(
                f"Loaded {len(voxels)} voxels, res={res:.3f} m, {len(poses)} poses"
            )

            # Rasterize (same convention as render_localization_video_frames)
            t1 = time.perf_counter()
            vals = voxels["intensity"].astype(np.float32)
            ix, iy = voxels["x"].copy(), -voxels["y"]
            ix_min, ix_max = ix.min(), ix.max()
            iy_min, iy_max = iy.min(), iy.max()
            vals = np.clip(vals, 0.0, 0.6) / 0.3
            img = np.full((ix_max - ix_min + 1, iy_max - iy_min + 1), np.nan, dtype=np.float32)
            img[ix - ix_min, iy - iy_min] = vals

            # Downsample to ~2x the figure's pixel size (nan-aware block max) so
            # the matplotlib draw cost stays constant instead of growing with map
            # area. Without this, imshow resamples the full-res raster every draw.
            target_px = 2 * self.fig_size * self.dpi
            factor = max(1, int(np.ceil(max(img.shape) / target_px)))
            if factor > 1:
                h, w = img.shape
                hp = -(-h // factor) * factor
                wp = -(-w // factor) * factor
                padded = np.full((hp, wp), np.nan, dtype=np.float32)
                padded[:h, :w] = img
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    img = np.nanmax(padded.reshape(hp // factor, factor, wp // factor, factor), axis=(1, 3))
                # Extent covers the padded area so pixel centers stay aligned
                extent = [ix_min * res, (ix_min + hp) * res,
                          iy_min * res, (iy_min + wp) * res]
            else:
                extent = [ix_min * res, (ix_max + 1) * res,
                          iy_min * res, (iy_max + 1) * res]
            t_raster = time.perf_counter() - t1

            t2 = time.perf_counter()
            new_fig, ax = plt.subplots(figsize=(self.fig_size, self.fig_size), dpi=self.dpi, facecolor="white")
            new_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_facecolor("white")
            ax.imshow(img.T, origin="lower", cmap="magma_r", vmin=0.0, vmax=1.0,
                      extent=extent, interpolation="nearest", zorder=1)

            if self.plot_poses and poses:
                for p in poses:
                    pts = np.array([[self.veh_len/2, 0.], [-self.veh_len/2, self.veh_wid/2], [-self.veh_len/2, -self.veh_wid/2]])
                    c, s = np.cos(-p[3]), np.sin(-p[3])
                    pts = pts @ np.array([[c, -s], [s, c]]).T
                    pts[:, 0] += p[1]; pts[:, 1] += -p[2]
                    ax.add_patch(Polygon(pts, closed=True, facecolor="#00FF00",
                                         edgecolor="black", linewidth=1.2, zorder=6))

            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal")
            ax.axis("off")

            # Dot artist — updated in-place each odometry callback
            new_dot, = ax.plot([], [], "o", color="#00FF00",
                                 markersize=self.dot_radius, markeredgecolor="black",
                                 markeredgewidth=0.8, zorder=10)
            last_x, last_y = self._last_dot_xy
            if last_x is not None:
                new_dot.set_data([last_x], [last_y])

            # Draw background once, then save it for blitting
            new_fig.canvas.draw()
            new_bg = new_fig.canvas.copy_from_bbox(new_fig.bbox)
            t_render = time.perf_counter() - t2
        except Exception as e:
            self.get_logger().error(
                f"Failed to load/render voxel map '{map_path}': {e}. Keeping previous map.")
            return

        # Everything above succeeded: swap in the new figure and close the old one.
        if self._fig is not None:
            plt.close(self._fig)
        self._fig, self._ax, self._dot, self._bg = new_fig, ax, new_dot, new_bg

        total = time.perf_counter() - t0
        self.get_logger().debug(
            f"Map ready: extent x=[{extent[0]:.0f}, {extent[1]:.0f}] m  "
            f"y=[{extent[2]:.0f}, {extent[3]:.0f}] m"
        )
        self.get_logger().debug(
            f"[timing] map viz reload: load {t_load * 1000:.1f} ms ({len(voxels)} voxels), "
            f"rasterize {t_raster * 1000:.1f} ms, render {t_render * 1000:.1f} ms, "
            f"total {total * 1000:.1f} ms (frame publishing blocked throughout)"
        )

        self._reload_count += 1
        self._report_load_sum += t_load
        self._report_raster_sum += t_raster
        self._report_render_sum += t_render
        self._report_total_sum += total
        if self._reload_count % self._report_interval == 0:
            n = self._reload_count
            self.get_logger().info(
                f"[timing] avg over {n} reloads: load {self._report_load_sum / n * 1000:.1f} ms, "
                f"rasterize {self._report_raster_sum / n * 1000:.1f} ms, "
                f"render {self._report_render_sum / n * 1000:.1f} ms, "
                f"total {self._report_total_sum / n * 1000:.1f} ms | "
                f"{len(voxels)} voxels, extent x=[{extent[0]:.0f}, {extent[1]:.0f}] m "
                f"y=[{extent[2]:.0f}, {extent[3]:.0f}] m"
            )

    def _estimate_callback(self, msg: DRLEstimate):
        # Just update dot position — rendering is driven by the timer
        self._last_dot_xy = (msg.x, -msg.y)
        if self._dot is not None:
            self._dot.set_data([msg.x], [-msg.y])

    def _publish_frame(self):
        if self._fig is None:
            return  # no map loaded yet

        # Blit: restore static background, draw only the dot
        self._fig.canvas.restore_region(self._bg)
        self._ax.draw_artist(self._dot)
        self._fig.canvas.blit(self._fig.bbox)

        w, h = self._fig.canvas.get_width_height()
        data = bytes(self._fig.canvas.buffer_rgba())

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.height, msg.width = h, w
        msg.encoding = "rgba8"
        msg.step = w * 4
        msg.data = data
        self.pub_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MapVizNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
