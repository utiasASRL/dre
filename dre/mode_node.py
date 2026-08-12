#!/usr/bin/env python3
"""Starts/stops the node processes for the current operational mode.

Each mode is fully process-level: switching modes kills whatever the old mode
needed and starts whatever the new mode needs from scratch (no shared state
carried across a switch). Publishes the current mode on /operational_mode
(latched) so nodes that stay running across modes (e.g. raplace) can still
tell what's going on around them.
"""

import os
import signal
import subprocess
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory, get_package_prefix

RVIZ_POGO_PATH = os.path.join(get_package_share_directory("dre"), "config", "rviz_pogo.rviz")
RVIZ_MAPPING_PATH = os.path.join(get_package_share_directory("dre"), "config", "rviz_mapping.rviz")
RVIZ_DRL_PATH = os.path.join(get_package_share_directory("dre"), "config", "rviz_drl.rviz")
CONFIG_POGO_PATH = os.path.join(get_package_share_directory("dre"), "config", "config_pogo.yaml")
CONFIG_MAPPING_PATH = os.path.join(get_package_share_directory("dre"), "config", "config_mapping.yaml")
CONFIG_LOC_PATH = os.path.join(get_package_share_directory("dre"), "config", "config_loc.yaml")

# Runtime output data deliberately lives in the dre package's own source
# directory (output/) rather than under install/dre/share/dre — output is live
# data, not a package resource, and shouldn't be at risk from a clean
# rebuild/reinstall. Resolved via package.xml rather than hardcoding a path:
# --symlink-install symlinks package.xml back to source (unlike install(PROGRAMS
# ...) for node scripts, which are real copies), so this is the one file
# guaranteed to point at the actual dre package root regardless of where the
# workspace lives on disk.
DRE_PACKAGE_ROOT = os.path.dirname(
    os.path.realpath(os.path.join(get_package_share_directory("dre"), "package.xml"))
)
# dro_node and pogo_node each already nest their own output under <root>/<seq>/
# themselves (learned from the first RadarInfo message), as does mapping_node —
# so giving dro/pogo one root and mapping_node a different one means both trees
# auto-populate by sequence with no risk of the two ever writing the same file.
POSES_OUTPUT_PATH = os.path.join(DRE_PACKAGE_ROOT, "output", "poses")
MAPS_OUTPUT_PATH = os.path.join(DRE_PACKAGE_ROOT, "output", "maps")

# Shared across modes that need them — same instance re-used, never mutated.
DRO_NODE_ENTRY = {
    "package": "dre",
    "executable": "dro_node",
    "args": ["--ros-args", "-p", f"output_path:={POSES_OUTPUT_PATH}"],
}
RAPLACE_NODE_ENTRY = {"package": "dre", "executable": "raplace_node", "args": []}
REGISTRATION_NODE_ENTRY = {"package": "dre", "executable": "registration_node", "args": []}
POGO_NODE_ENTRY = {
    "package": "dre",
    "executable": "pogo_node",
    "args": [
        "--ros-args",
        "-p", f"config_file:={CONFIG_POGO_PATH}",
        "-p", f"output_path:={POSES_OUTPUT_PATH}",
    ],
}
MAPPING_NODE_ENTRY = {
    "package": "dre",
    "executable": "mapping_node",
    "args": [
        "--ros-args",
        "-p", f"config_file:={CONFIG_MAPPING_PATH}",
        "-p", f"output_path:={MAPS_OUTPUT_PATH}",
    ],
}
MAP_VIZ_NODE_ENTRY = {"package": "dre", "executable": "map_viz_node", "args": []}
LOC_NODE_ENTRY = {
    "package": "dre",
    "executable": "loc_node",
    "args": ["--ros-args", "-p", f"config_file:={CONFIG_LOC_PATH}"],
}
INITIAL_POSE_SELECTOR_ENTRY = {"package": "dre", "executable": "initial_pose_selector", "args": []}
LOC_VIZ_NODE_ENTRY = {"package": "dre", "executable": "loc_viz_node", "args": []}
STATIC_TF_ENTRY = {
    "package": "tf2_ros",
    "executable": "static_transform_publisher",
    "args": ["0", "0", "0", "0", "0", "0", "odom", "map"],
}
RVIZ_POGO_ENTRY = {
    "package": "rviz2",
    "executable": "rviz2",
    "args": ["-d", RVIZ_POGO_PATH],
    "wait_for": ("dro_node", "DRO ready"),
    "skip_if_headless": True,
}
RVIZ_MAPPING_ENTRY = {
    "package": "rviz2",
    "executable": "rviz2",
    "args": ["-d", RVIZ_MAPPING_PATH],
    "wait_for": ("dro_node", "DRO ready"),
    "skip_if_headless": True,
}
RVIZ_DRL_ENTRY = {
    "package": "rviz2",
    "executable": "rviz2",
    "args": ["-d", RVIZ_DRL_PATH],
    "wait_for": ("dro_node", "DRO ready"),
    "skip_if_headless": True,
}

# mode -> list of node entries to run for that mode. Each entry:
#   package, executable: as passed to the installed binary lookup.
#   args: the exact argv appended after the executable path — including "--ros-args"
#     yourself if you need ROS param overrides (e.g. "-p"). Plain program args (like
#     rviz2's "-d <file>") must NOT go after "--ros-args", so this isn't done for you.
#   wait_for: optional (other_executable, marker) — don't start this entry until
#     `marker` shows up in `other_executable`'s stdout. Mirrors what dro_launch.py/
#     dr_pogo_launch.py do with an OnProcessIO event handler (DRO's startup time
#     varies from near-instant to ~30s under torch compile, so a fixed delay isn't
#     good enough).
#   skip_if_headless: if True, this entry is left out entirely when mode_node's
#     `headless` parameter is set (used for rviz).
# TODO: extend with the "dre" (live-switchable, all nodes) mode once each of
# these has an established way to demote to idle rather than fully stopping.
MODE_TABLE = {
    "dro": [
        DRO_NODE_ENTRY,
        RVIZ_POGO_ENTRY,
    ],
    "pogo": [
        DRO_NODE_ENTRY,
        RAPLACE_NODE_ENTRY,
        REGISTRATION_NODE_ENTRY,
        POGO_NODE_ENTRY,
        STATIC_TF_ENTRY,
        RVIZ_POGO_ENTRY,
    ],
    "mapping": [
        DRO_NODE_ENTRY,
        RAPLACE_NODE_ENTRY,
        REGISTRATION_NODE_ENTRY,
        POGO_NODE_ENTRY,
        MAPPING_NODE_ENTRY,
        MAP_VIZ_NODE_ENTRY,
        STATIC_TF_ENTRY,
        RVIZ_MAPPING_ENTRY,
    ],
    "localization": [
        DRO_NODE_ENTRY,
        RAPLACE_NODE_ENTRY,
        REGISTRATION_NODE_ENTRY,
        LOC_NODE_ENTRY,
        INITIAL_POSE_SELECTOR_ENTRY,
        LOC_VIZ_NODE_ENTRY,
        MAP_VIZ_NODE_ENTRY,
        STATIC_TF_ENTRY,
        RVIZ_DRL_ENTRY,
    ],
}


class ModeNode(Node):
    def __init__(self):
        super().__init__("mode_node")

        self.declare_parameter("mode", "dro")
        self.declare_parameter("headless", False)
        mode = self.get_parameter("mode").get_parameter_value().string_value
        self.headless_ = self.get_parameter("headless").get_parameter_value().bool_value
        self.mode_table_ = MODE_TABLE

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode_pub_ = self.create_publisher(String, "/operational_mode", qos)

        self.processes_ = {}  # executable name -> subprocess.Popen
        self.marker_events_ = {}  # (executable, marker) -> threading.Event
        self.current_mode_ = None

        self.set_mode(mode)

    def set_mode(self, mode: str):
        if mode not in self.mode_table_:
            self.get_logger().error(f"Unknown mode '{mode}'. Known modes: {list(self.mode_table_.keys())}")
            return

        entries = self.mode_table_[mode]
        desired = [e for e in entries if not (self.headless_ and e.get("skip_if_headless"))]
        desired_names = {e["executable"] for e in desired}

        # Every node restarts fresh on any mode switch, even ones needed in both
        # the old and new mode (e.g. raplace_node runs in pogo/mapping/localization
        # but behaves differently in each): several nodes hold internal state
        # (raplace_node's history, initial_pose_selector's/loc_node's map state)
        # that's specific to how they were configured for the mode that started
        # them, so leaving one running across a switch risked carrying stale
        # state into the new mode.
        for name in list(self.processes_.keys()):
            self.stop_node(name)

        # Fresh events every time set_mode runs: an old, already-set event from a
        # previous activation of this same mode must not make a freshly-restarted
        # node look "ready" before it actually logs its marker line again.
        self.marker_events_ = {
            entry["wait_for"]: threading.Event() for entry in desired if entry.get("wait_for")
        }
        watched = {target for target, _ in self.marker_events_.keys()}

        # Start entries with no dependency immediately; entries with wait_for are
        # started by a background thread once their marker line shows up.
        for entry in desired:
            executable = entry["executable"]
            wait_for = entry.get("wait_for")
            if wait_for is None:
                self.start_node(entry["package"], executable, entry.get("args", []),
                                 capture_stdout=executable in watched)
            else:
                event = self.marker_events_[wait_for]
                threading.Thread(target=self._start_when_ready, args=(entry, event), daemon=True).start()

        self.current_mode_ = mode
        msg = String()
        msg.data = mode
        self.mode_pub_.publish(msg)
        self.get_logger().info(
            f"Operational mode set to '{mode}' ({sorted(desired_names)} running"
            f"{', headless' if self.headless_ else ''})."
        )

    def _start_when_ready(self, entry, event: threading.Event):
        event.wait()
        self.start_node(entry["package"], entry["executable"], entry.get("args", []))

    def start_node(self, package: str, executable: str, extra_args, capture_stdout: bool = False):
        # Invoke the installed executable directly rather than going through
        # `ros2 run`: that CLI spawns the actual node as a further child process
        # of its own, so killing the `ros2 run` PID doesn't kill the node it
        # started — it just orphans it. Launching the binary ourselves means
        # proc.pid IS the node process.
        exe_path = os.path.join(get_package_prefix(package), "lib", package, executable)
        cmd = [exe_path] + list(extra_args)
        self.get_logger().info(f"Starting node '{executable}': {' '.join(cmd)}")

        # New process group (defense-in-depth): if the node itself ever spawns
        # children, killing the whole group still reaches them.
        if capture_stdout:
            proc = subprocess.Popen(
                cmd, preexec_fn=os.setsid,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            threading.Thread(target=self._pump_stdout, args=(proc, executable), daemon=True).start()
        else:
            proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
        self.processes_[executable] = proc

    def _pump_stdout(self, proc: subprocess.Popen, executable: str):
        # Forward the node's output as-is (so it's still visible in the terminal/log),
        # while watching for any marker line other entries are waiting on.
        for line in proc.stdout:
            print(line, end="", flush=True)
            for (target, marker), event in self.marker_events_.items():
                if target == executable and marker in line:
                    event.set()

    def stop_node(self, name: str):
        proc = self.processes_.pop(name, None)
        if proc is None:
            return
        self.get_logger().info(f"Stopping node '{name}' (pid {proc.pid}).")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.get_logger().warn(f"Node '{name}' did not exit in time, killing.")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    def shutdown(self):
        for name in list(self.processes_.keys()):
            self.stop_node(name)


def _raise_keyboard_interrupt(signum, frame):
    # SIGINT already surfaces as KeyboardInterrupt; SIGTERM doesn't by default
    # (Python just terminates immediately, skipping our except/finally cleanup
    # below), which orphaned this node's spawned children when killed that way
    # (e.g. `timeout` sends SIGTERM). Routing both through the same exception
    # means there's one cleanup path instead of two.
    raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)
    node = ModeNode()
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        # rclpy.spin(node) blocks in a C-level wait that's specially wired (inside
        # rclpy itself) to be interrupted by SIGINT — a plain signal.signal() handler
        # for SIGTERM never actually gets invoked while blocked there (confirmed by
        # testing directly), so it'd not actually fix the orphaning above. Polling in
        # short bursts instead means Python regularly gets control back between calls,
        # where pending signals (including our SIGTERM handler) really do run.
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
