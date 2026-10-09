#!/usr/bin/env python3
"""Run DRO over every Boreas sequence in a folder, then evaluate the results.

`dro_node` learns its output sequence folder from the first `RadarInfo` message and
never re-initializes, so a batch run cannot reuse a single node across sequences: this
script starts a fresh `dro_node` per sequence, waits for it to log "DRO ready" (which
can take ~30s when torch compilation is enabled), replays the sequence with
`boreas_player`, then tears the node down before moving on.

Example:
    python3 run_boreas_batch.py -d /home/clegentil/Documents/data/boreas_2

By default it replays offline (`-r 0`, i.e. the player publishes as fast as DRO can
consume), writes to <repo>/output/poses/<sequence_id>/ (the same tree mode_launch.py
uses), and runs boreas_eval.py at the end.
"""

import argparse
import json
import os
import os.path as osp
import re
import signal
import subprocess
import sys
import threading
import time

REPO_ROOT = osp.dirname(osp.abspath(__file__))
DEFAULT_OUTPUT_PATH = osp.join(REPO_ROOT, "output", "poses")

# dro_node logs this once its (optionally torch-compiled) pipeline is ready to accept
# data. Starting the player before then just means the first scans get dropped.
READY_MARKER = "DRO ready"

# boreas_player logs one of these per radar frame it publishes; it's the only place the
# sequence length is known, so it doubles as the progress source.
FRAME_RE = re.compile(r"Publishing radar frame (\d+) / (\d+)")


def find_sequences(data_dir, requested):
    available = sorted(
        d for d in os.listdir(data_dir)
        if d.startswith("boreas-") and osp.isdir(osp.join(data_dir, d))
    )
    if not requested:
        return available

    missing = [s for s in requested if s not in available]
    if missing:
        raise SystemExit(f"Sequences not found in {data_dir}: {', '.join(missing)}")
    # Keep the on-disk order rather than the order they were typed in, so logs and
    # output are comparable between a full run and a partial re-run.
    return [s for s in available if s in requested]


def start_dro_node(output_path, log_file):
    env = dict(os.environ)
    # Without these, the node's logs only reach us in ~4KB chunks, so we'd detect
    # READY_MARKER long after it was actually printed.
    env["PYTHONUNBUFFERED"] = "1"
    env["RCUTILS_LOGGING_BUFFERED_STREAM"] = "0"

    return subprocess.Popen(
        ["ros2", "run", "dre", "dro_node", "--ros-args", "-p", f"output_path:={output_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # ROS logging goes to stderr; merge so one reader sees both
        text=True,
        bufsize=1,
        env=env,
        # Own process group so we can kill the node and anything it spawned in one go
        preexec_fn=os.setsid,
    )


def pump_output(proc, log_file, ready_event, echo):
    """Drain the node's output into the log, flagging ready_event on READY_MARKER.

    This has to run in a thread and keep draining for the node's whole lifetime: if
    nobody reads the pipe, the node blocks on a full pipe buffer mid-sequence.
    """
    for line in proc.stdout:
        log_file.write(line)
        if echo:
            sys.stdout.write(f"  [dro] {line}")
        if not ready_event.is_set() and READY_MARKER in line:
            ready_event.set()
    log_file.flush()


def format_duration(seconds):
    if seconds is None or seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "--:--"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressReporter:
    """Renders per-frame replay progress for one sequence.

    On a terminal this rewrites a single line in place; when output is redirected
    (a batch left running under nohup, say) it prints a new line every
    `line_every` frames instead, so the file stays readable rather than filling
    with carriage returns.
    """

    def __init__(self, enabled, line_every=100):
        self.enabled = enabled
        self.line_every = line_every
        self.tty = sys.stdout.isatty()
        self.start = time.time()
        self.total = None
        self.current = 0
        self.last_width = 0
        self.last_drawn = -1
        # Wall-clock of the last frame advance, read by the stall watchdog. Tracked
        # even when rendering is off, so --no_progress/-v still get the watchdog.
        self.last_advance = time.time()

    def update(self, current, total):
        if current != self.current:
            self.last_advance = time.time()
        self.current, self.total = current, total
        if not self.enabled:
            return
        if not self.tty and current % self.line_every and current != total:
            return
        if self.tty and current == self.last_drawn:
            return
        self.last_drawn = current

        elapsed = time.time() - self.start
        rate = current / elapsed if elapsed > 0 else 0.0
        eta = (total - current) / rate if rate > 0 else None
        pct = 100.0 * current / total if total else 0.0
        line = (f"  frame {current}/{total} ({pct:5.1f}%) | {rate:4.1f} fps | "
                f"elapsed {format_duration(elapsed)} | eta {format_duration(eta)}")

        if self.tty:
            # Pad to erase the tail of a previously longer line
            sys.stdout.write("\r" + line.ljust(self.last_width))
            self.last_width = max(self.last_width, len(line))
        else:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def finish(self):
        """Close out the in-place line so the next print starts on a fresh row.

        Idempotent: it's called both on the timeout path (before the message that
        explains the abort) and from the cleanup that follows it.
        """
        if self.enabled and self.tty and self.last_drawn >= 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.last_drawn = -1


def pump_player_output(proc, log_file, progress, echo):
    """Drain the player's output into the log, driving the progress readout.

    Same reason this has to keep draining as for dro_node: an unread pipe eventually
    blocks the writer mid-sequence.
    """
    for line in proc.stdout:
        log_file.write(line)
        if echo:
            sys.stdout.write(f"  [player] {line}")
        match = FRAME_RE.search(line)
        if match:
            # The player's index is 0-based; report it 1-based so the readout ends at N/N
            progress.update(int(match.group(1)) + 1, int(match.group(2)))
    log_file.flush()


def wait_for_player(proc, progress, args):
    """Wait for the player, watching for an overall timeout and for a stalled replay.

    Returns None if it finished on its own, otherwise a message describing why it
    should be aborted. The stall check matters because a wedged pipeline doesn't
    exit or error — it just stops advancing, which would otherwise burn the whole
    `--sequence_timeout` (and, unattended, most of a night) on one dead sequence.
    """
    deadline = time.time() + args.sequence_timeout
    while True:
        try:
            proc.wait(timeout=1.0)
            return None
        except subprocess.TimeoutExpired:
            pass

        now = time.time()
        if now > deadline:
            return f"Sequence exceeded {args.sequence_timeout:.0f}s."

        stalled = now - progress.last_advance
        if args.stall_timeout > 0 and stalled > args.stall_timeout:
            return (f"No frame processed for {stalled:.0f}s "
                    f"(stuck at frame {progress.current}/{progress.total}).")


def stop_process_group(proc, name, timeout=10.0):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  {name} ignored SIGINT, killing it.")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            pass
    except ProcessLookupError:
        pass


def odometry_file(output_path, sequence):
    return osp.join(output_path, sequence, "odometry_result", sequence + ".txt")


def completion_marker(output_path, sequence):
    """Written only once a sequence replays all the way through.

    --skip_existing can't key off the odometry file: dro_node appends to it per frame,
    so a sequence that crashed or stalled half way leaves a perfectly real-looking file
    behind, and a resumed batch would skip exactly the sequences that need re-running.
    """
    return osp.join(output_path, sequence, "batch_complete.json")


def run_sequence(sequence, data_dir, output_path, log_dir, args):
    """Run one sequence end to end. Returns True if it produced an odometry file."""
    sequence_path = osp.join(data_dir, sequence)
    log_path = osp.join(log_dir, sequence + ".log")

    # Drop any marker from a previous run up front, so a re-run that fails can't be
    # mistaken for a completed one by a later --skip_existing.
    marker = completion_marker(output_path, sequence)
    if osp.exists(marker):
        os.remove(marker)

    # Line-buffered: a block-buffered log trails the live readout by thousands of
    # characters, which is exactly the wrong behaviour when diagnosing a wedged run.
    with open(log_path, "w", buffering=1) as log_file:
        ready_event = threading.Event()
        dro_proc = start_dro_node(output_path, log_file)
        reader = threading.Thread(
            target=pump_output,
            args=(dro_proc, log_file, ready_event, args.verbose),
            daemon=True,
        )
        reader.start()

        try:
            if not ready_event.wait(timeout=args.startup_timeout):
                print(f"  dro_node never reported '{READY_MARKER}' within "
                      f"{args.startup_timeout:.0f}s (see {log_path}).")
                return False
            if dro_proc.poll() is not None:
                print(f"  dro_node exited during startup (see {log_path}).")
                return False

            print(f"  dro_node ready, replaying (rate {args.playback_rate}) ...")
            player_env = dict(os.environ)
            player_env["PYTHONUNBUFFERED"] = "1"
            player_env["RCUTILS_LOGGING_BUFFERED_STREAM"] = "0"
            player_proc = subprocess.Popen(
                ["ros2", "run", "dre", "boreas_player",
                 "-p", sequence_path, "-r", str(args.playback_rate)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=player_env,
                preexec_fn=os.setsid,
            )
            # In verbose mode the raw echo already shows every frame line; a progress
            # line interleaved with it would just be noise (the watchdog still runs).
            progress = ProgressReporter(enabled=not (args.no_progress or args.verbose))
            player_reader = threading.Thread(
                target=pump_player_output,
                args=(player_proc, log_file, progress, args.verbose),
                daemon=True,
            )
            player_reader.start()
            try:
                abort = wait_for_player(player_proc, progress, args)
                if abort:
                    progress.finish()
                    print(f"  {abort} Aborting this sequence and moving on.")
                    stop_process_group(player_proc, "boreas_player")
                    return False
            finally:
                # Covers the case where wait() was interrupted by Ctrl-C
                stop_process_group(player_proc, "boreas_player")
                player_reader.join(timeout=5.0)
                progress.finish()

            if player_proc.returncode != 0:
                print(f"  boreas_player exited with code {player_proc.returncode} "
                      f"(see {log_path}).")
        finally:
            stop_process_group(dro_proc, "dro_node")
            reader.join(timeout=5.0)

    result_file = odometry_file(output_path, sequence)
    if not osp.exists(result_file):
        print(f"  No odometry output was written (see {log_path}).")
        return False

    with open(result_file) as f:
        poses = sum(1 for _ in f)
    with open(marker, "w") as f:
        json.dump({"sequence": sequence,
                   "poses": poses,
                   "frames_played": progress.current,
                   "frames_total": progress.total,
                   "playback_rate": args.playback_rate,
                   "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run DRO over every Boreas sequence in a folder and evaluate it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--data_dir", required=True,
                        help="Folder containing the boreas-* sequence folders")
    parser.add_argument("-o", "--output_path", default=DEFAULT_OUTPUT_PATH,
                        help="Root folder for the per-sequence DRO output")
    parser.add_argument("-s", "--sequences", nargs="+", default=None,
                        help="Only run these sequence IDs (default: all of them)")
    parser.add_argument("-r", "--playback_rate", type=float, default=0.0,
                        help="boreas_player playback rate; 0 means offline (no waiting "
                             "between frames)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip sequences that previously ran to completion (resume a "
                             "batch); sequences that stalled or crashed are re-run")
    parser.add_argument("--startup_timeout", type=float, default=300.0,
                        help="Seconds to wait for dro_node to report that it is ready")
    parser.add_argument("--sequence_timeout", type=float, default=7200.0,
                        help="Seconds to wait for a sequence to finish replaying")
    parser.add_argument("--stall_timeout", type=float, default=120.0,
                        help="Abort a sequence if no radar frame is processed for this "
                             "many seconds (0 disables the stall watchdog)")
    parser.add_argument("--no_eval", action="store_true",
                        help="Only run the sequences, don't run boreas_eval.py afterwards")
    parser.add_argument("--gt_path", default=None,
                        help="Ground-truth folder passed to boreas_eval.py "
                             "(default: same as --data_dir)")
    parser.add_argument("--no_progress", action="store_true",
                        help="Don't print the per-frame progress readout")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Echo dro_node's and boreas_player's output to the terminal "
                             "as well as the log (replaces the progress readout)")
    args = parser.parse_args()

    data_dir = osp.abspath(osp.expanduser(args.data_dir))
    output_path = osp.abspath(osp.expanduser(args.output_path))
    if not osp.isdir(data_dir):
        raise SystemExit(f"Data folder {data_dir} does not exist.")

    sequences = find_sequences(data_dir, args.sequences)
    if not sequences:
        raise SystemExit(f"No boreas-* sequence folders found in {data_dir}.")

    log_dir = osp.join(output_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print(f"Running DRO on {len(sequences)} sequence(s) from {data_dir}")
    print(f"Output: {output_path}")
    print(f"Logs:   {log_dir}")

    succeeded, failed, skipped = [], [], []
    batch_start = time.time()

    for idx, sequence in enumerate(sequences, start=1):
        if args.skip_existing and osp.exists(completion_marker(output_path, sequence)):
            print(f"\n[{idx}/{len(sequences)}] {sequence}: already completed, skipping.")
            skipped.append(sequence)
            continue

        print(f"\n[{idx}/{len(sequences)}] {sequence}")
        start = time.time()
        try:
            ok = run_sequence(sequence, data_dir, output_path, log_dir, args)
        except KeyboardInterrupt:
            print("\nInterrupted, stopping the batch.")
            failed.append(sequence)
            break
        elapsed = time.time() - start
        if ok:
            print(f"  Done in {elapsed:.0f}s.")
            succeeded.append(sequence)
        else:
            print(f"  Failed after {elapsed:.0f}s.")
            failed.append(sequence)

    print(f"\n{'=' * 70}")
    print(f"Batch finished in {(time.time() - batch_start) / 60:.1f} min: "
          f"{len(succeeded)} succeeded, {len(failed)} failed, {len(skipped)} skipped.")
    if failed:
        print(f"Failed sequences: {', '.join(failed)}")

    if args.no_eval:
        return 0 if not failed else 1

    print(f"\n{'=' * 70}")
    print("Running evaluation ...")
    eval_cmd = [sys.executable, osp.join(REPO_ROOT, "boreas_eval.py"),
                output_path, args.gt_path or data_dir]
    eval_ret = subprocess.call(eval_cmd)
    return eval_ret if eval_ret else (0 if not failed else 1)


if __name__ == "__main__":
    sys.exit(main())
