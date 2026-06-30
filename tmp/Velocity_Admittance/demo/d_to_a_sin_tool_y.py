#!/usr/bin/env python3
"""
Demo: transfer slot d → slot a, then tool-Y sin scan at pose a.

  source env.sh
  cd /media/camp/EXT_DRIVE/rm75_control
  python tmp/Velocity_Admittance/demo/d_to_a_sin_tool_y.py --log
  python tmp/Velocity_Admittance/demo/d_to_a_sin_tool_y.py --transfer-s 15 --duration 60 --log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rm75_control.control.hybrid_motion.loop import load_yaml, run_hybrid_motion_loop
from rm75_control.control.hybrid_motion.paths import CONFIG_D_TO_A_SIN_TOOL_Y

_VA_DIR = Path(__file__).resolve().parents[1]
if str(_VA_DIR) not in sys.path:
    sys.path.insert(0, str(_VA_DIR))
from demo_stack import build_demo_shaper, build_demo_source_factory, demo_trajectory_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="slot d→a transfer + tool-Y sin at pose a",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_D_TO_A_SIN_TOOL_Y)
    parser.add_argument("--from-slot", type=str, default=None, help="start pose slot (default: d)")
    parser.add_argument("--to-slot", type=str, default=None, help="scan pose slot (default: a)")
    parser.add_argument("--transfer-s", type=float, default=None, help="d→a transfer duration (s)")
    parser.add_argument("--y-pp-cm", type=float, default=None, help="tool-Y peak-to-peak at pose a (cm)")
    parser.add_argument("--desired-z", type=float, default=None, help="tool-Z force target (N)")
    parser.add_argument("--duration", type=float, default=None, help="scan duration after contact (s)")
    parser.add_argument("--log", action="store_true", help="record pose_d vs pose_act npz")
    parser.add_argument("--log-path", type=Path, default=None)
    args = parser.parse_args()

    raw = load_yaml(args.config)
    traj = raw.setdefault("trajectory_demo", {})
    traj.setdefault("type", "slot_transfer_sin_tool_y")
    traj.setdefault("from_slot", "d")
    traj.setdefault("to_slot", "a")
    traj.setdefault("scan_mode", "sin_tool_y")
    if args.from_slot:
        traj["from_slot"] = args.from_slot.lower()
    if args.to_slot:
        traj["to_slot"] = args.to_slot.lower()
    if args.transfer_s is not None:
        traj["transfer_s"] = args.transfer_s
    if args.y_pp_cm is not None:
        traj["y_peak_to_peak_cm"] = args.y_pp_cm
    if args.desired_z is not None:
        raw.setdefault("force", {})["desired_z_n"] = args.desired_z

    return run_hybrid_motion_loop(
        raw,
        source_factory=build_demo_source_factory(raw),
        shaper=build_demo_shaper(raw),
        reference_summary=demo_trajectory_summary(raw),
        title="Demo d→a transfer + tool-Y sin",
        duration_s=args.duration,
        log_enabled=args.log or args.log_path is not None,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
