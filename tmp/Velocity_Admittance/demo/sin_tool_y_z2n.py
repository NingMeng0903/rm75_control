#!/usr/bin/env python3
"""
Demo: 6D trajectory plugin + tool-frame force/motion hybrid.

  source env.sh
  cd /media/camp/EXT_DRIVE/rm75_control
  python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py
  python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py --trajectory sin_base_y --y-pp-cm 16
  python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py --trajectory sin_base_y_tool_rz --rz-deg 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rm75_control.control.hybrid_motion.loop import load_yaml, run_hybrid_motion_loop
from rm75_control.control.hybrid_motion.paths import CONFIG_SIN_TOOL_Y_Z2N

_VA_DIR = Path(__file__).resolve().parents[1]
if str(_VA_DIR) not in sys.path:
    sys.path.insert(0, str(_VA_DIR))
from demo_stack import build_demo_shaper, build_demo_source_factory, demo_trajectory_summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="6D trajectory + tool-Z force hybrid (trajectory is pluggable)",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_SIN_TOOL_Y_Z2N)
    parser.add_argument(
        "--trajectory", type=str, default=None,
        help="trajectory_demo.type: hold | sin_base_y | sin_base_y_tool_rz | sin_tool_y",
    )
    parser.add_argument("--desired-z", type=float, default=None, help="tool-Z force target (N)")
    parser.add_argument("--y-pp-cm", type=float, default=None, help="world-Y peak-to-peak (cm)")
    parser.add_argument("--rz-deg", type=float, default=None, help="tool +Z spin amplitude (deg)")
    parser.add_argument("--duration", type=float, default=None, help="scan duration (s)")
    parser.add_argument("--log", action="store_true", help="record pose_d vs pose_act npz every cycle")
    parser.add_argument("--log-path", type=Path, default=None, help="npz output (default: tmp/Velocity_Admittance/logs/)")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    traj = raw.setdefault("trajectory_demo", raw.get("trajectory", {}))
    if "trajectory" in raw and "trajectory_demo" not in raw:
        raw["trajectory_demo"] = traj
    if args.trajectory:
        traj["type"] = args.trajectory
    if args.y_pp_cm is not None:
        traj["y_peak_to_peak_cm"] = args.y_pp_cm
    if args.rz_deg is not None:
        traj["rz_amplitude_deg"] = args.rz_deg
    if args.desired_z is not None:
        raw.setdefault("force", {})["desired_z_n"] = args.desired_z

    return run_hybrid_motion_loop(
        raw,
        source_factory=build_demo_source_factory(raw),
        shaper=build_demo_shaper(raw),
        reference_summary=demo_trajectory_summary(raw),
        title="Demo 6D traj + tool-Z force",
        duration_s=args.duration,
        log_enabled=args.log or args.log_path is not None,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
