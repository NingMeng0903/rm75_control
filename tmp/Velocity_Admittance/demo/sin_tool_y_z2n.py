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
from pathlib import Path

from rm75_control.control.velocity_admittance.loop import load_yaml, run_velocity_admittance
from rm75_control.control.velocity_admittance.paths import CONFIG_SIN_TOOL_Y_Z2N


def main() -> int:
    parser = argparse.ArgumentParser(
        description="6D trajectory + tool-Z force hybrid (trajectory is pluggable)",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_SIN_TOOL_Y_Z2N)
    parser.add_argument(
        "--trajectory", type=str, default=None,
        help="trajectory.type: hold | sin_base_y | sin_base_y_tool_rz | sin_tool_y",
    )
    parser.add_argument("--desired-z", type=float, default=None, help="tool-Z force target (N)")
    parser.add_argument("--y-pp-cm", type=float, default=None, help="world-Y peak-to-peak (cm)")
    parser.add_argument("--rz-deg", type=float, default=None, help="tool +Z spin amplitude (deg)")
    parser.add_argument("--duration", type=float, default=None, help="scan duration (s)")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    traj = raw.setdefault("trajectory", {})
    if args.trajectory:
        traj["type"] = args.trajectory
    if args.y_pp_cm is not None:
        traj["y_peak_to_peak_cm"] = args.y_pp_cm
    if args.rz_deg is not None:
        traj["rz_amplitude_deg"] = args.rz_deg
    if args.desired_z is not None:
        raw.setdefault("force", {})["desired_z_n"] = args.desired_z

    return run_velocity_admittance(
        raw,
        title="Demo 6D traj + tool-Z force",
        duration_s=args.duration,
    )


if __name__ == "__main__":
    raise SystemExit(main())
