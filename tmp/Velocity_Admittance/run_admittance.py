#!/usr/bin/env python3
"""
Velocity-resolved admittance (generic entry).

  source env.sh
  python tmp/Velocity_Admittance/run_admittance.py
  python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rm75_control.control.velocity_admittance.loop import load_yaml, run_velocity_admittance
from rm75_control.control.velocity_admittance.paths import CONFIG_ADMITTANCE


def main() -> int:
    parser = argparse.ArgumentParser(description="RM75 velocity admittance control")
    parser.add_argument("--config", type=Path, default=CONFIG_ADMITTANCE)
    parser.add_argument("--trajectory", type=str, default=None)
    parser.add_argument("--desired-z", type=float, default=None, help="sensor Fz target (N)")
    parser.add_argument("--duration", type=float, default=None, help="run time (s)")
    parser.add_argument("--log", action="store_true", help="record scan npz (pose_d vs pose_act)")
    parser.add_argument("--log-path", type=Path, default=None)
    args = parser.parse_args()

    raw = load_yaml(args.config)
    if args.trajectory:
        raw.setdefault("trajectory", {})["type"] = args.trajectory
    if args.desired_z is not None:
        raw.setdefault("force", {})["desired_z_n"] = args.desired_z

    return run_velocity_admittance(
        raw,
        duration_s=args.duration,
        log_enabled=args.log or args.log_path is not None,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
