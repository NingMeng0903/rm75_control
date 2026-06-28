#!/usr/bin/env python3
"""
Demo: world-Y sin (±7.5 cm) + 3 N tool-Z force (fast, anti-overshoot).

  source env.sh
  python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rm75_control.control.velocity_admittance.loop import load_yaml, run_velocity_admittance
from rm75_control.control.velocity_admittance.paths import CONFIG_SIN_TOOL_Y_Z2N


def main() -> int:
    parser = argparse.ArgumentParser(
        description="World-Y sin scan (±7.5 cm) with 3 N tool-Z force",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_SIN_TOOL_Y_Z2N)
    parser.add_argument("--desired-z", type=float, default=None, help="Fz target (N)")
    parser.add_argument("--duration", type=float, default=None, help="run time (s)")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    if args.desired_z is not None:
        raw.setdefault("force", {})["desired_z_n"] = args.desired_z

    return run_velocity_admittance(
        raw,
        title="Demo sin_tool_y + Fz=3N",
        duration_s=args.duration,
    )


if __name__ == "__main__":
    raise SystemExit(main())
