#!/usr/bin/env python3
"""Print all 6D force fields from rm_get_force_data (loop until Ctrl+C).

After force-sensor gravity/tool (TCP) calibration on the teach pendant, the
controller exposes BOTH raw and compensated wrenches:

  force_data            — raw sensor reading (tool weight + gravity included)
  zero_force_data       — external wrench in sensor frame (compensated)
  work_zero_force_data  — external wrench in work frame (compensated)
  tool_zero_force_data  — external wrench in tool frame (compensated)

Force-position control and tcp_z_spring.py use tool_zero_force_data (compensated).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rm75f_default.yaml"
AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


def fmt6(w: list[float]) -> str:
    return "  ".join(f"{a}={v:+.3f}" for a, v in zip(AXES, w))


def main() -> int:
    p = argparse.ArgumentParser(description="Print 6D force data from RealMan API")
    p.add_argument("--hz", type=float, default=10.0, help="print rate (default 10)")
    p.add_argument(
        "--once",
        action="store_true",
        help="single sample then exit",
    )
    args = p.parse_args()
    dt = 1.0 / args.hz if args.hz > 0 else 0.1

    from rm75_control import RobotSession

    print("connecting...", flush=True)
    with RobotSession(config=CONFIG) as bot:
        print("connected. Ctrl+C to stop.\n", flush=True)
        print(
            "Legend: force_data=raw | zero_*=after gravity/tool calibration\n",
            flush=True,
        )
        n = 0
        try:
            while True:
                ret, f = bot.robot.rm_get_force_data()
                n += 1
                ts = time.strftime("%H:%M:%S")
                if ret != 0:
                    print(f"[{ts}] #{n} rm_get_force_data failed: {ret}", file=sys.stderr)
                else:
                    print(f"[{ts}] #{n}")
                    print(f"  raw (sensor)     {fmt6(list(f['force_data']))}")
                    print(f"  zero (sensor)    {fmt6(list(f['zero_force_data']))}")
                    print(f"  zero (work)      {fmt6(list(f['work_zero_force_data']))}")
                    print(f"  zero (tool)      {fmt6(list(f['tool_zero_force_data']))}")
                    print(flush=True)
                if args.once:
                    break
                time.sleep(dt)
        except KeyboardInterrupt:
            print("\nstopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
