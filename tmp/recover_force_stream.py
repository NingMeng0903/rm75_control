#!/usr/bin/env python3
"""Recover controller after rm_set_force_position / stream force conflict."""

from __future__ import annotations

import sys
import time
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rm75f_default.yaml"


def main() -> int:
    from rm75_control import RobotSession

    print("Recovering force-control modes...", flush=True)
    with RobotSession(config=CONFIG) as bot:
        for attempt in range(6):
            diag = bot.prepare_for_force_stream(settle_s=2.0)
            ret = bot.robot.rm_start_force_position_move()
            print(f"attempt {attempt}: {diag} start={ret}", flush=True)
            if ret == 0:
                bot.robot.rm_stop_force_position_move()
                time.sleep(1.0)
                print("Recovered. rm_start_force_position_move OK.", flush=True)
                return 0
        print(
            "Recovery failed. Stop all programs on the teach pendant, then power-cycle "
            "the arm controller and retry.",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
