#!/usr/bin/env python3
"""Read joint, pose, tool force."""

import sys
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rm75f_default.yaml"

print("connecting...", flush=True)
from rm75_control import RobotSession

with RobotSession(config=CONFIG) as bot:
    print("connected.", flush=True)
    print("info:", bot.robot.rm_get_robot_info())
    ret, s = bot.robot.rm_get_current_arm_state()
    if ret != 0:
        print("get state failed:", ret, file=sys.stderr)
        sys.exit(1)
    print("joint (deg):", s["joint"])
    print("pose (m, rad):", s["pose"])
    ret, f = bot.robot.rm_get_force_data()
    if ret == 0:
        print("tool force:", list(f["tool_zero_force_data"]))
    else:
        print("force read failed:", ret)
print("done.", flush=True)
