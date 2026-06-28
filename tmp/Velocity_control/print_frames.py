#!/usr/bin/env python3
"""
Print RealMan coordinate-frame relationships (read-only + optional probe).

  source env.sh
  python tmp/Velocity_control/print_frames.py
  python tmp/Velocity_control/print_frames.py --probe-y-mm 5

Frame summary (RM API):
  rm_get_current_arm_state().pose     — 末端位姿，基/work 坐标系 (m, rad euler xyz)
  rm_algo_end2tool(pose)              — 同一 TCP，表达到「工作系 + 当前 tool 系」
  rm_algo_pose_move(pose, delta, 1)   — delta 在 Tool 系：平移 m，旋转 deg
  rm_set_movev_canfd_init(..., ft)    — ft=0 速度在 Tool 系；ft=1 速度在 Work 系
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

ROBOT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "rm75f_default.yaml"
CONFIG_YAML = ROBOT_CONFIG


def load_expected_tool_name() -> str:
    import yaml

    data = yaml.safe_load(CONFIG_YAML.read_text()) or {}
    return str(data.get("tool", {}).get("name", ""))


def print_tool_frames(robot, *, expected_name: str) -> None:
    ret, cur = robot.rm_get_current_tool_frame()
    if ret != 0:
        print(f"  rm_get_current_tool_frame failed: {ret}")
        return

    p = cur["pose"]
    print("\n=== Tool frame from controller (TCP vs flange / link7 end) ===")
    print("  rm_frame_t.pose = Tool origin relative to flange (m, rad euler xyz)")
    print(f"  ACTIVE tool: {cur['name']!r}")
    print(
        f"    offset xyz_mm=[{p[0]*1000:.1f},{p[1]*1000:.1f},{p[2]*1000:.1f}]  "
        f"euler_deg=[{math.degrees(p[3]):.1f},{math.degrees(p[4]):.1f},{math.degrees(p[5]):.1f}]"
    )
    print(
        f"    payload={cur['payload']:.3f} kg  "
        f"CoM in tool (controller units) x,y,z={cur['x']},{cur['y']},{cur['z']}"
    )

    total = robot.rm_get_total_tool_frame()
    names = total.get("tool_names", []) if isinstance(total, dict) else []
    print(f"  Stored tools on arm: {names}")

    if expected_name and expected_name not in names:
        print(
            f"  WARNING: configs/rm75f_default.yaml expects tool {expected_name!r} "
            f"but it is NOT in stored tools — activate/create it in Web UI.",
        )
    elif expected_name and cur["name"] != expected_name:
        print(
            f"  WARNING: yaml expects {expected_name!r} but ACTIVE is {cur['name']!r}.",
        )

    if expected_name and expected_name in names:
        ret_g, given = robot.rm_get_given_tool_frame(expected_name)
        if ret_g == 0:
            pg = given["pose"]
            print(f"  Stored {expected_name!r} offset xyz_mm="
                  f"[{pg[0]*1000:.1f},{pg[1]*1000:.1f},{pg[2]*1000:.1f}]")


def pose_to_rm_pose(pose: list[float]):
    from Robotic_Arm.rm_ctypes_wrap import rm_euler_t, rm_pose_t, rm_position_t

    po = rm_pose_t()
    po.position = rm_position_t(*pose[:3])
    po.euler = rm_euler_t(*pose[3:6])
    return po


def fmt6(v: list[float]) -> str:
    return (
        f"xyz=[{v[0]:+.4f},{v[1]:+.4f},{v[2]:+.4f}]  "
        f"euler=[{v[3]:+.4f},{v[4]:+.4f},{v[5]:+.4f}]"
    )


def tool_offset_pose(robot, ref: list[float], dx: float, dy: float, dz: float) -> list[float]:
    return robot.rm_algo_pose_move(ref, [dx, dy, dz, 0.0, 0.0, 0.0], frameMode=1)


def main() -> int:
    p = argparse.ArgumentParser(description="RM75 frame diagnostic")
    p.add_argument(
        "--probe-y-mm",
        type=float,
        default=0.0,
        help="apply +dy in Tool frame via rm_algo_pose_move (no motion, math only)",
    )
    args = p.parse_args()

    from rm75_control import RobotSession

    print("=== RM75 coordinate frames ===\n")
    print("Work/Base (state.pose)     — rm_get_current_arm_state, rm_movej_p target")
    print("Tool (active TCP)          — rm_get_current_tool_frame(); gripper Z≈220mm on this arm")
    print("rm_algo_end2tool(pose)     — same TCP, numeric pose in work+tool convention")
    print("rm_algo_pose_move(..., 1)  — delta [dx,dy,dz,0,0,0] in Tool frame (m, deg)")
    print("rm_set_movev_canfd_init ft  — 0: v in Tool; 1: v in Work (per rm_interface.h)\n")

    with RobotSession(config=ROBOT_CONFIG) as bot:
        expected_tool = load_expected_tool_name()
        print_tool_frames(bot.robot, expected_name=expected_tool)

        ret, st = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            print(f"get state failed: {ret}", file=sys.stderr)
            return 1

        pose = list(st["pose"][:6])
        tool = list(bot.robot.rm_algo_end2tool(pose_to_rm_pose(pose)))

        print("Current sample:")
        print(f"  state.pose (work)  {fmt6(pose)}")
        print(f"  end2tool           {fmt6(tool)}")

        r = Rsc.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
        col_x, col_y, col_z = r[:, 0], r[:, 1], r[:, 2]
        print("\nTool axes unit vectors in Work frame (from TCP euler xyz):")
        print(f"  tool X -> work {col_x}")
        print(f"  tool Y -> work {col_y}")
        print(f"  tool Z -> work {col_z}")

        dy_m = args.probe_y_mm / 1000.0
        if abs(dy_m) > 1e-9:
            pose_y = tool_offset_pose(bot.robot, pose, 0.0, dy_m, 0.0)
            tool_y = list(bot.robot.rm_algo_end2tool(pose_to_rm_pose(pose_y)))
            d_base = np.array(pose_y[:3]) - np.array(pose[:3])
            d_tool = np.array(tool_y[:3]) - np.array(tool[:3])
            print(f"\nProbe Tool Y +{args.probe_y_mm:.1f} mm (rm_algo_pose_move frameMode=1):")
            print(f"  base xyz delta (m)   {d_base}  |Δ|={np.linalg.norm(d_base)*1000:.2f} mm")
            print(f"  end2tool xyz delta   {d_tool}  |Δ|={np.linalg.norm(d_tool)*1000:.2f} mm")
            print(f"  end2tool dy          {d_tool[1]*1000:+.2f} mm  (may != dy if end2tool is not pure offset coords)")
            proj_y = float(np.dot(d_base, col_y))
            print(f"  base motion projected on tool Y axis: {proj_y*1000:+.2f} mm  ← use this to verify Y")

        print("\nRecommended for Y sin scan:")
        print("  movev init frame_type=0  → send [0, vy, 0, 0, 0, 0] = Tool Y velocity")
        print("  ref pose for P loop: end2tool()[1] vs y0 + dy  (same as sin_y_movev_canfd.py)")
        print("  geometric ref: tool_offset_pose(base, 0, dy, 0) for base target pose")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
