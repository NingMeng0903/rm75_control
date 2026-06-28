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
    p.add_argument(
        "--vel-probe",
        choices=["ft0", "ft1", "none"],
        default="none",
        help=(
            "physically move: ft0=frame_type=0 [0,vy,0..], ft1=frame_type=1 [0,vy,0..]. "
            "Sends 2 cm/s for 0.3 s (~6 mm), measures actual work-frame TCP displacement "
            "and compares with tool-Y direction."
        ),
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

        if args.vel_probe != "none":
            ft = 0 if args.vel_probe == "ft0" else 1
            _vel_probe(bot.robot, frame_type=ft, col_y=col_y, dt_ms=10, vy_m_s=0.02, n_cycles=30)

    return 0


def _vel_probe(robot, *, frame_type: int, col_y, dt_ms: float, vy_m_s: float, n_cycles: int) -> None:
    """Send [0, vy, 0, 0, 0, 0] for n_cycles at dt_ms, measure actual TCP displacement."""
    import time
    from rm75_control.motion.canfd import send_velocity_canfd

    print(f"\n=== vel_probe: frame_type={frame_type}  [0, {vy_m_s:.3f}, 0, 0, 0, 0] ===")
    print(f"  Moving {n_cycles * dt_ms:.0f} ms at {vy_m_s*100:.1f} cm/s → expected ~{n_cycles*dt_ms/1000*vy_m_s*1000:.1f} mm")
    print("  KEEP CLEAR — robot will move ~6 mm")

    ret0, st0 = robot.rm_get_current_arm_state()
    if ret0 != 0:
        print(f"  get_state failed: {ret0}")
        return
    pose_before = list(st0["pose"][:3])

    ret_init = robot.rm_set_movev_canfd_init(0, frame_type, int(dt_ms))
    if ret_init != 0:
        print(f"  rm_set_movev_canfd_init failed: {ret_init}")
        return

    # settle
    for _ in range(5):
        send_velocity_canfd(robot, [0.0]*6, follow=True, trajectory_mode=0, radio=0)
        time.sleep(dt_ms / 1000.0)

    vel_cmd = [0.0, vy_m_s, 0.0, 0.0, 0.0, 0.0]
    t0 = time.monotonic()
    for i in range(n_cycles):
        tick = t0 + i * dt_ms / 1000.0
        now = time.monotonic()
        if now < tick:
            time.sleep(tick - now)
        send_velocity_canfd(robot, vel_cmd, follow=True, trajectory_mode=0, radio=0)

    # stop
    for _ in range(5):
        send_velocity_canfd(robot, [0.0]*6, follow=True, trajectory_mode=0, radio=0)
        time.sleep(dt_ms / 1000.0)
    time.sleep(0.1)
    robot.rm_set_arm_slow_stop()
    time.sleep(0.3)

    ret1, st1 = robot.rm_get_current_arm_state()
    if ret1 != 0:
        print(f"  get_state after failed: {ret1}")
        return
    pose_after = list(st1["pose"][:3])

    delta = np.array(pose_after) - np.array(pose_before)
    dist_mm = np.linalg.norm(delta) * 1000.0
    if dist_mm < 0.1:
        print("  TCP did not move — check robot is not in E-stop or paused")
        return

    delta_unit = delta / np.linalg.norm(delta)
    proj_tool_y = float(np.dot(delta_unit, col_y))
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, proj_tool_y))))

    print(f"  actual displacement (work xyz mm): {delta * 1000}")
    print(f"  displacement magnitude: {dist_mm:.2f} mm")
    print(f"  tool Y unit vec (work): {np.round(col_y, 4)}")
    print(f"  cos(angle to tool Y): {proj_tool_y:.4f}  → angle = {angle_deg:.1f} deg")
    if angle_deg < 10.0:
        print("  ✓ motion IS along tool Y  →  frame_type works as expected")
    elif angle_deg > 70.0:
        print("  ✗ motion is nearly PERPENDICULAR to tool Y  →  frame_type meaning reversed?")
    else:
        print(f"  ? motion has {angle_deg:.1f}° offset from tool Y  →  partial mismatch")


if __name__ == "__main__":
    raise SystemExit(main())
