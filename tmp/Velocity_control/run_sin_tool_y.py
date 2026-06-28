#!/usr/bin/env python3
"""
Tool TCP Y 正弦速度 — 仅 RealMan 官方两个接口。

  rm_set_movev_canfd_init(avoid, frame_type, dt)   # 一次
  rm_movev_canfd([0, vy, 0, 0, 0, 0], ...)       # 每 dt 循环

  推荐: follow=True, trajectory_mode=0（完全透传）, avoid=0, frame_type=0
  mode 1/2 + radio 在本场景实测不如透传。

  source env.sh
  python tmp/Velocity_control/run_sin_tool_y.py --pose-slot d --log --duration 55
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from rm75_control import RobotSession
from rm75_control.control.velocity_admittance.rm_algo import end2tool_pose
from rm75_control.force.compensation.collection import load_slot, move_j, wait_settle
from rm75_control.force.compensation.id_config import load_config
from rm75_control.force.compensation.paths import CONFIG_ID

ROBOT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "rm75f_default.yaml"
LOG_DIR = Path(__file__).resolve().parent / "logs"


def sin_period(amp_m: float, vmax_m_s: float) -> float:
    return 2.0 * math.pi * amp_m / vmax_m_s if amp_m > 0 and vmax_m_s > 0 else 1.0


def vy_sin(t_s: float, amp_m: float, omega: float, ramp_s: float) -> float:
    """TCP Y velocity; ramp_s>0 soft-starts from vy=0 (avoids step at scan onset)."""
    vy = amp_m * omega * math.cos(omega * t_s)
    if ramp_s > 0.0 and t_s < ramp_s:
        vy *= math.sin(0.5 * math.pi * t_s / ramp_s)
    return vy


def default_log_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"sin_tool_y_{stamp}.npz"


def wrap_angle_rad(d: np.ndarray) -> np.ndarray:
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def print_log_summary(
    *,
    log_path: Path,
    t: np.ndarray,
    vy_cmd: np.ndarray,
    pose: np.ndarray,
    tool: np.ndarray,
    expected_pp_mm: float,
    hold_s: float,
) -> None:
    if len(t) < 2:
        print("  log: too few samples for summary", flush=True)
        return

    scan = t >= hold_s
    pose_s = pose[scan] if np.any(scan) else pose
    tool_s = tool[scan] if np.any(scan) else tool

    y0 = float(tool_s[0, 1])
    y_mm = (tool_s[:, 1] - y0) * 1000.0
    stroke_mm = float(np.max(y_mm) - np.min(y_mm))
    y_center_mm = float(0.5 * (np.max(y_mm) + np.min(y_mm)))

    pose0 = pose_s[0]
    d_pos_mm = (pose_s[:, :3] - pose0[:3]) * 1000.0
    d_euler_deg = np.degrees(wrap_angle_rad(pose_s[:, 3:6] - pose0[3:6]))
    rot_max_deg = float(np.max(np.abs(d_euler_deg)))

    pct = f"  ({100.0 * stroke_mm / expected_pp_mm:.0f}%)" if expected_pp_mm > 0 else ""
    print("\n=== drift summary (from logged feedback) ===", flush=True)
    print(
        f"  samples={len(t)}  scan_samples={int(np.sum(scan))}  "
        f"tool_y stroke={stroke_mm:.1f} mm  "
        f"expected p-p={expected_pp_mm:.1f} mm{pct}",
        flush=True,
    )
    print(
        f"  tool_y center drift={y_center_mm:+.1f} mm  "
        f"(midpoint vs scan start; 0 = no center shift)",
        flush=True,
    )
    print(
        f"  base |Δx| max={float(np.max(np.abs(d_pos_mm[:, 0]))):.1f} mm  "
        f"|Δz| max={float(np.max(np.abs(d_pos_mm[:, 2]))):.1f} mm  "
        f"rot max={rot_max_deg:.2f}°",
        flush=True,
    )
    print(f"  saved → {log_path}", flush=True)


def save_log_npz(
    path: Path,
    *,
    t: np.ndarray,
    vy_cmd: np.ndarray,
    pose: np.ndarray,
    tool: np.ndarray,
    q_deg: np.ndarray,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        t=t,
        vy_cmd=vy_cmd,
        pose=pose,
        tool=tool,
        q_deg=q_deg,
        **{k: np.asarray(v) for k, v in meta.items()},
    )


def send_zero(robot, *, follow: bool, traj: int, radio: int, n: int, dt_s: float) -> None:
    zero = [0.0] * 6
    tick = time.monotonic()
    for _ in range(n):
        now = time.monotonic()
        if now < tick:
            time.sleep(tick - now)
        tick += dt_s
        robot.rm_movev_canfd(zero, follow, traj, radio)


def main() -> int:
    p = argparse.ArgumentParser(description="Tool Y sin via rm_movev_canfd (official API only)")
    p.add_argument("--peak-to-peak-mm", type=float, default=160.0)
    p.add_argument("--max-vel-cm-s", type=float, default=2.0)
    p.add_argument("--dt-ms", type=float, default=10.0)
    p.add_argument("--frame-type", type=int, default=0, choices=[0, 1])
    p.add_argument("--avoid-singularity", type=int, default=0, choices=[0, 1])
    p.add_argument("--follow", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trajectory-mode", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--radio", type=int, default=0)
    p.add_argument("--hold-s", type=float, default=1.0, help="init 后发零速静止")
    p.add_argument("--settle-frames", type=int, default=30, help="init 后零速帧数")
    p.add_argument("--ramp-s", type=float, default=2.0, help="扫描开始后 vy 软启动秒数")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument(
        "--pose-slot",
        type=str,
        default=None,
        choices=["a", "b", "c", "d"],
        help="move_j 到 force_id poses.yaml 中该位姿后再 init（与力采集相同起点）",
    )
    p.add_argument("--move-speed", type=int, default=None, help="move_j 速度，默认读 force_id.yaml")
    p.add_argument(
        "--log",
        nargs="?",
        const="auto",
        default=None,
        metavar="NPZ",
        help="记录 pose/end2tool 并结束时打印 stroke/中心漂/姿态漂；省略路径则写 logs/sin_tool_y_<ts>.npz",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="每 N 个控制周期采样一次反馈（默认 10 → 100ms，避免每拍读 state 拖慢循环）",
    )
    args = p.parse_args()

    amp_m = args.peak_to_peak_mm / 2000.0
    vmax_m_s = args.max_vel_cm_s / 100.0
    period = sin_period(amp_m, vmax_m_s)
    omega = 2.0 * math.pi / period
    dt_s = args.dt_ms / 1000.0

    print(
        f"rm_movev_canfd | frame_type={args.frame_type} dt={args.dt_ms}ms "
        f"traj={args.trajectory_mode} avoid={args.avoid_singularity}",
        flush=True,
    )
    print(
        f"  ±{args.peak_to_peak_mm/20:.1f}cm p-p  v_peak={args.max_vel_cm_s}cm/s  "
        f"period={period:.1f}s  hold={args.hold_s}s  ramp={args.ramp_s}s  Ctrl+C stop",
        flush=True,
    )

    with RobotSession(config=ROBOT_CONFIG) as bot:
        if args.pose_slot:
            fid = load_config(CONFIG_ID)
            move_speed = args.move_speed if args.move_speed is not None else fid.collect.move_speed
            q_tgt, pose_tgt, rec = load_slot(fid, args.pose_slot)
            print(
                f"move_j → pose {args.pose_slot} ({rec.get('label', '')}) "
                f"speed={move_speed}",
                flush=True,
            )
            move_j(bot.robot, q_tgt, speed=move_speed)
            pose_act, q_act = wait_settle(
                bot.robot, q_tgt, timeout_s=fid.collect.settle_timeout_s,
            )
            print(
                f"  settled q_max_err={float(np.max(np.abs(q_act - q_tgt))):.3f}° "
                f"pose={[round(v, 4) for v in pose_act]}",
                flush=True,
            )

        # 退出上一模式，避免 init 时位置/模式跳变（标定脚本同款）
        bot.robot.rm_set_arm_slow_stop()
        time.sleep(0.3)
        try:
            bot.robot.rm_set_arm_delete_trajectory()
        except Exception:
            pass
        time.sleep(0.2)

        ret = bot.robot.rm_set_movev_canfd_init(
            args.avoid_singularity, args.frame_type, int(args.dt_ms),
        )
        if ret != 0:
            print(f"rm_set_movev_canfd_init failed: {ret}", file=sys.stderr)
            return 1

        send_zero(
            bot.robot, follow=args.follow, traj=args.trajectory_mode,
            radio=args.radio, n=args.settle_frames, dt_s=dt_s,
        )

        log_enabled = args.log is not None
        log_path = (
            default_log_path() if args.log == "auto" else Path(args.log)
        ) if log_enabled else None
        log_t: list[float] = []
        log_vy: list[float] = []
        log_pose: list[list[float]] = []
        log_tool: list[list[float]] = []
        log_q: list[list[float]] = []

        t0 = time.monotonic()
        n = 0
        try:
            while True:
                tick = t0 + n * dt_s
                now = time.monotonic()
                if now < tick:
                    time.sleep(tick - now)

                elapsed = tick - t0
                t_scan = elapsed - args.hold_s
                if args.duration is not None and t_scan >= args.duration:
                    break

                if elapsed < args.hold_s:
                    vy = 0.0
                else:
                    vy = vy_sin(t_scan, amp_m, omega, args.ramp_s)

                vel = [0.0, vy, 0.0, 0.0, 0.0, 0.0]
                ret = bot.robot.rm_movev_canfd(
                    vel, args.follow, args.trajectory_mode, args.radio,
                )
                if ret != 0:
                    print(f"rm_movev_canfd failed: {ret}", file=sys.stderr)
                    return 1

                if log_enabled and n % args.log_every == 0:
                    ret_s, st = bot.robot.rm_get_current_arm_state()
                    if ret_s == 0:
                        pose6 = list(st["pose"][:6])
                        log_t.append(elapsed)
                        log_vy.append(vy)
                        log_pose.append(pose6)
                        log_tool.append(end2tool_pose(bot.robot, pose6))
                        log_q.append(list(st["joint"][:7]))

                n += 1
                if n % 100 == 0 and t_scan >= 0:
                    print(f"  t={elapsed:.1f}s vy={vy*100:+.2f}cm/s", flush=True)
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
        finally:
            if log_enabled and log_path is not None and log_t:
                t_arr = np.asarray(log_t, dtype=float)
                vy_arr = np.asarray(log_vy, dtype=float)
                pose_arr = np.asarray(log_pose, dtype=float)
                tool_arr = np.asarray(log_tool, dtype=float)
                q_arr = np.asarray(log_q, dtype=float)
                save_log_npz(
                    log_path,
                    t=t_arr,
                    vy_cmd=vy_arr,
                    pose=pose_arr,
                    tool=tool_arr,
                    q_deg=q_arr,
                    meta={
                        "peak_to_peak_mm": args.peak_to_peak_mm,
                        "max_vel_cm_s": args.max_vel_cm_s,
                        "frame_type": args.frame_type,
                        "avoid_singularity": args.avoid_singularity,
                        "follow": args.follow,
                        "pose_slot": args.pose_slot or "",
                    },
                )
                print_log_summary(
                    log_path=log_path,
                    t=t_arr,
                    vy_cmd=vy_arr,
                    pose=pose_arr,
                    tool=tool_arr,
                    expected_pp_mm=args.peak_to_peak_mm,
                    hold_s=args.hold_s,
                )
            send_zero(
                bot.robot, follow=args.follow, traj=args.trajectory_mode,
                radio=args.radio, n=args.settle_frames, dt_s=dt_s,
            )
            bot.robot.rm_set_arm_slow_stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
