#!/usr/bin/env python3
"""
Tool TCP Y sinusoid via rm_movev_canfd + outer position loop.

Single-threaded: movev every 10ms on a fixed schedule; pose feedback decimated
AFTER each send so state RTT never stalls the stream. P term uses ref aligned to
feedback timestamp + gain scheduling + slew limit (not integral windup).

Example:
  source /media/camp/EXT_DRIVE/rm75_control/env.sh
  python /media/camp/EXT_DRIVE/rm75_control/tmp/sin_y_movev_canfd.py \\
    --amplitude-mm 80 --max-vel-cm-s 1.5
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rm75f_default.yaml"

Y_AXIS = 1


@dataclass
class LagStats:
    n: int = 0
    sum_ref_mm: float = 0.0
    sum_sq_ref_mm: float = 0.0
    max_ref_mm: float = 0.0
    sum_jitter_ms: float = 0.0
    max_jitter_ms: float = 0.0
    max_x_drift_mm: float = 0.0
    max_z_drift_mm: float = 0.0
    max_vy_jump_cm_s: float = 0.0

    def update(
        self,
        y_ref: float,
        y_fb: float,
        x_fb: float,
        z_fb: float,
        x0: float,
        z0: float,
        vy_cmd: float,
        prev_vy: float,
        jitter_ms: float,
    ) -> None:
        ref_e = (y_ref - y_fb) * 1000.0
        self.n += 1
        self.sum_ref_mm += abs(ref_e)
        self.sum_sq_ref_mm += ref_e * ref_e
        self.max_ref_mm = max(self.max_ref_mm, abs(ref_e))
        self.max_x_drift_mm = max(self.max_x_drift_mm, abs(x_fb - x0) * 1000.0)
        self.max_z_drift_mm = max(self.max_z_drift_mm, abs(z_fb - z0) * 1000.0)
        self.max_vy_jump_cm_s = max(
            self.max_vy_jump_cm_s, abs(vy_cmd - prev_vy) * 100.0
        )
        self.sum_jitter_ms += abs(jitter_ms)
        self.max_jitter_ms = max(self.max_jitter_ms, abs(jitter_ms))

    def report(self, t: float, y_fb_mm: float, vy_cmd_m_s: float) -> str:
        if self.n == 0:
            return ""
        mean_ref = self.sum_ref_mm / self.n
        rms_ref = math.sqrt(self.sum_sq_ref_mm / self.n)
        mean_jit = self.sum_jitter_ms / self.n
        return (
            f"t={t:.1f}s y_fb={y_fb_mm:.1f}mm vy_cmd={vy_cmd_m_s*100:.2f}cm/s | "
            f"ref→fb: mean={mean_ref:.2f} max={self.max_ref_mm:.2f} rms={rms_ref:.2f} mm | "
            f"drift x={self.max_x_drift_mm:.1f} z={self.max_z_drift_mm:.1f} mm | "
            f"max_dvy={self.max_vy_jump_cm_s:.3f}cm/s/step | "
            f"loop jitter: mean={mean_jit:.2f} max={self.max_jitter_ms:.2f} ms"
        )

    def reset_window(self) -> None:
        self.n = 0
        self.sum_ref_mm = 0.0
        self.sum_sq_ref_mm = 0.0
        self.max_ref_mm = 0.0
        self.sum_jitter_ms = 0.0
        self.max_jitter_ms = 0.0
        self.max_x_drift_mm = 0.0
        self.max_z_drift_mm = 0.0
        self.max_vy_jump_cm_s = 0.0


def measure_state_rtt(robot, n: int = 5) -> float:
    samples: list[float] = []
    for _ in range(n):
        t0 = time.monotonic()
        ret, _ = robot.rm_get_current_arm_state()
        if ret == 0:
            samples.append((time.monotonic() - t0) * 1000.0)
    return sum(samples) / len(samples) if samples else float("nan")


def sin_period_for_peak_vel(amplitude_m: float, max_vel_m_s: float) -> float:
    if amplitude_m <= 0.0 or max_vel_m_s <= 0.0:
        return 1.0
    return 2.0 * math.pi * amplitude_m / max_vel_m_s


def y_ref_at(t: float, y0: float, amplitude_m: float, omega: float) -> float:
    return y0 + amplitude_m * math.sin(omega * t)


def vy_at(t: float, amplitude_m: float, omega: float) -> float:
    return amplitude_m * omega * math.cos(omega * t)


def pose_to_rm_pose(pose: list[float]):
    from Robotic_Arm.rm_ctypes_wrap import rm_euler_t, rm_pose_t, rm_position_t

    po = rm_pose_t()
    po.position = rm_position_t(*pose[:3])
    po.euler = rm_euler_t(*pose[3:6])
    return po


def read_tool_pose(robot) -> list[float] | None:
    ret, state = robot.rm_get_current_arm_state()
    if ret != 0:
        return None
    return list(robot.rm_algo_end2tool(pose_to_rm_pose(list(state["pose"]))))


def build_ref_tool_pose(pose0_tool: list[float], y_ref: float) -> list[float]:
    ref = list(pose0_tool)
    ref[Y_AXIS] = y_ref
    return ref


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tool TCP Y sin via rm_movev_canfd",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--amplitude-mm", type=float, default=80.0)
    parser.add_argument("--max-vel-cm-s", type=float, default=1.5)
    parser.add_argument("--period", type=float, default=None)
    parser.add_argument("--dt-ms", type=float, default=10.0)
    parser.add_argument(
        "--open-loop",
        action="store_true",
        help="vy feedforward only (will drift)",
    )
    parser.add_argument(
        "--pos-kp",
        type=float,
        default=0.8,
        help="Y P gain (m/s per m); reduced automatically at peak |vy|",
    )
    parser.add_argument("--pos-ki", type=float, default=0.0)
    parser.add_argument(
        "--fb-every",
        type=int,
        default=10,
        help="refresh pose after movev every N cycles (~100ms at N=10, dt=10ms)",
    )
    parser.add_argument(
        "--avoid-singularity",
        action="store_true",
        help="rm_set_movev_canfd_init avoid_singularity_flag=1",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="low follow (debug lag only; motion becomes unstable)",
    )
    parser.add_argument("--trajectory-mode", type=int, default=-1, choices=[-1, 0, 1, 2])
    parser.add_argument("--radio", type=int, default=-1)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--lag-every", type=int, default=10)
    args = parser.parse_args()

    amplitude_m = args.amplitude_mm / 1000.0
    dt_s = args.dt_ms / 1000.0
    max_vel_m_s = args.max_vel_cm_s / 100.0
    pos_closure = not args.open_loop
    follow = not args.no_follow

    if args.trajectory_mode == -1:
        trajectory_mode = 1 if follow else 0
    else:
        trajectory_mode = args.trajectory_mode
    if args.radio == -1:
        radio = 40 if follow and trajectory_mode == 1 else 0
    else:
        radio = args.radio
    if not follow and trajectory_mode != 0:
        trajectory_mode = 0
        radio = 0

    if args.period is None:
        period = sin_period_for_peak_vel(amplitude_m, max_vel_m_s)
    else:
        period = args.period
        v_peak = amplitude_m * (2.0 * math.pi / period)
        if v_peak > max_vel_m_s * 1.001:
            period = sin_period_for_peak_vel(amplitude_m, max_vel_m_s)
            print(
                f"period extended to {period:.2f}s to cap Y speed at "
                f"{args.max_vel_cm_s:.2f} cm/s",
                flush=True,
            )

    omega = 2.0 * math.pi / period
    v_peak = amplitude_m * omega

    from rm75_control import RobotSession
    from rm75_control.control.cartesian_velocity import (
        CartesianVelocityController,
        CartesianVelocityStreamConfig,
        CartesianVelocityTracker,
        CartesianVelocityTrackerConfig,
    )
    from rm75_control.core.exceptions import MotionError

    tracker_cfg = CartesianVelocityTrackerConfig.for_motion_axes(
        (Y_AXIS,),
        kp=args.pos_kp if pos_closure else 0.0,
        ki=args.pos_ki if pos_closure else 0.0,
        ref_speed_peak_m_s=v_peak,
    )

    print("Connecting...", flush=True)
    with RobotSession(config=CONFIG) as bot:
        ret, state = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            print(f"get state failed: {ret}", file=sys.stderr)
            return 1

        pose0_tool = read_tool_pose(bot.robot)
        if pose0_tool is None:
            print("failed to read start tool pose", file=sys.stderr)
            return 1
        x0, y0, z0 = pose0_tool[0], pose0_tool[1], pose0_tool[2]

        state_rtt_ms = measure_state_rtt(bot.robot)
        fb_ms = args.fb_every * args.dt_ms
        print(
            f"rm_get_current_arm_state RTT ~ {state_rtt_ms:.1f} ms; "
            f"pose refresh after movev every {args.fb_every} cycles (~{fb_ms:.0f}ms)",
            flush=True,
        )

        ret_init = bot.robot.rm_set_movev_canfd_init(
            1 if args.avoid_singularity else 0,
            0,
            int(args.dt_ms),
        )
        if ret_init != 0:
            raise MotionError(f"rm_set_movev_canfd_init failed with code {ret_init}")

        vel_ctrl = CartesianVelocityController(
            bot.robot,
            tracker=CartesianVelocityTracker(tracker_cfg),
            config=CartesianVelocityStreamConfig(
                follow=follow,
                trajectory_mode=trajectory_mode,
                radio=radio,
                period_ms=args.dt_ms,
            ),
        )

        loop_label = (
            f"pos_closure Kp_y={args.pos_kp} Ki_y={args.pos_ki} (gain-scheduled, slew-lim)"
            if pos_closure
            else "open-loop feedforward"
        )
        print("start tool pose (m, rad):", [round(v, 6) for v in pose0_tool])
        print(
            f"movev CANFD | tool Y +/-{args.amplitude_mm:.0f}mm "
            f"period={period:.2f}s peak_vel={v_peak*100:.2f}cm/s "
            f"dt={args.dt_ms}ms follow={follow} traj={trajectory_mode} radio={radio} | "
            f"{loop_label} | Ctrl+C stop",
            flush=True,
        )

        t_start = time.monotonic()
        cmd_count = 0
        stats = LagStats()
        last_fb = list(pose0_tool)
        last_fb_t = t_start
        last_vy = 0.0
        prev_vy_for_stats = 0.0
        last_jitter_ms = 0.0

        try:
            while True:
                tick = t_start + cmd_count * dt_s
                now = time.monotonic()
                if now < tick:
                    time.sleep(tick - now)
                last_jitter_ms = (time.monotonic() - tick) * 1000.0
                t_now = tick - t_start

                t_fb = last_fb_t - t_start
                y_ref_fb = y_ref_at(t_fb, y0, amplitude_m, omega)
                ref_pose = build_ref_tool_pose(pose0_tool, y_ref_fb)
                ref_vel = [0.0, vy_at(t_now, amplitude_m, omega), 0.0, 0.0, 0.0, 0.0]

                vel = vel_ctrl.step(
                    ref_pose=ref_pose,
                    ref_vel=ref_vel,
                    fb_pose=last_fb,
                    dt_s=dt_s,
                )
                prev_vy_for_stats = last_vy
                last_vy = vel[Y_AXIS]
                cmd_count += 1

                if cmd_count % args.fb_every == 0:
                    fb = read_tool_pose(bot.robot)
                    if fb is not None:
                        last_fb = fb
                        last_fb_t = time.monotonic()

                if cmd_count % args.lag_every == 0:
                    y_ref_sample = y_ref_at(t_now, y0, amplitude_m, omega)
                    stats.update(
                        y_ref_sample,
                        last_fb[1],
                        last_fb[0],
                        last_fb[2],
                        x0,
                        z0,
                        last_vy,
                        prev_vy_for_stats,
                        last_jitter_ms,
                    )

                if cmd_count % args.report_every == 0:
                    t_report = time.monotonic() - t_start
                    print(
                        stats.report(t_report, last_fb[Y_AXIS] * 1000.0, last_vy),
                        flush=True,
                    )
                    stats.reset_window()
        finally:
            try:
                vel_ctrl.send_velocity([0.0] * 6)
            except MotionError:
                pass
            bot.stop_all()

    print("stopped.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCtrl+C", file=sys.stderr)
        raise SystemExit(130)
