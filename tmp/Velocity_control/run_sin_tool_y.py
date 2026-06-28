#!/usr/bin/env python3
"""
Tool Y sin — pure velocity CANFD (same pattern as tmp/sin_y_movev_canfd.py).

movev frame_type=0 → velocity in Tool frame; vy = TCP Y axis.
Position P-term uses end2tool()[1] vs y0+dy (lag-aligned to fb time).

  source env.sh
  python tmp/Velocity_control/print_frames.py --probe-y-mm 10
  python tmp/Velocity_control/run_sin_tool_y.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from rm75_control import RobotSession
from rm75_control.control.cartesian_velocity import (
    CartesianVelocityController,
    CartesianVelocityStreamConfig,
    CartesianVelocityTracker,
    CartesianVelocityTrackerConfig,
)
from rm75_control.control.velocity_admittance.trajectory import sin_period_for_peak_vel
from rm75_control.core.exceptions import MotionError
from rm75_control.motion.canfd import send_velocity_canfd

CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_CONFIG = CONFIG_DIR / "sin_tool_y.yaml"
ROBOT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "rm75f_default.yaml"
Y_AXIS = 1


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


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


def sin_y_motion(t_s: float, amplitude_m: float, omega: float) -> tuple[float, float]:
    """±amplitude symmetric sin scan.  Slew limiter handles the startup velocity step."""
    dy = amplitude_m * math.sin(omega * t_s)
    vy = amplitude_m * omega * math.cos(omega * t_s)
    return dy, vy


def settle_zero_velocity(robot, *, dt_ms: float, follow: bool, traj: int, radio: int, n: int) -> None:
    dt_s = dt_ms / 1000.0
    zero = [0.0] * 6
    tick = time.monotonic()
    for _ in range(n):
        now = time.monotonic()
        if now < tick:
            time.sleep(min(0.002, tick - now))
        tick += dt_s
        send_velocity_canfd(robot, zero, follow=follow, trajectory_mode=traj, radio=radio)


@dataclass
class LagStats:
    n: int = 0
    max_y_err_mm: float = 0.0
    max_x_drift_mm: float = 0.0
    max_z_drift_mm: float = 0.0
    max_jitter_ms: float = 0.0

    def update(
        self,
        y_ref: float,
        y_fb: float,
        x_fb: float,
        z_fb: float,
        x0: float,
        z0: float,
        jitter_ms: float,
    ) -> None:
        self.n += 1
        self.max_y_err_mm = max(self.max_y_err_mm, abs(y_ref - y_fb) * 1000.0)
        self.max_x_drift_mm = max(self.max_x_drift_mm, abs(x_fb - x0) * 1000.0)
        self.max_z_drift_mm = max(self.max_z_drift_mm, abs(z_fb - z0) * 1000.0)
        self.max_jitter_ms = max(self.max_jitter_ms, jitter_ms)

    def line(self, t: float, vy_cm_s: float, y_fb_mm: float) -> str:
        if self.n == 0:
            return ""
        return (
            f"t={t:.1f}s y_tool={y_fb_mm:.1f}mm vy={vy_cm_s:+.2f}cm/s  "
            f"Y err_max={self.max_y_err_mm:.1f}mm  "
            f"tool_drift X={self.max_x_drift_mm:.1f} Z={self.max_z_drift_mm:.1f}mm  "
            f"jitter_max={self.max_jitter_ms:.1f}ms"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Tool Y sin velocity (frame_type=0)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--amplitude-mm", type=float, default=None, help="peak ± mm (150 = ±15 cm)")
    parser.add_argument("--max-vel-cm-s", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--open-loop", action="store_true")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    timing = raw.get("timing", {})
    traj_cfg = raw.get("trajectory", {})
    vc = raw.get("velocity_canfd", {})
    ctrl = raw.get("control", {})

    dt_ms = float(timing.get("dt_ms", 10.0))
    dt_s = dt_ms / 1000.0
    settle_frames = int(timing.get("settle_frames", 20))
    hold_s = float(timing.get("hold_s", 1.0))
    fb_every = max(1, int(timing.get("fb_every", 10)))

    amp_mm = float(args.amplitude_mm if args.amplitude_mm is not None else traj_cfg.get("amplitude_mm", 150.0))
    vmax_cm_s = float(args.max_vel_cm_s if args.max_vel_cm_s is not None else traj_cfg.get("y_max_vel_cm_s", 1.5))
    amp_m = amp_mm / 1000.0
    vmax_m_s = vmax_cm_s / 100.0
    ps = traj_cfg.get("period_s")
    period = sin_period_for_peak_vel(amp_m, vmax_m_s) if ps is None else float(ps)
    omega = 2.0 * math.pi / period if period > 0 else 0.0
    v_peak = amp_m * omega

    follow = bool(vc.get("follow", True))
    traj_mode = int(vc.get("trajectory_mode", 1))
    radio = int(vc.get("radio", 40))
    avoid = int(vc.get("avoid_singularity", 0))

    pos_kp = float(ctrl.get("pos_kp_y", 0.8))
    pos_ki = float(ctrl.get("pos_ki_y", 0.0))
    hold_kp = float(ctrl.get("hold_kp", 1.0))
    pos_closure = not args.open_loop

    print(
        f"sin_tool_y | movev frame_type=0 (Tool v) avoid={avoid} follow={follow} "
        f"traj={traj_mode} radio={radio}",
        flush=True,
    )
    print(
        f"  peak ±{amp_mm/10:.1f}cm ({amp_mm:.0f}mm)  v_peak={v_peak*100:.2f}cm/s  "
        f"period={period:.1f}s  hold={hold_s:.1f}s  kp_y={pos_kp}  hold_kp={hold_kp}",
        flush=True,
    )
    print("  Run: python tmp/Velocity_control/print_frames.py --probe-y-mm 10", flush=True)

    with RobotSession(config=ROBOT_CONFIG) as bot:
        ret, state = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            print(f"get state failed: {ret}", file=sys.stderr)
            return 1

        pose0_tool = read_tool_pose(bot.robot)
        if pose0_tool is None:
            print("read tool pose failed", file=sys.stderr)
            return 1
        x0, y0, z0 = pose0_tool[0], pose0_tool[1], pose0_tool[2]

        # Rotation matrix: columns = Tool X/Y/Z axes in Work frame.
        # Used to convert Work-frame drift into Tool-frame coordinates for reporting.
        import numpy as _np
        from scipy.spatial.transform import Rotation as _Rsc
        _euler = list(bot.robot.rm_get_current_arm_state()[1]["pose"])[3:6]
        _R = _Rsc.from_euler("xyz", _euler, degrees=False).as_matrix()  # Work←Tool

        ret_init = bot.robot.rm_set_movev_canfd_init(avoid, 0, int(dt_ms))
        if ret_init != 0:
            raise MotionError(f"rm_set_movev_canfd_init failed: {ret_init}")

        # Y axis only: ff + P closure.
        # No hold correction on X/Z/rot — Tool X/Z ≈ -Work X/Z so hold_kp in tool frame
        # would apply corrections in the wrong direction and actively destabilize the arm.
        # The robot's internal control maintains other DOF when only Y velocity is commanded.
        tracker_cfg = CartesianVelocityTrackerConfig.for_motion_axes(
            (Y_AXIS,),
            kp=pos_kp if pos_closure else 0.0,
            ki=pos_ki if pos_closure else 0.0,
            ref_speed_peak_m_s=v_peak,
        )
        vel_ctrl = CartesianVelocityController(
            bot.robot,
            tracker=CartesianVelocityTracker(tracker_cfg),
            config=CartesianVelocityStreamConfig(
                follow=follow,
                trajectory_mode=traj_mode,
                radio=radio,
                period_ms=dt_ms,
            ),
        )

        settle_zero_velocity(
            bot.robot, dt_ms=dt_ms, follow=follow, traj=traj_mode, radio=radio, n=settle_frames,
        )
        vel_ctrl.reset_slew()
        print(f"  tool0 end2tool: {[round(v, 4) for v in pose0_tool]}", flush=True)
        print("Ctrl+C to stop.", flush=True)

        t_start = time.monotonic()
        cmd_count = 0
        last_fb = list(pose0_tool)
        last_fb_t = t_start
        stats = LagStats()
        last_log = t_start
        last_jitter_ms = 0.0

        try:
            while True:
                tick = t_start + cmd_count * dt_s
                now = time.monotonic()
                if now < tick:
                    time.sleep(tick - now)
                last_jitter_ms = (time.monotonic() - tick) * 1000.0

                t_now = tick - t_start
                t_scan = t_now - hold_s
                if args.duration is not None and t_scan >= args.duration:
                    break
                cmd_count += 1

                if t_now < hold_s:
                    y_ref_cmd = y0
                    vy_cmd = 0.0
                    y_ref_fb = y0
                else:
                    dy_cmd, vy_cmd = sin_y_motion(t_scan, amp_m, omega)
                    y_ref_cmd = y0 + dy_cmd
                    # P-term reference aligned to when feedback was actually sampled
                    t_fb_scan = last_fb_t - t_start - hold_s
                    dy_fb, _ = sin_y_motion(max(0.0, t_fb_scan), amp_m, omega)
                    y_ref_fb = y0 + dy_fb

                ref_pose = list(pose0_tool)
                ref_pose[Y_AXIS] = y_ref_fb
                ref_vel = [0.0, vy_cmd, 0.0, 0.0, 0.0, 0.0]

                vel = vel_ctrl.step(ref_pose=ref_pose, ref_vel=ref_vel, fb_pose=last_fb, dt_s=dt_s)

                if cmd_count % fb_every == 0:
                    fb = read_tool_pose(bot.robot)
                    if fb is not None:
                        last_fb = fb
                        last_fb_t = time.monotonic()

                if t_now >= hold_s and now - last_log >= 1.0:
                    last_log = now
                    # Convert Work-frame displacement to Tool frame for drift stats.
                    d_work = _np.array([last_fb[0]-x0, last_fb[1]-y0, last_fb[2]-z0])
                    d_tool = _R.T @ d_work  # Work→Tool: R^T
                    stats.update(
                        y_ref_cmd, last_fb[1], d_tool[0], d_tool[2], 0.0, 0.0, last_jitter_ms
                    )
                    print(
                        stats.line(t_now, vel[Y_AXIS] * 100.0, d_tool[1] * 1000.0),
                        flush=True,
                    )
                    stats = LagStats()

        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
        finally:
            try:
                vel_ctrl.send_velocity([0.0] * 6)
                bot.robot.rm_set_arm_slow_stop()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
