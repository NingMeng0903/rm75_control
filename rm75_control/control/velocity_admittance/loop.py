"""Shared velocity-admittance control loop."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from rm75_control.motion.canfd import send_velocity_canfd

from .controller import AdmittanceConfig, AdmittanceController
from .observer import CompensatedForceObserver
from .paths import CONFIG_ROBOT, PHI_JSON
from .rm_algo import end2tool_xyz
from .trajectory import TrajectoryGenerator, sin_period_for_peak_vel


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def init_velocity_canfd(robot, vc: dict, dt_ms: float) -> None:
    ret = robot.rm_set_movev_canfd_init(
        int(vc.get("avoid_singularity", 1)),
        int(vc.get("frame_type", 1)),
        int(dt_ms),
    )
    if ret != 0:
        raise RuntimeError(f"rm_set_movev_canfd_init failed: {ret}")


def settle_movev_after_init(
    robot,
    *,
    dt_ms: float,
    follow: bool,
    trajectory_mode: int,
    radio: int,
    n_frames: int = 10,
) -> None:
    """Zero-velocity frames after rm_set_movev_canfd_init — cuts mode-switch jerk."""
    dt_s = dt_ms / 1000.0
    zero = [0.0] * 6
    next_tick = time.monotonic()
    for _ in range(n_frames):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
        next_tick += dt_s
        send_velocity_canfd(
            robot, zero,
            follow=follow, trajectory_mode=trajectory_mode, radio=radio,
        )


def trajectory_summary(raw: dict) -> str:
    t = raw.get("trajectory", {})
    kind = str(t.get("type", "sin_tool_y"))
    amp_mm = float(t.get("amplitude_mm", 5.0))
    vmax = float(t.get("y_max_vel_cm_s", 1.0))
    ps = t.get("period_s")
    if ps is None:
        period = sin_period_for_peak_vel(amp_mm / 1000.0, vmax / 100.0)
        period_s = f"{period:.1f}s (auto)"
    else:
        period_s = f"{float(ps):.1f}s"
    soft = " soft_start" if t.get("soft_start") else ""
    return (
        f"{kind}{soft}  amp=±{amp_mm:.1f}mm ({2 * amp_mm:.0f}mm p-p)  "
        f"v_peak≈{vmax:.1f}cm/s  period={period_s}"
    )


def _hold_controller_config(cfg: AdmittanceConfig) -> AdmittanceConfig:
    """All position hold — no force axis until contact latched."""
    fa = np.zeros(6, dtype=float)
    kp = cfg.kp_pos.copy()
    kp[0] = min(float(kp[0]), 4.0)
    kp[1] = min(float(kp[1]), 4.0)
    kp[2] = 3.0
    kp[3:6] = np.maximum(kp[3:6], 10.0)
    mv = cfg.max_velocity.copy()
    vz_cap = cfg.max_vz_tool_m_s
    mv[2] = min(float(mv[2]), vz_cap)
    mv[3:6] = np.minimum(mv[3:6], 0.08)
    ma = cfg.max_acceleration.copy()
    ma[2] = min(float(ma[2]), 0.05)
    ma[3:6] = np.minimum(ma[3:6], 0.15)
    return replace(
        cfg,
        force_axes=fa,
        motion_axes=np.zeros(6),
        lock_orientation=True,
        enable_normal_tracking=False,
        kp_pos=kp,
        max_velocity=mv,
        max_acceleration=ma,
    )


def run_velocity_admittance(
    raw: dict,
    *,
    title: str = "Velocity admittance",
    duration_s: float | None = None,
    tool_hint: bool = True,
) -> int:
    if not PHI_JSON.exists():
        raise SystemExit(f"Missing {PHI_JSON} — run force_calibrate.py first")

    timing = raw.get("timing", {})
    dt_ms = float(timing.get("dt_ms", 10.0))
    dt_s = dt_ms / 1000.0
    vc = raw.get("velocity_canfd", {})
    follow = bool(vc.get("follow", True))
    traj_mode = int(vc.get("trajectory_mode", 0))
    radio = int(vc.get("radio", 0))

    startup = raw.get("startup", {})
    settle_frames = int(startup.get("settle_frames", 10))
    hold_s = float(startup.get("hold_s", 0.0))
    wait_contact = bool(startup.get("wait_contact", False))
    contact_fz_n = float(startup.get("contact_fz_n", 1.0))
    contact_samples = int(startup.get("contact_samples", 30))

    ctrl_cfg = AdmittanceConfig.from_dict(raw)
    hold_cfg = _hold_controller_config(ctrl_cfg)
    observer = CompensatedForceObserver.from_yaml(raw)
    controller = AdmittanceController(dt_s, ctrl_cfg)
    hold_ctrl = AdmittanceController(dt_s, hold_cfg)

    f_cfg = raw.get("force", {})
    desired_z = float(f_cfg.get("desired_z_n", 3.0))
    f_des = np.zeros(6)
    f_des[2] = desired_z
    f_zero = np.zeros(6)

    print(f"{title} | rm_movev_canfd follow={follow} traj={traj_mode} radio={radio}")
    print(f"  phi: {PHI_JSON.name}  Fz_des={desired_z:.2f} N")
    print(f"  vz cap (tool TCP): ±{ctrl_cfg.max_vz_tool_m_s * 100:.1f} cm/s")
    print(f"  trajectory: {trajectory_summary(raw)}")
    if wait_contact:
        print(
            f"  startup: settle={settle_frames}f hold={hold_s:.1f}s  "
            f"position hold until Fz_ext≥{contact_fz_n:.1f}N (no auto descent)",
            flush=True,
        )
    if tool_hint:
        print("  Ensure gripper (or desired tool) is active in RM Web UI before contact tasks.")

    from rm75_control import RobotSession

    with RobotSession(config=CONFIG_ROBOT) as bot:
        ret, state = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            raise SystemExit(f"get state failed: {ret}")
        pose0 = np.asarray(state["pose"][:6], dtype=float)
        pose_anchor = pose0.copy()
        motion_axes = ctrl_cfg.motion_axes.copy()
        print(
            f"  start TCP pose (base): "
            f"xyz=[{pose0[0]:.3f},{pose0[1]:.3f},{pose0[2]:.3f}] m",
            flush=True,
        )
        traj = TrajectoryGenerator.from_dict(raw, pose0, bot.robot)

        init_velocity_canfd(bot.robot, vc, dt_ms)
        settle_movev_after_init(
            bot.robot, dt_ms=dt_ms, follow=follow,
            trajectory_mode=traj_mode, radio=radio, n_frames=settle_frames,
        )
        hold_ctrl.reset()
        controller.reset()
        print("Velocity CANFD initialized. Ctrl+C to stop.", flush=True)

        t0 = time.monotonic()
        next_tick = t0
        last_log = t0
        contact_latched = not wait_contact
        contact_streak = 0
        t_scan0: float | None = None

        try:
            while True:
                now = time.monotonic()
                if duration_s is not None and t_scan0 is not None:
                    if now - t_scan0 >= duration_s:
                        break
                if now < next_tick:
                    time.sleep(min(0.002, next_tick - now))
                    continue
                next_tick += dt_s
                t_s = now - t0

                ret_s, st = bot.robot.rm_get_current_arm_state()
                ret_f, fd = bot.robot.rm_get_force_data()
                if ret_s != 0 or ret_f != 0:
                    continue

                pose = np.asarray(st["pose"][:6], dtype=float)
                force_raw = np.asarray(fd["force_data"][:6], dtype=float)
                observer.append(t_s, pose, force_raw)

                wrench = observer.latest_wrench()
                f_ext = wrench[1] if wrench is not None else f_zero

                if t_s < hold_s or not contact_latched:
                    v_cmd = hold_ctrl.compute_velocity_command(
                        pose, pose_anchor, np.zeros(6), f_ext, f_zero,
                        pose_anchor=pose_anchor,
                    )
                    send_velocity_canfd(
                        bot.robot, v_cmd.tolist(),
                        follow=follow, trajectory_mode=traj_mode, radio=radio,
                    )
                    if t_s < hold_s:
                        continue

                if not contact_latched:
                    if f_ext[2] >= contact_fz_n:
                        contact_streak += 1
                    else:
                        contact_streak = 0
                    if contact_streak >= contact_samples:
                        contact_latched = True
                        t_scan0 = now
                        pose_anchor = pose.copy()
                        controller.force_error_integral.fill(0.0)
                        controller.last_v_cmd = hold_ctrl.last_v_cmd.copy()
                        print(
                            f"  contact latched @ t={t_s:.1f}s  Fz_ext={f_ext[2]:+.2f}N  "
                            "— scan + Fz track ON (tool Y only, attitude locked)",
                            flush=True,
                        )
                    else:
                        continue

                t_scan = (now - t_scan0) if t_scan0 is not None else t_s
                pose_d, vel_ff = traj.sample(t_scan)
                pose_d = TrajectoryGenerator.blend_tool_pose(
                    bot.robot, pose_d, pose_anchor, motion_axes,
                )
                vel_ff = TrajectoryGenerator.project_tool_motion_ff(
                    bot.robot, pose_anchor, vel_ff, motion_axes,
                )
                v_cmd = controller.compute_velocity_command(
                    pose, pose_d, vel_ff, f_ext, f_des, pose_anchor=pose_anchor,
                )
                send_velocity_canfd(
                    bot.robot, v_cmd.tolist(),
                    follow=follow, trajectory_mode=traj_mode, radio=radio,
                )

                if now - last_log >= 1.0:
                    last_log = now
                    dy_tool = float(
                        end2tool_xyz(bot.robot, list(pose))[1]
                        - end2tool_xyz(bot.robot, list(pose0))[1]
                    )
                    print(
                        f"  t={t_s:.1f}s  ΔY_tool={dy_tool * 1000:+.1f}mm  "
                        f"Fz_ext={f_ext[2]:+.2f}N  vz={v_cmd[2]:+.4f} vy={v_cmd[1]:+.4f}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
        finally:
            try:
                send_velocity_canfd(
                    bot.robot, [0.0] * 6,
                    follow=follow, trajectory_mode=traj_mode, radio=radio,
                )
                bot.robot.rm_set_arm_slow_stop()
            except Exception:
                pass

    return 0
