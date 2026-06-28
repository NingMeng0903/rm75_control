"""Shared velocity-admittance control loop."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from rm75_control.force.compensation.collection import load_slot, move_j, wait_settle
from rm75_control.force.compensation.id_config import load_config
from rm75_control.force.compensation.paths import CONFIG_ID
from rm75_control.motion.canfd import send_velocity_canfd

from .controller import AdmittanceConfig, AdmittanceController
from .observer import CompensatedForceObserver
from .paths import CONFIG_ROBOT, PHI_JSON
from .trajectory import TrajectoryGenerator, sin_period_for_peak_vel


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def init_velocity_canfd(robot, vc: dict, dt_ms: float) -> None:
    ret = robot.rm_set_movev_canfd_init(
        int(vc.get("avoid_singularity", 0)),
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
    kp[0:2] = np.minimum(kp[0:2], 2.5)
    kp[2] = max(float(kp[2]), 2.0)
    kp[3:6] = 0.0
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
    feedback_every = max(1, int(timing.get("feedback_every", 3)))
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
    pose_slot_raw = startup.get("pose_slot", "d")
    pose_slot = (
        None
        if pose_slot_raw in (None, "", "none", "null")
        else str(pose_slot_raw).lower()
    )
    move_speed = startup.get("move_speed")

    ctrl_cfg = AdmittanceConfig.from_dict(raw)
    hold_cfg = _hold_controller_config(ctrl_cfg)
    control_frame = ctrl_cfg.control_frame
    frame_type = int(vc.get("frame_type", 0 if control_frame == "tool" else 1))
    if control_frame == "tool" and frame_type != 0:
        print("  NOTE: control_frame=tool → forcing frame_type=0 (TCP movev)", flush=True)
        frame_type = 0
    elif control_frame == "base" and frame_type != 1:
        print("  NOTE: control_frame=base → forcing frame_type=1 (world movev)", flush=True)
        frame_type = 1
    vc_run = {**vc, "frame_type": frame_type}
    traj_kind = str(raw.get("trajectory", {}).get("type", "sin_tool_y"))
    base_world_scan = ctrl_cfg.motion_frame == "base" or traj_kind == "sin_base_y"
    observer = CompensatedForceObserver.from_yaml(raw)
    controller = AdmittanceController(dt_s, ctrl_cfg)
    hold_ctrl = AdmittanceController(dt_s, hold_cfg)

    f_cfg = raw.get("force", {})
    desired_z = float(f_cfg.get("desired_z_n", 3.0))
    f_des = np.zeros(6)
    f_des[2] = desired_z
    f_zero = np.zeros(6)

    print(
        f"{title} | rm_movev_canfd frame_type={frame_type} "
        f"follow={follow} traj={traj_mode} radio={radio}",
    )
    print(
        f"  kp_pos={ctrl_cfg.kp_pos.tolist()}  delay={ctrl_cfg.system_delay_s * 1000:.0f}ms  "
        f"feedback every {feedback_every} cycles (~{feedback_every * dt_ms:.0f}ms)",
    )
    print(f"  phi: {PHI_JSON.name}  Fz_des={desired_z:.2f} N")
    print(f"  vz cap (tool TCP): ±{ctrl_cfg.max_vz_tool_m_s * 100:.1f} cm/s")
    print(f"  trajectory: {trajectory_summary(raw)}")
    print(
        f"  plan=world ({traj_kind})  control/output={control_frame}  "
        f"force=tool-Z  orient=locked",
        flush=True,
    )
    if wait_contact:
        print(
            f"  startup: settle={settle_frames}f hold={hold_s:.1f}s  "
            f"position hold until Fz_ext≥{contact_fz_n:.1f}N (no auto descent)",
            flush=True,
        )
    if pose_slot:
        print(f"  startup pose: move_j → slot '{pose_slot}'", flush=True)
    if tool_hint:
        print("  Ensure gripper (or desired tool) is active in RM Web UI before contact tasks.")

    from rm75_control import RobotSession

    with RobotSession(config=CONFIG_ROBOT) as bot:
        if pose_slot:
            fid = load_config(CONFIG_ID)
            spd = int(move_speed) if move_speed is not None else fid.collect.move_speed
            q_tgt, _, rec = load_slot(fid, pose_slot)
            print(
                f"  move_j → {pose_slot} ({rec.get('label', '')}) speed={spd}",
                flush=True,
            )
            move_j(bot.robot, q_tgt, speed=spd)
            pose_act, q_act = wait_settle(
                bot.robot, q_tgt, timeout_s=fid.collect.settle_timeout_s,
            )
            print(
                f"  settled q_max_err={float(np.max(np.abs(q_act - q_tgt))):.3f}°",
                flush=True,
            )

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

        bot.robot.rm_set_arm_slow_stop()
        time.sleep(0.3)
        try:
            bot.robot.rm_set_arm_delete_trajectory()
        except Exception:
            pass
        time.sleep(0.2)

        init_velocity_canfd(bot.robot, vc_run, dt_ms)
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
        cycle = 0
        pose_fb = pose0.copy()
        f_ext = f_zero.copy()

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

                if cycle % feedback_every == 0:
                    ret_s, st = bot.robot.rm_get_current_arm_state()
                    ret_f, fd = bot.robot.rm_get_force_data()
                    if ret_s == 0 and ret_f == 0:
                        pose_fb = np.asarray(st["pose"][:6], dtype=float)
                        force_raw = np.asarray(fd["force_data"][:6], dtype=float)
                        observer.append(t_s, pose_fb, force_raw)
                        wrench = observer.latest_wrench()
                        if wrench is not None:
                            f_ext = wrench[1]
                cycle += 1
                pose = pose_fb

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
                        traj.set_origin(pose_anchor)
                        controller.force_error_integral.fill(0.0)
                        controller.last_v_cmd = hold_ctrl.last_v_cmd.copy()
                        print(
                            f"  contact latched @ t={t_s:.1f}s  Fz_ext={f_ext[2]:+.2f}N  "
                            "— world-Y sin + tool-Z force (pose D locked)",
                            flush=True,
                        )
                    else:
                        continue

                t_scan = (now - t_scan0) if t_scan0 is not None else t_s
                pose_d, vel_ff = traj.sample(t_scan)
                if base_world_scan:
                    pose_d, vel_ff = TrajectoryGenerator.world_scan_reference(
                        pose_d, vel_ff, pose_anchor, motion_axes,
                    )
                else:
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
                    dy_mm = float(pose[1] - pose_anchor[1]) * 1000.0
                    deuler = np.degrees(pose[3:6] - pose_anchor[3:6])
                    print(
                        f"  t={t_s:.1f}s  ΔY_world={dy_mm:+.1f}mm  "
                        f"Δeuler_deg=[{deuler[0]:+.2f},{deuler[1]:+.2f},{deuler[2]:+.2f}]  "
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
