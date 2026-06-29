"""Shared velocity-admittance control loop."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.force.compensation.collection import load_slot, move_j, wait_settle
from rm75_control.force.compensation.id_config import load_config
from rm75_control.force.compensation.paths import CONFIG_ID
from rm75_control.motion.canfd import send_velocity_canfd

from .async_state import AsyncStateObserver
from .controller import AdmittanceConfig, AdmittanceController, pose_error, wrap_pi
from .observer import CompensatedForceObserver
from .paths import CONFIG_ROBOT, PHI_JSON
from .scan_log import ScanLogRecorder, default_log_path, print_jerk_summary
from .trajectory import TrajectoryGenerator, sin_period_for_peak_vel


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def prepare_canfd_velocity_session(
    bot,
    *,
    settle_s: float = 0.5,
    clear_errors: bool = False,
) -> dict:
    """Full recover_controller before movev init — use only when stuck in force/plan mode."""
    return bot.recover_controller(
        settle_s=settle_s,
        clear_errors=clear_errors,
        probe_force_stream=False,
    )


def idle_before_movev_init(
    robot,
    *,
    mode: str = "light",
    extra_settle_s: float = 0.0,
) -> None:
    """
    Idle before rm_set_movev_canfd_init.

    Modes (see tmp/Velocity_control/run_sin_tool_y.py):
      skip     — init immediately (lowest snap if already idle after move_j)
      minimal  — delete_traj only, no slow_stop (after move_j settle)
      light    — slow_stop + delete_traj (run_sin_tool_y default)
      full     — use prepare_canfd_velocity_session instead (avoid after move_j)
    """
    m = mode.lower()
    if m in ("skip", "none", "false", "0"):
        return
    if m in ("light", "slow_stop"):
        robot.rm_set_arm_slow_stop()
        time.sleep(0.3)
    try:
        robot.rm_set_arm_delete_trajectory()
    except Exception:
        pass
    time.sleep(0.2)
    if extra_settle_s > 0.0:
        time.sleep(extra_settle_s)


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


def _pose_tracking_error_mm_deg(pose_act: np.ndarray, pose_tgt: np.ndarray) -> tuple[float, float]:
    err = pose_error(pose_tgt, pose_act)
    pos_mm = float(np.linalg.norm(err[:3]) * 1000.0)
    rot_deg = float(np.max(np.abs(np.degrees(err[3:6]))))
    return pos_mm, rot_deg


def velocity_realign_to_pose(
    robot,
    async_obs: AsyncStateObserver,
    pose_target: np.ndarray,
    *,
    dt_ms: float,
    follow: bool,
    trajectory_mode: int,
    radio: int,
    control_frame: str,
    euler_order: str,
    kp_pos: float,
    kp_rot: float,
    max_vel_m_s: float,
    max_omega_rad_s: float,
    max_accel_m_s2: float,
    max_alpha_rad_s2: float,
    pos_tol_mm: float,
    rot_tol_deg: float,
    timeout_s: float,
    settle_frames: int = 15,
) -> tuple[np.ndarray, bool]:
    """
    Post-init spatial homing while rm_movev_canfd stays active (no move_j).

    Uses base-frame pose error → velocity PBAC, output in tool or base per control_frame.
    All 6 axes are position-tracked (no force admittance) — for undoing init snap only.
    """
    dt_s = dt_ms / 1000.0
    pose_tgt = np.asarray(pose_target, dtype=float)
    last_v = np.zeros(6, dtype=float)
    next_tick = time.monotonic()
    t_start = time.monotonic()
    pose_act = pose_tgt.copy()

    while time.monotonic() - t_start < timeout_s:
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
            continue
        next_tick += dt_s

        snap = async_obs.read()
        if snap.pose is None:
            continue
        pose_act = snap.pose
        err = pose_error(pose_tgt, pose_act)
        pos_mm, rot_deg = _pose_tracking_error_mm_deg(pose_act, pose_tgt)
        if pos_mm <= pos_tol_mm and rot_deg <= rot_tol_deg:
            break

        v_base = np.zeros(6, dtype=float)
        v_base[:3] = kp_pos * err[:3]
        v_base[3:6] = kp_rot * err[3:6]
        v_base[:3] = np.clip(v_base[:3], -max_vel_m_s, max_vel_m_s)
        v_base[3:6] = np.clip(v_base[3:6], -max_omega_rad_s, max_omega_rad_s)

        if control_frame == "tool":
            r_mat = Rsc.from_euler(euler_order, pose_act[3:6], degrees=False).as_matrix()
            v_out = np.zeros(6, dtype=float)
            v_out[:3] = r_mat.T @ v_base[:3]
            v_out[3:6] = r_mat.T @ v_base[3:6]
        else:
            v_out = v_base

        dv_lin = max_accel_m_s2 * dt_s
        dv_ang = max_alpha_rad_s2 * dt_s
        for i in range(3):
            v_out[i] = float(np.clip(
                v_out[i], last_v[i] - dv_lin, last_v[i] + dv_lin,
            ))
        for i in range(3, 6):
            v_out[i] = float(np.clip(
                v_out[i], last_v[i] - dv_ang, last_v[i] + dv_ang,
            ))
        last_v = v_out.copy()
        send_velocity_canfd(
            robot, v_out.tolist(),
            follow=follow, trajectory_mode=trajectory_mode, radio=radio,
        )

    settle_movev_after_init(
        robot, dt_ms=dt_ms, follow=follow,
        trajectory_mode=trajectory_mode, radio=radio, n_frames=settle_frames,
    )
    snap = async_obs.read()
    if snap.pose is not None:
        pose_act = snap.pose
    pos_mm, rot_deg = _pose_tracking_error_mm_deg(pose_act, pose_tgt)
    ok = pos_mm <= pos_tol_mm and rot_deg <= rot_tol_deg
    return pose_act, ok


def trajectory_summary(raw: dict) -> str:
    t = raw.get("trajectory", {})
    kind = str(t.get("type", "sin_tool_y"))
    y_pp = t.get("y_peak_to_peak_cm")
    if y_pp is not None:
        pp_mm = float(y_pp) * 10.0
        amp_label = f"Y p-p={float(y_pp):.1f}cm ({pp_mm:.0f}mm)"
    else:
        amp_mm = float(t.get("amplitude_mm", 5.0))
        amp_label = f"amp=±{amp_mm:.1f}mm ({2 * amp_mm:.0f}mm p-p)"
    vmax = float(t.get("y_max_vel_cm_s", 1.0))
    ps = t.get("period_s")
    half_m = float(y_pp) * 0.01 / 2.0 if y_pp is not None else float(t.get("amplitude_mm", 5.0)) / 1000.0
    if ps is None:
        period = sin_period_for_peak_vel(half_m, vmax / 100.0)
        period_s = f"{period:.1f}s (auto)"
    else:
        period_s = f"{float(ps):.1f}s"
    soft = " soft_start" if t.get("soft_start") else ""
    rz = float(t.get("rz_amplitude_deg", 0.0))
    spin = f"  tool-Rz±{rz:.1f}°" if rz > 0 else ""
    return f"{kind}{soft}  {amp_label}{spin}  v_peak≈{vmax:.1f}cm/s  period={period_s}"


def run_velocity_admittance(
    raw: dict,
    *,
    title: str = "Velocity admittance",
    duration_s: float | None = None,
    tool_hint: bool = True,
    log_path: Path | None = None,
    log_enabled: bool | None = None,
) -> int:
    if not PHI_JSON.exists():
        raise SystemExit(f"Missing {PHI_JSON} — run force_calibrate.py first")

    timing = raw.get("timing", {})
    dt_ms = float(timing.get("dt_ms", 10.0))
    dt_s = dt_ms / 1000.0
    async_poll_ms = float(timing.get("async_poll_ms", 10.0))
    vc = raw.get("velocity_canfd", {})
    follow = bool(vc.get("follow", True))
    traj_mode = int(vc.get("trajectory_mode", 0))
    radio = int(vc.get("radio", 0))

    startup = raw.get("startup", {})
    settle_frames = int(startup.get("settle_frames", 10))
    hold_s = float(startup.get("hold_s", 0.0))
    wait_contact = bool(startup.get("wait_contact", True))
    auto_start_under_n = float(startup.get("auto_start_under_n", 0.5))
    auto_start_hold_s = float(startup.get("auto_start_hold_s", 0.5))
    auto_start_samples = max(1, int(round(auto_start_hold_s / dt_s)))
    approach_ramp_s = float(startup.get("approach_ramp_s", 1.0))
    require_observer = bool(startup.get("require_observer_ready", True))
    post_init_realign = bool(startup.get("post_init_realign", False))
    pre_movev_prep = str(startup.get("pre_movev_prep", "minimal")).lower()
    pre_movev_settle_s = float(startup.get("pre_movev_settle_s", 0.0))
    realign_min_snap_mm = float(startup.get("realign_min_snap_mm", 0.5))
    realign_pos_tol_mm = float(startup.get("realign_pos_tol_mm", 1.5))
    realign_rot_tol_deg = float(startup.get("realign_rot_tol_deg", 0.8))
    realign_timeout_s = float(startup.get("realign_timeout_s", 15.0))
    realign_kp_pos = float(startup.get("realign_kp_pos", 0.6))
    realign_kp_rot = float(startup.get("realign_kp_rot", 0.35))
    realign_max_vel_m_s = float(startup.get("realign_max_vel_m_s", 0.025))
    realign_max_omega = float(startup.get("realign_max_omega_rad_s", 0.12))
    realign_target_mode = str(startup.get("realign_target", "pre_init")).lower()
    pose_slot_raw = startup.get("pose_slot", "d")
    pose_slot = (
        None
        if pose_slot_raw in (None, "", "none", "null")
        else str(pose_slot_raw).lower()
    )
    move_speed = startup.get("move_speed")

    monitor = raw.get("monitor", {})
    if log_enabled is None:
        log_enabled = bool(monitor.get("log", False))
    log_every = max(1, int(monitor.get("log_every", 1)))
    if log_enabled and log_path is None:
        log_path = default_log_path()

    ctrl_cfg = AdmittanceConfig.from_dict(raw)
    control_frame = ctrl_cfg.control_frame
    frame_type = int(vc.get("frame_type", 0 if control_frame == "tool" else 1))
    if control_frame == "tool" and frame_type != 0:
        print("  NOTE: control_frame=tool → forcing frame_type=0 (TCP movev)", flush=True)
        frame_type = 0
    elif control_frame == "base" and frame_type != 1:
        print("  NOTE: control_frame=base → forcing frame_type=1 (world movev)", flush=True)
        frame_type = 1
    vc_run = {**vc, "frame_type": frame_type}
    traj_kind = str(raw.get("trajectory", {}).get("type", "hold"))
    observer = CompensatedForceObserver.from_yaml(raw)
    controller = AdmittanceController(dt_s, ctrl_cfg)

    f_cfg = raw.get("force", {})
    desired_z = float(f_cfg.get("desired_z_n", 3.0))
    f_des = np.zeros(6)
    f_des[2] = desired_z
    f_zero = np.zeros(6)
    auto_start_fz_n = float(startup.get("auto_start_fz_n", desired_z - auto_start_under_n))
    auto_recover = bool(startup.get("auto_recover", True))
    recover_probe_force = bool(startup.get("recover_probe_force_stream", False))

    print(
        f"{title} | rm_movev_canfd frame_type={frame_type} "
        f"follow={follow} traj={traj_mode} radio={radio}",
    )
    print(
        f"  kp_pos={ctrl_cfg.kp_pos.tolist()}  track_axes={ctrl_cfg.track_axes.tolist()}  "
        f"delay={ctrl_cfg.system_delay_s * 1000:.0f}ms  "
        f"async feedback ~{async_poll_ms:.0f}ms",
    )
    print(f"  phi: {PHI_JSON.name}  Fz_des={desired_z:.2f} N")
    print(f"  vz cap (tool TCP): ±{ctrl_cfg.max_vz_tool_m_s * 100:.1f} cm/s")
    print(f"  trajectory: {trajectory_summary(raw)}  kind={traj_kind}")
    scan_mode = "open-loop ff" if ctrl_cfg.open_loop else "closed-loop track"
    print(
        f"  hybrid: traj/Servo=6D base  fuse=tool_sleeve (Z force, XY ff) "
        f"S_f={ctrl_cfg.force_axes.tolist()}  "
        f"movev={control_frame} frame_type={frame_type}  scan={scan_mode}",
        flush=True,
    )
    if wait_contact:
        print(
            f"  auto-start: Fz≥{auto_start_fz_n:.1f}N for {auto_start_hold_s:.1f}s "
            f"→ scan; approach Fz ramp {approach_ramp_s:.1f}s (no step at hold end)",
            flush=True,
        )
    if pose_slot:
        print(f"  startup pose: move_j → slot '{pose_slot}'", flush=True)
    if post_init_realign:
        print(
            f"  post-init realign: ON → target={realign_target_mode}  "
            f"tol={realign_pos_tol_mm:.1f}mm / {realign_rot_tol_deg:.1f}°  "
            f"skip if snap<{realign_min_snap_mm:.1f}mm",
            flush=True,
        )
    print(
        f"  pre-movev prep: {pre_movev_prep} "
        f"(minimal/skip after move_j; light = run_sin_tool_y)",
        flush=True,
    )
    if tool_hint:
        print("  Ensure gripper (or desired tool) is active in RM Web UI before contact tasks.")
    if log_enabled:
        print(f"  scan log: ON → {log_path}  every {log_every} cycle(s)", flush=True)

    from rm75_control import RobotSession

    with RobotSession(config=CONFIG_ROBOT) as bot:
        if auto_recover:
            rec = bot.recover_controller(
                settle_s=1.0,
                clear_errors=True,
                probe_force_stream=recover_probe_force,
            )
            err = rec.get("system_err") or []
            print(
                f"  auto-recover: idle={rec.get('planning_idle')}  "
                f"traj={rec.get('trajectory_type_final')}  "
                f"sys_err={err or 'none'}",
                flush=True,
            )
            if err:
                print("  (cleared latched controller errors on connect)", flush=True)

        pose_slot_cartesian: np.ndarray | None = None
        if pose_slot:
            fid = load_config(CONFIG_ID)
            spd = int(move_speed) if move_speed is not None else fid.collect.move_speed
            q_tgt, pose_slot_cartesian, rec = load_slot(fid, pose_slot)
            pose_slot_cartesian = np.asarray(pose_slot_cartesian, dtype=float)
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
        traj_origin = pose0.copy()
        print(
            f"  start TCP pose (base): "
            f"xyz=[{pose0[0]:.3f},{pose0[1]:.3f},{pose0[2]:.3f}] m",
            flush=True,
        )
        traj = TrajectoryGenerator.from_dict(raw, pose0, bot.robot)
        pose_before_prep = pose0.copy()

        if pre_movev_prep == "full":
            prep = prepare_canfd_velocity_session(bot, settle_s=0.5)
            print(
                f"  CANFD prep (full recover): idle={prep.get('planning_idle')}  "
                f"traj={prep.get('trajectory_type_final')}  "
                f"euler_deg={prep.get('pose_euler_deg', [])}",
                flush=True,
            )
            if not prep.get("planning_idle", False):
                print("  WARN: planner not idle before movev init — snap more likely", flush=True)
        else:
            idle_before_movev_init(
                bot.robot, mode=pre_movev_prep, extra_settle_s=pre_movev_settle_s,
            )
            print(f"  CANFD prep: {pre_movev_prep}", flush=True)

        ret_pre, st_pre = bot.robot.rm_get_current_arm_state()
        pose_pre = pose_before_prep.copy()
        if ret_pre == 0:
            pose_pre = np.asarray(st_pre["pose"][:6], dtype=float)
        prep_dpos_mm = (pose_pre[:3] - pose_before_prep[:3]) * 1000.0
        prep_mm = float(np.linalg.norm(prep_dpos_mm))
        if prep_mm > 0.5:
            print(
                f"  prep drift Δpos_mm="
                f"[{prep_dpos_mm[0]:+.2f},{prep_dpos_mm[1]:+.2f},{prep_dpos_mm[2]:+.2f}]  "
                f"|Δ|={prep_mm:.1f}mm",
                flush=True,
            )

        init_velocity_canfd(bot.robot, vc_run, dt_ms)
        init_settle_frames = max(1, settle_frames)
        settle_movev_after_init(
            bot.robot, dt_ms=dt_ms, follow=follow,
            trajectory_mode=traj_mode, radio=radio, n_frames=init_settle_frames,
        )
        ret_post, st_post = bot.robot.rm_get_current_arm_state()
        pose_post = pose0.copy()
        if ret_post == 0:
            pose_post = np.asarray(st_post["pose"][:6], dtype=float)
            deuler = np.degrees(pose_post[3:6] - pose_pre[3:6])
            deuler = (deuler + 180.0) % 360.0 - 180.0
            dpos_mm = (pose_post[:3] - pose_pre[:3]) * 1000.0
            print(
                f"  post-init settle Δpos_mm="
                f"[{dpos_mm[0]:+.2f},{dpos_mm[1]:+.2f},{dpos_mm[2]:+.2f}]  "
                f"Δeuler_deg=[{deuler[0]:+.2f},{deuler[1]:+.2f},{deuler[2]:+.2f}]",
                flush=True,
            )
            snap_mm = float(np.linalg.norm(dpos_mm))
            if snap_mm > 3.0:
                print(
                    f"  init snap |Δ|={snap_mm:.1f}mm — "
                    f"try pre_movev_prep: skip or minimal after move_j",
                    flush=True,
                )
            pose0 = pose_post.copy()
            traj_origin = pose0.copy()
            traj.set_origin(pose0)
            print(
                f"  anchored pose0 (post-init): "
                f"xyz=[{pose0[0]:.3f},{pose0[1]:.3f},{pose0[2]:.3f}] m",
                flush=True,
            )

        async_obs = AsyncStateObserver(bot.robot, poll_s=async_poll_ms / 1000.0)
        async_obs.start()
        try:
            pose_fb = async_obs.wait_first_pose(timeout_s=5.0)
        except TimeoutError:
            async_obs.stop()
            raise SystemExit("AsyncStateObserver: no pose after CANFD init")
        q_fb = np.zeros(7, dtype=float)

        controller.reset(clear_velocity=True)

        if post_init_realign:
            if realign_target_mode == "pose_slot" and pose_slot_cartesian is not None:
                realign_target = pose_slot_cartesian.copy()
            else:
                realign_target = pose_pre.copy()
            snap_to_target_mm, snap_rot_deg = _pose_tracking_error_mm_deg(pose0, realign_target)
            if snap_to_target_mm >= realign_min_snap_mm or snap_rot_deg >= realign_rot_tol_deg:
                print(
                    f"  realign start: offset vs target "
                    f"{snap_to_target_mm:.1f}mm / {snap_rot_deg:.2f}°",
                    flush=True,
                )
                pose_realigned, realign_ok = velocity_realign_to_pose(
                    bot.robot,
                    async_obs,
                    realign_target,
                    dt_ms=dt_ms,
                    follow=follow,
                    trajectory_mode=traj_mode,
                    radio=radio,
                    control_frame=control_frame,
                    euler_order=ctrl_cfg.euler_order,
                    kp_pos=realign_kp_pos,
                    kp_rot=realign_kp_rot,
                    max_vel_m_s=realign_max_vel_m_s,
                    max_omega_rad_s=realign_max_omega,
                    max_accel_m_s2=0.4,
                    max_alpha_rad_s2=0.8,
                    pos_tol_mm=realign_pos_tol_mm,
                    rot_tol_deg=realign_rot_tol_deg,
                    timeout_s=realign_timeout_s,
                    settle_frames=max(10, settle_frames // 2),
                )
                pose0 = pose_realigned.copy()
                traj_origin = pose0.copy()
                traj.set_origin(pose0)
                err_mm, err_deg = _pose_tracking_error_mm_deg(pose0, realign_target)
                status = "OK" if realign_ok else "TIMEOUT"
                print(
                    f"  realign {status}: residual {err_mm:.1f}mm / {err_deg:.2f}°  "
                    f"xyz=[{pose0[0]:.3f},{pose0[1]:.3f},{pose0[2]:.3f}]",
                    flush=True,
                )
                controller.reset(clear_velocity=True)
            else:
                print(
                    f"  realign skip: snap {snap_to_target_mm:.1f}mm < "
                    f"{realign_min_snap_mm:.1f}mm",
                    flush=True,
                )

        print("Velocity CANFD initialized. Ctrl+C to stop.", flush=True)

        scan_log = ScanLogRecorder() if log_enabled else None
        log_tick = 0

        t0 = time.monotonic()
        next_tick = t0
        last_log = t0
        scan_started = not wait_contact
        pending_scan = False
        start_streak = 0
        t_scan0: float | None = None if wait_contact else t0
        f_ext = f_zero.copy()
        last_wait_msg = 0.0
        fz_buf: list[float] = []

        def _fz_smooth() -> float:
            if not fz_buf:
                return float(f_ext[2])
            return float(np.median(fz_buf))

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

                snap = async_obs.read()
                if snap.pose is not None:
                    pose_fb = snap.pose
                    if snap.q_deg is not None:
                        q_fb = snap.q_deg
                    observer.append(t_s, pose_fb, snap.force_raw)
                    wrench = observer.latest_wrench()
                    if wrench is not None:
                        f_ext = wrench[1]
                        fz_buf.append(float(f_ext[2]))
                        if len(fz_buf) > 7:
                            fz_buf.pop(0)
                pose = pose_fb

                if not scan_started and wait_contact and t_s >= hold_s:
                    if require_observer and not observer.ready():
                        if t_s - last_wait_msg >= 2.0:
                            print(
                                f"  waiting phi observer ({len(observer.buf)}/"
                                f"{observer.cfg.min_samples})…",
                                flush=True,
                            )
                            last_wait_msg = t_s
                        start_streak = 0
                    else:
                        fz_s = _fz_smooth()
                        if fz_s >= auto_start_fz_n:
                            start_streak += 1
                        else:
                            start_streak = 0
                        if start_streak >= auto_start_samples:
                            pending_scan = True
                            start_streak = 0

                if pending_scan and not scan_started:
                    pending_scan = False
                    scan_started = True
                    t_scan0 = now
                    traj_origin = pose.copy()
                    traj.set_origin(traj_origin)
                    controller.reset(clear_velocity=ctrl_cfg.open_loop)
                    print(
                        f"  scan ON @ t={t_s:.1f}s  Fz={f_ext[2]:+.2f}N  traj={traj_kind}",
                        flush=True,
                    )

                t_scan = (now - t_scan0) if (scan_started and t_scan0 is not None) else float("nan")
                phase = 2 if scan_started else (1 if t_s >= hold_s else 0)
                pose_d_log = np.zeros(6, dtype=float)
                vel_ff_log = np.zeros(6, dtype=float)
                f_des_z = float(desired_z)

                if not scan_started:
                    if t_s < hold_s or (require_observer and not observer.ready()):
                        v_cmd = np.zeros(6, dtype=float)
                        phase = 0
                        pose_d_log = pose.copy()
                    else:
                        since_approach = max(0.0, t_s - hold_s)
                        f_scale = (
                            min(1.0, since_approach / approach_ramp_s)
                            if approach_ramp_s > 0
                            else 1.0
                        )
                        f_des_z = float(desired_z * f_scale)
                        pose_d_log = pose0.copy()
                        v_cmd = controller.compute_velocity_command(
                            pose, pose0, np.zeros(6), f_ext, f_des * f_scale,
                            in_contact=False,
                            enable_pbac=False,
                        )
                        phase = 1
                    send_velocity_canfd(
                        bot.robot, v_cmd.tolist(),
                        follow=follow, trajectory_mode=traj_mode, radio=radio,
                    )
                    if scan_log is not None:
                        log_tick += 1
                        if log_tick >= log_every:
                            log_tick = 0
                            scan_log.append_row(
                                t_s=t_s, t_scan=t_scan, phase=phase,
                                pose_act=pose, q_deg=q_fb, pose_d=pose_d_log,
                                vel_ff=vel_ff_log, v_cmd=v_cmd, f_ext=f_ext,
                                f_des_z=f_des_z,
                            )
                    continue

                sample = traj.sample(t_scan)
                pose_d_log = sample.pose_d.copy()
                vel_ff_log = sample.vel_ff.copy()
                v_cmd = controller.compute_velocity_command(
                    pose, sample.pose_d, sample.vel_ff, f_ext, f_des,
                    in_contact=True,
                    enable_pbac=not ctrl_cfg.open_loop,
                )
                phase = 2

                v_cmd = np.asarray(v_cmd, dtype=float)
                if not np.all(np.isfinite(v_cmd)):
                    print(
                        f"  WARN: non-finite v_cmd {v_cmd} — sending zero (phase={phase})",
                        flush=True,
                    )
                    v_cmd = np.zeros(6, dtype=float)

                if scan_log is not None:
                    log_tick += 1
                    if log_tick >= log_every:
                        log_tick = 0
                        scan_log.append_row(
                            t_s=t_s,
                            t_scan=t_scan,
                            phase=phase,
                            pose_act=pose,
                            q_deg=q_fb,
                            pose_d=pose_d_log,
                            vel_ff=vel_ff_log,
                            v_cmd=v_cmd,
                            f_ext=f_ext,
                            f_des_z=f_des_z,
                        )

                send_velocity_canfd(
                    bot.robot, v_cmd.tolist(),
                    follow=follow, trajectory_mode=traj_mode, radio=radio,
                )

                if scan_started and now - last_log >= 1.0:
                    last_log = now

                    dy_mm = float(pose[1] - traj_origin[1]) * 1000.0
                    deuler = np.degrees([
                        wrap_pi(float(pose[i] - traj_origin[i])) for i in range(3, 6)
                    ])
                    print(
                        f"  t={t_s:.1f}s  ΔY_world={dy_mm:+.1f}mm  "
                        f"Δeuler_deg=[{deuler[0]:+.2f},{deuler[1]:+.2f},{deuler[2]:+.2f}]  "
                        f"Fz_ext={f_ext[2]:+.2f}N  "
                        f"vy={v_cmd[1]:+.4f} ({control_frame} movev)",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
        finally:
            async_obs.stop()
            if scan_log is not None and len(scan_log) > 0 and log_path is not None:
                try:
                    saved = scan_log.save(
                        log_path,
                        meta={
                            "traj_kind": traj_kind,
                            "dt_ms": dt_ms,
                            "async_poll_ms": async_poll_ms,
                            "control_frame": control_frame,
                            "frame_type": frame_type,
                        },
                    )
                    print(f"  scan log saved → {saved} ({len(scan_log)} samples)", flush=True)
                    print_jerk_summary(saved, dt_s=dt_s)
                except Exception as exc:
                    print(f"  scan log save failed: {exc}", flush=True)
            try:
                settle_movev_after_init(
                    bot.robot, dt_ms=dt_ms, follow=follow,
                    trajectory_mode=traj_mode, radio=radio, n_frames=15,
                )
                bot.robot.rm_set_arm_slow_stop()
            except Exception:
                pass

    return 0
