"""
Multi-pose force-ID data collection: A → B → C → D → return A.

  source env.sh
  python -m rm75_control.force.compensation.collection
  python tmp/force_compensation/force_calibrate.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import excitation as ex
from .id_config import ForceIdConfig, load_config
from .paths import CONFIG_ID, CONFIG_ROBOT, npz_for_slot
from .progress import stage_progress


from .tool_pose import get_active_tool_name, poses_calib_tool_frame, slot_tcp_pose


def load_slot(
    cfg: ForceIdConfig,
    slot: str,
    robot=None,
    *,
    calib_tool: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    data = ex.load_poses_yaml(cfg.poses_yaml)
    rec = ex.get_slot_record(data, slot)
    if rec is None:
        raise SystemExit(f"Pose slot '{slot}' missing in {cfg.poses_yaml}")
    q = np.asarray(rec["q_deg"], dtype=float)
    pose = np.asarray(rec["pose_base"], dtype=float)
    tool = calib_tool if calib_tool is not None else poses_calib_tool_frame(data)
    if robot is not None:
        pose = slot_tcp_pose(robot, q, pose, calib_tool=tool)
    return q, pose, rec


def move_j(robot, q_deg: np.ndarray, *, speed: int) -> None:
    ret = robot.rm_movej(q_deg.tolist(), speed, 0, 0, 1)
    if ret != 0:
        raise RuntimeError(f"rm_movej failed: {ret}")


def move_j_p(robot, pose: np.ndarray, *, speed: int) -> None:
    ret = robot.rm_movej_p(pose.tolist(), speed, 0, 0, 1)
    if ret != 0:
        raise RuntimeError(f"rm_movej_p failed: {ret}")


def require_tool_frame(robot, *, required: str) -> None:
    """Abort calibration unless the active TCP tool frame matches (default Arm_Tip)."""
    ret, cur = robot.rm_get_current_tool_frame()
    if ret != 0:
        raise SystemExit(
            f"ERROR: rm_get_current_tool_frame failed (code {ret}). "
            "Cannot verify tool frame before calibration."
        )
    active = str(cur.get("name", ""))
    if active != required:
        raise SystemExit(
            f"\nERROR: Active tool frame is {active!r}, but force calibration requires {required!r}.\n"
            "Switch the tool coordinate to Arm_Tip in the RealMan Web UI / teach pendant, then re-run.\n"
        )
    print(f"  Tool frame OK: {active!r}", flush=True)


def wait_settle(robot, target_q: np.ndarray, *, timeout_s: float) -> tuple[np.ndarray, np.ndarray]:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        ret, st = robot.rm_get_current_arm_state()
        if ret == 0:
            q = np.asarray(st["joint"][:7], dtype=float)
            if float(np.max(np.abs(q - target_q))) < 0.5:
                time.sleep(0.5)
                return np.asarray(st["pose"][:6], dtype=float), q
        time.sleep(0.1)
    ret, st = robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError("get state failed after movej")
    return np.asarray(st["pose"][:6], dtype=float), np.asarray(st["joint"][:7], dtype=float)


def wait_settle_pose(
    robot,
    target_pose: np.ndarray,
    *,
    timeout_s: float,
    tol_mm: float = 1.0,
    tol_deg: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        ret, st = robot.rm_get_current_arm_state()
        if ret == 0:
            pose = np.asarray(st["pose"][:6], dtype=float)
            d_mm, d_deg = ex.pose_drift_mm_deg(pose, target_pose)
            if d_mm < tol_mm and d_deg < tol_deg:
                time.sleep(0.5)
                return pose, np.asarray(st["joint"][:7], dtype=float)
        time.sleep(0.1)
    ret, st = robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError("get state failed after movej_p")
    return np.asarray(st["pose"][:6], dtype=float), np.asarray(st["joint"][:7], dtype=float)


def run_cartesian(bot, cfg: ForceIdConfig, slot: str) -> Path:
    from rm75_control.motion.canfd import exit_canfd_session, send_pose_canfd

    c = cfg.collect
    cart = c.cartesian
    max_deg = cart.max_deg_for_slot(slot)
    dt_s = c.dt_ms / 1000.0
    duration = cart.duration_s
    ramp_down_s = c.cartesian_ramp_down_s
    out = npz_for_slot(slot)
    exc = ex.CartesianExcitation.from_config(cart, c.scale, slot)

    ret, state = bot.robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"get state failed: {ret}")
    pose0 = np.asarray(state["pose"][:6], dtype=float)
    q0 = np.asarray(state["joint"][:7], dtype=float)

    n_cmd = int(duration / dt_s) + 1
    n_log = (n_cmd + c.log_every - 1) // c.log_every
    t_log = np.zeros(n_log)
    pose_log = np.zeros((n_log, 6))
    q_log = np.zeros((n_log, 7))
    f_log = np.zeros((n_log, 6))
    delta_log = np.zeros((n_log, 6))

    next_tick = time.monotonic()
    log_i = 0
    t_end = (n_cmd - 1) * dt_s
    try:
        for i in range(n_cmd):
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += dt_s
            t_cmd = i * dt_s
            stage_progress(slot, i + 1, n_cmd)
            ramp = min(1.0, t_cmd / c.warmup_s) if c.warmup_s > 0 else 1.0
            delta = ex.clamp_delta(
                exc.delta_pose(t_cmd) * ramp,
                max_mm=cart.max_delta_mm,
                max_rot_deg=max_deg,
            )
            send_pose_canfd(
                bot.robot, (pose0 + delta).tolist(),
                follow=c.follow,
            )
            if i % c.log_every == 0:
                ret_s, st = bot.robot.rm_get_current_arm_state()
                ret_f, fd = bot.robot.rm_get_force_data()
                if ret_s == 0:
                    pose_log[log_i] = st["pose"][:6]
                    q_log[log_i] = st["joint"][:7]
                if ret_f == 0:
                    f_log[log_i] = np.asarray(fd["force_data"][:6], dtype=float)
                delta_log[log_i] = delta
                t_log[log_i] = t_cmd
                log_i += 1
        ramp_end = min(1.0, t_end / c.warmup_s) if c.warmup_s > 0 else 1.0
        delta_end = ex.clamp_delta(
            exc.delta_pose(t_end) * ramp_end,
            max_mm=cart.max_delta_mm,
            max_rot_deg=max_deg,
        )
        next_tick = ex.ramp_down_cartesian(
            bot.robot, pose0, delta_end,
            follow=c.follow, dt_ms=c.dt_ms, ramp_s=ramp_down_s,
            next_tick=next_tick,
        )
    finally:
        exit_canfd_session(
            bot.robot,
            q_resync=q0,
            move_speed=c.move_speed,
            settle_timeout_s=c.settle_timeout_s,
            print_diag=True,
        )

    if out.exists():
        out.unlink()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        t=t_log[:log_i], pose=pose_log[:log_i], q_deg=q_log[:log_i],
        force_raw=f_log[:log_i], delta_pose=delta_log[:log_i],
        pose0=pose0, q0_deg=q0, pose_slot=slot, preset="cartesian",
        scale=c.scale, max_delta_mm=cart.max_delta_mm, max_delta_deg=max_deg,
        dt_ms=c.dt_ms, log_every=c.log_every, method="cartesian",
    )
    return out


def run_pose_d(bot, cfg: ForceIdConfig) -> Path:
    from rm75_control.motion.canfd import exit_canfd_session, send_velocity_canfd

    c = cfg.collect
    pd = c.pose_d
    vb = pd.velocity_burst
    dt_s = c.dt_ms / 1000.0
    total_s = pd.joint_duration_s + pd.burst_duration_s
    out = npz_for_slot("d")

    ret, state = bot.robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"get state failed: {ret}")
    pose0 = np.asarray(state["pose"][:6], dtype=float)
    q0 = np.asarray(state["joint"][:7], dtype=float)

    n_total = int(total_s / dt_s) + 1
    n_log = (n_total + c.log_every - 1) // c.log_every
    t_log = np.zeros(n_log)
    pose_log = np.zeros((n_log, 6))
    q_log = np.zeros((n_log, 7))
    f_log = np.zeros((n_log, 6))
    phase_log = np.zeros(n_log, dtype=np.int8)

    print("\n  d (joint + pose_d_vel_burst)", flush=True)
    next_tick = time.monotonic()
    log_i = 0
    movev_ready = False
    ramped_down = False
    last_vel = np.zeros(6, dtype=float)
    burst_pose0 = pose0.copy()

    try:
        for i in range(n_total):
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += dt_s
            t_cmd = i * dt_s
            stage_progress("d", i + 1, n_total)
            ramp = min(1.0, t_cmd / c.warmup_s) if c.warmup_s > 0 else 1.0

            if pd.joint_duration_s > 0 and t_cmd < pd.joint_duration_s:
                q_cmd = ex.joint_cmd(t_cmd, q0, pd, c.scale * ramp)
                ret = bot.robot.rm_movej_canfd(q_cmd.tolist(), False, 0, 0, 0)
                phase = 0
            else:
                if not movev_ready:
                    print("  joint→movev handoff…", flush=True)
                    print("  joint hold + settle…", flush=True)
                    _, pose_handoff = ex.stabilize_joint_canfd_hold(
                        bot.robot,
                        dt_ms=c.dt_ms,
                        n_frames=80,
                        dwell_s=c.pre_movev_settle_s,
                    )
                    burst_pose0 = pose_handoff
                    next_tick, handoff_diag = ex.begin_pose_d_vel_burst(
                        bot, vb=vb, dt_ms=c.dt_ms,
                        skip_resync=True,
                        settle_frames=c.movev_settle_frames,
                        quiescent_mm=c.movev_quiescent_mm,
                        quiescent_consecutive=c.movev_quiescent_consecutive,
                        quiescent_warmup_frames=c.quiescent_warmup_frames,
                        quiescent_reject_step_mm=c.quiescent_reject_step_mm,
                        pre_movev_settle_s=0.5,
                        post_handoff_zero_s=c.post_handoff_zero_s,
                        move_speed=c.move_speed,
                        settle_timeout_s=c.settle_timeout_s,
                        next_tick=next_tick,
                    )
                    pose_quiet = handoff_diag.get("quiescent_pose")
                    if pose_quiet is not None:
                        burst_pose0 = np.asarray(pose_quiet, dtype=float)
                    movev_ready = True
                t_burst = t_cmd - pd.joint_duration_s
                vel_cmd, _ = ex.vel_burst_cmd(t_burst, vb, scale=c.scale)
                if vb.ramp_s > 0.0 and t_burst < vb.ramp_s:
                    vel_cmd *= t_burst / vb.ramp_s
                last_vel = vel_cmd
                send_velocity_canfd(
                    bot.robot, vel_cmd.tolist(),
                    follow=vb.follow,
                )
                phase = 1
                ret = 0
            if ret != 0:
                raise RuntimeError(f"pose d command failed: {ret}")

            if i % c.log_every == 0:
                ret_s, st = bot.robot.rm_get_current_arm_state()
                ret_f, fd = bot.robot.rm_get_force_data()
                if ret_s == 0:
                    pose_log[log_i] = st["pose"][:6]
                    q_log[log_i] = st["joint"][:7]
                if ret_f == 0:
                    f_log[log_i] = np.asarray(fd["force_data"][:6], dtype=float)
                t_log[log_i] = t_cmd
                phase_log[log_i] = phase
                log_i += 1

        if movev_ready:
            ex.ramp_down_velocity(
                bot.robot, last_vel, vb=vb, dt_ms=c.dt_ms, next_tick=next_tick,
            )
            ramped_down = True

    finally:
        if movev_ready and not ramped_down:
            try:
                ex.ramp_down_velocity(bot.robot, last_vel, vb=vb, dt_ms=c.dt_ms)
            except Exception:
                pass
        exit_canfd_session(
            bot.robot,
            q_resync=None,
            move_speed=c.move_speed,
            settle_timeout_s=c.settle_timeout_s,
            print_diag=True,
        )

    burst_pose0_save = burst_pose0.copy()
    if log_i > 0 and np.any(phase_log[:log_i] == 1) and pd.joint_duration_s <= 0:
        burst_pose0_save = pose_log[:log_i][phase_log[:log_i] == 1][0].copy()

    if out.exists():
        out.unlink()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        t=t_log[:log_i], pose=pose_log[:log_i], q_deg=q_log[:log_i],
        force_raw=f_log[:log_i], phase=phase_log[:log_i],
        pose0=pose0, pose_burst0=burst_pose0_save, q0_deg=q0, pose_slot="d",
        preset="pose_d_vel_burst", scale=c.scale,
        joint_s=pd.joint_duration_s, burst_s=pd.burst_duration_s,
        dt_ms=c.dt_ms, log_every=c.log_every, method="pose_d_vel_burst",
        velocity_burst_profile=vb.profile,
    )
    return out


def slot_kind(slot: str) -> str:
    return "pose_d_vel_burst" if slot == "d" else "cartesian"


def dry_run(cfg: ForceIdConfig) -> None:
    seq = cfg.collect.sequence
    print(f"Collect {' → '.join(seq)} → {cfg.collect.return_home}")
    for slot in seq:
        _, _, rec = load_slot(cfg, slot)
        line = f"  {slot} [{slot_kind(slot)}]: {rec.get('label', f'pose_{slot}')}"
        if slot == "d":
            vb = cfg.collect.pose_d.velocity_burst
            line += (
                f" | burst={vb.profile} {vb.amp_deg_s}°/s frame={vb.frame_type} "
                f"order={list(vb.axis_order)} ramp_down={vb.ramp_down_s}s"
            )
        print(line)


def save_current_pose(cfg: ForceIdConfig, slot: str, label: str | None) -> None:
    from rm75_control import RobotSession

    with RobotSession(config=CONFIG_ROBOT) as bot:
        require_tool_frame(bot.robot, required=cfg.required_tool_frame)
        ret, st = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            raise SystemExit(f"get state failed: {ret}")
        pose = np.asarray(st["pose"][:6], dtype=float)
        q = np.asarray(st["joint"][:7], dtype=float)
    ex.save_pose_slot(cfg.poses_yaml, slot, pose, q, label)
    print(f"Saved pose '{slot}' → {cfg.poses_yaml}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A→B→C→D→A force-ID collection")
    parser.add_argument("--config", type=Path, default=CONFIG_ID)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-pose", type=str, default=None, metavar="SLOT")
    parser.add_argument("--pose-label", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.save_pose:
        save_current_pose(cfg, args.save_pose, args.pose_label)
        return 0
    if args.dry_run:
        dry_run(cfg)
        return 0

    c = cfg.collect
    seq = c.sequence
    slots = {s: load_slot(cfg, s) for s in set(seq) | {c.return_home}}
    print(f"Collect {' → '.join(seq)} → {c.return_home}")
    for s in seq:
        line = f"  {s} [{slot_kind(s)}]: {slots[s][2].get('label', f'pose_{s}')}"
        if s == "d":
            vb = c.pose_d.velocity_burst
            line += (
                f" | burst={vb.profile} {vb.amp_deg_s}°/s frame={vb.frame_type} "
                f"order={list(vb.axis_order)}"
            )
        print(line)

    from rm75_control import RobotSession

    with RobotSession(config=CONFIG_ROBOT) as bot:
        require_tool_frame(bot.robot, required=cfg.required_tool_frame)
        saved = []
        for slot in seq:
            q_tgt, pose_tgt, rec = slots[slot]
            print(f"\nMove {slot}")
            from rm75_control.motion.canfd import exit_canfd_session

            exit_canfd_session(
                bot.robot,
                q_resync=q_tgt if slot == "d" else None,
                move_speed=c.move_speed,
                settle_timeout_s=c.settle_timeout_s,
                print_diag=True,
            )
            if slot == "d":
                move_j(bot.robot, q_tgt, speed=c.move_speed)
                wait_settle(bot.robot, q_tgt, timeout_s=c.settle_timeout_s)
            else:
                move_j_p(bot.robot, pose_tgt, speed=c.move_speed)
                wait_settle_pose(bot.robot, pose_tgt, timeout_s=c.settle_timeout_s)
            if slot == "d":
                saved.append(run_pose_d(bot, cfg))
            else:
                saved.append(run_cartesian(bot, cfg, slot))

        home = c.return_home
        q_h, pose_h, _ = slots[home]
        print(f"\nReturn {home}")
        from rm75_control.motion.canfd import exit_canfd_session

        exit_canfd_session(
            bot.robot,
            q_resync=q_h if home == "d" else None,
            move_speed=c.move_speed,
            settle_timeout_s=c.settle_timeout_s,
            print_diag=True,
        )
        if home == "d":
            move_j(bot.robot, q_h, speed=c.move_speed)
        else:
            move_j_p(bot.robot, pose_h, speed=c.move_speed)
        if home == "d":
            wait_settle(bot.robot, q_h, timeout_s=c.settle_timeout_s)
        else:
            wait_settle_pose(bot.robot, pose_h, timeout_s=c.settle_timeout_s)
        print("\nCollection done:")
        for p in saved:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
