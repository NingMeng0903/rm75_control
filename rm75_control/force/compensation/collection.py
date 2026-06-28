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


def load_slot(cfg: ForceIdConfig, slot: str) -> tuple[np.ndarray, np.ndarray, dict]:
    data = ex.load_poses_yaml(cfg.poses_yaml)
    rec = ex.get_slot_record(data, slot)
    if rec is None:
        raise SystemExit(f"Pose slot '{slot}' missing in {cfg.poses_yaml}")
    return (
        np.asarray(rec["q_deg"], dtype=float),
        np.asarray(rec["pose_base"], dtype=float),
        rec,
    )


def move_j(robot, q_deg: np.ndarray, *, speed: int) -> None:
    ret = robot.rm_movej(q_deg.tolist(), speed, 0, 0, 1)
    if ret != 0:
        raise RuntimeError(f"rm_movej failed: {ret}")


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


def run_cartesian(bot, cfg: ForceIdConfig, slot: str) -> Path:
    from rm75_control.motion.canfd import send_pose_canfd

    c = cfg.collect
    cart = c.cartesian
    max_deg = cart.max_deg_for_slot(slot)
    dt_s = c.dt_ms / 1000.0
    duration = cart.duration_s
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

    print(f"\n  {slot}", flush=True)
    next_tick = time.monotonic()
    log_i = 0
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
                follow=c.follow, trajectory_mode=0, radio=0,
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
    finally:
        bot.stop_all()
        try:
            send_pose_canfd(bot.robot, pose0.tolist(), follow=c.follow, trajectory_mode=0, radio=0)
        except Exception:
            pass

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
    from rm75_control.motion.canfd import send_velocity_canfd

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
                    if pd.joint_duration_s > 0:
                        print("  resync pose d before burst…", flush=True)
                        # Joint movej_canfd must stop before planned movej + movev init.
                        bot.stop_all()
                        time.sleep(0.5)
                        move_j(bot.robot, q0, speed=c.move_speed)
                        burst_pose0, _ = wait_settle(
                            bot.robot, q0, timeout_s=c.settle_timeout_s,
                        )
                    else:
                        burst_pose0 = pose0.copy()
                    next_tick = ex.begin_pose_d_vel_burst(
                        bot, vb=vb, dt_ms=c.dt_ms,
                    )
                    movev_ready = True
                t_burst = t_cmd - pd.joint_duration_s
                vel_cmd, _ = ex.vel_burst_cmd(t_burst, vb, scale=c.scale)
                last_vel = vel_cmd
                send_velocity_canfd(
                    bot.robot, vel_cmd.tolist(),
                    follow=vb.follow,
                    trajectory_mode=vb.trajectory_mode,
                    radio=vb.radio,
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
        if movev_ready:
            try:
                bot.robot.rm_set_arm_slow_stop()
            except Exception:
                pass
        else:
            bot.stop_all()
        try:
            bot.robot.rm_movej_canfd(q0.tolist(), False, 0, 0, 0)
        except Exception:
            pass

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
        saved = []
        for slot in seq:
            q_tgt, _, rec = slots[slot]
            print(f"\nMove {slot}")
            move_j(bot.robot, q_tgt, speed=c.move_speed)
            wait_settle(bot.robot, q_tgt, timeout_s=c.settle_timeout_s)
            if slot == "d":
                saved.append(run_pose_d(bot, cfg))
            else:
                saved.append(run_cartesian(bot, cfg, slot))

        home = c.return_home
        q_h, _, _ = slots[home]
        print(f"\nReturn {home}")
        move_j(bot.robot, q_h, speed=c.move_speed)
        wait_settle(bot.robot, q_h, timeout_s=c.settle_timeout_s)
        print("\nCollection done:")
        for p in saved:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
