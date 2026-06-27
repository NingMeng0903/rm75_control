#!/usr/bin/env python3
"""
Multi-pose ID excitation: A → B → C → D → return A.

  a       — Cartesian ORIENT, 30 s default, cap 18°
  b, c    — Cartesian ORIENT, 30 s default, larger cap (28° default)
  d       — joint J7 excitation + Stage-2 inertia burst (same pose)

Saves force_id_pose_{a,b,c,d}.npz, then: python tmp/force_id/force_id_fit.py

Run:
  source env.sh
  python tmp/force_id/force_id_multi_pose.py --dry-run
  python tmp/force_id/force_id_multi_pose.py --yes
  python tmp/force_id/force_id_run_all.py --yes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import force_id_cartesian as fic  # noqa: E402
import force_id_collect_pose_d as posed  # noqa: E402

CONFIG = fic.CONFIG
POSES_YAML = fic.POSES_YAML
LOG_DIR = fic.LOG_DIR
DEG2RAD = fic.DEG2RAD
SEQUENCE = ("a", "b", "c", "d")
CARTESIAN_SLOTS = ("a", "b", "c")
JOINT_SLOTS = ("d",)


def load_slot_q(slot: str) -> tuple[np.ndarray, np.ndarray, dict]:
    data = fic.load_poses_yaml(POSES_YAML)
    rec = fic.get_slot_record(data, slot)
    if rec is None:
        raise SystemExit(f"Pose slot '{slot}' not saved in {POSES_YAML}")
    q = np.asarray(rec["q_deg"], dtype=float)
    pose = np.asarray(rec["pose_base"], dtype=float)
    return q, pose, rec


def move_j(robot, q_deg: np.ndarray, *, speed: int, block: bool = True) -> None:
    ret = robot.rm_movej(q_deg.tolist(), speed, 0, 0, 1 if block else 0)
    if ret != 0:
        raise RuntimeError(f"rm_movej to {q_deg.round(2)} failed: {ret}")


def wait_settle(robot, target_q: np.ndarray, *, timeout_s: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        ret, st = robot.rm_get_current_arm_state()
        if ret == 0:
            q = np.asarray(st["joint"][:7], dtype=float)
            if float(np.max(np.abs(q - target_q))) < 0.5:
                pose = np.asarray(st["pose"][:6], dtype=float)
                time.sleep(0.5)
                return pose, q
        time.sleep(0.1)
    ret, st = robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError("get state failed after movej")
    return np.asarray(st["pose"][:6], dtype=float), np.asarray(st["joint"][:7], dtype=float)


def preview_cart_slot(
    slot: str,
    rec: dict,
    *,
    duration: float,
    scale: float,
    orient_heavy: bool,
    max_mm: float,
    max_deg: float,
) -> None:
    if orient_heavy:
        amp_mm, amp_rot, freqs = fic.ORIENT_AMP_MM, fic.ORIENT_AMP_ROT_DEG, fic.ORIENT_FREQS_HZ
        preset = "ORIENT"
    else:
        amp_mm, amp_rot, freqs = fic.SLOW_AMP_MM, fic.SLOW_AMP_ROT_DEG, fic.SLOW_FREQS_HZ
        preset = "SLOW"
    cfg = fic.CartesianExcitation(amp_mm, amp_rot, freqs, scale)
    dt_s = 0.01
    ts = np.linspace(0, duration, int(duration / dt_s) + 1)
    max_mm_a = np.full(3, max_mm)
    max_rot_a = np.full(3, max_deg)
    deltas = []
    for t in ts:
        d = cfg.delta_pose(t)
        d = fic.clamp_delta(d, max_mm=max_mm_a, max_rot_deg=max_rot_a)
        deltas.append(d)
    deltas = np.asarray(deltas)
    v_lin, v_ang = fic.estimate_peak_speeds(amp_mm, amp_rot, freqs, scale)
    pose = np.asarray(rec["pose_base"], dtype=float)
    print(f"\n--- slot {slot} ({rec.get('label', '')}) cartesian ---")
    print("  pose_base:", [round(float(v), 4) for v in pose])
    print("  q_deg:", [round(float(v), 1) for v in rec["q_deg"]])
    print(
        "  preview mm:",
        [round(float(m * 1000), 1) for m in np.max(np.abs(deltas[:, 0:3]), axis=0)],
        "rot deg:",
        [round(float(m / DEG2RAD), 1) for m in np.max(np.abs(deltas[:, 3:6]), axis=0)],
        f"cap {max_deg}°",
    )
    print(
        "  peak ang vel deg/s:",
        [round(float(v / DEG2RAD), 1) for v in v_ang],
        f"| {preset} {duration}s",
    )
    print(f"  → {LOG_DIR / f'force_id_pose_{slot}.npz'}")


def preview_pose_d(rec: dict, *, joint_s: float, burst_s: float, scale: float) -> None:
    q0 = np.asarray(rec["q_deg"], dtype=float)
    print(f"\n--- slot d ({rec.get('label', '')}) joint + Stage-2 burst ---")
    print("  q_deg:", [round(float(v), 1) for v in q0])
    posed.preview_combined(q0, joint_s=joint_s, burst_s=burst_s, scale=scale)
    print(f"  total {joint_s + burst_s:.0f}s → {LOG_DIR / 'force_id_pose_d.npz'}")


def run_excitation(
    bot,
    *,
    slot: str,
    duration: float,
    dt_ms: float,
    scale: float,
    orient_heavy: bool,
    max_mm: float,
    max_deg: float,
    log_every: int,
    follow: bool,
    warmup_s: float,
) -> Path:
    from rm75_control.motion.canfd import send_pose_canfd

    if orient_heavy:
        amp_mm, amp_rot, freqs = fic.ORIENT_AMP_MM, fic.ORIENT_AMP_ROT_DEG, fic.ORIENT_FREQS_HZ
        preset = "ORIENT"
    else:
        amp_mm, amp_rot, freqs = fic.SLOW_AMP_MM, fic.SLOW_AMP_ROT_DEG, fic.SLOW_FREQS_HZ
        preset = "SLOW"

    dt_s = dt_ms / 1000.0
    max_mm_a = np.full(3, max_mm)
    max_rot_a = np.full(3, max_deg)
    out = LOG_DIR / f"force_id_pose_{slot}.npz"

    ret, state = bot.robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"get state failed: {ret}")
    pose0 = np.asarray(state["pose"][:6], dtype=float)
    q0 = np.asarray(state["joint"][:7], dtype=float)

    n_cmd = int(duration / dt_s) + 1
    n_log = (n_cmd + log_every - 1) // log_every
    t_log = np.zeros(n_log)
    pose_log = np.zeros((n_log, 6))
    q_log = np.zeros((n_log, 7))
    f_log = np.zeros((n_log, 6))
    delta_log = np.zeros((n_log, 6))

    print(
        f"\n>> Excite slot {slot} | {preset} | {duration}s cap {max_deg}° | {n_log} samples",
        flush=True,
    )
    t_start = time.monotonic()
    next_tick = t_start
    log_i = 0

    try:
        for i in range(n_cmd):
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += dt_s
            t_cmd = i * dt_s
            ramp = min(1.0, t_cmd / warmup_s) if warmup_s > 0 else 1.0
            exc = fic.CartesianExcitation(amp_mm, amp_rot, freqs, scale * ramp)
            delta = fic.clamp_delta(
                exc.delta_pose(t_cmd), max_mm=max_mm_a, max_rot_deg=max_rot_a
            )
            send_pose_canfd(
                bot.robot,
                (pose0 + delta).tolist(),
                follow=follow,
                trajectory_mode=0,
                radio=0,
            )
            if i % log_every == 0:
                ret_s, st = bot.robot.rm_get_current_arm_state()
                ret_f, fdata = bot.robot.rm_get_force_data()
                if ret_s == 0:
                    pose_log[log_i] = st["pose"][:6]
                    q_log[log_i] = st["joint"][:7]
                if ret_f == 0:
                    f_log[log_i] = np.asarray(fdata["force_data"][:6], dtype=float)
                delta_log[log_i] = delta
                t_log[log_i] = t_cmd
                log_i += 1
            if (i + 1) % int(1.0 / dt_s) == 0:
                print(f"   slot {slot} t={t_cmd:.0f}s", flush=True)
    finally:
        bot.stop_all()
        try:
            send_pose_canfd(bot.robot, pose0.tolist(), follow=follow, trajectory_mode=0, radio=0)
        except Exception:
            pass

    if out.exists():
        out.unlink()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        t=t_log[:log_i],
        pose=pose_log[:log_i],
        q_deg=q_log[:log_i],
        force_raw=f_log[:log_i],
        delta_pose=delta_log[:log_i],
        pose0=pose0,
        q0_deg=q0,
        amp_mm=amp_mm,
        amp_rot_deg=amp_rot,
        freqs_hz=np.array(freqs, dtype=object),
        scale=scale,
        max_delta_mm=max_mm,
        max_delta_deg=max_deg,
        dt_ms=dt_ms,
        log_every=log_every,
        pose_slot=slot,
        preset=preset,
        method="cartesian_fourier_multi",
    )
    print(f"   Saved {log_i} samples → {out}", flush=True)
    return out


def run_pose_d_combined(
    bot,
    *,
    joint_s: float,
    burst_s: float,
    scale: float,
    log_every: int,
    warmup_s: float,
    follow: bool,
) -> Path:
    from rm75_control.motion.canfd import send_pose_canfd

    dt_s = 0.01
    total_s = joint_s + burst_s
    out = LOG_DIR / "force_id_pose_d.npz"

    ret, state = bot.robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"get state failed: {ret}")
    pose0 = np.asarray(state["pose"][:6], dtype=float)
    q0 = np.asarray(state["joint"][:7], dtype=float)

    n_total = int(total_s / dt_s) + 1
    n_log = (n_total + log_every - 1) // log_every
    t_log = np.zeros(n_log)
    pose_log = np.zeros((n_log, 6))
    q_log = np.zeros((n_log, 7))
    f_log = np.zeros((n_log, 6))
    phase_log = np.zeros(n_log, dtype=np.int8)

    print(
        f"\n>> Excite slot d | joint {joint_s}s + burst {burst_s}s | {n_log} samples",
        flush=True,
    )
    t_start = time.monotonic()
    next_tick = t_start
    log_i = 0

    try:
        for i in range(n_total):
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += dt_s
            t_cmd = i * dt_s
            ramp = min(1.0, t_cmd / warmup_s) if warmup_s > 0 else 1.0

            if t_cmd < joint_s:
                q_cmd = posed.joint_cmd(t_cmd, q0, scale * ramp)
                ret = bot.robot.rm_movej_canfd(q_cmd.tolist(), False, 0, 0, 0)
                phase = 0
            else:
                t_burst = t_cmd - joint_s
                ramp_b = min(1.0, t_burst / min(3.0, burst_s * 0.1))
                delta = fic.inertia_burst_delta(t_burst) * ramp_b
                send_pose_canfd(
                    bot.robot,
                    (pose0 + delta).tolist(),
                    follow=follow,
                    trajectory_mode=0,
                    radio=0,
                )
                phase = 1
                ret = 0

            if ret != 0:
                raise RuntimeError(f"pose d command failed: {ret}")

            if i % log_every == 0:
                ret_s, st = bot.robot.rm_get_current_arm_state()
                ret_f, fdata = bot.robot.rm_get_force_data()
                if ret_s == 0:
                    pose_log[log_i] = st["pose"][:6]
                    q_log[log_i] = st["joint"][:7]
                if ret_f == 0:
                    f_log[log_i] = np.asarray(fdata["force_data"][:6], dtype=float)
                t_log[log_i] = t_cmd
                phase_log[log_i] = phase
                log_i += 1

            if (i + 1) % int(1.0 / dt_s) == 0:
                if t_cmd < joint_s:
                    q_now = posed.joint_cmd(t_cmd, q0, scale)
                    print(f"   slot d t={t_cmd:.0f}s joint J7={q_now[6]-q0[6]:+.1f}deg", flush=True)
                else:
                    seg = int((t_cmd - joint_s) // fic.INERTIA_BURST_SEGMENT_S) % 3
                    ax = ("rx", "ry", "rz")[seg]
                    print(f"   slot d t={t_cmd:.0f}s burst {ax}", flush=True)
    finally:
        bot.stop_all()
        try:
            bot.robot.rm_movej_canfd(q0.tolist(), False, 0, 0, 0)
            send_pose_canfd(bot.robot, pose0.tolist(), follow=follow, trajectory_mode=0, radio=0)
        except Exception:
            pass

    if out.exists():
        out.unlink()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        t=t_log[:log_i],
        pose=pose_log[:log_i],
        q_deg=q_log[:log_i],
        force_raw=f_log[:log_i],
        phase=phase_log[:log_i],
        pose0=pose0,
        q0_deg=q0,
        pose_slot="d",
        preset="pose_d_joint+inertia_burst",
        joint_s=joint_s,
        burst_s=burst_s,
        scale=scale,
        dt_ms=10.0,
        log_every=log_every,
        method="joint+inertia_burst",
    )
    print(f"   Saved {log_i} samples → {out}", flush=True)
    return out


def dry_run_all(args) -> int:
    print(f"Multi-pose dry-run | {' → '.join(SEQUENCE)} → a")
    print(f"Poses file: {POSES_YAML}")
    max_mm = 5.0 if args.orient_heavy else args.max_delta_mm
    for slot in SEQUENCE:
        _, _, rec = load_slot_q(slot)
        if slot in JOINT_SLOTS:
            preview_pose_d(
                rec,
                joint_s=args.duration_d_joint,
                burst_s=args.duration_d_burst,
                scale=args.scale,
            )
        else:
            max_deg = fic.slot_orient_max_deg(
                slot, default_deg=args.max_orient_deg, bc_deg=args.bc_max_deg
            )
            preview_cart_slot(
                slot,
                rec,
                duration=args.duration_abc,
                scale=args.scale,
                orient_heavy=args.orient_heavy,
                max_mm=max_mm,
                max_deg=max_deg,
            )
    abc_t = 3 * args.duration_abc
    d_t = args.duration_d_joint + args.duration_d_burst
    print(f"\nTiming: abc {abc_t:.0f}s + d {d_t:.0f}s = {abc_t + d_t:.0f}s excitation")
    print("\nPath plan (rm_movej):")
    plan = list(SEQUENCE) + ["a"]
    for i in range(len(plan) - 1):
        print(f"  {plan[i]} → {plan[i+1]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="A→B→C→D (abc 30s cart, d joint+burst), return A")
    parser.add_argument("--duration-abc", type=float, default=30.0, help="seconds at a,b,c")
    parser.add_argument("--duration-d-joint", type=float, default=30.0, help="pose d joint phase")
    parser.add_argument("--duration-d-burst", type=float, default=45.0, help="pose d Stage-2 burst")
    parser.add_argument("--duration", type=float, default=None, help="legacy: sets all phases to same value")
    parser.add_argument("--dt-ms", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--no-orient-heavy", action="store_true")
    parser.add_argument("--move-speed", type=int, default=15)
    parser.add_argument("--max-delta-mm", type=float, default=8.0)
    parser.add_argument("--max-orient-deg", type=float, default=18.0, help="attitude cap at pose a")
    parser.add_argument("--bc-max-deg", type=float, default=28.0, help="attitude cap at pose b,c")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    args.orient_heavy = not args.no_orient_heavy
    if args.duration is not None:
        args.duration_abc = args.duration
        args.duration_d_joint = args.duration
        args.duration_d_burst = max(45.0, args.duration * 0.75)
    if args.log_every < 1:
        parser.error("--log-every >= 1")

    if args.dry_run:
        return dry_run_all(args)

    max_mm = 5.0 if args.orient_heavy else args.max_delta_mm

    from rm75_control import RobotSession

    slots = {s: load_slot_q(s) for s in SEQUENCE + ("a",)}
    print("Sequence: A,B,C cartesian (30s) → D joint+Stage-2 → return A")
    print(f"  a cap {args.max_orient_deg}°  |  b,c cap {args.bc_max_deg}°")
    print(f"  d: joint {args.duration_d_joint}s + burst {args.duration_d_burst}s")
    for s in SEQUENCE:
        mode = "joint+burst" if s in JOINT_SLOTS else "cartesian"
        print(f"  {s} [{mode}]: {slots[s][2].get('label')}")

    if not args.yes:
        input("Confirm FREE SPACE for A→B→C→D→A. Enter...")

    with RobotSession(config=CONFIG) as bot:
        saved = []
        for slot in SEQUENCE:
            q_tgt, _, rec = slots[slot]
            print(f"\n== Move to {slot} ({rec.get('label')}) ==")
            move_j(bot.robot, q_tgt, speed=args.move_speed)
            pose0, _ = wait_settle(bot.robot, q_tgt)
            dmm, ddeg = fic.pose_drift_mm_deg(pose0, np.asarray(rec["pose_base"]))
            print(f"   settled drift vs yaml: {dmm:.1f} mm, {ddeg:.1f} deg")

            if slot in JOINT_SLOTS:
                out = run_pose_d_combined(
                    bot,
                    joint_s=args.duration_d_joint,
                    burst_s=args.duration_d_burst,
                    scale=args.scale,
                    log_every=args.log_every,
                    warmup_s=args.warmup_s,
                    follow=args.follow,
                )
            else:
                max_deg = fic.slot_orient_max_deg(
                    slot, default_deg=args.max_orient_deg, bc_deg=args.bc_max_deg
                )
                out = run_excitation(
                    bot,
                    slot=slot,
                    duration=args.duration_abc,
                    dt_ms=args.dt_ms,
                    scale=args.scale,
                    orient_heavy=args.orient_heavy,
                    max_mm=max_mm,
                    max_deg=max_deg,
                    log_every=args.log_every,
                    follow=args.follow,
                    warmup_s=args.warmup_s,
                )
            saved.append(out)

        print("\n== Return to pose A ==")
        q_a, _, rec_a = slots["a"]
        move_j(bot.robot, q_a, speed=args.move_speed)
        pose0, _ = wait_settle(bot.robot, q_a)
        dmm, ddeg = fic.pose_drift_mm_deg(pose0, np.asarray(rec_a["pose_base"]))
        print(f"   home drift: {dmm:.1f} mm, {ddeg:.1f} deg")
        print("\nDone.", flush=True)
        for p in saved:
            print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
