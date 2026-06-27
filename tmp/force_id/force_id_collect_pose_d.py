#!/usr/bin/env python3
"""Pose-D: joint Fourier (J7 boost) + Stage-2 inertia burst (Rx/Ry/Rz) at same pose."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import force_id_cartesian as fic

CONFIG = fic.CONFIG
LOG_DIR = fic.LOG_DIR
OUT = LOG_DIR / "force_id_pose_d.npz"

# Per-harmonic amps at scale=1; J7 (index 6) largest
POSE_D_AMP_DEG = np.array([10.0, 8.0, 8.0, 14.0, 16.0, 14.0, 32.0], dtype=float)
POSE_D_FREQS = [
    [0.14, 0.21],
    [0.13, 0.19],
    [0.12, 0.18],
    [0.22, 0.31],
    [0.24, 0.33],
    [0.23, 0.30],
    [0.20, 0.29],
]
MAX_DELTA = np.array([12.0, 12.0, 12.0, 18.0, 20.0, 18.0, 35.0], dtype=float)


def joint_cmd(t: float, q0: np.ndarray, scale: float) -> np.ndarray:
    q = q0.copy()
    for j in range(7):
        a = POSE_D_AMP_DEG[j] * scale
        for k, f in enumerate(POSE_D_FREQS[j]):
            ph = (j + 1) * 0.8 + k * 1.2
            q[j] += a * math.sin(2 * math.pi * f * t + ph)
    delta = q - q0
    delta = np.clip(delta, -MAX_DELTA, MAX_DELTA)
    return q0 + delta


def preview_combined(
    q0: np.ndarray,
    *,
    joint_s: float,
    burst_s: float,
    scale: float,
) -> None:
    dt_s = 0.01
    ts_j = np.linspace(0, joint_s, int(joint_s / dt_s) + 1)
    qs = np.array([joint_cmd(t, q0, scale) for t in ts_j])
    print("  joint |q-q0| max deg:", np.round(np.max(np.abs(qs - q0), axis=0), 1))
    print(f"  J7 max: {float(np.max(np.abs(qs[:, 6] - q0[6]))):.1f} deg  ({joint_s:.0f}s)")
    ts_b = np.linspace(0, burst_s, int(burst_s / dt_s) + 1)
    deltas = np.array([fic.inertia_burst_delta(t) for t in ts_b])
    rot_deg = np.max(np.abs(deltas[:, 3:6]), axis=0) / fic.DEG2RAD
    v_peak_deg_s = 2 * math.pi * max(fic.INERTIA_BURST_FREQS_HZ) * fic.INERTIA_BURST_MAX_ROT_DEG
    print(
        f"  burst rot max deg [rx,ry,rz]: {[round(float(v), 1) for v in rot_deg]}  "
        f"cap {fic.INERTIA_BURST_MAX_ROT_DEG:.0f}°  "
        f"freq {fic.INERTIA_BURST_FREQS_HZ} Hz  "
        f"peak ω≈{v_peak_deg_s:.0f} deg/s  "
        f"({burst_s:.0f}s, {fic.INERTIA_BURST_SEGMENT_S:.0f}s/axis)",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-s", type=float, default=30.0)
    parser.add_argument("--burst-s", type=float, default=45.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    from rm75_control import RobotSession
    from rm75_control.motion.canfd import send_pose_canfd

    dt_s = 0.01
    total_s = args.joint_s + args.burst_s
    with RobotSession(config=CONFIG) as bot:
        ret, st = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            return 1
        q0 = np.asarray(st["joint"][:7], dtype=float)
        pose0 = np.asarray(st["pose"][:6], dtype=float)
        print("pose0:", [round(v, 4) for v in pose0])
        print("q0:", [round(v, 1) for v in q0])
        preview_combined(q0, joint_s=args.joint_s, burst_s=args.burst_s, scale=args.scale)
        if args.dry_run:
            return 0
        if not args.yes:
            input("FREE SPACE. Enter to collect pose-d (joint + inertia burst)...")

        n_total = int(total_s / dt_s) + 1
        n_log = (n_total + args.log_every - 1) // args.log_every
        t_log = np.zeros(n_log)
        pose_log = np.zeros((n_log, 6))
        q_log = np.zeros((n_log, 7))
        f_log = np.zeros((n_log, 6))
        phase_log = np.zeros(n_log, dtype=np.int8)  # 0=joint, 1=burst

        li = 0
        t_start = time.monotonic()
        next_tick = t_start
        try:
            for i in range(n_total):
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)
                next_tick += dt_s
                t_cmd = i * dt_s
                ramp = min(1.0, t_cmd / args.warmup_s) if args.warmup_s > 0 else 1.0

                if t_cmd < args.joint_s:
                    q_cmd = joint_cmd(t_cmd, q0, args.scale * ramp)
                    ret = bot.robot.rm_movej_canfd(q_cmd.tolist(), False, 0, 0, 0)
                    phase = 0
                else:
                    t_burst = t_cmd - args.joint_s
                    ramp_b = min(1.0, t_burst / min(3.0, args.burst_s * 0.1))
                    delta = fic.inertia_burst_delta(t_burst) * ramp_b
                    send_pose_canfd(
                        bot.robot,
                        (pose0 + delta).tolist(),
                        follow=False,
                        trajectory_mode=0,
                        radio=0,
                    )
                    phase = 1
                    ret = 0

                if ret != 0:
                    print(f"command fail {ret} at t={t_cmd:.1f}s")
                    break

                if i % args.log_every == 0:
                    ret_s, st = bot.robot.rm_get_current_arm_state()
                    ret_f, fd = bot.robot.rm_get_force_data()
                    if ret_s == 0:
                        pose_log[li] = st["pose"][:6]
                        q_log[li] = st["joint"][:7]
                    if ret_f == 0:
                        f_log[li] = np.asarray(fd["force_data"][:6])
                    t_log[li] = t_cmd
                    phase_log[li] = phase
                    li += 1

                if (i + 1) % int(1.0 / dt_s) == 0:
                    if t_cmd < args.joint_s:
                        q_now = joint_cmd(t_cmd, q0, args.scale)
                        print(f"  t={t_cmd:.0f}s joint J7={q_now[6]-q0[6]:+.1f}deg", flush=True)
                    else:
                        seg = int((t_cmd - args.joint_s) // fic.INERTIA_BURST_SEGMENT_S) % 3
                        ax = ("rx", "ry", "rz")[seg]
                        print(f"  t={t_cmd:.0f}s burst {ax}", flush=True)
        finally:
            bot.stop_all()
            try:
                bot.robot.rm_movej_canfd(q0.tolist(), False, 0, 0, 0)
                send_pose_canfd(bot.robot, pose0.tolist(), follow=False, trajectory_mode=0, radio=0)
            except Exception:
                pass

        if OUT.exists():
            OUT.unlink()
        np.savez(
            OUT,
            t=t_log[:li],
            pose=pose_log[:li],
            q_deg=q_log[:li],
            force_raw=f_log[:li],
            phase=phase_log[:li],
            pose0=pose0,
            q0_deg=q0,
            pose_slot="d",
            preset="pose_d_joint+inertia_burst",
            joint_s=args.joint_s,
            burst_s=args.burst_s,
            scale=args.scale,
            dt_ms=10.0,
            log_every=args.log_every,
            method="joint+inertia_burst",
        )
        print(f"Saved {li} → {OUT}  (joint {args.joint_s}s + burst {args.burst_s}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
