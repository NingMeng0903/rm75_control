#!/usr/bin/env python3
"""
Plan A: joint-space excitation for 6D force sensor ID (no contact, no force-position stream).

For a safer workspace-limited alternative, see tmp/force_id/force_id_cartesian.py (Plan B).

Design (safe RM75):
  - Center trajectory at CURRENT joint pose q0 (you place arm in air manually first).
  - Small multi-harmonic oscillations per joint (Fourier), NOT large workspace moves.
  - Clamp to controller joint limits with margin (avoid self-collision / singular extremes).
  - 100Hz rm_movej_canfd ONLY; log raw force_data + joint + pose for offline OLS.

Run:
  source env.sh
  # NOT required to be joint [0,0,0,...]. Use ANY safe mid-range pose in free air
  # (e.g. your usual scan pose). Avoid folded elbow and joint limits.
  python tmp/force_id/force_id_excitation.py --dry-run
  python tmp/force_id/force_id_excitation.py --duration 50 --scale 0.5

What it sends every 10ms:
  rm_movej_canfd(q_target_deg, follow=..., traj=0)  — joint POSITION setpoints, not velocity.
  q_target = q0 + small sin waves (default SLOW freqs ~0.12–0.20 Hz).

After run: use filtfilt + numerical differentiation on logged q/pose to build W, phi.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _paths import CONFIG_ROBOT, LOG_DIR  # noqa: E402

CONFIG = CONFIG_ROBOT

# Slow ID preset (default): ~0.8–2.5 deg/s peak per joint at scale=1.0
SLOW_AMP_DEG = np.array([10.0, 6.0, 8.0, 10.0, 8.0, 12.0, 12.0], dtype=float)
SLOW_FREQS_HZ = [
    [0.12, 0.17],
    [0.13, 0.19],
    [0.14, 0.18],
    [0.11, 0.16],
    [0.15, 0.20],
    [0.12, 0.18],
    [0.13, 0.17],
]

# Faster preset (--fast): for validation after --slow works
FAST_AMP_DEG = np.array([18.0, 12.0, 15.0, 18.0, 15.0, 20.0, 20.0], dtype=float)
FAST_FREQS_HZ = [
    [0.23, 0.37],
    [0.29, 0.41],
    [0.31, 0.43],
    [0.27, 0.39],
    [0.33, 0.47],
    [0.25, 0.35],
    [0.28, 0.44],
]

DEFAULT_AMP_DEG = SLOW_AMP_DEG
DEFAULT_FREQS_HZ = SLOW_FREQS_HZ


@dataclass(frozen=True)
class ExcitationConfig:
    amp_deg: np.ndarray
    freqs_hz: list[list[float]]
    scale: float
    limit_margin_deg: float

    def joint_command(self, t_s: float, q0_deg: np.ndarray) -> np.ndarray:
        q = q0_deg.copy()
        for j in range(len(q0_deg)):
            amps = self.amp_deg[j] * self.scale
            for k, f in enumerate(self.freqs_hz[j]):
                # Two harmonics with phase offset per joint/harmonic for richer excitation
                phase = (j + 1) * 0.7 + k * 1.3
                q[j] += amps * math.sin(2.0 * math.pi * f * t_s + phase)
        return q


def estimate_peak_joint_speed_deg_s(
    amp_deg: np.ndarray, freqs_hz: list[list[float]], scale: float
) -> np.ndarray:
    """Upper bound: sum_k |2*pi*f_k*A_k| per joint (harmonics peak-aligned)."""
    peak = np.zeros(len(amp_deg))
    for j in range(len(amp_deg)):
        for f in freqs_hz[j]:
            peak[j] += 2.0 * math.pi * f * amp_deg[j] * scale
    return peak


def clamp_joints(
    q_deg: np.ndarray,
    q0_deg: np.ndarray,
    *,
    j_min: np.ndarray | None,
    j_max: np.ndarray | None,
    margin_deg: float,
    max_delta_deg: float,
) -> np.ndarray:
    """Clamp to limits and max deviation from q0."""
    out = q_deg.copy()
    if j_min is not None and j_max is not None:
        lo = np.asarray(j_min, dtype=float) + margin_deg
        hi = np.asarray(j_max, dtype=float) - margin_deg
        out = np.clip(out, lo, hi)
    delta = np.clip(out - q0_deg, -max_delta_deg, max_delta_deg)
    return q0_deg + delta


def read_joint_limits(robot) -> tuple[np.ndarray | None, np.ndarray | None]:
    ret_lo, j_min = robot.rm_get_joint_min_pos()
    ret_hi, j_max = robot.rm_get_joint_max_pos()
    if ret_lo != 0 or ret_hi != 0:
        return None, None
    return np.asarray(j_min[:7], dtype=float), np.asarray(j_max[:7], dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safe joint Fourier excitation + raw force logging for dynamic ID"
    )
    parser.add_argument("--duration", type=float, default=50.0, help="record length (s)")
    parser.add_argument("--dt-ms", type=float, default=10.0, help="CANFD period (ms)")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="scale amplitude table (0.3=very gentle, 0.5=default slow, 1.0=full table)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use faster/larger FAST_AMP/FREQ table (only after slow run OK)",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="rm_movej_canfd high follow (snappier; default OFF = gentler tracking)",
    )
    parser.add_argument(
        "--max-delta-deg",
        type=float,
        default=22.0,
        help="max |q - q0| per joint (hard cap regardless of amp table)",
    )
    parser.add_argument(
        "--limit-margin-deg",
        type=float,
        default=5.0,
        help="stay this many deg inside controller joint limits",
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=5.0,
        help="ramp amplitude 0→scale over this time at start",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print q range preview only, no motion",
    )
    args = parser.parse_args()

    dt_s = args.dt_ms / 1000.0
    amp_table = FAST_AMP_DEG if args.fast else SLOW_AMP_DEG
    freq_table = FAST_FREQS_HZ if args.fast else SLOW_FREQS_HZ
    cfg = ExcitationConfig(
        amp_deg=amp_table,
        freqs_hz=freq_table,
        scale=args.scale,
        limit_margin_deg=args.limit_margin_deg,
    )
    v_peak = estimate_peak_joint_speed_deg_s(amp_table, freq_table, args.scale)

    from rm75_control import RobotSession

    print("Connecting...", flush=True)
    with RobotSession(config=CONFIG) as bot:
        ret, state = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            print(f"get state failed: {ret}", file=sys.stderr)
            return 1

        q0 = np.asarray(state["joint"][:7], dtype=float)
        j_min, j_max = read_joint_limits(bot.robot)

        print("q0 (deg):", [round(v, 2) for v in q0])
        if j_min is not None:
            print("joint limits (deg): min", [round(v, 1) for v in j_min])
            print("                    max", [round(v, 1) for v in j_max])

        # Preview commanded range over one period envelope
        ts = np.linspace(0, args.duration, int(args.duration / dt_s) + 1)
        qs = []
        for t in ts:
            q_cmd = cfg.joint_command(t, q0)
            q_cmd = clamp_joints(
                q_cmd,
                q0,
                j_min=j_min,
                j_max=j_max,
                margin_deg=args.limit_margin_deg,
                max_delta_deg=args.max_delta_deg,
            )
            qs.append(q_cmd)
        qs = np.asarray(qs)
        delta = qs - q0
        print(
            f"preview |q-q0| max per joint (deg): "
            f"{[round(float(m), 1) for m in np.max(np.abs(delta), axis=0)]}"
        )
        print(
            f"est. peak joint speed bound (deg/s): "
            f"{[round(float(v), 2) for v in v_peak]}  "
            f"(~{[round(float(v)/360*60, 1) for v in v_peak]} rpm)"
        )
        if args.dry_run:
            print("dry-run: no motion.", flush=True)
            return 0

        print(
            f"Excitation: {'FAST' if args.fast else 'SLOW'} scale={args.scale} "
            f"duration={args.duration}s dt={args.dt_ms}ms follow={args.follow} | Ctrl+C abort",
            flush=True,
        )
        input("Confirm arm is in FREE SPACE (no contact). Press Enter to start...")

        n = int(args.duration / dt_s) + 1
        t_log = np.zeros(n)
        q_log = np.zeros((n, 7))
        pose_log = np.zeros((n, 6))
        f_log = np.zeros((n, 6))

        t_start = time.monotonic()
        next_tick = t_start
        i = 0

        try:
            while i < n:
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)
                t_now = now - t_start
                next_tick += dt_s

                ramp = min(1.0, t_now / args.warmup_s) if args.warmup_s > 0 else 1.0
                exc = ExcitationConfig(
                    amp_deg=cfg.amp_deg,
                    freqs_hz=cfg.freqs_hz,
                    scale=cfg.scale * ramp,
                    limit_margin_deg=cfg.limit_margin_deg,
                )
                q_cmd = exc.joint_command(t_now, q0)
                q_cmd = clamp_joints(
                    q_cmd,
                    q0,
                    j_min=j_min,
                    j_max=j_max,
                    margin_deg=args.limit_margin_deg,
                    max_delta_deg=args.max_delta_deg,
                )

                ret = bot.robot.rm_movej_canfd(
                    q_cmd.tolist(), args.follow, 0, 0, 0
                )
                if ret != 0:
                    print(f"rm_movej_canfd failed: {ret}", file=sys.stderr)
                    break

                ret_s, st = bot.robot.rm_get_current_arm_state()
                ret_f, fdata = bot.robot.rm_get_force_data()
                if ret_s == 0:
                    pose_log[i] = st["pose"][:6]
                    q_log[i] = st["joint"][:7]
                if ret_f == 0:
                    f_log[i] = np.asarray(fdata["force_data"][:6], dtype=float)
                t_log[i] = t_now
                i += 1

                if i % int(1.0 / dt_s) == 0:
                    print(f"  t={t_now:.1f}s", flush=True)
        except KeyboardInterrupt:
            print("\nCtrl+C — stopping", flush=True)
        finally:
            bot.stop_all()
            # Hold q0 on exit
            try:
                bot.robot.rm_movej_canfd(q0.tolist(), args.follow, 0, 0, 0)
            except Exception:
                pass

        n_saved = i
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = LOG_DIR / f"force_id_{stamp}.npz"
        np.savez(
            out,
            t=t_log[:n_saved],
            q_deg=q_log[:n_saved],
            pose=pose_log[:n_saved],
            force_raw=f_log[:n_saved],
            q0_deg=q0,
            amp_deg=DEFAULT_AMP_DEG,
            freqs_hz=np.array(DEFAULT_FREQS_HZ, dtype=object),
            scale=args.scale,
            max_delta_deg=args.max_delta_deg,
            limit_margin_deg=args.limit_margin_deg,
            dt_ms=args.dt_ms,
        )
        print(f"Saved {n_saved} samples → {out}", flush=True)
        print(
            "Next: filtfilt q→qd→qdd, FK→R_base_sensor, build W, lstsq phi; "
            "validate on NEW trajectory.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
