#!/usr/bin/env python3
"""
Plan B: Cartesian (base-frame) Fourier excitation for 6D force dynamic ID.

Safer / more intuitive than joint-space ID:
  - You teach a comfortable scan pose in free air → that pose is pose0 (center).
  - TCP moves in a SMALL box around pose0 (mm + deg), NOT ±80mm scan swings.
  - Pure rm_movep_canfd @ 10ms — NO force-position stream, NO contact.

Run:
  source env.sh
  python tmp/force_id/force_id_cartesian.py --dry-run
  python tmp/force_id/force_id_cartesian.py --duration 60

Output: tmp/force_id/logs/force_id_cartesian.npz
  Per pose: tmp/force_id/logs/force_id_pose_<a|b|c|d>.npz

Pose slots in configs/force_id_poses.yaml:
  python tmp/force_id/force_id_cartesian.py --save-pose a
  python tmp/force_id/force_id_run_all.py --yes   # full pipeline
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _paths import CONFIG_ROBOT, DEFAULT_NPZ, LOG_DIR, POSES_YAML  # noqa: E402

CONFIG = CONFIG_ROBOT
DEFAULT_OUT = DEFAULT_NPZ
POSE_SLOTS = ("a", "b", "c", "d")

DEG2RAD = math.pi / 180.0

# ID-oriented preset (scale=1): modest translation, larger attitude wobble
SLOW_AMP_MM = np.array([5.0, 6.0, 3.0], dtype=float)  # base X,Y,Z
SLOW_AMP_ROT_DEG = np.array([8.0, 10.0, 8.0], dtype=float)  # rx,ry,rz per harmonic
SLOW_FREQS_HZ = [
    [0.12, 0.18],
    [0.13, 0.19],
    [0.11, 0.16],
    [0.14, 0.21],
    [0.15, 0.22],
    [0.13, 0.20],
]

FAST_AMP_MM = np.array([8.0, 10.0, 5.0], dtype=float)
FAST_AMP_ROT_DEG = np.array([10.0, 12.0, 10.0], dtype=float)
FAST_FREQS_HZ = [
    [0.20, 0.31],
    [0.23, 0.35],
    [0.19, 0.28],
    [0.25, 0.37],
    [0.27, 0.39],
    [0.24, 0.33],
]

# Larger attitude excitation for Mx/My/Mz ID; small translation
ORIENT_AMP_MM = np.array([3.0, 4.0, 2.0], dtype=float)
ORIENT_AMP_ROT_DEG = np.array([12.0, 15.0, 12.0], dtype=float)
ORIENT_FREQS_HZ = [
    [0.12, 0.18],
    [0.13, 0.19],
    [0.11, 0.16],
    [0.22, 0.33],
    [0.25, 0.37],
    [0.24, 0.31],
]

# Stage-2 (pose d only): high-freq single-axis Rx→Ry→Rz bursts for inertia ID
# Aggressive preset (~4× α vs old 10°/1.15Hz/12° cap): amp 18°, cap 25°, 1.1–1.55 Hz
INERTIA_BURST_SEGMENT_S = 15.0
INERTIA_BURST_AMP_ROT_DEG = 18.0
INERTIA_BURST_FREQS_HZ = [1.1, 1.55]
INERTIA_BURST_MAX_ROT_DEG = 25.0


def inertia_burst_delta(t_s: float) -> np.ndarray:
    """15 s per axis: drx, dry, drz — higher frequency than ORIENT."""
    seg = int(t_s // INERTIA_BURST_SEGMENT_S) % 3
    t_loc = t_s - seg * INERTIA_BURST_SEGMENT_S
    axis = 3 + seg
    amp = INERTIA_BURST_AMP_ROT_DEG * DEG2RAD
    delta = np.zeros(6, dtype=float)
    for k, f in enumerate(INERTIA_BURST_FREQS_HZ):
        ph = seg * 1.4 + k * 0.85
        delta[axis] += amp * math.sin(2.0 * math.pi * f * t_loc + ph)
    cap = INERTIA_BURST_MAX_ROT_DEG * DEG2RAD
    delta[3:6] = np.clip(delta[3:6], -cap, cap)
    return delta


def slot_orient_max_deg(slot: str, *, default_deg: float, bc_deg: float) -> float:
    if slot in ("b", "c"):
        return bc_deg
    return default_deg


def load_poses_yaml(path: Path) -> dict:
    if not path.exists():
        return {"poses": {s: {"pose_base": None, "q_deg": None} for s in POSE_SLOTS}}
    return yaml.safe_load(path.read_text()) or {"poses": {}}


def save_poses_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True))


def save_pose_slot(
    path: Path, slot: str, pose6: np.ndarray, q_deg: np.ndarray, label: str | None
) -> None:
    data = load_poses_yaml(path)
    poses = data.setdefault("poses", {})
    poses[slot] = {
        "label": label or f"pose_{slot}",
        "note": f"saved {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "pose_base": [round(float(v), 6) for v in pose6],
        "q_deg": [round(float(v), 3) for v in q_deg],
    }
    save_poses_yaml(path, data)


def get_slot_record(data: dict, slot: str) -> dict | None:
    rec = data.get("poses", {}).get(slot)
    if not rec or rec.get("pose_base") is None:
        return None
    return rec


def pose_drift_mm_deg(current: np.ndarray, recorded: np.ndarray) -> tuple[float, float]:
    dpos = float(np.linalg.norm(current[:3] - recorded[:3])) * 1000.0
    deul = np.abs(current[3:6] - recorded[3:6])
    deul = np.minimum(deul, 2 * math.pi - deul)
    ddeg = float(np.max(deul) * 180.0 / math.pi)
    return dpos, ddeg


def output_for_pose(slot: str | None, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if slot:
        return LOG_DIR / f"force_id_pose_{slot}.npz"
    return DEFAULT_OUT
@dataclass(frozen=True)
class CartesianExcitation:
    amp_mm: np.ndarray
    amp_rot_deg: np.ndarray
    freqs_hz: list[list[float]]
    scale: float

    def delta_pose(self, t_s: float) -> np.ndarray:
        """6D offset from pose0: [dx_m, dy_m, dz_m, drx, dry, drz] rad."""
        out = np.zeros(6, dtype=float)
        amps_m = self.amp_mm * self.scale / 1000.0
        amps_rad = self.amp_rot_deg * self.scale * DEG2RAD
        amp_list = [
            amps_m[0],
            amps_m[1],
            amps_m[2],
            amps_rad[0],
            amps_rad[1],
            amps_rad[2],
        ]
        for axis in range(6):
            for k, f in enumerate(self.freqs_hz[axis]):
                phase = (axis + 1) * 0.9 + k * 1.1
                out[axis] += amp_list[axis] * math.sin(2.0 * math.pi * f * t_s + phase)
        return out

    def command_pose(self, t_s: float, pose0: np.ndarray) -> np.ndarray:
        return pose0 + self.delta_pose(t_s)


def clamp_delta(
    delta: np.ndarray,
    *,
    max_mm: np.ndarray,
    max_rot_deg: np.ndarray,
) -> np.ndarray:
    out = delta.copy()
    out[0:3] = np.clip(out[0:3], -max_mm / 1000.0, max_mm / 1000.0)
    max_rot_rad = max_rot_deg * DEG2RAD
    out[3:6] = np.clip(out[3:6], -max_rot_rad, max_rot_rad)
    return out


def estimate_peak_speeds(
    amp_mm: np.ndarray,
    amp_rot_deg: np.ndarray,
    freqs_hz: list[list[float]],
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Upper bound per axis: sum_k |2*pi*f*A| for linear (m/s) and angular (rad/s)."""
    lin = np.zeros(3)
    ang = np.zeros(3)
    amps_m = amp_mm * scale / 1000.0
    amps_rad = amp_rot_deg * scale * DEG2RAD
    for axis in range(3):
        for f in freqs_hz[axis]:
            lin[axis] += 2.0 * math.pi * f * amps_m[axis]
    for axis in range(3):
        for f in freqs_hz[axis + 3]:
            ang[axis] += 2.0 * math.pi * f * amps_rad[axis]
    return lin, ang


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safe Cartesian Fourier excitation + raw force logging (Plan B)"
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--dt-ms", type=float, default=10.0)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="amplitude scale (1.0=default ID table; use 0.5 for first touch)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="larger FAST amp/freq table (only after slow run is OK)",
    )
    parser.add_argument(
        "--orient-heavy",
        action="store_true",
        help="prioritize rx/ry/rz (12–15 deg/harmonic, faster rot freqs); small translation",
    )
    parser.add_argument(
        "--pose",
        choices=POSE_SLOTS,
        help="pose slot a/b/c (metadata + output force_id_pose_<slot>.npz)",
    )
    parser.add_argument(
        "--save-pose",
        choices=POSE_SLOTS,
        metavar="SLOT",
        help="save current arm to configs/force_id_poses.yaml and exit",
    )
    parser.add_argument(
        "--pose-label",
        type=str,
        default="",
        help="optional label when using --save-pose",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="high follow (default OFF = gentler rm_movep_canfd tracking)",
    )
    parser.add_argument(
        "--max-drift-mm",
        type=float,
        default=8.0,
        help="warn if current TCP differs from saved --pose slot by more than this",
    )
    parser.add_argument(
        "--max-delta-mm",
        type=float,
        default=8.0,
        help="hard cap |dx,dy,dz| from pose0 (mm)",
    )
    parser.add_argument(
        "--max-delta-deg",
        type=float,
        default=15.0,
        help="hard cap |drx,dry,drz| from pose0 (deg)",
    )
    parser.add_argument("--warmup-s", type=float, default=5.0)
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="log state+force every N CANFD ticks (10≈10Hz log, 100Hz motion; each read ~40–80ms)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="npz path (default: tmp/force_id/logs/force_id_cartesian.npz)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("--log-every must be >= 1")

    dt_s = args.dt_ms / 1000.0
    if args.orient_heavy:
        amp_mm = ORIENT_AMP_MM
        amp_rot = ORIENT_AMP_ROT_DEG
        freqs = ORIENT_FREQS_HZ
        preset = "ORIENT"
        if args.max_delta_mm == 8.0:
            args.max_delta_mm = 5.0
        if args.max_delta_deg == 15.0:
            args.max_delta_deg = 18.0
    elif args.fast:
        amp_mm = FAST_AMP_MM
        amp_rot = FAST_AMP_ROT_DEG
        freqs = FAST_FREQS_HZ
        preset = "FAST"
    else:
        amp_mm = SLOW_AMP_MM
        amp_rot = SLOW_AMP_ROT_DEG
        freqs = SLOW_FREQS_HZ
        preset = "SLOW"
    max_mm = np.full(3, args.max_delta_mm)
    max_rot = np.full(3, args.max_delta_deg)
    out_path = (
        output_for_pose(args.pose, None)
        if args.pose and args.output == DEFAULT_OUT
        else args.output
    )

    cfg = CartesianExcitation(
        amp_mm=amp_mm,
        amp_rot_deg=amp_rot,
        freqs_hz=freqs,
        scale=args.scale,
    )
    v_lin_peak, v_ang_peak = estimate_peak_speeds(amp_mm, amp_rot, freqs, args.scale)

    from rm75_control import RobotSession
    from rm75_control.motion.canfd import send_pose_canfd

    print("Connecting...", flush=True)
    with RobotSession(config=CONFIG) as bot:
        ret, state = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            print(f"get state failed: {ret}", file=sys.stderr)
            return 1

        pose0 = np.asarray(state["pose"][:6], dtype=float)
        q0 = np.asarray(state["joint"][:7], dtype=float)

        if args.save_pose:
            save_pose_slot(
                POSES_YAML,
                args.save_pose,
                pose0,
                q0,
                args.pose_label or None,
            )
            print(f"Saved pose slot '{args.save_pose}' → {POSES_YAML}", flush=True)
            print("  pose_base:", [round(float(v), 6) for v in pose0])
            print("  q_deg:", [round(float(v), 2) for v in q0])
            return 0

        poses_data = load_poses_yaml(POSES_YAML)
        slot_rec = get_slot_record(poses_data, args.pose) if args.pose else None
        if args.pose:
            label = slot_rec.get("label") if slot_rec else "(not saved yet)"
            print(f"pose slot '{args.pose}': {label}")
            if slot_rec:
                ref = np.asarray(slot_rec["pose_base"], dtype=float)
                dmm, ddeg = pose_drift_mm_deg(pose0, ref)
                print(f"  drift vs saved: {dmm:.1f} mm, {ddeg:.1f} deg")
                if dmm > args.max_drift_mm or ddeg > 15.0:
                    print(
                        f"  WARNING: arm not at saved slot '{args.pose}' — "
                        f"re-teach or --save-pose {args.pose}",
                        flush=True,
                    )
            print(f"  output → {out_path}")

        print("pose0 base (m, rad):", [round(float(v), 6) for v in pose0])
        print("q0 (deg):", [round(float(v), 2) for v in q0])

        ts = np.linspace(0, args.duration, int(args.duration / dt_s) + 1)
        deltas = []
        for t in ts:
            d = cfg.delta_pose(t)
            d = clamp_delta(d, max_mm=max_mm, max_rot_deg=max_rot)
            deltas.append(d)
        deltas = np.asarray(deltas)

        print(
            "preview |pose-pose0| max (mm):",
            [round(float(m * 1000), 1) for m in np.max(np.abs(deltas[:, 0:3]), axis=0)],
        )
        print(
            "preview |orient-pose0| max (deg):",
            [round(float(m / DEG2RAD), 1) for m in np.max(np.abs(deltas[:, 3:6]), axis=0)],
            f"(cap {args.max_delta_deg}°)",
        )
        print(
            "est. peak TCP speed bound (mm/s):",
            [round(float(v * 1000), 1) for v in v_lin_peak],
        )
        print(
            "est. peak angular speed bound (deg/s):",
            [round(float(v / DEG2RAD), 2) for v in v_ang_peak],
        )
        mode = preset + (f" scale={args.scale}" if args.scale != 1.0 else "")
        print(
            f"settings: {mode} max_mm={args.max_delta_mm} max_deg={args.max_delta_deg}",
            flush=True,
        )
        if args.dry_run:
            cmd = f"python tmp/force_id/force_id_cartesian.py --duration {args.duration}"
            if args.pose:
                cmd += f" --pose {args.pose}"
            if args.orient_heavy:
                cmd += " --orient-heavy"
            print(f"dry-run: no motion. OK → {cmd}", flush=True)
            return 0

        print(
            f"Plan B Cartesian | {preset} scale={args.scale} "
            f"duration={args.duration}s dt={args.dt_ms}ms follow={args.follow} "
            f"| rm_movep_canfd ONLY | Ctrl+C abort",
            flush=True,
        )
        input("Confirm TCP is in FREE SPACE (no contact). Press Enter to start...")

        n_cmd = int(args.duration / dt_s) + 1
        n_log = (n_cmd + args.log_every - 1) // args.log_every
        log_dt_s = dt_s * args.log_every
        t_log = np.zeros(n_log)
        pose_log = np.zeros((n_log, 6))
        q_log = np.zeros((n_log, 7))
        f_log = np.zeros((n_log, 6))
        delta_log = np.zeros((n_log, 6))
        cmd_delta_log = np.zeros((n_cmd, 6))

        print(
            f"cmd {1/dt_s:.0f}Hz | log every {args.log_every} ticks "
            f"({1/log_dt_s:.0f}Hz, ~{n_log} samples)",
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

                ramp = min(1.0, t_cmd / args.warmup_s) if args.warmup_s > 0 else 1.0
                exc = CartesianExcitation(
                    amp_mm=amp_mm,
                    amp_rot_deg=amp_rot,
                    freqs_hz=freqs,
                    scale=args.scale * ramp,
                )
                delta = exc.delta_pose(t_cmd)
                delta = clamp_delta(delta, max_mm=max_mm, max_rot_deg=max_rot)
                cmd_pose = pose0 + delta
                cmd_delta_log[i] = delta

                send_pose_canfd(
                    bot.robot,
                    cmd_pose.tolist(),
                    follow=args.follow,
                    trajectory_mode=0,
                    radio=0,
                )

                if i % args.log_every == 0:
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
                    wall = time.monotonic() - t_start
                    print(f"  t_cmd={t_cmd:.1f}s wall={wall:.1f}s", flush=True)
        except KeyboardInterrupt:
            print("\nCtrl+C — stopping", flush=True)
        finally:
            bot.stop_all()
            try:
                send_pose_canfd(
                    bot.robot,
                    pose0.tolist(),
                    follow=args.follow,
                    trajectory_mode=0,
                    radio=0,
                )
            except Exception:
                pass

        n_saved = log_i
        out = out_path
        if out.exists():
            out.unlink()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            t=t_log[:n_saved],
            pose=pose_log[:n_saved],
            q_deg=q_log[:n_saved],
            force_raw=f_log[:n_saved],
            delta_pose=delta_log[:n_saved],
            pose0=pose0,
            q0_deg=q0,
            amp_mm=amp_mm,
            amp_rot_deg=amp_rot,
            freqs_hz=np.array(freqs, dtype=object),
            scale=args.scale,
            max_delta_mm=args.max_delta_mm,
            max_delta_deg=args.max_delta_deg,
            dt_ms=args.dt_ms,
            log_every=args.log_every,
            log_dt_ms=args.log_every * args.dt_ms,
            cmd_dt_ms=args.dt_ms,
            n_cmd=n_cmd,
            frame="base",
            method="cartesian_fourier",
            preset=preset,
            pose_slot=args.pose or "",
        )
        print(f"Saved {n_saved} log samples ({n_cmd} cmds) → {out}", flush=True)
        print(
            "This script only records .npz — OLS identification is a separate offline step.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
