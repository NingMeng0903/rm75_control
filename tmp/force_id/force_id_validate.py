#!/usr/bin/env python3
"""Validate φ at current pose: excite ~30s, compare raw vs compensated RMS."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import force_id_cartesian as fic
import force_id_comp_demo as fid
from _paths import CONFIG_FORCE, PHI_JSON

CONFIG = fic.CONFIG

def compensate_block(
    pose: np.ndarray,
    force_raw: np.ndarray,
    t: np.ndarray,
    phi: np.ndarray,
    cfg: fid.FrameConfig,
    fc: float,
) -> tuple[np.ndarray, np.ndarray]:
    W, Y = fid.build_dataset(pose, force_raw, t, cfg, fc=fc, use_inertia=True)
    Yhat = W @ phi
    Fext = (Y - Yhat).reshape(-1, 6)
    return Y.reshape(-1, 6), Fext


def run_validate(duration: float, phi: np.ndarray, cfg: fid.FrameConfig, fc: float) -> dict:
    from rm75_control import RobotSession
    from rm75_control.motion.canfd import send_pose_canfd

    dt_s = 0.01
    amp_mm, amp_rot, freqs = fic.ORIENT_AMP_MM, fic.ORIENT_AMP_ROT_DEG, fic.ORIENT_FREQS_HZ
    max_mm = np.full(3, 5.0)
    max_rot = np.full(3, 35.0)

    poses, forces, ts = [], [], []
    with RobotSession(config=CONFIG) as bot:
        ret, st = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"get state failed: {ret}")
        pose0 = np.asarray(st["pose"][:6], dtype=float)
        q0 = np.asarray(st["joint"][:7], dtype=float)
        print("Validate pose0:", [round(v, 4) for v in pose0])
        print("q0:", [round(v, 1) for v in q0])

        n = int(duration / dt_s) + 1
        t_start = time.monotonic()
        next_tick = t_start
        for i in range(n):
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += dt_s
            t_cmd = i * dt_s
            ramp = min(1.0, t_cmd / 5.0)
            exc = fic.CartesianExcitation(amp_mm, amp_rot, freqs, ramp)
            delta = fic.clamp_delta(
                exc.delta_pose(t_cmd), max_mm=max_mm, max_rot_deg=max_rot
            )
            send_pose_canfd(
                bot.robot, (pose0 + delta).tolist(), follow=False, trajectory_mode=0, radio=0
            )
            if i % 10 == 0:
                ret_s, st = bot.robot.rm_get_current_arm_state()
                ret_f, fd = bot.robot.rm_get_force_data()
                if ret_s == 0 and ret_f == 0:
                    poses.append(st["pose"][:6])
                    forces.append(fd["force_data"][:6])
                    ts.append(t_cmd)
        bot.stop_all()
        try:
            send_pose_canfd(bot.robot, pose0.tolist(), follow=False, trajectory_mode=0, radio=0)
        except Exception:
            pass

    pose = np.asarray(poses)
    force = np.asarray(forces)
    t = np.asarray(ts)
    Fraw = force * np.array(cfg.force_sign)
    _, Fext = compensate_block(pose, force, t, phi, cfg, fc)
    return {
        "n": len(t),
        "raw_std": Fraw.std(0).tolist(),
        "raw_mean": Fraw.mean(0).tolist(),
        "ext_std": Fext.std(0).tolist(),
        "ext_mean": Fext.mean(0).tolist(),
        "rms_raw_F": float(np.sqrt(np.mean(Fraw[:, :3] ** 2))),
        "rms_ext_F": float(np.sqrt(np.mean(Fext[:, :3] ** 2))),
        "rms_raw_M": float(np.sqrt(np.mean(Fraw[:, 3:] ** 2))),
        "rms_ext_M": float(np.sqrt(np.mean(Fext[:, 3:] ** 2))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--phi", type=Path, default=PHI_JSON)
    args = parser.parse_args()

    data = json.loads(args.phi.read_text())
    cfg = fid.FrameConfig.from_yaml(CONFIG_FORCE)
    fc = 2.5
    src = "phi_recommended" if "phi_recommended" in data else "phi_16"
    phi = np.array([data[src][k] for k in fid.PHI_NAMES])

    print(f"Using φ ({src}) from {args.phi}  m={phi[0]:.3f} kg")
    r = run_validate(args.duration, phi, cfg, fc)
    print(f"\n=== Validation {r['n']} samples @ ~10Hz ===")
    print(f"  |F| RMS  raw={r['rms_raw_F']:.3f} N  →  ext={r['rms_ext_F']:.3f} N")
    print(f"  |M| RMS  raw={r['rms_raw_M']:.4f}  →  ext={r['rms_ext_M']:.4f} N·m")
    print(f"  F ext mean: {[round(x,3) for x in r['ext_mean'][:3]]}")
    print(f"  F ext std : {[round(x,3) for x in r['ext_std'][:3]]}")
    ok = r["rms_ext_F"] < 0.35 and abs(r["ext_mean"][2]) < 0.5
    print(f"\n  {'PASS' if ok else 'MARGINAL'} (target ext_F < 0.35 N, Fz_ext mean near 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
