#!/usr/bin/env python3
"""
Empirical demo: find force_sign + sensor frame convention for raw force_data ID.

Uses live robot (static sample) + optional npz log.
Run: source env.sh && python tmp/force_id/force_id_sign_demo.py
     python tmp/force_id/force_id_sign_demo.py --npz tmp/force_id/logs/force_id_cartesian.npz
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _paths import CONFIG_ROBOT, DEFAULT_NPZ  # noqa: E402

CONFIG = CONFIG_ROBOT
DEFAULT_NPZ = DEFAULT_NPZ
G = 9.80665

# RealMan doc: sensor Z up, Y opposite aviation plug, RH; at zero pose tool == sensor.
SENSOR_EULER_ORDERS = ("xyz", "zyx", "ZYX")


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def gravity_in_sensor(g_base: np.ndarray, R_base_sensor: np.ndarray) -> np.ndarray:
    return R_base_sensor.T @ g_base


def R_from_pose_euler(rxryrz: np.ndarray, order: str) -> np.ndarray:
    return Rsc.from_euler(order, rxryrz, degrees=False).as_matrix()


def R_with_fixed_offset(
    pose6: np.ndarray, order: str, offset_euler: tuple[float, float, float] | None
) -> np.ndarray:
    R = R_from_pose_euler(pose6[3:6], order)
    if offset_euler is None:
        return R
    R_off = Rsc.from_euler("xyz", offset_euler, degrees=False).as_matrix()
    return R @ R_off


def apply_force_sign(raw6: np.ndarray, signs: tuple[int, ...]) -> np.ndarray:
    return raw6 * np.array(signs, dtype=float)


def static_mass_estimate(
    F: np.ndarray, g_s: np.ndarray, *, model: str
) -> float:
    """Crude m from one static sample along dominant gravity axis."""
    gs = g_s.copy()
    if model == "F_eq_ma_minus_g":
        # F = m * (0 - g_s) => m = F · g_s / |g_s|^2  when a=0, using vector proj
        denom = float(np.dot(gs, gs))
        if denom < 1e-9:
            return float("nan")
        return float(np.dot(F, gs) / denom)
    if model == "F_eq_neg_m_times_g":
        # F = -m * g_s  (common wrench convention)
        denom = float(np.dot(gs, gs))
        if denom < 1e-9:
            return float("nan")
        return float(-np.dot(F, gs) / denom)
    raise ValueError(model)


def score_config_on_log(
    pose: np.ndarray,
    force: np.ndarray,
    *,
    order: str,
    offset: tuple[float, float, float] | None,
    f_sign: tuple[int, ...],
    g_base: np.ndarray,
    model: str,
) -> dict:
    """Score how consistent static F ≈ m * g model is across all samples (a≈0 band)."""
    m_est = []
    for i in range(len(pose)):
        R = R_with_fixed_offset(pose[i], order, offset)
        gs = gravity_in_sensor(g_base, R)
        F = apply_force_sign(force[i], f_sign)[:3]
        m = static_mass_estimate(F, gs, model=model)
        if math.isfinite(m) and abs(m) < 50:
            m_est.append(m)
    if len(m_est) < 10:
        return {"n": len(m_est), "m_mean": float("nan"), "m_std": float("nan"), "score": 1e9}
    arr = np.asarray(m_est)
    # want m > 0, stable, ~0.5-5 kg for their tool
    m_mean = float(arr.mean())
    m_std = float(arr.std())
    penalty = 0.0
    if m_mean <= 0:
        penalty += 10.0 + abs(m_mean)
    if m_std > 0.5:
        penalty += m_std
    if m_mean > 0 and (m_mean < 0.05 or m_mean > 15):
        penalty += 1.0
    score = m_std + penalty
    return {"n": len(arr), "m_mean": m_mean, "m_std": m_std, "score": score}


def search_best_on_log(pose: np.ndarray, force: np.ndarray) -> list[dict]:
    g_base = np.array([0.0, 0.0, -G])
    f_signs = [
        (1, 1, 1, 1, 1, 1),
        (-1, -1, -1, 1, 1, 1),
        (-1, -1, -1, -1, -1, -1),
        (1, 1, -1, 1, 1, 1),
    ]
    offsets: list[tuple[float, float, float] | None] = [
        None,
        (math.pi, 0.0, 0.0),
        (0.0, math.pi, 0.0),
        (0.0, 0.0, math.pi),
        (math.pi, 0.0, math.pi),
    ]
    models = ("F_eq_ma_minus_g", "F_eq_neg_m_times_g")
    results = []
    for order in SENSOR_EULER_ORDERS:
        for offset in offsets:
            for f_sign in f_signs:
                for model in models:
                    s = score_config_on_log(
                        pose,
                        force,
                        order=order,
                        offset=offset,
                        f_sign=f_sign,
                        g_base=g_base,
                        model=model,
                    )
                    results.append(
                        {
                            "order": order,
                            "offset_rad": offset,
                            "force_sign": f_sign,
                            "model": model,
                            **s,
                        }
                    )
    results.sort(key=lambda r: r["score"])
    return results


def live_static_sample(n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    from rm75_control import RobotSession

    poses, forces = [], []
    with RobotSession(config=CONFIG) as bot:
        print("Live: hold arm still; sampling force + pose...", flush=True)
        import time

        for _ in range(n):
            ret_s, st = bot.robot.rm_get_current_arm_state()
            ret_f, fd = bot.robot.rm_get_force_data()
            if ret_s == 0 and ret_f == 0:
                poses.append(st["pose"][:6])
                forces.append(fd["force_data"][:6])
            time.sleep(0.05)
    return np.asarray(poses), np.asarray(forces)


def print_top(results: list[dict], k: int = 8) -> None:
    print(f"\n{'rank':>4}  {'score':>7}  {'m kg':>7}  {'m_std':>6}  model  order  offset  force_sign")
    for i, r in enumerate(results[:k]):
        off = r["offset_rad"]
        off_s = "none" if off is None else f"({off[0]:+.2f},{off[1]:+.2f},{off[2]:+.2f})"
        fs = ",".join(str(x) for x in r["force_sign"])
        print(
            f"{i+1:4d}  {r['score']:7.3f}  {r['m_mean']:+7.3f}  {r['m_std']:6.3f}  "
            f"{r['model'][-20:]:20s} {r['order']:4s} {off_s:22s} [{fs}]"
        )


def recommend(best: dict) -> None:
    fs = best["force_sign"]
    print("\n=== RECOMMENDATION (empirical) ===")
    print(f"  force_sign: {list(fs)}")
    print(f"  euler order: {best['order']}")
    if best["offset_rad"] is None:
        print("  R_base_sensor: use TCP/base pose orientation directly (zero offset)")
    else:
        o = best["offset_rad"]
        print(
            f"  R_base_sensor: R_tcp * RxRyRz({o[0]:.4f}, {o[1]:.4f}, {o[2]:.4f}) rad"
        )
    if best["model"] == "F_eq_ma_minus_g":
        print("  regressor: F = m*(a - g_s) + ...  (keep current academic form)")
    else:
        print("  regressor: use F = -m*g_s at static OR equivalently force_sign on F")
    print(f"  estimated mass: {best['m_mean']:.3f} +/- {best['m_std']:.3f} kg")
    print("\nRealMan docs: sensor Z up, Y opp. connector; tool==sensor ONLY at zero pose.")
    print("If offset != none → TCP pose != sensor frame (expected after TCP change).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Find force sign / sensor frame convention")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--live", action="store_true", help="also sample live robot")
    parser.add_argument("--live-only", action="store_true")
    args = parser.parse_args()

    if args.live or args.live_only:
        try:
            pose_l, f_l = live_static_sample()
            print(f"\nLive static mean F (N): {pose_l.shape[0]} samples")
            print("  F mean:", np.round(f_l.mean(0), 3))
            print("  pose mean euler deg:", np.round(pose_l.mean(0)[3:6] * 180 / math.pi, 2))
            res_l = search_best_on_log(pose_l, f_l)
            print("\n--- LIVE static search (top 8) ---")
            print_top(res_l, 8)
            recommend(res_l[0])
        except Exception as e:
            print(f"Live failed: {e}", file=sys.stderr)
            if args.live_only:
                return 1

    if args.live_only:
        return 0

    if not args.npz.exists():
        print(f"No npz at {args.npz}", file=sys.stderr)
        return 1

    d = np.load(args.npz, allow_pickle=True)
    pose = d["pose"]
    force = d["force_raw"]
    print(f"\n=== NPZ {args.npz.name} ({len(pose)} samples) ===")
    print("F mean:", np.round(force.mean(0), 3))
    print("F std :", np.round(force.std(0), 4))

    res = search_best_on_log(pose, force)
    print("\n--- NPZ dynamic log search (top 10) ---")
    print_top(res, 10)
    recommend(res[0])

    # verify: full 10-param OLS with best config on filtered subset
    best = res[0]
    print("\n--- Quick 10p OLS with best config (first 200 samples, filtfilt) ---")
    from scipy.signal import butter, filtfilt

    n = min(200, len(pose))
    t = d["t"][:n]
    pose = pose[:n]
    force = force[:n]
    fs = 1.0 / np.mean(np.diff(t))

    def lpf(x, fc=2.0):
        b, a = butter(2, min(fc / (0.5 * fs), 0.99), btype="low")
        return filtfilt(b, a, x, axis=0)

    euler = pose[:, 3:6].copy()
    for j in range(3):
        euler[:, j] = np.unwrap(euler[:, j])
    ff = lpf(apply_force_sign(force, best["force_sign"]))
    a_b = np.gradient(lpf(np.gradient(lpf(pose[:, :3]), t, axis=0), 2.0), t, axis=0)
    omega_s = np.zeros((n, 3))
    alpha_s = np.zeros((n, 3))
    for i in range(n):
        R = R_with_fixed_offset(
            np.concatenate([pose[i, :3], euler[i]]), best["order"], best["offset_rad"]
        )
        if i == 0:
            R1 = R_with_fixed_offset(
                np.concatenate([pose[1, :3], euler[1]]), best["order"], best["offset_rad"]
            )
            dR = (R1 - R) / max(t[1] - t[0], 1e-6)
        elif i == n - 1:
            R0 = R_with_fixed_offset(
                np.concatenate([pose[i - 1, :3], euler[i - 1]]),
                best["order"],
                best["offset_rad"],
            )
            dR = (R - R0) / max(t[-1] - t[-2], 1e-6)
        else:
            Rp = R_with_fixed_offset(
                np.concatenate([pose[i + 1, :3], euler[i + 1]]),
                best["order"],
                best["offset_rad"],
            )
            Rm = R_with_fixed_offset(
                np.concatenate([pose[i - 1, :3], euler[i - 1]]),
                best["order"],
                best["offset_rad"],
            )
            dR = (Rp - Rm) / max(t[i + 1] - t[i - 1], 1e-6)
        sk = dR @ R.T
        w = np.array([sk[2, 1] - sk[1, 2], sk[0, 2] - sk[2, 0], sk[1, 0] - sk[0, 1]]) / 2
        omega_s[i] = R.T @ w
    omega_s = lpf(omega_s, 2.0)
    alpha_s = np.gradient(lpf(omega_s, 1.5), t, axis=0)

    rows, Y = [], []
    g_base = np.array([0.0, 0.0, -G])
    for i in range(n):
        R = R_with_fixed_offset(
            np.concatenate([pose[i, :3], euler[i]]), best["order"], best["offset_rad"]
        )
        gs = gravity_in_sensor(g_base, R)
        a_s = R.T @ a_b[i]
        aeq = a_s - gs
        w, al = omega_s[i], alpha_s[i]
        sw, sa = skew(w), skew(al)
        W = np.zeros((6, 10))
        W[0:3, 0] = aeq
        W[0:3, 1:4] = sa + sw @ sw
        W[3:6, 1:4] = -skew(aeq)
        W[:, 4:10] = np.eye(6)
        rows.append(W)
        Y.append(ff[i])
    Wmat = np.vstack(rows)
    Yvec = np.concatenate(Y)
    phi, *_ = np.linalg.lstsq(Wmat, Yvec, rcond=None)
    rms = float(np.sqrt(np.mean((Yvec - Wmat @ phi) ** 2)))
    print(f"  m={phi[0]:+.3f} kg  mc={np.round(phi[1:4],4)}  RMS={rms:.4f}")
    print(f"  bias F={np.round(phi[4:7],2)}  M={np.round(phi[7:10],3)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
