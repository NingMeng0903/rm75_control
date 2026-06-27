#!/usr/bin/env python3
"""
Grid-search force_sign / euler / sensor offset; fit phi; verify compensation.

Run:
  source env.sh
  python tmp/force_id/force_id_comp_demo.py
  python tmp/force_id/force_id_comp_demo.py --live
  python tmp/force_id/force_id_comp_demo.py --apply-yaml
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rsc

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _paths import CONFIG_FORCE, CONFIG_ROBOT, DEFAULT_NPZ, REPO  # noqa: E402

CONFIG_ROBOT = CONFIG_ROBOT
CONFIG_FORCE = CONFIG_FORCE
DEFAULT_NPZ = DEFAULT_NPZ

PHI_NAMES = [
    "m",
    "mc_x",
    "mc_y",
    "mc_z",
    "Ixx",
    "Iyy",
    "Izz",
    "Ixy",
    "Ixz",
    "Iyz",
    "Fx0",
    "Fy0",
    "Fz0",
    "Mx0",
    "My0",
    "Mz0",
]


@dataclass(frozen=True)
class FrameConfig:
    force_sign: tuple[int, ...]
    euler_order: str
    offset_rad: tuple[float, float, float]
    gravity_base: tuple[float, float, float] = (0.0, 0.0, -9.80665)

    def label(self) -> str:
        fs = ",".join(str(int(s)) for s in self.force_sign)
        off = "0" if self.offset_rad == (0.0, 0.0, 0.0) else ",".join(
            f"{v:.2f}" for v in self.offset_rad
        )
        return f"sign=[{fs}] order={self.euler_order} off=[{off}]"

    @classmethod
    def from_yaml(cls, path: Path) -> FrameConfig:
        data = yaml.safe_load(path.read_text())
        return cls(
            force_sign=tuple(int(x) for x in data["force_sign"]),
            euler_order=str(data["euler_order"]),
            offset_rad=tuple(float(x) for x in data["sensor_offset_euler_xyz_rad"]),
            gravity_base=tuple(float(x) for x in data["gravity_base"]),
        )


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def inertia_op(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array(
        [[wx, 0, 0, wy, wz, 0], [0, wy, 0, wx, 0, wz], [0, 0, wz, 0, wx, wy]]
    )


def R_base_sensor(pose6: np.ndarray, cfg: FrameConfig) -> np.ndarray:
    R = Rsc.from_euler(cfg.euler_order, pose6[3:6], degrees=False).as_matrix()
    if cfg.offset_rad != (0.0, 0.0, 0.0):
        R_off = Rsc.from_euler("xyz", cfg.offset_rad, degrees=False).as_matrix()
        R = R @ R_off
    return R


def apply_sign(raw6: np.ndarray, sign: tuple[int, ...]) -> np.ndarray:
    return raw6 * np.array(sign, dtype=float)


def filtfilt_cols(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    if len(x) < 30:
        return x
    b, a = butter(2, min(fc / (0.5 * fs), 0.99), btype="low")
    return filtfilt(b, a, x, axis=0)


def kinematics_sensor(
    pose: np.ndarray, t: np.ndarray, cfg: FrameConfig, fc: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fs = 1.0 / np.mean(np.diff(t))
    euler = pose[:, 3:6].copy()
    for j in range(3):
        euler[:, j] = np.unwrap(euler[:, j])

    p_f = filtfilt_cols(pose[:, :3], fs, fc)
    v_b = np.gradient(p_f, t, axis=0)
    a_b = np.gradient(filtfilt_cols(v_b, fs, fc * 0.8), t, axis=0)

    n = len(t)
    omega_s = np.zeros((n, 3))
    a_s = np.zeros((n, 3))
    g_s = np.zeros((n, 3))
    g_base = np.asarray(cfg.gravity_base, dtype=float)

    for i in range(n):
        p6 = np.concatenate([pose[i, :3], euler[i]])
        R = R_base_sensor(p6, cfg)
        if i == 0:
            R1 = R_base_sensor(np.concatenate([pose[1, :3], euler[1]]), cfg)
            dR = (R1 - R) / max(t[1] - t[0], 1e-6)
        elif i == n - 1:
            R0 = R_base_sensor(np.concatenate([pose[i - 1, :3], euler[i - 1]]), cfg)
            dR = (R - R0) / max(t[-1] - t[-2], 1e-6)
        else:
            Rp = R_base_sensor(np.concatenate([pose[i + 1, :3], euler[i + 1]]), cfg)
            Rm = R_base_sensor(np.concatenate([pose[i - 1, :3], euler[i - 1]]), cfg)
            dR = (Rp - Rm) / max(t[i + 1] - t[i - 1], 1e-6)
        sk = dR @ R.T
        w = np.array([sk[2, 1] - sk[1, 2], sk[0, 2] - sk[2, 0], sk[1, 0] - sk[0, 1]]) / 2
        omega_s[i] = R.T @ w
        a_s[i] = R.T @ a_b[i]
        g_s[i] = R.T @ g_base

    omega_s = filtfilt_cols(omega_s, fs, fc * 0.8)
    alpha_s = np.gradient(filtfilt_cols(omega_s, fs, fc * 0.6), t, axis=0)
    a_s = filtfilt_cols(a_s, fs, fc * 0.8)
    return omega_s, alpha_s, a_s, g_s


def regressor_row(
    a_s: np.ndarray,
    g_s: np.ndarray,
    omega_s: np.ndarray,
    alpha_s: np.ndarray,
    *,
    use_inertia: bool,
) -> np.ndarray:
    aeq = a_s - g_s
    w, al = omega_s, alpha_s
    sw, sa = skew(w), skew(al)
    W = np.zeros((6, 16))
    W[0:3, 0] = aeq
    W[0:3, 1:4] = sa + sw @ sw
    W[3:6, 1:4] = -skew(aeq)
    if use_inertia:
        W[3:6, 4:10] = inertia_op(al) + sw @ inertia_op(w)
    W[:, 10:16] = np.eye(6)
    return W


def build_dataset(
    pose: np.ndarray,
    force_raw: np.ndarray,
    t: np.ndarray,
    cfg: FrameConfig,
    *,
    fc: float,
    use_inertia: bool,
) -> tuple[np.ndarray, np.ndarray]:
    omega_s, alpha_s, a_s, g_s = kinematics_sensor(pose, t, cfg, fc)
    fs = 1.0 / np.mean(np.diff(t))
    f = filtfilt_cols(apply_sign(force_raw, cfg.force_sign), fs, fc)
    rows, Y = [], []
    for i in range(len(t)):
        rows.append(
            regressor_row(a_s[i], g_s[i], omega_s[i], alpha_s[i], use_inertia=use_inertia)
        )
        Y.append(f[i])
    return np.vstack(rows), np.concatenate(Y)


def fit_eval(
    W: np.ndarray,
    Y: np.ndarray,
    *,
    train_end: int,
    cols: list[int],
) -> dict:
    Wtr, Ytr = W[: 6 * train_end, :][:, cols], Y[: 6 * train_end]
    Wte, Yte = W[6 * train_end :, :][:, cols], Y[6 * train_end :]
    phi, *_ = np.linalg.lstsq(Wtr, Ytr, rcond=None)
    pred_tr = Wtr @ phi
    pred_te = Wte @ phi
    ext_tr = Ytr - pred_tr
    ext_te = Yte - pred_te
    return {
        "phi_full": _embed_phi(cols, phi),
        "m": float(phi[cols.index(0)]),
        "rms_train": float(np.sqrt(np.mean(ext_tr**2))),
        "rms_test": float(np.sqrt(np.mean(ext_te**2))),
        "rms_test_force": float(np.sqrt(np.mean(ext_te.reshape(-1, 6)[:, :3] ** 2))),
        "rms_test_moment": float(np.sqrt(np.mean(ext_te.reshape(-1, 6)[:, 3:] ** 2))),
        "ext_test": ext_te.reshape(-1, 6),
    }


def _embed_phi(cols: list[int], phi: np.ndarray) -> np.ndarray:
    out = np.zeros(16)
    for j, c in enumerate(cols):
        out[c] = phi[j]
    return out


def grid_configs() -> list[FrameConfig]:
    signs = [
        (1, 1, 1, 1, 1, 1),
        (-1, -1, -1, 1, 1, 1),
        (-1, -1, -1, -1, -1, -1),
        (1, 1, -1, 1, 1, 1),
    ]
    orders = ("xyz", "zyx", "ZYX")
    offsets = [
        (0.0, 0.0, 0.0),
        (0.0, math.pi, 0.0),
        (math.pi, 0.0, 0.0),
        (0.0, 0.0, math.pi),
        (math.pi, 0.0, math.pi),
    ]
    cfgs = []
    for sign, order, off in itertools.product(signs, orders, offsets):
        cfgs.append(FrameConfig(sign, order, off))
    return cfgs


def score_result(r: dict) -> float:
    m = r["m"]
    penalty = 0.0
    if m <= 0:
        penalty += 20.0 + abs(m)
    if m > 8:
        penalty += m - 8
    return r["rms_test"] + penalty + 0.1 * abs(m - 1.05)


def run_grid(
    pose: np.ndarray,
    force: np.ndarray,
    t: np.ndarray,
    *,
    fc: float,
    use_inertia: bool,
    top_k: int = 12,
) -> list[tuple[FrameConfig, dict]]:
    split = int(0.8 * len(t))
    cols10 = [0, 1, 2, 3] + list(range(10, 16))
    cols16 = list(range(16))
    results = []
    for cfg in grid_configs():
        W, Y = build_dataset(pose, force, t, cfg, fc=fc, use_inertia=use_inertia)
        cols = cols16 if use_inertia else cols10
        r = fit_eval(W, Y, train_end=split, cols=cols)
        results.append((cfg, r))
    results.sort(key=lambda x: score_result(x[1]))
    return results[:top_k]


def live_static_mass(cfg: FrameConfig, n: int = 25) -> float:
    from rm75_control import RobotSession

    F_sum = np.zeros(3)
    gs_sum = np.zeros(3)
    with RobotSession(config=CONFIG_ROBOT) as bot:
        import time

        for _ in range(n):
            ret_s, st = bot.robot.rm_get_current_arm_state()
            ret_f, fd = bot.robot.rm_get_force_data()
            if ret_s != 0 or ret_f != 0:
                continue
            p6 = np.asarray(st["pose"][:6], dtype=float)
            F = apply_sign(np.asarray(fd["force_data"][:6]), cfg.force_sign)[:3]
            gs = R_base_sensor(p6, cfg).T @ np.asarray(cfg.gravity_base, dtype=float)
            F_sum += F
            gs_sum += gs
            time.sleep(0.04)
    Fm = F_sum / n
    gsm = gs_sum / n
    denom = float(np.dot(gsm, gsm))
    if denom < 1e-9:
        return float("nan")
    # Static academic model: F = m * (a - g_s), a=0 => F = -m * g_s
    return float(-np.dot(Fm, gsm) / denom)


def print_top(results: list[tuple[FrameConfig, dict]], title: str) -> None:
    print(f"\n=== {title} ===")
    print(
        f"{'#':>2} {'score':>7} {'m kg':>7} {'trRMS':>7} {'teRMS':>7} "
        f"{'teF':>7} {'teM':>7}  config"
    )
    for i, (cfg, r) in enumerate(results):
        sc = score_result(r)
        print(
            f"{i+1:2d} {sc:7.3f} {r['m']:+7.3f} {r['rms_train']:7.4f} "
            f"{r['rms_test']:7.4f} {r['rms_test_force']:7.4f} "
            f"{r['rms_test_moment']:7.4f}  {cfg.label()}"
        )


def verify_yaml(
    pose: np.ndarray,
    force: np.ndarray,
    t: np.ndarray,
    cfg: FrameConfig,
    *,
    fc: float,
    use_inertia: bool,
) -> dict:
    split = int(0.8 * len(t))
    cols10 = [0, 1, 2, 3] + list(range(10, 16))
    cols16 = list(range(16))
    W, Y = build_dataset(pose, force, t, cfg, fc=fc, use_inertia=use_inertia)
    cols = cols16 if use_inertia else cols10
    r = fit_eval(W, Y, train_end=split, cols=cols)
    return r


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--apply-yaml", action="store_true", help="verify configs/force_sensor.yaml")
    parser.add_argument("--fc", type=float, default=2.5)
    parser.add_argument("--full-16", action="store_true", help="include inertia in grid")
    args = parser.parse_args()

    if not args.npz.exists():
        print(f"Missing {args.npz}", file=sys.stderr)
        return 1

    d = np.load(args.npz, allow_pickle=True)
    pose = d["pose"]
    force = d["force_raw"]
    t = d["t"]
    print(f"NPZ: {args.npz}  N={len(t)}  F_mean={np.round(force.mean(0),2)}")

    top10 = run_grid(pose, force, t, fc=args.fc, use_inertia=False, top_k=15)
    print_top(top10, "GRID 10-param (m,mc,bias) — lower teRMS + m>0 wins")

    best_cfg, best_r = top10[0]
    print("\n--- BEST combo ---")
    print(best_cfg.label())
    print(f"  m={best_r['m']:+.3f} kg")
    print(f"  test compensation RMS: all={best_r['rms_test']:.4f} N  "
          f"force={best_r['rms_test_force']:.4f}  moment={best_r['rms_test_moment']:.4f}")
    phi = best_r["phi_full"]
    shown = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15]
    print(
        "  phi:",
        ", ".join(f"{PHI_NAMES[i]}={phi[i]:+.4f}" for i in shown),
    )

    if args.apply_yaml or CONFIG_FORCE.exists():
        yaml_cfg = FrameConfig.from_yaml(CONFIG_FORCE)
        yr = verify_yaml(pose, force, t, yaml_cfg, fc=args.fc, use_inertia=args.full_16)
        print(f"\n=== configs/force_sensor.yaml verification ===")
        print(yaml_cfg.label())
        print(f"  m={yr['m']:+.3f} kg")
        print(f"  test RMS: all={yr['rms_test']:.4f}  force={yr['rms_test_force']:.4f}  "
              f"moment={yr['rms_test_moment']:.4f}")
        if args.live:
            m_live = live_static_mass(yaml_cfg)
            print(f"  live static m (F·g/|g|²): {m_live:+.3f} kg")

    if args.live:
        print("\n=== LIVE static mass check (top 3 combos) ===")
        for cfg, r in top10[:3]:
            m_live = live_static_mass(cfg)
            print(f"  {cfg.label()}  grid_m={r['m']:+.3f}  live_m={m_live:+.3f}")

    # write chosen yaml if best matches simple recommendation
    winner = top10[0][0]
    out = {
        "force_sign": list(winner.force_sign),
        "euler_order": winner.euler_order,
        "sensor_offset_euler_xyz_rad": list(winner.offset_rad),
        "gravity_base": list(winner.gravity_base),
        "filtfilt_cutoff_hz": args.fc,
        "identify_mass_bias_only": not args.full_16,
        "verified_test_rms": float(top10[0][1]["rms_test"]),
        "verified_mass_kg": float(top10[0][1]["m"]),
    }
    verified_path = REPO / "configs" / "force_sensor_verified.yaml"
    verified_path.write_text("# Auto-written by tmp/force_id/force_id_comp_demo.py\n" + yaml.dump(out, sort_keys=False))
    print(f"\nWrote grid winner → {verified_path}")

    if winner.force_sign == (-1, -1, -1, 1, 1, 1) and winner.offset_rad == (0.0, 0.0, 0.0):
        print("CONFIRMED: force_sign [-1,-1,-1,1,1,1], offset none, euler", winner.euler_order)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
