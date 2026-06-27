#!/usr/bin/env python3
"""
Merge multi-pose ID logs and run staged OLS.

  Stage-1: 10p (m, mc, bias) on all samples
  Stage-2: 16p joint OLS (baseline)
  Stage-3: 10p fixed + I fitted on pose-d burst rows (phase==1) only  ← recommended φ

Run:
  source env.sh
  python tmp/force_id/force_id_fit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _paths import CONFIG_FORCE, LOG_DIR, PHI_JSON  # noqa: E402

DEFAULT_NPZS = [
    LOG_DIR / "force_id_pose_a.npz",
    LOG_DIR / "force_id_pose_b.npz",
    LOG_DIR / "force_id_pose_c.npz",
    LOG_DIR / "force_id_pose_d.npz",
]
OUT_JSON = PHI_JSON

PHI_NAMES = [
    "m", "mc_x", "mc_y", "mc_z",
    "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz",
    "Fx0", "Fy0", "Fz0", "Mx0", "My0", "Mz0",
]

import force_id_comp_demo as fid  # noqa: E402

COLS10 = [0, 1, 2, 3] + list(range(10, 16))
COLS_I = list(range(4, 10))
COLS16 = list(range(16))
I_DIAG = [4, 5, 6]


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    slot = str(d.get("pose_slot", path.stem.split("_")[-1]))
    t = d["t"]
    phase = np.asarray(d["phase"], dtype=np.int8) if "phase" in d.files else np.zeros(len(t))
    return d["pose"], d["force_raw"], t, slot, phase


def sample_row_mask(n_samples: int, sample_mask: np.ndarray) -> np.ndarray:
    """Expand per-time-sample mask (N,) to regressor rows (6N,)."""
    return np.repeat(sample_mask, 6)


def build_merged(
    paths: list[Path],
    cfg: fid.FrameConfig,
    fc: float,
    *,
    use_inertia: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    W_parts, Y_parts, tags = [], [], []
    burst_parts, alpha_parts = [], []
    for p in paths:
        pose, force, t, slot, phase = load_npz(p)
        W, Y = fid.build_dataset(pose, force, t, cfg, fc=fc, use_inertia=use_inertia)
        omega_s, alpha_s, _, _ = fid.kinematics_sensor(pose, t, cfg, fc)
        alpha_norm = np.linalg.norm(alpha_s, axis=1)
        burst = phase == 1
        W_parts.append(W)
        Y_parts.append(Y)
        tags.extend([slot] * len(t))
        burst_parts.append(burst)
        alpha_parts.append(alpha_norm)
    burst_sample = np.concatenate(burst_parts)
    alpha_sample = np.concatenate(alpha_parts)
    return (
        np.vstack(W_parts),
        np.concatenate(Y_parts),
        tags,
        sample_row_mask(len(burst_sample), burst_sample),
        sample_row_mask(len(alpha_sample), alpha_sample),
    )


def fit_cols(W: np.ndarray, Y: np.ndarray, cols: list[int]) -> tuple[np.ndarray, float]:
    phi, *_ = np.linalg.lstsq(W[:, cols], Y, rcond=None)
    rms = float(np.sqrt(np.mean((Y - W[:, cols] @ phi) ** 2)))
    full = np.zeros(16)
    for j, c in enumerate(cols):
        full[c] = phi[j]
    return full, rms


def eval_phi(W: np.ndarray, Y: np.ndarray, phi: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = np.ones(len(Y), dtype=bool)
    Yh = W @ phi
    e = Y[mask] - Yh[mask]
    e6 = e.reshape(-1, 6)
    return {
        "rms_all": float(np.sqrt(np.mean(e**2))),
        "rms_force": float(np.sqrt(np.mean(e6[:, :3] ** 2))),
        "rms_moment": float(np.sqrt(np.mean(e6[:, 3:] ** 2))),
    }


def constrain_inertia(phi: np.ndarray, *, r_max: float = 0.12) -> np.ndarray:
    out = phi.copy()
    m = max(float(out[0]), 0.05)
    cap_diag = m * r_max * r_max
    cap_off = 0.35 * cap_diag
    for i in I_DIAG:
        out[i] = float(np.clip(out[i], 0.0, cap_diag))
    for i in COLS_I:
        if i not in I_DIAG:
            out[i] = float(np.clip(out[i], -cap_off, cap_off))
    return out


def fit_i_on_mask(
    W16: np.ndarray,
    Y: np.ndarray,
    phi10: np.ndarray,
    row_mask: np.ndarray,
    *,
    min_rows: int,
    moments_only: bool = True,
) -> tuple[np.ndarray, dict]:
    use = row_mask
    if moments_only:
        use = row_mask & (np.arange(len(row_mask)) % 6 >= 3)
    n = int(np.sum(use))
    if n < min_rows:
        return phi10.copy(), {"skipped": True, "n_rows": n}
    Yres = Y[use] - W16[use][:, COLS10] @ phi10[COLS10]
    phi_i, *_ = np.linalg.lstsq(W16[use][:, COLS_I], Yres, rcond=None)
    phi = phi10.copy()
    phi[COLS_I] = phi_i
    phi = constrain_inertia(phi)
    stats = eval_phi(W16, Y, phi, row_mask)
    stats["n_rows"] = n
    stats["skipped"] = False
    stats["moments_only"] = moments_only
    return phi, stats


def holdout_by_pose(
    paths: list[Path],
    cfg: fid.FrameConfig,
    fc: float,
    phi: np.ndarray,
) -> dict:
    out = {}
    for p in paths:
        pose, force, t, slot, phase = load_npz(p)
        W, Y = fid.build_dataset(pose, force, t, cfg, fc=fc, use_inertia=True)
        Yhat = W @ phi
        e = Y - Yhat
        e6 = e.reshape(-1, 6)
        entry = {
            "rms_all": float(np.sqrt(np.mean(e**2))),
            "rms_force": float(np.sqrt(np.mean(e6[:, :3] ** 2))),
            "rms_moment": float(np.sqrt(np.mean(e6[:, 3:] ** 2))),
            "per_axis": np.sqrt(np.mean(e6**2, axis=0)).tolist(),
        }
        burst = phase == 1
        if np.any(burst):
            rm = sample_row_mask(len(t), burst)
            eb = e[rm]
            eb6 = eb.reshape(-1, 6)
            entry["burst_rms_force"] = float(np.sqrt(np.mean(eb6[:, :3] ** 2)))
            entry["burst_rms_moment"] = float(np.sqrt(np.mean(eb6[:, 3:] ** 2)))
        out[slot] = entry
    return out


def print_phi(phi: np.ndarray, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"  m     = {phi[0]:+.4f} kg")
    print(f"  mc    = [{phi[1]:+.4f}, {phi[2]:+.4f}, {phi[3]:+.4f}] kg·m")
    print(f"  I     = [{phi[4]:+.5f}, {phi[5]:+.5f}, {phi[6]:+.5f}, "
          f"{phi[7]:+.5f}, {phi[8]:+.5f}, {phi[9]:+.5f}] kg·m²")
    print(f"  biasF = [{phi[10]:+.3f}, {phi[11]:+.3f}, {phi[12]:+.3f}] N")
    print(f"  biasM = [{phi[13]:+.4f}, {phi[14]:+.4f}, {phi[15]:+.4f}] N·m")


def phi_dict(phi: np.ndarray) -> dict[str, float]:
    return {PHI_NAMES[i]: float(phi[i]) for i in range(16)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge pose logs → staged OLS φ")
    parser.add_argument("--npz", type=Path, action="append", default=None)
    parser.add_argument("--config", type=Path, default=CONFIG_FORCE)
    parser.add_argument("--fc", type=float, default=None)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--alpha-percentile", type=float, default=70.0,
                        help="also fit I on burst samples with |alpha| above this percentile")
    args = parser.parse_args()

    paths = args.npz if args.npz else DEFAULT_NPZS
    for p in paths:
        if not p.exists():
            print(f"Missing {p}", file=sys.stderr)
            return 1

    cfg = fid.FrameConfig.from_yaml(args.config)
    fc = args.fc
    if fc is None:
        fc = float(yaml.safe_load(args.config.read_text()).get("filtfilt_cutoff_hz", 2.5))

    print("Merged ID fit (10p → 16p → burst-I)")
    print(f"  config: {args.config.name} → {cfg.label()}")
    print(f"  files ({len(paths)}):")
    for p in paths:
        d = np.load(p, allow_pickle=True)
        n_burst = int(np.sum(d["phase"] == 1)) if "phase" in d.files else 0
        print(f"    {p.name}: N={len(d['t'])} slot={d.get('pose_slot','?')} burst={n_burst}")

    W10, Y, tags, burst_rows, alpha_rows = build_merged(paths, cfg, fc, use_inertia=False)
    W16, Y16, _, burst_rows, alpha_rows = build_merged(paths, cfg, fc, use_inertia=True)
    assert np.allclose(Y, Y16)

    phi10, rms10 = fit_cols(W10, Y, COLS10)
    phi16, rms16 = fit_cols(W16, Y16, COLS16)

    Yres_all = Y16 - W16[:, COLS10] @ phi10[COLS10]
    phi_i_all, *_ = np.linalg.lstsq(W16[:, COLS_I], Yres_all, rcond=None)
    phi_seq = phi10.copy()
    phi_seq[COLS_I] = phi_i_all
    phi_seq = constrain_inertia(phi_seq)
    rms_seq = eval_phi(W16, Y16, phi_seq)["rms_all"]

    phi_burst, burst_fit = fit_i_on_mask(W16, Y16, phi10, burst_rows, min_rows=300)
    rms_burst_all = eval_phi(W16, Y16, phi_burst)["rms_all"]
    rms_burst_on_burst = eval_phi(W16, Y16, phi_burst, burst_rows)
    rms10_on_burst = eval_phi(W16, Y16, phi10, burst_rows)

    # high-|alpha| within burst rows
    alpha_vals = np.zeros(len(Y16) // 6)
    idx = 0
    for p in paths:
        pose, force, t, _, _ = load_npz(p)
        _, alpha_s, _, _ = fid.kinematics_sensor(pose, t, cfg, fc)
        alpha_vals[idx : idx + len(t)] = np.linalg.norm(alpha_s, axis=1)
        idx += len(t)
    burst_samples = burst_rows.reshape(-1, 6)[:, 0]
    if np.any(burst_samples):
        thr = float(np.percentile(alpha_vals[burst_samples], args.alpha_percentile))
        high_a_rows = sample_row_mask(
            len(alpha_vals), burst_samples & (alpha_vals >= thr)
        )
        phi_ha, ha_fit = fit_i_on_mask(W16, Y16, phi10, high_a_rows, min_rows=150)
    else:
        phi_ha, ha_fit = phi_burst.copy(), {"skipped": True, "n_rows": 0}

    if burst_fit.get("skipped"):
        phi_rec = phi10.copy()
        rec_label = "phi_10 (no burst data)"
    else:
        phi_rec = phi_burst
        rec_label = "phi_burst (10p + I@burst moments)"

    n = len(Y) // 6
    split = int((1.0 - args.holdout_frac) * n)
    row_tr = np.repeat(np.arange(n) < split, 6)
    row_te = ~row_tr
    phi16_tr, _ = fit_cols(W16[row_tr], Y16[row_tr], COLS16)
    Yhat_te = W16[row_te] @ phi16_tr
    rms16_te = float(np.sqrt(np.mean((Y16[row_te] - Yhat_te) ** 2)))

    per_pose = holdout_by_pose(paths, cfg, fc, phi_rec)

    print_phi(phi10, "Stage-1: m + mc + bias (10p, all data)")
    print(f"  RMS fit = {rms10:.4f}")
    print_phi(phi16, "Stage-2: full 16p joint OLS (all data)")
    print(f"  RMS fit = {rms16:.4f}")
    print_phi(phi_seq, "Stage-2b: sequential I on all data residual")
    print(f"  RMS fit = {rms_seq:.4f}")
    print_phi(phi_burst, "Stage-3: 10p + I on pose-d burst rows only (recommended I)")
    print(f"  RMS all = {rms_burst_all:.4f}")
    print(
        f"  on burst rows: M RMS {rms10_on_burst['rms_moment']:.4f} (10p) → "
        f"{rms_burst_on_burst['rms_moment']:.4f} (burst-I)  "
        f"F {rms10_on_burst['rms_force']:.3f} → {rms_burst_on_burst['rms_force']:.3f} N"
    )
    if not ha_fit.get("skipped"):
        ha_on = eval_phi(W16, Y16, phi_ha, high_a_rows)
        print_phi(phi_ha, f"Stage-3alt: I on burst & |α|≥p{args.alpha_percentile:.0f}")
        print(f"  on high-α burst: M RMS={ha_on['rms_moment']:.4f}  n={ha_fit['n_rows']}")

    print(f"\nRecommended φ: {rec_label}")
    print_phi(phi_rec, "φ_recommended")
    print(f"\nHold-out last {args.holdout_frac*100:.0f}% timeline (16p): test RMS={rms16_te:.4f}")

    print("\nPer-pose residual (φ_recommended):")
    for slot, st in per_pose.items():
        line = (
            f"  pose {slot}: RMS F={st['rms_force']:.3f} N  M={st['rms_moment']:.4f} N·m"
        )
        if "burst_rms_moment" in st:
            line += (
                f"  | burst M={st['burst_rms_moment']:.4f} F={st['burst_rms_force']:.3f} N"
            )
        print(line)

    result = {
        "config": cfg.label(),
        "files": [str(p) for p in paths],
        "n_samples": int(n),
        "recommended": rec_label,
        "phi_10": phi_dict(phi10),
        "phi_16": phi_dict(phi16),
        "phi_sequential": phi_dict(phi_seq),
        "phi_burst": phi_dict(phi_burst),
        "phi_recommended": phi_dict(phi_rec),
        "rms_10": rms10,
        "rms_16": rms16,
        "rms_16_test": rms16_te,
        "rms_burst_all": rms_burst_all,
        "burst_i_fit": burst_fit,
        "per_pose_residual": per_pose,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\nSaved φ → {OUT_JSON}  (validate uses phi_recommended)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
