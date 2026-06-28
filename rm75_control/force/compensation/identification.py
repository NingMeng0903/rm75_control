"""
Merge multi-pose ID logs → staged OLS φ.

Called by force_calibrate.py; not intended as a top-level entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from . import regressor as fid
from .id_config import load_config
from .paths import CONFIG_ID

PHI_NAMES = [
    "m", "mc_x", "mc_y", "mc_z",
    "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz",
    "Fx0", "Fy0", "Fz0", "Mx0", "My0", "Mz0",
]

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


def constrain_inertia(phi: np.ndarray, *, r_max: float) -> np.ndarray:
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
    r_max: float,
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
    phi = constrain_inertia(phi, r_max=r_max)
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


def com_report(phi: np.ndarray, cfg: fid.FrameConfig) -> dict:
    r_sensor, r_link7 = fid.com_from_phi(phi, cfg)
    return {
        "sensor_mm": fid.com_dict_mm(r_sensor),
        "link7_mm": fid.com_dict_mm(r_link7),
    }


def print_summary(
    phi: np.ndarray,
    cfg: fid.FrameConfig,
    *,
    rms_all: float,
    per_pose: dict,
    out_json: Path,
) -> None:
    com = com_report(phi, cfg)
    c_s = com["sensor_mm"]
    c_l = com["link7_mm"]
    print("\nIdentify done")
    print(f"  m     = {phi[0]:+.4f} kg")
    print(f"  mc    = [{phi[1]:+.4f}, {phi[2]:+.4f}, {phi[3]:+.4f}] kg·m")
    print(
        f"  CoM sensor  Cx,Cy,Cz = [{c_s['Cx']:+.2f}, {c_s['Cy']:+.2f}, {c_s['Cz']:+.2f}] mm"
    )
    print(
        f"  CoM link7   Cx,Cy,Cz = [{c_l['Cx']:+.2f}, {c_l['Cy']:+.2f}, {c_l['Cz']:+.2f}] mm"
    )
    print(f"  biasF = [{phi[10]:+.3f}, {phi[11]:+.3f}, {phi[12]:+.3f}] N")
    print(f"  biasM = [{phi[13]:+.4f}, {phi[14]:+.4f}, {phi[15]:+.4f}] N·m")
    print(f"  RMS   = {rms_all:.4f}")
    for slot, st in per_pose.items():
        print(f"  {slot}: F={st['rms_force']:.3f} N  M={st['rms_moment']:.4f} N·m")
    print(f"  {out_json}")


def phi_dict(phi: np.ndarray) -> dict[str, float]:
    return {PHI_NAMES[i]: float(phi[i]) for i in range(16)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge pose logs → staged OLS φ")
    parser.add_argument("--id-config", type=Path, default=CONFIG_ID)
    parser.add_argument("--npz", type=Path, action="append", default=None)
    parser.add_argument("--fc", type=float, default=None)
    args = parser.parse_args(argv)

    id_cfg = load_config(args.id_config)
    fcfg = id_cfg.fit
    paths = args.npz if args.npz else fcfg.npz_paths
    for p in paths:
        if not p.exists():
            print(f"Missing {p}", file=sys.stderr)
            return 1

    cfg = fid.FrameConfig.from_yaml(fcfg.force_sensor)
    fc = args.fc
    if fc is None:
        fc = float(yaml.safe_load(fcfg.force_sensor.read_text()).get("filtfilt_cutoff_hz", 2.5))

    r_max = fcfg.inertia_r_max_m
    holdout_frac = fcfg.holdout_frac
    alpha_pct = fcfg.alpha_percentile
    out_json = fcfg.phi_output

    W10, Y, tags, burst_rows, alpha_rows = build_merged(paths, cfg, fc, use_inertia=False)
    W16, Y16, _, burst_rows, alpha_rows = build_merged(paths, cfg, fc, use_inertia=True)
    assert np.allclose(Y, Y16)

    phi10, rms10 = fit_cols(W10, Y, COLS10)
    phi16, rms16 = fit_cols(W16, Y16, COLS16)

    Yres_all = Y16 - W16[:, COLS10] @ phi10[COLS10]
    phi_i_all, *_ = np.linalg.lstsq(W16[:, COLS_I], Yres_all, rcond=None)
    phi_seq = constrain_inertia(phi10.copy(), r_max=r_max)
    phi_seq[COLS_I] = phi_i_all
    phi_seq = constrain_inertia(phi_seq, r_max=r_max)
    rms_seq = eval_phi(W16, Y16, phi_seq)["rms_all"]

    phi_burst, burst_fit = fit_i_on_mask(
        W16, Y16, phi10, burst_rows, min_rows=fcfg.min_burst_rows, r_max=r_max,
    )
    rms_burst_all = eval_phi(W16, Y16, phi_burst)["rms_all"]
    rms_burst_on_burst = eval_phi(W16, Y16, phi_burst, burst_rows)
    rms10_on_burst = eval_phi(W16, Y16, phi10, burst_rows)

    alpha_vals = np.zeros(len(Y16) // 6)
    idx = 0
    for p in paths:
        pose, force, t, _, _ = load_npz(p)
        _, alpha_s, _, _ = fid.kinematics_sensor(pose, t, cfg, fc)
        alpha_vals[idx : idx + len(t)] = np.linalg.norm(alpha_s, axis=1)
        idx += len(t)
    burst_samples = burst_rows.reshape(-1, 6)[:, 0]
    if np.any(burst_samples):
        thr = float(np.percentile(alpha_vals[burst_samples], alpha_pct))
        high_a_rows = sample_row_mask(
            len(alpha_vals), burst_samples & (alpha_vals >= thr)
        )
        phi_ha, ha_fit = fit_i_on_mask(
            W16, Y16, phi10, high_a_rows, min_rows=fcfg.min_high_alpha_rows, r_max=r_max,
        )
    else:
        phi_ha, ha_fit = phi_burst.copy(), {"skipped": True, "n_rows": 0}

    if burst_fit.get("skipped"):
        phi_rec = phi10.copy()
        rec_label = "phi_10 (no burst data)"
    else:
        phi_rec = phi_burst
        rec_label = "phi_burst (10p + I@burst moments)"

    n = len(Y) // 6
    split = int((1.0 - holdout_frac) * n)
    row_tr = np.repeat(np.arange(n) < split, 6)
    row_te = ~row_tr
    phi16_tr, _ = fit_cols(W16[row_tr], Y16[row_tr], COLS16)
    Yhat_te = W16[row_te] @ phi16_tr
    rms16_te = float(np.sqrt(np.mean((Y16[row_te] - Yhat_te) ** 2)))

    per_pose = holdout_by_pose(paths, cfg, fc, phi_rec)
    rms_rec = eval_phi(W16, Y16, phi_rec)["rms_all"]

    result = {
        "config": cfg.label(),
        "id_config": str(args.id_config),
        "files": [str(p) for p in paths],
        "n_samples": int(n),
        "recommended": rec_label,
        "phi_10": phi_dict(phi10),
        "phi_16": phi_dict(phi16),
        "phi_sequential": phi_dict(phi_seq),
        "phi_burst": phi_dict(phi_burst),
        "phi_recommended": phi_dict(phi_rec),
        "com_recommended": com_report(phi_rec, cfg),
        "rms_10": rms10,
        "rms_16": rms16,
        "rms_16_test": rms16_te,
        "rms_burst_all": rms_burst_all,
        "burst_i_fit": burst_fit,
        "per_pose_residual": per_pose,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    print_summary(phi_rec, cfg, rms_all=rms_rec, per_pose=per_pose, out_json=out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
