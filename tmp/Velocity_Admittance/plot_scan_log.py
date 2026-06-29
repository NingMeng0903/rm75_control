#!/usr/bin/env python3
"""Plot admittance scan log — diagnose controller vs execution vs contact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.velocity_admittance.scan_log import (
    load_scan_log,
    scan_tracking_world_mm,
)


def _euler_delta_deg(pose: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = pose - ref
    d[:, 3:6] = (d[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(d[:, 3:6])


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot admittance scan log npz")
    parser.add_argument("npz", type=Path)
    parser.add_argument("--save", type=Path, default=None, help="PNG output (use if no display)")
    parser.add_argument("--tmax", type=float, default=None, help="limit time axis (s)")
    args = parser.parse_args()

    d = load_scan_log(args.npz)
    t = d["t_s"]
    if args.tmax is not None:
        m = t <= args.tmax
        d = {k: v[m] if hasattr(v, "__len__") and len(v) == len(t) else v for k, v in d.items()}
        t = d["t_s"]

    v = d["v_cmd"]
    vf = d["vel_ff"]
    pose = d["pose_act"]
    pose_d = d["pose_d"]
    f = d["f_ext"]
    phase = d["phase"]
    scan = phase >= 2
    si = int(np.where(scan)[0][0]) if np.any(scan) else 0
    t_scan_on = float(t[si]) if np.any(scan) else 0.0
    dt_ms = np.concatenate([[np.nan], np.diff(t) * 1000.0])

    deuler = _euler_delta_deg(pose, pose[si])
    tr = scan_tracking_world_mm(pose_d, pose, scan_mask=scan if np.any(scan) else np.ones(len(t), dtype=bool))
    d_cmd = tr["d_cmd_mm"].copy()
    d_act = tr["d_act_mm"].copy()
    s_cmd = tr["s_cmd_mm"].copy()
    s_act = tr["s_act_mm"].copy()
    track_err = tr["scan_track_err_mm"].copy()
    # Pre-scan: pose_d / ref frame differ (hold logs zeros, approach logs post-init pose0).
    # Mask so position panels only show scan tracking (cmd=act at scan ON).
    pre = ~scan
    d_cmd[pre] = np.nan
    d_act[pre] = np.nan
    s_cmd[pre] = np.nan
    s_act[pre] = np.nan
    track_err[pre] = np.nan

    vy_ff_tool = np.zeros(len(t))
    for i in range(len(t)):
        r = Rsc.from_euler("xyz", pose[i, 3:6], degrees=False).as_matrix()
        vy_ff_tool[i] = (r.T @ vf[i, :3])[1]

    fig, axes = plt.subplots(7, 1, figsize=(13, 14), sharex=True)
    fig.suptitle(
        f"{args.npz.name}  scan position vs scan0 (pre-scan masked); vel: full run"
    )

    ax = axes[0]
    ax.plot(t, v[:, 1], "C0", lw=0.8, label="v_cmd tool-Y")
    ax.plot(t, vy_ff_tool, "C1", lw=0.8, alpha=0.7, label="R.T@vel_ff tool-Y")
    ax.axvline(t[si], color="gray", ls="--", lw=0.8, label="scan ON")
    ax.set_ylabel("tool Y vel (m/s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, v[:, 2], "C2", lw=0.8, label="v_cmd tool-Z")
    ax.set_ylabel("tool Z vel")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, f[:, 2], "C3", lw=0.8, label="Fz ext")
    ax.axhline(3.0, color="r", ls="--", lw=0.8, label="Fz des 3N")
    ax.axhline(1.0, color="orange", ls=":", lw=0.8, label="contact loss ~1N")
    ax.set_ylabel("Fz (N)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(t, deuler[:, 1], label="Δpitch deg")
    ax.plot(t, deuler[:, 2], label="Δyaw deg", alpha=0.8)
    ax.set_ylabel("Δeuler vs scan0")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    ax.plot(t, s_cmd, "C1", lw=0.9, ls="--", label="cmd tool-Y→world")
    ax.plot(t, s_act, "C0", lw=0.9, label="act tool-Y→world")
    ax.plot(t, track_err, "C3", lw=0.6, alpha=0.7, label="track err")
    ax.axvline(t_scan_on, color="gray", ls="--", lw=0.8)
    if np.any(scan):
        ax.scatter([t_scan_on], [0.0], c="k", s=12, zorder=5, label="scan ON cmd≈act")
    ax.set_ylabel("scan axis mm")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[5]
    ax.plot(t, d_cmd[:, 1], "C3", lw=0.8, ls="--", label="cmd ΔY world")
    ax.plot(t, d_act[:, 1], "C2", lw=0.8, label="act ΔY world")
    ax.set_ylabel("world ΔY mm")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, d_cmd[:, 0], "C1", lw=0.8, ls=":", alpha=0.9, label="cmd ΔX")
    ax2.plot(t, d_act[:, 0], "C0", lw=0.6, alpha=0.8, label="act ΔX")
    ax2.set_ylabel("world ΔX mm (right)")
    cx = float(np.max(np.abs(d_cmd[scan, 0]))) if np.any(scan) else 0.0
    ax.text(
        0.02, 0.04,
        f"sin_tool_y: 1 DOF tool-Y only; cmd ΔX≈±{cx:.1f}mm from tilt\n"
        "act ΔX blow-up = contact slip + pitch drift (not missing X cmd)",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )

    ax = axes[6]
    ax.plot(t, dt_ms, "k", lw=0.6, alpha=0.7)
    ax.axhline(10, color="g", ls="--", lw=0.8)
    ax.axhline(15, color="r", ls=":", lw=0.8)
    ax.set_ylabel("loop dt ms")
    ax.set_xlabel("t (s)")
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.fill_between(t, ax.get_ylim()[0], ax.get_ylim()[1], where=scan, alpha=0.04, color="blue")

    fig.tight_layout()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=130)
        print(f"saved → {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
