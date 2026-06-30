"""High-rate scan log: target trajectory vs actual encoder feedback."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rm75_control.control.hybrid_motion.paths import VA_DATA_DIR


LOG_DIR = VA_DATA_DIR / "logs"
_GROW = 512


class ScanLogRecorder:
    """Pre-allocated ring growth — minimal per-tick overhead."""

    def __init__(self, *, capacity: int = 4096) -> None:
        self._cap = capacity
        self._n = 0
        self.t_s = np.zeros(capacity, dtype=float)
        self.t_scan = np.full(capacity, np.nan, dtype=float)
        self.phase = np.zeros(capacity, dtype=np.int8)
        self.pose_act = np.zeros((capacity, 6), dtype=float)
        self.q_deg = np.zeros((capacity, 7), dtype=float)
        self.pose_d = np.zeros((capacity, 6), dtype=float)
        self.vel_ff = np.zeros((capacity, 6), dtype=float)
        self.v_cmd = np.zeros((capacity, 6), dtype=float)
        self.f_ext = np.zeros((capacity, 6), dtype=float)
        self.f_des_z = np.zeros(capacity, dtype=float)

    def __len__(self) -> int:
        return self._n

    def _grow(self) -> None:
        new_cap = self._cap + _GROW
        for name in (
            "t_s", "t_scan", "phase", "f_des_z",
        ):
            old = getattr(self, name)
            ext = np.zeros(new_cap, dtype=old.dtype)
            ext[: self._cap] = old
            if name == "t_scan":
                ext[self._cap :] = np.nan
            setattr(self, name, ext)
        for name in ("pose_act", "pose_d", "vel_ff", "v_cmd", "f_ext"):
            old = getattr(self, name)
            ext = np.zeros((new_cap, 6), dtype=float)
            ext[: self._cap] = old
            setattr(self, name, ext)
        old = self.q_deg
        ext = np.zeros((new_cap, 7), dtype=float)
        ext[: self._cap] = old
        self.q_deg = ext
        self._cap = new_cap

    def append_row(
        self,
        *,
        t_s: float,
        t_scan: float,
        phase: int,
        pose_act: np.ndarray,
        q_deg: np.ndarray,
        pose_d: np.ndarray,
        vel_ff: np.ndarray,
        v_cmd: np.ndarray,
        f_ext: np.ndarray,
        f_des_z: float,
    ) -> None:
        if self._n >= self._cap:
            self._grow()
        i = self._n
        self.t_s[i] = t_s
        self.t_scan[i] = t_scan
        self.phase[i] = phase
        self.pose_act[i] = pose_act
        self.q_deg[i] = q_deg
        self.pose_d[i] = pose_d
        self.vel_ff[i] = vel_ff
        self.v_cmd[i] = v_cmd
        self.f_ext[i] = f_ext
        self.f_des_z[i] = f_des_z
        self._n += 1

    def save(self, path: Path, *, meta: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = self._n
        if n == 0:
            raise ValueError("ScanLogRecorder: no samples")
        pack = {
            "t_s": self.t_s[:n].copy(),
            "t_scan": self.t_scan[:n].copy(),
            "phase": self.phase[:n].copy(),
            "pose_act": self.pose_act[:n].copy(),
            "q_deg": self.q_deg[:n].copy(),
            "pose_d": self.pose_d[:n].copy(),
            "vel_ff": self.vel_ff[:n].copy(),
            "v_cmd": self.v_cmd[:n].copy(),
            "f_ext": self.f_ext[:n].copy(),
            "f_des_z": self.f_des_z[:n].copy(),
        }
        if meta:
            pack["meta_json"] = np.array([str(meta)])
        np.savez_compressed(path, **pack)
        return path


def default_log_path(prefix: str = "admittance") -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"{prefix}_{stamp}.npz"


def load_scan_log(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def scan_origin_r(pose_act: np.ndarray, scan_mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Scan ON index, pose0, and R0 (tool orientation in world at scan start)."""
    from scipy.spatial.transform import Rotation as Rsc

    idx = np.where(scan_mask)[0]
    if len(idx) == 0:
        return 0, pose_act[0].copy(), np.eye(3)
    si = int(idx[0])
    pose0 = pose_act[si].copy()
    r0 = Rsc.from_euler("xyz", pose0[3:6], degrees=False).as_matrix()
    return si, pose0, r0


def world_delta_mm(pose: np.ndarray, pose0: np.ndarray) -> np.ndarray:
    """TCP linear displacement vs scan origin, in world frame (mm)."""
    return (pose[:, :3] - pose0[:3]) * 1000.0


def tool_y_world_scalar_mm(delta_world_mm: np.ndarray, r0: np.ndarray) -> np.ndarray:
    """Tool-Y scan progress: world displacement projected onto tool +Y at scan ON."""
    e_scan = r0[:, 1]
    return delta_world_mm @ e_scan


def scan_tracking_world_mm(
    pose_d: np.ndarray,
    pose_act: np.ndarray,
    *,
    scan_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Decoupled scan tracking in world frame.

    Compare commanded vs actual TCP world deltas from scan origin. TCP-Z is force-controlled:
    use scan-axis scalar (tool-Y in world @ scan0) and world-XY cross-track, not world-Z.
    """
    si, pose0, r0 = scan_origin_r(pose_act, scan_mask)
    d_cmd = world_delta_mm(pose_d, pose0)
    d_act = world_delta_mm(pose_act, pose0)
    s_cmd = tool_y_world_scalar_mm(d_cmd, r0)
    s_act = tool_y_world_scalar_mm(d_act, r0)
    dxy_err = (d_cmd[:, :2] - d_act[:, :2])
    return {
        "d_cmd_mm": d_cmd,
        "d_act_mm": d_act,
        "s_cmd_mm": s_cmd,
        "s_act_mm": s_act,
        "scan_track_err_mm": s_cmd - s_act,
        "world_xy_err_mm": np.linalg.norm(dxy_err, axis=1),
        "scan_idx": si,
        "r0": r0,
    }


def print_jerk_summary(path: Path, *, dt_s: float) -> None:
    """Print v_cmd / pose tracking diagnostics to separate planner vs execution."""
    data = load_scan_log(path)
    t = data["t_s"]
    v = data["v_cmd"]
    pose_act = data["pose_act"]
    pose_d = data["pose_d"]
    phase = data["phase"]
    n = len(t)
    if n < 3:
        print(f"  log summary: too few samples ({n})", flush=True)
        return

    dv = np.diff(v, axis=0) / np.maximum(np.diff(t)[:, None], 1e-6)
    jerk_proxy = np.diff(dv, axis=0) / np.maximum(np.diff(t)[1:, None], 1e-6)
    scan_mask = phase >= 2
    mask = scan_mask if np.any(scan_mask) else np.ones(n, dtype=bool)

    idx = np.where(mask)[0]
    idx = idx[(idx > 0) & (idx < n - 2)]
    if len(idx) == 0:
        idx = np.arange(1, min(n - 2, n))

    finite_dv = np.isfinite(dv).all(axis=1)
    finite_jk = np.isfinite(jerk_proxy).all(axis=1) if len(jerk_proxy) else np.array([True])
    idx_dv = idx[np.isin(idx - 1, np.where(finite_dv)[0])]
    idx_jk = idx[np.isin(idx - 2, np.where(finite_jk)[0])]

    dv_n = np.linalg.norm(dv[idx_dv - 1], axis=1) if len(idx_dv) else np.array([0.0])
    jk_n = (
        np.linalg.norm(jerk_proxy[idx_jk - 2], axis=1)
        if len(idx_jk) and len(jerk_proxy)
        else np.array([0.0])
    )

    tr = scan_tracking_world_mm(pose_d, pose_act, scan_mask=mask)
    idx_scan = np.where(mask)[0]
    s_cmd = tr["s_cmd_mm"][idx_scan] if len(idx_scan) else np.array([0.0])
    s_act = tr["s_act_mm"][idx_scan] if len(idx_scan) else np.array([0.0])
    scan_track = np.abs(tr["scan_track_err_mm"][idx_scan]) if len(idx_scan) else np.array([0.0])
    xy_cross = tr["world_xy_err_mm"][idx_scan] if len(idx_scan) else np.array([0.0])

    loop_dt = np.diff(t[scan_mask]) * 1000.0 if np.any(scan_mask) else np.diff(t) * 1000.0
    loop_dt = loop_dt[np.isfinite(loop_dt)]

    print("\n=== scan log summary ===", flush=True)
    print(f"  file: {path}", flush=True)
    print(f"  samples={n}  scan_samples={int(np.sum(scan_mask))}  dt_nom={dt_s*1000:.1f}ms", flush=True)
    if len(loop_dt):
        print(
            f"  loop dt ms: median={float(np.median(loop_dt)):.2f}  "
            f"max={float(np.max(loop_dt)):.2f}  "
            f">15ms={int(np.sum(loop_dt > 15))}/{len(loop_dt)}",
            flush=True,
        )
    print(
        f"  |dv_cmd| max={float(np.nanmax(dv_n)) if len(dv_n) else 0:.4f} m/s²  "
        f"p95={float(np.nanpercentile(dv_n, 95)) if len(dv_n) else 0:.4f}  "
        f"|jerk_proxy| max={float(np.nanmax(jk_n)) if len(jk_n) else 0:.2f}",
        flush=True,
    )
    if len(idx_scan):
        print(
            f"  tool-Y world (scan axis @ scan0): track err max={float(np.max(scan_track)):.2f} mm  "
            f"p95={float(np.percentile(scan_track, 95)):.2f} mm  "
            f"(world-Z decoupled — force axis)",
            flush=True,
        )
        print(
            f"  tool-Y world stroke  cmd [{float(s_cmd.min()):+.1f}, {float(s_cmd.max()):+.1f}] mm  "
            f"act [{float(s_act.min()):+.1f}, {float(s_act.max()):+.1f}] mm  "
            f"world-XY |Δcmd−Δact| p95={float(np.percentile(xy_cross, 95)):.2f} mm",
            flush=True,
        )
        print(
            "  (large world track err + smooth v_cmd tool-Y → execution/contact slip)",
            flush=True,
        )
    for axis, name in enumerate(["vx", "vy", "vz", "wx", "wy", "wz"]):
        col = v[scan_mask, axis] if np.any(scan_mask) else v[:, axis]
        col = col[np.isfinite(col)]
        if len(col) < 2:
            continue
        dcol = np.diff(col) / dt_s
        dcol = dcol[np.isfinite(dcol)]
        if len(dcol) == 0:
            continue
        spikes = int(np.sum(np.abs(dcol) > 3.0 * float(np.std(dcol) + 1e-9)))
        print(
            f"  v_cmd {name}: std={float(np.std(col)):.5f}  "
            f"|dv/dt| max={float(np.max(np.abs(dcol))):.4f}  spikes(>3σ)={spikes}",
            flush=True,
        )
