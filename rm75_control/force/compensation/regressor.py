"""Newton-Euler regressor: kinematics + W matrix for force compensation ID."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rsc

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
    origin_in_link7_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
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
            origin_in_link7_m=tuple(
                float(x) for x in data.get("sensor_origin_in_link7_m", [0.0, 0.0, 0.0])
            ),
            gravity_base=tuple(float(x) for x in data["gravity_base"]),
        )


def com_from_phi(phi: np.ndarray, cfg: FrameConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Center of mass position (m) from identified phi.

    mc is the first mass moment in the **sensor** frame: mc = m * r_com_sensor.
    link7 frame: R_link7_sensor @ r_com_sensor + sensor origin in link7,
    with R_link7_sensor = R_off from sensor_offset_euler (same as regressor).
    """
    m = float(phi[0])
    if m <= 1e-9:
        z = np.zeros(3, dtype=float)
        return z, z
    r_sensor = np.asarray(phi[1:4], dtype=float) / m
    if cfg.offset_rad != (0.0, 0.0, 0.0):
        r_off = Rsc.from_euler("xyz", cfg.offset_rad, degrees=False).as_matrix()
        r_link7 = r_off @ r_sensor
    else:
        r_link7 = r_sensor.copy()
    r_link7 = r_link7 + np.asarray(cfg.origin_in_link7_m, dtype=float)
    return r_sensor, r_link7


def com_dict_mm(r_m: np.ndarray) -> dict[str, float]:
    r_mm = np.asarray(r_m, dtype=float) * 1000.0
    return {
        "Cx": float(r_mm[0]),
        "Cy": float(r_mm[1]),
        "Cz": float(r_mm[2]),
    }


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


def _angular_velocity_sensor(R: np.ndarray, R_prev: np.ndarray, dt: float) -> np.ndarray:
    """Sensor-frame angular velocity from two rotations via dR/dt @ R^T (causal)."""
    dR = (R - R_prev) / max(dt, 1e-6)
    sk = dR @ R.T
    w = np.array(
        [sk[2, 1] - sk[1, 2], sk[0, 2] - sk[2, 0], sk[1, 0] - sk[0, 1]],
        dtype=float,
    ) / 2.0
    return R.T @ w


def regressor_row_causal(
    poses: np.ndarray,
    times: np.ndarray,
    cfg: FrameConfig,
    *,
    use_inertia: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single-sample (current = last row) regressor for ONLINE causal force
    compensation. Returns (W_row 6x16, g_s 3).

    Gravity g_s = R^T g_base is exact from the current orientation (no
    derivative, no lag). Angular rate/accel and linear accel use causal backward
    finite differences over the supplied short history. These dynamic terms are
    tiny at cm/s scan speeds (m*a ~ 0.01 N for a 1 kg tool), so crude estimates
    plus a downstream causal LPF are sufficient. Mirrors regressor_row(); the
    offline build_dataset() path is left untouched for identification.

    poses: (K,6) base-frame pose history, oldest→newest (newest = current).
    times: (K,) monotonic timestamps matching poses.
    """
    poses = np.asarray(poses, dtype=float)
    times = np.asarray(times, dtype=float)
    n = len(times)
    cur = n - 1

    R = R_base_sensor(poses[cur], cfg)
    g_s = R.T @ np.asarray(cfg.gravity_base, dtype=float)

    omega_s = np.zeros(3)
    alpha_s = np.zeros(3)
    a_s = np.zeros(3)

    # Quasi-static compensation (use_inertia=False): keep ONLY the gravity term
    # (g_s = R^T g_base, exact and smooth from orientation alone) + the constant
    # bias. The dynamic terms (linear accel, ω, α) are obtained here by finite
    # differencing the measured pose; at scan speeds (~cm/s) their TRUE force
    # contribution is <0.05 N, but raw double-differencing of the quantised
    # encoder injects ~2 N of noise (a_s std ≈ 1.7-2.8 m/s² × tool mass). That
    # noise is what made the tool-Z force feel jittery. So we drop them unless
    # the caller explicitly wants the inertial model (use_inertia=True), in which
    # case they must be filtered upstream like the offline build_dataset path.
    if use_inertia and n >= 2:
        dt0 = max(times[cur] - times[cur - 1], 1e-6)
        R_prev = R_base_sensor(poses[cur - 1], cfg)
        omega_s = _angular_velocity_sensor(R, R_prev, dt0)
        if n >= 3:
            dt1 = max(times[cur - 1] - times[cur - 2], 1e-6)
            dt_c = 0.5 * (dt0 + dt1)
            v_now = (poses[cur, :3] - poses[cur - 1, :3]) / dt0
            v_prev = (poses[cur - 1, :3] - poses[cur - 2, :3]) / dt1
            a_s = R.T @ ((v_now - v_prev) / dt_c)
            R_prev2 = R_base_sensor(poses[cur - 2], cfg)
            omega_prev = _angular_velocity_sensor(R_prev, R_prev2, dt1)
            alpha_s = (omega_s - omega_prev) / dt_c

    W_row = regressor_row(a_s, g_s, omega_s, alpha_s, use_inertia=use_inertia)
    return W_row, g_s


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
