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
