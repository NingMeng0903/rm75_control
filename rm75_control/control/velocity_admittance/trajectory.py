"""6D trajectory producers (base frame). Hybrid controller consumes pose_d + vel_ff."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from .rm_algo import end2tool_pose


def tool_frame_delta_pose(
    pose_ref: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    *,
    euler_order: str = "xyz",
) -> np.ndarray:
    """Tool-frame translation without rm_algo RPC (matches frameMode=1 pure translation)."""
    pose = np.asarray(pose_ref, dtype=float).copy()
    r_mat = Rsc.from_euler(euler_order, pose[3:6], degrees=False).as_matrix()
    pose[:3] = pose[:3] + r_mat @ np.array([dx, dy, dz], dtype=float)
    return pose


def tool_offset_pose(robot, ref_pose: list[float], dx: float, dy: float, dz: float) -> list[float]:
    delta = [dx, dy, dz, 0.0, 0.0, 0.0]
    return robot.rm_algo_pose_move(ref_pose, delta, frameMode=1)


def sin_period_for_peak_vel(amplitude_m: float, max_vel_m_s: float) -> float:
    if amplitude_m <= 0.0 or max_vel_m_s <= 0.0:
        return 1.0
    return 2.0 * math.pi * amplitude_m / max_vel_m_s


@dataclass(frozen=True)
class TrajectorySample:
    """One tick of reference motion in base/world frame (6D pose + 6D velocity)."""

    pose_d: np.ndarray
    vel_ff: np.ndarray


class Trajectory6D(Protocol):
    """Any trajectory plugin: set contact origin, then stream 6D references."""

    def set_origin(self, pose0: np.ndarray) -> None: ...

    def sample(self, t_s: float) -> TrajectorySample: ...


@dataclass
class TrajectoryConfig:
    kind: str = "hold"
    amplitude_mm: float = 5.0
    y_peak_to_peak_cm: float | None = None
    period_s: float | None = None
    y_max_vel_cm_s: float = 1.0
    soft_start: bool = False
    ramp_s: float = 2.0
    rz_amplitude_deg: float = 0.0

    @property
    def half_amplitude_m(self) -> float:
        if self.y_peak_to_peak_cm is not None:
            return float(self.y_peak_to_peak_cm) * 0.01 / 2.0
        return self.amplitude_mm / 1000.0


def sin_y_motion(
    t_s: float,
    amplitude_m: float,
    omega: float,
    *,
    soft_start: bool,
    ramp_s: float = 2.0,
) -> tuple[float, float]:
    dy = amplitude_m * math.sin(omega * t_s)
    vy = amplitude_m * omega * math.cos(omega * t_s)
    if soft_start and ramp_s > 0.0 and t_s < ramp_s:
        vy *= math.sin(0.5 * math.pi * t_s / ramp_s)
    return dy, vy


def tool_z_spin_angle_rad(
    t_s: float, *, rz_amp_deg: float, omega: float, soft_start: bool, ramp_s: float,
) -> float:
    if rz_amp_deg <= 0.0:
        return 0.0
    ramp = 1.0
    if soft_start and ramp_s > 0.0 and t_s < ramp_s:
        ramp = math.sin(0.5 * math.pi * t_s / ramp_s)
    return math.radians(rz_amp_deg) * math.sin(omega * t_s) * ramp


def apply_tool_z_spin_pose(pose_ref: np.ndarray, phi_rad: float) -> np.ndarray:
    """Rotate pose_ref orientation by phi about tool +Z (base-frame axis)."""
    from scipy.spatial.transform import Rotation as Rsc

    pose = np.asarray(pose_ref, dtype=float).copy()
    if abs(phi_rad) < 1e-12:
        return pose
    r0 = Rsc.from_euler("xyz", pose_ref[3:6], degrees=False).as_matrix()
    axis = r0[:, 2]
    r_d = Rsc.from_rotvec(axis * phi_rad).as_matrix() @ r0
    pose[3:6] = Rsc.from_matrix(r_d).as_euler("xyz", degrees=False)
    return pose


def tool_z_spin_vel_base(pose_ref: np.ndarray, t_s: float, *, rz_amp_deg: float, omega: float,
                         soft_start: bool, ramp_s: float) -> np.ndarray:
    """Sinusoidal spin about tool +Z → base-frame angular velocity (small-angle)."""
    from scipy.spatial.transform import Rotation as Rsc

    if rz_amp_deg <= 0.0:
        return np.zeros(3)
    ramp = 1.0
    if soft_start and ramp_s > 0.0 and t_s < ramp_s:
        ramp = math.sin(0.5 * math.pi * t_s / ramp_s)
    wz_tool = math.radians(rz_amp_deg) * omega * math.cos(omega * t_s) * ramp
    r_mat = Rsc.from_euler("xyz", pose_ref[3:6], degrees=False).as_matrix()
    return r_mat @ np.array([0.0, 0.0, wz_tool], dtype=float)


class TrajectoryGenerator:
    """
    Built-in trajectory kinds (demos). Each sample() returns full 6D base-frame
    (pose_d, vel_ff). Drop tool-Z from vel_ff externally if desired; force hybrid
    fills tool-Z via force_axes.
    """

    def __init__(self, cfg: TrajectoryConfig, pose0: np.ndarray, robot) -> None:
        self.cfg = cfg
        self.pose0 = np.asarray(pose0, dtype=float)
        self.robot = robot
        amp_m = cfg.half_amplitude_m
        if cfg.period_s is None:
            period = sin_period_for_peak_vel(amp_m, cfg.y_max_vel_cm_s / 100.0)
        else:
            period = float(cfg.period_s)
        self.omega = 2.0 * math.pi / period if period > 0 else 0.0
        self.amplitude_m = amp_m

    def set_origin(self, pose0: np.ndarray) -> None:
        self.pose0 = np.asarray(pose0, dtype=float).copy()

    def sample(self, t_s: float) -> TrajectorySample:
        kind = self.cfg.kind
        if kind == "hold":
            return TrajectorySample(self.pose0.copy(), np.zeros(6))

        if kind in ("sin_base_y", "sin_base_y_tool_rz"):
            return self._sin_base_y(t_s, spin=(kind == "sin_base_y_tool_rz"))

        if kind == "sin_tool_y":
            return self._sin_tool_y(t_s)

        raise ValueError(f"Unknown trajectory type: {kind}")

    def _sin_base_y(self, t_s: float, *, spin: bool) -> TrajectorySample:
        dy, vy = sin_y_motion(
            t_s, self.amplitude_m, self.omega,
            soft_start=self.cfg.soft_start, ramp_s=self.cfg.ramp_s,
        )
        pose = self.pose0.copy()
        pose[1] += dy
        if spin:
            phi = tool_z_spin_angle_rad(
                t_s,
                rz_amp_deg=self.cfg.rz_amplitude_deg,
                omega=self.omega,
                soft_start=self.cfg.soft_start,
                ramp_s=self.cfg.ramp_s,
            )
            pose = apply_tool_z_spin_pose(pose, phi)
        vel = np.zeros(6, dtype=float)
        vel[1] = vy
        if spin:
            vel[3:6] = tool_z_spin_vel_base(
                self.pose0, t_s,
                rz_amp_deg=self.cfg.rz_amplitude_deg,
                omega=self.omega,
                soft_start=self.cfg.soft_start,
                ramp_s=self.cfg.ramp_s,
            )
        return TrajectorySample(pose, vel)

    def _sin_tool_y(self, t_s: float) -> TrajectorySample:
        dy, vy = sin_y_motion(
            t_s, self.amplitude_m, self.omega,
            soft_start=self.cfg.soft_start, ramp_s=self.cfg.ramp_s,
        )
        pose = np.asarray(
            tool_offset_pose(self.robot, list(self.pose0), 0.0, dy, 0.0), dtype=float
        )
        r_mat = Rsc.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
        vel = np.zeros(6, dtype=float)
        vel[:3] = r_mat @ np.array([0.0, vy, 0.0], dtype=float)
        return TrajectorySample(pose, vel)

    @classmethod
    def from_dict(cls, raw: dict, pose0: np.ndarray, robot) -> TrajectoryGenerator:
        t = raw.get("trajectory", {})
        ps = t.get("period_s")
        y_pp_cm = t.get("y_peak_to_peak_cm")
        return cls(
            TrajectoryConfig(
                kind=str(t.get("type", "hold")),
                amplitude_mm=float(t.get("amplitude_mm", 5.0)),
                y_peak_to_peak_cm=float(y_pp_cm) if y_pp_cm is not None else None,
                period_s=float(ps) if ps is not None else None,
                y_max_vel_cm_s=float(t.get("y_max_vel_cm_s", 1.0)),
                soft_start=bool(t.get("soft_start", False)),
                ramp_s=float(t.get("ramp_s", 2.0)),
                rz_amplitude_deg=float(t.get("rz_amplitude_deg", 0.0)),
            ),
            pose0,
            robot,
        )
