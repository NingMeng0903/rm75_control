"""Reference pose + analytic feedforward velocity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .rm_algo import end2tool_pose, pose_to_rm_pose


def tool_offset_pose(robot, ref_pose: list[float], dx: float, dy: float, dz: float) -> list[float]:
    delta = [dx, dy, dz, 0.0, 0.0, 0.0]
    return robot.rm_algo_pose_move(ref_pose, delta, frameMode=1)


def sin_period_for_peak_vel(amplitude_m: float, max_vel_m_s: float) -> float:
    if amplitude_m <= 0.0 or max_vel_m_s <= 0.0:
        return 1.0
    return 2.0 * math.pi * amplitude_m / max_vel_m_s


@dataclass
class TrajectoryConfig:
    kind: str = "sin_tool_y"
    amplitude_mm: float = 5.0
    period_s: float | None = None
    y_max_vel_cm_s: float = 1.0
    soft_start: bool = False
    ramp_s: float = 2.0


def sin_y_motion(
    t_s: float,
    amplitude_m: float,
    omega: float,
    *,
    soft_start: bool,
    ramp_s: float = 2.0,
) -> tuple[float, float]:
    """Tool-frame Y offset (m) and velocity (m/s). soft_start ramps vy from 0 (±sin ref)."""
    dy = amplitude_m * math.sin(omega * t_s)
    vy = amplitude_m * omega * math.cos(omega * t_s)
    if soft_start and ramp_s > 0.0 and t_s < ramp_s:
        vy *= math.sin(0.5 * math.pi * t_s / ramp_s)
    return dy, vy


class TrajectoryGenerator:
    def __init__(self, cfg: TrajectoryConfig, pose0: np.ndarray, robot) -> None:
        self.cfg = cfg
        self.pose0 = np.asarray(pose0, dtype=float)
        self.robot = robot
        amp_m = cfg.amplitude_mm / 1000.0
        if cfg.period_s is None:
            period = sin_period_for_peak_vel(amp_m, cfg.y_max_vel_cm_s / 100.0)
        else:
            period = float(cfg.period_s)
        self.omega = 2.0 * math.pi / period if period > 0 else 0.0
        self.amplitude_m = amp_m
        self._y0_tool = float(end2tool_pose(robot, list(pose0))[1])

    def set_origin(self, pose0: np.ndarray) -> None:
        """Reset sin centre to contact pose (e.g. pose D at latch)."""
        self.pose0 = np.asarray(pose0, dtype=float).copy()
        self._y0_tool = float(end2tool_pose(self.robot, list(self.pose0))[1])

    @staticmethod
    def world_scan_reference(
        pose_d: np.ndarray,
        vel_ff: np.ndarray,
        pose_anchor: np.ndarray,
        motion_axes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Base/world sin on motion_axes; lock all other pose dof to anchor."""
        pose_d = np.asarray(pose_d, dtype=float).copy()
        vel_ff = np.asarray(vel_ff, dtype=float).copy()
        anchor = np.asarray(pose_anchor, dtype=float)
        for i in range(3):
            if motion_axes[i] < 0.5:
                pose_d[i] = anchor[i]
                vel_ff[i] = 0.0
            else:
                vel_ff[i] = float(vel_ff[i])
        pose_d[3:6] = anchor[3:6]
        vel_ff[3:6] = 0.0
        return pose_d, vel_ff
    def blend_tool_pose(
        robot,
        pose_d: np.ndarray,
        pose_anchor: np.ndarray,
        motion_axes: np.ndarray,
    ) -> np.ndarray:
        """Tool-frame mask: only motion_axes[i]==1 follows pose_d; orientation locked to anchor."""
        t_des = np.asarray(end2tool_pose(robot, list(pose_d)), dtype=float)
        t_anc = np.asarray(end2tool_pose(robot, list(pose_anchor)), dtype=float)
        delta = np.zeros(3, dtype=float)
        for i in range(3):
            if motion_axes[i] > 0.5:
                delta[i] = t_des[i] - t_anc[i]
        return np.asarray(
            tool_offset_pose(
                robot, list(pose_anchor), float(delta[0]), float(delta[1]), float(delta[2])
            ),
            dtype=float,
        )

    @staticmethod
    def project_tool_motion_ff(
        robot,
        pose_ref: np.ndarray,
        vel_ff: np.ndarray,
        motion_axes: np.ndarray,
    ) -> np.ndarray:
        """Feedforward only on allowed tool linear axes; zero angular feedforward."""
        from scipy.spatial.transform import Rotation as Rsc

        vel_ff = np.asarray(vel_ff, dtype=float).copy()
        r_mat = Rsc.from_euler("xyz", pose_ref[3:6], degrees=False).as_matrix()
        v_tool = r_mat.T @ vel_ff[:3]
        for i in range(3):
            if motion_axes[i] < 0.5:
                v_tool[i] = 0.0
        out = np.zeros(6, dtype=float)
        out[:3] = r_mat @ v_tool
        return out

    def sample(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        kind = self.cfg.kind
        if kind == "hold":
            return self.pose0.copy(), np.zeros(6)

        if kind == "sin_base_y":
            dy, vy = sin_y_motion(
                t_s, self.amplitude_m, self.omega,
                soft_start=self.cfg.soft_start, ramp_s=self.cfg.ramp_s,
            )
            pose = self.pose0.copy()
            pose[1] += dy
            vel = np.zeros(6)
            vel[1] = vy
            return pose, vel

        if kind == "sin_tool_y":
            dy, vy = sin_y_motion(
                t_s, self.amplitude_m, self.omega,
                soft_start=self.cfg.soft_start, ramp_s=self.cfg.ramp_s,
            )
            pose = np.asarray(
                tool_offset_pose(self.robot, list(self.pose0), 0.0, dy, 0.0), dtype=float
            )
            pose_p = np.asarray(
                tool_offset_pose(
                    self.robot, list(self.pose0), 0.0, dy + vy * 1e-3, 0.0
                ),
                dtype=float,
            )
            vel = (pose_p - pose) / 1e-3
            return pose, vel

        raise ValueError(f"Unknown trajectory type: {kind}")

    @classmethod
    def from_dict(cls, raw: dict, pose0: np.ndarray, robot) -> TrajectoryGenerator:
        t = raw.get("trajectory", {})
        ps = t.get("period_s")
        return cls(
            TrajectoryConfig(
                kind=str(t.get("type", "sin_tool_y")),
                amplitude_mm=float(t.get("amplitude_mm", 5.0)),
                period_s=float(ps) if ps is not None else None,
                y_max_vel_cm_s=float(t.get("y_max_vel_cm_s", 1.0)),
                soft_start=bool(t.get("soft_start", False)),
                ramp_s=float(t.get("ramp_s", 2.0)),
            ),
            pose0,
            robot,
        )
