"""Reference pose + analytic feedforward velocity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


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
        self._y0_tool = float(robot.rm_algo_end2tool(list(pose0))[1])

    def sample(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        kind = self.cfg.kind
        if kind == "hold":
            return self.pose0.copy(), np.zeros(6)

        if kind == "sin_base_y":
            dy = self.amplitude_m * math.sin(self.omega * t_s)
            vy = self.amplitude_m * self.omega * math.cos(self.omega * t_s)
            pose = self.pose0.copy()
            pose[1] += dy
            vel = np.zeros(6)
            vel[1] = vy
            return pose, vel

        if kind == "sin_tool_y":
            dy = self.amplitude_m * math.sin(self.omega * t_s)
            vy = self.amplitude_m * self.omega * math.cos(self.omega * t_s)
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
            ),
            pose0,
            robot,
        )
