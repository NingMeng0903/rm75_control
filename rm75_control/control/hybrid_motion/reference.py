"""External motion reference — the only trajectory type the hybrid controller sees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class MotionReference:
    """One control tick: desired pose + feed-forward velocity (base/world frame)."""

    pose_d: np.ndarray
    vel_ff: np.ndarray
    t_ref: float = 0.0
    valid: bool = True

    @classmethod
    def from_pose_hold(cls, pose: np.ndarray) -> MotionReference:
        return cls(np.asarray(pose, dtype=float).copy(), np.zeros(6, dtype=float))

    @classmethod
    def from_pose_delta(
        cls,
        pose: np.ndarray,
        pose_prev: np.ndarray,
        dt: float,
        *,
        alpha: float = 0.2,
    ) -> MotionReference:
        """Degraded mode: finite-difference velocity with optional low-pass."""
        pose = np.asarray(pose, dtype=float)
        pose_prev = np.asarray(pose_prev, dtype=float)
        if dt <= 0.0:
            return cls.from_pose_hold(pose)
        vel = (pose - pose_prev) / dt
        vel_ff = np.zeros(6, dtype=float)
        vel_ff[:3] = vel[:3]
        vel_ff[3:6] = vel[3:6]
        if 0.0 < alpha < 1.0:
            vel_ff = alpha * vel_ff
        return cls(pose.copy(), vel_ff)


# Migration alias — demos may still import TrajectorySample.
TrajectorySample = MotionReference


class MotionReferenceSource(Protocol):
    """External planner / demo trajectory plugin."""

    def set_origin(self, pose0: np.ndarray) -> None: ...

    def sample(self, t_s: float) -> MotionReference: ...

    # Optional v2: def sample_ahead(self, t_s: float, tau_s: float) -> MotionReference: ...
