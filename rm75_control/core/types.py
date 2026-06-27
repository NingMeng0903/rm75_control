"""Control modes and shared data types."""

from __future__ import annotations

from enum import Enum, auto


class ControlMode(Enum):
    IDLE = auto()
    PTP_PLANNED = auto()
    CARTESIAN_POSE_CANFD = auto()
    CARTESIAN_VEL_CANFD = auto()
    JOINT_CANFD = auto()
    FORCE_SCAN = auto()
    FORCE_PTP = auto()
