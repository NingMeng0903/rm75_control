"""CANFD pose and velocity streaming via rm_movep_canfd / rm_movev_canfd."""

from __future__ import annotations

from typing import Protocol, Sequence

from rm75_control.core.exceptions import MotionError

Pose6 = Sequence[float]
Vel6 = Sequence[float]

MAX_LINEAR_V_M_S = 0.25
MAX_ANGULAR_V_RAD_S = 0.6


class PoseCanfdClient(Protocol):
    def rm_movep_canfd(
        self,
        pose: list[float],
        follow: bool,
        trajectory_mode: int = 0,
        radio: int = 0,
    ) -> int:
        ...


def send_pose_canfd(
    robot: PoseCanfdClient,
    pose: Pose6,
    *,
    follow: bool = True,
    trajectory_mode: int = 0,
    radio: int = 0,
) -> None:
    if len(pose) not in (6, 7):
        raise ValueError(f"pose must have 6 (euler) or 7 (quat) elements, got {len(pose)}")

    ret = robot.rm_movep_canfd(
        list(pose),
        follow,
        trajectory_mode,
        radio,
    )
    if ret != 0:
        raise MotionError(f"rm_movep_canfd failed with code {ret}")


class VelocityCanfdClient(Protocol):
    def rm_movev_canfd(
        self,
        cartesian_velocity: list[float],
        follow: bool,
        trajectory_mode: int = 0,
        radio: int = 0,
    ) -> int:
        ...


def clamp_cartesian_velocity(vel: Vel6) -> list[float]:
    out = list(vel)
    for i in range(3):
        out[i] = max(-MAX_LINEAR_V_M_S, min(MAX_LINEAR_V_M_S, out[i]))
    for i in range(3, 6):
        out[i] = max(-MAX_ANGULAR_V_RAD_S, min(MAX_ANGULAR_V_RAD_S, out[i]))
    return out


def send_velocity_canfd(
    robot: VelocityCanfdClient,
    cartesian_velocity: Vel6,
    *,
    follow: bool = False,
    trajectory_mode: int = 0,
    radio: int = 0,
) -> None:
    ret = robot.rm_movev_canfd(
        clamp_cartesian_velocity(cartesian_velocity),
        follow,
        trajectory_mode,
        radio,
    )
    if ret != 0:
        raise MotionError(f"rm_movev_canfd failed with code {ret}")
