"""CANFD pose streaming via rm_movep_canfd."""

from __future__ import annotations

from typing import Protocol, Sequence

from rm75_control.core.exceptions import MotionError

Pose6 = Sequence[float]


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
