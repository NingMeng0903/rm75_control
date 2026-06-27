"""Thin wrapper for rm_force_position_move (native params preserved)."""

from __future__ import annotations

from typing import Protocol, Sequence

from rm75_control.core.exceptions import MotionError


class ForcePositionClient(Protocol):
    def rm_force_position_move(self, param) -> int:
        ...


def send_force_position_move(robot: ForcePositionClient, param) -> None:
    ret = robot.rm_force_position_move(param)
    if ret != 0:
        raise MotionError(f"rm_force_position_move failed with code {ret}")
