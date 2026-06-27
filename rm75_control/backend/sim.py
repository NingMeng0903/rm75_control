"""Simulation backend — rm_algo FK + mock wrench."""

from __future__ import annotations

from rm75_control.backend.base import RobotBackend


class SimBackend(RobotBackend):
    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def get_joint_positions(self) -> list[float]:
        raise NotImplementedError

    def get_tcp_pose(self) -> list[float]:
        raise NotImplementedError
