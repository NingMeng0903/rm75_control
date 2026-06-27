"""Abstract robot backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class RobotBackend(ABC):
    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def get_joint_positions(self) -> Sequence[float]:
        ...

    @abstractmethod
    def get_tcp_pose(self) -> Sequence[float]:
        ...
