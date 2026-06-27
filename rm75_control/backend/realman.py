"""RealMan RM_API2 backend — thin wrapper over RoboticArm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rm75_control.backend.base import RobotBackend
from rm75_control.core.exceptions import RobotConnectionError

if TYPE_CHECKING:
    from Robotic_Arm.rm_robot_interface import RoboticArm


class RealManBackend(RobotBackend):
    """Maps rm75_control calls to Robotic_Arm.rm_* APIs."""

    def __init__(self, ip: str, port: int, thread_mode: int = 2) -> None:
        self.ip = ip
        self.port = port
        self.thread_mode = thread_mode
        self._robot: RoboticArm | None = None

    @property
    def robot(self) -> RoboticArm:
        if self._robot is None:
            raise RobotConnectionError("Robot is not connected")
        return self._robot

    def connect(self) -> None:
        from Robotic_Arm.rm_ctypes_wrap import rm_thread_mode_e
        from Robotic_Arm.rm_robot_interface import RoboticArm

        self._robot = RoboticArm(rm_thread_mode_e(self.thread_mode))
        handle = self._robot.rm_create_robot_arm(self.ip, self.port)
        if handle.id == -1:
            self._robot = None
            raise RobotConnectionError(
                f"Failed to connect to robot at {self.ip}:{self.port}"
            )

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.rm_delete_robot_arm()
            self._robot = None

    def get_joint_positions(self) -> list[float]:
        ret, state = self.robot.rm_get_current_arm_state()
        if ret != 0:
            raise RobotConnectionError(f"rm_get_current_arm_state failed: {ret}")
        return list(state["joint"])

    def get_tcp_pose(self) -> list[float]:
        ret, state = self.robot.rm_get_current_arm_state()
        if ret != 0:
            raise RobotConnectionError(f"rm_get_current_arm_state failed: {ret}")
        return list(state["pose"])
