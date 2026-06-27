"""Force-position scan streaming for surface following (native hybrid control)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from rm75_control.core.exceptions import MotionError
from rm75_control.motion.force_position import ForcePositionClient, send_force_position_move

Pose6 = Sequence[float]
Joint7 = Sequence[float]


@dataclass
class ForceScanConfig:
    """Native rm_force_position_move params + loop timing."""

    flag: int = 0
    sensor: int = 1
    mode: int = 1
    follow: bool = True
    control_mode: list[int] = field(default_factory=lambda: [3, 3, 4, 0, 0, 0])
    desired_force: list[float] = field(default_factory=lambda: [0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    limit_vel: list[float] = field(
        default_factory=lambda: [0.05, 0.05, 0.05, 0.3, 0.3, 0.3]
    )
    trajectory_mode: int = 0
    radio: int = 0
    period_ms: float = 10.0

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        **overrides: Any,
    ) -> ForceScanConfig:
        force_cfg = config.get("force", {})
        scan_cfg = force_cfg.get("scan", {})
        timing = config.get("timing", {})
        values: dict[str, Any] = {
            "flag": scan_cfg.get("flag", 0),
            "sensor": scan_cfg.get("sensor", force_cfg.get("sensor", 1)),
            "mode": scan_cfg.get("mode", force_cfg.get("coordinate_mode", 1)),
            "follow": scan_cfg.get("follow", True),
            "control_mode": scan_cfg.get(
                "control_mode", force_cfg.get("default_control_mode", [3, 3, 4, 0, 0, 0])
            ),
            "desired_force": scan_cfg.get(
                "desired_force", force_cfg.get("default_desired_force", [0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
            ),
            "limit_vel": scan_cfg.get("limit_vel", [0.05, 0.05, 0.05, 0.3, 0.3, 0.3]),
            "trajectory_mode": scan_cfg.get("trajectory_mode", 0),
            "radio": scan_cfg.get("radio", 0),
            "period_ms": scan_cfg.get("period_ms", timing.get("force_scan_period_ms", 10.0)),
        }
        values.update(overrides)
        return cls(**values)


class ForceScanController:
    """
    Periodic rm_force_position_move loop.

    MPC / xi-space planning stays outside this class: provide a callback that
    maps (xi, observation) -> joint or pose command each cycle.
    """

    def __init__(
        self,
        robot: ForcePositionClient,
        config: ForceScanConfig | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        self.robot = robot
        self.config = config or ForceScanConfig()
        self.dry_run = dry_run
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        if self.dry_run:
            self._active = True
            return
        ret = self.robot.rm_start_force_position_move()
        if ret != 0:
            raise MotionError(f"rm_start_force_position_move failed with code {ret}")
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        if not self.dry_run:
            ret = self.robot.rm_stop_force_position_move()
            if ret != 0:
                # -2 receive timeout is common after Ctrl+C; fall back to arm stop.
                try:
                    self.robot.rm_set_arm_slow_stop()
                except Exception:
                    pass
                if ret not in (-2,):
                    raise MotionError(
                        f"rm_stop_force_position_move failed with code {ret}"
                    )
        self._active = False

    def step_joint(
        self,
        joint: Joint7,
        *,
        desired_force: Sequence[float] | None = None,
        control_mode: Sequence[int] | None = None,
    ) -> None:
        self._send(flag=0, joint=joint, pose=None, desired_force=desired_force, control_mode=control_mode)

    def step_pose(
        self,
        pose: Pose6,
        *,
        desired_force: Sequence[float] | None = None,
        control_mode: Sequence[int] | None = None,
    ) -> None:
        self._send(flag=1, joint=None, pose=pose, desired_force=desired_force, control_mode=control_mode)

    def run_xi_loop(
        self,
        xi0: Sequence[float],
        step_fn: Callable[[Sequence[float], float], tuple[Sequence[float], int, Sequence[float] | None]],
        *,
        duration_s: float | None = None,
        max_steps: int | None = None,
    ) -> None:
        """
        Run scan loop driven by xi-space policy.

        step_fn(xi, t) -> (xi_next, command_flag, command)
            command_flag 0: command is joint[7]
            command_flag 1: command is pose[6]
        """
        if not self._active:
            raise RuntimeError("ForceScanController is not started")

        dt = self.config.period_ms / 1000.0
        xi = list(xi0)
        t = 0.0
        steps = 0
        while True:
            xi, flag, cmd = step_fn(xi, t)
            if flag == 0:
                self.step_joint(cmd)
            elif flag == 1:
                self.step_pose(cmd)
            else:
                raise ValueError(f"command_flag must be 0 or 1, got {flag}")

            t += dt
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
            if duration_s is not None and t >= duration_s:
                break
            if dt > 0.0:
                time.sleep(dt)

    def _send(
        self,
        *,
        flag: int,
        joint: Joint7 | None,
        pose: Pose6 | None,
        desired_force: Sequence[float] | None,
        control_mode: Sequence[int] | None,
    ) -> None:
        if not self._active:
            raise RuntimeError("ForceScanController is not started")

        from Robotic_Arm.rm_ctypes_wrap import (
            rm_euler_t,
            rm_force_position_move_t,
            rm_pose_t,
            rm_position_t,
        )
        from ctypes import c_float, c_int

        cfg = self.config
        param = rm_force_position_move_t()
        param.flag = flag
        param.sensor = cfg.sensor
        param.mode = cfg.mode
        param.follow = cfg.follow
        param.trajectory_mode = cfg.trajectory_mode
        param.radio = cfg.radio

        if flag == 1 and pose is not None:
            po = rm_pose_t()
            po.position = rm_position_t(*pose[:3])
            po.euler = rm_euler_t(*pose[3:6])
            param.pose = po
        elif flag == 0 and joint is not None:
            param.joint = (c_float * 7)(*joint)

        modes = list(control_mode) if control_mode is not None else cfg.control_mode
        forces = list(desired_force) if desired_force is not None else cfg.desired_force
        param.control_mode = (c_int * 6)(*modes)
        param.desired_force = (c_float * 6)(*forces)
        param.limit_vel = (c_float * 6)(*cfg.limit_vel)

        if not self.dry_run:
            send_force_position_move(self.robot, param)
