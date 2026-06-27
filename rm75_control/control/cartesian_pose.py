"""Cartesian pose CANFD (rm_movep_canfd) with optional Ruckig preprocessing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from rm75_control.core.exceptions import MotionError
from rm75_control.motion.canfd import PoseCanfdClient, send_pose_canfd

Pose6 = list[float]


@dataclass
class CartesianLimits:
    max_velocity: list[float] = field(
        default_factory=lambda: [0.05, 0.05, 0.05, 0.3, 0.3, 0.3]
    )
    max_acceleration: list[float] = field(
        default_factory=lambda: [0.5, 0.5, 0.5, 1.5, 1.5, 1.5]
    )
    max_jerk: list[float] = field(
        default_factory=lambda: [2.0, 2.0, 2.0, 5.0, 5.0, 5.0]
    )


@dataclass
class CartesianPoseStreamConfig:
    """Native rm_movep_canfd params + optional Ruckig upstream."""

    use_ruckig: bool = False
    period_ms: float = 10.0
    follow: bool = True
    trajectory_mode: int = 0
    radio: int = 0
    limits: CartesianLimits = field(default_factory=CartesianLimits)
    steps_per_segment: int = 50

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        use_ruckig: bool | None = None,
        follow: bool | None = None,
        trajectory_mode: int | None = None,
        radio: int | None = None,
        period_ms: float | None = None,
        steps_per_segment: int | None = None,
        **overrides: Any,
    ) -> CartesianPoseStreamConfig:
        motion = config.get("motion", {})
        cartesian = motion.get("cartesian_pose", {})
        ruckig_cfg = cartesian.get("ruckig", motion.get("ruckig", {}))
        timing = config.get("timing", {})

        limits = CartesianLimits(
            max_velocity=ruckig_cfg.get(
                "max_velocity", CartesianLimits().max_velocity
            ),
            max_acceleration=ruckig_cfg.get(
                "max_acceleration", CartesianLimits().max_acceleration
            ),
            max_jerk=ruckig_cfg.get("max_jerk", CartesianLimits().max_jerk),
        )

        def _pick(key: str, legacy_key: str, default: Any) -> Any:
            if key in cartesian:
                return cartesian[key]
            if legacy_key in motion:
                return motion[legacy_key]
            return default

        values: dict[str, Any] = {
            "use_ruckig": cartesian.get("use_ruckig", motion.get("use_ruckig", False)),
            "period_ms": cartesian.get(
                "period_ms", timing.get("canfd_period_ms", 10.0)
            ),
            "follow": _pick("follow", "canfd_follow", True),
            "trajectory_mode": _pick("trajectory_mode", "canfd_trajectory_mode", 0),
            "radio": _pick("radio", "canfd_radio", 0),
            "limits": limits,
            "steps_per_segment": cartesian.get(
                "steps_per_segment", motion.get("steps_per_segment", 50)
            ),
        }

        optional = {
            "use_ruckig": use_ruckig,
            "follow": follow,
            "trajectory_mode": trajectory_mode,
            "radio": radio,
            "period_ms": period_ms,
            "steps_per_segment": steps_per_segment,
        }
        for key, val in optional.items():
            if val is not None:
                values[key] = val
        values.update(overrides)
        return cls(**values)


class CartesianPoseController:
    """Stream Cartesian poses through rm_movep_canfd."""

    def __init__(
        self,
        robot: PoseCanfdClient,
        config: CartesianPoseStreamConfig | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        self.robot = robot
        self.config = config or CartesianPoseStreamConfig()
        self.dry_run = dry_run

    @property
    def use_ruckig(self) -> bool:
        return self.config.use_ruckig

    def run(
        self,
        waypoints: Sequence[Pose6],
        *,
        start_pose: Sequence[float] | None = None,
    ) -> None:
        if len(waypoints) == 0:
            raise ValueError("waypoints must not be empty")

        normalized = [_normalize_pose6(p) for p in waypoints]
        if self.config.use_ruckig:
            initial = (
                _normalize_pose6(start_pose)
                if start_pose is not None
                else normalized[0]
            )
            stream = self._generate_ruckig(normalized, initial)
        else:
            stream = self._generate_direct(normalized)

        dt = self.config.period_ms / 1000.0
        for pose in stream:
            if not self.dry_run:
                send_pose_canfd(
                    self.robot,
                    pose,
                    follow=self.config.follow,
                    trajectory_mode=self.config.trajectory_mode,
                    radio=self.config.radio,
                )
            if dt > 0.0:
                time.sleep(dt)

    def _generate_direct(self, waypoints: list[Pose6]) -> Iterator[Pose6]:
        if len(waypoints) == 1:
            yield waypoints[0]
            return

        steps = max(1, self.config.steps_per_segment)
        if steps == 1:
            for p in waypoints:
                yield p
            return

        for start, end in zip(waypoints[:-1], waypoints[1:]):
            start_arr = np.asarray(start, dtype=float)
            end_arr = np.asarray(end, dtype=float)
            for alpha in np.linspace(0.0, 1.0, steps, endpoint=True):
                yield list(start_arr + alpha * (end_arr - start_arr))

    def _generate_ruckig(
        self,
        waypoints: list[Pose6],
        initial_pose: Pose6,
    ) -> Iterator[Pose6]:
        dt = self.config.period_ms / 1000.0
        dofs = 6
        otg = Ruckig(dofs, dt)
        inp = InputParameter(dofs)
        out = OutputParameter(dofs)

        limits = self.config.limits
        inp.max_velocity = limits.max_velocity
        inp.max_acceleration = limits.max_acceleration
        inp.max_jerk = limits.max_jerk

        current = list(initial_pose)
        for target in waypoints:
            if np.allclose(current, target, atol=1e-9):
                continue
            inp.current_position = current
            inp.current_velocity = [0.0] * dofs
            inp.current_acceleration = [0.0] * dofs
            inp.target_position = _normalize_pose6(target)
            inp.target_velocity = [0.0] * dofs
            inp.target_acceleration = [0.0] * dofs

            result = Result.Working
            while result == Result.Working:
                result = otg.update(inp, out)
                if result not in (Result.Working, Result.Finished):
                    raise MotionError(f"Ruckig failed with result {result}")
                yield list(out.new_position)
                out.pass_to_input(inp)

            current = list(out.new_position)


def _normalize_pose6(pose: Sequence[float]) -> Pose6:
    if len(pose) == 6:
        return [float(v) for v in pose]
    if len(pose) == 7:
        raise ValueError(
            "Quaternion pose is not supported in Ruckig path yet; use 6D euler."
        )
    raise ValueError(f"pose must have 6 elements, got {len(pose)}")
