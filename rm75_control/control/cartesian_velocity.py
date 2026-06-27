"""Cartesian velocity CANFD with outer position loop (feedforward + P/I)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from rm75_control.motion.canfd import (
    VelocityCanfdClient,
    clamp_cartesian_velocity,
    send_velocity_canfd,
)

Pose6 = list[float]
Vel6 = list[float]


@dataclass
class AxisVelocityGains:
    """Position error -> velocity correction on one Cartesian axis."""

    kp: float = 0.0
    ki: float = 0.0
    integral_limit: float = 0.05


@dataclass
class CartesianVelocityTrackerConfig:
    """Per-axis outer loop gains for rm_movev_canfd streaming."""

    axis_gains: list[AxisVelocityGains] = field(
        default_factory=lambda: [AxisVelocityGains() for _ in range(6)]
    )
    max_correction_m_s: float = 0.015
    motion_axes: tuple[int, ...] = ()
    ref_speed_peak_m_s: float = 0.0

    @classmethod
    def for_motion_axes(
        cls,
        motion_axes: Sequence[int],
        *,
        kp: float = 2.0,
        ki: float = 0.0,
        hold_axes: Sequence[int] | None = None,
        hold_kp: float = 1.0,
        ref_speed_peak_m_s: float = 0.0,
    ) -> CartesianVelocityTrackerConfig:
        gains = [AxisVelocityGains() for _ in range(6)]
        for idx in motion_axes:
            gains[idx] = AxisVelocityGains(kp=kp, ki=ki)
        if hold_axes is not None:
            for idx in hold_axes:
                if idx not in motion_axes:
                    gains[idx] = AxisVelocityGains(kp=hold_kp)
        return cls(
            axis_gains=gains,
            motion_axes=tuple(motion_axes),
            ref_speed_peak_m_s=ref_speed_peak_m_s,
        )


class CartesianVelocityTracker:
    """
    Outer position loop on top of velocity CANFD.

    vy_cmd = vy_ff + Kp*(y_ref - y_fb) + Ki*integral(err)

    Intended building block for hybrid control: motion axes via movev + position
    closure; force axes stay on rm_force_position_move (see force/scan.py).
    """

    def __init__(self, config: CartesianVelocityTrackerConfig | None = None) -> None:
        self.config = config or CartesianVelocityTrackerConfig()
        self._integral = [0.0] * 6

    def reset(self) -> None:
        self._integral = [0.0] * 6

    def _kp_scale(self, axis: int, ref_vel: float) -> float:
        peak = self.config.ref_speed_peak_m_s
        if peak <= 0.0 or axis not in self.config.motion_axes:
            return 1.0
        ratio = min(1.0, abs(ref_vel) / peak)
        # Near max speed error is mostly phase lag — back off P to avoid spikes.
        return max(0.1, 1.0 - 0.9 * ratio * ratio)

    def compute(
        self,
        *,
        ref_pose: Sequence[float],
        ref_vel: Sequence[float],
        fb_pose: Sequence[float],
        dt_s: float,
    ) -> Vel6:
        vel = [float(ref_vel[i]) for i in range(6)]
        max_corr = self.config.max_correction_m_s
        for i in range(6):
            g = self.config.axis_gains[i]
            if g.kp == 0.0 and g.ki == 0.0:
                continue
            err = float(ref_pose[i]) - float(fb_pose[i])
            corr = 0.0
            if g.ki != 0.0 and dt_s > 0.0:
                self._integral[i] += err * dt_s
                lim = g.integral_limit
                self._integral[i] = max(-lim, min(lim, self._integral[i]))
                corr += g.ki * self._integral[i]
            corr += g.kp * err * self._kp_scale(i, float(ref_vel[i]))
            if max_corr > 0.0:
                corr = max(-max_corr, min(max_corr, corr))
            vel[i] += corr
        return clamp_cartesian_velocity(vel)


@dataclass
class CartesianVelocityStreamConfig:
    follow: bool = True
    trajectory_mode: int = 1
    radio: int = 40
    period_ms: float = 10.0


class CartesianVelocityController:
    """Stream clamped Cartesian velocities through rm_movev_canfd."""

    def __init__(
        self,
        robot: VelocityCanfdClient,
        tracker: CartesianVelocityTracker | None = None,
        config: CartesianVelocityStreamConfig | None = None,
        *,
        dry_run: bool = False,
        max_dv_m_s: float = 0.0004,
    ) -> None:
        self.robot = robot
        self.tracker = tracker or CartesianVelocityTracker()
        self.config = config or CartesianVelocityStreamConfig()
        self.dry_run = dry_run
        self.max_dv_m_s = max_dv_m_s
        self._last_vel: Vel6 = [0.0] * 6

    def reset_slew(self) -> None:
        self._last_vel = [0.0] * 6

    def send_velocity(self, vel: Vel6) -> None:
        if self.dry_run:
            return
        send_velocity_canfd(
            self.robot,
            vel,
            follow=self.config.follow,
            trajectory_mode=self.config.trajectory_mode,
            radio=self.config.radio,
        )

    def step(
        self,
        *,
        ref_pose: Sequence[float],
        ref_vel: Sequence[float],
        fb_pose: Sequence[float],
        dt_s: float | None = None,
    ) -> Vel6:
        dt = dt_s if dt_s is not None else self.config.period_ms / 1000.0
        vel = self.tracker.compute(
            ref_pose=ref_pose,
            ref_vel=ref_vel,
            fb_pose=fb_pose,
            dt_s=dt,
        )
        if self.max_dv_m_s > 0.0:
            for i in range(6):
                dv = vel[i] - self._last_vel[i]
                dv = max(-self.max_dv_m_s, min(self.max_dv_m_s, dv))
                vel[i] = self._last_vel[i] + dv
        self._last_vel = list(vel)
        self.send_velocity(vel)
        return vel
