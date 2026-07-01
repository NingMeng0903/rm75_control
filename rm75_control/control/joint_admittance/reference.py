"""Motion references for the joint-admittance loop.

Re-uses hybrid_motion.MotionReference so any existing MotionReferenceSource
(demo trajectories, planners) is equally usable with the joint-space loop.

Provided here, self-contained (no robot handle needed - pure kinematics/scipy):

* HoldReference          - hold the start pose (bring-up default).
* CartesianMoveReference - smoothstep point-to-point Cartesian move (position +
  Slerp orientation), analytic vel_ff.  Forces a STRAIGHT Cartesian line - only
  appropriate very close to the target or away from singularities, since a
  straight-line Cartesian constraint can force awkward joint reconfiguration.
* JointSmoothMoveReference - smoothstep interpolation IN JOINT SPACE from q_start
  to q_target (from our pose_ik.solve_pose_ik, NOT vendor IK).  Exposed to the
  loop as FK/J(q_ref) Cartesian references via sample(), plus sample_q() for
  nullspace q_ref tracking.  Execute through CartesianTrackOuterLoop +
  Phase.q_ref_provider + inner.update(..., q_ref=...) so CLIK/nullspace stay live.
  Do NOT use update_joint() for this - that bypasses nullspace entirely.
* SinToolYReference      - tool-frame Y sinusoid about a fixed origin (analogue
  of the tmp/Velocity_Admittance BuiltinTrajectorySource "sin_tool_y" mode, but
  computed directly instead of via robot.rm_algo_pose_move).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as Rsc
from scipy.spatial.transform import Slerp

from rm75_control.control.hybrid_motion.reference import MotionReference


class HoldReference:
    """Hold the start pose: pose_d = pose0, vel_ff = 0 (bring-up default).

    With force enabled and force_axes = tool-Z, this yields a pure constant-force
    hold - the safest first on-robot test of the cascade.
    """

    def __init__(self) -> None:
        self._pose0: np.ndarray | None = None

    def set_origin(self, pose0: np.ndarray) -> None:
        self._pose0 = np.asarray(pose0, dtype=float).copy()

    def sample(self, t_s: float) -> MotionReference:
        if self._pose0 is None:
            raise RuntimeError("HoldReference.set_origin must be called first")
        return MotionReference.from_pose_hold(self._pose0)


def interpolate_pose_smoothstep(
    pose_start: np.ndarray,
    pose_end: np.ndarray,
    t_s: float,
    duration_s: float,
    *,
    euler_order: str = "xyz",
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothstep (C1, zero end-velocity) pose blend with analytic vel_ff.

    Position: linear interpolation scaled by s(u) = 3u^2 - 2u^3, u = t/T.
    Orientation: Slerp along the same s(u); velocity via a small finite
    difference on the Slerp path (robust for any relative rotation angle).
    """
    pose_start = np.asarray(pose_start, dtype=float)
    pose_end = np.asarray(pose_end, dtype=float)
    if duration_s <= 0.0:
        return pose_end.copy(), np.zeros(6, dtype=float)

    u = float(np.clip(t_s / duration_s, 0.0, 1.0))
    s = u * u * (3.0 - 2.0 * u)
    ds_dt = 6.0 * u * (1.0 - u) / duration_s

    pose = np.zeros(6, dtype=float)
    pose[:3] = (1.0 - s) * pose_start[:3] + s * pose_end[:3]

    r0 = Rsc.from_euler(euler_order, pose_start[3:6], degrees=False)
    r1 = Rsc.from_euler(euler_order, pose_end[3:6], degrees=False)
    slerp = Slerp([0.0, 1.0], Rsc.concatenate([r0, r1]))
    rot_s = slerp([s])[0]
    pose[3:6] = rot_s.as_euler(euler_order, degrees=False)

    vel = np.zeros(6, dtype=float)
    vel[:3] = ds_dt * (pose_end[:3] - pose_start[:3])
    ds = 1e-4
    s2 = min(s + ds, 1.0)
    if s2 > s:
        rot_s2 = slerp([s2])[0]
        vel[3:6] = (rot_s2 * rot_s.inv()).as_rotvec() / (s2 - s) * ds_dt
    return pose, vel


class CartesianMoveReference:
    """Point-to-point Cartesian move: smoothstep pose0 -> pose_target over duration_s.

    Generic, reusable "Cartesian trajectory tracking" building block - drives the
    inner IK loop straight from the current pose to any target pose without any
    MoveJ/MoveV mode switch.  ``done(t_s)`` tells the caller when to advance to
    the next phase.
    """

    def __init__(
        self,
        pose_target: np.ndarray,
        duration_s: float,
        *,
        euler_order: str = "xyz",
    ) -> None:
        self.pose_target = np.asarray(pose_target, dtype=float).copy()
        self.duration_s = float(duration_s)
        self.euler_order = euler_order
        self._pose0: np.ndarray | None = None

    def set_origin(self, pose0: np.ndarray) -> None:
        self._pose0 = np.asarray(pose0, dtype=float).copy()

    def sample(self, t_s: float) -> MotionReference:
        if self._pose0 is None:
            raise RuntimeError("CartesianMoveReference.set_origin must be called first")
        pose, vel = interpolate_pose_smoothstep(
            self._pose0, self.pose_target, t_s, self.duration_s, euler_order=self.euler_order
        )
        return MotionReference(pose, vel, t_ref=t_s)

    def done(self, t_s: float) -> bool:
        return t_s >= self.duration_s


def smoothstep_scalar(t_s: float, duration_s: float) -> tuple[float, float]:
    """s(u) in [0,1] and ds/dt, u = clip(t/T, 0, 1), s = 3u^2 - 2u^3 (zero end-vel)."""
    if duration_s <= 0.0:
        return 1.0, 0.0
    u = float(np.clip(t_s / duration_s, 0.0, 1.0))
    s = u * u * (3.0 - 2.0 * u)
    ds_dt = 6.0 * u * (1.0 - u) / duration_s
    return s, ds_dt


class JointSmoothMoveReference:
    """Smoothstep move in JOINT SPACE (q_start -> q_target), exposed as a Cartesian
    MotionReferenceSource via FK/Jacobian - i.e. the "free-planned, natural motion"
    analogue of MoveJ (smooth joint interpolation, whatever curved Cartesian path
    that implies), rather than CartesianMoveReference's forced straight line.

    Feeding (pose(t), vel_ff(t) = J(q(t)) @ qdot(t)) into the CLIK/QP inner loop
    makes it track a target that is EXACTLY consistent with smooth joint motion,
    so the resulting q_cmd closely follows q(t) itself - the CLIK correction only
    has to cancel small linearization residuals, not fight a Cartesian constraint.
    Requires q_target to already be resolved (e.g. via the vendor IK or an offline
    CLIK solve) - this class itself does no IK, it only interpolates.
    """

    def __init__(
        self,
        kin,
        q_start_rad: np.ndarray,
        q_target_rad: np.ndarray,
        duration_s: float,
    ) -> None:
        self.kin = kin
        self.q_start = np.asarray(q_start_rad, dtype=float).copy()
        self.q_target = np.asarray(q_target_rad, dtype=float).copy()
        self.duration_s = float(duration_s)

    def set_origin(self, pose0: np.ndarray) -> None:
        # q_start already anchors this reference; pose0 is implied by FK(q_start).
        del pose0

    def sample_q(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Joint-space (q_ref(t), qdot_ff(t)) - pass to Phase.q_ref_provider so the
        CLIK/QP nullspace pins the redundant DOF to this smoothstep (no drift)."""
        s, ds_dt = smoothstep_scalar(t_s, self.duration_s)
        dq = self.q_target - self.q_start
        q = self.q_start + s * dq
        qdot = ds_dt * dq
        return q, qdot

    def sample(self, t_s: float) -> MotionReference:
        """Cartesian (pose, vel_ff) view via FK/Jacobian - feed through
        CartesianTrackOuterLoop; pair with q_ref_provider for nullspace tracking."""
        q, qdot = self.sample_q(t_s)
        pose = self.kin.fk_pose(q)
        vel = self.kin.jacobian(q) @ qdot
        return MotionReference(pose, vel, t_ref=t_s)

    def done(self, t_s: float) -> bool:
        return t_s >= self.duration_s


def sin_period_for_peak_vel(amplitude_m: float, max_vel_m_s: float) -> float:
    if amplitude_m <= 0.0 or max_vel_m_s <= 0.0:
        return 1.0
    return 2.0 * math.pi * amplitude_m / max_vel_m_s


def sin_y_motion(
    t_s: float,
    amplitude_m: float,
    omega: float,
    *,
    soft_start: bool,
    ramp_s: float = 2.0,
) -> tuple[float, float]:
    dy = amplitude_m * math.sin(omega * t_s)
    vy = amplitude_m * omega * math.cos(omega * t_s)
    if soft_start and ramp_s > 0.0 and t_s < ramp_s:
        vy *= math.sin(0.5 * math.pi * t_s / ramp_s)
    return dy, vy


class SinToolYReference:
    """Tool-frame Y sinusoid about a fixed origin (orientation held constant).

    origin is set once via ``set_origin`` (e.g. pose D once the arm has arrived);
    pose = origin + R(origin) @ [0, amplitude*sin(wt), 0], matching a pure
    tool-frame translation delta (equivalent to rm_algo_pose_move with a
    translation-only delta in tool frame, computed directly - no robot RPC).
    """

    def __init__(
        self,
        amplitude_m: float,
        *,
        period_s: float | None = None,
        max_vel_m_s: float | None = None,
        soft_start: bool = True,
        ramp_s: float = 2.0,
        euler_order: str = "xyz",
    ) -> None:
        if period_s is None:
            if max_vel_m_s is None:
                raise ValueError("provide either period_s or max_vel_m_s")
            period_s = sin_period_for_peak_vel(amplitude_m, max_vel_m_s)
        self.amplitude_m = float(amplitude_m)
        self.period_s = float(period_s)
        self.omega = 2.0 * math.pi / self.period_s if self.period_s > 0 else 0.0
        self.soft_start = soft_start
        self.ramp_s = ramp_s
        self.euler_order = euler_order
        self._origin: np.ndarray | None = None

    def set_origin(self, pose0: np.ndarray) -> None:
        self._origin = np.asarray(pose0, dtype=float).copy()

    def sample(self, t_s: float) -> MotionReference:
        if self._origin is None:
            raise RuntimeError("SinToolYReference.set_origin must be called first")
        dy, vy = sin_y_motion(
            t_s, self.amplitude_m, self.omega, soft_start=self.soft_start, ramp_s=self.ramp_s
        )
        r_mat = Rsc.from_euler(self.euler_order, self._origin[3:6], degrees=False).as_matrix()
        pose = self._origin.copy()
        pose[:3] = self._origin[:3] + r_mat @ np.array([0.0, dy, 0.0])
        vel = np.zeros(6, dtype=float)
        vel[:3] = r_mat @ np.array([0.0, vy, 0.0])
        return MotionReference(pose, vel, t_ref=t_s)
