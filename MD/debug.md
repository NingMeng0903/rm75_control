# Joint-Position QP-CLIK Controller — Full Source Dump

Auto-generated snapshot of the new cascaded joint-position controller.
Architecture: task-space admittance outer loop → joint-space CLIK/QP inner loop → `rm_movej_canfd`.

---

## FILE: `rm75_control/control/joint_admittance/__init__.py`

```py
"""Joint-space inner loop (Pinocchio CLIK / QP IK) for RM75-F.

Cascaded controller: the task-space admittance outer loop
(rm75_control.control.hybrid_motion.controller.AdmittanceController) produces a
6D Cartesian twist, and this package converts it to absolute joint angles that
are streamed through a single interface (rm_movej_canfd) - no MoveJ/MoveV mode
switching.

Imports are kept lazy: `import rm75_control.control.joint_admittance` does NOT
pull in Pinocchio.  Import the submodules explicitly when you need them, e.g.::

    from rm75_control.control.joint_admittance.model import RobotKinematics
    from rm75_control.control.joint_admittance.clik import ClikController
"""

from __future__ import annotations

__all__ = [
    "RobotKinematics",
    "ClikController",
    "ClikConfig",
    "JointIkController",
]


def __getattr__(name: str):  # PEP 562 lazy re-export (avoids importing pinocchio eagerly)
    if name in ("RobotKinematics",):
        from rm75_control.control.joint_admittance.model import RobotKinematics

        return RobotKinematics
    if name in ("ClikController", "ClikConfig"):
        from rm75_control.control.joint_admittance import clik

        return getattr(clik, name)
    if name in ("JointIkController",):
        from rm75_control.control.joint_admittance.loop import JointIkController

        return JointIkController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

## FILE: `rm75_control/control/joint_admittance/clik.py`

```py
"""Closed-Loop Inverse Kinematics (CLIK) with DLS + nullspace projection.

Phase 1 inner-loop core.  Converts a base-frame 6D Cartesian twist reference
(from the admittance outer loop) into a joint-velocity command, integrates a
Cartesian reference pose for drift-free feedback, and projects a secondary task
(joint centering / limit avoidance) into the nullspace.

References:
* Sciavicco & Siciliano 1988; Siciliano 1990 -> CLIK error feedback.
* Wampler 1986 / Nakamura & Hanafusa 1986 -> damped least squares.
* Liegeois 1977; Chiaverini 1997 -> nullspace / gradient projection.

Design contract with the loop:
* The loop owns the *actually commanded* joint state `q_prev` (post safety /
  smoothing) and passes it in every tick.  CLIK never assumes its raw integration
  survived - it always re-linearizes about `q_prev`, so the loop is truly closed.
* CLIK owns the integrated Cartesian reference `x_ref` (advanced by the twist).
  Feedback `K * (x_ref - fk(q_prev))` removes DLS / discretization drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.joint_admittance.model import RobotKinematics, pose_error


@dataclass
class ClikConfig:
    # Cartesian error feedback gain (per axis: x,y,z,rx,ry,rz), 1/s.
    k_task: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=float)
    )
    # DLS sigma-scheduled damping (Wampler/Nakamura).
    sigma_thresh: float = 0.04      # start damping when smallest singular value drops below this
    lambda_max: float = 0.08        # max damping factor at the singularity
    # Nullspace secondary task gain scaling (applied to the provided qdot0).
    nullspace_gain: float = 1.0
    # Anti-windup: cap the fed-back Cartesian error so a saturated axis can't
    # blow up qdot (position m, orientation rad).
    max_pos_err_m: float = 0.05
    max_rot_err_rad: float = 0.20
    euler_order: str = "xyz"


@dataclass
class ClikResult:
    q_next: np.ndarray        # proposed next joint position (rad), pre-safety
    qdot: np.ndarray          # joint velocity command (rad/s)
    x_ref: np.ndarray         # integrated Cartesian reference (pose6)
    x_cur: np.ndarray         # current FK pose (pose6) at q_prev
    cart_err: np.ndarray      # 6D Cartesian error used for feedback
    sigma_min: float          # smallest singular value of J
    lam: float                # damping factor applied
    manip: float              # Yoshikawa manipulability


def damping_from_sigma(sigma_min: float, sigma_thresh: float, lambda_max: float) -> float:
    """lambda = 0 for sigma_min > thresh (pure pinv), ramps to lambda_max as sigma->0.

    lambda^2 = (1 - (sigma_min / thresh)^2) * lambda_max^2   for sigma_min < thresh.
    """
    if sigma_thresh <= 0.0 or sigma_min >= sigma_thresh:
        return 0.0
    ratio = sigma_min / sigma_thresh
    lam_sq = (1.0 - ratio * ratio) * (lambda_max * lambda_max)
    return float(np.sqrt(max(lam_sq, 0.0)))


def dls_pinv(J: np.ndarray, lam: float) -> np.ndarray:
    """Damped least-squares right pseudo-inverse J^T (J J^T + lam^2 I)^-1."""
    m = J.shape[0]
    JJt = J @ J.T
    if lam > 0.0:
        JJt = JJt + (lam * lam) * np.eye(m)
    return J.T @ np.linalg.solve(JJt, np.eye(m))


def integrate_pose(pose: np.ndarray, twist: np.ndarray, dt: float, euler_order: str = "xyz") -> np.ndarray:
    """Advance a base-frame pose6 by a base-frame twist [v_lin, omega] over dt."""
    pose = np.asarray(pose, dtype=float)
    out = pose.copy()
    out[:3] = pose[:3] + np.asarray(twist[:3], dtype=float) * dt
    R = Rsc.from_euler(euler_order, pose[3:6], degrees=False).as_matrix()
    dR = Rsc.from_rotvec(np.asarray(twist[3:6], dtype=float) * dt).as_matrix()
    out[3:6] = Rsc.from_matrix(dR @ R).as_euler(euler_order, degrees=False)
    return out


def _saturate_error(err: np.ndarray, max_pos: float, max_rot: float) -> np.ndarray:
    out = np.asarray(err, dtype=float).copy()
    pos_n = float(np.linalg.norm(out[:3]))
    if max_pos > 0.0 and pos_n > max_pos:
        out[:3] *= max_pos / pos_n
    rot_n = float(np.linalg.norm(out[3:6]))
    if max_rot > 0.0 and rot_n > max_rot:
        out[3:6] *= max_rot / rot_n
    return out


class ClikController:
    """Stateful CLIK integrator (owns only the Cartesian reference pose)."""

    def __init__(self, kin: RobotKinematics, cfg: ClikConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or ClikConfig()
        self.x_ref = np.zeros(6, dtype=float)
        self._initialized = False

    def reset(self, q0_rad: np.ndarray, pose0: np.ndarray | None = None) -> None:
        self.x_ref = (
            np.asarray(pose0, dtype=float).copy()
            if pose0 is not None
            else self.kin.fk_pose(q0_rad)
        )
        self._initialized = True

    def set_reference(self, pose: np.ndarray) -> None:
        self.x_ref = np.asarray(pose, dtype=float).copy()

    def step(
        self,
        q_prev: np.ndarray,
        twist_ref: np.ndarray,
        dt: float,
        secondary_qdot: np.ndarray | None = None,
    ) -> ClikResult:
        cfg = self.cfg
        q_prev = np.asarray(q_prev, dtype=float)
        twist_ref = np.asarray(twist_ref, dtype=float)
        if not self._initialized:
            self.reset(q_prev)

        # 1) advance the integrated Cartesian reference by the feed-forward twist
        self.x_ref = integrate_pose(self.x_ref, twist_ref, dt, cfg.euler_order)

        # 2) linearize about the actually-commanded joint state
        J = self.kin.jacobian(q_prev)
        sigma = self.kin.singular_values(J)
        sigma_min = float(sigma.min())
        lam = damping_from_sigma(sigma_min, cfg.sigma_thresh, cfg.lambda_max)
        Jdls = dls_pinv(J, lam)

        # 3) CLIK feedback against the integrated reference (drift correction)
        x_cur = self.kin.fk_pose(q_prev)
        err = pose_error(self.x_ref, x_cur, cfg.euler_order)
        err_sat = _saturate_error(err, cfg.max_pos_err_m, cfg.max_rot_err_rad)
        v_cmd = twist_ref + cfg.k_task * err_sat

        # 4) primary + nullspace secondary task
        qdot = Jdls @ v_cmd
        if secondary_qdot is not None and cfg.nullspace_gain != 0.0:
            N = np.eye(self.kin.nv) - Jdls @ J
            qdot = qdot + N @ (cfg.nullspace_gain * np.asarray(secondary_qdot, dtype=float))

        q_next = q_prev + qdot * dt
        return ClikResult(
            q_next=q_next,
            qdot=qdot,
            x_ref=self.x_ref.copy(),
            x_cur=x_cur,
            cart_err=err,
            sigma_min=sigma_min,
            lam=lam,
            manip=self.kin.manipulability(J),
        )
```

## FILE: `rm75_control/control/joint_admittance/config.py`

```py
"""YAML -> JointIkConfig loader for the joint-space inner loop.

Keeps the inner-loop tuning (gains K, DLS lambda schedule, nullspace gains,
smoothing cutoff, safety limits) in one config section so bring-up is a matter of
editing yaml, not code.  The outer admittance loop is still configured with the
existing hybrid_motion keys and built via AdmittanceConfig.from_dict.
"""

from __future__ import annotations

import math

import numpy as np

from rm75_control.control.joint_admittance.clik import ClikConfig
from rm75_control.control.joint_admittance.loop import JointIkConfig
from rm75_control.control.joint_admittance.tasks.nullspace_task import NullspaceTaskConfig


def _arr(v, default):
    return np.asarray(v if v is not None else default, dtype=float)


def build_joint_ik_config(raw: dict) -> JointIkConfig:
    timing = raw.get("timing", {})
    dt = float(timing.get("dt_ms", 10.0)) / 1000.0

    inner = raw.get("inner", {})
    euler_order = str(raw.get("frames", {}).get("euler_order", inner.get("euler_order", "xyz")))

    c = inner.get("clik", {})
    clik = ClikConfig(
        k_task=_arr(c.get("k_task"), [2.0] * 6),
        sigma_thresh=float(c.get("sigma_thresh", 0.04)),
        lambda_max=float(c.get("lambda_max", 0.08)),
        nullspace_gain=float(c.get("nullspace_gain", 1.0)),
        max_pos_err_m=float(c.get("max_pos_err_m", 0.05)),
        max_rot_err_rad=float(c.get("max_rot_err_rad", 0.20)),
        euler_order=euler_order,
    )

    n = inner.get("nullspace", {})
    nullspace = NullspaceTaskConfig(
        k_center=float(n.get("k_center", 0.5)),
        k_limit=float(n.get("k_limit", 2.0)),
        activation=float(n.get("activation", 0.85)),
        weights=(np.asarray(n["weights"], dtype=float) if n.get("weights") is not None else None),
    )

    margin_deg = float(inner.get("position_margin_deg", 1.0))

    cfg = JointIkConfig(
        dt=dt,
        control_frame=str(inner.get("control_frame", "base")),
        euler_order=euler_order,
        solver=str(inner.get("solver", "clik")),
        clik=clik,
        nullspace=nullspace,
        v_scale=float(inner.get("v_scale", 0.5)),
        a_max=float(inner.get("a_max", 20.0)),
        position_margin_rad=math.radians(margin_deg),
        use_smoothing=bool(inner.get("use_smoothing", True)),
        smooth_cutoff_hz=float(inner.get("smooth_cutoff_hz", 15.0)),
    )

    if cfg.solver == "qp":
        from rm75_control.control.joint_admittance.solver.qp_builder import QpConfig

        q = inner.get("qp", {})
        cfg.qp = QpConfig(
            k_task=_arr(q.get("k_task"), [2.0] * 6),
            task_weight=_arr(q.get("task_weight"), [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]),
            reg=float(q.get("reg", 1e-3)),
            reg_secondary_scale=float(q.get("reg_secondary_scale", 1.0)),
            max_pos_err_m=float(q.get("max_pos_err_m", clik.max_pos_err_m)),
            max_rot_err_rad=float(q.get("max_rot_err_rad", clik.max_rot_err_rad)),
            euler_order=euler_order,
            backend=str(q.get("backend", "proxqp")),
            eps_abs=float(q.get("eps_abs", 1e-6)),
            max_iter=int(q.get("max_iter", 200)),
        )
    return cfg
```

## FILE: `rm75_control/control/joint_admittance/model.py`

```py
"""Pinocchio kinematics engine for RM75-F (FK / Jacobian / manipulability).

The whole cascade is only as correct as this model.  Two conventions are pinned
here and must match the Realman controller:

* Joint order  : joint_1..joint_7, radians internally.  The robot API speaks
  degrees (rm_get_current_arm_state()["joint"], rm_movej_canfd) - convert at the
  boundary with deg2rad / rad2deg helpers.
* Cartesian    : the TCP twist / Jacobian are expressed LOCAL_WORLD_ALIGNED,
  i.e. linear velocity of the TCP point and angular velocity, both in base-frame
  axes.  This matches the base-frame 6D twist the admittance outer loop emits
  (controller.py, control_frame="base").  Pose is returned as
  [x, y, z, rx, ry, rz] with intrinsic xyz Euler (Realman convention).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rsc

DEFAULT_URDF = (
    Path(__file__).resolve().parents[2] / "assets" / "robots" / "rm75_6f" / "RM75-6F.urdf"
)

JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]


def deg2rad(q_deg: np.ndarray) -> np.ndarray:
    return np.asarray(q_deg, dtype=float) * (np.pi / 180.0)


def rad2deg(q_rad: np.ndarray) -> np.ndarray:
    return np.asarray(q_rad, dtype=float) * (180.0 / np.pi)


def pose_distance(
    pose_a: np.ndarray, pose_b: np.ndarray, euler_order: str = "xyz"
) -> tuple[float, float]:
    """Position distance (mm) and orientation distance (deg) between two pose6."""
    a = np.asarray(pose_a, dtype=float)
    b = np.asarray(pose_b, dtype=float)
    d_mm = float(np.linalg.norm(a[:3] - b[:3]) * 1000.0)
    ra = Rsc.from_euler(euler_order, a[3:6], degrees=False).as_matrix()
    rb = Rsc.from_euler(euler_order, b[3:6], degrees=False).as_matrix()
    d_deg = float(np.degrees(np.linalg.norm(Rsc.from_matrix(ra @ rb.T).as_rotvec())))
    return d_mm, d_deg


def pose_error(desired: np.ndarray, current: np.ndarray, euler_order: str = "xyz") -> np.ndarray:
    """Base-frame 6D pose error: linear diff + SO(3) log (rotvec of R_des @ R_cur^T).

    Mirrors hybrid_motion.controller.pose_error so the inner loop's Cartesian
    error definition is identical to the outer loop's.
    """
    err = np.zeros(6, dtype=float)
    err[:3] = np.asarray(desired[:3], dtype=float) - np.asarray(current[:3], dtype=float)
    r_des = Rsc.from_euler(euler_order, desired[3:6], degrees=False).as_matrix()
    r_cur = Rsc.from_euler(euler_order, current[3:6], degrees=False).as_matrix()
    err[3:6] = Rsc.from_matrix(r_des @ r_cur.T).as_rotvec()
    return err


class RobotKinematics:
    """Thin Pinocchio wrapper exposing FK, Jacobian and manipulability at the TCP."""

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        tcp_frame: str = "tcp",
        euler_order: str = "xyz",
    ) -> None:
        self.urdf_path = Path(urdf_path) if urdf_path is not None else DEFAULT_URDF
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.euler_order = euler_order

        if not self.model.existFrame(tcp_frame):
            raise ValueError(f"frame {tcp_frame!r} not in URDF {self.urdf_path}")
        self.tcp_frame = tcp_frame
        self.tcp_id = self.model.getFrameId(tcp_frame)

        self.nq = self.model.nq
        self.nv = self.model.nv
        if self.nq != 7 or self.nv != 7:
            raise ValueError(f"expected 7-DOF model, got nq={self.nq} nv={self.nv}")

        # Position / velocity limits (radians, rad/s) straight from the URDF.
        self.q_lower = np.asarray(self.model.lowerPositionLimit, dtype=float).copy()
        self.q_upper = np.asarray(self.model.upperPositionLimit, dtype=float).copy()
        self.v_max = np.asarray(self.model.velocityLimit, dtype=float).copy()

    # ---- forward kinematics ------------------------------------------------
    def fk_placement(self, q_rad: np.ndarray) -> pin.SE3:
        q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.tcp_id)
        return self.data.oMf[self.tcp_id]

    def fk_pose(self, q_rad: np.ndarray) -> np.ndarray:
        """TCP pose as [x, y, z, rx, ry, rz] (m, rad; intrinsic xyz Euler)."""
        M = self.fk_placement(q_rad)
        pose = np.zeros(6, dtype=float)
        pose[:3] = M.translation
        pose[3:6] = Rsc.from_matrix(M.rotation).as_euler(self.euler_order, degrees=False)
        return pose

    def fk_position_quat(self, q_rad: np.ndarray) -> np.ndarray:
        """TCP pose as [x, y, z, qx, qy, qz, qw] (handy for logging / comparisons)."""
        M = self.fk_placement(q_rad)
        quat = Rsc.from_matrix(M.rotation).as_quat()  # [x, y, z, w]
        return np.concatenate([M.translation, quat])

    def frame_placement(self, q_rad: np.ndarray, frame_name: str) -> pin.SE3:
        """SE3 of an arbitrary frame (e.g. 'link_7' flange) in the base frame."""
        if not self.model.existFrame(frame_name):
            raise ValueError(f"frame {frame_name!r} not in URDF {self.urdf_path}")
        fid = self.model.getFrameId(frame_name)
        q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, fid)
        return self.data.oMf[fid]

    def frame_pose(self, q_rad: np.ndarray, frame_name: str) -> np.ndarray:
        """Pose [x, y, z, rx, ry, rz] of an arbitrary frame in the base frame."""
        M = self.frame_placement(q_rad, frame_name)
        pose = np.zeros(6, dtype=float)
        pose[:3] = M.translation
        pose[3:6] = Rsc.from_matrix(M.rotation).as_euler(self.euler_order, degrees=False)
        return pose

    # ---- differential kinematics ------------------------------------------
    def jacobian(self, q_rad: np.ndarray) -> np.ndarray:
        """6x7 TCP Jacobian, LOCAL_WORLD_ALIGNED (linear on top, angular below).

        Maps joint velocity (rad/s) -> [v_lin(base), omega(base)].
        """
        q = np.asarray(q_rad, dtype=float)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(
            self.model, self.data, self.tcp_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return np.asarray(J, dtype=float)

    @staticmethod
    def manipulability(J: np.ndarray) -> float:
        """Yoshikawa measure sqrt(det(J J^T)); 0 at a singularity."""
        JJt = J @ J.T
        det = float(np.linalg.det(JJt))
        return float(np.sqrt(max(det, 0.0)))

    @staticmethod
    def singular_values(J: np.ndarray) -> np.ndarray:
        return np.linalg.svd(J, compute_uv=False)

    def clamp_to_limits(self, q_rad: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return np.clip(q_rad, self.q_lower + margin, self.q_upper - margin)
```

## FILE: `rm75_control/control/joint_admittance/reference.py`

```py
"""Motion references for the joint-admittance loop.

Re-uses hybrid_motion.MotionReference so any existing MotionReferenceSource
(demo trajectories, planners) is equally usable with the joint-space loop.

Provided here, self-contained (no robot handle needed - pure kinematics/scipy):

* HoldReference          - hold the start pose (bring-up default).
* CartesianMoveReference - smoothstep point-to-point Cartesian move (position +
  Slerp orientation), analytic vel_ff.  Drives the "walk to pose D" phase.
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
```

## FILE: `rm75_control/control/joint_admittance/validation.py`

```py
"""FK validation: Pinocchio model vs the real Realman controller (重中之重).

The entire cascade is only as trustworthy as the URDF <-> robot frame match.
Before running ANY joint-position control, prove that Pinocchio FK agrees with
the Realman pose interface to <1 mm / <0.1 deg.  If it does not, the URDF base
rotation or the TCP offset is wrong and every downstream Jacobian is wrong.

Two robot comparisons (both use rm_get_current_arm_state + rm_get_current_tool_frame):

* flange  (default, tool-agnostic): recover the base->flange (link_7) transform
  from the reported base->tool pose and the active tool offset, then compare to
  Pinocchio's link_7 FK.  Validates the 7-DOF arm chain independent of any tool.
* tcp     : compare Pinocchio's `tcp` frame FK (link_7 +0.220 m Z) directly to
  the reported base->tool pose.  Requires the ACTIVE Realman tool frame to be the
  matching +220 mm tool; otherwise it will (correctly) report the offset mismatch.

Usage (source env.sh first):
    # read-only single-shot at the current configuration
    python -m rm75_control.control.joint_admittance.validation --ip 192.168.1.18

    # drive fixed MoveJ points from a poses yaml and assert thresholds
    python -m rm75_control.control.joint_admittance.validation \
        --ip 192.168.1.18 --poses tmp/force_compensation/config/poses.yaml --move

    # offline: compare recorded (q_deg, pose) pairs, no robot
    python -m rm75_control.control.joint_admittance.validation --npz run.npz
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.joint_admittance.model import RobotKinematics, deg2rad, pose_distance

POS_TOL_MM = 1.0
ROT_TOL_DEG = 0.1


def pose_to_se3(pose6: np.ndarray, euler_order: str = "xyz"):
    """[x,y,z,rx,ry,rz] -> (t(3), R(3x3))."""
    pose6 = np.asarray(pose6, dtype=float)
    t = pose6[:3].copy()
    R = Rsc.from_euler(euler_order, pose6[3:6], degrees=False).as_matrix()
    return t, R


def se3_to_pose(t: np.ndarray, R: np.ndarray, euler_order: str = "xyz") -> np.ndarray:
    pose = np.zeros(6, dtype=float)
    pose[:3] = t
    pose[3:6] = Rsc.from_matrix(R).as_euler(euler_order, degrees=False)
    return pose


def se3_inv(t: np.ndarray, R: np.ndarray):
    Rt = R.T
    return -Rt @ t, Rt


def se3_mul(ta, Ra, tb, Rb):
    return ta + Ra @ tb, Ra @ Rb


def pose_diff(pose_a: np.ndarray, pose_b: np.ndarray, euler_order: str = "xyz") -> tuple[float, float]:
    """Return (position error mm, orientation error deg) between two pose6."""
    return pose_distance(pose_a, pose_b, euler_order)


def base_flange_from_tool(tool_pose: np.ndarray, tool_offset: np.ndarray, euler_order: str = "xyz") -> np.ndarray:
    """base->flange = base->tool * (flange->tool)^-1."""
    tb, Rb = pose_to_se3(tool_pose, euler_order)
    to, Ro = pose_to_se3(tool_offset, euler_order)
    ti, Ri = se3_inv(to, Ro)
    tf, Rf = se3_mul(tb, Rb, ti, Ri)
    return se3_to_pose(tf, Rf, euler_order)


def _summary(rows: list[dict]) -> dict:
    max_mm = max((r["pos_mm"] for r in rows), default=0.0)
    max_deg = max((r["rot_deg"] for r in rows), default=0.0)
    ok = max_mm < POS_TOL_MM and max_deg < ROT_TOL_DEG
    return {"max_mm": max_mm, "max_deg": max_deg, "ok": ok, "n": len(rows)}


def _print_rows(rows: list[dict], mode: str) -> None:
    print(f"\n  {mode} comparison (Pinocchio vs Realman):", flush=True)
    print("   idx |  pos err (mm) | rot err (deg)", flush=True)
    for r in rows:
        flag = "" if (r["pos_mm"] < POS_TOL_MM and r["rot_deg"] < ROT_TOL_DEG) else "  <-- FAIL"
        print(f"   {r['idx']:>3} | {r['pos_mm']:>11.4f} | {r['rot_deg']:>11.5f}{flag}", flush=True)


def compare_offline(npz_path: str, kin: RobotKinematics, frame: str) -> dict:
    data = np.load(npz_path)
    q_deg = np.asarray(data["q_deg"] if "q_deg" in data else data["joint"], dtype=float)
    pose = np.asarray(data["pose"], dtype=float)
    if q_deg.ndim == 1:
        q_deg = q_deg[None, :]
        pose = pose[None, :]
    rows = []
    for i in range(len(q_deg)):
        q = deg2rad(q_deg[i][:7])
        fk = kin.fk_pose(q) if frame == "tcp" else kin.frame_pose(q, frame)
        d_mm, d_deg = pose_diff(fk, pose[i][:6], kin.euler_order)
        rows.append({"idx": i, "pos_mm": d_mm, "rot_deg": d_deg})
    _print_rows(rows, f"offline[{frame}]")
    return _summary(rows)


def _read_state(robot) -> tuple[np.ndarray, np.ndarray]:
    ret, st = robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"rm_get_current_arm_state failed: {ret}")
    q_deg = np.asarray(st["joint"][:7], dtype=float)
    pose = np.asarray(st["pose"][:6], dtype=float)
    return q_deg, pose


def _read_tool_offset(robot) -> tuple[str, np.ndarray]:
    ret, tf = robot.rm_get_current_tool_frame()
    if ret != 0:
        raise RuntimeError(f"rm_get_current_tool_frame failed: {ret}")
    return str(tf.get("name", "?")), np.asarray(tf["pose"][:6], dtype=float)


def compare_once(robot, kin: RobotKinematics, mode: str, idx: int, *, verbose: bool = False) -> dict:
    q_deg, tool_pose = _read_state(robot)
    q = deg2rad(q_deg)
    row: dict = {"idx": idx, "q_deg": q_deg.tolist()}

    if mode == "tcp":
        fk = kin.fk_pose(q)
        d_mm, d_deg = pose_diff(fk, tool_pose, kin.euler_order)
        row.update(pos_mm=d_mm, rot_deg=d_deg)
    elif mode == "rm_fk":
        fk = kin.fk_pose(q)
        rm_fk = np.asarray(robot.rm_algo_forward_kinematics(q_deg.tolist(), flag=1)[:6], dtype=float)
        d_mm, d_deg = pose_diff(fk, rm_fk, kin.euler_order)
        row.update(pos_mm=d_mm, rot_deg=d_deg, rm_fk=rm_fk.tolist())
    else:  # flange
        _tool_name, tool_offset = _read_tool_offset(robot)
        flange_meas = base_flange_from_tool(tool_pose, tool_offset, kin.euler_order)
        fk = kin.frame_pose(q, "link_7")
        d_mm, d_deg = pose_diff(fk, flange_meas, kin.euler_order)
        row.update(pos_mm=d_mm, rot_deg=d_deg, flange_meas=flange_meas.tolist(), fk_link7=fk.tolist())
        if verbose:
            r_mat = Rsc.from_euler(kin.euler_order, fk[3:6], degrees=False).as_matrix()
            delta_base = np.asarray(flange_meas[:3], dtype=float) - np.asarray(fk[:3], dtype=float)
            delta_link7 = r_mat.T @ delta_base
            row["flange_delta_link7_mm"] = (delta_link7 * 1000.0).tolist()
            print(
                f"  [{idx}] flange offset in link_7 frame (mm): "
                f"{np.round(delta_link7 * 1000.0, 3).tolist()}  |Δ|={d_mm:.3f} mm",
                flush=True,
            )
    return row


def run_robot(args, kin: RobotKinematics) -> dict:
    from rm75_control.core.session import RobotSession

    modes = ["flange", "tcp", "rm_fk"] if args.all_modes else [args.mode]
    summaries: dict[str, dict] = {}

    with RobotSession(ip=args.ip, port=args.port) as sess:
        robot = sess.robot
        tool_name, tool_offset = _read_tool_offset(robot)
        print(f"  active Realman tool frame: {tool_name!r}  offset={np.round(tool_offset, 5).tolist()}", flush=True)

        for mode in modes:
            if mode == "tcp":
                print(
                    "  NOTE: --mode tcp compares Pinocchio tcp vs state.pose (active tool).",
                    flush=True,
                )
            if mode == "rm_fk":
                print(
                    "  NOTE: --mode rm_fk compares Pinocchio tcp vs rm_algo_forward_kinematics.",
                    flush=True,
                )

            rows: list[dict] = []
            if args.move and args.poses:
                targets = _load_pose_targets(args.poses)
                print(f"  driving {len(targets)} MoveJ points from {args.poses} [{mode}]", flush=True)
                for i, q_tgt in enumerate(targets):
                    sess.move_joints(q_tgt, velocity_percent=args.speed, block=1)
                    time.sleep(0.6)
                    rows.append(compare_once(robot, kin, mode, i, verbose=args.verbose))
            else:
                print(f"  read-only: comparing at the current configuration [{mode}]", flush=True)
                rows.append(compare_once(robot, kin, mode, 0, verbose=args.verbose))

            _print_rows(rows, f"robot[{mode}]")
            summaries[mode] = _summary(rows)

            if mode == "flange" and rows and not summaries[mode]["ok"]:
                deltas = [r.get("flange_delta_link7_mm") for r in rows if "flange_delta_link7_mm" in r]
                if deltas:
                    mean_mm = np.mean(np.asarray(deltas, dtype=float), axis=0)
                    print(
                        f"  mean flange offset pin->rm in link_7 frame (mm): "
                        f"{np.round(mean_mm, 3).tolist()}  |mean|={np.linalg.norm(mean_mm):.3f} mm",
                        flush=True,
                    )
                    print(
                        "  If |mean| is constant across poses, fix joint_7 origin y in the URDF "
                        "(vendor -172.5 mm vs Realman ~-161.2 mm).",
                        flush=True,
                    )

    if len(summaries) == 1:
        return next(iter(summaries.values()))
    ok = all(s["ok"] for s in summaries.values())
    max_mm = max(s["max_mm"] for s in summaries.values())
    max_deg = max(s["max_deg"] for s in summaries.values())
    n = sum(s["n"] for s in summaries.values())
    return {"max_mm": max_mm, "max_deg": max_deg, "ok": ok, "n": n, "by_mode": summaries}


def _load_pose_targets(poses_yaml: str) -> list[np.ndarray]:
    import yaml

    with open(poses_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    targets: list[np.ndarray] = []
    # Accept either {poses: {a: {q_deg: [...]}, ...}} or {slots: [...]} or a plain list.
    src = data.get("poses", data.get("slots", data))
    if isinstance(src, dict):
        for _k, rec in src.items():
            if isinstance(rec, dict) and "q_deg" in rec:
                targets.append(np.asarray(rec["q_deg"][:7], dtype=float))
    elif isinstance(src, list):
        for rec in src:
            if isinstance(rec, dict) and "q_deg" in rec:
                targets.append(np.asarray(rec["q_deg"][:7], dtype=float))
            elif isinstance(rec, (list, tuple)):
                targets.append(np.asarray(rec[:7], dtype=float))
    if not targets:
        raise SystemExit(f"no q_deg pose targets found in {poses_yaml}")
    return targets


def main() -> None:
    ap = argparse.ArgumentParser(description="Pinocchio-vs-Realman FK validation")
    ap.add_argument("--ip", default="192.168.1.18")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--mode", choices=["flange", "tcp", "rm_fk"], default="flange")
    ap.add_argument("--all-modes", action="store_true", help="run flange + tcp + rm_fk in one session")
    ap.add_argument("--verbose", action="store_true", help="print per-pose flange offset in link_7 frame")
    ap.add_argument("--poses", default=None, help="poses yaml with q_deg entries")
    ap.add_argument("--move", action="store_true", help="drive MoveJ to each pose (needs --poses)")
    ap.add_argument("--speed", type=int, default=20, help="MoveJ velocity percent")
    ap.add_argument("--urdf", default=None, help="override URDF path")
    ap.add_argument("--npz", default=None, help="offline: compare recorded q_deg/pose arrays")
    args = ap.parse_args()

    kin = RobotKinematics(urdf_path=args.urdf)
    print(f"Loaded URDF: {kin.urdf_path}", flush=True)

    if args.npz:
        frame = "tcp" if args.mode == "tcp" else "link_7"
        summ = compare_offline(args.npz, kin, frame)
    else:
        summ = run_robot(args, kin)

    print(
        f"\n  RESULT: max pos {summ['max_mm']:.4f} mm | max rot {summ['max_deg']:.5f} deg "
        f"over {summ['n']} pose(s)  ->  {'PASS' if summ['ok'] else 'FAIL'}"
        f"  (tol {POS_TOL_MM} mm / {ROT_TOL_DEG} deg)",
        flush=True,
    )
    if not summ["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

## FILE: `rm75_control/control/joint_admittance/loop.py`

```py
"""Joint-space inner loop: Cartesian twist -> absolute joint angles (rm_movej_canfd).

Two layers:

* ``JointIkController`` - the reusable, hardware-free inner loop.  Given the last
  commanded joint state and a Cartesian twist, it runs CLIK (DLS + nullspace),
  integrates, smooths and safety-clamps, and returns the next joint command.
  This is what the offline sim validation exercises.

* ``run_joint_admittance_loop`` - the on-robot orchestration.  It reuses the
  task-space admittance outer loop (hybrid_motion.AdmittanceController) plus the
  compensated force observer and the async state reader, feeds the outer twist
  into ``JointIkController`` every tick, and streams the result through
  ``rm_movej_canfd`` on an absolute perf_counter schedule.  It NEVER calls a
  MoveV/MoveJ mode switch during the loop - the only motion interface is the
  joint CANFD passthrough (a single ``move_j`` positions the arm at start).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.joint_admittance.clik import ClikConfig, ClikController
from rm75_control.control.joint_admittance.model import (
    RobotKinematics,
    deg2rad,
    pose_distance,
    pose_error,
    rad2deg,
)
from rm75_control.control.joint_admittance.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance.utils.friction import (
    FrictionCompensator,
    FrictionConfig,
)
from rm75_control.control.joint_admittance.utils.safety import (
    SafetyLimiter,
    SafetyLimits,
    Watchdog,
)
from rm75_control.control.joint_admittance.utils.smoothing import SecondOrderLowPass


# ---------------------------------------------------------------------------
# Inner loop (hardware-free)
# ---------------------------------------------------------------------------
@dataclass
class JointIkConfig:
    dt: float = 0.01
    control_frame: str = "base"        # frame the incoming twist is expressed in
    euler_order: str = "xyz"
    solver: str = "clik"               # "clik" (Phase 1 DLS) | "qp" (Phase 2 ProxQP)
    clik: ClikConfig = field(default_factory=ClikConfig)
    qp: "QpConfig | None" = None       # only used when solver == "qp"
    nullspace: NullspaceTaskConfig = field(default_factory=NullspaceTaskConfig)
    # safety
    v_scale: float = 0.5               # fraction of URDF joint velocity limit allowed
    a_max: float = 20.0                # rad/s^2 acceleration clamp (per joint)
    position_margin_rad: float = 0.017
    # smoothing
    use_smoothing: bool = True
    smooth_cutoff_hz: float = 15.0
    # optional Phase 3 friction feed-forward (default off)
    friction: FrictionConfig = field(default_factory=FrictionConfig)


@dataclass
class JointIkStep:
    q_send: np.ndarray          # commanded joint position (rad) after smooth + clamp
    qdot: np.ndarray            # CLIK joint velocity (rad/s)
    twist_base: np.ndarray      # twist actually applied (base frame)
    sigma_min: float
    lam: float
    manip: float
    cart_err_mm: float
    vel_clamped: bool
    acc_clamped: bool
    pos_clamped: bool


class JointIkController:
    """Reusable inner loop: (q_prev, twist) -> next joint command, all in rad."""

    def __init__(self, kin: RobotKinematics, cfg: JointIkConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or JointIkConfig()
        self.cfg.clik.euler_order = self.cfg.euler_order
        self.task = JointCenteringTask.from_kinematics(kin, self.cfg.nullspace)
        self.limits = SafetyLimits.from_kinematics(
            kin,
            v_scale=self.cfg.v_scale,
            a_max=self.cfg.a_max,
            position_margin=self.cfg.position_margin_rad,
        )
        self.core = self._make_core()
        # backward-compatible alias (tests / diagnostics read `.clik.x_ref`)
        self.clik = self.core
        self.safety = SafetyLimiter(self.limits)
        self.smoother = (
            SecondOrderLowPass(self.cfg.smooth_cutoff_hz, self.cfg.dt, dim=kin.nv)
            if self.cfg.use_smoothing
            else None
        )
        self.friction = FrictionCompensator(self.cfg.friction)
        self.q_cmd = np.zeros(kin.nv, dtype=float)

    def _make_core(self):
        if self.cfg.solver == "qp":
            from rm75_control.control.joint_admittance.solver.qp_builder import (
                QpConfig,
                QpIkController,
            )

            qp_cfg = self.cfg.qp or QpConfig()
            qp_cfg.euler_order = self.cfg.euler_order
            self.cfg.qp = qp_cfg
            return QpIkController(self.kin, self.limits, qp_cfg)
        return ClikController(self.kin, self.cfg.clik)

    def reset(self, q0_rad: np.ndarray, pose0: np.ndarray | None = None) -> None:
        self.q_cmd = np.asarray(q0_rad, dtype=float).copy()
        self.core.reset(self.q_cmd, pose0)
        self.safety.reset(self.q_cmd)
        if self.smoother is not None:
            self.smoother.reset(self.q_cmd)

    def _twist_to_base(self, twist: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
        twist = np.asarray(twist, dtype=float)
        if self.cfg.control_frame != "tool":
            return twist
        # rotate tool-frame twist into base frame using the current TCP orientation
        R = self.kin.fk_placement(q_prev).rotation
        out = np.zeros(6, dtype=float)
        out[:3] = R @ twist[:3]
        out[3:6] = R @ twist[3:6]
        return out

    def update(self, twist: np.ndarray, dt: float | None = None) -> JointIkStep:
        dt = self.cfg.dt if dt is None else dt
        q_prev = self.q_cmd
        twist_base = self._twist_to_base(twist, q_prev)

        secondary = self.task(q_prev)
        r = self.core.step(q_prev, twist_base, dt, secondary_qdot=secondary)

        q_target = r.q_next
        if self.cfg.friction.enabled:
            q_target = q_target + self.friction(r.qdot) * dt
        if self.smoother is not None:
            q_target = self.smoother(q_target)

        rep = self.safety.clamp(q_prev, q_target, dt)
        if self.smoother is not None:
            self.smoother.sync(rep.q_safe)

        self.q_cmd = rep.q_safe
        return JointIkStep(
            q_send=rep.q_safe.copy(),
            qdot=r.qdot,
            twist_base=twist_base,
            sigma_min=r.sigma_min,
            lam=r.lam,
            manip=r.manip,
            cart_err_mm=float(np.linalg.norm(r.cart_err[:3]) * 1000.0),
            vel_clamped=rep.vel_clamped,
            acc_clamped=rep.acc_clamped,
            pos_clamped=rep.pos_clamped,
        )


# ---------------------------------------------------------------------------
# Outer loop adapter
# ---------------------------------------------------------------------------
class OuterLoop(Protocol):
    """Task-space controller producing a Cartesian twist each tick."""

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        """Return a 6D twist in the inner loop's control_frame."""
        ...


class AdmittanceOuterLoop:
    """Wrap hybrid_motion.AdmittanceController + a MotionReferenceSource.

    Emits the same 6D Cartesian twist the velocity stack streamed to MoveV - here
    it feeds the joint-space inner loop instead.  ``control_frame`` matches the
    AdmittanceController config (tool by default).
    """

    def __init__(self, controller, reference_source, *, desired_force: np.ndarray | None = None):
        self.controller = controller
        self.reference = reference_source
        self.desired_force = (
            np.zeros(6) if desired_force is None else np.asarray(desired_force, dtype=float)
        )

    def set_origin(self, pose0: np.ndarray) -> None:
        if hasattr(self.reference, "set_origin"):
            self.reference.set_origin(pose0)

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        ref = self.reference.sample(t_s)
        return self.controller.compute_velocity_command(
            current_pose,
            ref.pose_d,
            ref.vel_ff,
            f_ext,
            self.desired_force,
        )


@dataclass
class CartesianTrackConfig:
    """Pure position/orientation tracking, no force axis - used to reposition the
    arm (e.g. walk to a pose slot) through the SAME inner IK loop, so a Cartesian
    move and an admittance scan can be sequenced without any MoveJ/MoveV switch."""

    k_task: np.ndarray = field(default_factory=lambda: np.full(6, 1.5))
    max_lin_vel_m_s: float = 0.08
    max_ang_vel_rad_s: float = 0.5
    euler_order: str = "xyz"
    # MUST match the consuming JointIkConfig.control_frame. MotionReference.pose_d /
    # vel_ff are always base-frame (see reference.py); this outer loop computes the
    # PD+feedforward twist in base frame and then, if control_frame=="tool", rotates
    # it INTO tool-axis coordinates (R^T @ v) before returning - because the inner
    # loop's _twist_to_base() rotates a "tool" twist back OUT with R @ twist. Return
    # a base-frame vector here when control_frame=="tool" and you get a silent
    # double rotation: the command points along the wrong axes and the loop diverges
    # instead of converging (this bit us: cart_err growing to 100s of mm).
    control_frame: str = "base"


class CartesianTrackOuterLoop:
    """Generic Cartesian-trajectory-tracking outer loop (no force).

    Wraps any MotionReferenceSource (e.g. CartesianMoveReference for a point-to-
    point walk, or SinToolYReference for a scan without force control) and turns
    its (pose_d, vel_ff) into a twist via simple PD + feedforward, clamped to safe
    Cartesian rates.  Use AdmittanceOuterLoop instead when you want force control.

    The returned twist is expressed in ``cfg.control_frame`` - set it to match the
    ``JointIkConfig.control_frame`` of the inner loop this feeds (see CartesianTrackConfig).
    """

    def __init__(self, reference, cfg: CartesianTrackConfig | None = None) -> None:
        self.reference = reference
        self.cfg = cfg or CartesianTrackConfig()

    def set_origin(self, pose0: np.ndarray) -> None:
        if hasattr(self.reference, "set_origin"):
            self.reference.set_origin(pose0)

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        del f_ext
        cfg = self.cfg
        ref = self.reference.sample(t_s)
        err = pose_error(ref.pose_d, current_pose, cfg.euler_order)
        v = np.asarray(ref.vel_ff, dtype=float) + cfg.k_task * err  # base-frame twist

        lin_n = float(np.linalg.norm(v[:3]))
        if cfg.max_lin_vel_m_s > 0.0 and lin_n > cfg.max_lin_vel_m_s:
            v[:3] *= cfg.max_lin_vel_m_s / lin_n
        ang_n = float(np.linalg.norm(v[3:6]))
        if cfg.max_ang_vel_rad_s > 0.0 and ang_n > cfg.max_ang_vel_rad_s:
            v[3:6] *= cfg.max_ang_vel_rad_s / ang_n

        if cfg.control_frame == "tool":
            R = Rsc.from_euler(cfg.euler_order, current_pose[3:6], degrees=False).as_matrix()
            out = np.zeros(6, dtype=float)
            out[:3] = R.T @ v[:3]
            out[3:6] = R.T @ v[3:6]
            return out
        return v


def arrived(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    tol_mm: float = 1.0,
    tol_deg: float = 0.5,
    euler_order: str = "xyz",
) -> bool:
    """Convenience Phase.wait_until predicate: True once within tolerance of target."""
    d_mm, d_deg = pose_distance(current_pose, target_pose, euler_order)
    return d_mm <= tol_mm and d_deg <= tol_deg


# ---------------------------------------------------------------------------
# On-robot orchestration
# ---------------------------------------------------------------------------
def _set_realtime_priority(priority: int = 80) -> bool:
    """Best-effort SCHED_FIFO for the control thread (needs CAP_SYS_NICE / root)."""
    try:
        param = os.sched_param(priority)
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
        return True
    except (PermissionError, OSError, AttributeError):
        return False


@dataclass
class LoopResult:
    ticks: int
    duration_s: float
    max_jitter_ms: float
    stalled: bool


@dataclass
class Phase:
    """One leg of a multi-phase on-robot run, e.g. "walk to D" then "sin scan at D".

    All phases share the SAME inner loop, async state reader and watchdog - there
    is no MoveJ/MoveV switch and no gap in the joint-command stream at the phase
    boundary, only ``outer`` (and optionally ``force_observer``) changing.
    """

    outer: OuterLoop
    label: str = ""
    duration_s: float | None = None          # None -> run until wait_until (or max_duration_s)
    max_duration_s: float | None = None      # safety cap when duration_s is None
    wait_until: object | None = None         # Callable[[np.ndarray], bool] on current pose
    force_observer: object | None = None     # None -> reuse the loop-level force_observer
    on_enter: object | None = None           # Callable[[], None], fired right after set_origin


def run_joint_admittance_phases(
    session,
    phases: list[Phase],
    inner: JointIkController,
    *,
    q_start_deg: np.ndarray | None = None,
    dt: float | None = None,
    force_observer=None,
    follow: bool = True,
    move_speed: int = 20,
    realtime: bool = False,
    watchdog_timeout_s: float = 0.1,
    on_step=None,
    verbose: bool = True,
) -> LoopResult:
    """Run a sequence of ``Phase`` objects on the real robot, one continuous stream.

    Sequence:
      1. move_j to q_start (single planned motion; the only non-CANFD command).
      2. Start the async state reader; read q0 and reset the inner loop at it.
      3. For each phase, at fixed dt (perf_counter absolute schedule):
           outer.sample(t_phase, pose, f_ext) -> inner.update(q) -> rm_movej_canfd.
         A phase ends when t_phase >= duration_s, or wait_until(pose) is True, or
         (if duration_s is None) t_phase >= max_duration_s.
    """
    from rm75_control.control.hybrid_motion.async_state import AsyncStateObserver
    from rm75_control.motion.canfd import send_joint_canfd

    dt = inner.cfg.dt if dt is None else dt
    robot = session.robot

    if q_start_deg is not None:
        session.move_joints(list(np.asarray(q_start_deg, dtype=float)), velocity_percent=move_speed, block=1)
        time.sleep(0.5)

    async_obs = AsyncStateObserver(robot, poll_s=dt)
    async_obs.start()
    ticks = 0
    max_jitter_ms = 0.0
    stalled = False
    total_t0 = time.perf_counter()
    try:
        pose0 = async_obs.wait_first_pose(timeout_s=5.0)
        snap0 = async_obs.read()
        if snap0.q_deg is None:
            raise RuntimeError("no joint feedback from robot")
        q0_rad = deg2rad(snap0.q_deg)
        inner.reset(q0_rad, pose0)

        if realtime and not _set_realtime_priority():
            if verbose:
                print("  (SCHED_FIFO unavailable - running at normal priority)", flush=True)

        def _hold() -> None:
            # watchdog stall action: hold at the last commanded joint state
            try:
                send_joint_canfd(robot, rad2deg(inner.q_cmd), follow=False)
            except Exception:
                try:
                    robot.rm_set_arm_slow_stop()
                except Exception:
                    pass

        wd = Watchdog(watchdog_timeout_s, _hold)
        wd.start()
        try:
            pose = pose0
            for phase in phases:
                if verbose:
                    print(f"-- phase: {phase.label or phase.outer.__class__.__name__} --", flush=True)
                if hasattr(phase.outer, "set_origin"):
                    phase.outer.set_origin(pose)
                if phase.on_enter is not None:
                    phase.on_enter()

                obs = phase.force_observer if phase.force_observer is not None else force_observer
                phase_t0 = time.perf_counter()
                next_tick = phase_t0
                while True:
                    now = time.perf_counter()
                    t_phase = now - phase_t0
                    if phase.duration_s is not None and t_phase >= phase.duration_s:
                        break
                    if phase.max_duration_s is not None and t_phase >= phase.max_duration_s:
                        break
                    jitter_ms = abs(now - next_tick) * 1000.0
                    max_jitter_ms = max(max_jitter_ms, jitter_ms)

                    snap = async_obs.read()
                    pose = snap.pose if snap.pose is not None else pose
                    f_ext = np.zeros(6)
                    if obs is not None:
                        _signed, f_ext = obs.update(now - total_t0, pose, snap.force_raw)

                    twist = np.asarray(phase.outer.sample(t_phase, pose, f_ext), dtype=float)
                    step = inner.update(twist, dt)
                    send_joint_canfd(robot, rad2deg(step.q_send), follow=follow)
                    wd.beat()

                    if on_step is not None:
                        on_step(phase.label, t_phase, step, pose, f_ext)

                    if phase.wait_until is not None and phase.wait_until(pose):
                        break

                    ticks += 1
                    next_tick += dt
                    sleep_for = next_tick - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
        finally:
            wd.stop()
            stalled = wd.fired
    finally:
        async_obs.stop()

    total_s = time.perf_counter() - total_t0
    if verbose:
        print(
            f"  joint-admittance loop: {ticks} ticks, {total_s:.1f}s, "
            f"max jitter {max_jitter_ms:.2f} ms{' [WATCHDOG FIRED]' if stalled else ''}",
            flush=True,
        )
    return LoopResult(ticks=ticks, duration_s=total_s, max_jitter_ms=max_jitter_ms, stalled=stalled)


def run_joint_admittance_loop(
    session,
    outer: OuterLoop,
    inner: JointIkController,
    *,
    q_start_deg: np.ndarray | None = None,
    duration_s: float = 10.0,
    dt: float | None = None,
    force_observer=None,
    follow: bool = True,
    move_speed: int = 20,
    realtime: bool = False,
    watchdog_timeout_s: float = 0.1,
    on_step=None,
    verbose: bool = True,
) -> LoopResult:
    """Single-phase convenience wrapper around ``run_joint_admittance_phases``."""
    phase = Phase(outer=outer, label="run", duration_s=duration_s)
    on_step_1 = None if on_step is None else (lambda label, t, step, pose, f_ext: on_step(t, step, pose, f_ext))
    return run_joint_admittance_phases(
        session,
        [phase],
        inner,
        q_start_deg=q_start_deg,
        dt=dt,
        force_observer=force_observer,
        follow=follow,
        move_speed=move_speed,
        realtime=realtime,
        watchdog_timeout_s=watchdog_timeout_s,
        on_step=on_step_1,
        verbose=verbose,
    )
```

## FILE: `rm75_control/control/joint_admittance/tasks/__init__.py`

```py
"""Secondary / priority tasks for the joint-space inner loop."""
```

## FILE: `rm75_control/control/joint_admittance/tasks/nullspace_task.py`

```py
"""Nullspace secondary task: joint centering + limit avoidance (Liegeois 1977).

Produces a desired joint velocity `qdot0` that the CLIK/QP core projects into the
nullspace of the primary Cartesian task, so it never perturbs TCP tracking.  It
uses the redundancy of the 7-DOF arm to (a) pull joints toward the middle of
their range and (b) repel them harder as they approach a limit.

The cost being descended is the classic Liegeois manipulability/limit criterion
    H(q) = 1/2 * sum_i w_i * ((q_i - q_mid_i) / half_range_i)^2
    qdot0 = -k * dH/dq
plus a smooth activation term that grows near the limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance.model import RobotKinematics


@dataclass
class NullspaceTaskConfig:
    k_center: float = 1.0        # centering velocity gain (rad/s per normalized unit)
    k_limit: float = 2.0         # extra repulsion gain near a limit
    activation: float = 0.8      # |u| beyond which limit repulsion ramps in (u in [-1,1])
    weights: np.ndarray | None = None   # optional per-joint weighting (len 7)


class JointCenteringTask:
    """Callable secondary task: q (rad) -> qdot0 (rad/s)."""

    def __init__(
        self,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        cfg: NullspaceTaskConfig | None = None,
    ) -> None:
        self.q_lower = np.asarray(q_lower, dtype=float)
        self.q_upper = np.asarray(q_upper, dtype=float)
        self.cfg = cfg or NullspaceTaskConfig()
        self.q_mid = 0.5 * (self.q_lower + self.q_upper)
        self.half = 0.5 * (self.q_upper - self.q_lower)
        # guard against zero-range joints
        self.half = np.where(self.half > 1e-9, self.half, 1.0)
        self.w = (
            np.ones_like(self.q_mid)
            if self.cfg.weights is None
            else np.asarray(self.cfg.weights, dtype=float)
        )

    @classmethod
    def from_kinematics(
        cls, kin: RobotKinematics, cfg: NullspaceTaskConfig | None = None
    ) -> "JointCenteringTask":
        return cls(kin.q_lower, kin.q_upper, cfg)

    def __call__(self, q_rad: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        q = np.asarray(q_rad, dtype=float)
        u = (q - self.q_mid) / self.half              # normalized position in [-1, 1]

        # gradient-descent centering: -k * w * u
        qdot0 = -cfg.k_center * self.w * u

        # smooth limit repulsion beyond activation band
        if cfg.k_limit > 0.0 and cfg.activation < 1.0:
            span = max(1.0 - cfg.activation, 1e-6)
            over = np.clip((np.abs(u) - cfg.activation) / span, 0.0, 1.0)
            qdot0 = qdot0 - cfg.k_limit * np.sign(u) * (over * over)
        return qdot0
```

## FILE: `rm75_control/control/joint_admittance/solver/__init__.py`

```py
"""Phase 2 QP inner-loop solver (ProxQP preferred, OSQP fallback)."""
```

## FILE: `rm75_control/control/joint_admittance/solver/constraint_mgr.py`

```py
"""Per-tick box constraints on joint velocity for the QP inner loop.

Everything the safety layer enforces after the fact is expressed here as *hard*
inequality bounds on the decision variable qdot, so the QP respects them while
optimizing (rather than clipping an already-computed solution):

    qdot in [-v_max, v_max]                              (velocity)
    qdot in [(q_min+m - q)/dt, (q_max-m - q)/dt]         (position, look-ahead)
    qdot in [qdot_prev - a_max*dt, qdot_prev + a_max*dt] (acceleration)

The three boxes are intersected.  If a position bound and another bound cross
(the joint is already past a limit), the position bound wins and both l and u
collapse onto it, which drives the joint back inside next tick.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance.utils.safety import SafetyLimits


class VelocityBoxConstraints:
    def __init__(self, limits: SafetyLimits) -> None:
        self.lim = limits

    def bounds(
        self,
        q: np.ndarray,
        dt: float,
        qdot_prev: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        lim = self.lim
        q = np.asarray(q, dtype=float)

        lo = -lim.v_max.copy()
        hi = lim.v_max.copy()

        # position look-ahead
        m = lim.position_margin
        p_lo = (lim.q_lower + m - q) / dt
        p_hi = (lim.q_upper - m - q) / dt
        lo = np.maximum(lo, p_lo)
        hi = np.minimum(hi, p_hi)

        # acceleration
        if lim.a_max is not None and qdot_prev is not None:
            qdot_prev = np.asarray(qdot_prev, dtype=float)
            a = lim.a_max * dt
            lo = np.maximum(lo, qdot_prev - a)
            hi = np.minimum(hi, qdot_prev + a)

        # resolve crossings: position limit dominates
        crossed = lo > hi
        if np.any(crossed):
            mid = np.clip(0.0, p_lo, p_hi)  # bias toward staying in position range
            lo = np.where(crossed, np.minimum(p_lo, p_hi), lo)
            hi = np.where(crossed, np.maximum(p_lo, p_hi), hi)
            # if still crossed after using position bounds only, collapse to feasible point
            still = lo > hi
            if np.any(still):
                lo = np.where(still, mid, lo)
                hi = np.where(still, mid, hi)
        return lo, hi
```

## FILE: `rm75_control/control/joint_admittance/solver/qp_builder.py`

```py
"""QP-based inverse kinematics core (Phase 2), a drop-in for ClikController.

Solves, every tick, the box-constrained velocity-IK QP

    min_qdot  0.5 * || J qdot - v_task ||^2_{W} + 0.5 * reg * || qdot - qdot_0 ||^2
    s.t.      l <= qdot <= u          (velocity / position look-ahead / accel)

where v_task = twist_ref + K * e_x (the same CLIK closed-loop feedback as
Phase 1) and qdot_0 is the secondary (centering) task.  Redundancy is resolved
implicitly: the reg term pulls qdot toward qdot_0 in the directions the task
leaves free, so no explicit nullspace projection is needed, and all limits are
respected *as hard constraints* rather than post-hoc clipping.

Backends: ProxQP (preferred, warm-started -> sub-ms) with an OSQP fallback.
The ``step`` signature and returned ClikResult match ClikController, so
JointIkController can swap cores without any other change.

References: Escande/Kanoun task-priority QP; Diehl QP methods; Sentis WBC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rm75_control.control.joint_admittance.clik import (
    ClikResult,
    _saturate_error,
    integrate_pose,
)
from rm75_control.control.joint_admittance.model import RobotKinematics, pose_error
from rm75_control.control.joint_admittance.solver.constraint_mgr import VelocityBoxConstraints
from rm75_control.control.joint_admittance.utils.safety import SafetyLimits


@dataclass
class QpConfig:
    k_task: np.ndarray = field(default_factory=lambda: np.full(6, 2.0))
    task_weight: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]))
    reg: float = 1e-3               # regularization / secondary-task weight
    reg_secondary_scale: float = 1.0  # scales qdot_0 contribution to g
    max_pos_err_m: float = 0.05
    max_rot_err_rad: float = 0.20
    euler_order: str = "xyz"
    backend: str = "proxqp"         # "proxqp" | "osqp"
    eps_abs: float = 1e-6
    max_iter: int = 200
    sigma_thresh: float = 0.04      # kept only for diagnostics / manip reporting


class _ProxQpBackend:
    def __init__(self, n: int, cfg: QpConfig) -> None:
        import proxsuite

        self._px = proxsuite
        self.n = n
        # n_eq = 0, n_in = n (box via C = I)
        self.qp = proxsuite.proxqp.dense.QP(n, 0, n)
        self.C = np.eye(n)
        self.qp.settings.eps_abs = cfg.eps_abs
        self.qp.settings.max_iter = cfg.max_iter
        self.qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        self._initialized = False

    def solve(self, H, g, lo, hi):
        if not self._initialized:
            self.qp.init(H, g, None, None, self.C, lo, hi)
            self._initialized = True
        else:
            self.qp.update(H=H, g=g, C=self.C, l=lo, u=hi)
        self.qp.solve()
        return np.asarray(self.qp.results.x, dtype=float)


class _OsqpBackend:
    def __init__(self, n: int, cfg: QpConfig) -> None:
        import osqp
        import scipy.sparse as sp

        self._osqp = osqp
        self._sp = sp
        self.n = n
        self.cfg = cfg
        self.A = sp.identity(n, format="csc")
        self.prob = None

    def solve(self, H, g, lo, hi):
        sp = self._sp
        P = sp.csc_matrix(np.triu(H))
        if self.prob is None:
            self.prob = self._osqp.OSQP()
            self.prob.setup(
                P, g, self.A, lo, hi,
                verbose=False, warm_start=True,
                eps_abs=self.cfg.eps_abs, eps_rel=self.cfg.eps_abs,
                max_iter=self.cfg.max_iter,
            )
        else:
            self.prob.update(Px=P.data, q=g, l=lo, u=hi)
        res = self.prob.solve()
        if res.x is None or np.any(np.isnan(res.x)):
            return np.zeros(self.n)
        return np.asarray(res.x, dtype=float)


class QpIkController:
    """QP velocity-IK core with the same interface as ClikController."""

    def __init__(
        self,
        kin: RobotKinematics,
        limits: SafetyLimits,
        cfg: QpConfig | None = None,
    ) -> None:
        self.kin = kin
        self.cfg = cfg or QpConfig()
        self.constraints = VelocityBoxConstraints(limits)
        self.x_ref = np.zeros(6, dtype=float)
        self.qdot_prev = np.zeros(kin.nv, dtype=float)
        self._initialized = False
        self.backend = self._make_backend(kin.nv)

    def _make_backend(self, n: int):
        want = self.cfg.backend.lower()
        if want == "proxqp":
            try:
                return _ProxQpBackend(n, self.cfg)
            except Exception:
                pass
        if want in ("osqp", "proxqp"):
            try:
                return _OsqpBackend(n, self.cfg)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "No QP backend available (install proxsuite or osqp)"
                ) from exc
        raise ValueError(f"unknown QP backend {self.cfg.backend!r}")

    @property
    def backend_name(self) -> str:
        return type(self.backend).__name__.replace("_", "").replace("Backend", "").lower()

    def reset(self, q0_rad: np.ndarray, pose0: np.ndarray | None = None) -> None:
        self.x_ref = (
            np.asarray(pose0, dtype=float).copy()
            if pose0 is not None
            else self.kin.fk_pose(q0_rad)
        )
        self.qdot_prev = np.zeros(self.kin.nv, dtype=float)
        self._initialized = True

    def set_reference(self, pose: np.ndarray) -> None:
        self.x_ref = np.asarray(pose, dtype=float).copy()

    def step(
        self,
        q_prev: np.ndarray,
        twist_ref: np.ndarray,
        dt: float,
        secondary_qdot: np.ndarray | None = None,
    ) -> ClikResult:
        cfg = self.cfg
        q_prev = np.asarray(q_prev, dtype=float)
        twist_ref = np.asarray(twist_ref, dtype=float)
        if not self._initialized:
            self.reset(q_prev)

        # advance integrated Cartesian reference
        self.x_ref = integrate_pose(self.x_ref, twist_ref, dt, cfg.euler_order)

        J = self.kin.jacobian(q_prev)
        sigma = self.kin.singular_values(J)
        sigma_min = float(sigma.min())

        x_cur = self.kin.fk_pose(q_prev)
        err = pose_error(self.x_ref, x_cur, cfg.euler_order)
        err_sat = _saturate_error(err, cfg.max_pos_err_m, cfg.max_rot_err_rad)
        v_task = twist_ref + cfg.k_task * err_sat

        W = np.diag(cfg.task_weight)
        H = J.T @ W @ J + cfg.reg * np.eye(self.kin.nv)
        # symmetrize for solver numerics
        H = 0.5 * (H + H.T)
        g = -(J.T @ (W @ v_task))
        if secondary_qdot is not None:
            g = g - cfg.reg * cfg.reg_secondary_scale * np.asarray(secondary_qdot, dtype=float)

        lo, hi = self.constraints.bounds(q_prev, dt, self.qdot_prev)
        qdot = self.backend.solve(
            np.ascontiguousarray(H),
            np.ascontiguousarray(g),
            np.ascontiguousarray(lo),
            np.ascontiguousarray(hi),
        )
        self.qdot_prev = qdot
        q_next = q_prev + qdot * dt
        return ClikResult(
            q_next=q_next,
            qdot=qdot,
            x_ref=self.x_ref.copy(),
            x_cur=x_cur,
            cart_err=err,
            sigma_min=sigma_min,
            lam=cfg.reg,
            manip=self.kin.manipulability(J),
        )
```

## FILE: `rm75_control/control/joint_admittance/utils/__init__.py`

```py
"""Utilities for the joint-space inner loop (smoothing, safety, watchdog)."""
```

## FILE: `rm75_control/control/joint_admittance/utils/safety.py`

```py
"""Safety layer for direct joint-position streaming.

When you bypass MoveJ's built-in S-curve planner and push q_cmd straight into
rm_movej_canfd, the motor drivers will fault (over-current / following error) on
any discontinuity.  This module enforces, per tick, in order:

  1. velocity limit : |dq| <= v_max * dt          (per-frame dq clamp)
  2. acceleration   : |dq - dq_prev| <= a_max*dt^2 (jerk-free enough for CANFD)
  3. position limit : q in [q_lower+margin, q_upper-margin]

plus a Watchdog thread that trips (freeze / slow-stop) if the control loop stops
feeding heartbeats - so a stuck Python process can never leave the arm coasting.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class SafetyLimits:
    q_lower: np.ndarray
    q_upper: np.ndarray
    v_max: np.ndarray                       # rad/s (per joint)
    a_max: np.ndarray | None = None         # rad/s^2 (per joint); None disables accel clamp
    position_margin: float = 0.017          # ~1 deg back-off from hard limit

    @classmethod
    def from_kinematics(
        cls,
        kin,
        *,
        v_scale: float = 1.0,
        a_max: np.ndarray | float | None = None,
        position_margin: float = 0.017,
    ) -> "SafetyLimits":
        v_max = np.asarray(kin.v_max, dtype=float) * float(v_scale)
        if a_max is not None and np.isscalar(a_max):
            a_max = np.full_like(v_max, float(a_max))
        return cls(
            q_lower=np.asarray(kin.q_lower, dtype=float),
            q_upper=np.asarray(kin.q_upper, dtype=float),
            v_max=v_max,
            a_max=None if a_max is None else np.asarray(a_max, dtype=float),
            position_margin=position_margin,
        )


@dataclass
class SafetyReport:
    q_safe: np.ndarray
    dq: np.ndarray
    vel_clamped: bool = False
    acc_clamped: bool = False
    pos_clamped: bool = False


class SafetyLimiter:
    """Stateful per-tick clamp: velocity -> acceleration -> position."""

    def __init__(self, limits: SafetyLimits) -> None:
        self.lim = limits
        self._dq_prev: np.ndarray | None = None

    def reset(self, q0: np.ndarray | None = None) -> None:
        self._dq_prev = None

    def clamp(self, q_prev: np.ndarray, q_desired: np.ndarray, dt: float) -> SafetyReport:
        lim = self.lim
        q_prev = np.asarray(q_prev, dtype=float)
        q_desired = np.asarray(q_desired, dtype=float)
        dq = q_desired - q_prev

        vel_clamped = acc_clamped = pos_clamped = False

        # 1) velocity limit
        dq_max = lim.v_max * dt
        clipped = np.clip(dq, -dq_max, dq_max)
        if not np.allclose(clipped, dq):
            vel_clamped = True
        dq = clipped

        # 2) acceleration limit (change in dq between ticks)
        if lim.a_max is not None and self._dq_prev is not None:
            ddq_max = lim.a_max * dt * dt
            ddq = dq - self._dq_prev
            ddq_c = np.clip(ddq, -ddq_max, ddq_max)
            if not np.allclose(ddq_c, ddq):
                acc_clamped = True
            dq = self._dq_prev + ddq_c

        q_safe = q_prev + dq

        # 3) position limit
        lo = lim.q_lower + lim.position_margin
        hi = lim.q_upper - lim.position_margin
        q_clamped = np.clip(q_safe, lo, hi)
        if not np.allclose(q_clamped, q_safe):
            pos_clamped = True
            dq = q_clamped - q_prev
        q_safe = q_clamped

        self._dq_prev = dq
        return SafetyReport(
            q_safe=q_safe,
            dq=dq,
            vel_clamped=vel_clamped,
            acc_clamped=acc_clamped,
            pos_clamped=pos_clamped,
        )


class Watchdog:
    """Independent heartbeat monitor.

    The control loop calls `beat()` every tick.  If no beat arrives within
    `timeout_s`, the watchdog fires `on_stall` exactly once (e.g. slow-stop the
    arm / latch a hold).  Runs as a daemon thread so it survives a stuck loop.
    """

    def __init__(
        self,
        timeout_s: float,
        on_stall: Callable[[], None],
        *,
        poll_s: float = 0.005,
        name: str = "ja-watchdog",
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.on_stall = on_stall
        self.poll_s = float(poll_s)
        self._name = name
        self._last_beat = time.perf_counter()
        self._stop = threading.Event()
        self._fired = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def beat(self) -> None:
        with self._lock:
            self._last_beat = time.perf_counter()
            # allow re-arming after a transient recovery
            self._fired.clear()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._last_beat = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def fired(self) -> bool:
        return self._fired.is_set()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                dt = time.perf_counter() - self._last_beat
            if dt > self.timeout_s and not self._fired.is_set():
                self._fired.set()
                try:
                    self.on_stall()
                except Exception:
                    pass
            time.sleep(self.poll_s)

    def __enter__(self) -> "Watchdog":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
```

## FILE: `rm75_control/control/joint_admittance/utils/smoothing.py`

```py
"""Command smoothing for direct joint-position streaming.

Even a mathematically continuous q_cmd, sampled at 100-1000 Hz, carries
high-frequency content that shows up as motor current ripple.  These filters
smooth q_cmd before it hits rm_movej_canfd.  All are per-joint and stateful.

Provided:
* FirstOrderLowPass    - single-pole IIR (cheap, ~6 dB/oct).
* SecondOrderLowPass   - critically-damped 2nd order (S-curve-like step response,
                         no overshoot; ~12 dB/oct).
* MovingAverage        - boxcar window.

Cutoff is specified in Hz; alpha is derived from the control dt.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def alpha_from_cutoff(cutoff_hz: float, dt: float) -> float:
    """First-order IIR smoothing factor for a given cutoff and sample period."""
    if cutoff_hz <= 0.0:
        return 1.0  # no filtering
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return float(dt / (tau + dt))


class FirstOrderLowPass:
    def __init__(self, cutoff_hz: float, dt: float, dim: int = 7) -> None:
        self.alpha = alpha_from_cutoff(cutoff_hz, dt)
        self.dim = dim
        self._y: np.ndarray | None = None

    def reset(self, x0: np.ndarray | None = None) -> None:
        self._y = None if x0 is None else np.asarray(x0, dtype=float).copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._y is None:
            self._y = x.copy()
        else:
            self._y = self._y + self.alpha * (x - self._y)
        return self._y.copy()

    def sync(self, y: np.ndarray) -> None:
        """Force the filter output state to `y` (keep it aligned with the value
        actually sent after a downstream safety clamp)."""
        self._y = np.asarray(y, dtype=float).copy()


class SecondOrderLowPass:
    """Critically-damped-equivalent 2nd-order low-pass, monotone step response.

    Implemented as two cascaded first-order IIR stages at the SAME cutoff
    (two coincident real poles = the discrete analogue of critical damping).
    Each stage is `y += alpha*(x-y)` with `0 < alpha <= 1`, which is
    UNCONDITIONALLY STABLE for any cutoff_hz / dt (unlike an explicit-Euler
    integration of the continuous mass-spring-damper ODE, which only stays
    stable while omega*dt is small and produces large tracking lag / blows up
    numerically as cutoff approaches ~1/(2*dt)).  This lets you raise the
    cutoff at a fixed control dt without risking instability - the tradeoff
    between smoothing and ramp-tracking lag is then just alpha vs alpha^2.
    """

    def __init__(self, cutoff_hz: float, dt: float, dim: int = 7) -> None:
        self.cutoff_hz = cutoff_hz
        self.dt = dt
        self.dim = dim
        self.alpha = alpha_from_cutoff(cutoff_hz, dt)
        self._y1: np.ndarray | None = None
        self._y2: np.ndarray | None = None

    def reset(self, x0: np.ndarray | None = None) -> None:
        self._y1 = None if x0 is None else np.asarray(x0, dtype=float).copy()
        self._y2 = None if x0 is None else np.asarray(x0, dtype=float).copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._y1 is None:
            self._y1 = x.copy()
            self._y2 = x.copy()
            return self._y2.copy()
        self._y1 = self._y1 + self.alpha * (x - self._y1)
        self._y2 = self._y2 + self.alpha * (self._y1 - self._y2)
        return self._y2.copy()

    def sync(self, y: np.ndarray) -> None:
        """Re-seat both stages on the actually-sent value after a clamp."""
        y = np.asarray(y, dtype=float).copy()
        self._y1 = y.copy()
        self._y2 = y.copy()


class MovingAverage:
    def __init__(self, window: int, dim: int = 7) -> None:
        self.window = max(int(window), 1)
        self.dim = dim
        self._buf: deque[np.ndarray] = deque(maxlen=self.window)

    def reset(self, x0: np.ndarray | None = None) -> None:
        self._buf.clear()
        if x0 is not None:
            self._buf.append(np.asarray(x0, dtype=float).copy())

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        self._buf.append(x.copy())
        return np.mean(np.stack(self._buf, axis=0), axis=0)
```

## FILE: `rm75_control/control/joint_admittance/utils/friction.py`

```py
"""Optional joint friction / stiction feed-forward (Phase 3, experimental).

RM harmonic-drive joints have noticeable static friction; at very low scan speed
the force loop can stick-slip while the joint fights breakaway.  With only a
position/velocity interface (no direct torque command) we cannot inject a true
torque compensation, so this applies a *velocity-level* nudge: a smooth Coulomb
+ viscous term that adds a small breakaway velocity in the direction of motion.

Disabled by default.  Enable and tune ONLY after the basic loop is stable, and
keep the gains tiny - excessive compensation causes limit-cycle chatter.

    dqdot = fc * tanh(qdot / v_eps) + fv * qdot
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrictionConfig:
    enabled: bool = False
    coulomb: np.ndarray = field(default_factory=lambda: np.zeros(7))   # rad/s breakaway nudge
    viscous: np.ndarray = field(default_factory=lambda: np.zeros(7))   # rad/s per rad/s
    v_eps: float = 0.02                                                # rad/s smoothing of sign()


class FrictionCompensator:
    def __init__(self, cfg: FrictionConfig | None = None) -> None:
        self.cfg = cfg or FrictionConfig()

    def __call__(self, qdot: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        if not cfg.enabled:
            return np.zeros_like(qdot)
        qdot = np.asarray(qdot, dtype=float)
        return cfg.coulomb * np.tanh(qdot / max(cfg.v_eps, 1e-6)) + cfg.viscous * qdot
```

## FILE: `rm75_control/force/compensation/tool_pose.py`

```py
"""Map poses.yaml slot TCP into the active tool frame (FK from q_deg)."""

from __future__ import annotations

import numpy as np

# Scan / admittance standoff: poses.yaml slot d is force-ID Arm_Tip at contact;
# runtime scan pose D is +220 mm along tool +Z from that teach pose (not FK(q_deg)).
DEFAULT_SCAN_APPROACH_DZ_M = 0.220


def get_active_tool_name(robot) -> str:
    ret, cur = robot.rm_get_current_tool_frame()
    if ret != 0:
        return ""
    return str(cur.get("name", ""))


def poses_calib_tool_frame(poses_data: dict, *, default: str = "Arm_Tip") -> str:
    return str(poses_data.get("pose_tool_frame", default))


def slot_tcp_pose(
    robot,
    q_deg: np.ndarray,
    pose_stored: np.ndarray,
    *,
    calib_tool: str,
) -> np.ndarray:
    """
    TCP pose in base frame for the **active** tool at slot ``q_deg``.

    ``poses.yaml`` ``pose_base`` is recorded with ``calib_tool`` active (e.g. Arm_Tip).
    When the Web UI active tool differs (e.g. gripper, ~220 mm offset on RM75-6F),
    ``state.pose`` and stored ``pose_base`` disagree at the same ``q_deg`` — use FK.
    """
    q_deg = np.asarray(q_deg, dtype=float)
    pose_stored = np.asarray(pose_stored, dtype=float)
    active = get_active_tool_name(robot)
    if active and calib_tool and active != calib_tool:
        fk = robot.rm_algo_forward_kinematics(q_deg.tolist(), flag=1)
        return np.asarray(fk[:6], dtype=float)
    return pose_stored.copy()


def tool_frame_delta_pose(
    robot,
    pose_ref: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Apply a translation delta in the tool frame of ``pose_ref`` (Realman frameMode=1)."""
    delta = [float(dx), float(dy), float(dz), 0.0, 0.0, 0.0]
    out = robot.rm_algo_pose_move(list(np.asarray(pose_ref, dtype=float)), delta, frameMode=1)
    return np.asarray(out[:6], dtype=float)


def slot_scan_approach_pose(
    robot,
    pose_arm_tip: np.ndarray,
    *,
    approach_dz_m: float = DEFAULT_SCAN_APPROACH_DZ_M,
) -> np.ndarray:
    """
    Scan standoff pose D from a force-ID slot teach pose.

    ``poses.yaml`` slot ``d`` ``pose_base`` is saved with ``Arm_Tip`` at the
    contact / identification tip.  The velocity-admittance scan startup pose D is
    **+220 mm along tool +Z** (outward, away from tissue) — not the raw teach
    pose and not ``FK(q_deg)`` with ``gripper`` active (that lands on the tip).
    """
    return tool_frame_delta_pose(robot, pose_arm_tip, 0.0, 0.0, approach_dz_m)
```

## FILE: `rm75_control/motion/canfd.py`

```py
"""CANFD pose and velocity streaming + unified handoff (traj0 only)."""

from __future__ import annotations

import time
from typing import Any, Protocol, Sequence

import numpy as np

from rm75_control.core.exceptions import MotionError

Pose6 = Sequence[float]
Vel6 = Sequence[float]

MAX_LINEAR_V_M_S = 0.25
MAX_ANGULAR_V_RAD_S = 0.6

TRAJ0_MODE = 0
TRAJ0_RADIO = 0

LATCHED_ERR_CODES = frozenset({"4119", "1003", "4099", "0x1003"})


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
    del trajectory_mode, radio
    if len(pose) not in (6, 7):
        raise ValueError(f"pose must have 6 (euler) or 7 (quat) elements, got {len(pose)}")

    ret = robot.rm_movep_canfd(
        list(pose),
        follow,
        TRAJ0_MODE,
        TRAJ0_RADIO,
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
    del trajectory_mode, radio
    ret = robot.rm_movev_canfd(
        clamp_cartesian_velocity(cartesian_velocity),
        follow,
        TRAJ0_MODE,
        TRAJ0_RADIO,
    )
    if ret != 0:
        raise MotionError(f"rm_movev_canfd failed with code {ret}")


class JointCanfdClient(Protocol):
    def rm_movej_canfd(
        self,
        joint: list[float],
        follow: bool,
        expand: float = 0,
        trajectory_mode: int = 0,
        radio: int = 0,
    ) -> int:
        ...


def send_joint_canfd(
    robot: JointCanfdClient,
    joint_deg: Sequence[float],
    *,
    follow: bool = True,
    expand: float = 0.0,
    trajectory_mode: int = 0,
    radio: int = 0,
) -> None:
    """Stream absolute joint angles (degrees) via rm_movej_canfd passthrough.

    This is the single output interface for the joint-space inner loop - it never
    switches the controller between MoveJ/MoveV modes.  Joint values must already
    be smoothed / rate-limited (see joint_admittance.utils.safety); the driver
    faults on discontinuities.
    """
    del trajectory_mode, radio
    if len(joint_deg) != 7:
        raise ValueError(f"joint_deg must have 7 elements, got {len(joint_deg)}")
    ret = robot.rm_movej_canfd(list(joint_deg), follow, expand, TRAJ0_MODE, TRAJ0_RADIO)
    if ret != 0:
        raise MotionError(f"rm_movej_canfd failed with code {ret}")


def _wait_planning_idle(robot, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        traj = robot.rm_get_arm_current_trajectory()
        if traj.get("return_code") == 0 and traj.get("trajectory_type", 0) == 0:
            return True
        time.sleep(0.05)
    return False


def read_system_err(robot) -> list[str]:
    ret, st = robot.rm_get_current_arm_state()
    if ret != 0:
        return [f"get_state:{ret}"]
    err = st.get("err", {})
    n = int(err.get("err_len", 0))
    return [str(c) for c in list(err.get("err", []))[:n]]


def _should_clear_err(codes: list[str]) -> bool:
    if not codes or codes == ["0"]:
        return False
    for c in codes:
        if c in LATCHED_ERR_CODES:
            return True
        if "1003" in c or "4119" in c:
            return True
    return False


def _format_system_err(codes: list[str]) -> str:
    """Realman returns ['0'] when no latched error — not a fault."""
    if not codes:
        return "none"
    if len(codes) == 1 and str(codes[0]) in ("0", ""):
        return "none"
    return str(codes)


def print_handoff_diag(diag: dict[str, Any], *, prefix: str = "  handoff") -> None:
    parts = [
        f"slow_stop={diag.get('slow_stop')}",
        f"pause={diag.get('pause')}",
        f"delete={diag.get('delete_traj')}",
        f"idle={diag.get('planning_idle')}",
        f"traj={diag.get('trajectory_type')}",
    ]
    if diag.get("q_drift_deg") is not None:
        parts.append(f"q_drift={diag['q_drift_deg']:.2f}°")
    if diag.get("resync_done"):
        parts.append("resync=1")
    err = _format_system_err(diag.get("system_err") or [])
    parts.append(f"err={err}")
    if diag.get("quiescent_max_step_mm") is not None:
        parts.append(f"quiesce={diag.get('quiescent_frames')}f/{diag['quiescent_max_step_mm']:.2f}mm")
    print(f"{prefix}: " + "  ".join(parts), flush=True)


def exit_canfd_session(
    robot,
    *,
    q_resync: np.ndarray | None = None,
    settle_sleep_s: float = 0.3,
    move_speed: int = 15,
    settle_timeout_s: float = 15.0,
    resync_threshold_deg: float = 1.0,
    print_diag: bool = False,
) -> dict[str, Any]:
    """
    Unified CANFD exit: slow_stop → pause → delete → idle → sleep → resync → clear err.

    Call before move_j, between slots, and before enter_movev_session.
    """
    diag: dict[str, Any] = {}
    diag["slow_stop"] = robot.rm_set_arm_slow_stop()
    time.sleep(0.1)
    diag["pause"] = robot.rm_set_arm_pause()
    diag["delete_traj"] = robot.rm_set_arm_delete_trajectory()
    diag["planning_idle"] = _wait_planning_idle(robot)
    time.sleep(settle_sleep_s)

    if q_resync is not None:
        q_tgt = np.asarray(q_resync, dtype=float)
        ret, st = robot.rm_get_current_arm_state()
        if ret == 0:
            q_act = np.asarray(st["joint"][:7], dtype=float)
            q_drift = float(np.max(np.abs(q_act - q_tgt)))
            diag["q_drift_deg"] = q_drift
            if q_drift > resync_threshold_deg:
                from rm75_control.force.compensation.collection import move_j, wait_settle

                move_j(robot, q_tgt, speed=move_speed)
                wait_settle(robot, q_tgt, timeout_s=settle_timeout_s)
                diag["resync_done"] = True

    diag["system_err_before"] = read_system_err(robot)
    if _should_clear_err(diag["system_err_before"]):
        try:
            diag["clear_system_err"] = robot.rm_clear_system_err()
        except Exception:
            diag["clear_system_err"] = -999
        time.sleep(0.2)

    traj = robot.rm_get_arm_current_trajectory()
    diag["trajectory_type"] = traj.get("trajectory_type", -1)
    diag["system_err"] = read_system_err(robot)

    if print_diag:
        print_handoff_diag(diag)
    return diag


def settle_movev_after_init(
    robot,
    *,
    dt_ms: float,
    follow: bool,
    n_frames: int = 30,
    next_tick: float | None = None,
) -> float:
    """
    Zero-velocity frames after rm_set_movev_canfd_init — always low follow.

    rm_set_movev_canfd_init captures the current joint state as the IK
    reference.  If the arm is still micro-vibrating (which is common <200ms
    after move_j), with follow=True the high-bandwidth servo sees the residual
    error and issues a large corrective velocity → visible twitch/snap.

    Sending ALL settle frames with follow=False (低跟随, gentler servo) keeps
    the correction velocity small regardless of when init was called.  The
    actual scan commands use follow=True from the first tick of the main loop,
    by which point the arm is already confirmed quiescent.
    """
    dt_s = dt_ms / 1000.0
    if next_tick is None:
        next_tick = time.monotonic()
    zero = [0.0] * 6
    for _ in range(n_frames):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
        next_tick += dt_s
        send_velocity_canfd(robot, zero, follow=False)
    return next_tick


def wait_movev_quiescent(
    robot,
    *,
    dt_ms: float,
    follow: bool,
    settle_mm: float = 0.3,
    need_consecutive: int = 5,
    max_frames: int = 200,
    warmup_frames: int = 15,
    reject_max_step_mm: float = 2.0,
    next_tick: float | None = None,
) -> tuple[np.ndarray | None, float, int, float]:
    """
    Stream zero velocity until TCP motion < settle_mm for need_consecutive ticks.

    Uses follow=False (低跟随) to match settle_movev_after_init — quiescence is
    measured under the same gentle-servo conditions the arm will settle in.
    The actual session's follow mode takes effect from the first real command.

    warmup_frames: ignore motion for the first N ticks after init (init snap).
    reject_max_step_mm: the quiet streak's peak step must stay below this limit;
        prevents declaring quiescence when an early 9 mm snap is followed by
        sub-mm creep (global max_step was misleading).

    Returns (last_pose, max_step_mm, frames_used, next_tick).
    """
    dt_s = dt_ms / 1000.0
    if next_tick is None:
        next_tick = time.monotonic()
    del follow  # low follow throughout; see docstring
    zero = [0.0] * 6
    prev_xyz: np.ndarray | None = None
    last_pose: np.ndarray | None = None
    quiet = 0
    max_step_mm = 0.0
    streak_max_mm = 0.0
    for k in range(max_frames):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
        next_tick += dt_s
        send_velocity_canfd(robot, zero, follow=False)
        ret, st = robot.rm_get_current_arm_state()
        if ret != 0:
            continue
        pose = np.asarray(st["pose"][:6], dtype=float)
        last_pose = pose
        if k < warmup_frames:
            prev_xyz = pose[:3].copy()
            quiet = 0
            streak_max_mm = 0.0
            continue
        if prev_xyz is not None:
            step_mm = float(np.linalg.norm((pose[:3] - prev_xyz) * 1000.0))
            max_step_mm = max(max_step_mm, step_mm)
            if step_mm < settle_mm:
                quiet += 1
                streak_max_mm = max(streak_max_mm, step_mm)
                if (
                    quiet >= need_consecutive
                    and streak_max_mm <= reject_max_step_mm
                ):
                    return last_pose, max_step_mm, k + 1, next_tick
            else:
                quiet = 0
                streak_max_mm = 0.0
        prev_xyz = pose[:3].copy()
    return last_pose, max_step_mm, max_frames, next_tick


def enter_movev_session(
    robot,
    *,
    frame_type: int,
    avoid_singularity: int,
    dt_ms: float,
    follow: bool,
    q_resync: np.ndarray | None = None,
    skip_resync: bool = False,
    settle_frames: int = 50,
    quiescent_mm: float = 0.25,
    quiescent_consecutive: int = 10,
    quiescent_warmup_frames: int = 15,
    quiescent_reject_step_mm: float = 2.0,
    move_speed: int = 15,
    settle_timeout_s: float = 15.0,
    pre_init_settle_s: float = 1.5,
    next_tick: float | None = None,
    print_diag: bool = False,
) -> tuple[float, dict[str, Any]]:
    """
    Full movev handoff: exit → (settle) → init → zero frames → quiescence.

    pre_init_settle_s: extra sleep between exit_canfd_session and
    rm_set_movev_canfd_init.  exit_canfd_session already sleeps 0.3s, so the
    total static time before init = 0.3 + pre_init_settle_s.  Default 1.5s
    gives 1.8s total — arm must be fully static before init captures FK.

    skip_resync: when True, exit CANFD without joint resync (use after move_j
    already placed the arm at the target slot).

    Returns (next_tick, diag).
    """
    diag = exit_canfd_session(
        robot,
        q_resync=None if skip_resync else q_resync,
        move_speed=move_speed,
        settle_timeout_s=settle_timeout_s,
        print_diag=False,
    )

    if pre_init_settle_s > 0.0:
        time.sleep(pre_init_settle_s)

    # rm_set_movev_canfd_init can transiently refuse if the controller's internal
    # CANFD state machine (joint-CANFD layer) hasn't fully reset yet — the
    # trajectory-planner idle check doesn't cover that layer.  Retry with a
    # short light-cleanup cycle between attempts.
    _INIT_RETRIES = 3
    _INIT_RETRY_SLEEP_S = 0.5
    ret = -1
    for attempt in range(_INIT_RETRIES):
        ret = robot.rm_set_movev_canfd_init(avoid_singularity, frame_type, int(dt_ms))
        if ret == 0:
            break
        if attempt < _INIT_RETRIES - 1:
            robot.rm_set_arm_pause()
            robot.rm_set_arm_delete_trajectory()
            _wait_planning_idle(robot, timeout_s=5.0)
            time.sleep(_INIT_RETRY_SLEEP_S)
    diag["movev_init"] = ret
    diag["movev_init_attempts"] = attempt + 1
    if ret != 0:
        raise RuntimeError(
            f"rm_set_movev_canfd_init failed after {_INIT_RETRIES} attempts: {ret}"
        )

    next_tick = settle_movev_after_init(
        robot, dt_ms=dt_ms, follow=follow,
        n_frames=settle_frames, next_tick=next_tick,
    )
    pose_quiet, max_step_mm, q_frames, next_tick = wait_movev_quiescent(
        robot, dt_ms=dt_ms, follow=follow,
        settle_mm=quiescent_mm, need_consecutive=quiescent_consecutive,
        warmup_frames=quiescent_warmup_frames,
        reject_max_step_mm=quiescent_reject_step_mm,
        next_tick=next_tick,
    )
    diag["quiescent_max_step_mm"] = max_step_mm
    diag["quiescent_frames"] = q_frames
    if pose_quiet is not None:
        diag["quiescent_pose"] = pose_quiet.tolist()

    if print_diag:
        print_handoff_diag(diag)
        attempts = diag.get("movev_init_attempts", 1)
        if attempts > 1:
            print(f"  movev_init: needed {attempts} attempts", flush=True)
        if max_step_mm > 0:
            print(
                f"  movev quiescence: {q_frames} frames (max step {max_step_mm:.2f}mm/tick)",
                flush=True,
            )
    return next_tick, diag
```

## FILE: `configs/joint_admittance.yaml`

```yaml
# Joint-space inner-loop (QP/CLIK) + task-space admittance outer-loop config.
#
# Cascade: the admittance outer loop (hybrid_motion.AdmittanceController) emits a
# 6D Cartesian twist; the inner loop turns it into absolute joint angles streamed
# via rm_movej_canfd (single interface, no MoveJ/MoveV mode switching).
#
# Run:  source env.sh
#       python apps/joint_admittance/run_joint_admittance.py --config configs/joint_admittance.yaml
#
# For an on-robot test that also exercises Cartesian trajectory tracking (walk to
# a pose) + force-position hybrid control (sin scan there), both through this same
# cascade with no MoveJ/MoveV switch, see tmp/joint_admittance/d_sin_tool_y.py.

robot:
  ip: "192.168.1.18"
  port: 8080
  thread_mode: 2

timing:
  dt_ms: 10.0          # 100 Hz control loop
  async_poll_ms: 10.0

# ---------------------------------------------------------------------------
# Inner loop (joint-space IK)
# ---------------------------------------------------------------------------
inner:
  solver: clik          # "clik" (Phase 1 DLS) | "qp" (Phase 2 ProxQP, hard limits)
  control_frame: tool   # frame the outer-loop twist is expressed in (match hybrid_motion.control_frame)
  euler_order: xyz

  # safety (streaming q_cmd bypasses MoveJ's planner, so these MUST hold)
  v_scale: 0.35         # fraction of URDF joint velocity limit allowed (start conservative)
  a_max: 12.0           # rad/s^2 per-joint acceleration clamp
  position_margin_deg: 2.0
  use_smoothing: true
  # utils/smoothing.SecondOrderLowPass is now two cascaded 1st-order stages
  # (unconditionally stable at any cutoff/dt - see MD/JOINT_ADMITTANCE.md).
  # Ramp-tracking lag falls roughly with 1/cutoff; smoothing (jerk reduction)
  # falls as you raise it. 20 Hz is a safe starting point at 100 Hz control;
  # raise toward 30-40 Hz once bring-up confirms no current-spike faults.
  smooth_cutoff_hz: 20.0

  clik:
    k_task: [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]   # Cartesian error feedback gain (1/s)
    sigma_thresh: 0.04    # DLS: below this smallest singular value, damping ramps in
    lambda_max: 0.08      # DLS: max damping at a singularity
    nullspace_gain: 1.0
    max_pos_err_m: 0.05   # anti-windup cap on fed-back Cartesian error
    max_rot_err_rad: 0.20

  qp:                     # used only when solver == qp
    k_task: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    task_weight: [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
    reg: 1.0e-3           # regularization / secondary-task weight
    backend: proxqp       # proxqp (warm-started) | osqp
    eps_abs: 1.0e-6
    max_iter: 200

  nullspace:
    k_center: 0.5         # pull joints toward mid-range (rad/s per normalized unit)
    k_limit: 2.0          # extra repulsion near a joint limit
    activation: 0.85      # |normalized pos| beyond which repulsion ramps in

# ---------------------------------------------------------------------------
# Outer loop (task-space admittance) - hybrid_motion.AdmittanceConfig keys
# ---------------------------------------------------------------------------
frames:
  euler_order: xyz
  control_frame: tool

force:
  desired_z_n: 3.0        # constant normal-force setpoint on tool-Z [N]
  phi_source: phi_recommended
  fc_hz: 6.0
  min_samples: 35

hybrid_motion:
  force_axes: [0, 0, 1, 0, 0, 0]
  track_axes: [1, 1, 0, 1, 1, 1]
  kp_pos: [0.3, 0.6, 0.0, 0.4, 0.4, 0.4]
  system_delay_s: 0.015
  contact_threshold_n: 0.5
  deadband_n: 0.3
  deadband_width_n: 0.2
  max_velocity: [0.15, 0.15, 0.05, 0.5, 0.5, 0.5]
  max_acceleration: [1.0, 1.0, 0.8, 2.0, 2.0, 2.0]
  admittance_mass_z: 3.0
  admittance_damping_z: 60.0
  max_vz_tool_m_s: 0.05
  var_damping_enabled: true

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
startup:
  pose_slot_q_deg: null   # optional [7] MoveJ target before streaming; null = start where it is
  duration_s: 20.0
  move_speed: 20
  reference: hold         # "hold" = maintain the start pose (bring-up default)
  enable_force: false     # true requires a calibrated force_id_phi.json
  follow: true
  realtime: false         # true attempts SCHED_FIFO (needs privileges)
  watchdog_timeout_s: 0.1
```

## FILE: `apps/joint_admittance/run_joint_admittance.py`

```py
#!/usr/bin/env python3
"""Run the cascaded joint-position controller on the RM75-F.

  source env.sh
  # 1) FIRST prove FK matches the robot (<1mm / <0.1deg):
  python -m rm75_control.control.joint_admittance.validation --ip 192.168.1.18
  # 2) then run the loop (bring-up default: hold the start pose):
  python apps/joint_admittance/run_joint_admittance.py --config configs/joint_admittance.yaml

For a real test sequence (Cartesian move to a pose + force-position sinusoid
scan there, both through this same controller), see
tmp/joint_admittance/d_sin_tool_y.py instead.

The outer admittance loop emits a Cartesian twist; the inner CLIK/QP loop turns
it into absolute joint angles streamed through rm_movej_canfd only - no MoveV/
MoveJ mode switching once running.  A single MoveJ positions the arm at start.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.hybrid_motion.controller import AdmittanceConfig, AdmittanceController
from rm75_control.control.joint_admittance.config import build_joint_ik_config
from rm75_control.control.joint_admittance.loop import (
    AdmittanceOuterLoop,
    JointIkController,
    run_joint_admittance_loop,
)
from rm75_control.control.joint_admittance.model import RobotKinematics
from rm75_control.control.joint_admittance.reference import HoldReference
from rm75_control.core.session import RobotSession


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description="RM75 joint-position cascaded controller")
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance.yaml"))
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--solver", choices=["clik", "qp"], default=None)
    ap.add_argument("--dry-run", action="store_true", help="build everything, do not connect")
    args = ap.parse_args()

    raw = load_yaml(args.config)
    if args.solver:
        raw.setdefault("inner", {})["solver"] = args.solver

    startup = raw.get("startup", {})
    dt = float(raw.get("timing", {}).get("dt_ms", 10.0)) / 1000.0

    kin = RobotKinematics()
    inner_cfg = build_joint_ik_config(raw)
    inner = JointIkController(kin, inner_cfg)
    print(f"Inner loop: solver={inner_cfg.solver} control_frame={inner_cfg.control_frame} "
          f"dt={dt*1000:.0f}ms v_scale={inner_cfg.v_scale}", flush=True)
    if inner_cfg.solver == "qp":
        print(f"  QP backend: {inner.core.backend_name}", flush=True)

    outer_ctrl = AdmittanceController(dt, AdmittanceConfig.from_dict(raw))
    desired_force = np.zeros(6)
    desired_force[2] = float(raw.get("force", {}).get("desired_z_n", 0.0))
    reference = HoldReference()  # bring-up default: hold the start pose
    outer = AdmittanceOuterLoop(outer_ctrl, reference, desired_force=desired_force)

    force_observer = None
    if bool(startup.get("enable_force", False)):
        from rm75_control.control.hybrid_motion.observer import CompensatedForceObserver

        force_observer = CompensatedForceObserver.from_yaml(raw)
        print("  force observer: enabled (requires calibrated phi)", flush=True)

    q_start = startup.get("pose_slot_q_deg")
    q_start = None if q_start is None else np.asarray(q_start, dtype=float)
    duration = args.duration if args.duration is not None else float(startup.get("duration_s", 10.0))

    if args.dry_run:
        print("dry-run: controllers built OK, not connecting.", flush=True)
        return 0

    robot_cfg = raw.get("robot", {})
    with RobotSession(ip=robot_cfg.get("ip"), port=robot_cfg.get("port"), config=args.config) as sess:
        run_joint_admittance_loop(
            sess,
            outer,
            inner,
            q_start_deg=q_start,
            duration_s=duration,
            dt=dt,
            force_observer=force_observer,
            follow=bool(startup.get("follow", True)),
            move_speed=int(startup.get("move_speed", 20)),
            realtime=bool(startup.get("realtime", False)),
            watchdog_timeout_s=float(startup.get("watchdog_timeout_s", 0.1)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## FILE: `tmp/joint_admittance/d_sin_tool_y.py`

```py
#!/usr/bin/env python3
"""On-robot bring-up test: Cartesian move to pose D, then tool-Y sin scan at D -
both driven by OUR joint-position cascade (CLIK/QP inner loop -> rm_movej_canfd),
never rm_movev_canfd / MoveV. One continuous run, no mode switch at the D handoff.

  Phase 1  current pose -> scan pose D   (CartesianTrackOuterLoop, no force)
  Phase 2  hold D, sin sweep tool-Y       (AdmittanceOuterLoop, force-position hybrid)

``poses.yaml`` slot ``d`` is the **force-ID Arm_Tip teach pose** (contact tip).
Scan pose **D** = that ``pose_base`` + **220 mm along tool +Z outward** — NOT
``FK(q_deg)`` with gripper (that puts the TCP tip on the ID point).

This is a SPECIFIC test script (hardcodes pose slot "d" from poses.yaml) - the
generic, config-driven entry point lives at apps/joint_admittance/run_joint_admittance.py.

Usage:
  source env.sh
  # 0) FIRST prove FK matches the robot (<1mm / <0.1deg):
  python -m rm75_control.control.joint_admittance.validation --ip 192.168.1.18

  # 1) dry run (builds everything, no robot connection):
  python tmp/joint_admittance/d_sin_tool_y.py --dry-run

  # 2) move to D only, no scan (prove Cartesian trajectory tracking in isolation):
  python tmp/joint_admittance/d_sin_tool_y.py --scan-duration 0

  # 3) full test: move to D, then 30s of tool-Y sin scan with 3N tool-Z force hold
  #    (requires a calibrated tmp/force_compensation/logs/force_id_phi.json):
  python tmp/joint_admittance/d_sin_tool_y.py --enable-force --desired-z 3.0 --scan-duration 30

  # 4) tune the move and the sweep:
  python tmp/joint_admittance/d_sin_tool_y.py --move-duration 10 \
      --y-pp-cm 10 --max-vel-cm-s 2 --enable-force --desired-z 3.0 --scan-duration 60

Ctrl+C stops cleanly (watchdog holds the last q, RobotSession releases the arm).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.hybrid_motion.controller import AdmittanceConfig, AdmittanceController
from rm75_control.control.joint_admittance.config import build_joint_ik_config
from rm75_control.control.joint_admittance.loop import (
    AdmittanceOuterLoop,
    CartesianTrackConfig,
    CartesianTrackOuterLoop,
    JointIkController,
    Phase,
    arrived,
    run_joint_admittance_phases,
)
from rm75_control.control.joint_admittance.model import RobotKinematics
from rm75_control.control.joint_admittance.reference import (
    CartesianMoveReference,
    SinToolYReference,
)
from rm75_control.core.session import RobotSession
from rm75_control.force.compensation.collection import load_slot
from rm75_control.force.compensation.id_config import load_config as load_force_id_config
from rm75_control.force.compensation.paths import CONFIG_ID
from rm75_control.force.compensation.tool_pose import (
    DEFAULT_SCAN_APPROACH_DZ_M,
    get_active_tool_name,
    poses_calib_tool_frame,
    slot_scan_approach_pose,
)
from rm75_control.force.compensation import excitation as ex


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_scan_pose_d(
    slot: str,
    robot,
    *,
    approach_dz_m: float,
    use_force_id_pose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (q_deg, scan_pose_D, force_id_pose_base) for slot in poses.yaml."""
    fid = load_force_id_config(CONFIG_ID)
    poses_data = ex.load_poses_yaml(fid.poses_yaml)
    calib_tool = poses_calib_tool_frame(poses_data)
    active = get_active_tool_name(robot) if robot is not None else ""
    if active and calib_tool and active != calib_tool:
        print(
            f"  slot {slot!r}: teach frame {calib_tool!r}, active tool {active!r}",
            flush=True,
        )

    q_deg, _fk_pose, rec = load_slot(fid, slot, robot, calib_tool=calib_tool)
    pose_id = np.asarray(rec["pose_base"], dtype=float)

    if use_force_id_pose:
        pose_d = _fk_pose.copy()
        print(
            f"  scan target: force-ID FK pose (legacy) pose={np.round(pose_d, 4)}",
            flush=True,
        )
    else:
        pose_d = slot_scan_approach_pose(robot, pose_id, approach_dz_m=approach_dz_m)
        print(
            f"  force-ID slot {slot!r} Arm_Tip teach: pose={np.round(pose_id, 4)}",
            flush=True,
        )
        print(
            f"  scan pose D (+{approach_dz_m * 1000:.0f} mm tool-Z): "
            f"pose={np.round(pose_d, 4)}",
            flush=True,
        )
    return q_deg, pose_d, pose_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance.yaml"))
    ap.add_argument("--slot", type=str, default="d", help="poses.yaml slot (default: d)")
    ap.add_argument(
        "--approach-dz-mm",
        type=float,
        default=DEFAULT_SCAN_APPROACH_DZ_M * 1000.0,
        help="scan D = Arm_Tip teach pose + this offset along tool +Z (mm); default 220",
    )
    ap.add_argument(
        "--use-force-id-pose",
        action="store_true",
        help="legacy: track FK(q_deg) tip pose instead of teach+220mm standoff",
    )
    ap.add_argument("--solver", choices=["clik", "qp"], default=None)
    ap.add_argument("--move-duration", type=float, default=8.0, help="phase 1 smoothstep duration (s)")
    ap.add_argument("--move-kp", type=float, default=1.5, help="phase 1 Cartesian tracking gain (1/s)")
    ap.add_argument("--y-pp-cm", type=float, default=6.0, help="tool-Y sin peak-to-peak amplitude (cm)")
    ap.add_argument("--max-vel-cm-s", type=float, default=2.0, help="tool-Y sin peak velocity (cm/s)")
    ap.add_argument("--period-s", type=float, default=None, help="override sin period (s); default from max-vel")
    ap.add_argument("--desired-z", type=float, default=None, help="tool-Z force setpoint (N); default from config")
    ap.add_argument("--scan-duration", type=float, default=30.0, help="phase 2 duration (s); 0 = skip phase 2")
    ap.add_argument("--enable-force", action="store_true", default=None, help="force-enable phase 2 (default: config startup.enable_force)")
    ap.add_argument("--dry-run", action="store_true", help="build everything, do not connect")
    args = ap.parse_args()

    raw = load_yaml(args.config)
    if args.solver:
        raw.setdefault("inner", {})["solver"] = args.solver
    startup = raw.get("startup", {})
    dt = float(raw.get("timing", {}).get("dt_ms", 10.0)) / 1000.0

    kin = RobotKinematics()
    inner_cfg = build_joint_ik_config(raw)
    inner = JointIkController(kin, inner_cfg)
    print(
        f"Inner loop: solver={inner_cfg.solver} control_frame={inner_cfg.control_frame} "
        f"dt={dt * 1000:.0f}ms v_scale={inner_cfg.v_scale}",
        flush=True,
    )

    amplitude_m = float(args.y_pp_cm) * 0.01 / 2.0
    max_vel_m_s = float(args.max_vel_cm_s) * 0.01
    desired_z = args.desired_z if args.desired_z is not None else float(raw.get("force", {}).get("desired_z_n", 0.0))
    enable_force = args.enable_force if args.enable_force is not None else bool(startup.get("enable_force", False))

    if args.dry_run:
        print("dry-run: controllers built OK, not connecting.", flush=True)
        return 0

    robot_cfg = raw.get("robot", {})
    with RobotSession(ip=robot_cfg.get("ip"), port=robot_cfg.get("port"), config=args.config) as sess:
        q_d_deg, pose_d, _pose_id = resolve_scan_pose_d(
            args.slot,
            sess.robot,
            approach_dz_m=float(args.approach_dz_mm) * 0.001,
            use_force_id_pose=bool(args.use_force_id_pose),
        )

        # --- Phase 1: Cartesian move current -> pose D (pure tracking, no force) ---
        move_ref = CartesianMoveReference(pose_d, args.move_duration, euler_order=inner_cfg.euler_order)
        move_outer = CartesianTrackOuterLoop(
            move_ref,
            CartesianTrackConfig(
                k_task=np.full(6, args.move_kp),
                euler_order=inner_cfg.euler_order,
                control_frame=inner_cfg.control_frame,  # MUST match the inner loop
            ),
        )
        phase1 = Phase(
            outer=move_outer,
            label=f"move -> {args.slot}",
            duration_s=None,
            max_duration_s=args.move_duration + 5.0,
            wait_until=lambda pose: arrived(pose, pose_d, tol_mm=2.0, tol_deg=1.0, euler_order=inner_cfg.euler_order),
        )

        phases = [phase1]

        force_observer = None
        if args.scan_duration > 0.0:
            # --- Phase 2: hold D, tool-Y sin sweep (force-position hybrid via OUR inner loop) ---
            outer_ctrl = AdmittanceController(dt, AdmittanceConfig.from_dict(raw))
            desired_force = np.zeros(6)
            desired_force[2] = desired_z
            sin_ref = SinToolYReference(
                amplitude_m,
                period_s=args.period_s,
                max_vel_m_s=None if args.period_s is not None else max_vel_m_s,
                soft_start=True,
                ramp_s=2.0,
                euler_order=inner_cfg.euler_order,
            )
            scan_outer = AdmittanceOuterLoop(outer_ctrl, sin_ref, desired_force=desired_force)

            if enable_force:
                from rm75_control.control.hybrid_motion.observer import CompensatedForceObserver

                force_observer = CompensatedForceObserver.from_yaml(raw)
                print("  force observer: enabled (requires calibrated phi)", flush=True)
            else:
                print("  force observer: DISABLED (--enable-force not set) - Fz feedback is zero", flush=True)

            phase2 = Phase(
                outer=scan_outer,
                label="sin_tool_y @ D",
                duration_s=args.scan_duration,
                force_observer=force_observer,
            )
            phases.append(phase2)
            print(
                f"Phase 2: tool-Y sin amp={args.y_pp_cm:.1f}cm p-p, v_peak={args.max_vel_cm_s:.1f}cm/s, "
                f"desired_z={desired_z:.1f}N, duration={args.scan_duration:.1f}s",
                flush=True,
            )

        t_last_print = [0.0]

        def on_step(label: str, t_phase: float, step, pose, f_ext) -> None:
            now = time.perf_counter()
            if now - t_last_print[0] >= 1.0:
                t_last_print[0] = now
                print(
                    f"  [{label}] t={t_phase:5.1f}s  cart_err={step.cart_err_mm:5.2f}mm  "
                    f"Fz={f_ext[2]:+5.2f}N  manip={step.manip:.3f}",
                    flush=True,
                )

        run_joint_admittance_phases(
            sess,
            phases,
            inner,
            q_start_deg=None,  # start wherever the arm currently is
            dt=dt,
            follow=bool(startup.get("follow", True)),
            move_speed=int(startup.get("move_speed", 20)),
            realtime=bool(startup.get("realtime", False)),
            watchdog_timeout_s=float(startup.get("watchdog_timeout_s", 0.1)),
            on_step=on_step,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## FILE: `tests/test_joint_ik_offline.py`

```py
"""Offline closed-loop validation of the joint-space inner loop (no robot).

Under perfect joint tracking, JointIkController.q_cmd *is* the simulated robot
state, so repeatedly calling ``update(twist)`` closes the loop.  We assert:

* zero drift when the twist is zero,
* faithful Cartesian tracking of the integrated reference after settling,
* nullspace motion does NOT move the TCP (redundancy resolution is orthogonal),
* velocity / position limits are always respected.

Run as pytest, or ``python tests/test_joint_ik_offline.py`` for a printed report.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance.clik import ClikConfig
from rm75_control.control.joint_admittance.loop import JointIkConfig, JointIkController
from rm75_control.control.joint_admittance.model import RobotKinematics, deg2rad
from rm75_control.control.joint_admittance.tasks.nullspace_task import NullspaceTaskConfig

Q_HOME_DEG = np.array([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0])


def _make(control_frame: str = "base", k_center: float = 0.0, **cfg_kw) -> JointIkController:
    kin = RobotKinematics()
    cfg = JointIkConfig(
        control_frame=control_frame,
        clik=ClikConfig(k_task=np.full(6, 5.0)),
        nullspace=NullspaceTaskConfig(k_center=k_center, k_limit=0.0),
        v_scale=0.9,
        a_max=50.0,
        **cfg_kw,
    )
    ctrl = JointIkController(kin, cfg)
    ctrl.reset(deg2rad(Q_HOME_DEG))
    return ctrl


def test_zero_drift_on_hold():
    ctrl = _make(k_center=0.0)
    pose0 = ctrl.kin.fk_pose(ctrl.q_cmd)
    for _ in range(300):
        ctrl.update(np.zeros(6))
    pose1 = ctrl.kin.fk_pose(ctrl.q_cmd)
    assert np.linalg.norm(pose1[:3] - pose0[:3]) < 1e-5


def test_tracks_reference_after_settle():
    ctrl = _make()
    dt = ctrl.cfg.dt
    twist = np.array([0.01, 0.0, -0.005, 0.0, 0.0, 0.0])  # base frame
    for _ in range(100):        # 1 s of motion
        ctrl.update(twist)
    for _ in range(200):        # 2 s hold to settle (2nd-order smoother group delay)
        ctrl.update(np.zeros(6))
    x_ref = ctrl.clik.x_ref
    x_cur = ctrl.kin.fk_pose(ctrl.q_cmd)
    assert np.linalg.norm(x_ref[:3] - x_cur[:3]) * 1000.0 < 1.0  # < 1 mm


def test_nullspace_preserves_tcp():
    """A 6-DOF task on a 7-DOF arm leaves a 1-D nullspace: the centering task can
    only move joints along that single direction, but it must NEVER perturb the
    TCP.  Validate: TCP fixed, joints actually move, centering cost non-increasing.
    """
    from scipy.spatial.transform import Rotation as Rsc

    ctrl = _make(k_center=1.5)
    q_mid = 0.5 * (ctrl.kin.q_lower + ctrl.kin.q_upper)
    q_start = ctrl.q_cmd.copy()
    pose0 = ctrl.kin.fk_pose(ctrl.q_cmd)
    dist0 = np.linalg.norm(q_start - q_mid)
    max_pos_err_mm = 0.0
    max_rot_err_deg = 0.0
    for _ in range(400):
        ctrl.update(np.zeros(6))
        pose = ctrl.kin.fk_pose(ctrl.q_cmd)
        max_pos_err_mm = max(max_pos_err_mm, np.linalg.norm(pose[:3] - pose0[:3]) * 1000.0)
        r0 = Rsc.from_euler("xyz", pose0[3:6]).as_matrix()
        r1 = Rsc.from_euler("xyz", pose[3:6]).as_matrix()
        d_deg = np.degrees(np.linalg.norm(Rsc.from_matrix(r1 @ r0.T).as_rotvec()))
        max_rot_err_deg = max(max_rot_err_deg, d_deg)
    dist1 = np.linalg.norm(ctrl.q_cmd - q_mid)
    joint_motion = np.linalg.norm(ctrl.q_cmd - q_start)
    # TCP stayed put (nullspace projection is orthogonal to the task) ...
    assert max_pos_err_mm < 1.0, max_pos_err_mm
    assert max_rot_err_deg < 0.1, max_rot_err_deg
    # ... joints actually moved (nullspace is live) ...
    assert joint_motion > 1e-3, joint_motion
    # ... and centering never made things worse.
    assert dist1 <= dist0 + 1e-6, (dist0, dist1)


def test_velocity_and_position_limits():
    ctrl = _make(k_center=0.0)
    dt = ctrl.cfg.dt
    dq_max = ctrl.kin.v_max * ctrl.cfg.v_scale * dt
    q_prev = ctrl.q_cmd.copy()
    for _ in range(500):
        ctrl.update(np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]))  # absurdly large twist
        dq = ctrl.q_cmd - q_prev
        assert np.all(np.abs(dq) <= dq_max + 1e-9), "velocity limit violated"
        assert np.all(ctrl.q_cmd >= ctrl.kin.q_lower - 1e-9)
        assert np.all(ctrl.q_cmd <= ctrl.kin.q_upper + 1e-9)
        q_prev = ctrl.q_cmd.copy()


def test_smoothing_filter_stable_at_high_cutoff():
    """Regression guard: SecondOrderLowPass must stay stable (bounded, decaying
    error under a constant-velocity ramp) at ANY cutoff_hz for a fixed dt.

    An earlier explicit-Euler discretization of the 2nd-order ODE was only
    conditionally stable and blew up (>100mm error) above ~17 Hz at dt=0.01s.
    The cascaded-first-order implementation must never do that.
    """
    for cutoff in (5.0, 15.0, 30.0, 60.0, 100.0, 250.0):
        ctrl = _make(k_center=0.0, smooth_cutoff_hz=cutoff)
        for _ in range(500):
            s = ctrl.update(np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.1]))
        assert s.cart_err_mm < 100.0, (cutoff, s.cart_err_mm)  # bounded lag, not exploding
        assert np.all(np.isfinite(ctrl.q_cmd))


def test_cartesian_track_outer_loop_tool_frame_converges():
    """Regression guard: CartesianTrackOuterLoop(control_frame="tool") MUST return a
    twist expressed in tool-axis coordinates, matching JointIkConfig(control_frame=
    "tool")'s _twist_to_base(), which does R @ twist. A base-frame twist here gets
    double-rotated by the inner loop and DIVERGES instead of converging - this is
    exactly the bug seen on-robot (cart_err growing to 100s of mm instead of settling).
    Use a non-trivial start orientation so a frame mismatch actually shows up.
    """
    from rm75_control.control.joint_admittance.loop import CartesianTrackConfig, CartesianTrackOuterLoop
    from rm75_control.control.joint_admittance.reference import CartesianMoveReference

    ctrl = _make(control_frame="tool", k_center=0.0)
    dt = ctrl.cfg.dt

    # Rotate q1/q4/q6 off zero so the TCP orientation is far from identity - a
    # frame mixup is invisible at R=I but explosive away from it.
    q_start = deg2rad(np.array([20.0, -30.0, 10.0, 70.0, 15.0, 50.0, -10.0]))
    ctrl.reset(q_start)
    pose0 = ctrl.kin.fk_pose(ctrl.q_cmd)
    pose_target = pose0.copy()
    pose_target[0] += 0.05
    pose_target[1] -= 0.03
    pose_target[2] += 0.02

    move_ref = CartesianMoveReference(pose_target, duration_s=3.0)
    move_outer = CartesianTrackOuterLoop(move_ref, CartesianTrackConfig(control_frame="tool"))
    move_outer.set_origin(pose0)

    t_s = 0.0
    err_mm = []
    for _ in range(700):  # 3s move + 4s settle
        cur = ctrl.kin.fk_pose(ctrl.q_cmd)
        twist = move_outer.sample(t_s, cur, np.zeros(6))
        ctrl.update(twist, dt)
        err_mm.append(np.linalg.norm(ctrl.kin.fk_pose(ctrl.q_cmd)[:3] - pose_target[:3]) * 1000.0)
        t_s += dt

    # must CONVERGE (not diverge): final error small, and well below the initial error.
    assert err_mm[-1] < 2.5, err_mm[-1]
    assert err_mm[-1] < err_mm[0], (err_mm[0], err_mm[-1])


def test_cartesian_move_then_sin_reference_offline():
    """End-to-end offline check for the D-then-sin_tool_y bring-up test: a
    CartesianMoveReference driven through CartesianTrackOuterLoop must land the
    TCP on the target pose, and handing off to a SinToolYReference at that pose
    must not cause any discontinuity in q_cmd (no MoveJ/MoveV switch needed).
    """
    from rm75_control.control.joint_admittance.loop import CartesianTrackOuterLoop
    from rm75_control.control.joint_admittance.reference import (
        CartesianMoveReference,
        SinToolYReference,
    )

    ctrl = _make(k_center=0.0)
    dt = ctrl.cfg.dt
    pose0 = ctrl.kin.fk_pose(ctrl.q_cmd)
    pose_target = pose0.copy()
    pose_target[0] += 0.05
    pose_target[2] += 0.03
    pose_target[5] += np.radians(5.0)

    move_ref = CartesianMoveReference(pose_target, duration_s=3.0)
    move_outer = CartesianTrackOuterLoop(move_ref)
    move_outer.set_origin(pose0)

    t_s = 0.0
    for _ in range(700):  # 3s move + 4s settle
        twist = move_outer.sample(t_s, ctrl.kin.fk_pose(ctrl.q_cmd), np.zeros(6))
        ctrl.update(twist, dt)
        t_s += dt

    pose_arrived = ctrl.kin.fk_pose(ctrl.q_cmd)
    assert np.linalg.norm(pose_arrived[:3] - pose_target[:3]) * 1000.0 < 2.5

    sin_ref = SinToolYReference(amplitude_m=0.01, period_s=4.0, soft_start=True, ramp_s=1.0)
    sin_outer = CartesianTrackOuterLoop(sin_ref)
    sin_outer.set_origin(pose_arrived)

    t_s = 0.0
    y_positions = []
    first_tick_dq = None
    for _ in range(400):  # 4s = 1 sin period
        q_prev = ctrl.q_cmd.copy()
        twist = sin_outer.sample(t_s, ctrl.kin.fk_pose(ctrl.q_cmd), np.zeros(6))
        ctrl.update(twist, dt)
        if first_tick_dq is None:
            first_tick_dq = np.linalg.norm(ctrl.q_cmd - q_prev)
        y_positions.append(ctrl.kin.fk_pose(ctrl.q_cmd)[1])
        t_s += dt

    # no discontinuity at the phase boundary: the handoff tick's joint step must
    # be in line with a normal control tick, not a MoveJ-style snap.
    assert first_tick_dq < 0.02, first_tick_dq

    # the sweep must actually move the TCP along Y once ramped up
    y_positions = np.asarray(y_positions)
    assert (y_positions.max() - y_positions.min()) > 0.005


def _report() -> None:
    print("Running offline joint-IK validation report...\n")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                print(f"  FAIL  {name}: {e}")

    # quantitative summary for the tracking case
    ctrl = _make()
    twist = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.1])
    jit = []
    for _ in range(200):
        s = ctrl.update(twist)
        jit.append(s.cart_err_mm)
    print(
        f"\n  ramp+rotate: final cart_err={jit[-1]:.3f} mm, "
        f"sigma_min={s.sigma_min:.4f}, manip={s.manip:.5f}, lam={s.lam:.5f}"
    )


if __name__ == "__main__":
    _report()
```

## FILE: `rm75_control/assets/robots/rm75_6f/RM75-6F.urdf`

```xml
<?xml version="1.0" encoding="utf-8"?>
<!-- RM75-6F kinematics-only URDF for rm75_control.
     Derived from the vendor SolidWorks export
     (Among_US/assets/robots/rm75_6f/vendor/.../RM75-6F.urdf).
     Changes vs vendor:
       * All <visual>/<collision> mesh geometry stripped (package:// deps removed);
         Pinocchio kinematics only needs links + joints + inertials.
       * Inertials retained for future dynamics / QP effort limits.
       * Added fixed 'tcp' link: link_7 -> tcp, +0.220 m along link_7 +Z, no rotation.
     Joint limits copied verbatim from vendor; verify j7 (+-6.28) against the
     official RM75 manual before trusting it in hard QP constraints.
     joint_7 origin y: vendor -0.1725 m -> -0.1612 m (-11.3 mm) to match Realman
     controller flange FK (see validation.py / MD/JOINT_ADMITTANCE.md). -->
<robot
  name="RM75-6F">
  <link name="base_link">
    <inertial>
      <origin xyz="0.00049987 5.2709E-05 0.060019" rpy="0 0 0" />
      <mass value="1.862" />
      <inertia ixx="0.0017232" ixy="-3.1058E-06" ixz="-3.7924E-05"
               iyy="0.0017051" iyz="1.3691E-06" izz="0.00090158" />
    </inertial>
  </link>
  <link name="link_1">
    <inertial>
      <origin xyz="0.000241 -0.013273 -0.00995" rpy="0 0 0" />
      <mass value="1.574" />
      <inertia ixx="0.002487573" ixy="0.000009663" ixz="-0.000007909"
               iyy="0.002321038" iyz="0.000179393" izz="0.001450554" />
    </inertial>
  </link>
  <joint name="joint_1" type="revolute">
    <origin xyz="0 0 0.2405" rpy="0 0 0" />
    <parent link="base_link" />
    <child link="link_1" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="60" velocity="3.14" />
  </joint>
  <link name="link_2">
    <inertial>
      <origin xyz="-0.000357 -0.106789 0.005329" rpy="0 0 0" />
      <mass value="1.217" />
      <inertia ixx="0.003494121" ixy="0.000002921" ixz="-0.000005613"
               iyy="0.000892721" iyz="-0.000583884" izz="0.003444080" />
    </inertial>
  </link>
  <joint name="joint_2" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_1" />
    <child link="link_2" />
    <axis xyz="0 0 1" />
    <limit lower="-2.2689" upper="2.2689" effort="60" velocity="3.14" />
  </joint>
  <link name="link_3">
    <inertial>
      <origin xyz="0.000003 -0.01398 -0.011324" rpy="0 0 0" />
      <mass value="1.11" />
      <inertia ixx="0.001836663" ixy="0.000002259" ixz="-0.000004216"
               iyy="0.001498875" iyz="0.000037167" izz="0.001062545" />
    </inertial>
  </link>
  <joint name="joint_3" type="revolute">
    <origin xyz="0 -0.256 0" rpy="1.5708 0 0" />
    <parent link="link_2" />
    <child link="link_3" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="30" velocity="3.14" />
  </joint>
  <link name="link_4">
    <inertial>
      <origin xyz="-0.000005 -0.084658 0.004747" rpy="0 0 0" />
      <mass value="0.685" />
      <inertia ixx="0.001282444" ixy="-0.000000551" ixz="-0.000000630"
               iyy="0.000373013" iyz="-0.000232084" izz="0.001256177" />
    </inertial>
  </link>
  <joint name="joint_4" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_3" />
    <child link="link_4" />
    <axis xyz="0 0 1" />
    <limit lower="-2.356" upper="2.356" effort="30" velocity="3.14" />
  </joint>
  <link name="link_5">
    <inertial>
      <origin xyz="0.000078 -0.012937 -0.008781" rpy="0 0 0" />
      <mass value="0.619" />
      <inertia ixx="0.000627336" ixy="0.000001636" ixz="-0.000001345"
               iyy="0.000542455" iyz="0.000034970" izz="0.000370291" />
    </inertial>
  </link>
  <joint name="joint_5" type="revolute">
    <origin xyz="0 -0.21 0" rpy="1.5708 0 0" />
    <parent link="link_4" />
    <child link="link_5" />
    <axis xyz="0 0 1" />
    <limit lower="-3.106" upper="3.106" effort="10" velocity="3.14" />
  </joint>
  <link name="link_6">
    <inertial>
      <origin xyz="-0.000014 -0.078524 0.002819" rpy="0 0 0" />
      <mass value="0.602" />
      <inertia ixx="0.000780774" ixy="-0.000000121" ixz="-0.000000469"
               iyy="0.000289973" iyz="-0.000120513" izz="0.000763955" />
    </inertial>
  </link>
  <joint name="joint_6" type="revolute">
    <origin xyz="0 0 0" rpy="-1.5708 0 0" />
    <parent link="link_5" />
    <child link="link_6" />
    <axis xyz="0 0 1" />
    <limit lower="-2.234" upper="2.234" effort="10" velocity="3.14" />
  </joint>
  <link name="link_7">
    <inertial>
      <origin xyz="0.001094 -0.000077 -0.010119" rpy="0 0 0" />
      <mass value="0.144" />
      <inertia ixx="0.000044123" ixy="-0.000000064" ixz="0.0000003"
               iyy="0.000035078" iyz="-0.000000029" izz="0.000065445" />
    </inertial>
  </link>
  <joint name="joint_7" type="revolute">
    <!-- Vendor SolidWorks URDF uses y=-0.1725 m. Realman controller flange FK sits
         ~11.3 mm closer (less negative y in link_6 frame). FK validation: vendor
         gave constant +11.3 mm pos error; moving to -0.1838 doubled it to ~22.6 mm
         (wrong direction); -0.1612 is the corrected value. -->
    <origin xyz="0 -0.1612 0" rpy="1.5708 0 0" />
    <parent link="link_6" />
    <child link="link_7" />
    <axis xyz="0 0 1" />
    <limit lower="-6.28" upper="6.28" effort="10" velocity="3.14" />
  </joint>
  <!-- Tool control point: pure frame, no mesh. +0.220 m along link_7 +Z. -->
  <link name="tcp" />
  <joint name="link_7_to_tcp" type="fixed">
    <origin xyz="0 0 0.220" rpy="0 0 0" />
    <parent link="link_7" />
    <child link="tcp" />
  </joint>
</robot>
```
