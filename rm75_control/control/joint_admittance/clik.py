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
