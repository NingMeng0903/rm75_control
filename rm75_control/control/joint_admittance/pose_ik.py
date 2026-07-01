"""One-shot pose inverse kinematics using our Pinocchio model + DLS-CLIK.

Planning-only helper: resolves ``q_target`` for a desired TCP pose without any
vendor ``rm_algo_inverse_kinematics`` call.  The resolved ``q_target`` is then
fed to ``JointSmoothMoveReference`` for a joint-space smoothstep; execution still
runs through the live CLIK/QP inner loop with nullspace (centering, ``q_ref``
tracking, future obstacle gradients).

References: Wampler 1986 / Nakamura & Hanafusa 1986 (DLS); Sciavicco & Siciliano
1988 (pose error feedback).
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance.clik import (
    ClikConfig,
    damping_from_sigma,
    dls_pinv,
)
from rm75_control.control.joint_admittance.model import RobotKinematics, pose_error


def solve_pose_ik(
    kin: RobotKinematics,
    q_seed: np.ndarray,
    pose_target: np.ndarray,
    *,
    max_iters: int = 500,
    pos_tol_m: float = 1e-3,
    rot_tol_rad: float = 0.02,
    dt: float = 0.02,
    clik_cfg: ClikConfig | None = None,
) -> tuple[np.ndarray, bool]:
    """Iterative damped-least-squares IK: ``q_seed`` -> ``q`` with ``fk(q) ≈ pose_target``.

    Returns ``(q_sol_rad, converged)``.  ``converged`` is False if iteration
    budget is exhausted before tolerances are met (caller should abort or retry
    with a different seed / relaxed tolerances).
    """
    cfg = clik_cfg or ClikConfig()
    k = np.asarray(cfg.k_task, dtype=float)
    q = np.clip(np.asarray(q_seed, dtype=float).copy(), kin.q_lower, kin.q_upper)
    pose_target = np.asarray(pose_target, dtype=float)

    for _ in range(max_iters):
        err = pose_error(pose_target, kin.fk_pose(q), cfg.euler_order)
        if np.linalg.norm(err[:3]) < pos_tol_m and np.linalg.norm(err[3:6]) < rot_tol_rad:
            return q, True

        J = kin.jacobian(q)
        sigma_min = float(kin.singular_values(J).min())
        lam = damping_from_sigma(sigma_min, cfg.sigma_thresh, cfg.lambda_max)
        qdot = dls_pinv(J, lam) @ (k * err)
        q = np.clip(q + qdot * dt, kin.q_lower, kin.q_upper)

    return q, False
