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
