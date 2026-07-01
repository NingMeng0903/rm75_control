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

    def update_joint(
        self,
        q_target: np.ndarray,
        qdot_ff: np.ndarray,
        dt: float | None = None,
        k_joint: float = 2.0,
    ) -> JointIkStep:
        """Joint-space PD+feedforward tracking of a moving joint target.

        Bypasses CLIK/QP entirely - no Cartesian error, no Jacobian, no nullspace
        projection.  Use for point-to-point repositioning when ``q_target(t)`` is
        already known (e.g. one-shot vendor IK + JointSmoothMoveReference
        smoothstep) - this reproduces MoveJ-like coordinated joint motion exactly,
        with none of the redundant-DOF nullspace drift a Cartesian-tracking CLIK
        reference has on a 7-DOF arm (see reference.py JointSmoothMoveReference).
        Still runs through the same friction/smoothing/safety pipeline as
        ``update()``, so it is equally safe to stream via ``rm_movej_canfd``.
        """
        dt = self.cfg.dt if dt is None else dt
        q_prev = self.q_cmd
        q_target = np.asarray(q_target, dtype=float)
        qdot_ff = np.asarray(qdot_ff, dtype=float)
        qdot = qdot_ff + k_joint * (q_target - q_prev)
        q_next = q_prev + qdot * dt

        if self.cfg.friction.enabled:
            q_next = q_next + self.friction(qdot) * dt
        if self.smoother is not None:
            q_next = self.smoother(q_next)

        rep = self.safety.clamp(q_prev, q_next, dt)
        if self.smoother is not None:
            self.smoother.sync(rep.q_safe)

        self.q_cmd = rep.q_safe
        x_cur = self.kin.fk_pose(rep.q_safe)
        x_tgt = self.kin.fk_pose(q_target)
        return JointIkStep(
            q_send=rep.q_safe.copy(),
            qdot=qdot,
            twist_base=np.zeros(6),
            sigma_min=float("nan"),
            lam=0.0,
            manip=float("nan"),
            cart_err_mm=float(np.linalg.norm(x_tgt[:3] - x_cur[:3]) * 1000.0),
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


class JointPathOuterLoop:
    """Execute a pre-planned joint-space path through CLIK/QP without Cartesian PD.

    The path's ``q_ref(t)`` / ``qdot_ff(t)`` come from e.g.
    ``JointSmoothMoveReference.sample_q``.  This outer loop deliberately emits
    **zero** Cartesian twist - all motion is via ``qdot_ff`` + CLIK/QP closed-loop
    feedback toward ``FK(q_ref(t))`` (``x_ref`` synced each tick in
    ``run_joint_admittance_phases``).  Using ``CartesianTrackOuterLoop`` here
    adds a separate Cartesian PD that fights ``qdot_ff`` and integrates ``x_ref``
    away from the joint path -> ``cart_err`` explodes (100+ mm diverge).
    """

    def __init__(self, reference) -> None:
        self.reference = reference

    def set_origin(self, pose0: np.ndarray) -> None:
        if hasattr(self.reference, "set_origin"):
            self.reference.set_origin(pose0)

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        del t_s, current_pose, f_ext
        return np.zeros(6, dtype=float)

    def sample_path(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        return self.reference.sample_q(t_s)


class JointSpaceMoveOuterLoop:
    """Optional bypass outer loop (``Phase.mode == "joint"``) feeding
    ``JointIkController.update_joint`` - NO CLIK, NO nullspace.

    Not for production point-to-point moves: use ``JointSmoothMoveReference`` +
    ``CartesianTrackOuterLoop`` + ``Phase.q_ref_provider`` instead so nullspace
    (centering, ``q_ref`` tracking, obstacle avoidance) stays live.
    """

    def __init__(self, reference, k_joint: float = 2.0) -> None:
        self.reference = reference
        self.k_joint = k_joint

    def set_origin(self, pose0: np.ndarray) -> None:
        if hasattr(self.reference, "set_origin"):
            self.reference.set_origin(pose0)

    def sample_joint(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        return self.reference.sample_q(t_s)


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
    mode: str = "cartesian"                  # "cartesian" (outer.sample -> twist) | "joint"
                                              # (outer.sample_joint -> (q_target, qdot_ff), see
                                              # JointSpaceMoveOuterLoop - bypasses Cartesian IK)
    duration_s: float | None = None          # None -> run until wait_until (or max_duration_s)
    max_duration_s: float | None = None      # safety cap when duration_s is None
    wait_until: object | None = None         # Callable[[np.ndarray], bool] on current pose
    q_ref_provider: object | None = None     # Callable[[float], q_ref_rad], mild nullspace
                                              # trim only (centering-style) - NOT the primary
                                              # mechanism for tracking a planned joint path,
                                              # see qdot_ff_provider for that (avoids fighting
                                              # the primary term's own redundancy resolution -
                                              # ClikController.step docstring)
    qdot_ff_provider: object | None = None   # Callable[[float], qdot_ff_rad_s]: joint-space
                                              # feedforward from a planned path (e.g.
                                              # JointSmoothMoveReference.sample_q); preferred
                                              # way to keep a joint path's own coordinated
                                              # redundancy resolution while CLIK/nullspace stay live
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

                    if phase.mode == "joint":
                        q_tgt, qdot_ff = phase.outer.sample_joint(t_phase)
                        step = inner.update_joint(
                            q_tgt, qdot_ff, dt, k_joint=getattr(phase.outer, "k_joint", 2.0)
                        )
                    else:
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
