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
