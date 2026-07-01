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
