"""Unit tests for online environment stiffness + critical damping."""

import math

import numpy as np

from rm75_control.control.hybrid_motion.adaptive_ke import (
    AdaptiveKeConfig,
    EnvironmentStiffnessEstimator,
)


def test_critical_damping_formula():
    cfg = AdaptiveKeConfig(enabled=True, zeta=1.0, ke_initial=400.0, bd_slew_max=1e6)
    est = EnvironmentStiffnessEstimator(cfg, dt=0.01, mass_z=2.0)
    m, ke = 2.0, 400.0
    expected = 2.0 * math.sqrt(m * ke)
    assert abs(est.bd - expected) < 1.0


def test_ewma_stiffness_converges():
    cfg = AdaptiveKeConfig(
        enabled=True,
        zeta=1.0,
        ke_initial=100.0,
        ke_forgetting=0.9,
        ke_min=0.0,
        ke_max=10000.0,
        dx_threshold_m=1e-5,
        contact_force_n=0.1,
        bd_slew_max=1e6,
    )
    est = EnvironmentStiffnessEstimator(cfg, dt=0.01, mass_z=2.0)
    ref = np.zeros(6)
    ref[2] = 0.52
    true_ke = 800.0
    pose = ref.copy()
    f = 0.0
    for _ in range(300):
        pose[2] -= 0.00005  # 0.05 mm per tick along base Z
        f += true_ke * 0.00005
        est.update(f, pose, in_contact=True, mass_z=2.0)
    assert 400.0 < est.ke_est < 1200.0
    zeta = est.zeta_eff
    assert 0.85 < zeta <= 1.05
