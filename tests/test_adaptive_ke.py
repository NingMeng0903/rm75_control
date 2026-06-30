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
        ke_slew_max=1e6,
        df_spike_n=100.0,
        f_err_gate_n=100.0,
        scan_vel_gate_m_s=10.0,
    )
    est = EnvironmentStiffnessEstimator(cfg, dt=0.01, mass_z=2.0)
    ref = np.zeros(6)
    true_ke = 800.0
    pose = ref.copy()
    f = 0.0
    v_force_z = 0.00005 / 0.01  # 0.05 mm per 10 ms tick
    for _ in range(300):
        f += true_ke * 0.00005
        est.update(
            f, pose, in_contact=True, mass_z=2.0, v_force_z=v_force_z, v_scan_tool_y=0.0, f_err_z=0.0
        )
    assert 400.0 < est.ke_est < 1200.0
    zeta = est.zeta_eff
    assert 0.85 < zeta <= 1.05


def test_asymmetric_ke_rises_faster_than_symmetric():
    base = dict(
        enabled=True,
        zeta=1.0,
        ke_initial=200.0,
        ke_forgetting=0.995,
        ke_min=0.0,
        ke_max=10000.0,
        dx_threshold_m=1e-5,
        contact_force_n=0.1,
        bd_slew_max=1e6,
        ke_slew_max=1e6,
        gate_scan_velocity=False,
        gate_df_spike=False,
    )
    cfg_asym = AdaptiveKeConfig(**base, ke_forgetting_inc=0.80)
    cfg_sym = AdaptiveKeConfig(**base, ke_forgetting_inc=0.995)
    pose = np.zeros(6)
    v_z = 0.00005 / 0.01

    def run_ramp(est: EnvironmentStiffnessEstimator, true_ke: float, n: int) -> float:
        f = 0.0
        for _ in range(n):
            f += true_ke * 0.00005
            est.update(
                f, pose, in_contact=True, mass_z=2.0, v_force_z=v_z, f_err_z=0.0
            )
        return est.ke_est

    ke_asym = run_ramp(EnvironmentStiffnessEstimator(cfg_asym, dt=0.01, mass_z=2.0), 1200.0, 40)
    ke_sym = run_ramp(EnvironmentStiffnessEstimator(cfg_sym, dt=0.01, mass_z=2.0), 1200.0, 40)
    assert ke_asym > ke_sym
