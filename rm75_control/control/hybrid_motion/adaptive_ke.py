"""Online environment stiffness estimation + critical-damping admittance (Yanan Li / pHRI).

Coupled contact model (normal axis):

    m_d · ẍ + b_d · ẋ + K_e · x = F_ext

Damping ratio ζ = b_d / (2√(m_d K_e)).  Holding ζ fixed while K_e changes requires:

    b_d(t) = 2 ζ √(m_d · K̂_e(t))

**Scan caveat:** K_e = ΔF/Δx is only valid on the *normal admittance coordinate*
(∫v_force_z dt). Optional gates (scan velocity, ΔF spike) default off so K̂_e can
track stiffness during lateral sweeps; EWMA + asymmetric forgetting filter ripple.

References:
  - Yanan Li & Ge, IEEE T-CST 2014 — impedance learning
  - Yanan Li et al., IEEE TSMC 2022 — proactive HRI (Ḟ as intention; complements Kdf)
  - Keemink et al., IJRR 2018 — fixed M/D admittance tuning (G4 virtual mass)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rsc


@dataclass
class AdaptiveKeConfig:
    enabled: bool = False
    zeta: float = 1.0
    ke_initial: float = 500.0
    ke_forgetting: float = 0.995       # λ when surface softens (slow forget)
    ke_forgetting_inc: float = 0.80    # λ when surface stiffens (fast track)
    ke_min: float = 0.0
    ke_max: float = 50000.0
    dx_threshold_m: float = 1e-4
    contact_force_n: float = 0.5
    bd_max: float = 800.0
    bd_min: float = 0.0
    bd_slew_max: float = 2000.0
    ke_slew_max: float = 8000.0
    displacement_source: str = "admittance"
    scan_vel_gate_m_s: float = 0.002   # ignored when gate_scan_velocity=false
    df_spike_n: float = 6.0            # ignored when gate_df_spike=false
    f_err_gate_n: float = 4.0
    gate_scan_velocity: bool = False   # if true, freeze K̂_e during lateral scan
    gate_df_spike: bool = False        # if true, reject |ΔF| spikes

    @classmethod
    def from_dict(cls, raw: dict, parent: dict) -> AdaptiveKeConfig:
        a = raw.get("adaptive_ke", parent.get("adaptive_ke", {}))
        if not isinstance(a, dict):
            a = {}
        return cls(
            enabled=bool(a.get("enabled", parent.get("adaptive_ke_enabled", False))),
            zeta=float(a.get("zeta", parent.get("adaptive_zeta", 1.0))),
            ke_initial=float(a.get("ke_initial", parent.get("ke_initial", 500.0))),
            ke_forgetting=float(a.get("ke_forgetting", parent.get("ke_forgetting", 0.995))),
            ke_forgetting_inc=float(
                a.get("ke_forgetting_inc", parent.get("ke_forgetting_inc", 0.80))
            ),
            ke_min=float(a.get("ke_min", parent.get("ke_min", 0.0))),
            ke_max=float(a.get("ke_max", parent.get("ke_max", 50000.0))),
            dx_threshold_m=float(a.get("dx_threshold_m", parent.get("ke_dx_threshold_m", 1e-4))),
            contact_force_n=float(
                a.get("contact_force_n", parent.get("adaptive_contact_force_n", 0.5))
            ),
            bd_max=float(a.get("bd_max", parent.get("adaptive_bd_max", 800.0))),
            bd_min=float(a.get("bd_min", parent.get("adaptive_bd_min", 0.0))),
            bd_slew_max=float(a.get("bd_slew_max", parent.get("adaptive_bd_slew_max", 2000.0))),
            ke_slew_max=float(a.get("ke_slew_max", parent.get("ke_slew_max", 8000.0))),
            displacement_source=str(
                a.get("displacement_source", parent.get("ke_displacement_source", "admittance"))
            ).lower(),
            scan_vel_gate_m_s=float(
                a.get("scan_vel_gate_m_s", parent.get("ke_scan_vel_gate_m_s", 0.002))
            ),
            df_spike_n=float(a.get("df_spike_n", parent.get("ke_df_spike_n", 6.0))),
            f_err_gate_n=float(a.get("f_err_gate_n", parent.get("ke_f_err_gate_n", 4.0))),
            gate_scan_velocity=bool(a.get("gate_scan_velocity", False)),
            gate_df_spike=bool(a.get("gate_df_spike", False)),
        )


class EnvironmentStiffnessEstimator:
    """EWMA stiffness from ΔF/Δx on the normal admittance axis; outputs b_d for target ζ."""

    def __init__(self, cfg: AdaptiveKeConfig, *, dt: float, mass_z: float = 3.0) -> None:
        self.cfg = cfg
        self.dt = max(dt, 1e-6)
        self._mass_z = max(mass_z, 1e-3)
        self.ke_est = float(cfg.ke_initial)
        self.bd = self._critical_bd(self._mass_z)
        self._x_adm = 0.0
        self._last_f_z = 0.0
        self._last_x = 0.0
        self._have_prev = False
        self._contact_ref_pose: np.ndarray | None = None
        self._in_contact = False
        self._update_gated = False

    def reset(self) -> None:
        self.ke_est = float(self.cfg.ke_initial)
        self.bd = self._critical_bd(self._mass_z)
        self._x_adm = 0.0
        self._last_f_z = 0.0
        self._last_x = 0.0
        self._have_prev = False
        self._contact_ref_pose = None
        self._in_contact = False
        self._update_gated = False

    def _critical_bd(self, mass_z: float) -> float:
        ke = max(self.ke_est, self.cfg.ke_min)
        bd = 2.0 * self.cfg.zeta * math.sqrt(max(mass_z, 1e-3) * ke)
        lo = self.cfg.bd_min if self.cfg.bd_min > 0.0 else 0.0
        return float(np.clip(bd, lo, self.cfg.bd_max))

    @staticmethod
    def tool_z_displacement_m(
        pose: np.ndarray,
        ref_pose: np.ndarray,
        *,
        euler_order: str = "xyz",
    ) -> float:
        pose = np.asarray(pose, dtype=float)
        ref = np.asarray(ref_pose, dtype=float)
        d_base = pose[:3] - ref[:3]
        r_mat = Rsc.from_euler(euler_order, pose[3:6], degrees=False).as_matrix()
        return float((r_mat.T @ d_base)[2])

    def _normal_displacement_m(
        self,
        pose: np.ndarray,
        *,
        v_force_z: float,
        euler_order: str = "xyz",
    ) -> float:
        if self.cfg.displacement_source == "pose" and self._contact_ref_pose is not None:
            return self.tool_z_displacement_m(pose, self._contact_ref_pose, euler_order=euler_order)
        self._x_adm += float(v_force_z) * self.dt
        return self._x_adm

    def _should_update_ke(
        self,
        f_ext_z: float,
        f_err_z: float,
        v_scan_tool_y: float,
        df: float,
    ) -> bool:
        cfg = self.cfg
        if abs(f_ext_z) < cfg.contact_force_n:
            return False
        if abs(f_err_z) > cfg.f_err_gate_n:
            return False
        if cfg.gate_scan_velocity and abs(v_scan_tool_y) > cfg.scan_vel_gate_m_s:
            return False
        if cfg.gate_df_spike and abs(df) > cfg.df_spike_n:
            return False
        return True

    def _slew_ke(self, ke_target: float) -> float:
        max_dke = self.cfg.ke_slew_max * self.dt
        delta = float(np.clip(ke_target - self.ke_est, -max_dke, max_dke))
        return self.ke_est + delta

    def update(
        self,
        f_ext_z: float,
        pose: np.ndarray,
        *,
        in_contact: bool,
        mass_z: float,
        v_force_z: float = 0.0,
        v_scan_tool_y: float = 0.0,
        f_err_z: float = 0.0,
        euler_order: str = "xyz",
    ) -> tuple[float, float]:
        """Return (ke_est, bd) after one tick."""
        cfg = self.cfg
        self._mass_z = max(mass_z, 1e-3)
        if not cfg.enabled:
            return self.ke_est, self.bd

        if in_contact and not self._in_contact:
            self._contact_ref_pose = np.asarray(pose, dtype=float).copy()
            self._x_adm = 0.0
            self._have_prev = False

        if not in_contact:
            self._in_contact = False
            self._contact_ref_pose = None
            self._x_adm = 0.0
            self._have_prev = False
            self._update_gated = False
            self.ke_est = max(cfg.ke_min, cfg.ke_initial * 0.5)
            bd_target = self._critical_bd(mass_z)
            self.bd = self._slew_damping(bd_target)
            return self.ke_est, self.bd

        self._in_contact = True
        if self._contact_ref_pose is None:
            self._contact_ref_pose = np.asarray(pose, dtype=float).copy()

        x = self._normal_displacement_m(pose, v_force_z=v_force_z, euler_order=euler_order)

        gated = True
        if self._have_prev:
            df = f_ext_z - self._last_f_z
            dx = x - self._last_x
            gated = not self._should_update_ke(f_ext_z, f_err_z, v_scan_tool_y, df)
            if not gated and abs(dx) >= cfg.dx_threshold_m:
                ke_inst = abs(df / dx)
                ke_inst = float(np.clip(ke_inst, cfg.ke_min, cfg.ke_max))
                lam = (
                    cfg.ke_forgetting_inc if ke_inst > self.ke_est else cfg.ke_forgetting
                )
                ke_target = lam * self.ke_est + (1.0 - lam) * ke_inst
                self.ke_est = self._slew_ke(ke_target)

        self._update_gated = gated
        self._last_f_z = f_ext_z
        self._last_x = x
        self._have_prev = True

        bd_target = self._critical_bd(mass_z)
        self.bd = self._slew_damping(bd_target)
        return self.ke_est, self.bd

    def _slew_damping(self, bd_target: float) -> float:
        max_dbd = self.cfg.bd_slew_max * self.dt
        delta = float(np.clip(bd_target - self.bd, -max_dbd, max_dbd))
        return self.bd + delta

    @property
    def zeta_eff(self) -> float:
        denom = 2.0 * math.sqrt(max(self._mass_z, 1e-3) * max(self.ke_est, self.cfg.ke_min))
        if denom < 1e-9:
            return 0.0
        return self.bd / denom

    @property
    def update_gated(self) -> bool:
        return self._update_gated
