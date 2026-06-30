"""Online environment stiffness estimation + critical-damping admittance (Yanan Li / pHRI).

Coupled contact model (normal axis):

    m_d · ẍ + b_d · ẋ + K_e · x = F_ext

Damping ratio ζ = b_d / (2√(m_d K_e)).  Holding ζ fixed while K_e changes requires:

    b_d(t) = 2 ζ √(m_d · K̂_e(t))

References (environment / human impedance learning & variable admittance):
  - Yanan Li, S.S. Ge, "Impedance Learning for Robots Interacting With Unknown
    Environments", IEEE T-CST 22(4):1422-1432, 2014.
  - Yanan Li, S.S. Ge, C. Yang, "Learning impedance control for physical
    robot-environment interaction", Int. J. Control 85(2):182-193, 2011.
  - Yanan Li, S.S. Ge, C. Wang, "Impedance adaptation for optimal robot-
    environment interaction", Int. J. Control 87(2):249-263, 2013.
  - Hsieh-Yu Li et al., "Stable and Compliant Motion of pHRI Coupled With a
    Moving Environment Using Variable Admittance and Adaptive Control",
    IEEE RA-L 3(3):2493-2500, 2018 (cited in Li/Ge line of work).
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
    ke_forgetting: float = 0.92
    ke_min: float = 0.0
    ke_max: float = 50000.0
    dx_threshold_m: float = 1e-4
    contact_force_n: float = 0.5
    bd_max: float = 800.0
    bd_slew_max: float = 2000.0  # max |Δb_d/Δt| [N·s/m²]

    @classmethod
    def from_dict(cls, raw: dict, parent: dict) -> AdaptiveKeConfig:
        a = raw.get("adaptive_ke", parent.get("adaptive_ke", {}))
        if not isinstance(a, dict):
            a = {}
        return cls(
            enabled=bool(a.get("enabled", parent.get("adaptive_ke_enabled", False))),
            zeta=float(a.get("zeta", parent.get("adaptive_zeta", 1.0))),
            ke_initial=float(a.get("ke_initial", parent.get("ke_initial", 500.0))),
            ke_forgetting=float(a.get("ke_forgetting", parent.get("ke_forgetting", 0.92))),
            ke_min=float(a.get("ke_min", parent.get("ke_min", 0.0))),
            ke_max=float(a.get("ke_max", parent.get("ke_max", 50000.0))),
            dx_threshold_m=float(a.get("dx_threshold_m", parent.get("ke_dx_threshold_m", 1e-4))),
            contact_force_n=float(
                a.get("contact_force_n", parent.get("adaptive_contact_force_n", 0.5))
            ),
            bd_max=float(a.get("bd_max", parent.get("adaptive_bd_max", 800.0))),
            bd_slew_max=float(a.get("bd_slew_max", parent.get("adaptive_bd_slew_max", 2000.0))),
        )


class EnvironmentStiffnessEstimator:
    """
    EWMA stiffness from ΔF/Δx along tool-Z; outputs b_d for target ζ.

    x is tool-frame normal penetration relative to the pose captured at contact.
    """

    def __init__(self, cfg: AdaptiveKeConfig, *, dt: float, mass_z: float = 3.0) -> None:
        self.cfg = cfg
        self.dt = max(dt, 1e-6)
        self._mass_z = max(mass_z, 1e-3)
        self.ke_est = float(cfg.ke_initial)
        self.bd = self._critical_bd(self._mass_z)
        self._last_f_z = 0.0
        self._last_x_z = 0.0
        self._have_prev = False
        self._contact_ref_pose: np.ndarray | None = None
        self._in_contact = False

    def reset(self) -> None:
        self.ke_est = float(self.cfg.ke_initial)
        self.bd = self._critical_bd(self._mass_z)
        self._last_f_z = 0.0
        self._last_x_z = 0.0
        self._have_prev = False
        self._contact_ref_pose = None
        self._in_contact = False

    def _critical_bd(self, mass_z: float) -> float:
        ke = max(self.ke_est, self.cfg.ke_min)
        bd = 2.0 * self.cfg.zeta * math.sqrt(max(mass_z, 1e-3) * ke)
        return min(bd, self.cfg.bd_max)

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

    def update(
        self,
        f_ext_z: float,
        pose: np.ndarray,
        *,
        in_contact: bool,
        mass_z: float,
        euler_order: str = "xyz",
    ) -> tuple[float, float]:
        """Return (ke_est, bd) after one tick."""
        cfg = self.cfg
        self._mass_z = max(mass_z, 1e-3)
        if not cfg.enabled:
            return self.ke_est, self.bd

        if in_contact and not self._in_contact:
            self._contact_ref_pose = np.asarray(pose, dtype=float).copy()
            self._have_prev = False

        if not in_contact:
            self._in_contact = False
            self._contact_ref_pose = None
            self._have_prev = False
            self.ke_est = max(cfg.ke_min, 0.0)
            bd_target = self._critical_bd(mass_z)
            self.bd = self._slew_damping(bd_target)
            return self.ke_est, self.bd

        self._in_contact = True
        if self._contact_ref_pose is None:
            self._contact_ref_pose = np.asarray(pose, dtype=float).copy()

        x_z = self.tool_z_displacement_m(pose, self._contact_ref_pose, euler_order=euler_order)

        if abs(f_ext_z) >= cfg.contact_force_n and self._have_prev:
            dx = x_z - self._last_x_z
            df = f_ext_z - self._last_f_z
            if abs(dx) >= cfg.dx_threshold_m:
                ke_inst = abs(df / dx)
                ke_inst = float(np.clip(ke_inst, cfg.ke_min, cfg.ke_max))
                lam = cfg.ke_forgetting
                self.ke_est = lam * self.ke_est + (1.0 - lam) * ke_inst

        self._last_f_z = f_ext_z
        self._last_x_z = x_z
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
        """Achieved ζ given current ke_est, bd, and mass."""
        denom = 2.0 * math.sqrt(max(self._mass_z, 1e-3) * max(self.ke_est, self.cfg.ke_min))
        if denom < 1e-9:
            return 0.0
        return self.bd / denom
