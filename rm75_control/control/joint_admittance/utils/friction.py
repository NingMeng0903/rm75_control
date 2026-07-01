"""Optional joint friction / stiction feed-forward (Phase 3, experimental).

RM harmonic-drive joints have noticeable static friction; at very low scan speed
the force loop can stick-slip while the joint fights breakaway.  With only a
position/velocity interface (no direct torque command) we cannot inject a true
torque compensation, so this applies a *velocity-level* nudge: a smooth Coulomb
+ viscous term that adds a small breakaway velocity in the direction of motion.

Disabled by default.  Enable and tune ONLY after the basic loop is stable, and
keep the gains tiny - excessive compensation causes limit-cycle chatter.

    dqdot = fc * tanh(qdot / v_eps) + fv * qdot
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrictionConfig:
    enabled: bool = False
    coulomb: np.ndarray = field(default_factory=lambda: np.zeros(7))   # rad/s breakaway nudge
    viscous: np.ndarray = field(default_factory=lambda: np.zeros(7))   # rad/s per rad/s
    v_eps: float = 0.02                                                # rad/s smoothing of sign()


class FrictionCompensator:
    def __init__(self, cfg: FrictionConfig | None = None) -> None:
        self.cfg = cfg or FrictionConfig()

    def __call__(self, qdot: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        if not cfg.enabled:
            return np.zeros_like(qdot)
        qdot = np.asarray(qdot, dtype=float)
        return cfg.coulomb * np.tanh(qdot / max(cfg.v_eps, 1e-6)) + cfg.viscous * qdot
