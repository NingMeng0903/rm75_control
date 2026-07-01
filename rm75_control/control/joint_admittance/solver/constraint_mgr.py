"""Per-tick box constraints on joint velocity for the QP inner loop.

Everything the safety layer enforces after the fact is expressed here as *hard*
inequality bounds on the decision variable qdot, so the QP respects them while
optimizing (rather than clipping an already-computed solution):

    qdot in [-v_max, v_max]                              (velocity)
    qdot in [(q_min+m - q)/dt, (q_max-m - q)/dt]         (position, look-ahead)
    qdot in [qdot_prev - a_max*dt, qdot_prev + a_max*dt] (acceleration)

The three boxes are intersected.  If a position bound and another bound cross
(the joint is already past a limit), the position bound wins and both l and u
collapse onto it, which drives the joint back inside next tick.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance.utils.safety import SafetyLimits


class VelocityBoxConstraints:
    def __init__(self, limits: SafetyLimits) -> None:
        self.lim = limits

    def bounds(
        self,
        q: np.ndarray,
        dt: float,
        qdot_prev: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        lim = self.lim
        q = np.asarray(q, dtype=float)

        lo = -lim.v_max.copy()
        hi = lim.v_max.copy()

        # position look-ahead
        m = lim.position_margin
        p_lo = (lim.q_lower + m - q) / dt
        p_hi = (lim.q_upper - m - q) / dt
        lo = np.maximum(lo, p_lo)
        hi = np.minimum(hi, p_hi)

        # acceleration
        if lim.a_max is not None and qdot_prev is not None:
            qdot_prev = np.asarray(qdot_prev, dtype=float)
            a = lim.a_max * dt
            lo = np.maximum(lo, qdot_prev - a)
            hi = np.minimum(hi, qdot_prev + a)

        # resolve crossings: position limit dominates
        crossed = lo > hi
        if np.any(crossed):
            mid = np.clip(0.0, p_lo, p_hi)  # bias toward staying in position range
            lo = np.where(crossed, np.minimum(p_lo, p_hi), lo)
            hi = np.where(crossed, np.maximum(p_lo, p_hi), hi)
            # if still crossed after using position bounds only, collapse to feasible point
            still = lo > hi
            if np.any(still):
                lo = np.where(still, mid, lo)
                hi = np.where(still, mid, hi)
        return lo, hi
