"""Command smoothing for direct joint-position streaming.

Even a mathematically continuous q_cmd, sampled at 100-1000 Hz, carries
high-frequency content that shows up as motor current ripple.  These filters
smooth q_cmd before it hits rm_movej_canfd.  All are per-joint and stateful.

Provided:
* FirstOrderLowPass    - single-pole IIR (cheap, ~6 dB/oct).
* SecondOrderLowPass   - critically-damped 2nd order (S-curve-like step response,
                         no overshoot; ~12 dB/oct).
* MovingAverage        - boxcar window.

Cutoff is specified in Hz; alpha is derived from the control dt.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def alpha_from_cutoff(cutoff_hz: float, dt: float) -> float:
    """First-order IIR smoothing factor for a given cutoff and sample period."""
    if cutoff_hz <= 0.0:
        return 1.0  # no filtering
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return float(dt / (tau + dt))


class FirstOrderLowPass:
    def __init__(self, cutoff_hz: float, dt: float, dim: int = 7) -> None:
        self.alpha = alpha_from_cutoff(cutoff_hz, dt)
        self.dim = dim
        self._y: np.ndarray | None = None

    def reset(self, x0: np.ndarray | None = None) -> None:
        self._y = None if x0 is None else np.asarray(x0, dtype=float).copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._y is None:
            self._y = x.copy()
        else:
            self._y = self._y + self.alpha * (x - self._y)
        return self._y.copy()

    def sync(self, y: np.ndarray) -> None:
        """Force the filter output state to `y` (keep it aligned with the value
        actually sent after a downstream safety clamp)."""
        self._y = np.asarray(y, dtype=float).copy()


class SecondOrderLowPass:
    """Critically-damped-equivalent 2nd-order low-pass, monotone step response.

    Implemented as two cascaded first-order IIR stages at the SAME cutoff
    (two coincident real poles = the discrete analogue of critical damping).
    Each stage is `y += alpha*(x-y)` with `0 < alpha <= 1`, which is
    UNCONDITIONALLY STABLE for any cutoff_hz / dt (unlike an explicit-Euler
    integration of the continuous mass-spring-damper ODE, which only stays
    stable while omega*dt is small and produces large tracking lag / blows up
    numerically as cutoff approaches ~1/(2*dt)).  This lets you raise the
    cutoff at a fixed control dt without risking instability - the tradeoff
    between smoothing and ramp-tracking lag is then just alpha vs alpha^2.
    """

    def __init__(self, cutoff_hz: float, dt: float, dim: int = 7) -> None:
        self.cutoff_hz = cutoff_hz
        self.dt = dt
        self.dim = dim
        self.alpha = alpha_from_cutoff(cutoff_hz, dt)
        self._y1: np.ndarray | None = None
        self._y2: np.ndarray | None = None

    def reset(self, x0: np.ndarray | None = None) -> None:
        self._y1 = None if x0 is None else np.asarray(x0, dtype=float).copy()
        self._y2 = None if x0 is None else np.asarray(x0, dtype=float).copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._y1 is None:
            self._y1 = x.copy()
            self._y2 = x.copy()
            return self._y2.copy()
        self._y1 = self._y1 + self.alpha * (x - self._y1)
        self._y2 = self._y2 + self.alpha * (self._y1 - self._y2)
        return self._y2.copy()

    def sync(self, y: np.ndarray) -> None:
        """Re-seat both stages on the actually-sent value after a clamp."""
        y = np.asarray(y, dtype=float).copy()
        self._y1 = y.copy()
        self._y2 = y.copy()


class MovingAverage:
    def __init__(self, window: int, dim: int = 7) -> None:
        self.window = max(int(window), 1)
        self.dim = dim
        self._buf: deque[np.ndarray] = deque(maxlen=self.window)

    def reset(self, x0: np.ndarray | None = None) -> None:
        self._buf.clear()
        if x0 is not None:
            self._buf.append(np.asarray(x0, dtype=float).copy())

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        self._buf.append(x.copy())
        return np.mean(np.stack(self._buf, axis=0), axis=0)
