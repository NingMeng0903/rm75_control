"""Safety layer for direct joint-position streaming.

When you bypass MoveJ's built-in S-curve planner and push q_cmd straight into
rm_movej_canfd, the motor drivers will fault (over-current / following error) on
any discontinuity.  This module enforces, per tick, in order:

  1. velocity limit : |dq| <= v_max * dt          (per-frame dq clamp)
  2. acceleration   : |dq - dq_prev| <= a_max*dt^2 (jerk-free enough for CANFD)
  3. position limit : q in [q_lower+margin, q_upper-margin]

plus a Watchdog thread that trips (freeze / slow-stop) if the control loop stops
feeding heartbeats - so a stuck Python process can never leave the arm coasting.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class SafetyLimits:
    q_lower: np.ndarray
    q_upper: np.ndarray
    v_max: np.ndarray                       # rad/s (per joint)
    a_max: np.ndarray | None = None         # rad/s^2 (per joint); None disables accel clamp
    position_margin: float = 0.017          # ~1 deg back-off from hard limit

    @classmethod
    def from_kinematics(
        cls,
        kin,
        *,
        v_scale: float = 1.0,
        a_max: np.ndarray | float | None = None,
        position_margin: float = 0.017,
    ) -> "SafetyLimits":
        v_max = np.asarray(kin.v_max, dtype=float) * float(v_scale)
        if a_max is not None and np.isscalar(a_max):
            a_max = np.full_like(v_max, float(a_max))
        return cls(
            q_lower=np.asarray(kin.q_lower, dtype=float),
            q_upper=np.asarray(kin.q_upper, dtype=float),
            v_max=v_max,
            a_max=None if a_max is None else np.asarray(a_max, dtype=float),
            position_margin=position_margin,
        )


@dataclass
class SafetyReport:
    q_safe: np.ndarray
    dq: np.ndarray
    vel_clamped: bool = False
    acc_clamped: bool = False
    pos_clamped: bool = False


class SafetyLimiter:
    """Stateful per-tick clamp: velocity -> acceleration -> position."""

    def __init__(self, limits: SafetyLimits) -> None:
        self.lim = limits
        self._dq_prev: np.ndarray | None = None

    def reset(self, q0: np.ndarray | None = None) -> None:
        self._dq_prev = None

    def clamp(self, q_prev: np.ndarray, q_desired: np.ndarray, dt: float) -> SafetyReport:
        lim = self.lim
        q_prev = np.asarray(q_prev, dtype=float)
        q_desired = np.asarray(q_desired, dtype=float)
        dq = q_desired - q_prev

        vel_clamped = acc_clamped = pos_clamped = False

        # 1) velocity limit
        dq_max = lim.v_max * dt
        clipped = np.clip(dq, -dq_max, dq_max)
        if not np.allclose(clipped, dq):
            vel_clamped = True
        dq = clipped

        # 2) acceleration limit (change in dq between ticks)
        if lim.a_max is not None and self._dq_prev is not None:
            ddq_max = lim.a_max * dt * dt
            ddq = dq - self._dq_prev
            ddq_c = np.clip(ddq, -ddq_max, ddq_max)
            if not np.allclose(ddq_c, ddq):
                acc_clamped = True
            dq = self._dq_prev + ddq_c

        q_safe = q_prev + dq

        # 3) position limit
        lo = lim.q_lower + lim.position_margin
        hi = lim.q_upper - lim.position_margin
        q_clamped = np.clip(q_safe, lo, hi)
        if not np.allclose(q_clamped, q_safe):
            pos_clamped = True
            dq = q_clamped - q_prev
        q_safe = q_clamped

        self._dq_prev = dq
        return SafetyReport(
            q_safe=q_safe,
            dq=dq,
            vel_clamped=vel_clamped,
            acc_clamped=acc_clamped,
            pos_clamped=pos_clamped,
        )


class Watchdog:
    """Independent heartbeat monitor.

    The control loop calls `beat()` every tick.  If no beat arrives within
    `timeout_s`, the watchdog fires `on_stall` exactly once (e.g. slow-stop the
    arm / latch a hold).  Runs as a daemon thread so it survives a stuck loop.
    """

    def __init__(
        self,
        timeout_s: float,
        on_stall: Callable[[], None],
        *,
        poll_s: float = 0.005,
        name: str = "ja-watchdog",
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.on_stall = on_stall
        self.poll_s = float(poll_s)
        self._name = name
        self._last_beat = time.perf_counter()
        self._stop = threading.Event()
        self._fired = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def beat(self) -> None:
        with self._lock:
            self._last_beat = time.perf_counter()
            # allow re-arming after a transient recovery
            self._fired.clear()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._last_beat = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def fired(self) -> bool:
        return self._fired.is_set()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                dt = time.perf_counter() - self._last_beat
            if dt > self.timeout_s and not self._fired.is_set():
                self._fired.set()
                try:
                    self.on_stall()
                except Exception:
                    pass
            time.sleep(self.poll_s)

    def __enter__(self) -> "Watchdog":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
