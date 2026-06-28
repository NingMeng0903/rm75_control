"""Background pose/force polling — keeps the control loop non-blocking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AsyncStateSnapshot:
    pose: np.ndarray | None = None
    force_raw: np.ndarray = field(default_factory=lambda: np.zeros(6))
    t_s: float = 0.0
    ok: bool = False


class AsyncStateObserver:
    """
    Poll rm_get_current_arm_state / rm_get_force_data in a daemon thread.
    Main loop reads latest snapshot without blocking on RPC.
    """

    def __init__(self, robot, *, poll_s: float = 0.002) -> None:
        self.robot = robot
        self.poll_s = poll_s
        self._lock = threading.Lock()
        self._snap = AsyncStateSnapshot()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="va-async-state")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def wait_first_pose(self, timeout_s: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snap = self.read()
            if snap.pose is not None:
                return snap.pose.copy()
            time.sleep(0.005)
        raise TimeoutError("AsyncStateObserver: no pose within timeout")

    def read(self) -> AsyncStateSnapshot:
        with self._lock:
            if self._snap.pose is None:
                return AsyncStateSnapshot(
                    force_raw=self._snap.force_raw.copy(),
                    t_s=self._snap.t_s,
                    ok=False,
                )
            return AsyncStateSnapshot(
                pose=self._snap.pose.copy(),
                force_raw=self._snap.force_raw.copy(),
                t_s=self._snap.t_s,
                ok=self._snap.ok,
            )

    def _loop(self) -> None:
        while self._running:
            t_s = time.monotonic()
            ret_s, st = self.robot.rm_get_current_arm_state()
            ret_f, fd = self.robot.rm_get_force_data()
            snap = AsyncStateSnapshot(t_s=t_s)
            if ret_s == 0:
                snap.pose = np.asarray(st["pose"][:6], dtype=float)
            if ret_f == 0:
                snap.force_raw = np.asarray(fd["force_data"][:6], dtype=float)
            snap.ok = snap.pose is not None and ret_f == 0
            with self._lock:
                if snap.pose is not None:
                    self._snap.pose = snap.pose
                if ret_f == 0:
                    self._snap.force_raw = snap.force_raw
                self._snap.t_s = t_s
                self._snap.ok = snap.ok
            time.sleep(self.poll_s)
