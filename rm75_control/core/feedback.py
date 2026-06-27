"""Background pose feedback polling (keeps CANFD stream loop non-blocking)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Callable, Sequence

Pose6 = list[float]
ReadPoseFn = Callable[[], Pose6 | None]


@dataclass
class StampedPose:
    pose: Pose6
    t_mono: float


class PoseFeedbackPoller:
    """Poll pose in a daemon thread; control loop reads cached stamp without blocking."""

    def __init__(
        self,
        read_pose: ReadPoseFn,
        *,
        period_s: float = 0.02,
    ) -> None:
        self._read_pose = read_pose
        self._period_s = max(period_s, 0.005)
        self._lock = Lock()
        self._stamp = StampedPose([0.0] * 6, time.monotonic())
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self, initial: Sequence[float]) -> None:
        self._stamp = StampedPose([float(v) for v in initial], time.monotonic())
        self._stop.clear()
        self._thread = Thread(target=self._run, name="pose-feedback", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get(self) -> StampedPose:
        with self._lock:
            return StampedPose(list(self._stamp.pose), self._stamp.t_mono)

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            pose = self._read_pose()
            if pose is not None:
                with self._lock:
                    self._stamp = StampedPose(pose, time.monotonic())
            elapsed = time.monotonic() - t0
            delay = self._period_s - elapsed
            if delay > 0.0:
                self._stop.wait(delay)
