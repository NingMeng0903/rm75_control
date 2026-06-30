"""Demo wiring: built-in trajectories + optional reference shaper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rm75_control.control.hybrid_motion.reference import MotionReferenceSource
from rm75_control.control.hybrid_motion.reference_shaper import ReferenceShaper, build_shaper

_DEMO_DIR = Path(__file__).resolve().parent / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from trajectory_builtin import trajectory_source_factory, trajectory_summary  # noqa: E402


def build_demo_source_factory(raw: dict) -> Callable[[np.ndarray, Any], MotionReferenceSource]:
    return trajectory_source_factory(raw)


def build_demo_shaper(raw: dict) -> ReferenceShaper:
    return build_shaper(raw)


def demo_trajectory_summary(raw: dict) -> str:
    return trajectory_summary(raw)
