"""Optional reference shaping between external commands and the hybrid controller."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .reference import MotionReference


class ReferenceShaper(Protocol):
    def reset(self, pose0: np.ndarray) -> None: ...

    def step(self, raw: MotionReference, dt: float) -> MotionReference: ...


class PassThroughShaper:
    """Default: no shaping — raw reference goes straight to the controller."""

    def reset(self, pose0: np.ndarray) -> None:
        del pose0

    def step(self, raw: MotionReference, dt: float) -> MotionReference:
        del dt
        return raw


def build_shaper(raw: dict) -> ReferenceShaper:
    ref_cfg = raw.get("reference", {})
    kind = str(ref_cfg.get("shaper", "pass_through")).lower().replace("-", "_")
    if kind in ("pass_through", "passthrough", "none", ""):
        return PassThroughShaper()
    if kind == "ruckig":
        raise NotImplementedError(
            "RuckigReferenceShaper is reserved for low-rate / waypoint sources — not wired yet"
        )
    raise ValueError(f"Unknown reference shaper: {kind!r}")
