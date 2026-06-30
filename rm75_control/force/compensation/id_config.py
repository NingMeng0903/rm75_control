"""Load config/force_id.yaml into typed settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from rm75_control.motion.canfd import TRAJ0_MODE, TRAJ0_RADIO

from .paths import CONFIG_ID, LOG_DIR, REPO, npz_for_slot


@dataclass(frozen=True)
class VelocityBurstConfig:
    profile: str
    amp_deg_s: float
    freqs_hz: list[float]
    segment_s: float
    ramp_s: float
    ramp_down_s: float
    frame_type: int
    avoid_singularity: int
    follow: bool
    trajectory_mode: int
    radio: int
    axis_order: tuple[int, int, int]


# Validated pose D rm_movev_canfd burst (base frame, traj=0 passthrough + init settle).
POSE_D_VEL_BURST: dict = {
    "amp_deg_s": 12.0,
    "freqs_hz": [0.28],
    "segment_s": 15.0,
    "ramp_s": 3.0,
    "ramp_down_s": 4.0,
    "frame_type": 1,
    "avoid_singularity": 0,
    "follow": True,
    "trajectory_mode": 0,
    "radio": 0,
    "axis_order": (0, 1, 2),
}

BURST_PROFILES: dict[str, dict] = {
    "pose_d_vel_burst": POSE_D_VEL_BURST,
}

DEFAULT_BURST_PROFILE = "pose_d_vel_burst"


def load_velocity_burst(raw: dict) -> VelocityBurstConfig:
    name = str(raw.get("profile", DEFAULT_BURST_PROFILE))
    if name not in BURST_PROFILES:
        raise ValueError(
            f"Unknown pose_d.velocity_burst.profile {name!r}; "
            f"choose from {list(BURST_PROFILES)}"
        )
    base = BURST_PROFILES[name]
    overrides = {
        k: raw[k]
        for k in (
            "amp_deg_s", "freqs_hz", "segment_s", "ramp_s", "ramp_down_s",
            "frame_type", "avoid_singularity", "follow", "trajectory_mode",
            "radio", "axis_order",
        )
        if k in raw
    }
    if "axis_order" in overrides:
        overrides["axis_order"] = tuple(int(x) for x in overrides["axis_order"])
    if "freqs_hz" in overrides:
        overrides["freqs_hz"] = [float(x) for x in overrides["freqs_hz"]]
    p = {**base, **overrides}
    p["trajectory_mode"] = TRAJ0_MODE
    p["radio"] = TRAJ0_RADIO
    return VelocityBurstConfig(
        profile=name,
        amp_deg_s=float(p["amp_deg_s"]),
        freqs_hz=list(p["freqs_hz"]),
        segment_s=float(p["segment_s"]),
        ramp_s=float(p["ramp_s"]),
        ramp_down_s=float(p.get("ramp_down_s", 4.0)),
        frame_type=int(p["frame_type"]),
        avoid_singularity=int(p["avoid_singularity"]),
        follow=bool(p["follow"]),
        trajectory_mode=int(p["trajectory_mode"]),
        radio=int(p["radio"]),
        axis_order=tuple(int(x) for x in p["axis_order"]),
    )


@dataclass(frozen=True)
class CartesianConfig:
    duration_s: float
    max_delta_mm: float
    max_orient_deg: dict[str, float]
    amp_mm: np.ndarray
    amp_rot_deg: np.ndarray
    amp_rot_deg_slots: dict[str, np.ndarray]
    amp_mm_slots: dict[str, np.ndarray]
    freqs_hz: list[list[float]]
    ramp_down_s: float

    def max_deg_for_slot(self, slot: str) -> float:
        return float(self.max_orient_deg.get(slot, self.max_orient_deg.get("a", 18.0)))

    def amp_rot_for_slot(self, slot: str) -> np.ndarray:
        if slot in self.amp_rot_deg_slots:
            return self.amp_rot_deg_slots[slot]
        return self.amp_rot_deg

    def amp_mm_for_slot(self, slot: str) -> np.ndarray:
        if slot in self.amp_mm_slots:
            return self.amp_mm_slots[slot]
        return self.amp_mm


@dataclass(frozen=True)
class PoseDConfig:
    joint_duration_s: float
    burst_duration_s: float
    joint_amp_deg: np.ndarray
    joint_max_delta_deg: np.ndarray
    joint_freqs_hz: list[list[float]]
    velocity_burst: VelocityBurstConfig


@dataclass(frozen=True)
class CollectConfig:
    move_speed: int
    settle_timeout_s: float
    dt_ms: float
    log_every: int
    scale: float
    warmup_s: float
    follow: bool
    cartesian_ramp_down_s: float
    movev_settle_frames: int
    movev_quiescent_mm: float
    cartesian: CartesianConfig
    pose_d: PoseDConfig
    sequence: tuple[str, ...]
    return_home: str


@dataclass(frozen=True)
class FitConfig:
    force_sensor: Path
    holdout_frac: float
    alpha_percentile: float
    min_burst_rows: int
    min_high_alpha_rows: int
    inertia_r_max_m: float
    npz_paths: list[Path]
    phi_output: Path
    phi_recommended_key: str


@dataclass(frozen=True)
class MonitorConfig:
    poll_ms: float
    window_s: float
    buffer_s: float
    min_samples: int
    refresh_hz: float
    phi_source: str
    use_inertia: bool


@dataclass(frozen=True)
class ForceIdConfig:
    poses_yaml: Path
    log_dir: Path
    collect: CollectConfig
    fit: FitConfig
    monitor: MonitorConfig


def _resolve_path(path: str, *, config_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "configs":
        return REPO / p
    return config_dir / p


def load_config(path: Path | None = None) -> ForceIdConfig:
    path = path or CONFIG_ID
    config_dir = path.parent
    raw = yaml.safe_load(path.read_text()) or {}

    c = raw.get("collect", {})
    cart = c.get("cartesian", {})
    pd = c.get("pose_d", {})
    br = pd.get("velocity_burst") or {}
    if not br:
        raise ValueError("pose_d.velocity_burst required (profile: pose_d_vel_burst)")
    rot_slots = cart.get("amp_rot_deg_slots", {})
    mm_slots = cart.get("amp_mm_slots", {})
    f = raw.get("fit", {})
    m = raw.get("monitor", {})

    sequence = tuple(str(s) for s in raw.get("sequence", ["a", "b", "c", "d"]))
    slots = f.get("npz_slots", list(sequence))
    phi_name = f.get("phi_output", "force_id_phi.json")

    return ForceIdConfig(
        poses_yaml=_resolve_path(raw.get("poses_yaml", "poses.yaml"), config_dir=config_dir),
        log_dir=LOG_DIR,
        collect=CollectConfig(
            move_speed=int(c.get("move_speed", 15)),
            settle_timeout_s=float(c.get("settle_timeout_s", 15.0)),
            dt_ms=float(c.get("dt_ms", 10.0)),
            log_every=int(c.get("log_every", 10)),
            scale=float(c.get("scale", 1.0)),
            warmup_s=float(c.get("warmup_s", 3.0)),
            follow=bool(c.get("follow", False)),
            cartesian_ramp_down_s=float(c.get("cartesian_ramp_down_s", cart.get("ramp_down_s", 2.5))),
            movev_settle_frames=int(c.get("movev_settle_frames", 30)),
            movev_quiescent_mm=float(c.get("movev_quiescent_mm", 0.3)),
            sequence=sequence,
            return_home=str(raw.get("return_home", "a")),
            cartesian=CartesianConfig(
                duration_s=float(cart.get("duration_s", 30.0)),
                max_delta_mm=float(cart.get("max_delta_mm", 5.0)),
                max_orient_deg={str(k): float(v) for k, v in cart.get("max_orient_deg", {}).items()},
                amp_mm=np.asarray(cart.get("amp_mm", [3, 4, 2]), dtype=float),
                amp_rot_deg=np.asarray(cart.get("amp_rot_deg", [12, 15, 12]), dtype=float),
                amp_rot_deg_slots={
                    str(k): np.asarray(v, dtype=float)
                    for k, v in rot_slots.items()
                },
                amp_mm_slots={
                    str(k): np.asarray(v, dtype=float)
                    for k, v in mm_slots.items()
                },
                freqs_hz=[list(map(float, row)) for row in cart.get("freqs_hz", [])],
                ramp_down_s=float(cart.get("ramp_down_s", c.get("cartesian_ramp_down_s", 2.5))),
            ),
            pose_d=PoseDConfig(
                joint_duration_s=float(pd.get("joint_duration_s", 30.0)),
                burst_duration_s=float(pd.get("burst_duration_s", 45.0)),
                joint_amp_deg=np.asarray(pd.get("joint_amp_deg", [10] * 7), dtype=float),
                joint_max_delta_deg=np.asarray(pd.get("joint_max_delta_deg", [12] * 7), dtype=float),
                joint_freqs_hz=[list(map(float, row)) for row in pd.get("joint_freqs_hz", [])],
                velocity_burst=load_velocity_burst(br),
            ),
        ),
        fit=FitConfig(
            force_sensor=_resolve_path(f.get("force_sensor", "configs/force_sensor.yaml"), config_dir=config_dir),
            holdout_frac=float(f.get("holdout_frac", 0.2)),
            alpha_percentile=float(f.get("alpha_percentile", 70.0)),
            min_burst_rows=int(f.get("min_burst_rows", 300)),
            min_high_alpha_rows=int(f.get("min_high_alpha_rows", 150)),
            inertia_r_max_m=float(f.get("inertia_r_max_m", 0.12)),
            npz_paths=[npz_for_slot(str(s)) for s in slots],
            phi_output=LOG_DIR / phi_name,
            phi_recommended_key=str(f.get("phi_recommended_key", "phi_burst")),
        ),
        monitor=MonitorConfig(
            poll_ms=float(m.get("poll_ms", 50.0)),
            window_s=float(m.get("window_s", 25.0)),
            buffer_s=float(m.get("buffer_s", 4.0)),
            min_samples=int(m.get("min_samples", 35)),
            refresh_hz=float(m.get("refresh_hz", 12.0)),
            phi_source=str(m.get("phi_source", "phi_recommended")),
            use_inertia=bool(m.get("use_inertia", True)),
        ),
    )
