"""Excitation trajectories and pose YAML helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .id_config import BurstConfig, CartesianConfig, PoseDConfig

DEG2RAD = math.pi / 180.0


def load_poses_yaml(path: Path) -> dict:
    if not path.exists():
        return {"poses": {}}
    return yaml.safe_load(path.read_text()) or {"poses": {}}


def save_poses_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True))


def save_pose_slot(
    path: Path, slot: str, pose6: np.ndarray, q_deg: np.ndarray, label: str | None
) -> None:
    data = load_poses_yaml(path)
    data.setdefault("poses", {})[slot] = {
        "label": label or f"pose_{slot}",
        "note": f"saved {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "pose_base": [round(float(v), 6) for v in pose6],
        "q_deg": [round(float(v), 3) for v in q_deg],
    }
    save_poses_yaml(path, data)


def get_slot_record(data: dict, slot: str) -> dict | None:
    rec = data.get("poses", {}).get(slot)
    if not rec or rec.get("pose_base") is None:
        return None
    return rec


def pose_drift_mm_deg(current: np.ndarray, recorded: np.ndarray) -> tuple[float, float]:
    dpos = float(np.linalg.norm(current[:3] - recorded[:3])) * 1000.0
    deul = np.abs(current[3:6] - recorded[3:6])
    deul = np.minimum(deul, 2 * math.pi - deul)
    ddeg = float(np.max(deul) * 180.0 / math.pi)
    return dpos, ddeg


@dataclass(frozen=True)
class CartesianExcitation:
    amp_mm: np.ndarray
    amp_rot_deg: np.ndarray
    freqs_hz: list[list[float]]
    scale: float

    @classmethod
    def from_config(cls, cart: CartesianConfig, scale: float) -> CartesianExcitation:
        return cls(cart.amp_mm, cart.amp_rot_deg, cart.freqs_hz, scale)

    def delta_pose(self, t_s: float) -> np.ndarray:
        out = np.zeros(6, dtype=float)
        amps_m = self.amp_mm * self.scale / 1000.0
        amps_rad = self.amp_rot_deg * self.scale * DEG2RAD
        amp_list = [
            amps_m[0], amps_m[1], amps_m[2],
            amps_rad[0], amps_rad[1], amps_rad[2],
        ]
        for axis in range(6):
            for k, f in enumerate(self.freqs_hz[axis]):
                ph = (axis + 1) * 0.9 + k * 1.1
                out[axis] += amp_list[axis] * math.sin(2.0 * math.pi * f * t_s + ph)
        return out


def clamp_delta(
    delta: np.ndarray,
    *,
    max_mm: float | np.ndarray,
    max_rot_deg: float | np.ndarray,
) -> np.ndarray:
    out = delta.copy()
    mm = np.full(3, max_mm, dtype=float) if np.isscalar(max_mm) else np.asarray(max_mm, dtype=float)
    deg = np.full(3, max_rot_deg, dtype=float) if np.isscalar(max_rot_deg) else np.asarray(max_rot_deg, dtype=float)
    out[0:3] = np.clip(out[0:3], -mm / 1000.0, mm / 1000.0)
    cap = deg * DEG2RAD
    out[3:6] = np.clip(out[3:6], -cap, cap)
    return out


def inertia_burst_delta(t_s: float, burst: BurstConfig) -> np.ndarray:
    seg = int(t_s // burst.segment_s) % 3
    t_loc = t_s - seg * burst.segment_s
    axis = 3 + seg
    amp = burst.amp_rot_deg * DEG2RAD
    delta = np.zeros(6, dtype=float)
    for k, f in enumerate(burst.freqs_hz):
        ph = seg * 1.4 + k * 0.85
        delta[axis] += amp * math.sin(2.0 * math.pi * f * t_loc + ph)
    cap = burst.max_rot_deg * DEG2RAD
    delta[3:6] = np.clip(delta[3:6], -cap, cap)
    return delta


def joint_cmd(t: float, q0: np.ndarray, pd: PoseDConfig, scale: float) -> np.ndarray:
    q = q0.copy()
    for j in range(7):
        a = pd.joint_amp_deg[j] * scale
        for k, f in enumerate(pd.joint_freqs_hz[j]):
            ph = (j + 1) * 0.8 + k * 1.2
            q[j] += a * math.sin(2 * math.pi * f * t + ph)
    delta = np.clip(q - q0, -pd.joint_max_delta_deg, pd.joint_max_delta_deg)
    return q0 + delta


def preview_cartesian(cart: CartesianConfig, *, duration: float, scale: float, max_deg: float) -> dict:
    exc = CartesianExcitation.from_config(cart, scale)
    dt_s = 0.01
    ts = np.linspace(0, duration, int(duration / dt_s) + 1)
    deltas = np.array([
        clamp_delta(exc.delta_pose(t), max_mm=cart.max_delta_mm, max_rot_deg=max_deg) for t in ts
    ])
    return {
        "mm_max": (np.max(np.abs(deltas[:, 0:3]), axis=0) * 1000).tolist(),
        "rot_deg_max": (np.max(np.abs(deltas[:, 3:6]), axis=0) / DEG2RAD).tolist(),
    }


def preview_pose_d(q0: np.ndarray, pd: PoseDConfig, *, scale: float) -> dict:
    dt_s = 0.01
    ts_j = np.linspace(0, pd.joint_duration_s, int(pd.joint_duration_s / dt_s) + 1)
    qs = np.array([joint_cmd(t, q0, pd, scale) for t in ts_j])
    ts_b = np.linspace(0, pd.burst_duration_s, int(pd.burst_duration_s / dt_s) + 1)
    deltas = np.array([inertia_burst_delta(t, pd.burst) for t in ts_b])
    br = pd.burst
    v_peak = 2 * math.pi * max(br.freqs_hz) * br.max_rot_deg
    return {
        "joint_max_deg": np.max(np.abs(qs - q0), axis=0).tolist(),
        "j7_max_deg": float(np.max(np.abs(qs[:, 6] - q0[6]))),
        "burst_rot_deg_max": (np.max(np.abs(deltas[:, 3:6]), axis=0) / DEG2RAD).tolist(),
        "burst_peak_omega_deg_s": v_peak,
    }
