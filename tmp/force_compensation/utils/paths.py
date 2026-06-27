"""Shared paths for tmp/force_compensation."""

from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
REPO = PKG.parent.parent
CONFIG_DIR = PKG / "config"
LOG_DIR = PKG / "logs"
CONFIG_ROBOT = REPO / "configs" / "rm75f_default.yaml"
CONFIG_FORCE = REPO / "configs" / "force_sensor.yaml"
CONFIG_ID = CONFIG_DIR / "force_id.yaml"
POSES_YAML = CONFIG_DIR / "poses.yaml"
PHI_JSON = LOG_DIR / "force_id_phi.json"

POSE_SLOTS = ("a", "b", "c", "d")


def npz_for_slot(slot: str) -> Path:
    return LOG_DIR / f"force_id_pose_{slot}.npz"
