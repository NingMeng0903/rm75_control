"""Shared paths for force-ID data and configs (under tmp/force_compensation)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "tmp" / "force_compensation"
CONFIG_DIR = DATA_DIR / "config"
LOG_DIR = DATA_DIR / "logs"
CONFIG_ROBOT = REPO / "configs" / "rm75f_default.yaml"
CONFIG_FORCE = REPO / "configs" / "force_sensor.yaml"
CONFIG_ID = CONFIG_DIR / "force_id.yaml"
POSES_YAML = CONFIG_DIR / "poses.yaml"
PHI_JSON = LOG_DIR / "force_id_phi.json"

POSE_SLOTS = ("a", "b", "c", "d")


def npz_for_slot(slot: str) -> Path:
    return LOG_DIR / f"force_id_pose_{slot}.npz"
