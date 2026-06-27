"""Shared paths for tmp/force_id package (scripts + logs)."""

from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = PKG.parent.parent
LOG_DIR = PKG / "logs"
CONFIG_ROBOT = REPO / "configs" / "rm75f_default.yaml"
CONFIG_FORCE = REPO / "configs" / "force_sensor.yaml"
POSES_YAML = REPO / "configs" / "force_id_poses.yaml"
PHI_JSON = LOG_DIR / "force_id_phi.json"
DEFAULT_NPZ = LOG_DIR / "force_id_cartesian.npz"
