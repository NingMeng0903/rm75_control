"""Paths for tmp/Admittance."""

from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = PKG.parent.parent
CONFIG_DIR = PKG / "config"
CONFIG_ADMITTANCE = CONFIG_DIR / "admittance.yaml"
CONFIG_ROBOT = REPO / "configs" / "rm75f_default.yaml"
CONFIG_FORCE = REPO / "configs" / "force_sensor.yaml"
PHI_JSON = REPO / "tmp" / "force_compensation" / "logs" / "force_id_phi.json"
FORCE_COMP_UTILS = REPO / "tmp" / "force_compensation"
