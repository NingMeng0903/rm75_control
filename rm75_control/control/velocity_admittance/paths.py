"""Paths for velocity admittance configs (under tmp/Velocity_Admittance)."""

from __future__ import annotations

from pathlib import Path

from rm75_control.force.compensation.paths import CONFIG_FORCE, CONFIG_ROBOT, PHI_JSON, REPO

VA_DATA_DIR = REPO / "tmp" / "Velocity_Admittance"
CONFIG_DIR = VA_DATA_DIR / "config"
DEMO_CONFIG_DIR = VA_DATA_DIR / "demo" / "config"
CONFIG_ADMITTANCE = CONFIG_DIR / "admittance.yaml"
CONFIG_SIN_TOOL_Y_Z2N = DEMO_CONFIG_DIR / "sin_tool_y_z2n.yaml"
