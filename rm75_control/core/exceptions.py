"""Shared exceptions for rm75_control."""

from __future__ import annotations


class RM75ControlError(Exception):
    """Base error for this package."""


class RobotConnectionError(RM75ControlError):
    """Failed to connect or lost connection."""


class ControlModeError(RM75ControlError):
    """Invalid control mode transition or concurrent mode usage."""


class MotionError(RM75ControlError):
    """Motion command rejected by robot or backend."""
