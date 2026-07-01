"""RM75 FK / Jacobian entry point.

The real implementation lives in
rm75_control.control.joint_admittance.model.RobotKinematics (Pinocchio-based).
This module keeps a stable import path and a lazy loader so that importing the
kinematics package does not require Pinocchio unless a model is actually built.
"""

from __future__ import annotations


def load_kinematics(*args, **kwargs):
    """Build the Pinocchio-backed RobotKinematics (imported lazily)."""
    from rm75_control.control.joint_admittance.model import RobotKinematics

    return RobotKinematics(*args, **kwargs)


def __getattr__(name: str):
    if name == "RobotKinematics":
        from rm75_control.control.joint_admittance.model import RobotKinematics

        return RobotKinematics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
