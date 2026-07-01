"""Joint-space inner loop (Pinocchio CLIK / QP IK) for RM75-F.

Cascaded controller: the task-space admittance outer loop
(rm75_control.control.hybrid_motion.controller.AdmittanceController) produces a
6D Cartesian twist, and this package converts it to absolute joint angles that
are streamed through a single interface (rm_movej_canfd) - no MoveJ/MoveV mode
switching.

Imports are kept lazy: `import rm75_control.control.joint_admittance` does NOT
pull in Pinocchio.  Import the submodules explicitly when you need them, e.g.::

    from rm75_control.control.joint_admittance.model import RobotKinematics
    from rm75_control.control.joint_admittance.clik import ClikController
"""

from __future__ import annotations

__all__ = [
    "RobotKinematics",
    "ClikController",
    "ClikConfig",
    "JointIkController",
]


def __getattr__(name: str):  # PEP 562 lazy re-export (avoids importing pinocchio eagerly)
    if name in ("RobotKinematics",):
        from rm75_control.control.joint_admittance.model import RobotKinematics

        return RobotKinematics
    if name in ("ClikController", "ClikConfig"):
        from rm75_control.control.joint_admittance import clik

        return getattr(clik, name)
    if name in ("JointIkController",):
        from rm75_control.control.joint_admittance.loop import JointIkController

        return JointIkController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
