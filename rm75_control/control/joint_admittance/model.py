"""Pinocchio kinematics engine for RM75-F (FK / Jacobian / manipulability).

The whole cascade is only as correct as this model.  Two conventions are pinned
here and must match the Realman controller:

* Joint order  : joint_1..joint_7, radians internally.  The robot API speaks
  degrees (rm_get_current_arm_state()["joint"], rm_movej_canfd) - convert at the
  boundary with deg2rad / rad2deg helpers.
* Cartesian    : the TCP twist / Jacobian are expressed LOCAL_WORLD_ALIGNED,
  i.e. linear velocity of the TCP point and angular velocity, both in base-frame
  axes.  This matches the base-frame 6D twist the admittance outer loop emits
  (controller.py, control_frame="base").  Pose is returned as
  [x, y, z, rx, ry, rz] with intrinsic xyz Euler (Realman convention).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rsc

DEFAULT_URDF = (
    Path(__file__).resolve().parents[2] / "assets" / "robots" / "rm75_6f" / "RM75-6F.urdf"
)

JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]


def deg2rad(q_deg: np.ndarray) -> np.ndarray:
    return np.asarray(q_deg, dtype=float) * (np.pi / 180.0)


def rad2deg(q_rad: np.ndarray) -> np.ndarray:
    return np.asarray(q_rad, dtype=float) * (180.0 / np.pi)


def pose_distance(
    pose_a: np.ndarray, pose_b: np.ndarray, euler_order: str = "xyz"
) -> tuple[float, float]:
    """Position distance (mm) and orientation distance (deg) between two pose6."""
    a = np.asarray(pose_a, dtype=float)
    b = np.asarray(pose_b, dtype=float)
    d_mm = float(np.linalg.norm(a[:3] - b[:3]) * 1000.0)
    ra = Rsc.from_euler(euler_order, a[3:6], degrees=False).as_matrix()
    rb = Rsc.from_euler(euler_order, b[3:6], degrees=False).as_matrix()
    d_deg = float(np.degrees(np.linalg.norm(Rsc.from_matrix(ra @ rb.T).as_rotvec())))
    return d_mm, d_deg


def pose_error(desired: np.ndarray, current: np.ndarray, euler_order: str = "xyz") -> np.ndarray:
    """Base-frame 6D pose error: linear diff + SO(3) log (rotvec of R_des @ R_cur^T).

    Mirrors hybrid_motion.controller.pose_error so the inner loop's Cartesian
    error definition is identical to the outer loop's.
    """
    err = np.zeros(6, dtype=float)
    err[:3] = np.asarray(desired[:3], dtype=float) - np.asarray(current[:3], dtype=float)
    r_des = Rsc.from_euler(euler_order, desired[3:6], degrees=False).as_matrix()
    r_cur = Rsc.from_euler(euler_order, current[3:6], degrees=False).as_matrix()
    err[3:6] = Rsc.from_matrix(r_des @ r_cur.T).as_rotvec()
    return err


class RobotKinematics:
    """Thin Pinocchio wrapper exposing FK, Jacobian and manipulability at the TCP."""

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        tcp_frame: str = "tcp",
        euler_order: str = "xyz",
    ) -> None:
        self.urdf_path = Path(urdf_path) if urdf_path is not None else DEFAULT_URDF
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.euler_order = euler_order

        if not self.model.existFrame(tcp_frame):
            raise ValueError(f"frame {tcp_frame!r} not in URDF {self.urdf_path}")
        self.tcp_frame = tcp_frame
        self.tcp_id = self.model.getFrameId(tcp_frame)

        self.nq = self.model.nq
        self.nv = self.model.nv
        if self.nq != 7 or self.nv != 7:
            raise ValueError(f"expected 7-DOF model, got nq={self.nq} nv={self.nv}")

        # Position / velocity limits (radians, rad/s) straight from the URDF.
        self.q_lower = np.asarray(self.model.lowerPositionLimit, dtype=float).copy()
        self.q_upper = np.asarray(self.model.upperPositionLimit, dtype=float).copy()
        self.v_max = np.asarray(self.model.velocityLimit, dtype=float).copy()

    # ---- forward kinematics ------------------------------------------------
    def fk_placement(self, q_rad: np.ndarray) -> pin.SE3:
        q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.tcp_id)
        return self.data.oMf[self.tcp_id]

    def fk_pose(self, q_rad: np.ndarray) -> np.ndarray:
        """TCP pose as [x, y, z, rx, ry, rz] (m, rad; intrinsic xyz Euler)."""
        M = self.fk_placement(q_rad)
        pose = np.zeros(6, dtype=float)
        pose[:3] = M.translation
        pose[3:6] = Rsc.from_matrix(M.rotation).as_euler(self.euler_order, degrees=False)
        return pose

    def fk_position_quat(self, q_rad: np.ndarray) -> np.ndarray:
        """TCP pose as [x, y, z, qx, qy, qz, qw] (handy for logging / comparisons)."""
        M = self.fk_placement(q_rad)
        quat = Rsc.from_matrix(M.rotation).as_quat()  # [x, y, z, w]
        return np.concatenate([M.translation, quat])

    def frame_placement(self, q_rad: np.ndarray, frame_name: str) -> pin.SE3:
        """SE3 of an arbitrary frame (e.g. 'link_7' flange) in the base frame."""
        if not self.model.existFrame(frame_name):
            raise ValueError(f"frame {frame_name!r} not in URDF {self.urdf_path}")
        fid = self.model.getFrameId(frame_name)
        q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, fid)
        return self.data.oMf[fid]

    def frame_pose(self, q_rad: np.ndarray, frame_name: str) -> np.ndarray:
        """Pose [x, y, z, rx, ry, rz] of an arbitrary frame in the base frame."""
        M = self.frame_placement(q_rad, frame_name)
        pose = np.zeros(6, dtype=float)
        pose[:3] = M.translation
        pose[3:6] = Rsc.from_matrix(M.rotation).as_euler(self.euler_order, degrees=False)
        return pose

    # ---- differential kinematics ------------------------------------------
    def jacobian(self, q_rad: np.ndarray) -> np.ndarray:
        """6x7 TCP Jacobian, LOCAL_WORLD_ALIGNED (linear on top, angular below).

        Maps joint velocity (rad/s) -> [v_lin(base), omega(base)].
        """
        q = np.asarray(q_rad, dtype=float)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.getFrameJacobian(
            self.model, self.data, self.tcp_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return np.asarray(J, dtype=float)

    @staticmethod
    def manipulability(J: np.ndarray) -> float:
        """Yoshikawa measure sqrt(det(J J^T)); 0 at a singularity."""
        JJt = J @ J.T
        det = float(np.linalg.det(JJt))
        return float(np.sqrt(max(det, 0.0)))

    @staticmethod
    def singular_values(J: np.ndarray) -> np.ndarray:
        return np.linalg.svd(J, compute_uv=False)

    def clamp_to_limits(self, q_rad: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return np.clip(q_rad, self.q_lower + margin, self.q_upper - margin)
