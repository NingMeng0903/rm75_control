"""Map poses.yaml slot TCP into the active tool frame (FK from q_deg)."""

from __future__ import annotations

import numpy as np

# Scan / admittance standoff: poses.yaml slot d is force-ID Arm_Tip at contact;
# runtime scan pose D is +220 mm along tool +Z from that teach pose (not FK(q_deg)).
DEFAULT_SCAN_APPROACH_DZ_M = 0.220


def get_active_tool_name(robot) -> str:
    ret, cur = robot.rm_get_current_tool_frame()
    if ret != 0:
        return ""
    return str(cur.get("name", ""))


def poses_calib_tool_frame(poses_data: dict, *, default: str = "Arm_Tip") -> str:
    return str(poses_data.get("pose_tool_frame", default))


def slot_tcp_pose(
    robot,
    q_deg: np.ndarray,
    pose_stored: np.ndarray,
    *,
    calib_tool: str,
) -> np.ndarray:
    """
    TCP pose in base frame for the **active** tool at slot ``q_deg``.

    ``poses.yaml`` ``pose_base`` is recorded with ``calib_tool`` active (e.g. Arm_Tip).
    When the Web UI active tool differs (e.g. gripper, ~220 mm offset on RM75-6F),
    ``state.pose`` and stored ``pose_base`` disagree at the same ``q_deg`` — use FK.
    """
    q_deg = np.asarray(q_deg, dtype=float)
    pose_stored = np.asarray(pose_stored, dtype=float)
    active = get_active_tool_name(robot)
    if active and calib_tool and active != calib_tool:
        fk = robot.rm_algo_forward_kinematics(q_deg.tolist(), flag=1)
        return np.asarray(fk[:6], dtype=float)
    return pose_stored.copy()


def tool_frame_delta_pose(
    robot,
    pose_ref: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Apply a translation delta in the tool frame of ``pose_ref`` (Realman frameMode=1)."""
    delta = [float(dx), float(dy), float(dz), 0.0, 0.0, 0.0]
    out = robot.rm_algo_pose_move(list(np.asarray(pose_ref, dtype=float)), delta, frameMode=1)
    return np.asarray(out[:6], dtype=float)


def slot_scan_approach_pose(
    robot,
    pose_arm_tip: np.ndarray,
    *,
    approach_dz_m: float = DEFAULT_SCAN_APPROACH_DZ_M,
) -> np.ndarray:
    """
    Scan standoff pose D from a force-ID slot teach pose.

    ``poses.yaml`` slot ``d`` ``pose_base`` is saved with ``Arm_Tip`` at the
    contact / identification tip.  The velocity-admittance scan startup pose D is
    **+220 mm along tool +Z** (outward, away from tissue) — not the raw teach
    pose and not ``FK(q_deg)`` with ``gripper`` active (that lands on the tip).
    """
    return tool_frame_delta_pose(robot, pose_arm_tip, 0.0, 0.0, approach_dz_m)
