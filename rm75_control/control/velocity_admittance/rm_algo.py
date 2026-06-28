"""RM algo helpers (pose structs for rm_algo_* calls)."""

from __future__ import annotations


def pose_to_rm_pose(pose: list[float]):
    from Robotic_Arm.rm_ctypes_wrap import rm_euler_t, rm_pose_t, rm_position_t

    po = rm_pose_t()
    po.position = rm_position_t(*pose[:3])
    po.euler = rm_euler_t(*pose[3:6])
    return po


def end2tool_pose(robot, pose6: list[float]) -> list[float]:
    return list(robot.rm_algo_end2tool(pose_to_rm_pose(pose6)))


def end2tool_xyz(robot, pose6: list[float]) -> list[float]:
    return end2tool_pose(robot, pose6)[:3]
