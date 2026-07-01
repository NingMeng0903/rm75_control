"""Demo / test trajectory sources (sin, hold). Not part of the hybrid_motion core."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as Rsc
from scipy.spatial.transform import Slerp

from rm75_control.control.hybrid_motion.reference import MotionReference, MotionReferenceSource
from rm75_control.force.compensation.collection import load_slot
from rm75_control.force.compensation.id_config import load_config
from rm75_control.force.compensation.paths import CONFIG_ID
from rm75_control.force.compensation.tool_pose import (
    get_active_tool_name,
    poses_calib_tool_frame,
)
from rm75_control.force.compensation import excitation as ex


def tool_frame_delta_pose(
    pose_ref: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    *,
    euler_order: str = "xyz",
) -> np.ndarray:
    pose = np.asarray(pose_ref, dtype=float).copy()
    r_mat = Rsc.from_euler(euler_order, pose[3:6], degrees=False).as_matrix()
    pose[:3] = pose[:3] + r_mat @ np.array([dx, dy, dz], dtype=float)
    return pose


def anchor_slot_transfer_poses(
    actual: np.ndarray,
    yaml_from: np.ndarray,
    yaml_to: np.ndarray,
    *,
    euler_order: str = "xyz",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply slot d→a delta (translation + rotation) from the live TCP.

    ``poses.yaml`` pose_base often disagrees with rm_movej(q_deg) FK. Raw yaml
    endpoints make pose_d ~220 mm away at scan ON → pitch snap, speed fault, lost
    contact. The relative d→a smoothstep (pos + Slerp rot) is unchanged.
    """
    actual = np.asarray(actual, dtype=float)
    y0 = np.asarray(yaml_from, dtype=float)
    y1 = np.asarray(yaml_to, dtype=float)

    pose_from = actual.copy()
    pose_to = actual.copy()
    pose_to[:3] = actual[:3] + (y1[:3] - y0[:3])

    r0 = Rsc.from_euler(euler_order, y0[3:6], degrees=False)
    r1 = Rsc.from_euler(euler_order, y1[3:6], degrees=False)
    r_act = Rsc.from_euler(euler_order, actual[3:6], degrees=False)
    r_to = (r1 * r0.inv()) * r_act
    pose_to[3:6] = r_to.as_euler(euler_order, degrees=False)
    return pose_from, pose_to


def tool_offset_pose(robot, ref_pose: list[float], dx: float, dy: float, dz: float) -> list[float]:
    delta = [dx, dy, dz, 0.0, 0.0, 0.0]
    return robot.rm_algo_pose_move(ref_pose, delta, frameMode=1)


def sin_period_for_peak_vel(amplitude_m: float, max_vel_m_s: float) -> float:
    if amplitude_m <= 0.0 or max_vel_m_s <= 0.0:
        return 1.0
    return 2.0 * math.pi * amplitude_m / max_vel_m_s


def sin_y_motion(
    t_s: float,
    amplitude_m: float,
    omega: float,
    *,
    soft_start: bool,
    ramp_s: float = 2.0,
) -> tuple[float, float]:
    dy = amplitude_m * math.sin(omega * t_s)
    vy = amplitude_m * omega * math.cos(omega * t_s)
    if soft_start and ramp_s > 0.0 and t_s < ramp_s:
        vy *= math.sin(0.5 * math.pi * t_s / ramp_s)
    return dy, vy


def tool_z_spin_angle_rad(
    t_s: float, *, rz_amp_deg: float, omega: float, soft_start: bool, ramp_s: float,
) -> float:
    if rz_amp_deg <= 0.0:
        return 0.0
    ramp = 1.0
    if soft_start and ramp_s > 0.0 and t_s < ramp_s:
        ramp = math.sin(0.5 * math.pi * t_s / ramp_s)
    return math.radians(rz_amp_deg) * math.sin(omega * t_s) * ramp


def apply_tool_z_spin_pose(pose_ref: np.ndarray, phi_rad: float) -> np.ndarray:
    pose = np.asarray(pose_ref, dtype=float).copy()
    if abs(phi_rad) < 1e-12:
        return pose
    r0 = Rsc.from_euler("xyz", pose_ref[3:6], degrees=False).as_matrix()
    axis = r0[:, 2]
    r_d = Rsc.from_rotvec(axis * phi_rad).as_matrix() @ r0
    pose[3:6] = Rsc.from_matrix(r_d).as_euler("xyz", degrees=False)
    return pose


def tool_z_spin_vel_base(
    pose_ref: np.ndarray,
    t_s: float,
    *,
    rz_amp_deg: float,
    omega: float,
    soft_start: bool,
    ramp_s: float,
) -> np.ndarray:
    if rz_amp_deg <= 0.0:
        return np.zeros(3)
    ramp = 1.0
    if soft_start and ramp_s > 0.0 and t_s < ramp_s:
        ramp = math.sin(0.5 * math.pi * t_s / ramp_s)
    wz_tool = math.radians(rz_amp_deg) * omega * math.cos(omega * t_s) * ramp
    r_mat = Rsc.from_euler("xyz", pose_ref[3:6], degrees=False).as_matrix()
    return r_mat @ np.array([0.0, 0.0, wz_tool], dtype=float)


def interpolate_pose_smoothstep(
    pose_start: np.ndarray,
    pose_end: np.ndarray,
    t_s: float,
    duration_s: float,
    *,
    euler_order: str = "xyz",
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothstep pose blend with analytic translational + rotational vel_ff."""
    pose_start = np.asarray(pose_start, dtype=float)
    pose_end = np.asarray(pose_end, dtype=float)
    if duration_s <= 0.0:
        return pose_end.copy(), np.zeros(6, dtype=float)

    u = float(np.clip(t_s / duration_s, 0.0, 1.0))
    s = u * u * (3.0 - 2.0 * u)
    ds_dt = 6.0 * u * (1.0 - u) / duration_s

    pose = np.zeros(6, dtype=float)
    pose[:3] = (1.0 - s) * pose_start[:3] + s * pose_end[:3]

    r0 = Rsc.from_euler(euler_order, pose_start[3:6], degrees=False)
    r1 = Rsc.from_euler(euler_order, pose_end[3:6], degrees=False)
    rot_s = Slerp([0.0, 1.0], Rsc.concatenate([r0, r1]))([s])[0]
    pose[3:6] = rot_s.as_euler(euler_order, degrees=False)

    vel = np.zeros(6, dtype=float)
    vel[:3] = ds_dt * (pose_end[:3] - pose_start[:3])

    ds = 1e-4
    s2 = min(s + ds, 1.0)
    if s2 > s:
        rot_s2 = Slerp([0.0, 1.0], Rsc.concatenate([r0, r1]))([s2])[0]
        vel[3:6] = (rot_s2 * rot_s.inv()).as_rotvec() / (s2 - s) * ds_dt
    return pose, vel


@dataclass
class TrajectoryConfig:
    kind: str = "hold"
    amplitude_mm: float = 5.0
    y_peak_to_peak_cm: float | None = None
    period_s: float | None = None
    y_max_vel_cm_s: float = 1.0
    soft_start: bool = False
    ramp_s: float = 2.0
    rz_amplitude_deg: float = 0.0
    from_slot: str = "d"
    to_slot: str = "a"
    transfer_s: float = 12.0
    scan_mode: str = "sin_tool_y"
    euler_order: str = "xyz"

    @property
    def half_amplitude_m(self) -> float:
        if self.y_peak_to_peak_cm is not None:
            return float(self.y_peak_to_peak_cm) * 0.01 / 2.0
        return self.amplitude_mm / 1000.0


class BuiltinTrajectorySource:
    """Built-in sin/hold demos implementing MotionReferenceSource."""

    def __init__(self, cfg: TrajectoryConfig, pose0: np.ndarray, robot) -> None:
        self.cfg = cfg
        self.pose0 = np.asarray(pose0, dtype=float)
        self.robot = robot
        amp_m = cfg.half_amplitude_m
        if cfg.period_s is None:
            period = sin_period_for_peak_vel(amp_m, cfg.y_max_vel_cm_s / 100.0)
        else:
            period = float(cfg.period_s)
        self.omega = 2.0 * math.pi / period if period > 0 else 0.0
        self.amplitude_m = amp_m
        self._pose_from: np.ndarray | None = None
        self._pose_to: np.ndarray | None = None
        self._yaml_pose_from: np.ndarray | None = None
        self._yaml_pose_to: np.ndarray | None = None
        self._yaml_tcp_drift_mm: float = 0.0
        if cfg.kind == "slot_transfer_sin_tool_y":
            self._load_slot_poses()
            self._anchor_transfer(self.pose0)

    def _load_slot_poses(self) -> None:
        fid = load_config(CONFIG_ID)
        poses_data = ex.load_poses_yaml(fid.poses_yaml)
        calib_tool = poses_calib_tool_frame(poses_data)
        active = get_active_tool_name(self.robot) if self.robot is not None else ""
        if active and calib_tool and active != calib_tool:
            print(
                f"  slot d→a TCP: FK(q_deg) in active tool {active!r} "
                f"(poses.yaml recorded in {calib_tool!r})",
                flush=True,
            )
        _, pose_from, rec_from = load_slot(
            fid, self.cfg.from_slot, self.robot, calib_tool=calib_tool,
        )
        _, pose_to, rec_to = load_slot(
            fid, self.cfg.to_slot, self.robot, calib_tool=calib_tool,
        )
        self._yaml_pose_from = np.asarray(pose_from, dtype=float)
        self._yaml_pose_to = np.asarray(pose_to, dtype=float)
        self._slot_labels = (
            str(rec_from.get("label", self.cfg.from_slot)),
            str(rec_to.get("label", self.cfg.to_slot)),
        )

    def _anchor_transfer(self, actual: np.ndarray) -> None:
        if self._yaml_pose_from is None or self._yaml_pose_to is None:
            return
        actual = np.asarray(actual, dtype=float)
        self._pose_from, self._pose_to = anchor_slot_transfer_poses(
            actual,
            self._yaml_pose_from,
            self._yaml_pose_to,
            euler_order=self.cfg.euler_order,
        )
        self._yaml_tcp_drift_mm = float(
            np.linalg.norm((actual[:3] - self._yaml_pose_from[:3]) * 1000.0)
        )

    @property
    def kind(self) -> str:
        return self.cfg.kind

    def set_origin(self, pose0: np.ndarray) -> None:
        self.pose0 = np.asarray(pose0, dtype=float).copy()
        if self.cfg.kind == "slot_transfer_sin_tool_y":
            self._anchor_transfer(self.pose0)

    def sample(self, t_s: float) -> MotionReference:
        kind = self.cfg.kind
        if kind == "hold":
            return MotionReference(self.pose0.copy(), np.zeros(6))

        if kind in ("sin_base_y", "sin_base_y_tool_rz"):
            return self._sin_base_y(t_s, spin=(kind == "sin_base_y_tool_rz"))

        if kind == "sin_tool_y":
            return self._sin_tool_y(t_s)

        if kind == "slot_transfer_sin_tool_y":
            return self._slot_transfer_sin_tool_y(t_s)

        raise ValueError(f"Unknown trajectory type: {kind}")

    def _sin_base_y(self, t_s: float, *, spin: bool) -> MotionReference:
        dy, vy = sin_y_motion(
            t_s, self.amplitude_m, self.omega,
            soft_start=self.cfg.soft_start, ramp_s=self.cfg.ramp_s,
        )
        pose = self.pose0.copy()
        pose[1] += dy
        if spin:
            phi = tool_z_spin_angle_rad(
                t_s,
                rz_amp_deg=self.cfg.rz_amplitude_deg,
                omega=self.omega,
                soft_start=self.cfg.soft_start,
                ramp_s=self.cfg.ramp_s,
            )
            pose = apply_tool_z_spin_pose(pose, phi)
        vel = np.zeros(6, dtype=float)
        vel[1] = vy
        if spin:
            vel[3:6] = tool_z_spin_vel_base(
                self.pose0, t_s,
                rz_amp_deg=self.cfg.rz_amplitude_deg,
                omega=self.omega,
                soft_start=self.cfg.soft_start,
                ramp_s=self.cfg.ramp_s,
            )
        return MotionReference(pose, vel, t_ref=t_s)

    def _sin_tool_y_at(
        self,
        t_s: float,
        origin: np.ndarray,
    ) -> MotionReference:
        dy, vy = sin_y_motion(
            t_s, self.amplitude_m, self.omega,
            soft_start=self.cfg.soft_start, ramp_s=self.cfg.ramp_s,
        )
        pose = np.asarray(
            tool_offset_pose(self.robot, list(origin), 0.0, dy, 0.0), dtype=float
        )
        r_mat = Rsc.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
        vel = np.zeros(6, dtype=float)
        vel[:3] = r_mat @ np.array([0.0, vy, 0.0], dtype=float)
        return MotionReference(pose, vel, t_ref=t_s)

    def _sin_tool_y(self, t_s: float) -> MotionReference:
        return self._sin_tool_y_at(t_s, self.pose0)

    def _slot_transfer_sin_tool_y(self, t_s: float) -> MotionReference:
        if self._pose_from is None or self._pose_to is None:
            raise RuntimeError("slot poses not loaded")

        transfer_s = float(self.cfg.transfer_s)
        if t_s < transfer_s:
            pose, vel = interpolate_pose_smoothstep(
                self._pose_from,
                self._pose_to,
                t_s,
                transfer_s,
                euler_order=self.cfg.euler_order,
            )
            return MotionReference(pose, vel, t_ref=t_s)

        t_scan = t_s - transfer_s
        if self.cfg.scan_mode == "sin_tool_y":
            return self._sin_tool_y_at(t_scan, self._pose_to)
        if self.cfg.scan_mode == "sin_base_y":
            return self._sin_base_y(t_scan, spin=False)
        raise ValueError(f"Unknown scan_mode: {self.cfg.scan_mode!r}")

    @classmethod
    def from_dict(cls, raw: dict, pose0: np.ndarray, robot) -> BuiltinTrajectorySource:
        t = raw.get("trajectory_demo", raw.get("trajectory", {}))
        ps = t.get("period_s")
        y_pp_cm = t.get("y_peak_to_peak_cm")
        frames = raw.get("frames", {})
        return cls(
            TrajectoryConfig(
                kind=str(t.get("type", "hold")),
                amplitude_mm=float(t.get("amplitude_mm", 5.0)),
                y_peak_to_peak_cm=float(y_pp_cm) if y_pp_cm is not None else None,
                period_s=float(ps) if ps is not None else None,
                y_max_vel_cm_s=float(t.get("y_max_vel_cm_s", 1.0)),
                soft_start=bool(t.get("soft_start", False)),
                ramp_s=float(t.get("ramp_s", 2.0)),
                rz_amplitude_deg=float(t.get("rz_amplitude_deg", 0.0)),
                from_slot=str(t.get("from_slot", "d")).lower(),
                to_slot=str(t.get("to_slot", "a")).lower(),
                transfer_s=float(t.get("transfer_s", 12.0)),
                scan_mode=str(t.get("scan_mode", "sin_tool_y")),
                euler_order=str(frames.get("euler_order", "xyz")),
            ),
            pose0,
            robot,
        )


# Backward-compatible names for scripts that still say TrajectoryGenerator.
TrajectoryGenerator = BuiltinTrajectorySource


def trajectory_summary(raw: dict) -> str:
    t = raw.get("trajectory_demo", raw.get("trajectory", {}))
    kind = str(t.get("type", "sin_tool_y"))
    y_pp = t.get("y_peak_to_peak_cm")
    if y_pp is not None:
        pp_mm = float(y_pp) * 10.0
        amp_label = f"Y p-p={float(y_pp):.1f}cm ({pp_mm:.0f}mm)"
    else:
        amp_mm = float(t.get("amplitude_mm", 5.0))
        amp_label = f"amp=±{amp_mm:.1f}mm ({2 * amp_mm:.0f}mm p-p)"
    vmax = float(t.get("y_max_vel_cm_s", 1.0))
    ps = t.get("period_s")
    half_m = float(y_pp) * 0.01 / 2.0 if y_pp is not None else float(t.get("amplitude_mm", 5.0)) / 1000.0
    if ps is None:
        period = sin_period_for_peak_vel(half_m, vmax / 100.0)
        period_s = f"{period:.1f}s (auto)"
    else:
        period_s = f"{float(ps):.1f}s"
    soft = " soft_start" if t.get("soft_start") else ""
    rz = float(t.get("rz_amplitude_deg", 0.0))
    spin_kinds = ("sin_base_y_tool_rz",)
    spin = f"  tool-Rz±{rz:.1f}°" if (kind in spin_kinds and rz > 0) else ""
    if kind == "slot_transfer_sin_tool_y":
        from_slot = str(t.get("from_slot", "d"))
        to_slot = str(t.get("to_slot", "a"))
        transfer_s = float(t.get("transfer_s", 12.0))
        scan_mode = str(t.get("scan_mode", "sin_tool_y"))
        return (
            f"{kind}  {from_slot}→{to_slot}  transfer={transfer_s:.1f}s  "
            f"then {scan_mode}{soft}  {amp_label}  v_peak≈{vmax:.1f}cm/s  period={period_s}"
        )
    return f"{kind}{soft}  {amp_label}{spin}  v_peak≈{vmax:.1f}cm/s  period={period_s}"


def trajectory_source_factory(raw: dict):
    """Return a (pose0, robot) → MotionReferenceSource factory for run_hybrid_motion_loop."""

    def factory(pose0: np.ndarray, robot: Any) -> MotionReferenceSource:
        return BuiltinTrajectorySource.from_dict(raw, pose0, robot)

    return factory
