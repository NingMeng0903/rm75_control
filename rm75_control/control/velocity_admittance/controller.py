"""Decoupled first-order admittance + position closed-loop (velocity output)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rsc


def wrap_pi(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def pose_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    err = np.asarray(desired, dtype=float) - np.asarray(current, dtype=float)
    for i in range(3, 6):
        err[i] = wrap_pi(err[i])
    return err


@dataclass
class AdmittanceConfig:
    euler_order: str = "xyz"
    force_axes: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    )
    motion_axes: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    )
    lock_orientation: bool = True
    kp_pos: np.ndarray = field(
        default_factory=lambda: np.array([8.0, 8.0, 0.0, 5.0, 5.0, 5.0])
    )
    k_fp_press: float = 0.015
    k_fp_release: float = 0.005
    k_fi: float = 0.008
    integral_limit: float = 0.05
    k_align: float = 0.02
    enable_normal_tracking: bool = False
    contact_threshold_n: float = 0.5
    deadband_n: float = 0.3
    max_velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.2, 0.08, 0.5, 0.5, 0.5])
    )
    max_acceleration: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 0.8, 2.0, 2.0, 2.0])
    )
    release_vz_up_m_s: float = 0.02
    release_vz_down_m_s: float = 0.02
    max_vz_tool_m_s: float = 0.05  # hard cap on tool-frame linear Z (TCP normal)

    @classmethod
    def from_dict(cls, raw: dict) -> AdmittanceConfig:
        c = raw.get("controller", raw)
        frames = raw.get("frames", {})
        fa = np.asarray(c.get("force_axes", [0, 0, 1, 0, 0, 0]), dtype=float)
        ma = np.asarray(c.get("motion_axes", [0, 1, 0, 0, 0, 0]), dtype=float)
        return cls(
            euler_order=str(frames.get("euler_order", "xyz")),
            force_axes=fa,
            motion_axes=ma,
            lock_orientation=bool(c.get("lock_orientation", True)),
            kp_pos=np.asarray(c.get("kp_pos", [8, 8, 0, 5, 5, 5]), dtype=float),
            k_fp_press=float(c.get("k_fp_press", 0.015)),
            k_fp_release=float(c.get("k_fp_release", 0.005)),
            k_fi=float(c.get("k_fi", 0.008)),
            integral_limit=float(c.get("integral_limit", 0.05)),
            k_align=float(c.get("k_align", 0.02)),
            enable_normal_tracking=bool(c.get("enable_normal_tracking", True)),
            contact_threshold_n=float(c.get("contact_threshold_n", 0.5)),
            deadband_n=float(c.get("deadband_n", 0.3)),
            max_velocity=np.asarray(
                c.get("max_velocity", [0.2, 0.2, 0.08, 0.5, 0.5, 0.5]), dtype=float
            ),
            max_acceleration=np.asarray(
                c.get("max_acceleration", [1.0, 1.0, 0.8, 2.0, 2.0, 2.0]), dtype=float
            ),
            release_vz_up_m_s=float(c.get("release_vz_up_m_s", 0.05)),
            release_vz_down_m_s=float(c.get("release_vz_down_m_s", 0.05)),
            max_vz_tool_m_s=float(c.get("max_vz_tool_m_s", 0.05)),
        )


class AdmittanceController:
    """
    Hybrid admittance-position controller (PBAC-style, Keemink et al.).

    Position loop + feedforward run in base frame; force admittance in sensor/tool
    frame. Commands are merged in tool frame, then rotated back to base for
    rm_movev_canfd (frame_type=1).
    """

    def __init__(self, dt: float, config: AdmittanceConfig | None = None) -> None:
        self.dt = dt
        self.cfg = config or AdmittanceConfig()
        self.force_error_integral = np.zeros(6)
        self.last_v_cmd = np.zeros(6)

    def reset(self) -> None:
        self.force_error_integral.fill(0.0)
        self.last_v_cmd.fill(0.0)

    def _selection(self, *, normal_track: bool) -> tuple[np.ndarray, np.ndarray]:
        s_f = np.diag(self.cfg.force_axes.copy())
        if (
            normal_track
            and self.cfg.enable_normal_tracking
            and not self.cfg.lock_orientation
        ):
            s_f[3, 3] = 1.0
            s_f[4, 4] = 1.0
        s_p = np.eye(6) - s_f
        return s_p, s_f

    def compute_velocity_command(
        self,
        current_pose: np.ndarray,
        desired_pose: np.ndarray,
        desired_vel_ff: np.ndarray,
        f_ext: np.ndarray,
        desired_force: np.ndarray,
        pose_anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        cfg = self.cfg
        desired_pose = np.asarray(desired_pose, dtype=float).copy()
        if pose_anchor is not None and cfg.lock_orientation:
            desired_pose[3:6] = np.asarray(pose_anchor, dtype=float)[3:6]

        err_pose = pose_error(desired_pose, current_pose)
        vel_ff = np.asarray(desired_vel_ff, dtype=float).copy()
        if cfg.lock_orientation:
            vel_ff[3:6] = 0.0
        v_pos_base = vel_ff + cfg.kp_pos * err_pose

        r_mat = Rsc.from_euler(
            cfg.euler_order, current_pose[3:6], degrees=False
        ).as_matrix()

        v_pos_tool = np.zeros(6, dtype=float)
        v_pos_tool[:3] = r_mat.T @ v_pos_base[:3]
        v_pos_tool[3:] = r_mat.T @ v_pos_base[3:]

        f_ext = np.asarray(f_ext, dtype=float)
        f_des = np.asarray(desired_force, dtype=float)
        v_force_tool = np.zeros(6, dtype=float)

        in_contact = float(np.linalg.norm(f_ext[:3])) >= cfg.contact_threshold_n

        for axis in range(6):
            if cfg.force_axes[axis] < 0.5:
                continue
            f_err = f_des[axis] - f_ext[axis]
            v_force_tool[axis] = self._admittance_axis(axis, f_err, in_contact)

        normal_track = in_contact and cfg.enable_normal_tracking
        if normal_track:
            v_force_tool[3] = -cfg.k_align * f_ext[1]
            v_force_tool[4] = cfg.k_align * f_ext[0]

        s_p, s_f = self._selection(normal_track=normal_track)
        v_cmd_tool = s_p @ v_pos_tool + s_f @ v_force_tool
        v_cmd_tool[3:6] = (s_p @ v_pos_tool)[3:6]
        vz_cap = cfg.max_vz_tool_m_s
        if vz_cap > 0.0:
            v_cmd_tool[2] = float(np.clip(v_cmd_tool[2], -vz_cap, vz_cap))

        v_cmd_base = np.zeros(6, dtype=float)
        v_cmd_base[:3] = r_mat @ v_cmd_tool[:3]
        v_cmd_base[3:] = r_mat @ v_cmd_tool[3:]

        v_clamp = np.clip(v_cmd_base, -cfg.max_velocity, cfg.max_velocity)
        dv_max = cfg.max_acceleration * self.dt
        v_final = np.clip(v_clamp, self.last_v_cmd - dv_max, self.last_v_cmd + dv_max)
        self.last_v_cmd = v_final.copy()
        return v_final

    def _admittance_axis(self, axis: int, f_err: float, in_contact: bool) -> float:
        cfg = self.cfg
        if axis == 2 and not in_contact:
            self.force_error_integral[axis] = 0.0
            v = cfg.k_fp_release * f_err
            return float(np.clip(v, -cfg.release_vz_down_m_s, cfg.release_vz_up_m_s))

        if abs(f_err) > 5.0:
            self.force_error_integral[axis] = 0.0

        k_fp = cfg.k_fp_press if f_err < 0 else cfg.k_fp_release
        if abs(f_err) > cfg.deadband_n:
            eff = f_err - math.copysign(cfg.deadband_n, f_err)
            self.force_error_integral[axis] += eff * self.dt
            self.force_error_integral[axis] = float(
                np.clip(self.force_error_integral[axis], -cfg.integral_limit, cfg.integral_limit)
            )
            v = k_fp * eff + cfg.k_fi * self.force_error_integral[axis]
        else:
            v = cfg.k_fi * self.force_error_integral[axis]
        if axis == 2:
            v = float(np.clip(v, -cfg.max_vz_tool_m_s, cfg.max_vz_tool_m_s))
        return v
