"""Tool-frame force/motion decoupling + base-frame 6D trajectory tracking."""

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


def smooth_deadband_eff(f_err: float, deadband_n: float, width_n: float) -> float:
    """
    C1 smooth deadband: zero inside |f|<=db, ramps to f-sign*db outside transition.
    Reduces PI limit cycles at the deadband edge (Z inertia ripple).
    """
    if width_n <= 0.0:
        if abs(f_err) <= deadband_n:
            return 0.0
        return f_err - math.copysign(deadband_n, f_err)
    af = abs(f_err)
    if af <= deadband_n:
        return 0.0
    if af >= deadband_n + width_n:
        return f_err - math.copysign(deadband_n + 0.5 * width_n, f_err)
    t = (af - deadband_n) / width_n
    gain = t * t * (3.0 - 2.0 * t)
    return math.copysign(gain * (af - deadband_n), f_err)


@dataclass
class AdmittanceConfig:
    """
    force_axes: tool-frame mask for admittance (typ. [0,0,1,0,0,0] = TCP normal).
    f_ext from phi is in sensor frame; with sensor_offset=0 and TCP pure translation,
    f_ext[2] is used as tool-Z force (see observer docstring).
    Trajectory pose_d / vel_ff are base-frame 6D (Servo / scan feedforward).
    Fusion is tool-frame sleeve decoupling only — no world-XY lstsq lock.
    """

    euler_order: str = "xyz"
    force_axes: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    )
    control_frame: str = "tool"
    kp_pos: np.ndarray = field(default_factory=lambda: np.zeros(6))
    track_axes: np.ndarray = field(default_factory=lambda: np.ones(6))
    system_delay_s: float = 0.015
    k_fp_press: float = 0.015
    k_fp_release: float = 0.005
    k_fi: float = 0.008
    integral_limit: float = 0.05
    k_align: float = 0.02
    enable_normal_tracking: bool = False
    contact_threshold_n: float = 0.5
    deadband_n: float = 0.3
    deadband_width_n: float = 0.2
    max_velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.2, 0.08, 0.5, 0.5, 0.5])
    )
    max_acceleration: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 0.8, 2.0, 2.0, 2.0])
    )
    release_vz_up_m_s: float = 0.02
    release_vz_down_m_s: float = 0.02
    approach_vz_tool_m_s: float = 0.03
    max_vz_tool_m_s: float = 0.05
    open_loop: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> AdmittanceConfig:
        c = raw.get("controller", raw)
        frames = raw.get("frames", {})
        traj = raw.get("trajectory", {})
        fa = np.asarray(c.get("force_axes", [0, 0, 1, 0, 0, 0]), dtype=float)
        open_loop = bool(c.get("open_loop", c.get("open_loop_scan", traj.get("open_loop", False))))
        return cls(
            euler_order=str(frames.get("euler_order", "xyz")),
            control_frame=str(frames.get("control_frame", c.get("control_frame", "tool"))),
            force_axes=fa,
            kp_pos=np.asarray(c.get("kp_pos", [0, 0, 0, 0, 0, 0]), dtype=float),
            track_axes=np.asarray(c.get("track_axes", [1, 1, 1, 1, 1, 1]), dtype=float),
            system_delay_s=float(c.get("system_delay_s", 0.015)),
            k_fp_press=float(c.get("k_fp_press", 0.015)),
            k_fp_release=float(c.get("k_fp_release", 0.005)),
            k_fi=float(c.get("k_fi", 0.008)),
            integral_limit=float(c.get("integral_limit", 0.05)),
            k_align=float(c.get("k_align", 0.02)),
            enable_normal_tracking=bool(c.get("enable_normal_tracking", False)),
            contact_threshold_n=float(c.get("contact_threshold_n", 0.5)),
            deadband_n=float(c.get("deadband_n", 0.3)),
            deadband_width_n=float(c.get("deadband_width_n", 0.2)),
            max_velocity=np.asarray(
                c.get("max_velocity", [0.2, 0.2, 0.08, 0.5, 0.5, 0.5]), dtype=float
            ),
            max_acceleration=np.asarray(
                c.get("max_acceleration", [1.0, 1.0, 0.8, 2.0, 2.0, 2.0]), dtype=float
            ),
            release_vz_up_m_s=float(c.get("release_vz_up_m_s", 0.05)),
            release_vz_down_m_s=float(c.get("release_vz_down_m_s", 0.05)),
            approach_vz_tool_m_s=float(c.get("approach_vz_tool_m_s", 0.03)),
            max_vz_tool_m_s=float(c.get("max_vz_tool_m_s", 0.05)),
            open_loop=open_loop,
        )


class AdmittanceController:
    """
    Pipeline (base trajectory/Servo → sleeve fusion → movev):
      1. v_pos_base = vel_ff + kp * (pose_d - pose)
      2. fuse_tool_sleeve: Tool-X/Y from R.T @ v_pos_base; Tool-Z from force admittance
      3. output v_cmd_tool (frame_type=0) or v_cmd_base

    Sleeve vs old S_p @ (R.T v_base):
      - Old S_p zeroed tool-Z trajectory rate → lost tilted scan component.
      - Sleeve keeps tool-X/Y intact, replaces only [2] with force — no world-XY lock.
    """

    def __init__(self, dt: float, config: AdmittanceConfig | None = None) -> None:
        self.dt = dt
        self.cfg = config or AdmittanceConfig()
        self.force_error_integral = np.zeros(6)
        self.last_v_cmd = np.zeros(6)

    def reset(self) -> None:
        self.force_error_integral.fill(0.0)
        self.last_v_cmd.fill(0.0)

    @staticmethod
    def fuse_tool_sleeve(
        v_pos_base: np.ndarray,
        v_force_tool: np.ndarray,
        r_mat: np.ndarray,
        *,
        normal_track: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Tool-frame orthogonal decoupling (sleeve / slider):
          Tool-X/Y ← trajectory or visual Servo feedforward
          Tool-Z   ← force admittance only — no lateral compensation for Z motion.
        """
        v_pos_tool = np.zeros(6, dtype=float)
        v_pos_tool[:3] = r_mat.T @ np.asarray(v_pos_base[:3], dtype=float)
        v_pos_tool[3:6] = r_mat.T @ np.asarray(v_pos_base[3:6], dtype=float)

        v_cmd_tool = v_pos_tool.copy()
        v_cmd_tool[2] = float(v_force_tool[2])
        if normal_track:
            v_cmd_tool[3:6] += np.asarray(v_force_tool[3:6], dtype=float)

        v_cmd_base = np.zeros(6, dtype=float)
        v_cmd_base[:3] = r_mat @ v_cmd_tool[:3]
        v_cmd_base[3:] = r_mat @ v_cmd_tool[3:6]
        return v_cmd_tool, v_cmd_base

    def compute_velocity_command(
        self,
        current_pose: np.ndarray,
        desired_pose: np.ndarray,
        desired_vel_ff: np.ndarray,
        f_ext: np.ndarray,
        desired_force: np.ndarray,
        *,
        in_contact: bool | None = None,
    ) -> np.ndarray:
        cfg = self.cfg
        r_mat = Rsc.from_euler(
            cfg.euler_order, current_pose[3:6], degrees=False
        ).as_matrix()

        pose_predicted = np.asarray(current_pose, dtype=float).copy()
        if cfg.system_delay_s > 0.0:
            if cfg.control_frame == "tool":
                pose_predicted[:3] += r_mat @ self.last_v_cmd[:3] * cfg.system_delay_s
            else:
                pose_predicted[:3] += self.last_v_cmd[:3] * cfg.system_delay_s

        err_pose = pose_error(desired_pose, pose_predicted)
        vel_ff = np.asarray(desired_vel_ff, dtype=float).copy()
        if cfg.open_loop:
            err_pose[:] = 0.0
        kp = cfg.kp_pos * cfg.track_axes
        v_pos_base = vel_ff + kp * err_pose

        f_ext = np.asarray(f_ext, dtype=float)
        f_des = np.asarray(desired_force, dtype=float)
        v_force_tool = np.zeros(6, dtype=float)

        if in_contact is None:
            in_contact = float(np.linalg.norm(f_ext[:3])) >= cfg.contact_threshold_n
        else:
            in_contact = bool(in_contact)

        for axis in range(6):
            if cfg.force_axes[axis] < 0.5:
                continue
            f_err = f_des[axis] - f_ext[axis]
            v_force_tool[axis] = self._admittance_axis(axis, f_err, in_contact)

        normal_track = in_contact and cfg.enable_normal_tracking
        if normal_track:
            v_force_tool[3] = -cfg.k_align * f_ext[1]
            v_force_tool[4] = cfg.k_align * f_ext[0]

        v_cmd_tool, v_cmd_base = self.fuse_tool_sleeve(
            v_pos_base, v_force_tool, r_mat, normal_track=normal_track,
        )
        if cfg.max_vz_tool_m_s > 0.0:
            v_cmd_tool[2] = float(np.clip(v_cmd_tool[2], -cfg.max_vz_tool_m_s, cfg.max_vz_tool_m_s))
            if cfg.control_frame == "base":
                v_cmd_base[:3] = r_mat @ v_cmd_tool[:3]
                v_cmd_base[3:] = r_mat @ v_cmd_tool[3:6]

        v_out = v_cmd_tool if cfg.control_frame == "tool" else v_cmd_base
        v_clamp = np.clip(v_out, -cfg.max_velocity, cfg.max_velocity)
        dv_max = cfg.max_acceleration * self.dt
        v_final = np.clip(v_clamp, self.last_v_cmd - dv_max, self.last_v_cmd + dv_max)
        self.last_v_cmd = v_final.copy()
        return v_final

    def _admittance_axis(self, axis: int, f_err: float, in_contact: bool) -> float:
        cfg = self.cfg
        if axis == 2 and not in_contact:
            self.force_error_integral[axis] = 0.0
            v = cfg.k_fp_release * f_err
            cap = cfg.approach_vz_tool_m_s
            return float(np.clip(v, -cap, cap))

        if abs(f_err) > 5.0:
            self.force_error_integral[axis] = 0.0

        eff = smooth_deadband_eff(f_err, cfg.deadband_n, cfg.deadband_width_n)
        k_fp = cfg.k_fp_press if f_err < 0 else cfg.k_fp_release

        if abs(eff) > 1e-9:
            self.force_error_integral[axis] += eff * self.dt
            if axis == 2:
                if f_err < 0:
                    self.force_error_integral[axis] = min(
                        0.0, float(self.force_error_integral[axis]),
                    )
                else:
                    self.force_error_integral[axis] = max(
                        0.0, float(self.force_error_integral[axis]),
                    )
            self.force_error_integral[axis] = float(
                np.clip(self.force_error_integral[axis], -cfg.integral_limit, cfg.integral_limit)
            )
            v = k_fp * eff + cfg.k_fi * self.force_error_integral[axis]
        else:
            v = 0.0

        if axis == 2:
            v = float(np.clip(v, -cfg.max_vz_tool_m_s, cfg.max_vz_tool_m_s))
        return v
