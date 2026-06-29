"""Tool-frame force/motion decoupling + base-frame 6D trajectory tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, lfilter
from scipy.spatial.transform import Rotation as Rsc


def wrap_pi(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def pose_error(
    desired: np.ndarray,
    current: np.ndarray,
    euler_order: str = "xyz",
) -> np.ndarray:
    """
    Base-frame 6D pose error for PBAC.

    Position: linear difference in base frame.
    Orientation: SO(3) log map — rotvec of R_des @ R_cur^T, NOT Euler subtraction.
    """
    err = np.zeros(6, dtype=float)
    err[:3] = np.asarray(desired[:3], dtype=float) - np.asarray(current[:3], dtype=float)

    r_des = Rsc.from_euler(euler_order, desired[3:6], degrees=False).as_matrix()
    r_cur = Rsc.from_euler(euler_order, current[3:6], degrees=False).as_matrix()
    r_err = r_des @ r_cur.T
    err[3:6] = Rsc.from_matrix(r_err).as_rotvec()
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
    vz_filter_alpha: float = 0.3
    slew_skip_force_axes: bool = True
    # Force-axis virtual mass (Keemink 2018, Guideline 4/6): a finite acceleration
    # limit on the tool-Z admittance velocity renders a virtual inertia → bounded
    # jerk + coupled-stability margin. Replaces the slew-skip hack (which left the
    # force axis with no acceleration limit → the press/pull jerk). 0 ⇒ legacy skip.
    vz_accel_limit_m_s2: float = 0.6
    # --- True 2nd-order admittance on the tool-Z (force) axis ---
    # M·v̇ + D·v = (F_des − F_ext). With an environment spring k_e this closes to
    # M·ẍ + D·ẋ + k_e·x = F_des → stable for any M,D,k_e>0 (ζ = D/(2√(M k_e))).
    # Large contact force ⇒ bounded, damped velocity (no saturating P limit cycle).
    # M intrinsically bounds acceleration (=F_err/M), so it replaces the
    # vz_accel_limit hack; max_vz_tool_m_s stays only as a hard safety clamp.
    admittance_mass_z: float = 3.0      # virtual mass M [kg]
    admittance_damping_z: float = 60.0  # base virtual damping D [N·s/m]
    # --- Dimeas & Aspragathos 2016 online variable damping ---
    # Instability index Is = λ·Is + (1−λ)·Iω·Irms from the normal-force signal;
    # Iω = HF/AC energy ratio (high-pass at omega_c), Irms = rms(AC)/f_max. Raises
    # damping (and optionally mass) only when high-frequency oscillation appears,
    # then decays — responsive at low force, stiff/stable when poked hard.
    var_damping_enabled: bool = True
    var_damping_omega_c_hz: float = 3.5
    var_damping_lambda: float = 0.99
    var_damping_f_max_n: float = 30.0
    var_damping_d_u: float = 60.0       # additive D at Is=1 [N·s/m]
    var_damping_m_u: float = 0.0        # additive M at Is=1 [kg] (0 = damping only)
    var_damping_dc_alpha: float = 0.02  # slow EWMA splitting DC bias from AC
    # Position-loop conditioning (PBAC). Small deadband kills FT/encoder-noise jitter
    # on the velocity-controlled (tracking) directions; the correction clamp keeps a
    # transient contact slip from surging the velocity command independent of vel_ff.
    pos_err_deadband_m: float = 0.0
    pos_correction_max_m_s: float = 0.0

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
            vz_filter_alpha=float(c.get("vz_filter_alpha", 0.3)),
            slew_skip_force_axes=bool(c.get("slew_skip_force_axes", True)),
            vz_accel_limit_m_s2=float(c.get("vz_accel_limit_m_s2", 0.6)),
            admittance_mass_z=float(c.get("admittance_mass_z", 3.0)),
            admittance_damping_z=float(c.get("admittance_damping_z", 60.0)),
            var_damping_enabled=bool(c.get("var_damping_enabled", True)),
            var_damping_omega_c_hz=float(c.get("var_damping_omega_c_hz", 3.5)),
            var_damping_lambda=float(c.get("var_damping_lambda", 0.99)),
            var_damping_f_max_n=float(c.get("var_damping_f_max_n", 30.0)),
            var_damping_d_u=float(c.get("var_damping_d_u", 60.0)),
            var_damping_m_u=float(c.get("var_damping_m_u", 0.0)),
            var_damping_dc_alpha=float(c.get("var_damping_dc_alpha", 0.02)),
            pos_err_deadband_m=float(c.get("pos_err_deadband_m", 0.0)),
            pos_correction_max_m_s=float(c.get("pos_correction_max_m_s", 0.0)),
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
        self._contact_ticks = 0
        self.filtered_vz = 0.0
        # 2nd-order admittance state (tool-Z velocity carried across ticks).
        self.v_force_z = 0.0
        # Dimeas variable-damping state.
        self.instability_index = 0.0
        self.damping_z_eff = float(self.cfg.admittance_damping_z)
        self._f_dc = 0.0          # slow EWMA → DC (setpoint) component
        self._p_hi = 0.0          # EWMA of high-pass energy
        self._p_ac = 0.0          # EWMA of AC energy
        self._init_hp_filter()

    def _init_hp_filter(self) -> None:
        """2nd-order Butterworth high-pass (persistent biquad) for the Is index."""
        fs = 1.0 / self.dt if self.dt > 0 else 100.0
        wn = min(max(self.cfg.var_damping_omega_c_hz / (0.5 * fs), 1e-3), 0.99)
        self._hp_b, self._hp_a = butter(2, wn, btype="high")
        self._hp_zi = np.zeros(max(len(self._hp_a), len(self._hp_b)) - 1)

    def reset(self, *, clear_velocity: bool = False) -> None:
        self.force_error_integral.fill(0.0)
        self._contact_ticks = 0
        self.filtered_vz = 0.0
        self.v_force_z = 0.0
        self.instability_index = 0.0
        self.damping_z_eff = float(self.cfg.admittance_damping_z)
        self._f_dc = 0.0
        self._p_hi = 0.0
        self._p_ac = 0.0
        self._hp_zi.fill(0.0)
        if clear_velocity:
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
        enable_pbac: bool | None = None,
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

        err_pose = pose_error(desired_pose, pose_predicted, cfg.euler_order)
        vel_ff = np.asarray(desired_vel_ff, dtype=float).copy()
        use_pbac = (not cfg.open_loop) if enable_pbac is None else bool(enable_pbac)
        if not use_pbac:
            err_pose[:] = 0.0
        # --- Translation PBAC in the TOOL frame (task-frame formalism) ---
        # The force axis is the tool-Z (probe normal). When the probe is tilted, the
        # force-driven tool-Z excursion projects onto base-X/Z; a base-frame position
        # loop mis-reads that as lateral tracking error and injects spurious tool-X/Y
        # velocity (probe slides sideways). De Schutter & Van Brussel 1988 / Bruyninckx
        # & De Schutter 1996: force- and velocity-controlled directions must be
        # orthogonal IN THE TASK FRAME → compute the correction in the tool frame and
        # drop the tool-Z (force) component before applying gains.
        err_tool = r_mat.T @ err_pose[:3]
        err_tool[2] = 0.0
        if cfg.pos_err_deadband_m > 0.0:
            for i in (0, 1):
                if abs(err_tool[i]) <= cfg.pos_err_deadband_m:
                    err_tool[i] = 0.0
        kp_xy = np.array([
            cfg.kp_pos[0] * cfg.track_axes[0],
            cfg.kp_pos[1] * cfg.track_axes[1],
            0.0,
        ])
        v_corr_tool = kp_xy * err_tool
        if cfg.pos_correction_max_m_s > 0.0:
            v_corr_tool[:2] = np.clip(
                v_corr_tool[:2], -cfg.pos_correction_max_m_s, cfg.pos_correction_max_m_s
            )
        v_corr = np.zeros(6, dtype=float)
        v_corr[:3] = r_mat @ v_corr_tool
        # Rotational PBAC in the tool/task frame too (mirror the translation path):
        # express the SO(3) error in tool axes, apply per-axis gains there, then
        # rotate back. Keeps the velocity-controlled rotational directions in the
        # SAME frame the force axis is decoupled in (TFF: De Schutter / Bruyninckx).
        err_rot_tool = r_mat.T @ err_pose[3:6]
        kp_rot = cfg.kp_pos[3:6] * cfg.track_axes[3:6]
        v_corr[3:6] = r_mat @ (kp_rot * err_rot_tool)
        v_pos_base = vel_ff + v_corr

        f_ext = np.asarray(f_ext, dtype=float)
        f_des = np.asarray(desired_force, dtype=float)
        v_force_tool = np.zeros(6, dtype=float)

        if in_contact is None:
            in_contact = float(np.linalg.norm(f_ext[:3])) >= cfg.contact_threshold_n
        else:
            in_contact = bool(in_contact)

        if in_contact:
            self._contact_ticks += 1
        else:
            self._contact_ticks = 0
        db_alpha = min(1.0, self._contact_ticks / 50.0)

        # Dimeas online instability index from the normal-force signal → drives the
        # variable damping used by the tool-Z admittance below. Updated every tick.
        self._update_instability_index(float(f_ext[2]))

        for axis in range(6):
            if cfg.force_axes[axis] < 0.5:
                continue
            f_err = f_des[axis] - f_ext[axis]
            v_force_tool[axis] = self._admittance_axis(
                axis, f_err, in_contact, db_alpha,
            )

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
        v_final = np.asarray(v_clamp, dtype=float).copy()
        for i in range(6):
            if cfg.force_axes[i] > 0.5:
                # The 2nd-order admittance (M·v̇ + D·v = F_err) already bounds the
                # force-axis acceleration (=F_err/M) and is smooth by construction.
                # Re-applying a per-tick Δv clamp here was the old jerk-inducing hack
                # (Keemink G4/G6: let virtual inertia, not a slew gate, shape it).
                continue
            dvf = dv_max[i]
            v_final[i] = float(np.clip(
                v_final[i],
                self.last_v_cmd[i] - dvf,
                self.last_v_cmd[i] + dvf,
            ))
        self.last_v_cmd = v_final.copy()
        return v_final

    def _update_instability_index(self, f_z: float) -> None:
        """
        Dimeas & Aspragathos 2016 online instability index from the normal-force
        signal (Eqs. 4-6, adapted to an O(1) energy-ratio form):

          hp     = highpass(f_z, omega_c)          # oscillation band only
          f_ac   = f_z − dc(f_z)                    # strip the force setpoint bias
          Iω     = E[hp²] / E[f_ac²]                # HF / AC energy ratio  ∈[0,1]
          Irms   = rms(f_ac) / f_max                # bounded magnitude term ∈[0,1]
          Is     = λ·Is + (1−λ)·Iω·Irms

        damping_z_eff = D + d_u·Is  (and M += m_u·Is, applied in _admittance_z).
        Stable cooperation ⇒ Is≈0 ⇒ base D; a poke that excites the contact
        resonance raises Is ⇒ more damping, then Is decays via λ when it settles.
        """
        cfg = self.cfg
        if not cfg.var_damping_enabled:
            self.instability_index = 0.0
            self.damping_z_eff = float(cfg.admittance_damping_z)
            return

        y, self._hp_zi = lfilter(self._hp_b, self._hp_a, [f_z], zi=self._hp_zi)
        hp = float(y[0])

        self._f_dc += cfg.var_damping_dc_alpha * (f_z - self._f_dc)
        f_ac = f_z - self._f_dc

        a_e = 0.05  # energy EWMA rate (~0.2 s memory at 100 Hz)
        self._p_hi += a_e * (hp * hp - self._p_hi)
        self._p_ac += a_e * (f_ac * f_ac - self._p_ac)

        i_omega = min(max(self._p_hi / (self._p_ac + 1e-6), 0.0), 1.0)
        i_rms = min(np.sqrt(max(self._p_ac, 0.0)) / max(cfg.var_damping_f_max_n, 1e-6), 1.0)
        lam = cfg.var_damping_lambda
        self.instability_index = lam * self.instability_index + (1.0 - lam) * (i_omega * i_rms)
        self.damping_z_eff = float(
            cfg.admittance_damping_z + cfg.var_damping_d_u * self.instability_index
        )

    def _admittance_z(self, f_err: float, in_contact: bool, db_alpha: float) -> float:
        """
        Discrete 2nd-order velocity admittance on tool-Z (the force axis):

            M·v̇ + D·v = F_err  ⇒  v[k] = v[k-1] + (dt/M)·(F_err − D·v[k-1])

        Against an environment spring k_e this closes to M·ẍ + D·ẋ + k_e·x = F_des,
        stable for any M,D,k_e>0 (ζ = D/(2√(M k_e))) — so an over-design-force poke
        gives a bounded, damped retraction instead of the old saturating-P limit
        cycle. M bounds the acceleration; the velocity cap is a pure safety clamp.
        """
        cfg = self.cfg
        eff = smooth_deadband_eff(
            f_err, cfg.deadband_n * db_alpha, cfg.deadband_width_n * db_alpha
        )
        m = max(cfg.admittance_mass_z + cfg.var_damping_m_u * self.instability_index, 1e-3)
        d = self.damping_z_eff
        v = self.v_force_z + (self.dt / m) * (eff - d * self.v_force_z)

        cap = cfg.max_vz_tool_m_s if in_contact else cfg.approach_vz_tool_m_s
        if cap > 0.0:
            v = float(np.clip(v, -cap, cap))
        self.v_force_z = v
        return v

    def _admittance_axis(
        self,
        axis: int,
        f_err: float,
        in_contact: bool,
        db_alpha: float = 1.0,
    ) -> float:
        cfg = self.cfg
        if axis == 2:
            return self._admittance_z(f_err, in_contact, db_alpha)

        # Legacy proportional + bounded-integral path, retained for any non-Z force
        # axis (none in the default force_axes mask).
        if abs(f_err) > 5.0:
            self.force_error_integral[axis] = 0.0
        actual_deadband = cfg.deadband_n * db_alpha
        actual_width = cfg.deadband_width_n * db_alpha
        eff = smooth_deadband_eff(f_err, actual_deadband, actual_width)
        k_fp = cfg.k_fp_press if f_err < 0 else cfg.k_fp_release
        if abs(eff) > 1e-9:
            self.force_error_integral[axis] += eff * self.dt
            self.force_error_integral[axis] = float(
                np.clip(self.force_error_integral[axis], -cfg.integral_limit, cfg.integral_limit)
            )
            v = k_fp * eff + cfg.k_fi * self.force_error_integral[axis]
        else:
            v = 0.0
        return v
