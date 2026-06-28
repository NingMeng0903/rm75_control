# RM75 力位混合速度控制 — 第三方代码包

> 生成目的：完整展示标定 → 读力 → 解算 → 速度 CANFD → 力位混合控制器 相关源码。
> 运行 demo：`python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py --trajectory sin_base_y_tool_rz --rz-deg 12`
> 外部依赖：RealMan `RM_API2/Python`（`Robotic_Arm`）、numpy、pyyaml、scipy。
> 标定产物：`tmp/force_compensation/logs/force_id_phi.json`（运行 admittance 前需存在）。

## 目录

- [一、入口与配置](#一入口与配置)
- [二、rm75_control/control（控制器与轨迹）](#二rm75_control/control控制器与轨迹)
- [三、标定与力补偿](#三标定与力补偿)
- [四、速度下发与运动](#四速度下发与运动)
- [五、Session 与后端](#五Session与后端)
- [六、调用关系](#六调用关系)
- [七、环境与依赖](#七环境与依赖)
- [八、YAML 参数附录](#八yaml-参数附录全文--说明)

## 一、入口与配置

### `tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py`

```python
#!/usr/bin/env python3
"""
Demo: 6D trajectory plugin + tool-frame force/motion hybrid.

  source env.sh
  cd /media/camp/EXT_DRIVE/rm75_control
  python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py
  python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py --trajectory sin_base_y --y-pp-cm 16
  python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py --trajectory sin_base_y_tool_rz --rz-deg 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rm75_control.control.velocity_admittance.loop import load_yaml, run_velocity_admittance
from rm75_control.control.velocity_admittance.paths import CONFIG_SIN_TOOL_Y_Z2N


def main() -> int:
    parser = argparse.ArgumentParser(
        description="6D trajectory + tool-Z force hybrid (trajectory is pluggable)",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_SIN_TOOL_Y_Z2N)
    parser.add_argument(
        "--trajectory", type=str, default=None,
        help="trajectory.type: hold | sin_base_y | sin_base_y_tool_rz | sin_tool_y",
    )
    parser.add_argument("--desired-z", type=float, default=None, help="tool-Z force target (N)")
    parser.add_argument("--y-pp-cm", type=float, default=None, help="world-Y peak-to-peak (cm)")
    parser.add_argument("--rz-deg", type=float, default=None, help="tool +Z spin amplitude (deg)")
    parser.add_argument("--duration", type=float, default=None, help="scan duration (s)")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    traj = raw.setdefault("trajectory", {})
    if args.trajectory:
        traj["type"] = args.trajectory
    if args.y_pp_cm is not None:
        traj["y_peak_to_peak_cm"] = args.y_pp_cm
    if args.rz_deg is not None:
        traj["rz_amplitude_deg"] = args.rz_deg
    if args.desired_z is not None:
        raw.setdefault("force", {})["desired_z_n"] = args.desired_z

    return run_velocity_admittance(
        raw,
        title="Demo 6D traj + tool-Z force",
        duration_s=args.duration,
    )


if __name__ == "__main__":
    raise SystemExit(main())
```

### `tmp/Velocity_Admittance/run_admittance.py`

```python
#!/usr/bin/env python3
"""
Velocity-resolved admittance (generic entry).

  source env.sh
  python tmp/Velocity_Admittance/run_admittance.py
  python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rm75_control.control.velocity_admittance.loop import load_yaml, run_velocity_admittance
from rm75_control.control.velocity_admittance.paths import CONFIG_ADMITTANCE


def main() -> int:
    parser = argparse.ArgumentParser(description="RM75 velocity admittance control")
    parser.add_argument("--config", type=Path, default=CONFIG_ADMITTANCE)
    parser.add_argument("--trajectory", type=str, default=None)
    parser.add_argument("--desired-z", type=float, default=None, help="sensor Fz target (N)")
    parser.add_argument("--duration", type=float, default=None, help="run time (s)")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    if args.trajectory:
        raw.setdefault("trajectory", {})["type"] = args.trajectory
    if args.desired_z is not None:
        raw.setdefault("force", {})["desired_z_n"] = args.desired_z

    return run_velocity_admittance(raw, duration_s=args.duration)


if __name__ == "__main__":
    raise SystemExit(main())
```

### `tmp/Velocity_Admittance/demo/config/sin_tool_y_z2n.yaml`

```yaml
# Demo trajectory plugin + tool-Z force hybrid.
# Architecture:
#   trajectory → 6D (pose_d, vel_ff) in base frame  [you may zero tool-Z in vel_ff]
#   controller → S_f on tool force_axes; all other DOFs follow trajectory (S_p = I - S_f)
#
# Run:
#   source env.sh && cd /media/camp/EXT_DRIVE/rm75_control
#   python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py

timing:
  dt_ms: 10.0
  feedback_every: 2

startup:
  pose_slot: d
  settle_frames: 25
  hold_s: 1.0
  auto_recover: true
  wait_contact: true
  auto_start_under_n: 0.5
  auto_start_hold_s: 0.5
  approach_ramp_s: 1.5

frames:
  euler_order: xyz
  control_frame: tool

velocity_canfd:
  frame_type: 0
  avoid_singularity: 0
  follow: true
  trajectory_mode: 0
  radio: 0

force:
  phi_source: phi_recommended
  buffer_s: 2.0
  min_samples: 22
  fc_hz: 4.0
  use_inertia: false
  desired_z_n: 3.0

trajectory:
  type: sin_base_y_tool_rz    # demo plugin; replace with any Trajectory6D
  y_peak_to_peak_cm: 16.0
  rz_amplitude_deg: 12.0
  y_max_vel_cm_s: 3.0
  soft_start: true
  ramp_s: 2.0
  open_loop: true

controller:
  force_axes: [0, 0, 1, 0, 0, 0]   # tool TCP-Z → force; other 5 DOF → trajectory
  open_loop: true
  kp_pos: [0, 0, 0, 0, 0, 0]
  system_delay_s: 0.015
  k_fp_press: 0.045
  k_fp_release: 0.025
  k_fi: 0.001
  integral_limit: 0.015
  max_vz_tool_m_s: 0.15
  approach_vz_tool_m_s: 0.03
  max_velocity: [0.03, 0.10, 0.15, 0.15, 0.15, 0.35]
  max_acceleration: [0.5, 1.0, 0.5, 1.0, 1.0, 1.5]
```

### `tmp/Velocity_Admittance/config/admittance.yaml`

```yaml
# Velocity admittance — library: rm75_control.control.velocity_admittance
# Generic: python tmp/Velocity_Admittance/run_admittance.py
# Demo:    python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py

timing:
  dt_ms: 10.0
  feedback_every: 3

startup:
  pose_slot: d

frames:
  euler_order: xyz   # rm_get_current_arm_state pose; match configs/force_sensor.yaml

velocity_canfd:
  frame_type: 1          # 0 tool, 1 work/base — v_cmd output is base frame
  avoid_singularity: 0
  follow: true
  trajectory_mode: 0     # 0 passthrough — smoothing in controller only
  radio: 0

force:
  phi_source: phi_recommended
  buffer_s: 4.0
  min_samples: 35
  use_inertia: false     # no virtual mass — PI admittance only (Keemink §5.4)
  desired_z_n: 3.0

trajectory:
  type: sin_tool_y       # hold | sin_tool_y | sin_base_y
  amplitude_mm: 5.0
  period_s: null         # auto from y_max_vel_cm_s if null
  y_max_vel_cm_s: 1.0

controller:
  force_axes: [0, 0, 1, 0, 0, 0]   # 1 = admittance on axis (sensor frame)
  motion_axes: [0, 1, 0, 0, 0, 0]
  lock_orientation: true
  enable_normal_tracking: false
  kp_pos: [2.0, 2.0, 0.0, 1.5, 1.5, 1.5]
  system_delay_s: 0.015
  k_fp_press: 0.015
  k_fp_release: 0.005
  k_fi: 0.008
  integral_limit: 0.05
  k_align: 0.0
  contact_threshold_n: 0.5
  deadband_n: 0.3
  max_vz_tool_m_s: 0.05
  max_velocity: [0.2, 0.2, 0.05, 0.08, 0.08, 0.08]
  max_acceleration: [1.0, 1.0, 0.05, 0.15, 0.15, 0.15]
  release_vz_up_m_s: 0.05
  release_vz_down_m_s: 0.05

monitor:
  enabled: false
  window_s: 25.0
  refresh_hz: 12.0
```

### `configs/rm75f_default.yaml`

```yaml
robot:
  ip: "192.168.1.18"
  port: 8080
  thread_mode: 2  # RM_TRIPLE_MODE_E

timing:
  canfd_period_ms: 10
  force_scan_period_ms: 10

motion:
  default_velocity_percent: 20

  # Cartesian pose CANFD (rm_movep_canfd) — native params + use_ruckig switch
  cartesian_pose:
    use_ruckig: false
    period_ms: 10
    follow: true
    trajectory_mode: 0  # 0 passthrough, 1 curve fit, 2 filter
    radio: 0
    steps_per_segment: 50  # used only when use_ruckig=false
    ruckig:
      max_velocity: [0.05, 0.05, 0.05, 0.3, 0.3, 0.3]
      max_acceleration: [0.5, 0.5, 0.5, 1.5, 1.5, 1.5]
      max_jerk: [2.0, 2.0, 2.0, 5.0, 5.0, 5.0]

force:
  sensor: 1
  coordinate_mode: 1
  default_desired_force: [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
  default_control_mode: [3, 3, 4, 3, 3, 3]
  scan:
    flag: 0  # 0 joint (after IK), 1 pose
    follow: true
    trajectory_mode: 0
    radio: 0
    period_ms: 10
    control_mode: [3, 3, 4, 3, 3, 3]
    desired_force: [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
    limit_vel: [0.05, 0.05, 0.25, 10.0, 10.0, 10.0]

tool:
  name: "gripper"
  payload_kg: 0.982
```

### `configs/force_sensor.yaml`

```yaml
# RealMan RM75 raw force_data — dynamic ID / compensation (verified 2026-06-27)
# Verified force_sign / euler — used by rm75_control.force.compensation.regressor

force_sign: [-1, -1, -1, 1, 1, 1]   # flip Fx,Fy,Fz only; moments unchanged
euler_order: xyz
sensor_offset_euler_xyz_rad: [0.0, 0.0, 0.0]
# Sensor origin position in link7 frame (m). Default 0 = co-located with link7 origin.
sensor_origin_in_link7_m: [0.0, 0.0, 0.0]
gravity_base: [0.0, 0.0, -9.80665]

filtfilt_cutoff_hz: 2.5
identify_mass_bias_only: true

# Verified on force_id_cartesian.npz (601 samples) + live static:
#   m_fit = +0.997 kg,  live m = +1.10 kg,  Fz_raw ≈ +10.5 N
#   hold-out compensation RMS: force 0.219 N, moment 0.015 Nm
# Equivalent combos (same RMS): flip all 6, or Rz180 offset — not needed if using above.
```

### `tmp/force_compensation/config/force_id.yaml`

```yaml
# Force compensation pipeline — config/force_id.yaml, config/poses.yaml
# Library: rm75_control.force.compensation
# Entry: tmp/force_compensation/force_calibrate.py, force_monitor.py

poses_yaml: poses.yaml

sequence: [a, b, c, d]
return_home: a

collect:
  move_speed: 15
  settle_timeout_s: 15.0
  dt_ms: 10.0
  log_every: 10
  scale: 1.0
  warmup_s: 3.0
  follow: false

  cartesian:
    duration_s: 30.0
    max_delta_mm: 5.0
    max_orient_deg:
      a: 18.0
      b: 32.0
      c: 32.0
    amp_mm: [3.0, 4.0, 2.0]
    amp_rot_deg: [12.0, 15.0, 12.0]
    amp_rot_deg_slots:
      b: [16.0, 20.0, 16.0]
      c: [16.0, 20.0, 16.0]
    freqs_hz:
      - [0.12, 0.18]
      - [0.13, 0.19]
      - [0.11, 0.16]
      - [0.22, 0.33]
      - [0.25, 0.37]
      - [0.24, 0.31]

  pose_d:
    # Phase 0: joint 45s. Phase 1: pose_d_vel_burst 45s (resync to q0 before burst).
    joint_duration_s: 45.0
    burst_duration_s: 45.0
    joint_amp_deg: [10.0, 8.0, 8.0, 14.0, 16.0, 14.0, 32.0]
    joint_max_delta_deg: [12.0, 12.0, 12.0, 18.0, 20.0, 18.0, 35.0]
    joint_freqs_hz:
      - [0.14, 0.21]
      - [0.13, 0.19]
      - [0.12, 0.18]
      - [0.22, 0.31]
      - [0.24, 0.33]
      - [0.23, 0.30]
      - [0.20, 0.29]
    velocity_burst:
      profile: pose_d_vel_burst
      # base wx→wy→wz 12°/s 0.28Hz traj=0; init settle 100ms (see excitation.settle_movev_after_init)
      ramp_down_s: 4.0

fit:
  force_sensor: configs/force_sensor.yaml
  holdout_frac: 0.2
  alpha_percentile: 70.0
  min_burst_rows: 300
  min_high_alpha_rows: 150
  inertia_r_max_m: 0.12
  npz_slots: [a, b, c, d]
  phi_output: force_id_phi.json
  phi_recommended_key: phi_burst

monitor:
  poll_ms: 50.0
  window_s: 25.0
  buffer_s: 4.0
  min_samples: 35
  refresh_hz: 12.0
  phi_source: phi_recommended
  use_inertia: true
```

### `tmp/force_compensation/config/poses.yaml`

```yaml
poses:
  a:
    label: pose_a
    note: 2026-06-27 verified
    pose_base:
    - 0.284324
    - -0.002917
    - 0.332434
    - -3.083
    - 0.043
    - 2.892
    q_deg:
    - 4.483
    - 15.902
    - -4.011
    - 72.358
    - -2.767
    - 90.212
    - 14.96
  b:
    label: pose_b
    note: saved 2026-06-27 12:42 UTC
    pose_base:
    - 0.278374
    - -0.095107
    - 0.358767
    - -2.449
    - 0.055
    - 3.059
    q_deg:
    - 4.253
    - 15.898
    - -3.625
    - 75.219
    - -38.729
    - 90.175
    - 6.913
  c:
    label: pose_c
    note: saved 2026-06-27 12:42 UTC
    pose_base:
    - 0.287775
    - 0.11732
    - 0.363931
    - 2.398
    - -0.047
    - 3.017
    q_deg:
    - 4.343
    - 15.893
    - -3.619
    - 75.287
    - 43.582
    - 85.76
    - 6.911
  d:
    label: pose_d
    note: saved 2026-06-27 13:02 UTC
    pose_base:
    - 0.210513
    - 0.003795
    - 0.515802
    - 2.642
    - 0.96
    - 2.639
    q_deg:
    - 4.987
    - -23.07
    - -3.953
    - 77.84
    - 2.451
    - 65.541
    - 14.407
```

## 二、rm75_control/control（控制器与轨迹）

### `rm75_control/control/__init__.py`

```python
"""High-level control modes (Cartesian CANFD pose/velocity + optional Ruckig)."""

from rm75_control.control.cartesian_pose import (
    CartesianLimits,
    CartesianPoseController,
    CartesianPoseStreamConfig,
)
from rm75_control.control.cartesian_velocity import (
    AxisVelocityGains,
    CartesianVelocityController,
    CartesianVelocityStreamConfig,
    CartesianVelocityTracker,
    CartesianVelocityTrackerConfig,
)

from rm75_control.control.velocity_admittance import (
    AdmittanceConfig,
    AdmittanceController,
    CompensatedForceObserver,
    run_velocity_admittance,
)

__all__ = [
    "AdmittanceConfig",
    "AdmittanceController",
    "AxisVelocityGains",
    "CartesianLimits",
    "CartesianPoseController",
    "CartesianPoseStreamConfig",
    "CartesianVelocityController",
    "CartesianVelocityStreamConfig",
    "CartesianVelocityTracker",
    "CartesianVelocityTrackerConfig",
    "CompensatedForceObserver",
    "run_velocity_admittance",
]
```

### `rm75_control/control/cartesian_pose.py`

```python
"""Cartesian pose CANFD (rm_movep_canfd) with optional Ruckig preprocessing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from rm75_control.core.exceptions import MotionError
from rm75_control.motion.canfd import PoseCanfdClient, send_pose_canfd

Pose6 = list[float]


@dataclass
class CartesianLimits:
    max_velocity: list[float] = field(
        default_factory=lambda: [0.05, 0.05, 0.05, 0.3, 0.3, 0.3]
    )
    max_acceleration: list[float] = field(
        default_factory=lambda: [0.5, 0.5, 0.5, 1.5, 1.5, 1.5]
    )
    max_jerk: list[float] = field(
        default_factory=lambda: [2.0, 2.0, 2.0, 5.0, 5.0, 5.0]
    )


@dataclass
class CartesianPoseStreamConfig:
    """Native rm_movep_canfd params + optional Ruckig upstream."""

    use_ruckig: bool = False
    period_ms: float = 10.0
    follow: bool = True
    trajectory_mode: int = 0
    radio: int = 0
    limits: CartesianLimits = field(default_factory=CartesianLimits)
    steps_per_segment: int = 50

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        use_ruckig: bool | None = None,
        follow: bool | None = None,
        trajectory_mode: int | None = None,
        radio: int | None = None,
        period_ms: float | None = None,
        steps_per_segment: int | None = None,
        **overrides: Any,
    ) -> CartesianPoseStreamConfig:
        motion = config.get("motion", {})
        cartesian = motion.get("cartesian_pose", {})
        ruckig_cfg = cartesian.get("ruckig", motion.get("ruckig", {}))
        timing = config.get("timing", {})

        limits = CartesianLimits(
            max_velocity=ruckig_cfg.get(
                "max_velocity", CartesianLimits().max_velocity
            ),
            max_acceleration=ruckig_cfg.get(
                "max_acceleration", CartesianLimits().max_acceleration
            ),
            max_jerk=ruckig_cfg.get("max_jerk", CartesianLimits().max_jerk),
        )

        def _pick(key: str, legacy_key: str, default: Any) -> Any:
            if key in cartesian:
                return cartesian[key]
            if legacy_key in motion:
                return motion[legacy_key]
            return default

        values: dict[str, Any] = {
            "use_ruckig": cartesian.get("use_ruckig", motion.get("use_ruckig", False)),
            "period_ms": cartesian.get(
                "period_ms", timing.get("canfd_period_ms", 10.0)
            ),
            "follow": _pick("follow", "canfd_follow", True),
            "trajectory_mode": _pick("trajectory_mode", "canfd_trajectory_mode", 0),
            "radio": _pick("radio", "canfd_radio", 0),
            "limits": limits,
            "steps_per_segment": cartesian.get(
                "steps_per_segment", motion.get("steps_per_segment", 50)
            ),
        }

        optional = {
            "use_ruckig": use_ruckig,
            "follow": follow,
            "trajectory_mode": trajectory_mode,
            "radio": radio,
            "period_ms": period_ms,
            "steps_per_segment": steps_per_segment,
        }
        for key, val in optional.items():
            if val is not None:
                values[key] = val
        values.update(overrides)
        return cls(**values)


class CartesianPoseController:
    """Stream Cartesian poses through rm_movep_canfd."""

    def __init__(
        self,
        robot: PoseCanfdClient,
        config: CartesianPoseStreamConfig | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        self.robot = robot
        self.config = config or CartesianPoseStreamConfig()
        self.dry_run = dry_run

    @property
    def use_ruckig(self) -> bool:
        return self.config.use_ruckig

    def run(
        self,
        waypoints: Sequence[Pose6],
        *,
        start_pose: Sequence[float] | None = None,
    ) -> None:
        if len(waypoints) == 0:
            raise ValueError("waypoints must not be empty")

        normalized = [_normalize_pose6(p) for p in waypoints]
        if self.config.use_ruckig:
            initial = (
                _normalize_pose6(start_pose)
                if start_pose is not None
                else normalized[0]
            )
            stream = self._generate_ruckig(normalized, initial)
        else:
            stream = self._generate_direct(normalized)

        dt = self.config.period_ms / 1000.0
        for pose in stream:
            if not self.dry_run:
                send_pose_canfd(
                    self.robot,
                    pose,
                    follow=self.config.follow,
                    trajectory_mode=self.config.trajectory_mode,
                    radio=self.config.radio,
                )
            if dt > 0.0:
                time.sleep(dt)

    def _generate_direct(self, waypoints: list[Pose6]) -> Iterator[Pose6]:
        if len(waypoints) == 1:
            yield waypoints[0]
            return

        steps = max(1, self.config.steps_per_segment)
        if steps == 1:
            for p in waypoints:
                yield p
            return

        for start, end in zip(waypoints[:-1], waypoints[1:]):
            start_arr = np.asarray(start, dtype=float)
            end_arr = np.asarray(end, dtype=float)
            for alpha in np.linspace(0.0, 1.0, steps, endpoint=True):
                yield list(start_arr + alpha * (end_arr - start_arr))

    def _generate_ruckig(
        self,
        waypoints: list[Pose6],
        initial_pose: Pose6,
    ) -> Iterator[Pose6]:
        dt = self.config.period_ms / 1000.0
        dofs = 6
        otg = Ruckig(dofs, dt)
        inp = InputParameter(dofs)
        out = OutputParameter(dofs)

        limits = self.config.limits
        inp.max_velocity = limits.max_velocity
        inp.max_acceleration = limits.max_acceleration
        inp.max_jerk = limits.max_jerk

        current = list(initial_pose)
        for target in waypoints:
            if np.allclose(current, target, atol=1e-9):
                continue
            inp.current_position = current
            inp.current_velocity = [0.0] * dofs
            inp.current_acceleration = [0.0] * dofs
            inp.target_position = _normalize_pose6(target)
            inp.target_velocity = [0.0] * dofs
            inp.target_acceleration = [0.0] * dofs

            result = Result.Working
            while result == Result.Working:
                result = otg.update(inp, out)
                if result not in (Result.Working, Result.Finished):
                    raise MotionError(f"Ruckig failed with result {result}")
                yield list(out.new_position)
                out.pass_to_input(inp)

            current = list(out.new_position)


def _normalize_pose6(pose: Sequence[float]) -> Pose6:
    if len(pose) == 6:
        return [float(v) for v in pose]
    if len(pose) == 7:
        raise ValueError(
            "Quaternion pose is not supported in Ruckig path yet; use 6D euler."
        )
    raise ValueError(f"pose must have 6 elements, got {len(pose)}")
```

### `rm75_control/control/cartesian_velocity.py`

```python
"""Cartesian velocity CANFD with outer position loop (feedforward + P/I)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from rm75_control.motion.canfd import (
    VelocityCanfdClient,
    clamp_cartesian_velocity,
    send_velocity_canfd,
)

Pose6 = list[float]
Vel6 = list[float]


@dataclass
class AxisVelocityGains:
    """Position error -> velocity correction on one Cartesian axis."""

    kp: float = 0.0
    ki: float = 0.0
    integral_limit: float = 0.05


@dataclass
class CartesianVelocityTrackerConfig:
    """Per-axis outer loop gains for rm_movev_canfd streaming."""

    axis_gains: list[AxisVelocityGains] = field(
        default_factory=lambda: [AxisVelocityGains() for _ in range(6)]
    )
    max_correction_m_s: float = 0.015
    motion_axes: tuple[int, ...] = ()
    ref_speed_peak_m_s: float = 0.0

    @classmethod
    def for_motion_axes(
        cls,
        motion_axes: Sequence[int],
        *,
        kp: float = 2.0,
        ki: float = 0.0,
        hold_axes: Sequence[int] | None = None,
        hold_kp: float = 1.0,
        ref_speed_peak_m_s: float = 0.0,
    ) -> CartesianVelocityTrackerConfig:
        gains = [AxisVelocityGains() for _ in range(6)]
        for idx in motion_axes:
            gains[idx] = AxisVelocityGains(kp=kp, ki=ki)
        if hold_axes is not None:
            for idx in hold_axes:
                if idx not in motion_axes:
                    gains[idx] = AxisVelocityGains(kp=hold_kp)
        return cls(
            axis_gains=gains,
            motion_axes=tuple(motion_axes),
            ref_speed_peak_m_s=ref_speed_peak_m_s,
        )


class CartesianVelocityTracker:
    """
    Outer position loop on top of velocity CANFD.

    vy_cmd = vy_ff + Kp*(y_ref - y_fb) + Ki*integral(err)

    Intended building block for hybrid control: motion axes via movev + position
    closure; force axes stay on rm_force_position_move (see force/scan.py).
    """

    def __init__(self, config: CartesianVelocityTrackerConfig | None = None) -> None:
        self.config = config or CartesianVelocityTrackerConfig()
        self._integral = [0.0] * 6

    def reset(self) -> None:
        self._integral = [0.0] * 6

    def _kp_scale(self, axis: int, ref_vel: float) -> float:
        peak = self.config.ref_speed_peak_m_s
        if peak <= 0.0 or axis not in self.config.motion_axes:
            return 1.0
        ratio = min(1.0, abs(ref_vel) / peak)
        # Near max speed error is mostly phase lag — back off P to avoid spikes.
        return max(0.1, 1.0 - 0.9 * ratio * ratio)

    def compute(
        self,
        *,
        ref_pose: Sequence[float],
        ref_vel: Sequence[float],
        fb_pose: Sequence[float],
        dt_s: float,
    ) -> Vel6:
        vel = [float(ref_vel[i]) for i in range(6)]
        max_corr = self.config.max_correction_m_s
        for i in range(6):
            g = self.config.axis_gains[i]
            if g.kp == 0.0 and g.ki == 0.0:
                continue
            err = float(ref_pose[i]) - float(fb_pose[i])
            corr = 0.0
            if g.ki != 0.0 and dt_s > 0.0:
                self._integral[i] += err * dt_s
                lim = g.integral_limit
                self._integral[i] = max(-lim, min(lim, self._integral[i]))
                corr += g.ki * self._integral[i]
            corr += g.kp * err * self._kp_scale(i, float(ref_vel[i]))
            if max_corr > 0.0:
                corr = max(-max_corr, min(max_corr, corr))
            vel[i] += corr
        return clamp_cartesian_velocity(vel)


@dataclass
class CartesianVelocityStreamConfig:
    follow: bool = True
    trajectory_mode: int = 1
    radio: int = 40
    period_ms: float = 10.0


class CartesianVelocityController:
    """Stream clamped Cartesian velocities through rm_movev_canfd."""

    def __init__(
        self,
        robot: VelocityCanfdClient,
        tracker: CartesianVelocityTracker | None = None,
        config: CartesianVelocityStreamConfig | None = None,
        *,
        dry_run: bool = False,
        max_dv_m_s: float = 0.0004,
    ) -> None:
        self.robot = robot
        self.tracker = tracker or CartesianVelocityTracker()
        self.config = config or CartesianVelocityStreamConfig()
        self.dry_run = dry_run
        self.max_dv_m_s = max_dv_m_s
        self._last_vel: Vel6 = [0.0] * 6

    def reset_slew(self) -> None:
        self._last_vel = [0.0] * 6

    def send_velocity(self, vel: Vel6) -> None:
        if self.dry_run:
            return
        send_velocity_canfd(
            self.robot,
            vel,
            follow=self.config.follow,
            trajectory_mode=self.config.trajectory_mode,
            radio=self.config.radio,
        )

    def step(
        self,
        *,
        ref_pose: Sequence[float],
        ref_vel: Sequence[float],
        fb_pose: Sequence[float],
        dt_s: float | None = None,
    ) -> Vel6:
        dt = dt_s if dt_s is not None else self.config.period_ms / 1000.0
        vel = self.tracker.compute(
            ref_pose=ref_pose,
            ref_vel=ref_vel,
            fb_pose=fb_pose,
            dt_s=dt,
        )
        if self.max_dv_m_s > 0.0:
            for i in range(6):
                dv = vel[i] - self._last_vel[i]
                dv = max(-self.max_dv_m_s, min(self.max_dv_m_s, dv))
                vel[i] = self._last_vel[i] + dv
        self._last_vel = list(vel)
        self.send_velocity(vel)
        return vel
```

### `rm75_control/control/velocity_admittance/__init__.py`

```python
"""Velocity-resolved admittance control loop and trajectory."""

from rm75_control.control.velocity_admittance.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.velocity_admittance.loop import load_yaml, run_velocity_admittance
from rm75_control.control.velocity_admittance.observer import CompensatedForceObserver
from rm75_control.control.velocity_admittance.trajectory import (
    Trajectory6D,
    TrajectoryGenerator,
    TrajectorySample,
)
from rm75_control.control.velocity_admittance.paths import (
    CONFIG_ADMITTANCE,
    CONFIG_SIN_TOOL_Y_Z2N,
)

__all__ = [
    "AdmittanceConfig",
    "AdmittanceController",
    "CompensatedForceObserver",
    "Trajectory6D",
    "TrajectoryGenerator",
    "TrajectorySample",
    "CONFIG_ADMITTANCE",
    "CONFIG_SIN_TOOL_Y_Z2N",
    "load_yaml",
    "run_velocity_admittance",
]
```

### `rm75_control/control/velocity_admittance/paths.py`

```python
"""Paths for velocity admittance configs (under tmp/Velocity_Admittance)."""

from __future__ import annotations

from pathlib import Path

from rm75_control.force.compensation.paths import CONFIG_FORCE, CONFIG_ROBOT, PHI_JSON, REPO

VA_DATA_DIR = REPO / "tmp" / "Velocity_Admittance"
CONFIG_DIR = VA_DATA_DIR / "config"
DEMO_CONFIG_DIR = VA_DATA_DIR / "demo" / "config"
CONFIG_ADMITTANCE = CONFIG_DIR / "admittance.yaml"
CONFIG_SIN_TOOL_Y_Z2N = DEMO_CONFIG_DIR / "sin_tool_y_z2n.yaml"
```

### `rm75_control/control/velocity_admittance/rm_algo.py`

```python
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
```

### `rm75_control/control/velocity_admittance/controller.py`

```python
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


@dataclass
class AdmittanceConfig:
    """
    force_axes: tool-frame mask for admittance (typ. [0,0,1,0,0,0] = TCP normal).
    All other tool DOFs follow trajectory via S_p = I - S_f.
    Trajectory pose_d / vel_ff are always base-frame 6D from a Trajectory6D producer.
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
    Pipeline (base trajectory → tool decouple → movev):
      1. v_pos_base = vel_ff + kp * (pose_d - pose)     [base 6D from trajectory]
      2. v_pos_tool = R^T v_pos_base
      3. v_cmd_tool = S_p v_pos_tool + S_f v_force_tool
      4. output v_cmd_tool or R v_cmd_tool per control_frame
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
        if normal_track and self.cfg.enable_normal_tracking:
            s_f[3, 3] = 1.0
            s_f[4, 4] = 1.0
        s_p = np.eye(6) - s_f
        return s_p, s_f

    @staticmethod
    def fuse_tool_decoupled(
        v_pos_base: np.ndarray,
        v_force_tool: np.ndarray,
        r_mat: np.ndarray,
        s_p: np.ndarray,
        s_f: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        v_pos_tool = np.zeros(6, dtype=float)
        v_pos_tool[:3] = r_mat.T @ v_pos_base[:3]
        v_pos_tool[3:] = r_mat.T @ v_pos_base[3:]
        v_cmd_tool = s_p @ v_pos_tool + s_f @ v_force_tool
        v_cmd_base = np.zeros(6, dtype=float)
        v_cmd_base[:3] = r_mat @ v_cmd_tool[:3]
        v_cmd_base[3:] = r_mat @ v_cmd_tool[3:]
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

        s_p, s_f = self._selection(normal_track=normal_track)
        v_cmd_tool, v_cmd_base = self.fuse_tool_decoupled(
            v_pos_base, v_force_tool, r_mat, s_p, s_f,
        )
        if cfg.max_vz_tool_m_s > 0.0:
            v_cmd_tool[2] = float(np.clip(v_cmd_tool[2], -cfg.max_vz_tool_m_s, cfg.max_vz_tool_m_s))
            if cfg.control_frame == "base":
                v_cmd_base[:3] = r_mat @ v_cmd_tool[:3]
                v_cmd_base[3:] = r_mat @ v_cmd_tool[3:]

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

        k_fp = cfg.k_fp_press if f_err < 0 else cfg.k_fp_release
        if abs(f_err) > cfg.deadband_n:
            eff = f_err - math.copysign(cfg.deadband_n, f_err)
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
            v = cfg.k_fi * self.force_error_integral[axis]
        if axis == 2:
            v = float(np.clip(v, -cfg.max_vz_tool_m_s, cfg.max_vz_tool_m_s))
        return v
```

### `rm75_control/control/velocity_admittance/trajectory.py`

```python
"""6D trajectory producers (base frame). Hybrid controller consumes pose_d + vel_ff."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .rm_algo import end2tool_pose


def tool_offset_pose(robot, ref_pose: list[float], dx: float, dy: float, dz: float) -> list[float]:
    delta = [dx, dy, dz, 0.0, 0.0, 0.0]
    return robot.rm_algo_pose_move(ref_pose, delta, frameMode=1)


def sin_period_for_peak_vel(amplitude_m: float, max_vel_m_s: float) -> float:
    if amplitude_m <= 0.0 or max_vel_m_s <= 0.0:
        return 1.0
    return 2.0 * math.pi * amplitude_m / max_vel_m_s


@dataclass(frozen=True)
class TrajectorySample:
    """One tick of reference motion in base/world frame (6D pose + 6D velocity)."""

    pose_d: np.ndarray
    vel_ff: np.ndarray


class Trajectory6D(Protocol):
    """Any trajectory plugin: set contact origin, then stream 6D references."""

    def set_origin(self, pose0: np.ndarray) -> None: ...

    def sample(self, t_s: float) -> TrajectorySample: ...


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

    @property
    def half_amplitude_m(self) -> float:
        if self.y_peak_to_peak_cm is not None:
            return float(self.y_peak_to_peak_cm) * 0.01 / 2.0
        return self.amplitude_mm / 1000.0


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


def tool_z_spin_vel_base(pose_ref: np.ndarray, t_s: float, *, rz_amp_deg: float, omega: float,
                         soft_start: bool, ramp_s: float) -> np.ndarray:
    """Sinusoidal spin about tool +Z → base-frame angular velocity (small-angle)."""
    from scipy.spatial.transform import Rotation as Rsc

    if rz_amp_deg <= 0.0:
        return np.zeros(3)
    ramp = 1.0
    if soft_start and ramp_s > 0.0 and t_s < ramp_s:
        ramp = math.sin(0.5 * math.pi * t_s / ramp_s)
    wz_tool = math.radians(rz_amp_deg) * omega * math.cos(omega * t_s) * ramp
    r_mat = Rsc.from_euler("xyz", pose_ref[3:6], degrees=False).as_matrix()
    return r_mat @ np.array([0.0, 0.0, wz_tool], dtype=float)


class TrajectoryGenerator:
    """
    Built-in trajectory kinds (demos). Each sample() returns full 6D base-frame
    (pose_d, vel_ff). Drop tool-Z from vel_ff externally if desired; force hybrid
    fills tool-Z via force_axes.
    """

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

    def set_origin(self, pose0: np.ndarray) -> None:
        self.pose0 = np.asarray(pose0, dtype=float).copy()

    def sample(self, t_s: float) -> TrajectorySample:
        kind = self.cfg.kind
        if kind == "hold":
            return TrajectorySample(self.pose0.copy(), np.zeros(6))

        if kind in ("sin_base_y", "sin_base_y_tool_rz"):
            return self._sin_base_y(t_s, spin=(kind == "sin_base_y_tool_rz"))

        if kind == "sin_tool_y":
            return self._sin_tool_y(t_s)

        raise ValueError(f"Unknown trajectory type: {kind}")

    def _sin_base_y(self, t_s: float, *, spin: bool) -> TrajectorySample:
        dy, vy = sin_y_motion(
            t_s, self.amplitude_m, self.omega,
            soft_start=self.cfg.soft_start, ramp_s=self.cfg.ramp_s,
        )
        pose = self.pose0.copy()
        pose[1] += dy
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
        return TrajectorySample(pose, vel)

    def _sin_tool_y(self, t_s: float) -> TrajectorySample:
        dy, vy = sin_y_motion(
            t_s, self.amplitude_m, self.omega,
            soft_start=self.cfg.soft_start, ramp_s=self.cfg.ramp_s,
        )
        pose = np.asarray(
            tool_offset_pose(self.robot, list(self.pose0), 0.0, dy, 0.0), dtype=float
        )
        pose_p = np.asarray(
            tool_offset_pose(self.robot, list(self.pose0), 0.0, dy + vy * 1e-3, 0.0),
            dtype=float,
        )
        vel = (pose_p - pose) / 1e-3
        return TrajectorySample(pose, vel)

    @classmethod
    def from_dict(cls, raw: dict, pose0: np.ndarray, robot) -> TrajectoryGenerator:
        t = raw.get("trajectory", {})
        ps = t.get("period_s")
        y_pp_cm = t.get("y_peak_to_peak_cm")
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
            ),
            pose0,
            robot,
        )
```

### `rm75_control/control/velocity_admittance/observer.py`

```python
"""Compensated external wrench from rolling pose/force buffer + phi."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from rm75_control.force.compensation import regressor as fid
from rm75_control.force.compensation.paths import CONFIG_FORCE, PHI_JSON


@dataclass
class ForceObserverConfig:
    phi_path: Path = PHI_JSON
    phi_source: str = "phi_recommended"
    force_sensor: Path = CONFIG_FORCE
    fc_hz: float = 2.5
    buffer_s: float = 4.0
    min_samples: int = 35
    use_inertia: bool = False
    poll_hz: float = 100.0


@dataclass
class ForceSampleBuffer:
    max_len: int
    t: deque = field(default_factory=deque)
    pose: deque = field(default_factory=deque)
    force: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.t = deque(maxlen=self.max_len)
        self.pose = deque(maxlen=self.max_len)
        self.force = deque(maxlen=self.max_len)

    def append(self, t_s: float, pose6: np.ndarray, force6: np.ndarray) -> None:
        self.t.append(t_s)
        self.pose.append(np.asarray(pose6, dtype=float))
        self.force.append(np.asarray(force6, dtype=float))

    def __len__(self) -> int:
        return len(self.t)


class CompensatedForceObserver:
    def __init__(self, cfg: ForceObserverConfig) -> None:
        self._fid = fid
        self.cfg = cfg
        self.phi = self._load_phi(cfg.phi_path, cfg.phi_source)
        self.frame = fid.FrameConfig.from_yaml(cfg.force_sensor)
        max_len = max(cfg.min_samples + 5, int(cfg.buffer_s * cfg.poll_hz) + 5)
        self.buf = ForceSampleBuffer(max_len=max_len)

    @staticmethod
    def _load_phi(path: Path, source: str) -> np.ndarray:
        data = json.loads(path.read_text())
        if source not in data:
            raise SystemExit(f"Key '{source}' not in {path}")
        return np.array([data[source][k] for k in fid.PHI_NAMES])

    def append(self, t_s: float, pose6: np.ndarray, force_raw: np.ndarray) -> None:
        self.buf.append(t_s, pose6, force_raw)

    def ready(self) -> bool:
        return len(self.buf) >= self.cfg.min_samples

    def latest_wrench(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (signed_filtered_raw, f_ext) in sensor frame, or None while warming up."""
        if not self.ready():
            return None
        t = np.asarray(self.buf.t)
        pose = np.asarray(self.buf.pose)
        force = np.asarray(self.buf.force)
        W, Y = self._fid.build_dataset(
            pose, force, t, self.frame, fc=self.cfg.fc_hz, use_inertia=self.cfg.use_inertia
        )
        k = len(t) - 1
        sl = slice(6 * k, 6 * k + 6)
        raw_show = Y[sl].copy()
        f_ext = (Y[sl] - W[sl] @ self.phi).reshape(6)
        return raw_show, f_ext

    @classmethod
    def from_yaml(cls, raw: dict) -> CompensatedForceObserver:
        f = raw.get("force", {})
        fc_cfg = float(yaml.safe_load(CONFIG_FORCE.read_text()).get("filtfilt_cutoff_hz", 2.5))
        fc_hz = float(f.get("fc_hz", fc_cfg))
        timing = raw.get("timing", {})
        dt_ms = float(timing.get("dt_ms", 10.0))
        return cls(
            ForceObserverConfig(
                phi_path=PHI_JSON,
                phi_source=str(f.get("phi_source", "phi_recommended")),
                fc_hz=fc_hz,
                buffer_s=float(f.get("buffer_s", 4.0)),
                min_samples=int(f.get("min_samples", 35)),
                use_inertia=bool(f.get("use_inertia", False)),
                poll_hz=1000.0 / dt_ms,
            )
        )
```

### `rm75_control/control/velocity_admittance/loop.py`

```python
"""Shared velocity-admittance control loop."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

from rm75_control.force.compensation.collection import load_slot, move_j, wait_settle
from rm75_control.force.compensation.id_config import load_config
from rm75_control.force.compensation.paths import CONFIG_ID
from rm75_control.motion.canfd import send_velocity_canfd

from .controller import AdmittanceConfig, AdmittanceController
from .observer import CompensatedForceObserver
from .paths import CONFIG_ROBOT, PHI_JSON
from .trajectory import TrajectoryGenerator, sin_period_for_peak_vel


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def prepare_canfd_velocity_session(
    bot,
    *,
    settle_s: float = 0.5,
    clear_errors: bool = False,
) -> dict:
    """Re-sync planner idle immediately before rm_set_movev_canfd_init."""
    return bot.recover_controller(
        settle_s=settle_s,
        clear_errors=clear_errors,
        probe_force_stream=False,
    )


def init_velocity_canfd(robot, vc: dict, dt_ms: float) -> None:
    ret = robot.rm_set_movev_canfd_init(
        int(vc.get("avoid_singularity", 0)),
        int(vc.get("frame_type", 1)),
        int(dt_ms),
    )
    if ret != 0:
        raise RuntimeError(f"rm_set_movev_canfd_init failed: {ret}")


def settle_movev_after_init(
    robot,
    *,
    dt_ms: float,
    follow: bool,
    trajectory_mode: int,
    radio: int,
    n_frames: int = 10,
) -> None:
    """Zero-velocity frames after rm_set_movev_canfd_init — cuts mode-switch jerk."""
    dt_s = dt_ms / 1000.0
    zero = [0.0] * 6
    next_tick = time.monotonic()
    for _ in range(n_frames):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
        next_tick += dt_s
        send_velocity_canfd(
            robot, zero,
            follow=follow, trajectory_mode=trajectory_mode, radio=radio,
        )


def trajectory_summary(raw: dict) -> str:
    t = raw.get("trajectory", {})
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
    spin = f"  tool-Rz±{rz:.1f}°" if rz > 0 else ""
    return f"{kind}{soft}  {amp_label}{spin}  v_peak≈{vmax:.1f}cm/s  period={period_s}"


def run_velocity_admittance(
    raw: dict,
    *,
    title: str = "Velocity admittance",
    duration_s: float | None = None,
    tool_hint: bool = True,
) -> int:
    if not PHI_JSON.exists():
        raise SystemExit(f"Missing {PHI_JSON} — run force_calibrate.py first")

    timing = raw.get("timing", {})
    dt_ms = float(timing.get("dt_ms", 10.0))
    dt_s = dt_ms / 1000.0
    feedback_every = max(1, int(timing.get("feedback_every", 3)))
    vc = raw.get("velocity_canfd", {})
    follow = bool(vc.get("follow", True))
    traj_mode = int(vc.get("trajectory_mode", 0))
    radio = int(vc.get("radio", 0))

    startup = raw.get("startup", {})
    settle_frames = int(startup.get("settle_frames", 10))
    hold_s = float(startup.get("hold_s", 0.0))
    wait_contact = bool(startup.get("wait_contact", True))
    auto_start_under_n = float(startup.get("auto_start_under_n", 0.5))
    auto_start_hold_s = float(startup.get("auto_start_hold_s", 0.5))
    auto_start_samples = max(1, int(round(auto_start_hold_s / dt_s)))
    approach_ramp_s = float(startup.get("approach_ramp_s", 1.0))
    require_observer = bool(startup.get("require_observer_ready", True))
    pose_slot_raw = startup.get("pose_slot", "d")
    pose_slot = (
        None
        if pose_slot_raw in (None, "", "none", "null")
        else str(pose_slot_raw).lower()
    )
    move_speed = startup.get("move_speed")

    ctrl_cfg = AdmittanceConfig.from_dict(raw)
    control_frame = ctrl_cfg.control_frame
    frame_type = int(vc.get("frame_type", 0 if control_frame == "tool" else 1))
    if control_frame == "tool" and frame_type != 0:
        print("  NOTE: control_frame=tool → forcing frame_type=0 (TCP movev)", flush=True)
        frame_type = 0
    elif control_frame == "base" and frame_type != 1:
        print("  NOTE: control_frame=base → forcing frame_type=1 (world movev)", flush=True)
        frame_type = 1
    vc_run = {**vc, "frame_type": frame_type}
    traj_kind = str(raw.get("trajectory", {}).get("type", "hold"))
    observer = CompensatedForceObserver.from_yaml(raw)
    controller = AdmittanceController(dt_s, ctrl_cfg)

    f_cfg = raw.get("force", {})
    desired_z = float(f_cfg.get("desired_z_n", 3.0))
    f_des = np.zeros(6)
    f_des[2] = desired_z
    f_zero = np.zeros(6)
    auto_start_fz_n = float(startup.get("auto_start_fz_n", desired_z - auto_start_under_n))
    auto_recover = bool(startup.get("auto_recover", True))
    recover_probe_force = bool(startup.get("recover_probe_force_stream", False))

    print(
        f"{title} | rm_movev_canfd frame_type={frame_type} "
        f"follow={follow} traj={traj_mode} radio={radio}",
    )
    print(
        f"  kp_pos={ctrl_cfg.kp_pos.tolist()}  delay={ctrl_cfg.system_delay_s * 1000:.0f}ms  "
        f"feedback every {feedback_every} cycles (~{feedback_every * dt_ms:.0f}ms)",
    )
    print(f"  phi: {PHI_JSON.name}  Fz_des={desired_z:.2f} N")
    print(f"  vz cap (tool TCP): ±{ctrl_cfg.max_vz_tool_m_s * 100:.1f} cm/s")
    print(f"  trajectory: {trajectory_summary(raw)}  kind={traj_kind}")
    scan_mode = "open-loop ff" if ctrl_cfg.open_loop else "closed-loop track"
    print(
        f"  hybrid: traj=6D base  decouple=tool S_f={ctrl_cfg.force_axes.tolist()}  "
        f"movev={control_frame} frame_type={frame_type}  scan={scan_mode}",
        flush=True,
    )
    if wait_contact:
        print(
            f"  auto-start: Fz≥{auto_start_fz_n:.1f}N for {auto_start_hold_s:.1f}s "
            f"→ scan; approach Fz ramp {approach_ramp_s:.1f}s (no step at hold end)",
            flush=True,
        )
    if pose_slot:
        print(f"  startup pose: move_j → slot '{pose_slot}'", flush=True)
    if tool_hint:
        print("  Ensure gripper (or desired tool) is active in RM Web UI before contact tasks.")

    from rm75_control import RobotSession

    with RobotSession(config=CONFIG_ROBOT) as bot:
        if auto_recover:
            rec = bot.recover_controller(
                settle_s=1.0,
                clear_errors=True,
                probe_force_stream=recover_probe_force,
            )
            err = rec.get("system_err") or []
            print(
                f"  auto-recover: idle={rec.get('planning_idle')}  "
                f"traj={rec.get('trajectory_type_final')}  "
                f"sys_err={err or 'none'}",
                flush=True,
            )
            if err:
                print("  (cleared latched controller errors on connect)", flush=True)

        if pose_slot:
            fid = load_config(CONFIG_ID)
            spd = int(move_speed) if move_speed is not None else fid.collect.move_speed
            q_tgt, _, rec = load_slot(fid, pose_slot)
            print(
                f"  move_j → {pose_slot} ({rec.get('label', '')}) speed={spd}",
                flush=True,
            )
            move_j(bot.robot, q_tgt, speed=spd)
            pose_act, q_act = wait_settle(
                bot.robot, q_tgt, timeout_s=fid.collect.settle_timeout_s,
            )
            print(
                f"  settled q_max_err={float(np.max(np.abs(q_act - q_tgt))):.3f}°",
                flush=True,
            )

        ret, state = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            raise SystemExit(f"get state failed: {ret}")
        pose0 = np.asarray(state["pose"][:6], dtype=float)
        traj_origin = pose0.copy()
        print(
            f"  start TCP pose (base): "
            f"xyz=[{pose0[0]:.3f},{pose0[1]:.3f},{pose0[2]:.3f}] m",
            flush=True,
        )
        traj = TrajectoryGenerator.from_dict(raw, pose0, bot.robot)

        prep = prepare_canfd_velocity_session(bot, settle_s=0.5)
        print(
            f"  CANFD prep: idle={prep.get('planning_idle')}  "
            f"traj={prep.get('trajectory_type_final')}  "
            f"euler_deg={prep.get('pose_euler_deg', [])}",
            flush=True,
        )
        if not prep.get("planning_idle", False):
            print("  WARN: planner not idle before movev init — snap more likely", flush=True)

        pose_pre = pose0.copy()
        ret_pre, st_pre = bot.robot.rm_get_current_arm_state()
        if ret_pre == 0:
            pose_pre = np.asarray(st_pre["pose"][:6], dtype=float)

        init_velocity_canfd(bot.robot, vc_run, dt_ms)
        settle_movev_after_init(
            bot.robot, dt_ms=dt_ms, follow=follow,
            trajectory_mode=traj_mode, radio=radio, n_frames=max(settle_frames, 30),
        )
        ret_post, st_post = bot.robot.rm_get_current_arm_state()
        if ret_post == 0:
            pose_post = np.asarray(st_post["pose"][:6], dtype=float)
            deuler = np.degrees(pose_post[3:6] - pose_pre[3:6])
            deuler = (deuler + 180.0) % 360.0 - 180.0
            dpos_mm = (pose_post[:3] - pose_pre[:3]) * 1000.0
            print(
                f"  post-init settle Δpos_mm="
                f"[{dpos_mm[0]:+.2f},{dpos_mm[1]:+.2f},{dpos_mm[2]:+.2f}]  "
                f"Δeuler_deg=[{deuler[0]:+.2f},{deuler[1]:+.2f},{deuler[2]:+.2f}]",
                flush=True,
            )
            if float(np.max(np.abs(deuler))) > 0.5 or float(np.linalg.norm(dpos_mm)) > 2.0:
                print(
                    "  WARN: init snap detected — last run may leave force/CANFD mode; "
                    "retry or run tmp/recover_force_stream.py",
                    flush=True,
                )

        controller.reset()
        print("Velocity CANFD initialized. Ctrl+C to stop.", flush=True)

        t0 = time.monotonic()
        next_tick = t0
        last_log = t0
        scan_started = not wait_contact
        start_streak = 0
        t_scan0: float | None = None if wait_contact else t0
        cycle = 0
        pose_fb = pose0.copy()
        f_ext = f_zero.copy()
        last_wait_msg = 0.0
        fz_buf: list[float] = []

        def _fz_smooth() -> float:
            if not fz_buf:
                return float(f_ext[2])
            return float(np.median(fz_buf))

        try:
            while True:
                now = time.monotonic()
                if duration_s is not None and t_scan0 is not None:
                    if now - t_scan0 >= duration_s:
                        break
                if now < next_tick:
                    time.sleep(min(0.002, next_tick - now))
                    continue
                next_tick += dt_s
                t_s = now - t0

                if cycle % feedback_every == 0:
                    ret_s, st = bot.robot.rm_get_current_arm_state()
                    ret_f, fd = bot.robot.rm_get_force_data()
                    if ret_s == 0 and ret_f == 0:
                        pose_fb = np.asarray(st["pose"][:6], dtype=float)
                        force_raw = np.asarray(fd["force_data"][:6], dtype=float)
                        observer.append(t_s, pose_fb, force_raw)
                        wrench = observer.latest_wrench()
                        if wrench is not None:
                            f_ext = wrench[1]
                            fz_buf.append(float(f_ext[2]))
                            if len(fz_buf) > 7:
                                fz_buf.pop(0)
                cycle += 1
                pose = pose_fb

                if not scan_started and wait_contact and t_s >= hold_s:
                    if require_observer and not observer.ready():
                        if t_s - last_wait_msg >= 2.0:
                            print(
                                f"  waiting phi observer ({len(observer.buf)}/"
                                f"{observer.cfg.min_samples})…",
                                flush=True,
                            )
                            last_wait_msg = t_s
                        start_streak = 0
                    else:
                        fz_s = _fz_smooth()
                        if fz_s >= auto_start_fz_n:
                            start_streak += 1
                        else:
                            start_streak = 0
                        if start_streak >= auto_start_samples:
                            scan_started = True
                            t_scan0 = now
                            traj_origin = pose.copy()
                            traj.set_origin(traj_origin)
                            controller.force_error_integral.fill(0.0)
                            controller.last_v_cmd.fill(0.0)
                            print(
                                f"  scan ON @ t={t_s:.1f}s  Fz={f_ext[2]:+.2f}N  traj={traj_kind}",
                                flush=True,
                            )

                if not scan_started:
                    if t_s < hold_s or (require_observer and not observer.ready()):
                        v_cmd = [0.0] * 6
                    else:
                        since_approach = max(0.0, t_s - hold_s)
                        f_scale = (
                            min(1.0, since_approach / approach_ramp_s)
                            if approach_ramp_s > 0
                            else 1.0
                        )
                        v_cmd = controller.compute_velocity_command(
                            pose, pose0, np.zeros(6), f_ext, f_des * f_scale,
                            in_contact=False,
                        )
                    send_velocity_canfd(
                        bot.robot, np.asarray(v_cmd, dtype=float).tolist(),
                        follow=follow, trajectory_mode=traj_mode, radio=radio,
                    )
                    continue

                t_scan = now - t_scan0 if t_scan0 is not None else 0.0
                sample = traj.sample(t_scan)
                v_cmd = controller.compute_velocity_command(
                    pose, sample.pose_d, sample.vel_ff, f_ext, f_des,
                )
                send_velocity_canfd(
                    bot.robot, np.asarray(v_cmd, dtype=float).tolist(),
                    follow=follow, trajectory_mode=traj_mode, radio=radio,
                )

                if now - last_log >= 1.0:
                    last_log = now
                    from .controller import wrap_pi

                    dy_mm = float(pose[1] - traj_origin[1]) * 1000.0
                    deuler = np.degrees([
                        wrap_pi(float(pose[i] - traj_origin[i])) for i in range(3, 6)
                    ])
                    print(
                        f"  t={t_s:.1f}s  ΔY_world={dy_mm:+.1f}mm  "
                        f"Δeuler_deg=[{deuler[0]:+.2f},{deuler[1]:+.2f},{deuler[2]:+.2f}]  "
                        f"Fz_ext={f_ext[2]:+.2f}N  "
                        f"vy={v_cmd[1]:+.4f} ({control_frame} movev)",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
        finally:
            try:
                settle_movev_after_init(
                    bot.robot, dt_ms=dt_ms, follow=follow,
                    trajectory_mode=traj_mode, radio=radio, n_frames=15,
                )
                bot.robot.rm_set_arm_slow_stop()
            except Exception:
                pass

    return 0
```

## 三、标定与力补偿

### `tmp/force_compensation/force_calibrate.py`

```python
#!/usr/bin/env python3
"""
One-shot force compensation calibration: collect A→B→C→D→A, then identify φ.

  source env.sh
  python tmp/force_compensation/force_calibrate.py
  python tmp/force_compensation/force_calibrate.py --dry-run
  python tmp/force_compensation/force_calibrate.py --identify-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rm75_control.force.compensation import collection, identification
from rm75_control.force.compensation.paths import CONFIG_ID


def _collect_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    if args.config != CONFIG_ID:
        argv += ["--config", str(args.config)]
    if args.dry_run:
        argv.append("--dry-run")
    if args.save_pose:
        argv += ["--save-pose", args.save_pose]
    if args.pose_label:
        argv += ["--pose-label", args.pose_label]
    return argv


def _identify_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    if args.config != CONFIG_ID:
        argv += ["--id-config", str(args.config)]
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect + identify force compensation φ")
    parser.add_argument("--config", type=Path, default=CONFIG_ID, help="config/force_id.yaml")
    parser.add_argument("--dry-run", action="store_true", help="preview collection only")
    parser.add_argument("--save-pose", type=str, default=None, metavar="SLOT")
    parser.add_argument("--pose-label", type=str, default=None)
    parser.add_argument("--identify-only", action="store_true", help="skip collection, fit existing npz")
    args = parser.parse_args()

    if args.identify_only and (args.dry_run or args.save_pose):
        parser.error("--identify-only cannot combine with --dry-run or --save-pose")

    collect_argv = _collect_argv(args)

    if args.save_pose:
        return collection.main(collect_argv)

    if args.identify_only:
        return identification.main(_identify_argv(args))

    if args.dry_run:
        rc = collection.main(collect_argv)
        if rc != 0:
            return rc
        print("\n(dry-run: identification would run after collection)")
        return 0

    rc = collection.main(collect_argv)
    if rc != 0:
        return rc

    return identification.main(_identify_argv(args))


if __name__ == "__main__":
    raise SystemExit(main())
```

### `tmp/force_compensation/force_monitor.py`

```python
#!/usr/bin/env python3
"""
Live 6D force compensation monitor — drag arm manually, watch raw vs F_ext.

  source env.sh
  python tmp/force_compensation/force_monitor.py

Config: config/force_id.yaml (monitor section), logs/force_id_phi.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import numpy as np
import yaml

from rm75_control.force.compensation import regressor as fid
from rm75_control.force.compensation.id_config import load_config
from rm75_control.force.compensation.paths import CONFIG_ID, CONFIG_ROBOT

AXIS_LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
FORCE_IDX = (0, 1, 2)
MOM_IDX = (3, 4, 5)


def load_phi(path: Path, source: str) -> tuple[np.ndarray, str]:
    data = json.loads(path.read_text())
    if source not in data:
        raise SystemExit(f"Key '{source}' not in {path}. Keys: {list(data.keys())}")
    phi = np.array([data[source][k] for k in fid.PHI_NAMES])
    return phi, source


@dataclass
class SampleBuffer:
    max_len: int
    t: deque = field(default_factory=deque)
    pose: deque = field(default_factory=deque)
    force: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.t = deque(maxlen=self.max_len)
        self.pose = deque(maxlen=self.max_len)
        self.force = deque(maxlen=self.max_len)

    def append(self, t_s: float, pose6: np.ndarray, force6: np.ndarray) -> None:
        self.t.append(t_s)
        self.pose.append(np.asarray(pose6, dtype=float))
        self.force.append(np.asarray(force6, dtype=float))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.asarray(self.t), np.asarray(self.pose), np.asarray(self.force)


def compensate_latest(
    buf: SampleBuffer,
    phi: np.ndarray,
    cfg: fid.FrameConfig,
    fc: float,
    *,
    use_inertia: bool,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(buf.t) < min_samples:
        return None
    t, pose, force = buf.arrays()
    W, Y = fid.build_dataset(pose, force, t, cfg, fc=fc, use_inertia=use_inertia)
    k = len(t) - 1
    sl = slice(6 * k, 6 * k + 6)
    return Y[sl].copy(), (Y[sl] - W[sl] @ phi).reshape(6)


class CompMonitor:
    def __init__(self, *, window_s: float, refresh_hz: float = 12.0) -> None:
        import matplotlib.pyplot as plt

        self.window_s = window_s
        self.refresh_interval = 1.0 / refresh_hz
        self._lock = Lock()
        max_pts = max(int(window_s * refresh_hz * 1.5) + 20, 200)
        self._t: deque[float] = deque(maxlen=max_pts)
        self._raw: list[deque[float]] = [deque(maxlen=max_pts) for _ in range(6)]
        self._ext: list[deque[float]] = [deque(maxlen=max_pts) for _ in range(6)]
        self._status = "Collecting buffer..."
        self._last_refresh = 0.0

        plt.ion()
        self._fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        self._fig.suptitle("6D force: raw (signed, filtered) vs compensated F_ext")
        self._axes = axes.ravel()
        self._line_raw: list = []
        self._line_ext: list = []
        for i, ax in enumerate(self._axes):
            unit = "N" if i < 3 else "N·m"
            (lr,) = ax.plot([], [], color="#2563eb", linewidth=1.2, alpha=0.85, label="raw")
            (le,) = ax.plot([], [], color="#ea580c", linewidth=1.4, label="F_ext")
            ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.35)
            ax.set_ylabel(f"{AXIS_LABELS[i]} ({unit})")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            self._line_raw.append(lr)
            self._line_ext.append(le)
        for ax in self._axes[3:]:
            ax.set_xlabel("Time (s)")
        self._text = self._fig.text(0.01, 0.01, "", fontsize=9, family="monospace")
        self._fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        try:
            self._fig.canvas.manager.set_window_title("RM75 force compensation monitor")
        except Exception:
            pass
        self._fig.show()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def append(self, t_s: float, raw6: np.ndarray, ext6: np.ndarray | None = None) -> None:
        with self._lock:
            self._t.append(t_s)
            for i in range(6):
                self._raw[i].append(float(raw6[i]))
                self._ext[i].append(float(ext6[i]) if ext6 is not None else float("nan"))

    def set_status(self, msg: str) -> None:
        with self._lock:
            self._status = msg

    def refresh(self, now: float) -> None:
        if now - self._last_refresh < self.refresh_interval:
            return
        self._last_refresh = now
        with self._lock:
            if not self._t:
                return
            ts = np.asarray(self._t)
            t_end = float(ts[-1])
            t_start = max(0.0, t_end - self.window_s)
            mask = ts >= t_start
            xs = ts[mask]
            status = self._status
            raw_pts = [np.asarray(self._raw[i])[mask] for i in range(6)]
            ext_pts = [np.asarray(self._ext[i])[mask] for i in range(6)]

        for i in range(6):
            self._line_raw[i].set_data(xs, raw_pts[i])
            self._line_ext[i].set_data(xs, ext_pts[i])
            vals = np.concatenate([raw_pts[i], ext_pts[i]])
            finite = vals[np.isfinite(vals)]
            if len(finite):
                y0, y1 = float(np.min(finite)), float(np.max(finite))
                pad = max(0.5, 0.15 * (y1 - y0 + 1e-6))
                self._axes[i].set_ylim(y0 - pad, y1 + pad)
            self._axes[i].set_xlim(t_start, max(t_end, t_start + 1.0))

        if len(xs) >= 5:
            rf = float(np.sqrt(np.nanmean(
                np.sum(np.stack([raw_pts[j] for j in FORCE_IDX], axis=1) ** 2, axis=1)
            )))
            ef = np.stack([ext_pts[j] for j in FORCE_IDX], axis=1)
            ef_ok = np.isfinite(ef).all(axis=1)
            if np.any(ef_ok):
                re = float(np.sqrt(np.mean(np.sum(ef[ef_ok] ** 2, axis=1))))
                em = np.stack([ext_pts[j] for j in MOM_IDX], axis=1)
                em_ok = np.isfinite(em).all(axis=1)
                rm = float(np.sqrt(np.mean(np.sum(em[em_ok] ** 2, axis=1)))) if np.any(em_ok) else float("nan")
                status = f"{status}  |  |F| raw={rf:.2f}N ext={re:.2f}N  |M| ext={rm:.3f}N·m"
            else:
                status = f"{status}  |  |F| raw={rf:.2f}N  (F_ext warming up)"
        self._text.set_text(status)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        import matplotlib.pyplot as plt
        plt.close(self._fig)
        plt.ioff()


def main() -> int:
    parser = argparse.ArgumentParser(description="Live 6D compensated force plot")
    parser.add_argument("--id-config", type=Path, default=CONFIG_ID)
    parser.add_argument("--phi", type=Path, default=None)
    parser.add_argument("--phi-source", type=str, default=None)
    parser.add_argument("--10p-only", dest="only_10p", action="store_true")
    args = parser.parse_args()

    id_cfg = load_config(args.id_config)
    mc = id_cfg.monitor
    fc_cfg = id_cfg.fit
    phi_path = args.phi or fc_cfg.phi_output
    phi_src = args.phi_source or mc.phi_source

    phi, src = load_phi(phi_path, phi_src)
    if args.only_10p:
        phi = phi.copy()
        phi[4:10] = 0.0
        src = f"{src} (I=0)"

    frame = fid.FrameConfig.from_yaml(fc_cfg.force_sensor)
    fc = float(yaml.safe_load(fc_cfg.force_sensor.read_text()).get("filtfilt_cutoff_hz", 2.5))
    use_inertia = mc.use_inertia and float(np.max(np.abs(phi[4:10]))) > 1e-9

    max_buf = max(mc.min_samples + 10, int(mc.buffer_s * 1000 / mc.poll_ms) + 5)
    buf = SampleBuffer(max_len=max_buf)
    sign = np.array(frame.force_sign, dtype=float)

    print(f"φ ({src}) from {phi_path}  m={phi[0]:.3f} kg")
    print(f"poll={mc.poll_ms}ms  window={mc.window_s}s  buffer≈{mc.buffer_s}s")
    print("Drag arm in FREE SPACE. Close plot or Ctrl+C to stop.")

    from rm75_control import RobotSession

    monitor = CompMonitor(window_s=mc.window_s, refresh_hz=mc.refresh_hz)
    dt_s = mc.poll_ms / 1000.0

    try:
        with RobotSession(config=CONFIG_ROBOT) as bot:
            t0 = time.monotonic()
            next_poll = t0
            while True:
                import matplotlib.pyplot as plt
                if not plt.fignum_exists(monitor._fig.number):
                    print("Plot closed — exiting.")
                    break
                now = time.monotonic()
                if now < next_poll:
                    time.sleep(min(0.02, next_poll - now))
                    continue
                next_poll += dt_s
                t_s = now - t0

                ret_s, st = bot.robot.rm_get_current_arm_state()
                ret_f, fd = bot.robot.rm_get_force_data()
                if ret_s != 0 or ret_f != 0:
                    monitor.set_status(f"API err s={ret_s} f={ret_f}")
                    monitor.refresh(now)
                    continue

                pose = np.asarray(st["pose"][:6], dtype=float)
                force = np.asarray(fd["force_data"][:6], dtype=float)
                buf.append(t_s, pose, force)

                comp = compensate_latest(
                    buf, phi, frame, fc,
                    use_inertia=use_inertia, min_samples=mc.min_samples,
                )
                if comp is None:
                    monitor.set_status(f"buffer {len(buf.t)}/{mc.min_samples}")
                    monitor.append(t_s, force * sign, None)
                else:
                    raw_show, ext_show = comp
                    monitor.set_status(f"OK n={len(buf.t)}")
                    monitor.append(t_s, raw_show, ext_show)
                monitor.refresh(now)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `rm75_control/force/__init__.py`

```python
"""Force sensor, scan streaming, and compensation identification."""

from rm75_control.force.compensation.paths import CONFIG_FORCE, CONFIG_ID, PHI_JSON
from rm75_control.force.scan import ForceScanConfig, ForceScanController

__all__ = [
    "CONFIG_FORCE",
    "CONFIG_ID",
    "ForceScanConfig",
    "ForceScanController",
    "PHI_JSON",
]
```

### `rm75_control/force/wrench.py`

```python
"""Force sensor read / zero / calibration."""
```

### `rm75_control/force/scan.py`

```python
"""Force-position scan streaming for surface following (native hybrid control)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from rm75_control.core.exceptions import MotionError
from rm75_control.motion.force_position import ForcePositionClient, send_force_position_move

Pose6 = Sequence[float]
Joint7 = Sequence[float]


@dataclass
class ForceScanConfig:
    """Native rm_force_position_move params + loop timing."""

    flag: int = 0
    sensor: int = 1
    mode: int = 1
    follow: bool = True
    control_mode: list[int] = field(default_factory=lambda: [3, 3, 4, 0, 0, 0])
    desired_force: list[float] = field(default_factory=lambda: [0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    limit_vel: list[float] = field(
        default_factory=lambda: [0.05, 0.05, 0.05, 0.3, 0.3, 0.3]
    )
    trajectory_mode: int = 0
    radio: int = 0
    period_ms: float = 10.0

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        **overrides: Any,
    ) -> ForceScanConfig:
        force_cfg = config.get("force", {})
        scan_cfg = force_cfg.get("scan", {})
        timing = config.get("timing", {})
        values: dict[str, Any] = {
            "flag": scan_cfg.get("flag", 0),
            "sensor": scan_cfg.get("sensor", force_cfg.get("sensor", 1)),
            "mode": scan_cfg.get("mode", force_cfg.get("coordinate_mode", 1)),
            "follow": scan_cfg.get("follow", True),
            "control_mode": scan_cfg.get(
                "control_mode", force_cfg.get("default_control_mode", [3, 3, 4, 0, 0, 0])
            ),
            "desired_force": scan_cfg.get(
                "desired_force", force_cfg.get("default_desired_force", [0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
            ),
            "limit_vel": scan_cfg.get("limit_vel", [0.05, 0.05, 0.05, 0.3, 0.3, 0.3]),
            "trajectory_mode": scan_cfg.get("trajectory_mode", 0),
            "radio": scan_cfg.get("radio", 0),
            "period_ms": scan_cfg.get("period_ms", timing.get("force_scan_period_ms", 10.0)),
        }
        values.update(overrides)
        return cls(**values)


class ForceScanController:
    """
    Periodic rm_force_position_move loop.

    MPC / xi-space planning stays outside this class: provide a callback that
    maps (xi, observation) -> joint or pose command each cycle.
    """

    def __init__(
        self,
        robot: ForcePositionClient,
        config: ForceScanConfig | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        self.robot = robot
        self.config = config or ForceScanConfig()
        self.dry_run = dry_run
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        if self.dry_run:
            self._active = True
            return
        ret = self.robot.rm_start_force_position_move()
        if ret != 0:
            raise MotionError(f"rm_start_force_position_move failed with code {ret}")
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        if not self.dry_run:
            ret = self.robot.rm_stop_force_position_move()
            if ret != 0:
                # -2 receive timeout is common after Ctrl+C; fall back to arm stop.
                try:
                    self.robot.rm_set_arm_slow_stop()
                except Exception:
                    pass
                if ret not in (-2,):
                    raise MotionError(
                        f"rm_stop_force_position_move failed with code {ret}"
                    )
        self._active = False

    def step_joint(
        self,
        joint: Joint7,
        *,
        desired_force: Sequence[float] | None = None,
        control_mode: Sequence[int] | None = None,
    ) -> None:
        self._send(flag=0, joint=joint, pose=None, desired_force=desired_force, control_mode=control_mode)

    def step_pose(
        self,
        pose: Pose6,
        *,
        desired_force: Sequence[float] | None = None,
        control_mode: Sequence[int] | None = None,
    ) -> None:
        self._send(flag=1, joint=None, pose=pose, desired_force=desired_force, control_mode=control_mode)

    def run_xi_loop(
        self,
        xi0: Sequence[float],
        step_fn: Callable[[Sequence[float], float], tuple[Sequence[float], int, Sequence[float] | None]],
        *,
        duration_s: float | None = None,
        max_steps: int | None = None,
    ) -> None:
        """
        Run scan loop driven by xi-space policy.

        step_fn(xi, t) -> (xi_next, command_flag, command)
            command_flag 0: command is joint[7]
            command_flag 1: command is pose[6]
        """
        if not self._active:
            raise RuntimeError("ForceScanController is not started")

        dt = self.config.period_ms / 1000.0
        xi = list(xi0)
        t = 0.0
        steps = 0
        while True:
            xi, flag, cmd = step_fn(xi, t)
            if flag == 0:
                self.step_joint(cmd)
            elif flag == 1:
                self.step_pose(cmd)
            else:
                raise ValueError(f"command_flag must be 0 or 1, got {flag}")

            t += dt
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
            if duration_s is not None and t >= duration_s:
                break
            if dt > 0.0:
                time.sleep(dt)

    def _send(
        self,
        *,
        flag: int,
        joint: Joint7 | None,
        pose: Pose6 | None,
        desired_force: Sequence[float] | None,
        control_mode: Sequence[int] | None,
    ) -> None:
        if not self._active:
            raise RuntimeError("ForceScanController is not started")

        from Robotic_Arm.rm_ctypes_wrap import (
            rm_euler_t,
            rm_force_position_move_t,
            rm_pose_t,
            rm_position_t,
        )
        from ctypes import c_float, c_int

        cfg = self.config
        param = rm_force_position_move_t()
        param.flag = flag
        param.sensor = cfg.sensor
        param.mode = cfg.mode
        param.follow = cfg.follow
        param.trajectory_mode = cfg.trajectory_mode
        param.radio = cfg.radio

        if flag == 1 and pose is not None:
            po = rm_pose_t()
            po.position = rm_position_t(*pose[:3])
            po.euler = rm_euler_t(*pose[3:6])
            param.pose = po
        elif flag == 0 and joint is not None:
            param.joint = (c_float * 7)(*joint)

        modes = list(control_mode) if control_mode is not None else cfg.control_mode
        forces = list(desired_force) if desired_force is not None else cfg.desired_force
        param.control_mode = (c_int * 6)(*modes)
        param.desired_force = (c_float * 6)(*forces)
        param.limit_vel = (c_float * 6)(*cfg.limit_vel)

        if not self.dry_run:
            send_force_position_move(self.robot, param)
```

### `rm75_control/force/compensation/__init__.py`

```python
"""Force compensation identification (multi-pose φ)."""

from rm75_control.force.compensation.regressor import (
    PHI_NAMES,
    FrameConfig,
    build_dataset,
    com_dict_mm,
    com_from_phi,
    kinematics_sensor,
)

__all__ = [
    "PHI_NAMES",
    "FrameConfig",
    "build_dataset",
    "com_dict_mm",
    "com_from_phi",
    "kinematics_sensor",
]
```

### `rm75_control/force/compensation/paths.py`

```python
"""Shared paths for force-ID data and configs (under tmp/force_compensation)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "tmp" / "force_compensation"
CONFIG_DIR = DATA_DIR / "config"
LOG_DIR = DATA_DIR / "logs"
CONFIG_ROBOT = REPO / "configs" / "rm75f_default.yaml"
CONFIG_FORCE = REPO / "configs" / "force_sensor.yaml"
CONFIG_ID = CONFIG_DIR / "force_id.yaml"
POSES_YAML = CONFIG_DIR / "poses.yaml"
PHI_JSON = LOG_DIR / "force_id_phi.json"

POSE_SLOTS = ("a", "b", "c", "d")


def npz_for_slot(slot: str) -> Path:
    return LOG_DIR / f"force_id_pose_{slot}.npz"
```

### `rm75_control/force/compensation/id_config.py`

```python
"""Load config/force_id.yaml into typed settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .paths import CONFIG_ID, LOG_DIR, REPO, npz_for_slot


@dataclass(frozen=True)
class VelocityBurstConfig:
    profile: str
    amp_deg_s: float
    freqs_hz: list[float]
    segment_s: float
    ramp_s: float
    ramp_down_s: float
    frame_type: int
    avoid_singularity: int
    follow: bool
    trajectory_mode: int
    radio: int
    axis_order: tuple[int, int, int]


# Validated pose D rm_movev_canfd burst (base frame, traj=0 passthrough + init settle).
POSE_D_VEL_BURST: dict = {
    "amp_deg_s": 12.0,
    "freqs_hz": [0.28],
    "segment_s": 15.0,
    "ramp_s": 3.0,
    "ramp_down_s": 4.0,
    "frame_type": 1,
    "avoid_singularity": 0,
    "follow": True,
    "trajectory_mode": 0,
    "radio": 0,
    "axis_order": (0, 1, 2),
}

BURST_PROFILES: dict[str, dict] = {
    "pose_d_vel_burst": POSE_D_VEL_BURST,
}

DEFAULT_BURST_PROFILE = "pose_d_vel_burst"


def load_velocity_burst(raw: dict) -> VelocityBurstConfig:
    name = str(raw.get("profile", DEFAULT_BURST_PROFILE))
    if name not in BURST_PROFILES:
        raise ValueError(
            f"Unknown pose_d.velocity_burst.profile {name!r}; "
            f"choose from {list(BURST_PROFILES)}"
        )
    base = BURST_PROFILES[name]
    overrides = {
        k: raw[k]
        for k in (
            "amp_deg_s", "freqs_hz", "segment_s", "ramp_s", "ramp_down_s",
            "frame_type", "avoid_singularity", "follow", "trajectory_mode",
            "radio", "axis_order",
        )
        if k in raw
    }
    if "axis_order" in overrides:
        overrides["axis_order"] = tuple(int(x) for x in overrides["axis_order"])
    if "freqs_hz" in overrides:
        overrides["freqs_hz"] = [float(x) for x in overrides["freqs_hz"]]
    p = {**base, **overrides}
    return VelocityBurstConfig(
        profile=name,
        amp_deg_s=float(p["amp_deg_s"]),
        freqs_hz=list(p["freqs_hz"]),
        segment_s=float(p["segment_s"]),
        ramp_s=float(p["ramp_s"]),
        ramp_down_s=float(p.get("ramp_down_s", 4.0)),
        frame_type=int(p["frame_type"]),
        avoid_singularity=int(p["avoid_singularity"]),
        follow=bool(p["follow"]),
        trajectory_mode=int(p["trajectory_mode"]),
        radio=int(p["radio"]),
        axis_order=tuple(int(x) for x in p["axis_order"]),
    )


@dataclass(frozen=True)
class CartesianConfig:
    duration_s: float
    max_delta_mm: float
    max_orient_deg: dict[str, float]
    amp_mm: np.ndarray
    amp_rot_deg: np.ndarray
    amp_rot_deg_slots: dict[str, np.ndarray]
    freqs_hz: list[list[float]]

    def max_deg_for_slot(self, slot: str) -> float:
        return float(self.max_orient_deg.get(slot, self.max_orient_deg.get("a", 18.0)))

    def amp_rot_for_slot(self, slot: str) -> np.ndarray:
        if slot in self.amp_rot_deg_slots:
            return self.amp_rot_deg_slots[slot]
        return self.amp_rot_deg


@dataclass(frozen=True)
class PoseDConfig:
    joint_duration_s: float
    burst_duration_s: float
    joint_amp_deg: np.ndarray
    joint_max_delta_deg: np.ndarray
    joint_freqs_hz: list[list[float]]
    velocity_burst: VelocityBurstConfig


@dataclass(frozen=True)
class CollectConfig:
    move_speed: int
    settle_timeout_s: float
    dt_ms: float
    log_every: int
    scale: float
    warmup_s: float
    follow: bool
    cartesian: CartesianConfig
    pose_d: PoseDConfig
    sequence: tuple[str, ...]
    return_home: str


@dataclass(frozen=True)
class FitConfig:
    force_sensor: Path
    holdout_frac: float
    alpha_percentile: float
    min_burst_rows: int
    min_high_alpha_rows: int
    inertia_r_max_m: float
    npz_paths: list[Path]
    phi_output: Path
    phi_recommended_key: str


@dataclass(frozen=True)
class MonitorConfig:
    poll_ms: float
    window_s: float
    buffer_s: float
    min_samples: int
    refresh_hz: float
    phi_source: str
    use_inertia: bool


@dataclass(frozen=True)
class ForceIdConfig:
    poses_yaml: Path
    log_dir: Path
    collect: CollectConfig
    fit: FitConfig
    monitor: MonitorConfig


def _resolve_path(path: str, *, config_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "configs":
        return REPO / p
    return config_dir / p


def load_config(path: Path | None = None) -> ForceIdConfig:
    path = path or CONFIG_ID
    config_dir = path.parent
    raw = yaml.safe_load(path.read_text()) or {}

    c = raw.get("collect", {})
    cart = c.get("cartesian", {})
    pd = c.get("pose_d", {})
    br = pd.get("velocity_burst") or {}
    if not br:
        raise ValueError("pose_d.velocity_burst required (profile: pose_d_vel_burst)")
    rot_slots = cart.get("amp_rot_deg_slots", {})
    f = raw.get("fit", {})
    m = raw.get("monitor", {})

    sequence = tuple(str(s) for s in raw.get("sequence", ["a", "b", "c", "d"]))
    slots = f.get("npz_slots", list(sequence))
    phi_name = f.get("phi_output", "force_id_phi.json")

    return ForceIdConfig(
        poses_yaml=_resolve_path(raw.get("poses_yaml", "poses.yaml"), config_dir=config_dir),
        log_dir=LOG_DIR,
        collect=CollectConfig(
            move_speed=int(c.get("move_speed", 15)),
            settle_timeout_s=float(c.get("settle_timeout_s", 15.0)),
            dt_ms=float(c.get("dt_ms", 10.0)),
            log_every=int(c.get("log_every", 10)),
            scale=float(c.get("scale", 1.0)),
            warmup_s=float(c.get("warmup_s", 3.0)),
            follow=bool(c.get("follow", False)),
            sequence=sequence,
            return_home=str(raw.get("return_home", "a")),
            cartesian=CartesianConfig(
                duration_s=float(cart.get("duration_s", 30.0)),
                max_delta_mm=float(cart.get("max_delta_mm", 5.0)),
                max_orient_deg={str(k): float(v) for k, v in cart.get("max_orient_deg", {}).items()},
                amp_mm=np.asarray(cart.get("amp_mm", [3, 4, 2]), dtype=float),
                amp_rot_deg=np.asarray(cart.get("amp_rot_deg", [12, 15, 12]), dtype=float),
                amp_rot_deg_slots={
                    str(k): np.asarray(v, dtype=float)
                    for k, v in rot_slots.items()
                },
                freqs_hz=[list(map(float, row)) for row in cart.get("freqs_hz", [])],
            ),
            pose_d=PoseDConfig(
                joint_duration_s=float(pd.get("joint_duration_s", 30.0)),
                burst_duration_s=float(pd.get("burst_duration_s", 45.0)),
                joint_amp_deg=np.asarray(pd.get("joint_amp_deg", [10] * 7), dtype=float),
                joint_max_delta_deg=np.asarray(pd.get("joint_max_delta_deg", [12] * 7), dtype=float),
                joint_freqs_hz=[list(map(float, row)) for row in pd.get("joint_freqs_hz", [])],
                velocity_burst=load_velocity_burst(br),
            ),
        ),
        fit=FitConfig(
            force_sensor=_resolve_path(f.get("force_sensor", "configs/force_sensor.yaml"), config_dir=config_dir),
            holdout_frac=float(f.get("holdout_frac", 0.2)),
            alpha_percentile=float(f.get("alpha_percentile", 70.0)),
            min_burst_rows=int(f.get("min_burst_rows", 300)),
            min_high_alpha_rows=int(f.get("min_high_alpha_rows", 150)),
            inertia_r_max_m=float(f.get("inertia_r_max_m", 0.12)),
            npz_paths=[npz_for_slot(str(s)) for s in slots],
            phi_output=LOG_DIR / phi_name,
            phi_recommended_key=str(f.get("phi_recommended_key", "phi_burst")),
        ),
        monitor=MonitorConfig(
            poll_ms=float(m.get("poll_ms", 50.0)),
            window_s=float(m.get("window_s", 25.0)),
            buffer_s=float(m.get("buffer_s", 4.0)),
            min_samples=int(m.get("min_samples", 35)),
            refresh_hz=float(m.get("refresh_hz", 12.0)),
            phi_source=str(m.get("phi_source", "phi_recommended")),
            use_inertia=bool(m.get("use_inertia", True)),
        ),
    )
```

### `rm75_control/force/compensation/excitation.py`

```python
"""Excitation trajectories and pose YAML helpers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .id_config import CartesianConfig, PoseDConfig, VelocityBurstConfig

DEG2RAD = math.pi / 180.0


def load_poses_yaml(path: Path) -> dict:
    if not path.exists():
        return {"poses": {}}
    return yaml.safe_load(path.read_text()) or {"poses": {}}


def save_poses_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True))


def save_pose_slot(
    path: Path, slot: str, pose6: np.ndarray, q_deg: np.ndarray, label: str | None
) -> None:
    data = load_poses_yaml(path)
    data.setdefault("poses", {})[slot] = {
        "label": label or f"pose_{slot}",
        "note": f"saved {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "pose_base": [round(float(v), 6) for v in pose6],
        "q_deg": [round(float(v), 3) for v in q_deg],
    }
    save_poses_yaml(path, data)


def get_slot_record(data: dict, slot: str) -> dict | None:
    rec = data.get("poses", {}).get(slot)
    if not rec or rec.get("pose_base") is None:
        return None
    return rec


def pose_drift_mm_deg(current: np.ndarray, recorded: np.ndarray) -> tuple[float, float]:
    dpos = float(np.linalg.norm(current[:3] - recorded[:3])) * 1000.0
    deul = np.abs(current[3:6] - recorded[3:6])
    deul = np.minimum(deul, 2 * math.pi - deul)
    ddeg = float(np.max(deul) * 180.0 / math.pi)
    return dpos, ddeg


def vel_burst_cmd(
    t_s: float,
    vb: VelocityBurstConfig,
    *,
    scale: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Cartesian angular velocity burst; returns (6D vel rad/s, axis_idx 0=wx..2=wz)."""
    amp_rad_s = vb.amp_deg_s * scale * DEG2RAD
    segment_s = vb.segment_s
    ramp_s = vb.ramp_s
    axis_order = vb.axis_order
    freqs_hz = vb.freqs_hz

    seg_slot = int(t_s // segment_s) % 3
    axis_idx = axis_order[seg_slot]
    t_loc = t_s - seg_slot * segment_s
    axis = 3 + axis_idx

    ramp_global = min(1.0, t_s / ramp_s) if ramp_s > 0 else 1.0
    ramp_seg = min(1.0, t_loc / min(ramp_s, segment_s * 0.2)) if ramp_s > 0 else 1.0
    env = ramp_global * ramp_seg

    vel = np.zeros(6, dtype=float)
    for k, f in enumerate(freqs_hz):
        ph = seg_slot * 1.4 + k * 0.85
        vel[axis] += amp_rad_s * math.sin(2.0 * math.pi * f * t_loc + ph)
    vel *= env
    return vel, axis_idx


def prepare_movev_session(bot) -> dict:
    diag = bot.prepare_for_force_stream(settle_s=1.0)
    diag["delete_traj"] = bot.robot.rm_set_arm_delete_trajectory()
    diag["slow_stop"] = bot.robot.rm_set_arm_slow_stop()
    time.sleep(0.5)
    traj = bot.robot.rm_get_arm_current_trajectory()
    diag["trajectory_type"] = traj.get("trajectory_type", -1)
    ret, st = bot.robot.rm_get_current_arm_state()
    if ret == 0:
        diag["err"] = st.get("err", {})
    return diag


def init_velocity_canfd(robot, *, vb: VelocityBurstConfig, dt_ms: float) -> None:
    ret = robot.rm_set_movev_canfd_init(vb.avoid_singularity, vb.frame_type, int(dt_ms))
    if ret != 0:
        raise RuntimeError(f"rm_set_movev_canfd_init failed: {ret}")


def settle_movev_after_init(
    robot,
    *,
    vb: VelocityBurstConfig,
    dt_ms: float,
    next_tick: float | None = None,
    n_frames: int = 10,
) -> float:
    """Zero velocity hold after rm_set_movev_canfd_init — cuts mode-switch jerk before burst."""
    from rm75_control.motion.canfd import send_velocity_canfd

    dt_s = dt_ms / 1000.0
    if next_tick is None:
        next_tick = time.monotonic()
    zero = [0.0] * 6
    for _ in range(n_frames):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
        next_tick += dt_s
        send_velocity_canfd(
            robot, zero,
            follow=vb.follow, trajectory_mode=vb.trajectory_mode, radio=vb.radio,
        )
    return next_tick


def settle_movev_stream(
    robot,
    *,
    dt_ms: float,
    follow: bool,
    trajectory_mode: int,
    radio: int,
    next_tick: float | None = None,
    n_frames: int = 10,
) -> float:
    """Same as settle_movev_after_init for test paths without VelocityBurstConfig."""
    from rm75_control.motion.canfd import send_velocity_canfd

    dt_s = dt_ms / 1000.0
    if next_tick is None:
        next_tick = time.monotonic()
    zero = [0.0] * 6
    for _ in range(n_frames):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
        next_tick += dt_s
        send_velocity_canfd(
            robot, zero,
            follow=follow, trajectory_mode=trajectory_mode, radio=radio,
        )
    return next_tick


def begin_pose_d_vel_burst(
    bot,
    *,
    vb: VelocityBurstConfig,
    dt_ms: float,
    next_tick: float | None = None,
) -> float:
    """prepare → init → zero-velocity settle; same handoff as test_pose_d_vel_burst.py."""
    prepare_movev_session(bot)
    init_velocity_canfd(bot.robot, vb=vb, dt_ms=dt_ms)
    return settle_movev_after_init(
        bot.robot, vb=vb, dt_ms=dt_ms,
        next_tick=next_tick if next_tick is not None else time.monotonic(),
    )


def ramp_down_velocity(
    robot,
    start_vel: np.ndarray,
    *,
    vb: VelocityBurstConfig,
    dt_ms: float,
    next_tick: float | None = None,
) -> float:
    from rm75_control.motion.canfd import send_velocity_canfd

    start_vel = np.asarray(start_vel, dtype=float)
    ramp_s = vb.ramp_down_s
    dt_s = dt_ms / 1000.0
    if ramp_s <= 0 or float(np.max(np.abs(start_vel))) < 1e-9:
        send_velocity_canfd(
            robot, [0.0] * 6,
            follow=vb.follow, trajectory_mode=vb.trajectory_mode, radio=vb.radio,
        )
        return next_tick if next_tick is not None else time.monotonic()

    n = max(2, int(ramp_s / dt_s) + 1)
    if next_tick is None:
        next_tick = time.monotonic()
    for i in range(n):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
        next_tick += dt_s
        scale = 0.5 * (1.0 + math.cos(math.pi * i / (n - 1)))
        send_velocity_canfd(
            robot, (start_vel * scale).tolist(),
            follow=vb.follow, trajectory_mode=vb.trajectory_mode, radio=vb.radio,
        )
    for _ in range(3):
        send_velocity_canfd(
            robot, [0.0] * 6,
            follow=vb.follow, trajectory_mode=vb.trajectory_mode, radio=vb.radio,
        )
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
        next_tick += dt_s
    return next_tick


@dataclass(frozen=True)
class CartesianExcitation:
    amp_mm: np.ndarray
    amp_rot_deg: np.ndarray
    freqs_hz: list[list[float]]
    slot: str

    @classmethod
    def from_config(cls, cart: CartesianConfig, scale: float, slot: str) -> "CartesianExcitation":
        amp_rot = cart.amp_rot_deg_slots.get(slot, cart.amp_rot_deg) * scale
        return cls(
            amp_mm=cart.amp_mm * scale,
            amp_rot_deg=amp_rot,
            freqs_hz=cart.freqs_hz,
            slot=slot,
        )

    def delta_pose(self, t_s: float) -> np.ndarray:
        delta = np.zeros(6, dtype=float)
        for j, (amp_mm, amp_deg) in enumerate(
            zip(self.amp_mm, self.amp_rot_deg)
        ):
            for k, (f_lin, f_rot) in enumerate(
                zip(self.freqs_hz[j % len(self.freqs_hz)],
                    self.freqs_hz[(j + 1) % len(self.freqs_hz)])
            ):
                ph = j * 0.7 + k * 1.1
                delta[j] += amp_mm / 1000.0 * math.sin(2.0 * math.pi * f_lin * t_s + ph)
                delta[3 + j % 3] += amp_deg * DEG2RAD * math.sin(
                    2.0 * math.pi * f_rot * t_s + ph + 0.4
                )
        return delta


def joint_cmd(
    t_s: float,
    q0: np.ndarray,
    pd: PoseDConfig,
    scale: float,
) -> np.ndarray:
    q = q0.copy()
    for j in range(min(7, len(pd.joint_amp_deg))):
        freqs = pd.joint_freqs_hz[j % len(pd.joint_freqs_hz)]
        amp = pd.joint_amp_deg[j] * scale
        max_d = pd.joint_max_delta_deg[j]
        delta = 0.0
        for k, f in enumerate(freqs):
            ph = j * 0.9 + k * 1.3
            delta += amp * math.sin(2.0 * math.pi * f * t_s + ph)
        delta = max(-max_d, min(max_d, delta))
        q[j] += delta
    return q


def clamp_delta(
    delta: np.ndarray,
    *,
    max_mm: float,
    max_rot_deg: float,
) -> np.ndarray:
    out = delta.copy()
    norm_pos = float(np.linalg.norm(out[:3])) * 1000.0
    if norm_pos > max_mm:
        out[:3] *= max_mm / norm_pos / 1000.0
    for j in range(3):
        d_deg = abs(out[3 + j]) * 180.0 / math.pi
        if d_deg > max_rot_deg:
            out[3 + j] *= max_rot_deg / d_deg
    return out


def preview_pose_d(q0: np.ndarray, pd: PoseDConfig, *, scale: float) -> dict:
    dt_s = 0.01
    ts_j = np.linspace(0, pd.joint_duration_s, int(pd.joint_duration_s / dt_s) + 1)
    qs = np.array([joint_cmd(t, q0, pd, scale) for t in ts_j])
    vb = pd.velocity_burst
    ts_b = np.linspace(0, pd.burst_duration_s, int(pd.burst_duration_s / dt_s) + 1)
    vels = np.array([vel_burst_cmd(t, vb, scale=scale)[0] for t in ts_b])
    omega_peak = float(np.max(np.abs(vels[:, 3:6])) / DEG2RAD)
    return {
        "joint_max_deg": np.max(np.abs(qs - q0), axis=0).tolist(),
        "j7_max_deg": float(np.max(np.abs(qs[:, 6] - q0[6]))),
        "burst_omega_deg_s_peak": omega_peak,
        "burst_profile": (
            f"{vb.profile} movev frame={vb.frame_type} amp={vb.amp_deg_s}°/s "
            f"order={list(vb.axis_order)} traj={vb.trajectory_mode} radio={vb.radio}"
        ),
    }
```

### `rm75_control/force/compensation/progress.py`

```python
"""Terminal progress bar (no extra dependencies)."""

from __future__ import annotations

import sys


def stage_progress(label: str, step: int, total: int, *, width: int = 36) -> None:
    total = max(int(total), 1)
    step = min(max(int(step), 0), total)
    frac = step / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    pct = int(100 * frac)
    sys.stdout.write(f"\r  {label} [{bar}] {pct:3d}%")
    sys.stdout.flush()
    if step >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def finish_progress() -> None:
    sys.stdout.write("\n")
    sys.stdout.flush()
```

### `rm75_control/force/compensation/collection.py`

```python
"""
Multi-pose force-ID data collection: A → B → C → D → return A.

  source env.sh
  python -m rm75_control.force.compensation.collection
  python tmp/force_compensation/force_calibrate.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import excitation as ex
from .id_config import ForceIdConfig, load_config
from .paths import CONFIG_ID, CONFIG_ROBOT, npz_for_slot
from .progress import stage_progress


def load_slot(cfg: ForceIdConfig, slot: str) -> tuple[np.ndarray, np.ndarray, dict]:
    data = ex.load_poses_yaml(cfg.poses_yaml)
    rec = ex.get_slot_record(data, slot)
    if rec is None:
        raise SystemExit(f"Pose slot '{slot}' missing in {cfg.poses_yaml}")
    return (
        np.asarray(rec["q_deg"], dtype=float),
        np.asarray(rec["pose_base"], dtype=float),
        rec,
    )


def move_j(robot, q_deg: np.ndarray, *, speed: int) -> None:
    ret = robot.rm_movej(q_deg.tolist(), speed, 0, 0, 1)
    if ret != 0:
        raise RuntimeError(f"rm_movej failed: {ret}")


def wait_settle(robot, target_q: np.ndarray, *, timeout_s: float) -> tuple[np.ndarray, np.ndarray]:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        ret, st = robot.rm_get_current_arm_state()
        if ret == 0:
            q = np.asarray(st["joint"][:7], dtype=float)
            if float(np.max(np.abs(q - target_q))) < 0.5:
                time.sleep(0.5)
                return np.asarray(st["pose"][:6], dtype=float), q
        time.sleep(0.1)
    ret, st = robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError("get state failed after movej")
    return np.asarray(st["pose"][:6], dtype=float), np.asarray(st["joint"][:7], dtype=float)


def run_cartesian(bot, cfg: ForceIdConfig, slot: str) -> Path:
    from rm75_control.motion.canfd import send_pose_canfd

    c = cfg.collect
    cart = c.cartesian
    max_deg = cart.max_deg_for_slot(slot)
    dt_s = c.dt_ms / 1000.0
    duration = cart.duration_s
    out = npz_for_slot(slot)
    exc = ex.CartesianExcitation.from_config(cart, c.scale, slot)

    ret, state = bot.robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"get state failed: {ret}")
    pose0 = np.asarray(state["pose"][:6], dtype=float)
    q0 = np.asarray(state["joint"][:7], dtype=float)

    n_cmd = int(duration / dt_s) + 1
    n_log = (n_cmd + c.log_every - 1) // c.log_every
    t_log = np.zeros(n_log)
    pose_log = np.zeros((n_log, 6))
    q_log = np.zeros((n_log, 7))
    f_log = np.zeros((n_log, 6))
    delta_log = np.zeros((n_log, 6))

    print(f"\n  {slot}", flush=True)
    next_tick = time.monotonic()
    log_i = 0
    try:
        for i in range(n_cmd):
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += dt_s
            t_cmd = i * dt_s
            stage_progress(slot, i + 1, n_cmd)
            ramp = min(1.0, t_cmd / c.warmup_s) if c.warmup_s > 0 else 1.0
            delta = ex.clamp_delta(
                exc.delta_pose(t_cmd) * ramp,
                max_mm=cart.max_delta_mm,
                max_rot_deg=max_deg,
            )
            send_pose_canfd(
                bot.robot, (pose0 + delta).tolist(),
                follow=c.follow, trajectory_mode=0, radio=0,
            )
            if i % c.log_every == 0:
                ret_s, st = bot.robot.rm_get_current_arm_state()
                ret_f, fd = bot.robot.rm_get_force_data()
                if ret_s == 0:
                    pose_log[log_i] = st["pose"][:6]
                    q_log[log_i] = st["joint"][:7]
                if ret_f == 0:
                    f_log[log_i] = np.asarray(fd["force_data"][:6], dtype=float)
                delta_log[log_i] = delta
                t_log[log_i] = t_cmd
                log_i += 1
    finally:
        bot.stop_all()
        try:
            send_pose_canfd(bot.robot, pose0.tolist(), follow=c.follow, trajectory_mode=0, radio=0)
        except Exception:
            pass

    if out.exists():
        out.unlink()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        t=t_log[:log_i], pose=pose_log[:log_i], q_deg=q_log[:log_i],
        force_raw=f_log[:log_i], delta_pose=delta_log[:log_i],
        pose0=pose0, q0_deg=q0, pose_slot=slot, preset="cartesian",
        scale=c.scale, max_delta_mm=cart.max_delta_mm, max_delta_deg=max_deg,
        dt_ms=c.dt_ms, log_every=c.log_every, method="cartesian",
    )
    return out


def run_pose_d(bot, cfg: ForceIdConfig) -> Path:
    from rm75_control.motion.canfd import send_velocity_canfd

    c = cfg.collect
    pd = c.pose_d
    vb = pd.velocity_burst
    dt_s = c.dt_ms / 1000.0
    total_s = pd.joint_duration_s + pd.burst_duration_s
    out = npz_for_slot("d")

    ret, state = bot.robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"get state failed: {ret}")
    pose0 = np.asarray(state["pose"][:6], dtype=float)
    q0 = np.asarray(state["joint"][:7], dtype=float)

    n_total = int(total_s / dt_s) + 1
    n_log = (n_total + c.log_every - 1) // c.log_every
    t_log = np.zeros(n_log)
    pose_log = np.zeros((n_log, 6))
    q_log = np.zeros((n_log, 7))
    f_log = np.zeros((n_log, 6))
    phase_log = np.zeros(n_log, dtype=np.int8)

    print("\n  d (joint + pose_d_vel_burst)", flush=True)
    next_tick = time.monotonic()
    log_i = 0
    movev_ready = False
    ramped_down = False
    last_vel = np.zeros(6, dtype=float)
    burst_pose0 = pose0.copy()

    try:
        for i in range(n_total):
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += dt_s
            t_cmd = i * dt_s
            stage_progress("d", i + 1, n_total)
            ramp = min(1.0, t_cmd / c.warmup_s) if c.warmup_s > 0 else 1.0

            if pd.joint_duration_s > 0 and t_cmd < pd.joint_duration_s:
                q_cmd = ex.joint_cmd(t_cmd, q0, pd, c.scale * ramp)
                ret = bot.robot.rm_movej_canfd(q_cmd.tolist(), False, 0, 0, 0)
                phase = 0
            else:
                if not movev_ready:
                    if pd.joint_duration_s > 0:
                        print("  resync pose d before burst…", flush=True)
                        # Joint movej_canfd must stop before planned movej + movev init.
                        bot.stop_all()
                        time.sleep(0.5)
                        move_j(bot.robot, q0, speed=c.move_speed)
                        burst_pose0, _ = wait_settle(
                            bot.robot, q0, timeout_s=c.settle_timeout_s,
                        )
                    else:
                        burst_pose0 = pose0.copy()
                    next_tick = ex.begin_pose_d_vel_burst(
                        bot, vb=vb, dt_ms=c.dt_ms,
                    )
                    movev_ready = True
                t_burst = t_cmd - pd.joint_duration_s
                vel_cmd, _ = ex.vel_burst_cmd(t_burst, vb, scale=c.scale)
                last_vel = vel_cmd
                send_velocity_canfd(
                    bot.robot, vel_cmd.tolist(),
                    follow=vb.follow,
                    trajectory_mode=vb.trajectory_mode,
                    radio=vb.radio,
                )
                phase = 1
                ret = 0
            if ret != 0:
                raise RuntimeError(f"pose d command failed: {ret}")

            if i % c.log_every == 0:
                ret_s, st = bot.robot.rm_get_current_arm_state()
                ret_f, fd = bot.robot.rm_get_force_data()
                if ret_s == 0:
                    pose_log[log_i] = st["pose"][:6]
                    q_log[log_i] = st["joint"][:7]
                if ret_f == 0:
                    f_log[log_i] = np.asarray(fd["force_data"][:6], dtype=float)
                t_log[log_i] = t_cmd
                phase_log[log_i] = phase
                log_i += 1

        if movev_ready:
            ex.ramp_down_velocity(
                bot.robot, last_vel, vb=vb, dt_ms=c.dt_ms, next_tick=next_tick,
            )
            ramped_down = True

    finally:
        if movev_ready and not ramped_down:
            try:
                ex.ramp_down_velocity(bot.robot, last_vel, vb=vb, dt_ms=c.dt_ms)
            except Exception:
                pass
        if movev_ready:
            try:
                bot.robot.rm_set_arm_slow_stop()
            except Exception:
                pass
        else:
            bot.stop_all()
        try:
            bot.robot.rm_movej_canfd(q0.tolist(), False, 0, 0, 0)
        except Exception:
            pass

    burst_pose0_save = burst_pose0.copy()
    if log_i > 0 and np.any(phase_log[:log_i] == 1) and pd.joint_duration_s <= 0:
        burst_pose0_save = pose_log[:log_i][phase_log[:log_i] == 1][0].copy()

    if out.exists():
        out.unlink()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        t=t_log[:log_i], pose=pose_log[:log_i], q_deg=q_log[:log_i],
        force_raw=f_log[:log_i], phase=phase_log[:log_i],
        pose0=pose0, pose_burst0=burst_pose0_save, q0_deg=q0, pose_slot="d",
        preset="pose_d_vel_burst", scale=c.scale,
        joint_s=pd.joint_duration_s, burst_s=pd.burst_duration_s,
        dt_ms=c.dt_ms, log_every=c.log_every, method="pose_d_vel_burst",
        velocity_burst_profile=vb.profile,
    )
    return out


def slot_kind(slot: str) -> str:
    return "pose_d_vel_burst" if slot == "d" else "cartesian"


def dry_run(cfg: ForceIdConfig) -> None:
    seq = cfg.collect.sequence
    print(f"Collect {' → '.join(seq)} → {cfg.collect.return_home}")
    for slot in seq:
        _, _, rec = load_slot(cfg, slot)
        line = f"  {slot} [{slot_kind(slot)}]: {rec.get('label', f'pose_{slot}')}"
        if slot == "d":
            vb = cfg.collect.pose_d.velocity_burst
            line += (
                f" | burst={vb.profile} {vb.amp_deg_s}°/s frame={vb.frame_type} "
                f"order={list(vb.axis_order)} ramp_down={vb.ramp_down_s}s"
            )
        print(line)


def save_current_pose(cfg: ForceIdConfig, slot: str, label: str | None) -> None:
    from rm75_control import RobotSession

    with RobotSession(config=CONFIG_ROBOT) as bot:
        ret, st = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            raise SystemExit(f"get state failed: {ret}")
        pose = np.asarray(st["pose"][:6], dtype=float)
        q = np.asarray(st["joint"][:7], dtype=float)
    ex.save_pose_slot(cfg.poses_yaml, slot, pose, q, label)
    print(f"Saved pose '{slot}' → {cfg.poses_yaml}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A→B→C→D→A force-ID collection")
    parser.add_argument("--config", type=Path, default=CONFIG_ID)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-pose", type=str, default=None, metavar="SLOT")
    parser.add_argument("--pose-label", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.save_pose:
        save_current_pose(cfg, args.save_pose, args.pose_label)
        return 0
    if args.dry_run:
        dry_run(cfg)
        return 0

    c = cfg.collect
    seq = c.sequence
    slots = {s: load_slot(cfg, s) for s in set(seq) | {c.return_home}}
    print(f"Collect {' → '.join(seq)} → {c.return_home}")
    for s in seq:
        line = f"  {s} [{slot_kind(s)}]: {slots[s][2].get('label', f'pose_{s}')}"
        if s == "d":
            vb = c.pose_d.velocity_burst
            line += (
                f" | burst={vb.profile} {vb.amp_deg_s}°/s frame={vb.frame_type} "
                f"order={list(vb.axis_order)}"
            )
        print(line)

    from rm75_control import RobotSession

    with RobotSession(config=CONFIG_ROBOT) as bot:
        saved = []
        for slot in seq:
            q_tgt, _, rec = slots[slot]
            print(f"\nMove {slot}")
            move_j(bot.robot, q_tgt, speed=c.move_speed)
            wait_settle(bot.robot, q_tgt, timeout_s=c.settle_timeout_s)
            if slot == "d":
                saved.append(run_pose_d(bot, cfg))
            else:
                saved.append(run_cartesian(bot, cfg, slot))

        home = c.return_home
        q_h, _, _ = slots[home]
        print(f"\nReturn {home}")
        move_j(bot.robot, q_h, speed=c.move_speed)
        wait_settle(bot.robot, q_h, timeout_s=c.settle_timeout_s)
        print("\nCollection done:")
        for p in saved:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `rm75_control/force/compensation/regressor.py`

```python
"""Newton-Euler regressor: kinematics + W matrix for force compensation ID."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rsc

PHI_NAMES = [
    "m",
    "mc_x",
    "mc_y",
    "mc_z",
    "Ixx",
    "Iyy",
    "Izz",
    "Ixy",
    "Ixz",
    "Iyz",
    "Fx0",
    "Fy0",
    "Fz0",
    "Mx0",
    "My0",
    "Mz0",
]


@dataclass(frozen=True)
class FrameConfig:
    force_sign: tuple[int, ...]
    euler_order: str
    offset_rad: tuple[float, float, float]
    origin_in_link7_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity_base: tuple[float, float, float] = (0.0, 0.0, -9.80665)

    def label(self) -> str:
        fs = ",".join(str(int(s)) for s in self.force_sign)
        off = "0" if self.offset_rad == (0.0, 0.0, 0.0) else ",".join(
            f"{v:.2f}" for v in self.offset_rad
        )
        return f"sign=[{fs}] order={self.euler_order} off=[{off}]"

    @classmethod
    def from_yaml(cls, path: Path) -> FrameConfig:
        data = yaml.safe_load(path.read_text())
        return cls(
            force_sign=tuple(int(x) for x in data["force_sign"]),
            euler_order=str(data["euler_order"]),
            offset_rad=tuple(float(x) for x in data["sensor_offset_euler_xyz_rad"]),
            origin_in_link7_m=tuple(
                float(x) for x in data.get("sensor_origin_in_link7_m", [0.0, 0.0, 0.0])
            ),
            gravity_base=tuple(float(x) for x in data["gravity_base"]),
        )


def com_from_phi(phi: np.ndarray, cfg: FrameConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Center of mass position (m) from identified phi.

    mc is the first mass moment in the **sensor** frame: mc = m * r_com_sensor.
    link7 frame: R_link7_sensor @ r_com_sensor + sensor origin in link7,
    with R_link7_sensor = R_off from sensor_offset_euler (same as regressor).
    """
    m = float(phi[0])
    if m <= 1e-9:
        z = np.zeros(3, dtype=float)
        return z, z
    r_sensor = np.asarray(phi[1:4], dtype=float) / m
    if cfg.offset_rad != (0.0, 0.0, 0.0):
        r_off = Rsc.from_euler("xyz", cfg.offset_rad, degrees=False).as_matrix()
        r_link7 = r_off @ r_sensor
    else:
        r_link7 = r_sensor.copy()
    r_link7 = r_link7 + np.asarray(cfg.origin_in_link7_m, dtype=float)
    return r_sensor, r_link7


def com_dict_mm(r_m: np.ndarray) -> dict[str, float]:
    r_mm = np.asarray(r_m, dtype=float) * 1000.0
    return {
        "Cx": float(r_mm[0]),
        "Cy": float(r_mm[1]),
        "Cz": float(r_mm[2]),
    }


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def inertia_op(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array(
        [[wx, 0, 0, wy, wz, 0], [0, wy, 0, wx, 0, wz], [0, 0, wz, 0, wx, wy]]
    )


def R_base_sensor(pose6: np.ndarray, cfg: FrameConfig) -> np.ndarray:
    R = Rsc.from_euler(cfg.euler_order, pose6[3:6], degrees=False).as_matrix()
    if cfg.offset_rad != (0.0, 0.0, 0.0):
        R_off = Rsc.from_euler("xyz", cfg.offset_rad, degrees=False).as_matrix()
        R = R @ R_off
    return R


def apply_sign(raw6: np.ndarray, sign: tuple[int, ...]) -> np.ndarray:
    return raw6 * np.array(sign, dtype=float)


def filtfilt_cols(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    if len(x) < 30:
        return x
    b, a = butter(2, min(fc / (0.5 * fs), 0.99), btype="low")
    return filtfilt(b, a, x, axis=0)


def kinematics_sensor(
    pose: np.ndarray, t: np.ndarray, cfg: FrameConfig, fc: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fs = 1.0 / np.mean(np.diff(t))
    euler = pose[:, 3:6].copy()
    for j in range(3):
        euler[:, j] = np.unwrap(euler[:, j])

    p_f = filtfilt_cols(pose[:, :3], fs, fc)
    v_b = np.gradient(p_f, t, axis=0)
    a_b = np.gradient(filtfilt_cols(v_b, fs, fc * 0.8), t, axis=0)

    n = len(t)
    omega_s = np.zeros((n, 3))
    a_s = np.zeros((n, 3))
    g_s = np.zeros((n, 3))
    g_base = np.asarray(cfg.gravity_base, dtype=float)

    for i in range(n):
        p6 = np.concatenate([pose[i, :3], euler[i]])
        R = R_base_sensor(p6, cfg)
        if i == 0:
            R1 = R_base_sensor(np.concatenate([pose[1, :3], euler[1]]), cfg)
            dR = (R1 - R) / max(t[1] - t[0], 1e-6)
        elif i == n - 1:
            R0 = R_base_sensor(np.concatenate([pose[i - 1, :3], euler[i - 1]]), cfg)
            dR = (R - R0) / max(t[-1] - t[-2], 1e-6)
        else:
            Rp = R_base_sensor(np.concatenate([pose[i + 1, :3], euler[i + 1]]), cfg)
            Rm = R_base_sensor(np.concatenate([pose[i - 1, :3], euler[i - 1]]), cfg)
            dR = (Rp - Rm) / max(t[i + 1] - t[i - 1], 1e-6)
        sk = dR @ R.T
        w = np.array([sk[2, 1] - sk[1, 2], sk[0, 2] - sk[2, 0], sk[1, 0] - sk[0, 1]]) / 2
        omega_s[i] = R.T @ w
        a_s[i] = R.T @ a_b[i]
        g_s[i] = R.T @ g_base

    omega_s = filtfilt_cols(omega_s, fs, fc * 0.8)
    alpha_s = np.gradient(filtfilt_cols(omega_s, fs, fc * 0.6), t, axis=0)
    a_s = filtfilt_cols(a_s, fs, fc * 0.8)
    return omega_s, alpha_s, a_s, g_s


def regressor_row(
    a_s: np.ndarray,
    g_s: np.ndarray,
    omega_s: np.ndarray,
    alpha_s: np.ndarray,
    *,
    use_inertia: bool,
) -> np.ndarray:
    aeq = a_s - g_s
    w, al = omega_s, alpha_s
    sw, sa = skew(w), skew(al)
    W = np.zeros((6, 16))
    W[0:3, 0] = aeq
    W[0:3, 1:4] = sa + sw @ sw
    W[3:6, 1:4] = -skew(aeq)
    if use_inertia:
        W[3:6, 4:10] = inertia_op(al) + sw @ inertia_op(w)
    W[:, 10:16] = np.eye(6)
    return W


def build_dataset(
    pose: np.ndarray,
    force_raw: np.ndarray,
    t: np.ndarray,
    cfg: FrameConfig,
    *,
    fc: float,
    use_inertia: bool,
) -> tuple[np.ndarray, np.ndarray]:
    omega_s, alpha_s, a_s, g_s = kinematics_sensor(pose, t, cfg, fc)
    fs = 1.0 / np.mean(np.diff(t))
    f = filtfilt_cols(apply_sign(force_raw, cfg.force_sign), fs, fc)
    rows, Y = [], []
    for i in range(len(t)):
        rows.append(
            regressor_row(a_s[i], g_s[i], omega_s[i], alpha_s[i], use_inertia=use_inertia)
        )
        Y.append(f[i])
    return np.vstack(rows), np.concatenate(Y)
```

### `rm75_control/force/compensation/identification.py`

```python
"""
Merge multi-pose ID logs → staged OLS φ.

Called by force_calibrate.py; not intended as a top-level entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from . import regressor as fid
from .id_config import load_config
from .paths import CONFIG_ID

PHI_NAMES = [
    "m", "mc_x", "mc_y", "mc_z",
    "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz",
    "Fx0", "Fy0", "Fz0", "Mx0", "My0", "Mz0",
]

COLS10 = [0, 1, 2, 3] + list(range(10, 16))
COLS_I = list(range(4, 10))
COLS16 = list(range(16))
I_DIAG = [4, 5, 6]


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    slot = str(d.get("pose_slot", path.stem.split("_")[-1]))
    t = d["t"]
    phase = np.asarray(d["phase"], dtype=np.int8) if "phase" in d.files else np.zeros(len(t))
    return d["pose"], d["force_raw"], t, slot, phase


def sample_row_mask(n_samples: int, sample_mask: np.ndarray) -> np.ndarray:
    """Expand per-time-sample mask (N,) to regressor rows (6N,)."""
    return np.repeat(sample_mask, 6)


def build_merged(
    paths: list[Path],
    cfg: fid.FrameConfig,
    fc: float,
    *,
    use_inertia: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    W_parts, Y_parts, tags = [], [], []
    burst_parts, alpha_parts = [], []
    for p in paths:
        pose, force, t, slot, phase = load_npz(p)
        W, Y = fid.build_dataset(pose, force, t, cfg, fc=fc, use_inertia=use_inertia)
        omega_s, alpha_s, _, _ = fid.kinematics_sensor(pose, t, cfg, fc)
        alpha_norm = np.linalg.norm(alpha_s, axis=1)
        burst = phase == 1
        W_parts.append(W)
        Y_parts.append(Y)
        tags.extend([slot] * len(t))
        burst_parts.append(burst)
        alpha_parts.append(alpha_norm)
    burst_sample = np.concatenate(burst_parts)
    alpha_sample = np.concatenate(alpha_parts)
    return (
        np.vstack(W_parts),
        np.concatenate(Y_parts),
        tags,
        sample_row_mask(len(burst_sample), burst_sample),
        sample_row_mask(len(alpha_sample), alpha_sample),
    )


def fit_cols(W: np.ndarray, Y: np.ndarray, cols: list[int]) -> tuple[np.ndarray, float]:
    phi, *_ = np.linalg.lstsq(W[:, cols], Y, rcond=None)
    rms = float(np.sqrt(np.mean((Y - W[:, cols] @ phi) ** 2)))
    full = np.zeros(16)
    for j, c in enumerate(cols):
        full[c] = phi[j]
    return full, rms


def eval_phi(W: np.ndarray, Y: np.ndarray, phi: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = np.ones(len(Y), dtype=bool)
    Yh = W @ phi
    e = Y[mask] - Yh[mask]
    e6 = e.reshape(-1, 6)
    return {
        "rms_all": float(np.sqrt(np.mean(e**2))),
        "rms_force": float(np.sqrt(np.mean(e6[:, :3] ** 2))),
        "rms_moment": float(np.sqrt(np.mean(e6[:, 3:] ** 2))),
    }


def constrain_inertia(phi: np.ndarray, *, r_max: float) -> np.ndarray:
    out = phi.copy()
    m = max(float(out[0]), 0.05)
    cap_diag = m * r_max * r_max
    cap_off = 0.35 * cap_diag
    for i in I_DIAG:
        out[i] = float(np.clip(out[i], 0.0, cap_diag))
    for i in COLS_I:
        if i not in I_DIAG:
            out[i] = float(np.clip(out[i], -cap_off, cap_off))
    return out


def fit_i_on_mask(
    W16: np.ndarray,
    Y: np.ndarray,
    phi10: np.ndarray,
    row_mask: np.ndarray,
    *,
    min_rows: int,
    r_max: float,
    moments_only: bool = True,
) -> tuple[np.ndarray, dict]:
    use = row_mask
    if moments_only:
        use = row_mask & (np.arange(len(row_mask)) % 6 >= 3)
    n = int(np.sum(use))
    if n < min_rows:
        return phi10.copy(), {"skipped": True, "n_rows": n}
    Yres = Y[use] - W16[use][:, COLS10] @ phi10[COLS10]
    phi_i, *_ = np.linalg.lstsq(W16[use][:, COLS_I], Yres, rcond=None)
    phi = phi10.copy()
    phi[COLS_I] = phi_i
    phi = constrain_inertia(phi, r_max=r_max)
    stats = eval_phi(W16, Y, phi, row_mask)
    stats["n_rows"] = n
    stats["skipped"] = False
    stats["moments_only"] = moments_only
    return phi, stats


def holdout_by_pose(
    paths: list[Path],
    cfg: fid.FrameConfig,
    fc: float,
    phi: np.ndarray,
) -> dict:
    out = {}
    for p in paths:
        pose, force, t, slot, phase = load_npz(p)
        W, Y = fid.build_dataset(pose, force, t, cfg, fc=fc, use_inertia=True)
        Yhat = W @ phi
        e = Y - Yhat
        e6 = e.reshape(-1, 6)
        entry = {
            "rms_all": float(np.sqrt(np.mean(e**2))),
            "rms_force": float(np.sqrt(np.mean(e6[:, :3] ** 2))),
            "rms_moment": float(np.sqrt(np.mean(e6[:, 3:] ** 2))),
            "per_axis": np.sqrt(np.mean(e6**2, axis=0)).tolist(),
        }
        burst = phase == 1
        if np.any(burst):
            rm = sample_row_mask(len(t), burst)
            eb = e[rm]
            eb6 = eb.reshape(-1, 6)
            entry["burst_rms_force"] = float(np.sqrt(np.mean(eb6[:, :3] ** 2)))
            entry["burst_rms_moment"] = float(np.sqrt(np.mean(eb6[:, 3:] ** 2)))
        out[slot] = entry
    return out


def com_report(phi: np.ndarray, cfg: fid.FrameConfig) -> dict:
    r_sensor, r_link7 = fid.com_from_phi(phi, cfg)
    return {
        "sensor_mm": fid.com_dict_mm(r_sensor),
        "link7_mm": fid.com_dict_mm(r_link7),
    }


def print_summary(
    phi: np.ndarray,
    cfg: fid.FrameConfig,
    *,
    rms_all: float,
    per_pose: dict,
    out_json: Path,
) -> None:
    com = com_report(phi, cfg)
    c_s = com["sensor_mm"]
    c_l = com["link7_mm"]
    print("\nIdentify done")
    print(f"  m     = {phi[0]:+.4f} kg")
    print(f"  mc    = [{phi[1]:+.4f}, {phi[2]:+.4f}, {phi[3]:+.4f}] kg·m")
    print(
        f"  CoM sensor  Cx,Cy,Cz = [{c_s['Cx']:+.2f}, {c_s['Cy']:+.2f}, {c_s['Cz']:+.2f}] mm"
    )
    print(
        f"  CoM link7   Cx,Cy,Cz = [{c_l['Cx']:+.2f}, {c_l['Cy']:+.2f}, {c_l['Cz']:+.2f}] mm"
    )
    print(f"  biasF = [{phi[10]:+.3f}, {phi[11]:+.3f}, {phi[12]:+.3f}] N")
    print(f"  biasM = [{phi[13]:+.4f}, {phi[14]:+.4f}, {phi[15]:+.4f}] N·m")
    print(f"  RMS   = {rms_all:.4f}")
    for slot, st in per_pose.items():
        print(f"  {slot}: F={st['rms_force']:.3f} N  M={st['rms_moment']:.4f} N·m")
    print(f"  {out_json}")


def phi_dict(phi: np.ndarray) -> dict[str, float]:
    return {PHI_NAMES[i]: float(phi[i]) for i in range(16)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge pose logs → staged OLS φ")
    parser.add_argument("--id-config", type=Path, default=CONFIG_ID)
    parser.add_argument("--npz", type=Path, action="append", default=None)
    parser.add_argument("--fc", type=float, default=None)
    args = parser.parse_args(argv)

    id_cfg = load_config(args.id_config)
    fcfg = id_cfg.fit
    paths = args.npz if args.npz else fcfg.npz_paths
    for p in paths:
        if not p.exists():
            print(f"Missing {p}", file=sys.stderr)
            return 1

    cfg = fid.FrameConfig.from_yaml(fcfg.force_sensor)
    fc = args.fc
    if fc is None:
        fc = float(yaml.safe_load(fcfg.force_sensor.read_text()).get("filtfilt_cutoff_hz", 2.5))

    r_max = fcfg.inertia_r_max_m
    holdout_frac = fcfg.holdout_frac
    alpha_pct = fcfg.alpha_percentile
    out_json = fcfg.phi_output

    W10, Y, tags, burst_rows, alpha_rows = build_merged(paths, cfg, fc, use_inertia=False)
    W16, Y16, _, burst_rows, alpha_rows = build_merged(paths, cfg, fc, use_inertia=True)
    assert np.allclose(Y, Y16)

    phi10, rms10 = fit_cols(W10, Y, COLS10)
    phi16, rms16 = fit_cols(W16, Y16, COLS16)

    Yres_all = Y16 - W16[:, COLS10] @ phi10[COLS10]
    phi_i_all, *_ = np.linalg.lstsq(W16[:, COLS_I], Yres_all, rcond=None)
    phi_seq = constrain_inertia(phi10.copy(), r_max=r_max)
    phi_seq[COLS_I] = phi_i_all
    phi_seq = constrain_inertia(phi_seq, r_max=r_max)
    rms_seq = eval_phi(W16, Y16, phi_seq)["rms_all"]

    phi_burst, burst_fit = fit_i_on_mask(
        W16, Y16, phi10, burst_rows, min_rows=fcfg.min_burst_rows, r_max=r_max,
    )
    rms_burst_all = eval_phi(W16, Y16, phi_burst)["rms_all"]
    rms_burst_on_burst = eval_phi(W16, Y16, phi_burst, burst_rows)
    rms10_on_burst = eval_phi(W16, Y16, phi10, burst_rows)

    alpha_vals = np.zeros(len(Y16) // 6)
    idx = 0
    for p in paths:
        pose, force, t, _, _ = load_npz(p)
        _, alpha_s, _, _ = fid.kinematics_sensor(pose, t, cfg, fc)
        alpha_vals[idx : idx + len(t)] = np.linalg.norm(alpha_s, axis=1)
        idx += len(t)
    burst_samples = burst_rows.reshape(-1, 6)[:, 0]
    if np.any(burst_samples):
        thr = float(np.percentile(alpha_vals[burst_samples], alpha_pct))
        high_a_rows = sample_row_mask(
            len(alpha_vals), burst_samples & (alpha_vals >= thr)
        )
        phi_ha, ha_fit = fit_i_on_mask(
            W16, Y16, phi10, high_a_rows, min_rows=fcfg.min_high_alpha_rows, r_max=r_max,
        )
    else:
        phi_ha, ha_fit = phi_burst.copy(), {"skipped": True, "n_rows": 0}

    if burst_fit.get("skipped"):
        phi_rec = phi10.copy()
        rec_label = "phi_10 (no burst data)"
    else:
        phi_rec = phi_burst
        rec_label = "phi_burst (10p + I@burst moments)"

    n = len(Y) // 6
    split = int((1.0 - holdout_frac) * n)
    row_tr = np.repeat(np.arange(n) < split, 6)
    row_te = ~row_tr
    phi16_tr, _ = fit_cols(W16[row_tr], Y16[row_tr], COLS16)
    Yhat_te = W16[row_te] @ phi16_tr
    rms16_te = float(np.sqrt(np.mean((Y16[row_te] - Yhat_te) ** 2)))

    per_pose = holdout_by_pose(paths, cfg, fc, phi_rec)
    rms_rec = eval_phi(W16, Y16, phi_rec)["rms_all"]

    result = {
        "config": cfg.label(),
        "id_config": str(args.id_config),
        "files": [str(p) for p in paths],
        "n_samples": int(n),
        "recommended": rec_label,
        "phi_10": phi_dict(phi10),
        "phi_16": phi_dict(phi16),
        "phi_sequential": phi_dict(phi_seq),
        "phi_burst": phi_dict(phi_burst),
        "phi_recommended": phi_dict(phi_rec),
        "com_recommended": com_report(phi_rec, cfg),
        "rms_10": rms10,
        "rms_16": rms16,
        "rms_16_test": rms16_te,
        "rms_burst_all": rms_burst_all,
        "burst_i_fit": burst_fit,
        "per_pose_residual": per_pose,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    print_summary(phi_rec, cfg, rms_all=rms_rec, per_pose=per_pose, out_json=out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## 四、速度下发与运动

### `rm75_control/motion/__init__.py`

```python
"""Planned motion and CANFD streaming."""

from rm75_control.motion.canfd import send_pose_canfd, send_velocity_canfd

__all__ = ["send_pose_canfd", "send_velocity_canfd"]
```

### `rm75_control/motion/canfd.py`

```python
"""CANFD pose and velocity streaming via rm_movep_canfd / rm_movev_canfd."""

from __future__ import annotations

from typing import Protocol, Sequence

from rm75_control.core.exceptions import MotionError

Pose6 = Sequence[float]
Vel6 = Sequence[float]

MAX_LINEAR_V_M_S = 0.25
MAX_ANGULAR_V_RAD_S = 0.6


class PoseCanfdClient(Protocol):
    def rm_movep_canfd(
        self,
        pose: list[float],
        follow: bool,
        trajectory_mode: int = 0,
        radio: int = 0,
    ) -> int:
        ...


def send_pose_canfd(
    robot: PoseCanfdClient,
    pose: Pose6,
    *,
    follow: bool = True,
    trajectory_mode: int = 0,
    radio: int = 0,
) -> None:
    if len(pose) not in (6, 7):
        raise ValueError(f"pose must have 6 (euler) or 7 (quat) elements, got {len(pose)}")

    ret = robot.rm_movep_canfd(
        list(pose),
        follow,
        trajectory_mode,
        radio,
    )
    if ret != 0:
        raise MotionError(f"rm_movep_canfd failed with code {ret}")


class VelocityCanfdClient(Protocol):
    def rm_movev_canfd(
        self,
        cartesian_velocity: list[float],
        follow: bool,
        trajectory_mode: int = 0,
        radio: int = 0,
    ) -> int:
        ...


def clamp_cartesian_velocity(vel: Vel6) -> list[float]:
    out = list(vel)
    for i in range(3):
        out[i] = max(-MAX_LINEAR_V_M_S, min(MAX_LINEAR_V_M_S, out[i]))
    for i in range(3, 6):
        out[i] = max(-MAX_ANGULAR_V_RAD_S, min(MAX_ANGULAR_V_RAD_S, out[i]))
    return out


def send_velocity_canfd(
    robot: VelocityCanfdClient,
    cartesian_velocity: Vel6,
    *,
    follow: bool = False,
    trajectory_mode: int = 0,
    radio: int = 0,
) -> None:
    ret = robot.rm_movev_canfd(
        clamp_cartesian_velocity(cartesian_velocity),
        follow,
        trajectory_mode,
        radio,
    )
    if ret != 0:
        raise MotionError(f"rm_movev_canfd failed with code {ret}")
```

### `rm75_control/motion/force_position.py`

```python
"""Thin wrapper for rm_force_position_move (native params preserved)."""

from __future__ import annotations

from typing import Protocol, Sequence

from rm75_control.core.exceptions import MotionError


class ForcePositionClient(Protocol):
    def rm_force_position_move(self, param) -> int:
        ...


def send_force_position_move(robot: ForcePositionClient, param) -> None:
    ret = robot.rm_force_position_move(param)
    if ret != 0:
        raise MotionError(f"rm_force_position_move failed with code {ret}")
```

### `rm75_control/motion/ptp.py`

```python
"""Planned point-to-point motion: movej, movel, movej_p."""
```

## 五、Session 与后端

### `rm75_control/core/__init__.py`

```python
"""Session, control modes, shared types."""

from rm75_control.core.types import ControlMode

__all__ = ["ControlMode"]
```

### `rm75_control/core/types.py`

```python
"""Control modes and shared data types."""

from __future__ import annotations

from enum import Enum, auto


class ControlMode(Enum):
    IDLE = auto()
    PTP_PLANNED = auto()
    CARTESIAN_POSE_CANFD = auto()
    CARTESIAN_VEL_CANFD = auto()
    JOINT_CANFD = auto()
    FORCE_SCAN = auto()
    FORCE_PTP = auto()
```

### `rm75_control/core/exceptions.py`

```python
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
```

### `rm75_control/core/feedback.py`

```python
"""Background pose feedback polling (keeps CANFD stream loop non-blocking)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Callable, Sequence

Pose6 = list[float]
ReadPoseFn = Callable[[], Pose6 | None]


@dataclass
class StampedPose:
    pose: Pose6
    t_mono: float


class PoseFeedbackPoller:
    """Poll pose in a daemon thread; control loop reads cached stamp without blocking."""

    def __init__(
        self,
        read_pose: ReadPoseFn,
        *,
        period_s: float = 0.02,
    ) -> None:
        self._read_pose = read_pose
        self._period_s = max(period_s, 0.005)
        self._lock = Lock()
        self._stamp = StampedPose([0.0] * 6, time.monotonic())
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self, initial: Sequence[float]) -> None:
        self._stamp = StampedPose([float(v) for v in initial], time.monotonic())
        self._stop.clear()
        self._thread = Thread(target=self._run, name="pose-feedback", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get(self) -> StampedPose:
        with self._lock:
            return StampedPose(list(self._stamp.pose), self._stamp.t_mono)

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            pose = self._read_pose()
            if pose is not None:
                with self._lock:
                    self._stamp = StampedPose(pose, time.monotonic())
            elapsed = time.monotonic() - t0
            delay = self._period_s - elapsed
            if delay > 0.0:
                self._stop.wait(delay)
```

### `rm75_control/core/session.py`

```python
"""Robot session: connect, mode switching, high-level entry point."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import yaml

from rm75_control.backend.realman import RealManBackend
from rm75_control.control.cartesian_pose import (
    CartesianPoseController,
    CartesianPoseStreamConfig,
)
from rm75_control.core.exceptions import ControlModeError, MotionError
from rm75_control.core.types import ControlMode
from rm75_control.force.scan import ForceScanConfig, ForceScanController

Pose6 = Sequence[float]


class RobotSession:
    """Top-level facade for init -> cartesian path -> reset workflows."""

    def __init__(
        self,
        ip: str | None = None,
        port: int | None = None,
        config: str | Path | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        self._config = self._load_config(config)
        robot_cfg = self._config.get("robot", {})
        self.ip = ip or robot_cfg.get("ip", "192.168.1.18")
        self.port = port or robot_cfg.get("port", 8080)
        self.thread_mode = robot_cfg.get("thread_mode", 2)
        self.dry_run = dry_run
        self.mode = ControlMode.IDLE
        self._backend: RealManBackend | None = None
        self._force_scan: ForceScanController | None = None

    @staticmethod
    def _load_config(config: str | Path | None) -> dict[str, Any]:
        if config is None:
            return {}
        path = Path(config)
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def backend(self) -> RealManBackend:
        if self._backend is None:
            raise RuntimeError("RobotSession is not connected")
        return self._backend

    @property
    def robot(self):
        return self.backend.robot

    def connect(self) -> None:
        print(f"Connecting to {self.ip}:{self.port}...", flush=True)
        self._backend = RealManBackend(
            self.ip,
            self.port,
            thread_mode=self.thread_mode,
        )
        if not self.dry_run:
            self._backend.connect()
        self.mode = ControlMode.IDLE
        print("Connected.", flush=True)

    def disconnect(self) -> None:
        if self._backend is not None and not self.dry_run:
            self.stop_all(hard=False)
            self._backend.disconnect()
        self._backend = None
        self.mode = ControlMode.IDLE

    def stop_motion(self, *, hard: bool = False) -> None:
        if self.dry_run or self._backend is None:
            self.mode = ControlMode.IDLE
            return
        if self.mode == ControlMode.FORCE_SCAN:
            raise ControlModeError(
                "In FORCE_SCAN mode; call stop_force_scan() before stop_motion()"
            )
        ret = (
            self.robot.rm_set_arm_stop()
            if hard
            else self.robot.rm_set_arm_slow_stop()
        )
        if ret != 0:
            raise MotionError(f"stop motion failed with code {ret}")
        self.mode = ControlMode.IDLE

    def stop_force_scan(self) -> None:
        if self._force_scan is not None:
            self._force_scan.stop()
            self._force_scan = None
        self.mode = ControlMode.IDLE

    def stop_all(self, *, hard: bool = False) -> None:
        """Stop force scan (if active) then stop planned/canfd motion."""
        self.stop_force_scan()
        if self._backend is not None and not self.dry_run:
            self.stop_motion(hard=hard)

    def _wait_planning_idle(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            traj = self.robot.rm_get_arm_current_trajectory()
            if traj.get("return_code") == 0 and traj.get("trajectory_type", 0) == 0:
                return True
            time.sleep(0.05)
        return False

    def prepare_for_force_stream(self, *, settle_s: float = 1.0) -> dict[str, Any]:
        """
        Exit planned force (rm_set_force_position) and stale CANFD force modes.

        rm_set_force_position + rm_start_force_position_move must never overlap;
        a failed mix can leave the controller rejecting stream start until recovery.
        """
        diag: dict[str, Any] = {}
        try:
            diag["stop_force_move"] = self.robot.rm_stop_force_position_move()
        except Exception:
            diag["stop_force_move"] = -999
        diag["stop_force"] = self.robot.rm_stop_force_position()
        traj = self.robot.rm_get_arm_current_trajectory()
        diag["trajectory_type"] = traj.get("trajectory_type", -1)
        if traj.get("trajectory_type", 0) != 0:
            diag["slow_stop"] = self.robot.rm_set_arm_slow_stop()
            diag["planning_idle"] = self._wait_planning_idle()
        if settle_s > 0.0:
            time.sleep(settle_s)
        return diag

    def recover_controller(
        self,
        *,
        settle_s: float = 1.0,
        clear_errors: bool = True,
        probe_force_stream: bool = False,
    ) -> dict[str, Any]:
        """
        Full controller cleanup before velocity CANFD (run every session start).

        Clears latched errors, exits force/plan modes, waits for planner idle.
        Optional force-stream probe unsticks rm_set_force_position conflicts.
        """
        diag: dict[str, Any] = {}
        if clear_errors:
            try:
                diag["clear_system_err"] = self.robot.rm_clear_system_err()
            except Exception:
                diag["clear_system_err"] = -999
            time.sleep(0.3)
            ret, st = self.robot.rm_get_current_arm_state()
            if ret == 0:
                err = st.get("err", {})
                diag["system_err"] = list(err.get("err", []))[: int(err.get("err_len", 0))]

        if not self.dry_run and self._backend is not None:
            self.stop_all(hard=False)
        diag.update(self.prepare_for_force_stream(settle_s=0.0))
        try:
            diag["delete_traj"] = self.robot.rm_set_arm_delete_trajectory()
        except Exception:
            diag["delete_traj"] = -999
        diag["slow_stop"] = self.robot.rm_set_arm_slow_stop()
        diag["planning_idle"] = self._wait_planning_idle(timeout_s=8.0)

        if probe_force_stream and not self.dry_run:
            try:
                ret = self.robot.rm_start_force_position_move()
                diag["force_stream_probe"] = ret
                if ret == 0:
                    self.robot.rm_stop_force_position_move()
                    time.sleep(0.3)
            except Exception:
                diag["force_stream_probe"] = -999

        if settle_s > 0.0:
            time.sleep(settle_s)

        traj = self.robot.rm_get_arm_current_trajectory()
        diag["trajectory_type_final"] = traj.get("trajectory_type", -1)
        ret, st = self.robot.rm_get_current_arm_state()
        if ret == 0:
            diag["pose_euler_deg"] = [
                round(float(v) * 180.0 / 3.141592653589793, 3) for v in st["pose"][3:6]
            ]
        self.mode = ControlMode.IDLE
        return diag

    def start_force_scan(self, **overrides: Any) -> ForceScanController:
        self.stop_force_scan()
        if self._backend is not None and not self.dry_run and self.mode != ControlMode.IDLE:
            self.stop_motion(hard=False)
        cfg = ForceScanConfig.from_config(self._config, **overrides)
        self._force_scan = ForceScanController(
            self.robot if not self.dry_run else _DryRunForceClient(),
            cfg,
            dry_run=self.dry_run,
        )
        last_err: MotionError | None = None
        for attempt in range(5):
            if self._backend is not None and not self.dry_run:
                diag = self.prepare_for_force_stream(settle_s=2.0 if attempt else 1.0)
                if attempt:
                    print(f"force stream recover attempt {attempt}: {diag}", flush=True)
            try:
                self._force_scan.start()
                last_err = None
                break
            except MotionError as exc:
                last_err = exc
        if last_err is not None:
            raise MotionError(
                f"{last_err}. Planned force (rm_set_force_position) and stream force "
                f"(rm_start_force_position_move) cannot be mixed. Run "
                f"python /media/camp/EXT_DRIVE/rm75_control/tmp/recover_force_stream.py "
                f"or stop/power-cycle on the teach pendant."
            ) from last_err
        self.mode = ControlMode.FORCE_SCAN
        return self._force_scan

    def _ensure_idle(self) -> None:
        if self.mode == ControlMode.FORCE_SCAN:
            raise ControlModeError(
                "In FORCE_SCAN mode; call stop_force_scan() first"
            )
        if self.mode not in (ControlMode.IDLE, ControlMode.PTP_PLANNED):
            raise ControlModeError(
                f"Cannot start motion while mode={self.mode.name}; call stop_motion() first"
            )

    def move_joints(
        self,
        joint: Sequence[float],
        *,
        velocity_percent: int | None = None,
        block: int = 1,
    ) -> None:
        if self.dry_run:
            self.mode = ControlMode.IDLE
            return

        self._ensure_idle()
        motion = self._config.get("motion", {})
        v = velocity_percent or motion.get("default_velocity_percent", 20)
        self.mode = ControlMode.PTP_PLANNED
        ret = self.robot.rm_movej(list(joint), v, 0, 0, block)
        self.mode = ControlMode.IDLE
        if ret != 0:
            raise MotionError(f"rm_movej failed with code {ret}")

    def move_cartesian_path(
        self,
        waypoints: Sequence[Pose6],
        *,
        use_ruckig: bool | None = None,
        follow: bool | None = None,
        trajectory_mode: int | None = None,
        radio: int | None = None,
        period_ms: float | None = None,
        steps_per_segment: int | None = None,
    ) -> None:
        """Cartesian pose CANFD. Native rm_movep_canfd params preserved; use_ruckig is the only extra switch."""
        self._ensure_idle()
        cfg = CartesianPoseStreamConfig.from_config(
            self._config,
            use_ruckig=use_ruckig,
            follow=follow,
            trajectory_mode=trajectory_mode,
            radio=radio,
            period_ms=period_ms,
            steps_per_segment=steps_per_segment,
        )
        controller = CartesianPoseController(
            self.robot if not self.dry_run else _DryRunCanfdClient(),
            cfg,
            dry_run=self.dry_run,
        )
        start_pose = None
        if not self.dry_run and self._backend is not None:
            try:
                start_pose = self._backend.get_tcp_pose()
            except Exception:
                start_pose = None
        self.mode = ControlMode.CARTESIAN_POSE_CANFD
        try:
            controller.run(waypoints, start_pose=start_pose)
        finally:
            self.mode = ControlMode.IDLE

    def run_init_path_reset(
        self,
        home_joint: Sequence[float],
        path_waypoints: Sequence[Pose6],
        *,
        use_ruckig: bool | None = None,
        follow: bool | None = None,
        trajectory_mode: int | None = None,
        radio: int | None = None,
    ) -> None:
        self.move_joints(home_joint)
        self.move_cartesian_path(
            path_waypoints,
            use_ruckig=use_ruckig,
            follow=follow,
            trajectory_mode=trajectory_mode,
            radio=radio,
        )
        self.stop_motion(hard=False)
        self.move_joints(home_joint)

    def __enter__(self) -> RobotSession:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()


class _DryRunCanfdClient:
    def rm_movep_canfd(self, pose, follow, trajectory_mode=0, radio=0) -> int:
        return 0


class _DryRunForceClient:
    def rm_start_force_position_move(self) -> int:
        return 0

    def rm_stop_force_position_move(self) -> int:
        return 0

    def rm_force_position_move(self, param) -> int:
        return 0
```

### `rm75_control/backend/__init__.py`

```python
"""Robot backend abstractions (Real / Sim)."""
```

### `rm75_control/backend/base.py`

```python
"""Abstract robot backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class RobotBackend(ABC):
    @abstractmethod
    def connect(self) -> None:
        ...

    @abstractmethod
    def disconnect(self) -> None:
        ...

    @abstractmethod
    def get_joint_positions(self) -> Sequence[float]:
        ...

    @abstractmethod
    def get_tcp_pose(self) -> Sequence[float]:
        ...
```

### `rm75_control/backend/realman.py`

```python
"""RealMan RM_API2 backend — thin wrapper over RoboticArm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rm75_control.backend.base import RobotBackend
from rm75_control.core.exceptions import RobotConnectionError

if TYPE_CHECKING:
    from Robotic_Arm.rm_robot_interface import RoboticArm


class RealManBackend(RobotBackend):
    """Maps rm75_control calls to Robotic_Arm.rm_* APIs."""

    def __init__(self, ip: str, port: int, thread_mode: int = 2) -> None:
        self.ip = ip
        self.port = port
        self.thread_mode = thread_mode
        self._robot: RoboticArm | None = None

    @property
    def robot(self) -> RoboticArm:
        if self._robot is None:
            raise RobotConnectionError("Robot is not connected")
        return self._robot

    def connect(self) -> None:
        from Robotic_Arm.rm_ctypes_wrap import rm_thread_mode_e
        from Robotic_Arm.rm_robot_interface import RoboticArm

        self._robot = RoboticArm(rm_thread_mode_e(self.thread_mode))
        handle = self._robot.rm_create_robot_arm(self.ip, self.port)
        if handle.id == -1:
            self._robot = None
            raise RobotConnectionError(
                f"Failed to connect to robot at {self.ip}:{self.port}"
            )

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.rm_delete_robot_arm()
            self._robot = None

    def get_joint_positions(self) -> list[float]:
        ret, state = self.robot.rm_get_current_arm_state()
        if ret != 0:
            raise RobotConnectionError(f"rm_get_current_arm_state failed: {ret}")
        return list(state["joint"])

    def get_tcp_pose(self) -> list[float]:
        ret, state = self.robot.rm_get_current_arm_state()
        if ret != 0:
            raise RobotConnectionError(f"rm_get_current_arm_state failed: {ret}")
        return list(state["pose"])
```

### `rm75_control/backend/sim.py`

```python
"""Simulation backend — rm_algo FK + mock wrench."""

from __future__ import annotations

from rm75_control.backend.base import RobotBackend


class SimBackend(RobotBackend):
    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def get_joint_positions(self) -> list[float]:
        raise NotImplementedError

    def get_tcp_pose(self) -> list[float]:
        raise NotImplementedError
```

## 六、调用关系

### 6.1 总览

```
sin_tool_y_z2n.py
  └─ run_velocity_admittance()          [loop.py]
       ├─ RobotSession                   [session.py → realman.py → RM API]
       ├─ recover_controller()           [session.py]
       ├─ load_slot / move_j             [collection.py → poses.yaml]
       ├─ rm_set_movev_canfd_init        [RM API]
       ├─ TrajectoryGenerator.sample()   [trajectory.py]  → pose_d, vel_ff (base 6D)
       ├─ CompensatedForceObserver       [observer.py → regressor.py + force_id_phi.json]
       │    └─ rm_get_force_data + phi 补偿 → f_ext (tool frame)
       ├─ AdmittanceController.compute_velocity_command()  [controller.py]
       │    └─ fuse_tool_decoupled: base轨迹 + tool-Z力导纳 → v_cmd (tool movev)
       └─ send_velocity_canfd()          [canfd.py → rm_movev_canfd]
```

### 6.2 标定链路（生成 phi，运行前一次性）

```
force_calibrate.py
  └─ identification / collection      [identification.py, collection.py]
       ├─ 多 pose 采集 rm_get_force_data
       ├─ regressor 拟合               [regressor.py]
       └─ 输出 force_id_phi.json       [observer 读取]
```

### 6.3 单周期数据流（scan 阶段）

| 步骤 | 模块 | 输入 | 输出 |
|------|------|------|------|
| 1 | trajectory.py | t_scan, pose0 | pose_d[6], vel_ff[6] base |
| 2 | observer.py | pose, force_raw | f_ext[6] tool（phi 补偿后） |
| 3 | controller.py | pose, pose_d, vel_ff, f_ext, Fz_des | v_cmd[6] tool |
| 4 | canfd.py | v_cmd | rm_movev_canfd(frame_type=0) |

### 6.4 rm75_control/control 文件职责

| 文件 | 职责 |
|------|------|
| velocity_admittance/loop.py | 主循环、auto-start、CANFD 会话 |
| velocity_admittance/controller.py | 力位混合解算 S_f/S_p |
| velocity_admittance/trajectory.py | 6D 轨迹插件 |
| velocity_admittance/observer.py | 外力观测（phi） |
| velocity_admittance/rm_algo.py | end2tool 坐标变换 |
| cartesian_pose.py | Session 位姿 CANFD（本 demo 未直接调用） |
| cartesian_velocity.py | 旧轴速度跟踪（本 demo 未直接调用） |

## 七、环境与依赖

### `env.sh`

```bash
#!/usr/bin/env bash
# Usage: source /media/camp/EXT_DRIVE/rm75_control/env.sh

RM75_ENV="/media/camp/EXT_DRIVE/envs/rm75"

if [ ! -d "${RM75_ENV}/bin" ]; then
  echo "rm75 env not found: ${RM75_ENV}" >&2
  return 1 2>/dev/null || exit 1
fi

# Prefer direct PATH (works even if conda name lookup fails)
export PATH="${RM75_ENV}/bin:${PATH}"

# Optional: also hook conda if available
CONDA_BASE="${CONDA_BASE:-/home/camp/miniconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${RM75_ENV}" 2>/dev/null || true
fi

export RM75_CONTROL_ROOT="/media/camp/EXT_DRIVE/rm75_control"
export RM_API2_PYTHON="/media/camp/EXT_DRIVE/RM_API2/Python"
export PYTHONPATH="${RM75_CONTROL_ROOT}:${RM_API2_PYTHON}:${PYTHONPATH:-}"

echo "rm75 env: $(which python)"
echo "PYTHONPATH includes RM_API2 and rm75_control"
```

### `requirements.txt`

```text
# RealMan SDK: add RM_API2/Python to PYTHONPATH, e.g.
#   export PYTHONPATH=/media/camp/EXT_DRIVE/RM_API2/Python:$PYTHONPATH
numpy
pyyaml
ruckig==0.17.3
```


## 八、YAML 参数附录（全文 + 说明）

以下为力位混合 demo 相关的 **全部 YAML 原文**（便于第三方离线查阅）。第一节已有相同内容，此处集中附于文档末尾。

### 8.0 参数速查（sin_tool_y_z2n.yaml 主 demo）

| 块 | 键 | 含义 |
|----|-----|------|
| **timing** | `dt_ms` | 控制周期 ms，传给 `rm_set_movev_canfd_init` |
| | `feedback_every` | 每 N 周期读一次 pose/force |
| **startup** | `pose_slot` | 启动 `move_j` 到 poses.yaml 的 slot（demo 用 `d`） |
| | `hold_s` | init 后零速等待秒数 |
| | `auto_recover` | 连接后 `recover_controller()` 清模式 |
| | `wait_contact` | true=等 Fz 达标再 scan |
| | `auto_start_under_n` | Fz ≥ F_des - 此值 持续 `auto_start_hold_s` 启动 scan |
| | `approach_ramp_s` | 下压阶段 Fz_des 线性 ramp，避免阶跃 |
| **frames** | `euler_order` | pose 欧拉顺序，须与 force_sensor.yaml 一致 |
| | `control_frame` | `tool`/`base`；demo 为 tool → `frame_type=0` |
| **velocity_canfd** | `frame_type` | 0=TCP movev，1=world movev |
| | `follow` | rm_movev_canfd follow 标志 |
| | `trajectory_mode` / `radio` | 0=透传 |
| **force** | `phi_source` | 读 force_id_phi.json 的键 |
| | `buffer_s` / `min_samples` | phi 补偿滤波缓冲 |
| | `fc_hz` | 低通截止 |
| | `desired_z_n` | tool-Z 目标力 N |
| **trajectory** | `type` | `hold` / `sin_base_y` / `sin_base_y_tool_rz` / `sin_tool_y` |
| | `y_peak_to_peak_cm` | world-Y 峰峰值 cm |
| | `rz_amplitude_deg` | tool+Z 旋转幅度 ±deg |
| | `y_max_vel_cm_s` | Y 方向峰值速度，自动算 period |
| | `soft_start` / `ramp_s` | scan 起始 vy 软启动 |
| **controller** | `force_axes` | tool 系力控轴 mask，典型 `[0,0,1,0,0,0]` |
| | `open_loop` | true=仅 ff；false=ff+kp·(pose_d-pose) |
| | `kp_pos` | 6D 位置 P 增益 base |
| | `k_fp_press/release` | 力导纳 P |
| | `k_fi` / `integral_limit` | 力导纳 I |
| | `max_vz_tool_m_s` | tool-Z 速度限幅 |
| | `max_velocity` / `max_acceleration` | movev 输出限幅 |

### 8.0.1 force_sensor.yaml

| 键 | 含义 |
|----|------|
| `force_sign` | 原始 Fx,Fy,Fz 符号翻转 |
| `euler_order` | 与 arm pose 一致 |
| `gravity_base` | 重力向量 base 系 |
| `filtfilt_cutoff_hz` | 标定/补偿滤波 |

### 8.0.2 force_id.yaml / poses.yaml

| 键 | 含义 |
|----|------|
| `sequence` / `return_home` | 标定遍历 pose 顺序 |
| `collect.move_speed` | move_j 速度 |
| `fit.phi_output` | 输出 phi JSON 文件名 |
| `poses.*.q_deg` | 各 slot 关节角；demo 启动位 `d` |

### 8.x `tmp/Velocity_Admittance/demo/config/sin_tool_y_z2n.yaml`
*demo 力位混合（当前主配置）*

```yaml
# Demo trajectory plugin + tool-Z force hybrid.
# Architecture:
#   trajectory → 6D (pose_d, vel_ff) in base frame  [you may zero tool-Z in vel_ff]
#   controller → S_f on tool force_axes; all other DOFs follow trajectory (S_p = I - S_f)
#
# Run:
#   source env.sh && cd /media/camp/EXT_DRIVE/rm75_control
#   python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py

timing:
  dt_ms: 10.0
  feedback_every: 2

startup:
  pose_slot: d
  settle_frames: 25
  hold_s: 1.0
  auto_recover: true
  wait_contact: true
  auto_start_under_n: 0.5
  auto_start_hold_s: 0.5
  approach_ramp_s: 1.5

frames:
  euler_order: xyz
  control_frame: tool

velocity_canfd:
  frame_type: 0
  avoid_singularity: 0
  follow: true
  trajectory_mode: 0
  radio: 0

force:
  phi_source: phi_recommended
  buffer_s: 2.0
  min_samples: 22
  fc_hz: 4.0
  use_inertia: false
  desired_z_n: 3.0

trajectory:
  type: sin_base_y_tool_rz    # demo plugin; replace with any Trajectory6D
  y_peak_to_peak_cm: 16.0
  rz_amplitude_deg: 12.0
  y_max_vel_cm_s: 3.0
  soft_start: true
  ramp_s: 2.0
  open_loop: true

controller:
  force_axes: [0, 0, 1, 0, 0, 0]   # tool TCP-Z → force; other 5 DOF → trajectory
  open_loop: true
  kp_pos: [0, 0, 0, 0, 0, 0]
  system_delay_s: 0.015
  k_fp_press: 0.045
  k_fp_release: 0.025
  k_fi: 0.001
  integral_limit: 0.015
  max_vz_tool_m_s: 0.15
  approach_vz_tool_m_s: 0.03
  max_velocity: [0.03, 0.10, 0.15, 0.15, 0.15, 0.35]
  max_acceleration: [0.5, 1.0, 0.5, 1.0, 1.0, 1.5]
```

### 8.x `tmp/Velocity_Admittance/config/admittance.yaml`
*通用 admittance 模板（旧字段，供对照）*

```yaml
# Velocity admittance — library: rm75_control.control.velocity_admittance
# Generic: python tmp/Velocity_Admittance/run_admittance.py
# Demo:    python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py

timing:
  dt_ms: 10.0
  feedback_every: 3

startup:
  pose_slot: d

frames:
  euler_order: xyz   # rm_get_current_arm_state pose; match configs/force_sensor.yaml

velocity_canfd:
  frame_type: 1          # 0 tool, 1 work/base — v_cmd output is base frame
  avoid_singularity: 0
  follow: true
  trajectory_mode: 0     # 0 passthrough — smoothing in controller only
  radio: 0

force:
  phi_source: phi_recommended
  buffer_s: 4.0
  min_samples: 35
  use_inertia: false     # no virtual mass — PI admittance only (Keemink §5.4)
  desired_z_n: 3.0

trajectory:
  type: sin_tool_y       # hold | sin_tool_y | sin_base_y
  amplitude_mm: 5.0
  period_s: null         # auto from y_max_vel_cm_s if null
  y_max_vel_cm_s: 1.0

controller:
  force_axes: [0, 0, 1, 0, 0, 0]   # 1 = admittance on axis (sensor frame)
  motion_axes: [0, 1, 0, 0, 0, 0]
  lock_orientation: true
  enable_normal_tracking: false
  kp_pos: [2.0, 2.0, 0.0, 1.5, 1.5, 1.5]
  system_delay_s: 0.015
  k_fp_press: 0.015
  k_fp_release: 0.005
  k_fi: 0.008
  integral_limit: 0.05
  k_align: 0.0
  contact_threshold_n: 0.5
  deadband_n: 0.3
  max_vz_tool_m_s: 0.05
  max_velocity: [0.2, 0.2, 0.05, 0.08, 0.08, 0.08]
  max_acceleration: [1.0, 1.0, 0.05, 0.15, 0.15, 0.15]
  release_vz_up_m_s: 0.05
  release_vz_down_m_s: 0.05

monitor:
  enabled: false
  window_s: 25.0
  refresh_hz: 12.0
```

### 8.x `configs/rm75f_default.yaml`
*机器人连接 / CANFD 全局*

```yaml
robot:
  ip: "192.168.1.18"
  port: 8080
  thread_mode: 2  # RM_TRIPLE_MODE_E

timing:
  canfd_period_ms: 10
  force_scan_period_ms: 10

motion:
  default_velocity_percent: 20

  # Cartesian pose CANFD (rm_movep_canfd) — native params + use_ruckig switch
  cartesian_pose:
    use_ruckig: false
    period_ms: 10
    follow: true
    trajectory_mode: 0  # 0 passthrough, 1 curve fit, 2 filter
    radio: 0
    steps_per_segment: 50  # used only when use_ruckig=false
    ruckig:
      max_velocity: [0.05, 0.05, 0.05, 0.3, 0.3, 0.3]
      max_acceleration: [0.5, 0.5, 0.5, 1.5, 1.5, 1.5]
      max_jerk: [2.0, 2.0, 2.0, 5.0, 5.0, 5.0]

force:
  sensor: 1
  coordinate_mode: 1
  default_desired_force: [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
  default_control_mode: [3, 3, 4, 3, 3, 3]
  scan:
    flag: 0  # 0 joint (after IK), 1 pose
    follow: true
    trajectory_mode: 0
    radio: 0
    period_ms: 10
    control_mode: [3, 3, 4, 3, 3, 3]
    desired_force: [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
    limit_vel: [0.05, 0.05, 0.25, 10.0, 10.0, 10.0]

tool:
  name: "gripper"
  payload_kg: 0.982
```

### 8.x `configs/force_sensor.yaml`
*力传感器符号 / 重力 / 滤波*

```yaml
# RealMan RM75 raw force_data — dynamic ID / compensation (verified 2026-06-27)
# Verified force_sign / euler — used by rm75_control.force.compensation.regressor

force_sign: [-1, -1, -1, 1, 1, 1]   # flip Fx,Fy,Fz only; moments unchanged
euler_order: xyz
sensor_offset_euler_xyz_rad: [0.0, 0.0, 0.0]
# Sensor origin position in link7 frame (m). Default 0 = co-located with link7 origin.
sensor_origin_in_link7_m: [0.0, 0.0, 0.0]
gravity_base: [0.0, 0.0, -9.80665]

filtfilt_cutoff_hz: 2.5
identify_mass_bias_only: true

# Verified on force_id_cartesian.npz (601 samples) + live static:
#   m_fit = +0.997 kg,  live m = +1.10 kg,  Fz_raw ≈ +10.5 N
#   hold-out compensation RMS: force 0.219 N, moment 0.015 Nm
# Equivalent combos (same RMS): flip all 6, or Rz180 offset — not needed if using above.
```

### 8.x `tmp/force_compensation/config/force_id.yaml`
*力标定流程*

```yaml
# Force compensation pipeline — config/force_id.yaml, config/poses.yaml
# Library: rm75_control.force.compensation
# Entry: tmp/force_compensation/force_calibrate.py, force_monitor.py

poses_yaml: poses.yaml

sequence: [a, b, c, d]
return_home: a

collect:
  move_speed: 15
  settle_timeout_s: 15.0
  dt_ms: 10.0
  log_every: 10
  scale: 1.0
  warmup_s: 3.0
  follow: false

  cartesian:
    duration_s: 30.0
    max_delta_mm: 5.0
    max_orient_deg:
      a: 18.0
      b: 32.0
      c: 32.0
    amp_mm: [3.0, 4.0, 2.0]
    amp_rot_deg: [12.0, 15.0, 12.0]
    amp_rot_deg_slots:
      b: [16.0, 20.0, 16.0]
      c: [16.0, 20.0, 16.0]
    freqs_hz:
      - [0.12, 0.18]
      - [0.13, 0.19]
      - [0.11, 0.16]
      - [0.22, 0.33]
      - [0.25, 0.37]
      - [0.24, 0.31]

  pose_d:
    # Phase 0: joint 45s. Phase 1: pose_d_vel_burst 45s (resync to q0 before burst).
    joint_duration_s: 45.0
    burst_duration_s: 45.0
    joint_amp_deg: [10.0, 8.0, 8.0, 14.0, 16.0, 14.0, 32.0]
    joint_max_delta_deg: [12.0, 12.0, 12.0, 18.0, 20.0, 18.0, 35.0]
    joint_freqs_hz:
      - [0.14, 0.21]
      - [0.13, 0.19]
      - [0.12, 0.18]
      - [0.22, 0.31]
      - [0.24, 0.33]
      - [0.23, 0.30]
      - [0.20, 0.29]
    velocity_burst:
      profile: pose_d_vel_burst
      # base wx→wy→wz 12°/s 0.28Hz traj=0; init settle 100ms (see excitation.settle_movev_after_init)
      ramp_down_s: 4.0

fit:
  force_sensor: configs/force_sensor.yaml
  holdout_frac: 0.2
  alpha_percentile: 70.0
  min_burst_rows: 300
  min_high_alpha_rows: 150
  inertia_r_max_m: 0.12
  npz_slots: [a, b, c, d]
  phi_output: force_id_phi.json
  phi_recommended_key: phi_burst

monitor:
  poll_ms: 50.0
  window_s: 25.0
  buffer_s: 4.0
  min_samples: 35
  refresh_hz: 12.0
  phi_source: phi_recommended
  use_inertia: true
```

### 8.x `tmp/force_compensation/config/poses.yaml`
*标定位姿 slot a/b/c/d*

```yaml
poses:
  a:
    label: pose_a
    note: 2026-06-27 verified
    pose_base:
    - 0.284324
    - -0.002917
    - 0.332434
    - -3.083
    - 0.043
    - 2.892
    q_deg:
    - 4.483
    - 15.902
    - -4.011
    - 72.358
    - -2.767
    - 90.212
    - 14.96
  b:
    label: pose_b
    note: saved 2026-06-27 12:42 UTC
    pose_base:
    - 0.278374
    - -0.095107
    - 0.358767
    - -2.449
    - 0.055
    - 3.059
    q_deg:
    - 4.253
    - 15.898
    - -3.625
    - 75.219
    - -38.729
    - 90.175
    - 6.913
  c:
    label: pose_c
    note: saved 2026-06-27 12:42 UTC
    pose_base:
    - 0.287775
    - 0.11732
    - 0.363931
    - 2.398
    - -0.047
    - 3.017
    q_deg:
    - 4.343
    - 15.893
    - -3.619
    - 75.287
    - 43.582
    - 85.76
    - 6.911
  d:
    label: pose_d
    note: saved 2026-06-27 13:02 UTC
    pose_base:
    - 0.210513
    - 0.003795
    - 0.515802
    - 2.642
    - 0.96
    - 2.639
    q_deg:
    - 4.987
    - -23.07
    - -3.953
    - 77.84
    - 2.451
    - 65.541
    - 14.407
```

### 8.x `tmp/Velocity_control/config/sin_tool_y.yaml`
*tool-Y 参考 demo 配置（对照）*

```yaml
# 参数已移到 run_sin_tool_y.py 命令行；此处仅作说明备份。
#
# 官方接口 (RM_API2 rm_interface.h):
#   rm_set_movev_canfd_init(avoid, frame_type, dt)  — 一次
#   rm_movev_canfd([vx,vy,vz,wx,wy,wz], follow, trajectory_mode, radio)  — 循环
#
# frame_type=0 → TCP/gripper 坐标系速度
# 推荐: follow=True, trajectory_mode=0, radio=0（完全透传；mode1/2 扫 sin 实测更差）
```

