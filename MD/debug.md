# RM75 力位混合速度导纳 — 完整代码包（debug）

> 用途：第三方审阅 / 离线对照。内容与仓库源码 **一字不差**。

> 运行 demo：`source env.sh && python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py --trajectory sin_base_y`

> 前置：`tmp/force_compensation/logs/force_id_phi.json`（先跑 force_calibrate.py）

> 外部依赖：RealMan `RM_API2/Python`、numpy、scipy、pyyaml

## 目录

- [零、模块关系](#零模块关系)
- [一、入口](#一入口)
- [二、velocity_admittance 包](#二velocity_admittance-包)
- [三、运动下发与 Session](#三运动下发与-session)
- [四、力补偿（phi → f_ext）](#四力补偿（phi-→-f_ext）)
- [五、YAML 配置](#五yaml-配置)
- [六、控制公式与修复要点](#六控制公式与修复要点)

## 零、模块关系

```
sin_tool_y_z2n.py / run_admittance.py
    └─ loop.run_velocity_admittance()
           ├─ RobotSession.recover_controller()     # 清 planner / 力控残留
           ├─ move_j + wait_settle (collection)     # 可选 slot 位姿
           ├─ rm_set_movev_canfd_init + 零速 settle
           ├─ post-init 锚定 pose0 / traj.set_origin
           ├─ AsyncStateObserver (2ms 后台读 pose+force)
           ├─ CompensatedForceObserver (phi → f_ext[2]≡tool-Z)
           ├─ TrajectoryGenerator → (pose_d, vel_ff) base 6D
           ├─ AdmittanceController
           │     v_pos_base = vel_ff + Kp⊙(pose_d - pose)
           │     fuse_constrained_xy: Tool-Z 力控 + 2×2 lstsq → Base-XY
           └─ send_velocity_canfd → rm_movev_canfd (frame_type=0 tool)
```

| 模块 | 职责 |
|------|------|
| `trajectory.py` | base 系 6D 参考 `(pose_d, vel_ff)`；spin 写入 pose_d |
| `observer.py` + `regressor.py` | 滚动 buffer + phi 回归 → sensor 系 `f_ext` |
| `controller.py` | PBAC + tool-Z PI 导纳 + **2×2 lstsq 融合**（替代旧 S_p 清零） |
| `async_state.py` | 后台高频反馈，主循环无 decimation |
| `loop.py` | 会话编排：init 锚定、接触检测、scan 切换 |
| `canfd.py` | `rm_movev_canfd` 封装 |


## 一、入口

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

## 二、velocity_admittance 包

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

### `rm75_control/control/velocity_admittance/async_state.py`

```python
"""Background pose/force polling — keeps the control loop non-blocking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AsyncStateSnapshot:
    pose: np.ndarray | None = None
    force_raw: np.ndarray = field(default_factory=lambda: np.zeros(6))
    t_s: float = 0.0
    ok: bool = False


class AsyncStateObserver:
    """
    Poll rm_get_current_arm_state / rm_get_force_data in a daemon thread.
    Main loop reads latest snapshot without blocking on RPC.
    """

    def __init__(self, robot, *, poll_s: float = 0.002) -> None:
        self.robot = robot
        self.poll_s = poll_s
        self._lock = threading.Lock()
        self._snap = AsyncStateSnapshot()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="va-async-state")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def wait_first_pose(self, timeout_s: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snap = self.read()
            if snap.pose is not None:
                return snap.pose.copy()
            time.sleep(0.005)
        raise TimeoutError("AsyncStateObserver: no pose within timeout")

    def read(self) -> AsyncStateSnapshot:
        with self._lock:
            if self._snap.pose is None:
                return AsyncStateSnapshot(
                    force_raw=self._snap.force_raw.copy(),
                    t_s=self._snap.t_s,
                    ok=False,
                )
            return AsyncStateSnapshot(
                pose=self._snap.pose.copy(),
                force_raw=self._snap.force_raw.copy(),
                t_s=self._snap.t_s,
                ok=self._snap.ok,
            )

    def _loop(self) -> None:
        while self._running:
            t_s = time.monotonic()
            ret_s, st = self.robot.rm_get_current_arm_state()
            ret_f, fd = self.robot.rm_get_force_data()
            snap = AsyncStateSnapshot(t_s=t_s)
            if ret_s == 0:
                snap.pose = np.asarray(st["pose"][:6], dtype=float)
            if ret_f == 0:
                snap.force_raw = np.asarray(fd["force_data"][:6], dtype=float)
            snap.ok = snap.pose is not None and ret_f == 0
            with self._lock:
                if snap.pose is not None:
                    self._snap.pose = snap.pose
                if ret_f == 0:
                    self._snap.force_raw = snap.force_raw
                self._snap.t_s = t_s
                self._snap.ok = snap.ok
            time.sleep(self.poll_s)
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
    f_ext from phi is in sensor frame; with sensor_offset=0 and TCP pure translation,
    f_ext[2] is used as tool-Z force (see observer docstring).
    Trajectory pose_d / vel_ff are base-frame 6D from a Trajectory6D producer.
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
    Pipeline (base trajectory → constrained fusion → movev):
      1. v_pos_base = vel_ff + kp * (pose_d - pose)
      2. fuse_constrained_xy: Tool-Z = force admittance; Tool-X/Y lstsq → Base-X/Y
      3. output v_cmd_tool (frame_type=0) or v_cmd_base
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
    def fuse_constrained_xy(
        v_pos_base: np.ndarray,
        v_force_tool: np.ndarray,
        r_mat: np.ndarray,
        *,
        normal_track: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        2×2 constrained projection: match Base-X/Y while Tool-Z is force-controlled.

        Replaces S_p @ (R^T v_base) which zeroed tool-Z trajectory rate and shrank
        world-Y stroke when the TCP is tilted.
        """
        v_cmd_tool = np.zeros(6, dtype=float)
        a_mat = r_mat[0:2, 0:2]
        b_vec = np.asarray(v_pos_base[0:2], dtype=float) - r_mat[0:2, 2] * float(v_force_tool[2])
        v_xy, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        v_cmd_tool[0:2] = v_xy
        v_cmd_tool[2] = float(v_force_tool[2])
        v_cmd_tool[3:6] = r_mat.T @ np.asarray(v_pos_base[3:6], dtype=float)
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

        v_cmd_tool, v_cmd_base = self.fuse_constrained_xy(
            v_pos_base, v_force_tool, r_mat, normal_track=normal_track,
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
    """Rotate pose_ref orientation by phi about tool +Z (base-frame axis)."""
    from scipy.spatial.transform import Rotation as Rsc

    pose = np.asarray(pose_ref, dtype=float).copy()
    if abs(phi_rad) < 1e-12:
        return pose
    r0 = Rsc.from_euler("xyz", pose_ref[3:6], degrees=False).as_matrix()
    axis = r0[:, 2]
    r_d = Rsc.from_rotvec(axis * phi_rad).as_matrix() @ r0
    pose[3:6] = Rsc.from_matrix(r_d).as_euler("xyz", degrees=False)
    return pose


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
        """
        Return (signed_filtered_raw, f_ext).

        f_ext is computed in the sensor frame (phi regressor). With
        sensor_offset_euler=0 and TCP offset a pure translation, linear
        f_ext[0:3] matches tool-frame force components — use f_ext[2] as tool-Z.
        """
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

from .async_state import AsyncStateObserver
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
    async_poll_ms = float(timing.get("async_poll_ms", 2.0))
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
        f"  kp_pos={ctrl_cfg.kp_pos.tolist()}  track_axes={ctrl_cfg.track_axes.tolist()}  "
        f"delay={ctrl_cfg.system_delay_s * 1000:.0f}ms  "
        f"async feedback ~{async_poll_ms:.0f}ms",
    )
    print(f"  phi: {PHI_JSON.name}  Fz_des={desired_z:.2f} N")
    print(f"  vz cap (tool TCP): ±{ctrl_cfg.max_vz_tool_m_s * 100:.1f} cm/s")
    print(f"  trajectory: {trajectory_summary(raw)}  kind={traj_kind}")
    scan_mode = "open-loop ff" if ctrl_cfg.open_loop else "closed-loop track"
    print(
        f"  hybrid: traj=6D base  fuse=2x2 lstsq + tool-Z force "
        f"S_f={ctrl_cfg.force_axes.tolist()}  "
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
        snap_detected = False
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
            snap_detected = (
                float(np.max(np.abs(deuler))) > 0.5
                or float(np.linalg.norm(dpos_mm)) > 2.0
            )
            if snap_detected:
                print(
                    "  WARN: init snap detected — extra zero-velocity settle",
                    flush=True,
                )
                settle_movev_after_init(
                    bot.robot, dt_ms=dt_ms, follow=follow,
                    trajectory_mode=traj_mode, radio=radio, n_frames=50,
                )
                ret_post2, st_post2 = bot.robot.rm_get_current_arm_state()
                if ret_post2 == 0:
                    pose_post = np.asarray(st_post2["pose"][:6], dtype=float)
            pose0 = pose_post.copy()
            traj_origin = pose0.copy()
            traj.set_origin(pose0)
            print(
                f"  anchored pose0 (post-init): "
                f"xyz=[{pose0[0]:.3f},{pose0[1]:.3f},{pose0[2]:.3f}] m",
                flush=True,
            )

        async_obs = AsyncStateObserver(bot.robot, poll_s=async_poll_ms / 1000.0)
        async_obs.start()
        try:
            pose_fb = async_obs.wait_first_pose(timeout_s=5.0)
        except TimeoutError:
            async_obs.stop()
            raise SystemExit("AsyncStateObserver: no pose after CANFD init")

        controller.reset()
        print("Velocity CANFD initialized. Ctrl+C to stop.", flush=True)

        t0 = time.monotonic()
        next_tick = t0
        last_log = t0
        scan_started = not wait_contact
        start_streak = 0
        t_scan0: float | None = None if wait_contact else t0
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

                snap = async_obs.read()
                if snap.pose is not None:
                    pose_fb = snap.pose
                    observer.append(t_s, pose_fb, snap.force_raw)
                    wrench = observer.latest_wrench()
                    if wrench is not None:
                        f_ext = wrench[1]
                        fz_buf.append(float(f_ext[2]))
                        if len(fz_buf) > 7:
                            fz_buf.pop(0)
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
                            controller.reset()
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
            async_obs.stop()
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

## 三、运动下发与 Session

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

## 四、力补偿（phi → f_ext）

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

## 五、YAML 配置

### `tmp/Velocity_Admittance/demo/config/sin_tool_y_z2n.yaml`

```yaml
# Demo trajectory plugin + tool-Z force hybrid.
# Architecture:
#   trajectory → 6D (pose_d, vel_ff) in base frame
#   controller → 2×2 lstsq (Base-X/Y track) + tool-Z force admittance
#
# Phase 1 (validate 16 cm stroke): open_loop: true, kp_pos all zero
# Phase 2 (weak PBAC Y drift): open_loop: false, uncomment phase2 block below
#
# Run:
#   source env.sh && cd /media/camp/EXT_DRIVE/rm75_control
#   python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py --trajectory sin_base_y

timing:
  dt_ms: 10.0
  async_poll_ms: 2.0

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
  force_axes: [0, 0, 1, 0, 0, 0]   # tool TCP-Z → force (f_ext[2] ≡ tool-Z when sensor∥flange)
  open_loop: true
  track_axes: [0, 1, 0, 0, 0, 0]   # Phase 1: Y only when closed-loop enabled
  kp_pos: [0, 0, 0, 0, 0, 0]
  # --- Phase 2 weak PBAC (after open-loop stroke OK) ---
  # open_loop: false
  # track_axes: [0, 1, 0, 1, 1, 1]   # Y + euler (pose_d includes spin)
  # kp_pos: [0, 1.0, 0, 0.3, 0.3, 0.3]
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
  async_poll_ms: 2.0

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

## 六、控制公式与修复要点

```text
# 每 10ms 控制周期
snap = AsyncStateObserver.read()          # ~2ms 后台更新
f_ext = phi_compensate(snap.force, snap.pose)
(pose_d, vel_ff) = traj.sample(t_scan)

v_pos_base = vel_ff + kp_pos ⊙ track_axes ⊙ (pose_d - pose_predicted)
v_force_tool[2] = PI_admittance(f_des[2] - f_ext[2])

# 2×2 约束融合（倾斜 TCP 下保留 world-Y 行程）
A = R[0:2, 0:2]
b = v_pos_base[0:2] - R[0:2, 2] * v_force_tool[2]
v_cmd_tool[0:2] = lstsq(A, b)
v_cmd_tool[2] = v_force_tool[2]
v_cmd_tool[3:6] = R.T @ v_pos_base[3:6]

rm_movev_canfd(v_cmd_tool, frame_type=0)
```

**Phase 1**：`open_loop: true`，验证 Y 16cm 行程。

**Phase 2**：弱 PBAC（yaml 注释块）：`track_axes` 含 Y（+ euler 若 spin 已写入 pose_d）。

