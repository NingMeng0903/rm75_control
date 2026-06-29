# RM75 力位混合速度导纳 — 完整代码包（debug）

> 用途：第三方审阅 / 离线对照。内容与仓库源码 **一字不差**（由 `scripts/gen_debug_va.py` 生成）。

> 运行：
> ```bash
> cd /media/camp/EXT_DRIVE/rm75_control && source env.sh
> python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py --trajectory sin_tool_y --log --duration 60
> ```

> 出图：
> ```bash
> python tmp/Velocity_Admittance/plot_scan_log.py tmp/Velocity_Admittance/logs/admittance_*.npz
> python tmp/Velocity_Admittance/plot_scan_log.py logs/admittance_xxx.npz --save /tmp/plot.png
> ```

> 前置：`tmp/force_compensation/logs/force_id_phi.json`

## 目录

- [零、架构与 log 诊断](#零架构与-log-诊断)
- [一、入口与出图](#一入口与出图)
- [二、velocity_admittance 包](#二velocity_admittance-包)
- [三、YAML](#三yaml)
- [四、CANFD 下发](#四canfd-下发)

## 零、架构与 log 诊断

```
sin_tool_y_z2n.py → loop.run_velocity_admittance()
  recover → move_j(slot) → CANFD init → post-init anchor pose0
  AsyncStateObserver (async_poll_ms=10)
  phases: 0 hold | 1 approach (Z force, pose_d=anchor) | 2 scan (traj.sample)
  scan ON: pending_scan → traj.set_origin(current pose); controller.reset(clear_velocity=False)
  deadband ramp: _contact_ticks/50 → db_alpha scales deadband (scan ON bumpless Z)
  CANFD init: settle max(frames,40); snap → deep settle 60 frames; re-anchor pose0
  Trajectory → pose_d + vel_ff (constant DOF: vel_ff=0; sin_tool_y: tool-Y only)
  fuse_tool_sleeve: v_cmd_tool[0:1]=R.T@v_pos; v_cmd_tool[2]=force PI
  rm_movev_canfd(frame_type=0 tool)
  --log → ScanLogRecorder → print_jerk_summary + plot_scan_log.py
```

| 维度 | 轨迹约束 | demo yaml（Phase2 已启用） |
|------|----------|---------------------------|
| tool-Y | sin, vel_ff | Kp=1.0 on track_axes[1] |
| tool-X | 常数 | Kp=0 |
| Rx,Ry,Rz | 常数 | Kp=1.0 姿态弱闭环 |
| tool-Z | 力控 | kp=0；deadband ramp 0.5s；k_fp_press=0.035 |

**三类跳变（勿混淆）**

1. **CANFD init**（t≈0~2s）：movev 模式切入，非接触 — deep settle + re-anchor
2. **scan ON**（t≈4s）：approach→scan 力律；db_alpha ramp 抹 vz 台阶
3. **pitch slip**（t≈5~8s）：Fz 丢失 + 软接触 — Phase2 PBAC 抑制

**跟踪评估（scan_log / plot）**

- 原点：`pose_act` @ scan ON（`phase≥2` 首样本）
- 指标：`tool-Y→world` = dot(Δpos, R0[:,1])；**不含 world-Z**（力控轴）
- `world-XY |Δcmd−Δact|`：交叉轨
- **Plot 仅画 scan 段位置**（pre-scan 用 scan0 参考会假 -200mm 误差）
- scan ON 瞬间 log 中 cmd≈act≈0；CANFD init 与 scan ON 是不同事件

**典型现象（软接触）**

- Fz 周期性 &lt;1N → 滑移；pitch 漂 → world ΔX 放大（非少给 X 维）
- v_cmd tool-Y 平滑 + pose 误差大 → 执行/接触，不是 fusion 尖峰

## 一、入口与出图

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
    parser.add_argument("--log", action="store_true", help="record pose_d vs pose_act npz every cycle")
    parser.add_argument("--log-path", type=Path, default=None, help="npz output (default: tmp/Velocity_Admittance/logs/)")
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
        log_enabled=args.log or args.log_path is not None,
        log_path=args.log_path,
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
    parser.add_argument("--log", action="store_true", help="record scan npz (pose_d vs pose_act)")
    parser.add_argument("--log-path", type=Path, default=None)
    args = parser.parse_args()

    raw = load_yaml(args.config)
    if args.trajectory:
        raw.setdefault("trajectory", {})["type"] = args.trajectory
    if args.desired_z is not None:
        raw.setdefault("force", {})["desired_z_n"] = args.desired_z

    return run_velocity_admittance(
        raw,
        duration_s=args.duration,
        log_enabled=args.log or args.log_path is not None,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
```
### `tmp/Velocity_Admittance/plot_scan_log.py`

```python
#!/usr/bin/env python3
"""Plot admittance scan log — diagnose controller vs execution vs contact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.velocity_admittance.scan_log import (
    load_scan_log,
    scan_tracking_world_mm,
)


def _euler_delta_deg(pose: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = pose - ref
    d[:, 3:6] = (d[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(d[:, 3:6])


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot admittance scan log npz")
    parser.add_argument("npz", type=Path)
    parser.add_argument("--save", type=Path, default=None, help="PNG output (use if no display)")
    parser.add_argument("--tmax", type=float, default=None, help="limit time axis (s)")
    args = parser.parse_args()

    d = load_scan_log(args.npz)
    t = d["t_s"]
    if args.tmax is not None:
        m = t <= args.tmax
        d = {k: v[m] if hasattr(v, "__len__") and len(v) == len(t) else v for k, v in d.items()}
        t = d["t_s"]

    v = d["v_cmd"]
    vf = d["vel_ff"]
    pose = d["pose_act"]
    pose_d = d["pose_d"]
    f = d["f_ext"]
    phase = d["phase"]
    scan = phase >= 2
    si = int(np.where(scan)[0][0]) if np.any(scan) else 0
    t_scan_on = float(t[si]) if np.any(scan) else 0.0
    dt_ms = np.concatenate([[np.nan], np.diff(t) * 1000.0])

    deuler = _euler_delta_deg(pose, pose[si])
    tr = scan_tracking_world_mm(pose_d, pose, scan_mask=scan if np.any(scan) else np.ones(len(t), dtype=bool))
    d_cmd = tr["d_cmd_mm"].copy()
    d_act = tr["d_act_mm"].copy()
    s_cmd = tr["s_cmd_mm"].copy()
    s_act = tr["s_act_mm"].copy()
    track_err = tr["scan_track_err_mm"].copy()
    # Pre-scan: pose_d / ref frame differ (hold logs zeros, approach logs post-init pose0).
    # Mask so position panels only show scan tracking (cmd=act at scan ON).
    pre = ~scan
    d_cmd[pre] = np.nan
    d_act[pre] = np.nan
    s_cmd[pre] = np.nan
    s_act[pre] = np.nan
    track_err[pre] = np.nan

    vy_ff_tool = np.zeros(len(t))
    for i in range(len(t)):
        r = Rsc.from_euler("xyz", pose[i, 3:6], degrees=False).as_matrix()
        vy_ff_tool[i] = (r.T @ vf[i, :3])[1]

    fig, axes = plt.subplots(7, 1, figsize=(13, 14), sharex=True)
    fig.suptitle(
        f"{args.npz.name}  scan position vs scan0 (pre-scan masked); vel: full run"
    )

    ax = axes[0]
    ax.plot(t, v[:, 1], "C0", lw=0.8, label="v_cmd tool-Y")
    ax.plot(t, vy_ff_tool, "C1", lw=0.8, alpha=0.7, label="R.T@vel_ff tool-Y")
    ax.axvline(t[si], color="gray", ls="--", lw=0.8, label="scan ON")
    ax.set_ylabel("tool Y vel (m/s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, v[:, 2], "C2", lw=0.8, label="v_cmd tool-Z")
    ax.set_ylabel("tool Z vel")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, f[:, 2], "C3", lw=0.8, label="Fz ext")
    ax.axhline(3.0, color="r", ls="--", lw=0.8, label="Fz des 3N")
    ax.axhline(1.0, color="orange", ls=":", lw=0.8, label="contact loss ~1N")
    ax.set_ylabel("Fz (N)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(t, deuler[:, 1], label="Δpitch deg")
    ax.plot(t, deuler[:, 2], label="Δyaw deg", alpha=0.8)
    ax.set_ylabel("Δeuler vs scan0")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    ax.plot(t, s_cmd, "C1", lw=0.9, ls="--", label="cmd tool-Y→world")
    ax.plot(t, s_act, "C0", lw=0.9, label="act tool-Y→world")
    ax.plot(t, track_err, "C3", lw=0.6, alpha=0.7, label="track err")
    ax.axvline(t_scan_on, color="gray", ls="--", lw=0.8)
    if np.any(scan):
        ax.scatter([t_scan_on], [0.0], c="k", s=12, zorder=5, label="scan ON cmd≈act")
    ax.set_ylabel("scan axis mm")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[5]
    ax.plot(t, d_cmd[:, 1], "C3", lw=0.8, ls="--", label="cmd ΔY world")
    ax.plot(t, d_act[:, 1], "C2", lw=0.8, label="act ΔY world")
    ax.set_ylabel("world ΔY mm")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t, d_cmd[:, 0], "C1", lw=0.8, ls=":", alpha=0.9, label="cmd ΔX")
    ax2.plot(t, d_act[:, 0], "C0", lw=0.6, alpha=0.8, label="act ΔX")
    ax2.set_ylabel("world ΔX mm (right)")
    cx = float(np.max(np.abs(d_cmd[scan, 0]))) if np.any(scan) else 0.0
    ax.text(
        0.02, 0.04,
        f"sin_tool_y: 1 DOF tool-Y only; cmd ΔX≈±{cx:.1f}mm from tilt\n"
        "act ΔX blow-up = contact slip + pitch drift (not missing X cmd)",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )

    ax = axes[6]
    ax.plot(t, dt_ms, "k", lw=0.6, alpha=0.7)
    ax.axhline(10, color="g", ls="--", lw=0.8)
    ax.axhline(15, color="r", ls=":", lw=0.8)
    ax.set_ylabel("loop dt ms")
    ax.set_xlabel("t (s)")
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.fill_between(t, ax.get_ylim()[0], ax.get_ylim()[1], where=scan, alpha=0.04, color="blue")

    fig.tight_layout()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=130)
        print(f"saved → {args.save}")
    else:
        plt.show()
    return 0


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
from rm75_control.control.velocity_admittance.scan_log import (
    ScanLogRecorder,
    load_scan_log,
    print_jerk_summary,
    scan_tracking_world_mm,
)
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
    "ScanLogRecorder",
    "load_scan_log",
    "print_jerk_summary",
    "scan_tracking_world_mm",
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
    q_deg: np.ndarray | None = None
    force_raw: np.ndarray = field(default_factory=lambda: np.zeros(6))
    t_s: float = 0.0
    ok: bool = False


class AsyncStateObserver:
    """
    Poll rm_get_current_arm_state / rm_get_force_data in a daemon thread.
    Main loop reads latest snapshot without blocking on RPC.
    """

    def __init__(self, robot, *, poll_s: float = 0.01) -> None:
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
                q_deg=self._snap.q_deg.copy() if self._snap.q_deg is not None else None,
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
                snap.q_deg = np.asarray(st["joint"][:7], dtype=float)
            if ret_f == 0:
                snap.force_raw = np.asarray(fd["force_data"][:6], dtype=float)
            snap.ok = snap.pose is not None and ret_f == 0
            with self._lock:
                if snap.pose is not None:
                    self._snap.pose = snap.pose
                if snap.q_deg is not None:
                    self._snap.q_deg = snap.q_deg
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
        self._contact_ticks = 0

    def reset(self, *, clear_velocity: bool = False) -> None:
        self.force_error_integral.fill(0.0)
        self._contact_ticks = 0
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

        err_pose = pose_error(desired_pose, pose_predicted)
        vel_ff = np.asarray(desired_vel_ff, dtype=float).copy()
        use_pbac = (not cfg.open_loop) if enable_pbac is None else bool(enable_pbac)
        if not use_pbac:
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

        if in_contact:
            self._contact_ticks += 1
        else:
            self._contact_ticks = 0
        db_alpha = min(1.0, self._contact_ticks / 50.0)

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
        v_final = np.clip(v_clamp, self.last_v_cmd - dv_max, self.last_v_cmd + dv_max)
        self.last_v_cmd = v_final.copy()
        return v_final

    def _admittance_axis(
        self,
        axis: int,
        f_err: float,
        in_contact: bool,
        db_alpha: float = 1.0,
    ) -> float:
        cfg = self.cfg
        if axis == 2 and not in_contact:
            self.force_error_integral[axis] = 0.0
            v = cfg.k_fp_release * f_err
            cap = cfg.approach_vz_tool_m_s
            return float(np.clip(v, -cap, cap))

        if abs(f_err) > 5.0:
            self.force_error_integral[axis] = 0.0

        actual_deadband = cfg.deadband_n * db_alpha
        actual_width = cfg.deadband_width_n * db_alpha
        eff = smooth_deadband_eff(f_err, actual_deadband, actual_width)
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
```
### `rm75_control/control/velocity_admittance/trajectory.py`

```python
"""6D trajectory producers (base frame). Hybrid controller consumes pose_d + vel_ff."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from .rm_algo import end2tool_pose


def tool_frame_delta_pose(
    pose_ref: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    *,
    euler_order: str = "xyz",
) -> np.ndarray:
    """Tool-frame translation without rm_algo RPC (matches frameMode=1 pure translation)."""
    pose = np.asarray(pose_ref, dtype=float).copy()
    r_mat = Rsc.from_euler(euler_order, pose[3:6], degrees=False).as_matrix()
    pose[:3] = pose[:3] + r_mat @ np.array([dx, dy, dz], dtype=float)
    return pose


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
        r_mat = Rsc.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
        vel = np.zeros(6, dtype=float)
        vel[:3] = r_mat @ np.array([0.0, vy, 0.0], dtype=float)
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
### `rm75_control/control/velocity_admittance/scan_log.py`

```python
"""High-rate scan log: target trajectory vs actual encoder feedback."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rm75_control.control.velocity_admittance.paths import VA_DATA_DIR


LOG_DIR = VA_DATA_DIR / "logs"
_GROW = 512


class ScanLogRecorder:
    """Pre-allocated ring growth — minimal per-tick overhead."""

    def __init__(self, *, capacity: int = 4096) -> None:
        self._cap = capacity
        self._n = 0
        self.t_s = np.zeros(capacity, dtype=float)
        self.t_scan = np.full(capacity, np.nan, dtype=float)
        self.phase = np.zeros(capacity, dtype=np.int8)
        self.pose_act = np.zeros((capacity, 6), dtype=float)
        self.q_deg = np.zeros((capacity, 7), dtype=float)
        self.pose_d = np.zeros((capacity, 6), dtype=float)
        self.vel_ff = np.zeros((capacity, 6), dtype=float)
        self.v_cmd = np.zeros((capacity, 6), dtype=float)
        self.f_ext = np.zeros((capacity, 6), dtype=float)
        self.f_des_z = np.zeros(capacity, dtype=float)

    def __len__(self) -> int:
        return self._n

    def _grow(self) -> None:
        new_cap = self._cap + _GROW
        for name in (
            "t_s", "t_scan", "phase", "f_des_z",
        ):
            old = getattr(self, name)
            ext = np.zeros(new_cap, dtype=old.dtype)
            ext[: self._cap] = old
            if name == "t_scan":
                ext[self._cap :] = np.nan
            setattr(self, name, ext)
        for name in ("pose_act", "pose_d", "vel_ff", "v_cmd", "f_ext"):
            old = getattr(self, name)
            ext = np.zeros((new_cap, 6), dtype=float)
            ext[: self._cap] = old
            setattr(self, name, ext)
        old = self.q_deg
        ext = np.zeros((new_cap, 7), dtype=float)
        ext[: self._cap] = old
        self.q_deg = ext
        self._cap = new_cap

    def append_row(
        self,
        *,
        t_s: float,
        t_scan: float,
        phase: int,
        pose_act: np.ndarray,
        q_deg: np.ndarray,
        pose_d: np.ndarray,
        vel_ff: np.ndarray,
        v_cmd: np.ndarray,
        f_ext: np.ndarray,
        f_des_z: float,
    ) -> None:
        if self._n >= self._cap:
            self._grow()
        i = self._n
        self.t_s[i] = t_s
        self.t_scan[i] = t_scan
        self.phase[i] = phase
        self.pose_act[i] = pose_act
        self.q_deg[i] = q_deg
        self.pose_d[i] = pose_d
        self.vel_ff[i] = vel_ff
        self.v_cmd[i] = v_cmd
        self.f_ext[i] = f_ext
        self.f_des_z[i] = f_des_z
        self._n += 1

    def save(self, path: Path, *, meta: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = self._n
        if n == 0:
            raise ValueError("ScanLogRecorder: no samples")
        pack = {
            "t_s": self.t_s[:n].copy(),
            "t_scan": self.t_scan[:n].copy(),
            "phase": self.phase[:n].copy(),
            "pose_act": self.pose_act[:n].copy(),
            "q_deg": self.q_deg[:n].copy(),
            "pose_d": self.pose_d[:n].copy(),
            "vel_ff": self.vel_ff[:n].copy(),
            "v_cmd": self.v_cmd[:n].copy(),
            "f_ext": self.f_ext[:n].copy(),
            "f_des_z": self.f_des_z[:n].copy(),
        }
        if meta:
            pack["meta_json"] = np.array([str(meta)])
        np.savez_compressed(path, **pack)
        return path


def default_log_path(prefix: str = "admittance") -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"{prefix}_{stamp}.npz"


def load_scan_log(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def scan_origin_r(pose_act: np.ndarray, scan_mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Scan ON index, pose0, and R0 (tool orientation in world at scan start)."""
    from scipy.spatial.transform import Rotation as Rsc

    idx = np.where(scan_mask)[0]
    if len(idx) == 0:
        return 0, pose_act[0].copy(), np.eye(3)
    si = int(idx[0])
    pose0 = pose_act[si].copy()
    r0 = Rsc.from_euler("xyz", pose0[3:6], degrees=False).as_matrix()
    return si, pose0, r0


def world_delta_mm(pose: np.ndarray, pose0: np.ndarray) -> np.ndarray:
    """TCP linear displacement vs scan origin, in world frame (mm)."""
    return (pose[:, :3] - pose0[:3]) * 1000.0


def tool_y_world_scalar_mm(delta_world_mm: np.ndarray, r0: np.ndarray) -> np.ndarray:
    """Tool-Y scan progress: world displacement projected onto tool +Y at scan ON."""
    e_scan = r0[:, 1]
    return delta_world_mm @ e_scan


def scan_tracking_world_mm(
    pose_d: np.ndarray,
    pose_act: np.ndarray,
    *,
    scan_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Decoupled scan tracking in world frame.

    Compare commanded vs actual TCP world deltas from scan origin. TCP-Z is force-controlled:
    use scan-axis scalar (tool-Y in world @ scan0) and world-XY cross-track, not world-Z.
    """
    si, pose0, r0 = scan_origin_r(pose_act, scan_mask)
    d_cmd = world_delta_mm(pose_d, pose0)
    d_act = world_delta_mm(pose_act, pose0)
    s_cmd = tool_y_world_scalar_mm(d_cmd, r0)
    s_act = tool_y_world_scalar_mm(d_act, r0)
    dxy_err = (d_cmd[:, :2] - d_act[:, :2])
    return {
        "d_cmd_mm": d_cmd,
        "d_act_mm": d_act,
        "s_cmd_mm": s_cmd,
        "s_act_mm": s_act,
        "scan_track_err_mm": s_cmd - s_act,
        "world_xy_err_mm": np.linalg.norm(dxy_err, axis=1),
        "scan_idx": si,
        "r0": r0,
    }


def print_jerk_summary(path: Path, *, dt_s: float) -> None:
    """Print v_cmd / pose tracking diagnostics to separate planner vs execution."""
    data = load_scan_log(path)
    t = data["t_s"]
    v = data["v_cmd"]
    pose_act = data["pose_act"]
    pose_d = data["pose_d"]
    phase = data["phase"]
    n = len(t)
    if n < 3:
        print(f"  log summary: too few samples ({n})", flush=True)
        return

    dv = np.diff(v, axis=0) / np.maximum(np.diff(t)[:, None], 1e-6)
    jerk_proxy = np.diff(dv, axis=0) / np.maximum(np.diff(t)[1:, None], 1e-6)
    scan_mask = phase >= 2
    mask = scan_mask if np.any(scan_mask) else np.ones(n, dtype=bool)

    idx = np.where(mask)[0]
    idx = idx[(idx > 0) & (idx < n - 2)]
    if len(idx) == 0:
        idx = np.arange(1, min(n - 2, n))

    finite_dv = np.isfinite(dv).all(axis=1)
    finite_jk = np.isfinite(jerk_proxy).all(axis=1) if len(jerk_proxy) else np.array([True])
    idx_dv = idx[np.isin(idx - 1, np.where(finite_dv)[0])]
    idx_jk = idx[np.isin(idx - 2, np.where(finite_jk)[0])]

    dv_n = np.linalg.norm(dv[idx_dv - 1], axis=1) if len(idx_dv) else np.array([0.0])
    jk_n = (
        np.linalg.norm(jerk_proxy[idx_jk - 2], axis=1)
        if len(idx_jk) and len(jerk_proxy)
        else np.array([0.0])
    )

    tr = scan_tracking_world_mm(pose_d, pose_act, scan_mask=mask)
    idx_scan = np.where(mask)[0]
    s_cmd = tr["s_cmd_mm"][idx_scan] if len(idx_scan) else np.array([0.0])
    s_act = tr["s_act_mm"][idx_scan] if len(idx_scan) else np.array([0.0])
    scan_track = np.abs(tr["scan_track_err_mm"][idx_scan]) if len(idx_scan) else np.array([0.0])
    xy_cross = tr["world_xy_err_mm"][idx_scan] if len(idx_scan) else np.array([0.0])

    loop_dt = np.diff(t[scan_mask]) * 1000.0 if np.any(scan_mask) else np.diff(t) * 1000.0
    loop_dt = loop_dt[np.isfinite(loop_dt)]

    print("\n=== scan log summary ===", flush=True)
    print(f"  file: {path}", flush=True)
    print(f"  samples={n}  scan_samples={int(np.sum(scan_mask))}  dt_nom={dt_s*1000:.1f}ms", flush=True)
    if len(loop_dt):
        print(
            f"  loop dt ms: median={float(np.median(loop_dt)):.2f}  "
            f"max={float(np.max(loop_dt)):.2f}  "
            f">15ms={int(np.sum(loop_dt > 15))}/{len(loop_dt)}",
            flush=True,
        )
    print(
        f"  |dv_cmd| max={float(np.nanmax(dv_n)) if len(dv_n) else 0:.4f} m/s²  "
        f"p95={float(np.nanpercentile(dv_n, 95)) if len(dv_n) else 0:.4f}  "
        f"|jerk_proxy| max={float(np.nanmax(jk_n)) if len(jk_n) else 0:.2f}",
        flush=True,
    )
    if len(idx_scan):
        print(
            f"  tool-Y world (scan axis @ scan0): track err max={float(np.max(scan_track)):.2f} mm  "
            f"p95={float(np.percentile(scan_track, 95)):.2f} mm  "
            f"(world-Z decoupled — force axis)",
            flush=True,
        )
        print(
            f"  tool-Y world stroke  cmd [{float(s_cmd.min()):+.1f}, {float(s_cmd.max()):+.1f}] mm  "
            f"act [{float(s_act.min()):+.1f}, {float(s_act.max()):+.1f}] mm  "
            f"world-XY |Δcmd−Δact| p95={float(np.percentile(xy_cross, 95)):.2f} mm",
            flush=True,
        )
        print(
            "  (large world track err + smooth v_cmd tool-Y → execution/contact slip)",
            flush=True,
        )
    for axis, name in enumerate(["vx", "vy", "vz", "wx", "wy", "wz"]):
        col = v[scan_mask, axis] if np.any(scan_mask) else v[:, axis]
        col = col[np.isfinite(col)]
        if len(col) < 2:
            continue
        dcol = np.diff(col) / dt_s
        dcol = dcol[np.isfinite(dcol)]
        if len(dcol) == 0:
            continue
        spikes = int(np.sum(np.abs(dcol) > 3.0 * float(np.std(dcol) + 1e-9)))
        print(
            f"  v_cmd {name}: std={float(np.std(col)):.5f}  "
            f"|dv/dt| max={float(np.max(np.abs(dcol))):.4f}  spikes(>3σ)={spikes}",
            flush=True,
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
from .scan_log import ScanLogRecorder, default_log_path, print_jerk_summary
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
    log_path: Path | None = None,
    log_enabled: bool | None = None,
) -> int:
    if not PHI_JSON.exists():
        raise SystemExit(f"Missing {PHI_JSON} — run force_calibrate.py first")

    timing = raw.get("timing", {})
    dt_ms = float(timing.get("dt_ms", 10.0))
    dt_s = dt_ms / 1000.0
    async_poll_ms = float(timing.get("async_poll_ms", 10.0))
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

    monitor = raw.get("monitor", {})
    if log_enabled is None:
        log_enabled = bool(monitor.get("log", False))
    log_every = max(1, int(monitor.get("log_every", 1)))
    if log_enabled and log_path is None:
        log_path = default_log_path()

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
        f"  hybrid: traj/Servo=6D base  fuse=tool_sleeve (Z force, XY ff) "
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
    if log_enabled:
        print(f"  scan log: ON → {log_path}  every {log_every} cycle(s)", flush=True)

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
            trajectory_mode=traj_mode, radio=radio, n_frames=max(settle_frames, 40),
        )
        ret_post, st_post = bot.robot.rm_get_current_arm_state()
        pose_post = pose0.copy()
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
                float(np.max(np.abs(deuler))) > 0.8
                or float(np.linalg.norm(dpos_mm)) > 3.0
            )
            if snap_detected:
                print(
                    f"  WARN: init snap detected (Δpos={float(np.linalg.norm(dpos_mm)):.1f}mm)"
                    f" — deep settle...",
                    flush=True,
                )
                settle_movev_after_init(
                    bot.robot, dt_ms=dt_ms, follow=follow,
                    trajectory_mode=traj_mode, radio=radio, n_frames=60,
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
        q_fb = np.zeros(7, dtype=float)

        controller.reset(clear_velocity=True)
        print("Velocity CANFD initialized. Ctrl+C to stop.", flush=True)

        scan_log = ScanLogRecorder() if log_enabled else None
        log_tick = 0

        t0 = time.monotonic()
        next_tick = t0
        last_log = t0
        scan_started = not wait_contact
        pending_scan = False
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
                    if snap.q_deg is not None:
                        q_fb = snap.q_deg
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
                            pending_scan = True
                            start_streak = 0

                if pending_scan and not scan_started:
                    pending_scan = False
                    scan_started = True
                    t_scan0 = now
                    traj_origin = pose.copy()
                    traj.set_origin(traj_origin)
                    controller.reset(clear_velocity=False)
                    print(
                        f"  scan ON @ t={t_s:.1f}s  Fz={f_ext[2]:+.2f}N  traj={traj_kind}",
                        flush=True,
                    )

                t_scan = (now - t_scan0) if (scan_started and t_scan0 is not None) else float("nan")
                phase = 2 if scan_started else (1 if t_s >= hold_s else 0)
                pose_d_log = np.zeros(6, dtype=float)
                vel_ff_log = np.zeros(6, dtype=float)
                f_des_z = float(desired_z)

                if not scan_started:
                    if t_s < hold_s or (require_observer and not observer.ready()):
                        v_cmd = np.zeros(6, dtype=float)
                        phase = 0
                        pose_d_log = pose.copy()
                    else:
                        since_approach = max(0.0, t_s - hold_s)
                        f_scale = (
                            min(1.0, since_approach / approach_ramp_s)
                            if approach_ramp_s > 0
                            else 1.0
                        )
                        f_des_z = float(desired_z * f_scale)
                        pose_d_log = pose0.copy()
                        v_cmd = controller.compute_velocity_command(
                            pose, pose0, np.zeros(6), f_ext, f_des * f_scale,
                            in_contact=False,
                            enable_pbac=False,
                        )
                        phase = 1
                    send_velocity_canfd(
                        bot.robot, v_cmd.tolist(),
                        follow=follow, trajectory_mode=traj_mode, radio=radio,
                    )
                    if scan_log is not None:
                        log_tick += 1
                        if log_tick >= log_every:
                            log_tick = 0
                            scan_log.append_row(
                                t_s=t_s, t_scan=t_scan, phase=phase,
                                pose_act=pose, q_deg=q_fb, pose_d=pose_d_log,
                                vel_ff=vel_ff_log, v_cmd=v_cmd, f_ext=f_ext,
                                f_des_z=f_des_z,
                            )
                    continue

                sample = traj.sample(t_scan)
                pose_d_log = sample.pose_d.copy()
                vel_ff_log = sample.vel_ff.copy()
                v_cmd = controller.compute_velocity_command(
                    pose, sample.pose_d, sample.vel_ff, f_ext, f_des,
                    in_contact=True,
                    enable_pbac=True,
                )
                phase = 2

                v_cmd = np.asarray(v_cmd, dtype=float)
                if not np.all(np.isfinite(v_cmd)):
                    print(
                        f"  WARN: non-finite v_cmd {v_cmd} — sending zero (phase={phase})",
                        flush=True,
                    )
                    v_cmd = np.zeros(6, dtype=float)

                if scan_log is not None:
                    log_tick += 1
                    if log_tick >= log_every:
                        log_tick = 0
                        scan_log.append_row(
                            t_s=t_s,
                            t_scan=t_scan,
                            phase=phase,
                            pose_act=pose,
                            q_deg=q_fb,
                            pose_d=pose_d_log,
                            vel_ff=vel_ff_log,
                            v_cmd=v_cmd,
                            f_ext=f_ext,
                            f_des_z=f_des_z,
                        )

                send_velocity_canfd(
                    bot.robot, v_cmd.tolist(),
                    follow=follow, trajectory_mode=traj_mode, radio=radio,
                )

                if scan_started and now - last_log >= 1.0:
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
            if scan_log is not None and len(scan_log) > 0 and log_path is not None:
                try:
                    saved = scan_log.save(
                        log_path,
                        meta={
                            "traj_kind": traj_kind,
                            "dt_ms": dt_ms,
                            "async_poll_ms": async_poll_ms,
                            "control_frame": control_frame,
                            "frame_type": frame_type,
                        },
                    )
                    print(f"  scan log saved → {saved} ({len(scan_log)} samples)", flush=True)
                    print_jerk_summary(saved, dt_s=dt_s)
                except Exception as exc:
                    print(f"  scan log save failed: {exc}", flush=True)
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
## 三、YAML

### `tmp/Velocity_Admittance/demo/config/sin_tool_y_z2n.yaml`

```yaml
# Demo trajectory plugin + tool-Z force hybrid.
# Architecture (sleeve / slider):
#   trajectory → pose_d + vel_ff (base frame); other DOF constant except scan axis + optional spin
#   controller → Tool-X/Y from R.T @ v_pos_base; Tool-Z from force admittance only
#   open_loop: false + scan-only PBAC (see loop enable_pbac); attitude kp only, Y=vel_ff
#
# Scan log (--log): phase 0 hold pose_d=pose_act; phase 1 pose_d=post-init anchor;
#   phase 2 pose_d from traj.sample; scan ON → traj.set_origin(current pose), cmd≈act@scan0
# Plot: python tmp/Velocity_Admittance/plot_scan_log.py logs/admittance_*.npz
#   Position panels mask pre-scan (avoid false -200mm cmd vs scan0 ref)
#
# Run:
#   source /media/camp/EXT_DRIVE/rm75_control/env.sh
#   cd /media/camp/EXT_DRIVE/rm75_control
#   python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py --trajectory sin_tool_y --log

timing:
  dt_ms: 10.0
  async_poll_ms: 10.0

startup:
  pose_slot: d
  settle_frames: 40
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
  type: sin_base_y_tool_rz    # hold | sin_tool_y | sin_base_y | sin_base_y_tool_rz
  y_peak_to_peak_cm: 16.0
  rz_amplitude_deg: 12.0
  y_max_vel_cm_s: 3.0
  soft_start: true
  ramp_s: 2.0
  open_loop: true

controller:
  force_axes: [0, 0, 1, 0, 0, 0]   # tool TCP-Z force; f_ext[2] when sensor parallel flange
  open_loop: false
  # Scan-only PBAC (loop enable_pbac=False during approach). Y via vel_ff only; lock attitude.
  track_axes: [0, 0, 0, 1, 1, 1]
  kp_pos: [0.0, 0.0, 0.0, 0.3, 0.3, 0.3]
  deadband_n: 0.3
  deadband_width_n: 0.2
  system_delay_s: 0.015
  k_fp_press: 0.035
  k_fp_release: 0.025
  k_fi: 0.001
  integral_limit: 0.015
  max_vz_tool_m_s: 0.15
  approach_vz_tool_m_s: 0.03
  max_velocity: [0.03, 0.10, 0.15, 0.10, 0.10, 0.10]
  max_acceleration: [0.5, 1.0, 0.5, 0.3, 0.3, 0.3]

monitor:
  log: false   # or --log on CLI
  log_every: 1  # 2 = half rate if needed; control loop always 100Hz
```
### `tmp/Velocity_Admittance/config/admittance.yaml`

```yaml
# Velocity admittance — library: rm75_control.control.velocity_admittance
# Generic: python tmp/Velocity_Admittance/run_admittance.py
# Demo:    python tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py

timing:
  dt_ms: 10.0
  async_poll_ms: 10.0

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
## 四、CANFD 下发

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

## 五、套筒融合公式

```text
v_pos_base = vel_ff + kp ⊙ track_axes ⊙ (pose_d - pose)   # open_loop: err=0
v_pos_tool[:3] = R.T @ v_pos_base[:3]
v_pos_tool[3:6] = R.T @ v_pos_base[3:6]
v_cmd_tool = v_pos_tool;  v_cmd_tool[2] = PI_admittance(f_des - f_ext)
rm_movev_canfd(v_cmd_tool, frame_type=0)
```

重新生成本文档：`python scripts/gen_debug_va.py`
