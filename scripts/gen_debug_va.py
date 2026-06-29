#!/usr/bin/env python3
"""Regenerate MD/debug.md from velocity admittance source (verbatim embed)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "MD" / "debug.md"

SECTIONS: list[tuple[str, Path]] = [
    ("tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py", REPO / "tmp/Velocity_Admittance/demo/sin_tool_y_z2n.py"),
    ("tmp/Velocity_Admittance/run_admittance.py", REPO / "tmp/Velocity_Admittance/run_admittance.py"),
    ("tmp/Velocity_Admittance/plot_scan_log.py", REPO / "tmp/Velocity_Admittance/plot_scan_log.py"),
    ("rm75_control/control/velocity_admittance/__init__.py", REPO / "rm75_control/control/velocity_admittance/__init__.py"),
    ("rm75_control/control/velocity_admittance/paths.py", REPO / "rm75_control/control/velocity_admittance/paths.py"),
    ("rm75_control/control/velocity_admittance/async_state.py", REPO / "rm75_control/control/velocity_admittance/async_state.py"),
    ("rm75_control/control/velocity_admittance/controller.py", REPO / "rm75_control/control/velocity_admittance/controller.py"),
    ("rm75_control/control/velocity_admittance/trajectory.py", REPO / "rm75_control/control/velocity_admittance/trajectory.py"),
    ("rm75_control/control/velocity_admittance/observer.py", REPO / "rm75_control/control/velocity_admittance/observer.py"),
    ("rm75_control/control/velocity_admittance/scan_log.py", REPO / "rm75_control/control/velocity_admittance/scan_log.py"),
    ("rm75_control/control/velocity_admittance/loop.py", REPO / "rm75_control/control/velocity_admittance/loop.py"),
    ("rm75_control/control/velocity_admittance/rm_algo.py", REPO / "rm75_control/control/velocity_admittance/rm_algo.py"),
    ("tmp/Velocity_Admittance/demo/config/sin_tool_y_z2n.yaml", REPO / "tmp/Velocity_Admittance/demo/config/sin_tool_y_z2n.yaml"),
    ("tmp/Velocity_Admittance/config/admittance.yaml", REPO / "tmp/Velocity_Admittance/config/admittance.yaml"),
    ("rm75_control/motion/canfd.py", REPO / "rm75_control/motion/canfd.py"),
]


def lang(path: Path) -> str:
    if path.suffix in (".yaml", ".yml"):
        return "yaml"
    return "python"


def embed(rel: str, path: Path) -> str:
    body = path.read_text(encoding="utf-8")
    return f"### `{rel}`\n\n```{lang(path)}\n{body}```\n"


def main() -> None:
    header = """# RM75 力位混合速度导纳 — 完整代码包（debug）

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

"""
    parts = [header]
    for rel, path in SECTIONS[:3]:
        parts.append(embed(rel, path))

    parts.append("## 二、velocity_admittance 包\n\n")
    for rel, path in SECTIONS[3:12]:
        parts.append(embed(rel, path))

    parts.append("## 三、YAML\n\n")
    for rel, path in SECTIONS[12:14]:
        parts.append(embed(rel, path))

    parts.append("## 四、CANFD 下发\n\n")
    for rel, path in SECTIONS[14:]:
        parts.append(embed(rel, path))

    parts.append("""
## 五、套筒融合公式

```text
v_pos_base = vel_ff + kp ⊙ track_axes ⊙ (pose_d - pose)   # open_loop: err=0
v_pos_tool[:3] = R.T @ v_pos_base[:3]
v_pos_tool[3:6] = R.T @ v_pos_base[3:6]
v_cmd_tool = v_pos_tool;  v_cmd_tool[2] = PI_admittance(f_des - f_ext)
rm_movev_canfd(v_cmd_tool, frame_type=0)
```

重新生成本文档：`python scripts/gen_debug_va.py`
""")

    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
