#!/usr/bin/env python3
"""Regenerate MD/debug.md — verbatim hybrid_motion source mirror."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "MD" / "debug.md"

SECTIONS: list[tuple[str, Path]] = [
    # Demo entry + trajectory plugin
    ("tmp/Velocity_Admittance/demo/d_to_a_sin_tool_y.py", REPO / "tmp/Velocity_Admittance/demo/d_to_a_sin_tool_y.py"),
    ("tmp/Velocity_Admittance/demo/trajectory_builtin.py", REPO / "tmp/Velocity_Admittance/demo/trajectory_builtin.py"),
    ("tmp/Velocity_Admittance/plot_scan_log.py", REPO / "tmp/Velocity_Admittance/plot_scan_log.py"),
    # Active scan config
    ("tmp/Velocity_Admittance/demo/config/human_soft_scan.yaml", REPO / "tmp/Velocity_Admittance/demo/config/human_soft_scan.yaml"),
    # hybrid_motion — velocity force-position decoupling stack
    ("rm75_control/control/hybrid_motion/__init__.py", REPO / "rm75_control/control/hybrid_motion/__init__.py"),
    ("rm75_control/control/hybrid_motion/paths.py", REPO / "rm75_control/control/hybrid_motion/paths.py"),
    ("rm75_control/control/hybrid_motion/async_state.py", REPO / "rm75_control/control/hybrid_motion/async_state.py"),
    ("rm75_control/control/hybrid_motion/reference.py", REPO / "rm75_control/control/hybrid_motion/reference.py"),
    ("rm75_control/control/hybrid_motion/reference_shaper.py", REPO / "rm75_control/control/hybrid_motion/reference_shaper.py"),
    ("rm75_control/control/hybrid_motion/observer.py", REPO / "rm75_control/control/hybrid_motion/observer.py"),
    ("rm75_control/control/hybrid_motion/adaptive_ke.py", REPO / "rm75_control/control/hybrid_motion/adaptive_ke.py"),
    ("rm75_control/control/hybrid_motion/controller.py", REPO / "rm75_control/control/hybrid_motion/controller.py"),
    ("rm75_control/control/hybrid_motion/scan_log.py", REPO / "rm75_control/control/hybrid_motion/scan_log.py"),
    ("rm75_control/control/hybrid_motion/loop.py", REPO / "rm75_control/control/hybrid_motion/loop.py"),
    ("rm75_control/control/hybrid_motion/rm_algo.py", REPO / "rm75_control/control/hybrid_motion/rm_algo.py"),
    # CANFD velocity I/O
    ("rm75_control/motion/canfd.py", REPO / "rm75_control/motion/canfd.py"),
]


def lang(path: Path) -> str:
    if path.suffix in (".yaml", ".yml"):
        return "yaml"
    return "python"


def embed(rel: str, path: Path) -> str:
    body = path.read_text(encoding="utf-8")
    return f"## `{rel}`\n\n```{lang(path)}\n{body}```\n\n"


def main() -> None:
    lines = [
        "# hybrid_motion 源码镜像（速度控制力-位置混合）\n",
        "\n",
        "由 `python scripts/gen_debug_va.py` 生成，与仓库文件一字不差。\n",
        "\n",
        "栈概览：\n",
        "- **loop.py** — 10ms CANFD 主循环、hold/scan 相位、日志\n",
        "- **controller.py** — PBAC + sleeve 融合、2阶导纳、Dimeas、带限 v_r\n",
        "- **adaptive_ke.py** — ΔF/Δx EWMA → K̂_e → b_d = 2ζ√(mK̂_e)\n",
        "- **observer.py** — φ 补偿 + 因果 LPF → f_ext\n",
        "- **canfd.py** — movev 速度下发 / quiescence handoff\n",
        "\n",
    ]
    for rel, path in SECTIONS:
        if not path.is_file():
            raise SystemExit(f"missing: {path}")
        lines.append(embed(rel, path))
    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(SECTIONS)} files)")


if __name__ == "__main__":
    main()
