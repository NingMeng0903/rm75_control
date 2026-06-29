#!/usr/bin/env python3
"""Regenerate MD/debug.md — verbatim source mirror only."""

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
    return f"## `{rel}`\n\n```{lang(path)}\n{body}```\n\n"


def main() -> None:
    lines = [
        "# velocity_admittance 源码镜像\n",
        "\n",
        "由 `python scripts/gen_debug_va.py` 生成，与仓库文件一字不差。\n",
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
