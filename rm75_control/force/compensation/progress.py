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
