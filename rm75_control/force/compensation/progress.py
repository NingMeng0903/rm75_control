"""In-place progress bar: erase-then-overwrite on /dev/tty."""

from __future__ import annotations

import os
import sys

_last_filled: dict[str, int] = {}
_tty_fd: int | None = None
_tty_tried: bool = False


def _ttyfd() -> int | None:
    global _tty_fd, _tty_tried
    if _tty_tried:
        return _tty_fd
    _tty_tried = True
    try:
        _tty_fd = os.open("/dev/tty", os.O_WRONLY | os.O_NOCTTY)
    except OSError:
        _tty_fd = None
    return _tty_fd


def _write_tty(s: str) -> None:
    fd = _ttyfd()
    if fd is not None:
        os.write(fd, s.encode())


def stage_progress(label: str, step: int, total: int, *, width: int = 36) -> None:
    total = max(int(total), 1)
    step = min(max(int(step), 0), total)
    pct = int(100 * step / total)
    filled = int(width * step / total)
    done = step >= total

    if not done and filled == _last_filled.get(label, -1):
        return
    _last_filled[label] = filled

    bar = "#" * filled + "-" * (width - filled)
    line = f"  {label} [{bar}] {pct:3d}%"

    fd = _ttyfd()
    if fd is not None:
        # \033[2K  erase entire current line
        # \r       go to column 0
        # line     draw bar (no padding needed — line is already erased)
        payload = f"\033[2K\r{line}"
        if done:
            payload += "\n"
        os.write(fd, payload.encode())
    else:
        # No controlling tty (pipe / CI): print at each bar increment.
        print(line, flush=True)
        if not done:
            _last_filled[label] = filled


def close_progress() -> None:
    global _tty_fd, _tty_tried
    _last_filled.clear()
    if _tty_fd is not None:
        try:
            os.close(_tty_fd)
        except OSError:
            pass
        _tty_fd = None
    _tty_tried = False
