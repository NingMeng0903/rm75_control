#!/usr/bin/env python3
"""
Live 6D force compensation monitor — drag the arm manually, watch raw vs F_ext.

Uses φ from force_id_phi.json (phi_recommended by default) and the same
regressor as force_id_fit.py. Keeps a rolling pose/force buffer for ω/α.

Run:
  source env.sh
  python tmp/force_id/force_id_comp_monitor.py
  python tmp/force_id/force_id_comp_monitor.py --window-s 30 --poll-ms 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import force_id_comp_demo as fid  # noqa: E402
from _paths import CONFIG_FORCE, CONFIG_ROBOT, PHI_JSON  # noqa: E402

AXIS_LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
FORCE_IDX = (0, 1, 2)
MOM_IDX = (3, 4, 5)


def load_phi(path: Path, source: str | None) -> tuple[np.ndarray, str]:
    data = json.loads(path.read_text())
    if source is None:
        source = "phi_recommended" if "phi_recommended" in data else "phi_16"
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
    f_model_in = Y[sl].copy()
    f_ext = (Y[sl] - W[sl] @ phi).reshape(6)
    return f_model_in, f_ext


class CompMonitor:
    """Rolling plot: signed raw (filtered, model input) vs compensated F_ext."""

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
        self._fig.suptitle("6D force: model input (raw signed, filtered) vs compensated F_ext")
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
            rf = float(
                np.sqrt(
                    np.nanmean(
                        np.sum(np.stack([raw_pts[j] for j in FORCE_IDX], axis=1) ** 2, axis=1)
                    )
                )
            )
            ef = np.stack([ext_pts[j] for j in FORCE_IDX], axis=1)
            ef_ok = np.isfinite(ef).all(axis=1)
            if np.any(ef_ok):
                re = float(np.sqrt(np.mean(np.sum(ef[ef_ok] ** 2, axis=1))))
                em = np.stack([ext_pts[j] for j in MOM_IDX], axis=1)
                em_ok = np.isfinite(em).all(axis=1)
                rm = float(np.sqrt(np.mean(np.sum(em[em_ok] ** 2, axis=1)))) if np.any(em_ok) else float("nan")
                status = (
                    f"{status}  |  window |F| raw={rf:.2f}N ext={re:.2f}N  |M| ext={rm:.3f}N·m"
                )
            else:
                status = f"{status}  |  window |F| raw={rf:.2f}N  (F_ext warming up)"
        self._text.set_text(status)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self._fig)
        plt.ioff()


def main() -> int:
    parser = argparse.ArgumentParser(description="Live 6D compensated force plot (manual motion)")
    parser.add_argument("--phi", type=Path, default=PHI_JSON)
    parser.add_argument("--phi-source", type=str, default=None)
    parser.add_argument("--config", type=Path, default=CONFIG_FORCE)
    parser.add_argument("--fc", type=float, default=None)
    parser.add_argument("--poll-ms", type=float, default=50.0, help="robot poll period")
    parser.add_argument("--window-s", type=float, default=25.0, help="plot time window")
    parser.add_argument("--buffer-s", type=float, default=4.0, help="kinematics history for ω/α")
    parser.add_argument("--min-samples", type=int, default=35, help="min buffer before F_ext")
    parser.add_argument("--10p-only", dest="only_10p", action="store_true", help="zero I terms in φ")
    parser.add_argument("--refresh-hz", type=float, default=12.0)
    args = parser.parse_args()

    phi, src = load_phi(args.phi, args.phi_source)
    if args.only_10p:
        phi = phi.copy()
        phi[4:10] = 0.0
        src = f"{src} (I=0)"

    cfg = fid.FrameConfig.from_yaml(args.config)
    fc = args.fc
    if fc is None:
        fc = float(yaml.safe_load(args.config.read_text()).get("filtfilt_cutoff_hz", 2.5))

    max_buf = max(args.min_samples + 10, int(args.buffer_s * 1000 / args.poll_ms) + 5)
    buf = SampleBuffer(max_len=max_buf)
    use_inertia = float(np.max(np.abs(phi[4:10]))) > 1e-9

    print(f"φ source: {src}  m={phi[0]:.3f} kg  inertia={'yes' if use_inertia else 'no'}")
    print(f"poll={args.poll_ms}ms  buffer≈{args.buffer_s}s  plot window={args.window_s}s")
    print("Drag the arm in FREE SPACE. Close plot window or Ctrl+C to stop.")

    from rm75_control import RobotSession

    monitor = CompMonitor(window_s=args.window_s, refresh_hz=args.refresh_hz)
    dt_s = args.poll_ms / 1000.0
    sign = np.array(cfg.force_sign, dtype=float)

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
                    monitor.set_status(f"API error state={ret_s} force={ret_f}")
                    monitor.refresh(now)
                    continue

                pose = np.asarray(st["pose"][:6], dtype=float)
                force = np.asarray(fd["force_data"][:6], dtype=float)
                buf.append(t_s, pose, force)

                comp = compensate_latest(
                    buf, phi, cfg, fc, use_inertia=use_inertia, min_samples=args.min_samples
                )
                if comp is None:
                    raw_show = force * sign
                    monitor.set_status(f"buffer {len(buf.t)}/{args.min_samples}")
                    monitor.append(t_s, raw_show, None)
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
