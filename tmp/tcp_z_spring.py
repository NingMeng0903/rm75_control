#!/usr/bin/env python3
"""
Tool-frame Y sinusoid + tool-frame Fz constant force (official stream API).

Run:
  source /media/camp/EXT_DRIVE/rm75_control/env.sh
  python /media/camp/EXT_DRIVE/rm75_control/tmp/tcp_z_spring.py \\
    --prepress --trajectory sin_tool_y --z-force 3
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Deque

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
CONFIG = SCRIPT_PATH.parents[1] / "configs" / "rm75f_default.yaml"

STREAM_CONTROL_MODE_FORCE = [3, 3, 4, 3, 3, 3]
STREAM_CONTROL_MODE_POSITION = [3, 3, 3, 3, 3, 3]


def build_stream_control_mode(z_force_n: float) -> list[int]:
    """Z force tracking (mode 4) only when |Fz target| > 0."""
    if abs(z_force_n) < 1e-6:
        return list(STREAM_CONTROL_MODE_POSITION)
    return list(STREAM_CONTROL_MODE_FORCE)


class ForceMonitor:
    """Live plot: tool Fz measured vs desired (display only)."""

    def __init__(
        self,
        target_fz: float,
        *,
        window_s: float = 30.0,
        refresh_hz: float = 10.0,
        invert_meas: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt

        self.target_fz = target_fz
        self.invert_meas = invert_meas
        self.window_s = window_s
        self.refresh_interval = 1.0 / refresh_hz
        self._lock = Lock()
        max_pts = max(int(window_s * refresh_hz) + 10, 100)
        self._t: Deque[float] = deque(maxlen=max_pts)
        self._fz: Deque[float] = deque(maxlen=max_pts)
        self._last_refresh = 0.0

        label = "Fz measured (×-1)" if invert_meas else "Fz measured"
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(9, 4))
        (self._line_meas,) = self._ax.plot([], [], "b-", linewidth=1.5, label=label)
        (self._line_des,) = self._ax.plot([], [], "r--", linewidth=1.2, label="Fz desired")
        self._ax.set_xlabel("Time (s)")
        self._ax.set_ylabel("Tool Fz (N)")
        self._ax.set_title("Tool-frame Fz: desired vs measured")
        self._ax.grid(True, alpha=0.3)
        self._ax.legend(loc="upper right")
        self._fig.tight_layout()
        try:
            self._fig.canvas.manager.set_window_title("RM75 Tool Fz Monitor")
        except Exception:
            pass
        self._fig.show()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def append(self, t_s: float, fz_meas: float) -> None:
        fz_plot = -fz_meas if self.invert_meas else fz_meas
        with self._lock:
            self._t.append(t_s)
            self._fz.append(fz_plot)

    def refresh(self, now: float) -> None:
        if now - self._last_refresh < self.refresh_interval:
            return
        self._last_refresh = now
        with self._lock:
            if not self._t:
                return
            ts = list(self._t)
            fz = list(self._fz)
        t_end = ts[-1]
        t_start = max(0.0, t_end - self.window_s)
        xs = [ts[i] for i, t in enumerate(ts) if t >= t_start]
        ys = [fz[i] for i, t in enumerate(ts) if t >= t_start]
        self._line_meas.set_data(xs, ys)
        if xs:
            self._line_des.set_data([xs[0], xs[-1]], [self.target_fz, self.target_fz])
        self._ax.set_xlim(t_start, max(t_end, t_start + 1.0))
        y_vals = ys + [self.target_fz]
        y_min, y_max = min(y_vals) - 1.0, max(y_vals) + 1.0
        if y_max - y_min < 2.0:
            mid = 0.5 * (y_max + y_min)
            y_min, y_max = mid - 1.0, mid + 1.0
        self._ax.set_ylim(y_min, y_max)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self._fig)
        plt.ioff()


def open_force_monitor(
    target_fz: float, window_s: float, *, invert_meas: bool = True
) -> ForceMonitor | None:
    try:
        return ForceMonitor(target_fz, window_s=window_s, invert_meas=invert_meas)
    except Exception as exc:
        print(f"Force monitor disabled: {exc}", flush=True)
        return None


class LagMonitor:
    """Live plot: raw tool Y fb vs cmd vs sin fit (same sample rate as lag fit)."""

    def __init__(
        self,
        *,
        y0_mm: float,
        window_s: float = 30.0,
        refresh_hz: float = 10.0,
    ) -> None:
        import matplotlib.pyplot as plt

        self.y0_mm = y0_mm
        self.window_s = window_s
        self.refresh_interval = 1.0 / refresh_hz
        self._last_refresh = 0.0
        self._fit: SinLagFit | None = None

        plt.ion()
        self._fig, (self._ax_abs, self._ax_dy, self._ax_err) = plt.subplots(
            3, 1, figsize=(10, 8), sharex=True
        )
        (self._line_cmd,) = self._ax_abs.plot(
            [], [], "C0--", linewidth=1.2, label="Y cmd"
        )
        (self._line_fb,) = self._ax_abs.plot(
            [], [], "C2-", linewidth=1.8, label="Y fb raw (robot)"
        )
        (self._line_fit,) = self._ax_abs.plot(
            [], [], "C1:", linewidth=1.6, label="Y fb sin fit"
        )
        self._ax_abs.set_ylabel("tool Y (mm)")
        self._ax_abs.set_title("Absolute tool Y: cmd vs raw fb vs sin fit")
        self._ax_abs.grid(True, alpha=0.3)
        self._ax_abs.legend(loc="upper right", fontsize=8)

        (self._line_dy_cmd,) = self._ax_dy.plot(
            [], [], "C0--", linewidth=1.2, label="ΔY cmd"
        )
        (self._line_dy_fb,) = self._ax_dy.plot(
            [], [], "C2-", linewidth=1.8, label="ΔY fb raw"
        )
        (self._line_dy_fit,) = self._ax_dy.plot(
            [], [], "C1:", linewidth=1.6, label="ΔY sin fit"
        )
        self._ax_dy.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
        self._ax_dy.set_ylabel("ΔY from y₀ (mm)")
        self._ax_dy.set_title("Offset from start y₀ (amplitude check)")
        self._ax_dy.grid(True, alpha=0.3)
        self._ax_dy.legend(loc="upper right", fontsize=8)

        (self._line_err,) = self._ax_err.plot(
            [], [], "C3-", linewidth=1.2, label="err = cmd − fb raw"
        )
        self._ax_err.axhline(0.0, color="k", linewidth=0.6, alpha=0.4)
        self._ax_err.set_xlabel("Time (s)")
        self._ax_err.set_ylabel("error (mm)")
        self._ax_err.set_title("Tracking error")
        self._ax_err.grid(True, alpha=0.3)
        self._ax_err.legend(loc="upper right", fontsize=8)
        self._fig.tight_layout()
        try:
            self._fig.canvas.manager.set_window_title("RM75 Tool Y Lag Monitor")
        except Exception:
            pass
        self._fig.show()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def update_from_estimator(
        self,
        estimator: SinLagEstimator,
        fit: SinLagFit | None,
    ) -> None:
        snap = estimator.snapshot(self.window_s)
        if snap is None:
            return
        self._fit = fit
        t_s = snap["t"]
        y_cmd = snap["y_cmd"]
        y_fb = snap["y_fb"]
        y_fit = snap["y_fit"]
        dy_cmd = [y - self.y0_mm for y in y_cmd]
        dy_fb = [y - self.y0_mm for y in y_fb]
        dy_fit = [y - self.y0_mm for y in y_fit]
        err = [c - f for c, f in zip(y_cmd, y_fb)]

        self._line_cmd.set_data(t_s, y_cmd)
        self._line_fb.set_data(t_s, y_fb)
        self._line_fit.set_data(t_s, y_fit)
        self._line_dy_cmd.set_data(t_s, dy_cmd)
        self._line_dy_fb.set_data(t_s, dy_fb)
        self._line_dy_fit.set_data(t_s, dy_fit)
        self._line_err.set_data(t_s, err)

        t_end = t_s[-1]
        t_start = max(0.0, t_end - self.window_s)
        for ax in (self._ax_abs, self._ax_dy, self._ax_err):
            ax.set_xlim(t_start, max(t_end, t_start + 1.0))

        if fit is not None:
            amp_line = (
                f"amp: cmd={fit.amp_cmd_nom_mm:.0f}mm "
                f"fb_peak={fit.amp_fb_peak_mm:.1f}mm "
                f"fb_fit={fit.amp_fb_fit_mm:.1f}mm "
                f"lag={fit.lag_ms:+.0f}ms ({fit.lag_deg:+.1f}°) "
                f"rmse={fit.rmse_mm:.2f}mm"
            )
            self._ax_abs.set_title(f"Absolute tool Y | {amp_line}")
            self._ax_dy.set_title(
                f"ΔY from y₀={self.y0_mm:.1f}mm | "
                f"peak raw={fit.amp_fb_peak_mm:.1f} vs fit={fit.amp_fb_fit_mm:.1f} "
                f"vs cmd={fit.amp_cmd_nom_mm:.0f}"
            )

        abs_vals = list(y_cmd) + list(y_fb) + list(y_fit)
        if abs_vals:
            y_min, y_max = min(abs_vals), max(abs_vals)
            pad = max(2.0, (y_max - y_min) * 0.08)
            self._ax_abs.set_ylim(y_min - pad, y_max + pad)
        dy_vals = list(dy_cmd) + list(dy_fb) + list(dy_fit)
        if dy_vals:
            d_max = max(abs(v) for v in dy_vals)
            d_lim = max(5.0, d_max * 1.12)
            self._ax_dy.set_ylim(-d_lim, d_lim)
        if err:
            e_max = max(abs(v) for v in err)
            e_lim = max(1.0, e_max * 1.12)
            self._ax_err.set_ylim(-e_lim, e_lim)

    def refresh(self, now: float) -> None:
        if now - self._last_refresh < self.refresh_interval:
            return
        self._last_refresh = now
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self._fig)
        plt.ioff()


def open_lag_monitor(window_s: float, y0_mm: float, *, refresh_hz: float = 5.0) -> LagMonitor | None:
    try:
        return LagMonitor(window_s=window_s, y0_mm=y0_mm, refresh_hz=refresh_hz)
    except Exception as exc:
        print(f"Lag monitor disabled: {exc}", flush=True)
        return None


@dataclass(frozen=True)
class SinLagFit:
    """Least-squares fit: y_fb ≈ offset + amp·sin(ωt − φ); φ>0 ⇒ fb lags cmd."""

    lag_ms: float
    lag_deg: float
    amp_fb_fit_mm: float
    amp_fb_peak_mm: float
    amp_cmd_peak_mm: float
    amp_cmd_nom_mm: float
    amp_ratio_fit: float
    amp_ratio_peak: float
    offset_mm: float
    rmse_mm: float
    mean_abs_err_mm: float
    peak_err_theory_mm: float
    n_samples: int
    span_s: float


class SinLagEstimator:
    """Rolling sin fit on measured tool Y to estimate phase/time lag vs command."""

    def __init__(
        self,
        omega: float,
        cmd_amp_mm: float,
        *,
        window_s: float,
        min_cycles: float = 1.5,
    ) -> None:
        self.omega = omega
        self.cmd_amp_mm = cmd_amp_mm
        self.window_s = window_s
        self.min_span_s = (
            (2.0 * math.pi / omega) * min_cycles if omega > 0.0 else window_s
        )
        max_pts = max(int(window_s * 120.0) + 20, 200)
        self._t: Deque[float] = deque(maxlen=max_pts)
        self._y_cmd: Deque[float] = deque(maxlen=max_pts)
        self._y_fb: Deque[float] = deque(maxlen=max_pts)

    def append(self, t_s: float, y_cmd_mm: float, y_fb_mm: float) -> None:
        self._t.append(t_s)
        self._y_cmd.append(y_cmd_mm)
        self._y_fb.append(y_fb_mm)
        t_cut = t_s - self.window_s
        while self._t and self._t[0] < t_cut:
            self._t.popleft()
            self._y_cmd.popleft()
            self._y_fb.popleft()

    def fit(self) -> SinLagFit | None:
        if self.omega <= 0.0 or len(self._t) < 20:
            return None
        t = np.asarray(self._t, dtype=float)
        y_fb = np.asarray(self._y_fb, dtype=float)
        y_cmd = np.asarray(self._y_cmd, dtype=float)
        span = float(t[-1] - t[0])
        if span < self.min_span_s:
            return None

        wt = self.omega * t
        x = np.column_stack([np.sin(wt), np.cos(wt), np.ones_like(t)])
        coef, _, _, _ = np.linalg.lstsq(x, y_fb, rcond=None)
        a, b, offset = coef
        amp_fb_fit = float(np.hypot(a, b))
        amp_fb_peak = 0.5 * float(np.max(y_fb) - np.min(y_fb))
        amp_cmd_peak = 0.5 * float(np.max(y_cmd) - np.min(y_cmd))
        # y_fb = offset + amp_fb·sin(ωt − φ),  a=amp·cos φ, b=−amp·sin φ
        phi_rad = float(math.atan2(-b, a))
        lag_s = phi_rad / self.omega
        lag_ms = lag_s * 1000.0
        lag_deg = math.degrees(phi_rad)

        y_pred = x @ coef
        rmse = float(np.sqrt(np.mean((y_fb - y_pred) ** 2)))
        err = y_cmd - y_fb
        mean_abs_err = float(np.mean(np.abs(err)))
        peak_err_theory = self.cmd_amp_mm * 2.0 * math.sin(abs(phi_rad) / 2.0)
        amp_ratio_fit = (
            amp_fb_fit / self.cmd_amp_mm if self.cmd_amp_mm > 0.0 else float("nan")
        )
        amp_ratio_peak = (
            amp_fb_peak / self.cmd_amp_mm if self.cmd_amp_mm > 0.0 else float("nan")
        )

        return SinLagFit(
            lag_ms=lag_ms,
            lag_deg=lag_deg,
            amp_fb_fit_mm=amp_fb_fit,
            amp_fb_peak_mm=amp_fb_peak,
            amp_cmd_peak_mm=amp_cmd_peak,
            amp_cmd_nom_mm=self.cmd_amp_mm,
            amp_ratio_fit=amp_ratio_fit,
            amp_ratio_peak=amp_ratio_peak,
            offset_mm=float(offset),
            rmse_mm=rmse,
            mean_abs_err_mm=mean_abs_err,
            peak_err_theory_mm=peak_err_theory,
            n_samples=len(t),
            span_s=span,
        )

    def _lstsq_coef(self) -> tuple[np.ndarray, np.ndarray, float, float, float] | None:
        if self.omega <= 0.0 or len(self._t) < 20:
            return None
        t = np.asarray(self._t, dtype=float)
        y_fb = np.asarray(self._y_fb, dtype=float)
        wt = self.omega * t
        x = np.column_stack([np.sin(wt), np.cos(wt), np.ones_like(t)])
        coef, _, _, _ = np.linalg.lstsq(x, y_fb, rcond=None)
        a, b, offset = coef
        amp_fb = float(np.hypot(a, b))
        phi_rad = float(math.atan2(-b, a))
        return t, y_fb, float(offset), amp_fb, phi_rad

    def snapshot(self, plot_window_s: float) -> dict[str, list[float]] | None:
        coef_data = self._lstsq_coef()
        if coef_data is None:
            return None
        t, y_fb, offset, amp_fb, phi_rad = coef_data
        t_end = float(t[-1])
        t_start = max(0.0, t_end - plot_window_s)
        mask = t >= t_start
        if not np.any(mask):
            return None
        t_w = t[mask]
        wt = self.omega * t_w
        y_fit = offset + amp_fb * np.sin(wt - phi_rad)
        y_cmd = np.asarray(self._y_cmd, dtype=float)[mask]
        return {
            "t": t_w.tolist(),
            "y_cmd": y_cmd.tolist(),
            "y_fb": y_fb[mask].tolist(),
            "y_fit": y_fit.tolist(),
        }


def format_sin_lag_fit(fit: SinLagFit) -> str:
    return (
        f"sin_fit lag={fit.lag_ms:+.0f}ms ({fit.lag_deg:+.1f}°) "
        f"amp_peak={fit.amp_fb_peak_mm:.1f}mm amp_fit={fit.amp_fb_fit_mm:.1f}mm "
        f"cmd_nom={fit.amp_cmd_nom_mm:.0f}mm cmd_peak={fit.amp_cmd_peak_mm:.1f}mm "
        f"ratio_peak={fit.amp_ratio_peak:.3f} ratio_fit={fit.amp_ratio_fit:.3f} "
        f"rmse={fit.rmse_mm:.2f}mm |err|={fit.mean_abs_err_mm:.2f}mm "
        f"n={fit.n_samples} span={fit.span_s:.1f}s"
    )


def read_tool_fz(robot) -> float | None:
    ret_f, fdata = robot.rm_get_force_data()
    if ret_f != 0:
        return None
    return float(fdata["tool_zero_force_data"][2])


def pose_to_rm_pose(pose: list[float]):
    from Robotic_Arm.rm_ctypes_wrap import rm_euler_t, rm_pose_t, rm_position_t

    po = rm_pose_t()
    po.position = rm_position_t(*pose[:3])
    po.euler = rm_euler_t(*pose[3:6])
    return po


def tool_frame_offset_pose(
    robot, ref_pose: list[float], dx: float, dy: float, dz: float
) -> list[float]:
    delta = [dx, dy, dz, 0.0, 0.0, 0.0]
    return robot.rm_algo_pose_move(ref_pose, delta, frameMode=1)


def build_desired_force(z_force_n: float) -> list[float]:
    f = [0.0] * 6
    f[2] = z_force_n
    return f


def sin_period_for_peak_vel(amplitude_m: float, max_vel_m_s: float) -> float:
    if amplitude_m <= 0.0 or max_vel_m_s <= 0.0:
        return 1.0
    return 2.0 * math.pi * amplitude_m / max_vel_m_s


def run_movel_hybrid(
    robot,
    pose: list[float],
    *,
    z_force: float,
    tool_mode: int,
    speed: int,
) -> None:
    ret = robot.rm_set_force_position(1, tool_mode, 2, z_force)
    if ret != 0:
        raise RuntimeError(f"rm_set_force_position failed: {ret}")
    ret = robot.rm_movel(pose, speed, 0, 0, 1)
    if ret != 0:
        raise RuntimeError(f"rm_movel failed: {ret}")
    time.sleep(2.0)
    ret = robot.rm_stop_force_position()
    if ret != 0:
        raise RuntimeError(f"rm_stop_force_position failed: {ret}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tool Y sin + tool Fz force-position hybrid scan",
        epilog=f"Example: python {SCRIPT_PATH} --prepress --trajectory sin_tool_y --z-force 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--z-force", type=float, default=3.0, help="tool Fz (N)")
    parser.add_argument(
        "--tool-mode", type=int, default=1, choices=[0, 1], help="0=base, 1=tool"
    )
    parser.add_argument(
        "--prepress",
        action="store_true",
        help="rm_set_force_position + rm_movel along tool Z",
    )
    parser.add_argument("--prepress-dz-mm", type=float, default=30.0)
    parser.add_argument("--movel-speed", type=int, default=20)
    parser.add_argument(
        "--trajectory",
        choices=("hold", "sin_tool_y", "sin_y"),
        default="hold",
    )
    parser.add_argument(
        "--amplitude-mm",
        type=float,
        default=50.0,
        help="peak tool Y offset ±mm (default 50 = ±5cm)",
    )
    parser.add_argument(
        "--y-max-vel-cm-s",
        type=float,
        default=1.5,
        help="peak tool Y speed (cm/s); auto period if --period omitted",
    )
    parser.add_argument("--period", type=float, default=None)
    parser.add_argument("--dt-ms", type=float, default=10.0)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument(
        "--plot-every",
        type=int,
        default=10,
        help="sample Fz for plot every N cycles (10=100ms)",
    )
    parser.add_argument("--plot-window-s", type=float, default=30.0)
    parser.add_argument(
        "--no-plot-invert",
        action="store_true",
        help="plot raw sensor Fz (default: ×-1 for display)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="disable Fz and lag plot windows",
    )
    parser.add_argument(
        "--no-lag-plot",
        action="store_true",
        help="disable lag plot only (Fz plot still on unless --no-plot)",
    )
    parser.add_argument(
        "--lag-plot-every",
        type=int,
        default=10,
        help="(deprecated) plot now follows --lag-fit-every at 10ms fb rate",
    )
    parser.add_argument(
        "--lag-fit-every",
        type=int,
        default=10,
        help="sample tool Y every N cycles (10=100ms; use >=10 with --follow)",
    )
    parser.add_argument(
        "--lag-fit-compute-every",
        type=int,
        default=50,
        help="recompute sin fit every N cycles (50=500ms; keep >>1 with --follow)",
    )
    parser.add_argument(
        "--lag-plot-hz",
        type=float,
        default=5.0,
        help="max lag plot refresh rate (Hz)",
    )
    parser.add_argument(
        "--lag-fit-window-s",
        type=float,
        default=None,
        help="rolling window for sin fit (default max(2*period, 15s))",
    )
    parser.add_argument(
        "--no-lag-fit",
        action="store_true",
        help="disable sin fit on measured tool Y",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="high follow (default off; better Y amplitude, may jitter Fz in contact)",
    )
    parser.add_argument(
        "--trajectory-mode",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="only when --follow: 0=passthrough, 1=curve fit, 2=filter",
    )
    parser.add_argument(
        "--radio",
        type=int,
        default=0,
        help="only when --follow: smooth factor (mode1: 0-100, mode2: 0-1000)",
    )
    args = parser.parse_args()

    dt_s = args.dt_ms / 1000.0
    amplitude_m = args.amplitude_mm / 1000.0
    max_vel_m_s = args.y_max_vel_cm_s / 100.0
    use_sin = args.trajectory in ("sin_tool_y", "sin_y")

    if use_sin and args.period is None:
        period = sin_period_for_peak_vel(amplitude_m, max_vel_m_s)
    elif use_sin:
        period = args.period
        v_peak = amplitude_m * (2.0 * math.pi / period)
        if v_peak > max_vel_m_s * 1.001:
            period = sin_period_for_peak_vel(amplitude_m, max_vel_m_s)
            print(
                f"period extended to {period:.2f}s to cap tool Y at "
                f"{args.y_max_vel_cm_s:.2f} cm/s",
                flush=True,
            )
    else:
        period = args.period or 6.0

    omega = 2.0 * math.pi / period if use_sin else 0.0
    v_peak = amplitude_m * omega if use_sin else 0.0
    desired_force = build_desired_force(args.z_force)
    stream_control_mode = build_stream_control_mode(args.z_force)
    limit_vel = [0.1, max_vel_m_s, 0.1, 10.0, 10.0, 10.0]

    monitor: ForceMonitor | None = None
    lag_monitor: LagMonitor | None = None
    lag_estimator: SinLagEstimator | None = None
    last_lag_fit: SinLagFit | None = None
    if not args.no_plot:
        monitor = open_force_monitor(
            args.z_force,
            args.plot_window_s,
            invert_meas=not args.no_plot_invert,
        )
        if monitor is not None:
            print("Realtime Fz monitor opened (blue = sensor × -1).", flush=True)
    if use_sin and not args.no_lag_fit:
        fit_window_s = args.lag_fit_window_s
        if fit_window_s is None:
            fit_window_s = max(2.0 * period, 15.0)
        lag_estimator = SinLagEstimator(
            omega,
            args.amplitude_mm,
            window_s=fit_window_s,
        )
        print(
            f"Sin lag fit: window={fit_window_s:.1f}s sample every "
            f"{args.lag_fit_every} cycle(s) ({args.lag_fit_every * args.dt_ms:.0f}ms) "
            f"compute every {args.lag_fit_compute_every} plot {args.lag_plot_hz:.0f}Hz",
            flush=True,
        )
        if args.follow and args.lag_fit_every < 10:
            print(
                "WARN: --follow needs steady pose stream; "
                "use --lag-fit-every 10 (or --no-lag-plot) to avoid blocking.",
                flush=True,
            )

    from rm75_control import RobotSession

    print("Connecting...", flush=True)
    with RobotSession(config=CONFIG) as bot:
        ret, state = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            print(f"get state failed: {ret}", file=sys.stderr)
            if monitor is not None:
                monitor.close()
            if lag_monitor is not None:
                lag_monitor.close()
            return 1

        pose0 = list(state["pose"])
        pose0_tool = bot.robot.rm_algo_end2tool(pose_to_rm_pose(pose0))
        y0_tool = pose0_tool[1]
        y0_mm = y0_tool * 1000.0

        if not args.no_plot and not args.no_lag_plot and use_sin:
            lag_monitor = open_lag_monitor(
                args.plot_window_s, y0_mm, refresh_hz=args.lag_plot_hz
            )
            if lag_monitor is not None:
                print(
                    "Lag monitor: raw fb (10ms) + sin fit + ΔY amplitude panel.",
                    flush=True,
                )

        print("start pose (base, m rad):", [round(v, 6) for v in pose0])
        print("start pose (tool, m rad):", [round(v, 6) for v in pose0_tool])
        print(
            f"stream: flag=1 pose control_mode={stream_control_mode} "
            f"desired_force={desired_force} mode={args.tool_mode} follow={args.follow} "
            f"limit_vel={limit_vel} traj={args.trajectory_mode if args.follow else 0} "
            f"radio={args.radio if args.follow else 0}",
            flush=True,
        )
        if use_sin:
            print(
                f"tool Y sin: amp={args.amplitude_mm:.1f}mm period={period:.2f}s "
                f"peak_vel={v_peak*100:.2f}cm/s (cap {args.y_max_vel_cm_s:.2f}cm/s)",
                flush=True,
            )

        if args.prepress:
            dz_m = args.prepress_dz_mm / 1000.0
            press_dz = dz_m if args.z_force >= 0.0 else -dz_m
            press_pose = tool_frame_offset_pose(bot.robot, pose0, 0.0, 0.0, press_dz)
            print(
                f"prepress: rm_set_force_position(1,{args.tool_mode},2,{args.z_force}) "
                f"+ rm_movel tool_Z {press_dz*1000:+.1f}mm",
                flush=True,
            )
            run_movel_hybrid(
                bot.robot,
                press_pose,
                z_force=args.z_force,
                tool_mode=args.tool_mode,
                speed=args.movel_speed,
            )

        fc = bot.start_force_scan(
            flag=1,
            mode=args.tool_mode,
            control_mode=stream_control_mode,
            desired_force=desired_force,
            limit_vel=limit_vel,
            follow=args.follow,
            trajectory_mode=args.trajectory_mode if args.follow else 0,
            radio=args.radio if args.follow else 0,
        )

        t_start = time.monotonic()
        next_tick = t_start
        cmd_count = 0
        last_fz: float | None = None

        last_y_cmd_mm = y0_tool * 1000.0
        last_y_fb_mm = y0_tool * 1000.0

        print("Stream started (pose @ 100Hz). Ctrl+C to stop.", flush=True)

        try:
            while True:
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)
                next_tick += dt_s
                t_now = now - t_start
                ret_s = -1
                st = None

                dy_tool = amplitude_m * math.sin(omega * t_now) if use_sin else 0.0
                y_tool_cmd_mm = (y0_tool + dy_tool) * 1000.0
                cmd_pose = tool_frame_offset_pose(bot.robot, pose0, 0.0, dy_tool, 0.0)
                fc.step_pose(cmd_pose)
                cmd_count += 1
                last_y_cmd_mm = y_tool_cmd_mm

                sample_plot = cmd_count % args.plot_every == 0
                sample_state = (
                    lag_estimator is not None
                    and cmd_count % args.lag_fit_every == 0
                )
                sample_fit = (
                    lag_estimator is not None
                    and cmd_count % args.lag_fit_compute_every == 0
                )
                if sample_state:
                    ret_s, st = bot.robot.rm_get_current_arm_state()
                    if ret_s == 0:
                        fb_tool = bot.robot.rm_algo_end2tool(
                            pose_to_rm_pose(list(st["pose"]))
                        )
                        last_y_fb_mm = fb_tool[1] * 1000.0
                        lag_estimator.append(
                            t_now, last_y_cmd_mm, last_y_fb_mm
                        )

                if sample_fit:
                    fit = lag_estimator.fit()
                    if fit is not None:
                        last_lag_fit = fit
                    if lag_monitor is not None:
                        lag_monitor.update_from_estimator(
                            lag_estimator, last_lag_fit
                        )

                if lag_monitor is not None:
                    lag_monitor.refresh(now)

                if monitor is not None and sample_plot:
                    fz = read_tool_fz(bot.robot)
                    if fz is not None:
                        last_fz = fz
                        monitor.append(t_now, fz)
                        monitor.refresh(now)

                if cmd_count % args.report_every == 0:
                    y_tool_fb = last_y_fb_mm
                    z_fb = float("nan")
                    if not sample_state:
                        ret_s, st = bot.robot.rm_get_current_arm_state()
                        if ret_s == 0:
                            fb_tool = bot.robot.rm_algo_end2tool(
                                pose_to_rm_pose(list(st["pose"]))
                            )
                            y_tool_fb = fb_tool[1] * 1000.0
                            last_y_fb_mm = y_tool_fb
                            z_fb = st["pose"][2] * 1000.0
                    elif ret_s == 0:
                        z_fb = st["pose"][2] * 1000.0
                    fz = last_fz if last_fz is not None else float("nan")
                    y_err_mm = last_y_cmd_mm - y_tool_fb
                    fz_disp = -fz if not math.isnan(fz) else float("nan")
                    line = (
                        f"t={t_now:.1f}s tool_y_cmd={last_y_cmd_mm:.1f} "
                        f"tool_y_fb={y_tool_fb:.1f} y_err={y_err_mm:+.2f}mm "
                        f"base_z_fb={z_fb:.1f} "
                        f"tool_Fz={fz:.2f} Fz_disp={fz_disp:+.2f} "
                        f"target={args.z_force:+.2f}"
                    )
                    if last_lag_fit is not None:
                        line += f" | {format_sin_lag_fit(last_lag_fit)}"
                    print(line, flush=True)
        except KeyboardInterrupt:
            print("\nCtrl+C stopping...", flush=True)
        finally:
            bot.stop_all()
            if monitor is not None:
                monitor.close()
            if lag_monitor is not None:
                lag_monitor.close()

    print("stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
