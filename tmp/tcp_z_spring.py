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
from pathlib import Path
from threading import Lock
from typing import Deque

SCRIPT_PATH = Path(__file__).resolve()
CONFIG = SCRIPT_PATH.parents[1] / "configs" / "rm75f_default.yaml"

STREAM_CONTROL_MODE = [3, 3, 4, 3, 3, 3]


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
    parser.add_argument("--no-plot", action="store_true")
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
    limit_vel = [0.1, max_vel_m_s, 0.1, 10.0, 10.0, 10.0]

    monitor: ForceMonitor | None = None
    if not args.no_plot:
        monitor = open_force_monitor(
            args.z_force,
            args.plot_window_s,
            invert_meas=not args.no_plot_invert,
        )
        if monitor is not None:
            print("Realtime Fz monitor opened (blue = sensor × -1).", flush=True)

    from rm75_control import RobotSession

    print("Connecting...", flush=True)
    with RobotSession(config=CONFIG) as bot:
        ret, state = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            print(f"get state failed: {ret}", file=sys.stderr)
            if monitor is not None:
                monitor.close()
            return 1

        pose0 = list(state["pose"])
        pose0_tool = bot.robot.rm_algo_end2tool(pose_to_rm_pose(pose0))
        y0_tool = pose0_tool[1]

        print("start pose (base, m rad):", [round(v, 6) for v in pose0])
        print("start pose (tool, m rad):", [round(v, 6) for v in pose0_tool])
        print(
            f"stream: flag=1 pose control_mode={STREAM_CONTROL_MODE} "
            f"desired_force={desired_force} mode={args.tool_mode} follow=True "
            f"limit_vel={limit_vel}",
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
            control_mode=STREAM_CONTROL_MODE,
            desired_force=desired_force,
            limit_vel=limit_vel,
            follow=True,
            trajectory_mode=0,
            radio=0,
        )

        t_start = time.monotonic()
        next_tick = t_start
        cmd_count = 0
        last_fz: float | None = None

        try:
            while True:
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)
                next_tick += dt_s
                t_now = now - t_start

                dy_tool = amplitude_m * math.sin(omega * t_now) if use_sin else 0.0
                cmd_pose = tool_frame_offset_pose(bot.robot, pose0, 0.0, dy_tool, 0.0)
                fc.step_pose(cmd_pose)
                cmd_count += 1

                if monitor is not None and cmd_count % args.plot_every == 0:
                    fz = read_tool_fz(bot.robot)
                    if fz is not None:
                        last_fz = fz
                        monitor.append(t_now, fz)
                    monitor.refresh(now)

                if cmd_count % args.report_every == 0:
                    ret_s, st = bot.robot.rm_get_current_arm_state()
                    y_tool_fb = float("nan")
                    z_fb = float("nan")
                    if ret_s == 0:
                        fb_tool = bot.robot.rm_algo_end2tool(
                            pose_to_rm_pose(list(st["pose"]))
                        )
                        y_tool_fb = fb_tool[1] * 1000.0
                        z_fb = st["pose"][2] * 1000.0
                    fz = last_fz if last_fz is not None else float("nan")
                    y_tool_cmd = (y0_tool + dy_tool) * 1000.0
                    fz_disp = -fz if not math.isnan(fz) else float("nan")
                    print(
                        f"t={t_now:.1f}s tool_y_cmd={y_tool_cmd:.1f} "
                        f"tool_y_fb={y_tool_fb:.1f} base_z_fb={z_fb:.1f} "
                        f"tool_Fz={fz:.2f} Fz_disp={fz_disp:+.2f} target={args.z_force:+.2f}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\nCtrl+C stopping...", flush=True)
        finally:
            bot.stop_all()
            if monitor is not None:
                monitor.close()

    print("stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
