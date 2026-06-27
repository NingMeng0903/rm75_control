#!/usr/bin/env python3
"""
TCP Y sin loop (direct CANFD) + lag monitoring. Ctrl+C to stop.

Lag is sampled on a decimated schedule so rm_get_current_arm_state (~40ms RTT)
does not stall the 10ms CANFD stream.

Example:
  python tmp/sin_y_cartesian.py --no-ruckig
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rm75f_default.yaml"


@dataclass
class LagStats:
    n: int = 0
    sum_ref_mm: float = 0.0
    sum_cmd_mm: float = 0.0
    sum_sq_ref_mm: float = 0.0
    max_ref_mm: float = 0.0
    max_cmd_mm: float = 0.0
    sum_jitter_ms: float = 0.0
    max_jitter_ms: float = 0.0
    sum_state_rtt_ms: float = 0.0

    def update(
        self,
        y_ref: float,
        y_cmd: float,
        y_fb: float,
        jitter_ms: float,
        state_rtt_ms: float,
    ) -> None:
        ref_e = (y_ref - y_fb) * 1000.0
        cmd_e = (y_cmd - y_fb) * 1000.0
        self.n += 1
        self.sum_ref_mm += abs(ref_e)
        self.sum_cmd_mm += abs(cmd_e)
        self.sum_sq_ref_mm += ref_e * ref_e
        self.max_ref_mm = max(self.max_ref_mm, abs(ref_e))
        self.max_cmd_mm = max(self.max_cmd_mm, abs(cmd_e))
        self.sum_jitter_ms += abs(jitter_ms)
        self.max_jitter_ms = max(self.max_jitter_ms, abs(jitter_ms))
        self.sum_state_rtt_ms += state_rtt_ms

    def report(self, t: float, y_fb_mm: float) -> str:
        if self.n == 0:
            return ""
        mean_ref = self.sum_ref_mm / self.n
        mean_cmd = self.sum_cmd_mm / self.n
        rms_ref = math.sqrt(self.sum_sq_ref_mm / self.n)
        mean_jit = self.sum_jitter_ms / self.n
        mean_rtt = self.sum_state_rtt_ms / self.n
        return (
            f"t={t:.1f}s y_fb={y_fb_mm:.1f}mm | "
            f"ref→fb: mean={mean_ref:.2f} max={self.max_ref_mm:.2f} rms={rms_ref:.2f} mm | "
            f"cmd→fb: mean={mean_cmd:.2f} max={self.max_cmd_mm:.2f} mm | "
            f"loop jitter: mean={mean_jit:.2f} max={self.max_jitter_ms:.2f} ms | "
            f"state RTT: {mean_rtt:.1f} ms"
        )

    def reset_window(self) -> None:
        self.n = 0
        self.sum_ref_mm = 0.0
        self.sum_cmd_mm = 0.0
        self.sum_sq_ref_mm = 0.0
        self.max_ref_mm = 0.0
        self.max_cmd_mm = 0.0
        self.sum_jitter_ms = 0.0
        self.max_jitter_ms = 0.0
        self.sum_state_rtt_ms = 0.0


def measure_state_rtt(robot, n: int = 5) -> float:
    samples: list[float] = []
    for _ in range(n):
        t0 = time.monotonic()
        ret, _ = robot.rm_get_current_arm_state()
        if ret == 0:
            samples.append((time.monotonic() - t0) * 1000.0)
    return sum(samples) / len(samples) if samples else float("nan")


def sin_y(t: float, y0: float, amplitude_m: float, omega: float) -> float:
    return y0 + amplitude_m * math.sin(omega * t)


def main() -> int:
    parser = argparse.ArgumentParser(description="TCP Y sin loop + lag stats")
    parser.add_argument(
        "--no-ruckig",
        action="store_true",
        help="direct sin -> CANFD (default; flag kept for explicit launch)",
    )
    parser.add_argument("--amplitude-mm", type=float, default=80.0)
    parser.add_argument("--period", type=float, default=4.0)
    parser.add_argument("--dt-ms", type=float, default=10.0)
    parser.add_argument("--trajectory-mode", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--report-every", type=int, default=100, help="commands per report line")
    parser.add_argument(
        "--lag-every",
        type=int,
        default=10,
        help="sample rm_get_current_arm_state every N commands (keep >=5 to avoid stalling loop)",
    )
    args = parser.parse_args()

    amplitude_m = args.amplitude_mm / 1000.0
    dt_s = args.dt_ms / 1000.0
    omega = 2.0 * math.pi / args.period

    from rm75_control import RobotSession
    from rm75_control.motion.canfd import send_pose_canfd

    print("Connecting...", flush=True)
    with RobotSession(config=CONFIG) as bot:
        ret, state = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            print(f"get state failed: {ret}", file=sys.stderr)
            return 1

        state_rtt_ms = measure_state_rtt(bot.robot)
        print(
            f"rm_get_current_arm_state RTT ~ {state_rtt_ms:.1f} ms "
            f"(lag sampled every {args.lag_every} cmds)",
            flush=True,
        )

        pose0 = list(state["pose"])
        x0, y0, z0 = pose0[0], pose0[1], pose0[2]
        orient = pose0[3:6]

        print("center pose (m, rad):", [round(v, 6) for v in pose0])
        print(
            f"sin→CANFD | Y +/-{args.amplitude_mm}mm period={args.period}s dt={args.dt_ms}ms | Ctrl+C stop",
            flush=True,
        )
        print("ref→fb: sin reference vs measured TCP | cmd→fb: sent pose vs measured", flush=True)

        t_start = time.monotonic()
        next_tick = t_start
        cmd_count = 0
        stats = LagStats()
        last_cmd_y = y0
        last_y_fb = y0
        last_jitter_ms = 0.0

        try:
            while True:
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)
                tick = next_tick
                next_tick += dt_s
                last_jitter_ms = (time.monotonic() - tick) * 1000.0

                t_now = tick - t_start
                y_ref = sin_y(t_now, y0, amplitude_m, omega)
                cmd_pose = [x0, y_ref, z0, *orient]

                last_cmd_y = cmd_pose[1]
                send_pose_canfd(
                    bot.robot,
                    cmd_pose,
                    follow=True,
                    trajectory_mode=args.trajectory_mode,
                )
                cmd_count += 1

                if cmd_count % args.lag_every == 0:
                    sample_t0 = time.monotonic()
                    ret, st = bot.robot.rm_get_current_arm_state()
                    sample_rtt_ms = (time.monotonic() - sample_t0) * 1000.0
                    if ret == 0:
                        y_fb = st["pose"][1]
                        last_y_fb = y_fb
                        t_sample = sample_t0 - t_start
                        y_ref_sample = sin_y(t_sample, y0, amplitude_m, omega)
                        stats.update(
                            y_ref_sample,
                            last_cmd_y,
                            y_fb,
                            last_jitter_ms,
                            sample_rtt_ms,
                        )

                if cmd_count % args.report_every == 0:
                    t_report = time.monotonic() - t_start
                    print(stats.report(t_report, last_y_fb * 1000.0), flush=True)
                    stats.reset_window()
        finally:
            bot.stop_all()

    print("stopped.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCtrl+C", file=sys.stderr)
        raise SystemExit(130)
