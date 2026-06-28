#!/usr/bin/env python3
"""
Velocity-resolved admittance + position closed-loop via rm_movev_canfd.

Uses compensated F_ext from force_compensation phi (direct SDK read, no ZMQ).

  source env.sh
  python tmp/Admittance/run_admittance.py
  python tmp/Admittance/run_admittance.py --config tmp/Admittance/config/admittance.yaml
  python tmp/Admittance/run_admittance.py --trajectory hold --desired-z 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from controller import AdmittanceConfig, AdmittanceController  # noqa: E402
from force_observer import CompensatedForceObserver  # noqa: E402
from paths import CONFIG_ADMITTANCE, CONFIG_ROBOT, PHI_JSON  # noqa: E402
from trajectory import TrajectoryGenerator  # noqa: E402

from rm75_control.motion.canfd import send_velocity_canfd  # noqa: E402


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def init_velocity_canfd(robot, vc: dict, dt_ms: float) -> None:
    ret = robot.rm_set_movev_canfd_init(
        int(vc.get("avoid_singularity", 1)),
        int(vc.get("frame_type", 1)),
        int(dt_ms),
    )
    if ret != 0:
        raise RuntimeError(f"rm_set_movev_canfd_init failed: {ret}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RM75 velocity admittance control")
    parser.add_argument("--config", type=Path, default=CONFIG_ADMITTANCE)
    parser.add_argument("--trajectory", type=str, default=None)
    parser.add_argument("--desired-z", type=float, default=None, help="sensor Fz target (N)")
    parser.add_argument("--duration", type=float, default=None, help="run time (s), default infinite")
    args = parser.parse_args()

    if not PHI_JSON.exists():
        print(f"Missing {PHI_JSON} — run force_calibrate.py first", file=sys.stderr)
        return 1

    raw = load_yaml(args.config)
    if args.trajectory:
        raw.setdefault("trajectory", {})["type"] = args.trajectory
    if args.desired_z is not None:
        raw.setdefault("force", {})["desired_z_n"] = args.desired_z

    timing = raw.get("timing", {})
    dt_ms = float(timing.get("dt_ms", 10.0))
    dt_s = dt_ms / 1000.0
    vc = raw.get("velocity_canfd", {})
    follow = bool(vc.get("follow", True))
    traj_mode = int(vc.get("trajectory_mode", 0))
    radio = int(vc.get("radio", 0))

    ctrl_cfg = AdmittanceConfig.from_dict(raw)
    observer = CompensatedForceObserver.from_yaml(raw)
    controller = AdmittanceController(dt_s, ctrl_cfg)

    f_cfg = raw.get("force", {})
    desired_z = float(f_cfg.get("desired_z_n", 3.0))
    f_des = np.zeros(6)
    f_des[2] = desired_z

    traj_kind = raw.get("trajectory", {}).get("type", "sin_tool_y")
    print(f"Admittance | rm_movev_canfd follow={follow} traj_mode={traj_mode}")
    print(f"  phi: {PHI_JSON.name}  Fz_des={desired_z:.2f} N  trajectory={traj_kind}")

    from rm75_control import RobotSession

    with RobotSession(config=CONFIG_ROBOT) as bot:
        ret, state = bot.robot.rm_get_current_arm_state()
        if ret != 0:
            print(f"get state failed: {ret}", file=sys.stderr)
            return 1
        pose0 = np.asarray(state["pose"][:6], dtype=float)
        traj = TrajectoryGenerator.from_dict(raw, pose0, bot.robot)

        init_velocity_canfd(bot.robot, vc, dt_ms)
        print("Velocity CANFD initialized. Ctrl+C to stop.")

        t0 = time.monotonic()
        next_tick = t0
        step = 0
        last_log = t0

        try:
            while True:
                now = time.monotonic()
                if args.duration is not None and now - t0 >= args.duration:
                    break
                if now < next_tick:
                    time.sleep(min(0.002, next_tick - now))
                    continue
                next_tick += dt_s
                t_s = now - t0
                step += 1

                ret_s, st = bot.robot.rm_get_current_arm_state()
                ret_f, fd = bot.robot.rm_get_force_data()
                if ret_s != 0 or ret_f != 0:
                    continue

                pose = np.asarray(st["pose"][:6], dtype=float)
                force_raw = np.asarray(fd["force_data"][:6], dtype=float)
                observer.append(t_s, pose, force_raw)

                wrench = observer.latest_wrench()
                if wrench is None:
                    send_velocity_canfd(
                        bot.robot, [0.0] * 6,
                        follow=follow, trajectory_mode=traj_mode, radio=radio,
                    )
                    continue

                _, f_ext = wrench
                pose_d, vel_ff = traj.sample(t_s)
                v_cmd = controller.compute_velocity_command(
                    pose, pose_d, vel_ff, f_ext, f_des,
                )
                send_velocity_canfd(
                    bot.robot, v_cmd.tolist(),
                    follow=follow, trajectory_mode=traj_mode, radio=radio,
                )

                if now - last_log >= 1.0:
                    last_log = now
                    print(
                        f"  t={t_s:.1f}s  Fz_ext={f_ext[2]:+.2f}N  "
                        f"vz_cmd={v_cmd[2]:+.4f}  vy_cmd={v_cmd[1]:+.4f}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            try:
                send_velocity_canfd(
                    bot.robot, [0.0] * 6,
                    follow=follow, trajectory_mode=traj_mode, radio=radio,
                )
                bot.robot.rm_set_arm_slow_stop()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
