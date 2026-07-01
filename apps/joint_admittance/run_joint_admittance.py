#!/usr/bin/env python3
"""Run the cascaded joint-position controller on the RM75-F.

  source env.sh
  # 1) FIRST prove FK matches the robot (<1mm / <0.1deg):
  python -m rm75_control.control.joint_admittance.validation --ip 192.168.1.18
  # 2) then run the loop (bring-up default: hold the start pose):
  python apps/joint_admittance/run_joint_admittance.py --config configs/joint_admittance.yaml

For a real test sequence (Cartesian move to a pose + force-position sinusoid
scan there, both through this same controller), see
tmp/joint_admittance/d_sin_tool_y.py instead.

The outer admittance loop emits a Cartesian twist; the inner CLIK/QP loop turns
it into absolute joint angles streamed through rm_movej_canfd only - no MoveV/
MoveJ mode switching once running.  A single MoveJ positions the arm at start.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.hybrid_motion.controller import AdmittanceConfig, AdmittanceController
from rm75_control.control.joint_admittance.config import build_joint_ik_config
from rm75_control.control.joint_admittance.loop import (
    AdmittanceOuterLoop,
    JointIkController,
    run_joint_admittance_loop,
)
from rm75_control.control.joint_admittance.model import RobotKinematics
from rm75_control.control.joint_admittance.reference import HoldReference
from rm75_control.core.session import RobotSession


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description="RM75 joint-position cascaded controller")
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance.yaml"))
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--solver", choices=["clik", "qp"], default=None)
    ap.add_argument("--dry-run", action="store_true", help="build everything, do not connect")
    args = ap.parse_args()

    raw = load_yaml(args.config)
    if args.solver:
        raw.setdefault("inner", {})["solver"] = args.solver

    startup = raw.get("startup", {})
    dt = float(raw.get("timing", {}).get("dt_ms", 10.0)) / 1000.0

    kin = RobotKinematics()
    inner_cfg = build_joint_ik_config(raw)
    inner = JointIkController(kin, inner_cfg)
    print(f"Inner loop: solver={inner_cfg.solver} control_frame={inner_cfg.control_frame} "
          f"dt={dt*1000:.0f}ms v_scale={inner_cfg.v_scale}", flush=True)
    if inner_cfg.solver == "qp":
        print(f"  QP backend: {inner.core.backend_name}", flush=True)

    outer_ctrl = AdmittanceController(dt, AdmittanceConfig.from_dict(raw))
    desired_force = np.zeros(6)
    desired_force[2] = float(raw.get("force", {}).get("desired_z_n", 0.0))
    reference = HoldReference()  # bring-up default: hold the start pose
    outer = AdmittanceOuterLoop(outer_ctrl, reference, desired_force=desired_force)

    force_observer = None
    if bool(startup.get("enable_force", False)):
        from rm75_control.control.hybrid_motion.observer import CompensatedForceObserver

        force_observer = CompensatedForceObserver.from_yaml(raw)
        print("  force observer: enabled (requires calibrated phi)", flush=True)

    q_start = startup.get("pose_slot_q_deg")
    q_start = None if q_start is None else np.asarray(q_start, dtype=float)
    duration = args.duration if args.duration is not None else float(startup.get("duration_s", 10.0))

    if args.dry_run:
        print("dry-run: controllers built OK, not connecting.", flush=True)
        return 0

    robot_cfg = raw.get("robot", {})
    with RobotSession(ip=robot_cfg.get("ip"), port=robot_cfg.get("port"), config=args.config) as sess:
        run_joint_admittance_loop(
            sess,
            outer,
            inner,
            q_start_deg=q_start,
            duration_s=duration,
            dt=dt,
            force_observer=force_observer,
            follow=bool(startup.get("follow", True)),
            move_speed=int(startup.get("move_speed", 20)),
            realtime=bool(startup.get("realtime", False)),
            watchdog_timeout_s=float(startup.get("watchdog_timeout_s", 0.1)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
