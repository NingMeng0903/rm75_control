#!/usr/bin/env python3
"""On-robot bring-up test: Cartesian move to pose D, then tool-Y sin scan at D -
both driven by OUR joint-position cascade (CLIK inner loop -> rm_movej_canfd),
never rm_movev_canfd / MoveV. One continuous run, no mode switch at the D handoff.

  Phase 1  current pose -> scan pose D   (CartesianTrackOuterLoop + CLIK, no force)
  Phase 2  hold D, sin sweep tool-Y       (AdmittanceOuterLoop, force-position hybrid)

``poses.yaml`` slot ``d`` is the **force-ID Arm_Tip teach pose** (contact tip).
Scan pose **D** = that ``pose_base`` + **220 mm along tool +Z outward**.

Usage:
  source env.sh
  python -m rm75_control.control.joint_admittance.validation --ip 192.168.1.18
  python tmp/joint_admittance/d_sin_tool_y.py --dry-run
  python tmp/joint_admittance/d_sin_tool_y.py --scan-duration 0
  python tmp/joint_admittance/d_sin_tool_y.py --enable-force --desired-z 3.0 --scan-duration 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.hybrid_motion.controller import AdmittanceConfig, AdmittanceController
from rm75_control.control.joint_admittance.config import build_joint_ik_config
from rm75_control.control.joint_admittance.loop import (
    AdmittanceOuterLoop,
    CartesianTrackConfig,
    CartesianTrackOuterLoop,
    JointIkController,
    Phase,
    arrived,
    run_joint_admittance_phases,
)
from rm75_control.control.joint_admittance.model import RobotKinematics
from rm75_control.control.joint_admittance.reference import CartesianMoveReference, SinToolYReference
from rm75_control.core.session import RobotSession
from rm75_control.force.compensation.collection import load_slot
from rm75_control.force.compensation.id_config import load_config as load_force_id_config
from rm75_control.force.compensation.paths import CONFIG_ID
from rm75_control.force.compensation.tool_pose import (
    DEFAULT_SCAN_APPROACH_DZ_M,
    get_active_tool_name,
    poses_calib_tool_frame,
    slot_scan_approach_pose,
)
from rm75_control.force.compensation import excitation as ex


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_scan_pose_d(
    slot: str,
    robot,
    *,
    approach_dz_m: float,
    use_force_id_pose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (q_deg, scan_pose_D, force_id_pose_base) for slot in poses.yaml."""
    fid = load_force_id_config(CONFIG_ID)
    poses_data = ex.load_poses_yaml(fid.poses_yaml)
    calib_tool = poses_calib_tool_frame(poses_data)
    active = get_active_tool_name(robot) if robot is not None else ""
    if active and calib_tool and active != calib_tool:
        print(
            f"  slot {slot!r}: teach frame {calib_tool!r}, active tool {active!r}",
            flush=True,
        )

    q_deg, _fk_pose, rec = load_slot(fid, slot, robot, calib_tool=calib_tool)
    pose_id = np.asarray(rec["pose_base"], dtype=float)

    if use_force_id_pose:
        pose_d = _fk_pose.copy()
        print(f"  scan target: force-ID FK pose (legacy) pose={np.round(pose_d, 4)}", flush=True)
    else:
        pose_d = slot_scan_approach_pose(robot, pose_id, approach_dz_m=approach_dz_m)
        print(f"  force-ID slot {slot!r} Arm_Tip teach: pose={np.round(pose_id, 4)}", flush=True)
        print(
            f"  scan pose D (+{approach_dz_m * 1000:.0f} mm tool-Z): "
            f"pose={np.round(pose_d, 4)}",
            flush=True,
        )
    return q_deg, pose_d, pose_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance.yaml"))
    ap.add_argument("--slot", type=str, default="d")
    ap.add_argument("--approach-dz-mm", type=float, default=DEFAULT_SCAN_APPROACH_DZ_M * 1000.0)
    ap.add_argument("--use-force-id-pose", action="store_true")
    ap.add_argument("--solver", choices=["clik", "qp"], default=None)
    ap.add_argument("--move-duration", type=float, default=8.0)
    ap.add_argument("--move-kp", type=float, default=1.5, help="phase 1 Cartesian tracking gain (1/s)")
    ap.add_argument("--y-pp-cm", type=float, default=6.0)
    ap.add_argument("--max-vel-cm-s", type=float, default=2.0)
    ap.add_argument("--period-s", type=float, default=None)
    ap.add_argument("--desired-z", type=float, default=None)
    ap.add_argument("--scan-duration", type=float, default=30.0)
    ap.add_argument("--enable-force", action="store_true", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw = load_yaml(args.config)
    if args.solver:
        raw.setdefault("inner", {})["solver"] = args.solver
    startup = raw.get("startup", {})
    dt = float(raw.get("timing", {}).get("dt_ms", 10.0)) / 1000.0

    kin = RobotKinematics()
    inner_cfg = build_joint_ik_config(raw)
    inner = JointIkController(kin, inner_cfg)
    print(
        f"Inner loop: solver={inner_cfg.solver} control_frame={inner_cfg.control_frame} "
        f"dt={dt * 1000:.0f}ms v_scale={inner_cfg.v_scale}",
        flush=True,
    )

    amplitude_m = float(args.y_pp_cm) * 0.01 / 2.0
    max_vel_m_s = float(args.max_vel_cm_s) * 0.01
    desired_z = args.desired_z if args.desired_z is not None else float(raw.get("force", {}).get("desired_z_n", 0.0))
    enable_force = args.enable_force if args.enable_force is not None else bool(startup.get("enable_force", False))

    if args.dry_run:
        print("dry-run: controllers built OK, not connecting.", flush=True)
        return 0

    robot_cfg = raw.get("robot", {})
    with RobotSession(ip=robot_cfg.get("ip"), port=robot_cfg.get("port"), config=args.config) as sess:
        _q_slot_deg, pose_d, _pose_id = resolve_scan_pose_d(
            args.slot,
            sess.robot,
            approach_dz_m=float(args.approach_dz_mm) * 0.001,
            use_force_id_pose=bool(args.use_force_id_pose),
        )

        move_ref = CartesianMoveReference(pose_d, args.move_duration, euler_order=inner_cfg.euler_order)
        move_outer = CartesianTrackOuterLoop(
            move_ref,
            CartesianTrackConfig(
                k_task=np.full(6, args.move_kp),
                euler_order=inner_cfg.euler_order,
                control_frame=inner_cfg.control_frame,
            ),
        )
        phase1 = Phase(
            outer=move_outer,
            label=f"move -> {args.slot}",
            duration_s=None,
            max_duration_s=args.move_duration + 5.0,
            wait_until=lambda pose: arrived(
                pose, pose_d, tol_mm=2.0, tol_deg=1.0, euler_order=inner_cfg.euler_order
            ),
        )
        phases = [phase1]

        force_observer = None
        if args.scan_duration > 0.0:
            outer_ctrl = AdmittanceController(dt, AdmittanceConfig.from_dict(raw))
            desired_force = np.zeros(6)
            desired_force[2] = desired_z
            sin_ref = SinToolYReference(
                amplitude_m,
                period_s=args.period_s,
                max_vel_m_s=None if args.period_s is not None else max_vel_m_s,
                soft_start=True,
                ramp_s=2.0,
                euler_order=inner_cfg.euler_order,
            )
            scan_outer = AdmittanceOuterLoop(outer_ctrl, sin_ref, desired_force=desired_force)

            if enable_force:
                from rm75_control.control.hybrid_motion.observer import CompensatedForceObserver

                force_observer = CompensatedForceObserver.from_yaml(raw)
                print("  force observer: enabled (requires calibrated phi)", flush=True)
            else:
                print("  force observer: DISABLED (--enable-force not set)", flush=True)

            phases.append(
                Phase(
                    outer=scan_outer,
                    label="sin_tool_y @ D",
                    duration_s=args.scan_duration,
                    force_observer=force_observer,
                )
            )
            print(
                f"Phase 2: tool-Y sin amp={args.y_pp_cm:.1f}cm p-p, v_peak={args.max_vel_cm_s:.1f}cm/s, "
                f"desired_z={desired_z:.1f}N, duration={args.scan_duration:.1f}s",
                flush=True,
            )

        t_last_print = [0.0]

        def on_step(label: str, t_phase: float, step, pose, f_ext) -> None:
            now = time.perf_counter()
            if now - t_last_print[0] >= 1.0:
                t_last_print[0] = now
                print(
                    f"  [{label}] t={t_phase:5.1f}s  cart_err={step.cart_err_mm:5.2f}mm  "
                    f"Fz={f_ext[2]:+5.2f}N  manip={step.manip:.3f}",
                    flush=True,
                )

        run_joint_admittance_phases(
            sess,
            phases,
            inner,
            q_start_deg=None,
            dt=dt,
            follow=bool(startup.get("follow", True)),
            move_speed=int(startup.get("move_speed", 20)),
            realtime=bool(startup.get("realtime", False)),
            watchdog_timeout_s=float(startup.get("watchdog_timeout_s", 0.1)),
            on_step=on_step,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
