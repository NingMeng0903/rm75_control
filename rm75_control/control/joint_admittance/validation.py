"""FK validation: Pinocchio model vs the real Realman controller (重中之重).

The entire cascade is only as trustworthy as the URDF <-> robot frame match.
Before running ANY joint-position control, prove that Pinocchio FK agrees with
the Realman pose interface to <1 mm / <0.1 deg.  If it does not, the URDF base
rotation or the TCP offset is wrong and every downstream Jacobian is wrong.

Two robot comparisons (both use rm_get_current_arm_state + rm_get_current_tool_frame):

* flange  (default, tool-agnostic): recover the base->flange (link_7) transform
  from the reported base->tool pose and the active tool offset, then compare to
  Pinocchio's link_7 FK.  Validates the 7-DOF arm chain independent of any tool.
* tcp     : compare Pinocchio's `tcp` frame FK (link_7 +0.220 m Z) directly to
  the reported base->tool pose.  Requires the ACTIVE Realman tool frame to be the
  matching +220 mm tool; otherwise it will (correctly) report the offset mismatch.

Usage (source env.sh first):
    # read-only single-shot at the current configuration
    python -m rm75_control.control.joint_admittance.validation --ip 192.168.1.18

    # drive fixed MoveJ points from a poses yaml and assert thresholds
    python -m rm75_control.control.joint_admittance.validation \
        --ip 192.168.1.18 --poses tmp/force_compensation/config/poses.yaml --move

    # offline: compare recorded (q_deg, pose) pairs, no robot
    python -m rm75_control.control.joint_admittance.validation --npz run.npz
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.joint_admittance.model import RobotKinematics, deg2rad, pose_distance

POS_TOL_MM = 1.0
ROT_TOL_DEG = 0.1


def pose_to_se3(pose6: np.ndarray, euler_order: str = "xyz"):
    """[x,y,z,rx,ry,rz] -> (t(3), R(3x3))."""
    pose6 = np.asarray(pose6, dtype=float)
    t = pose6[:3].copy()
    R = Rsc.from_euler(euler_order, pose6[3:6], degrees=False).as_matrix()
    return t, R


def se3_to_pose(t: np.ndarray, R: np.ndarray, euler_order: str = "xyz") -> np.ndarray:
    pose = np.zeros(6, dtype=float)
    pose[:3] = t
    pose[3:6] = Rsc.from_matrix(R).as_euler(euler_order, degrees=False)
    return pose


def se3_inv(t: np.ndarray, R: np.ndarray):
    Rt = R.T
    return -Rt @ t, Rt


def se3_mul(ta, Ra, tb, Rb):
    return ta + Ra @ tb, Ra @ Rb


def pose_diff(pose_a: np.ndarray, pose_b: np.ndarray, euler_order: str = "xyz") -> tuple[float, float]:
    """Return (position error mm, orientation error deg) between two pose6."""
    return pose_distance(pose_a, pose_b, euler_order)


def base_flange_from_tool(tool_pose: np.ndarray, tool_offset: np.ndarray, euler_order: str = "xyz") -> np.ndarray:
    """base->flange = base->tool * (flange->tool)^-1."""
    tb, Rb = pose_to_se3(tool_pose, euler_order)
    to, Ro = pose_to_se3(tool_offset, euler_order)
    ti, Ri = se3_inv(to, Ro)
    tf, Rf = se3_mul(tb, Rb, ti, Ri)
    return se3_to_pose(tf, Rf, euler_order)


def _summary(rows: list[dict]) -> dict:
    max_mm = max((r["pos_mm"] for r in rows), default=0.0)
    max_deg = max((r["rot_deg"] for r in rows), default=0.0)
    ok = max_mm < POS_TOL_MM and max_deg < ROT_TOL_DEG
    return {"max_mm": max_mm, "max_deg": max_deg, "ok": ok, "n": len(rows)}


def _print_rows(rows: list[dict], mode: str) -> None:
    print(f"\n  {mode} comparison (Pinocchio vs Realman):", flush=True)
    print("   idx |  pos err (mm) | rot err (deg)", flush=True)
    for r in rows:
        flag = "" if (r["pos_mm"] < POS_TOL_MM and r["rot_deg"] < ROT_TOL_DEG) else "  <-- FAIL"
        print(f"   {r['idx']:>3} | {r['pos_mm']:>11.4f} | {r['rot_deg']:>11.5f}{flag}", flush=True)


def compare_offline(npz_path: str, kin: RobotKinematics, frame: str) -> dict:
    data = np.load(npz_path)
    q_deg = np.asarray(data["q_deg"] if "q_deg" in data else data["joint"], dtype=float)
    pose = np.asarray(data["pose"], dtype=float)
    if q_deg.ndim == 1:
        q_deg = q_deg[None, :]
        pose = pose[None, :]
    rows = []
    for i in range(len(q_deg)):
        q = deg2rad(q_deg[i][:7])
        fk = kin.fk_pose(q) if frame == "tcp" else kin.frame_pose(q, frame)
        d_mm, d_deg = pose_diff(fk, pose[i][:6], kin.euler_order)
        rows.append({"idx": i, "pos_mm": d_mm, "rot_deg": d_deg})
    _print_rows(rows, f"offline[{frame}]")
    return _summary(rows)


def _read_state(robot) -> tuple[np.ndarray, np.ndarray]:
    ret, st = robot.rm_get_current_arm_state()
    if ret != 0:
        raise RuntimeError(f"rm_get_current_arm_state failed: {ret}")
    q_deg = np.asarray(st["joint"][:7], dtype=float)
    pose = np.asarray(st["pose"][:6], dtype=float)
    return q_deg, pose


def _read_tool_offset(robot) -> tuple[str, np.ndarray]:
    ret, tf = robot.rm_get_current_tool_frame()
    if ret != 0:
        raise RuntimeError(f"rm_get_current_tool_frame failed: {ret}")
    return str(tf.get("name", "?")), np.asarray(tf["pose"][:6], dtype=float)


def compare_once(robot, kin: RobotKinematics, mode: str, idx: int, *, verbose: bool = False) -> dict:
    q_deg, tool_pose = _read_state(robot)
    q = deg2rad(q_deg)
    row: dict = {"idx": idx, "q_deg": q_deg.tolist()}

    if mode == "tcp":
        fk = kin.fk_pose(q)
        d_mm, d_deg = pose_diff(fk, tool_pose, kin.euler_order)
        row.update(pos_mm=d_mm, rot_deg=d_deg)
    elif mode == "rm_fk":
        fk = kin.fk_pose(q)
        rm_fk = np.asarray(robot.rm_algo_forward_kinematics(q_deg.tolist(), flag=1)[:6], dtype=float)
        d_mm, d_deg = pose_diff(fk, rm_fk, kin.euler_order)
        row.update(pos_mm=d_mm, rot_deg=d_deg, rm_fk=rm_fk.tolist())
    else:  # flange
        _tool_name, tool_offset = _read_tool_offset(robot)
        flange_meas = base_flange_from_tool(tool_pose, tool_offset, kin.euler_order)
        fk = kin.frame_pose(q, "link_7")
        d_mm, d_deg = pose_diff(fk, flange_meas, kin.euler_order)
        row.update(pos_mm=d_mm, rot_deg=d_deg, flange_meas=flange_meas.tolist(), fk_link7=fk.tolist())
        if verbose:
            r_mat = Rsc.from_euler(kin.euler_order, fk[3:6], degrees=False).as_matrix()
            delta_base = np.asarray(flange_meas[:3], dtype=float) - np.asarray(fk[:3], dtype=float)
            delta_link7 = r_mat.T @ delta_base
            row["flange_delta_link7_mm"] = (delta_link7 * 1000.0).tolist()
            print(
                f"  [{idx}] flange offset in link_7 frame (mm): "
                f"{np.round(delta_link7 * 1000.0, 3).tolist()}  |Δ|={d_mm:.3f} mm",
                flush=True,
            )
    return row


def run_robot(args, kin: RobotKinematics) -> dict:
    from rm75_control.core.session import RobotSession

    modes = ["flange", "tcp", "rm_fk"] if args.all_modes else [args.mode]
    summaries: dict[str, dict] = {}

    with RobotSession(ip=args.ip, port=args.port) as sess:
        robot = sess.robot
        tool_name, tool_offset = _read_tool_offset(robot)
        print(f"  active Realman tool frame: {tool_name!r}  offset={np.round(tool_offset, 5).tolist()}", flush=True)

        for mode in modes:
            if mode == "tcp":
                print(
                    "  NOTE: --mode tcp compares Pinocchio tcp vs state.pose (active tool).",
                    flush=True,
                )
            if mode == "rm_fk":
                print(
                    "  NOTE: --mode rm_fk compares Pinocchio tcp vs rm_algo_forward_kinematics.",
                    flush=True,
                )

            rows: list[dict] = []
            if args.move and args.poses:
                targets = _load_pose_targets(args.poses)
                print(f"  driving {len(targets)} MoveJ points from {args.poses} [{mode}]", flush=True)
                for i, q_tgt in enumerate(targets):
                    sess.move_joints(q_tgt, velocity_percent=args.speed, block=1)
                    time.sleep(0.6)
                    rows.append(compare_once(robot, kin, mode, i, verbose=args.verbose))
            else:
                print(f"  read-only: comparing at the current configuration [{mode}]", flush=True)
                rows.append(compare_once(robot, kin, mode, 0, verbose=args.verbose))

            _print_rows(rows, f"robot[{mode}]")
            summaries[mode] = _summary(rows)

            if mode == "flange" and rows and not summaries[mode]["ok"]:
                deltas = [r.get("flange_delta_link7_mm") for r in rows if "flange_delta_link7_mm" in r]
                if deltas:
                    mean_mm = np.mean(np.asarray(deltas, dtype=float), axis=0)
                    print(
                        f"  mean flange offset pin->rm in link_7 frame (mm): "
                        f"{np.round(mean_mm, 3).tolist()}  |mean|={np.linalg.norm(mean_mm):.3f} mm",
                        flush=True,
                    )
                    print(
                        "  If |mean| is constant across poses, fix joint_7 origin y in the URDF "
                        "(vendor -172.5 mm vs Realman ~-161.2 mm).",
                        flush=True,
                    )

    if len(summaries) == 1:
        return next(iter(summaries.values()))
    ok = all(s["ok"] for s in summaries.values())
    max_mm = max(s["max_mm"] for s in summaries.values())
    max_deg = max(s["max_deg"] for s in summaries.values())
    n = sum(s["n"] for s in summaries.values())
    return {"max_mm": max_mm, "max_deg": max_deg, "ok": ok, "n": n, "by_mode": summaries}


def _load_pose_targets(poses_yaml: str) -> list[np.ndarray]:
    import yaml

    with open(poses_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    targets: list[np.ndarray] = []
    # Accept either {poses: {a: {q_deg: [...]}, ...}} or {slots: [...]} or a plain list.
    src = data.get("poses", data.get("slots", data))
    if isinstance(src, dict):
        for _k, rec in src.items():
            if isinstance(rec, dict) and "q_deg" in rec:
                targets.append(np.asarray(rec["q_deg"][:7], dtype=float))
    elif isinstance(src, list):
        for rec in src:
            if isinstance(rec, dict) and "q_deg" in rec:
                targets.append(np.asarray(rec["q_deg"][:7], dtype=float))
            elif isinstance(rec, (list, tuple)):
                targets.append(np.asarray(rec[:7], dtype=float))
    if not targets:
        raise SystemExit(f"no q_deg pose targets found in {poses_yaml}")
    return targets


def main() -> None:
    ap = argparse.ArgumentParser(description="Pinocchio-vs-Realman FK validation")
    ap.add_argument("--ip", default="192.168.1.18")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--mode", choices=["flange", "tcp", "rm_fk"], default="flange")
    ap.add_argument("--all-modes", action="store_true", help="run flange + tcp + rm_fk in one session")
    ap.add_argument("--verbose", action="store_true", help="print per-pose flange offset in link_7 frame")
    ap.add_argument("--poses", default=None, help="poses yaml with q_deg entries")
    ap.add_argument("--move", action="store_true", help="drive MoveJ to each pose (needs --poses)")
    ap.add_argument("--speed", type=int, default=20, help="MoveJ velocity percent")
    ap.add_argument("--urdf", default=None, help="override URDF path")
    ap.add_argument("--npz", default=None, help="offline: compare recorded q_deg/pose arrays")
    args = ap.parse_args()

    kin = RobotKinematics(urdf_path=args.urdf)
    print(f"Loaded URDF: {kin.urdf_path}", flush=True)

    if args.npz:
        frame = "tcp" if args.mode == "tcp" else "link_7"
        summ = compare_offline(args.npz, kin, frame)
    else:
        summ = run_robot(args, kin)

    print(
        f"\n  RESULT: max pos {summ['max_mm']:.4f} mm | max rot {summ['max_deg']:.5f} deg "
        f"over {summ['n']} pose(s)  ->  {'PASS' if summ['ok'] else 'FAIL'}"
        f"  (tol {POS_TOL_MM} mm / {ROT_TOL_DEG} deg)",
        flush=True,
    )
    if not summ["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
