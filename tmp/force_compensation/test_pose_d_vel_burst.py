#!/usr/bin/env python3
"""
Pose D rm_movev_canfd angular velocity burst (validated preset: pose_d_vel_burst).

  python tmp/force_compensation/test_pose_d_vel_burst.py --clear-err
  python tmp/force_compensation/test_pose_d_vel_burst.py --clear-err --diag
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from utils import excitation as ex  # noqa: E402
from utils.collection import move_j, wait_settle  # noqa: E402
from utils.id_config import BURST_PROFILES, DEFAULT_BURST_PROFILE, load_velocity_burst  # noqa: E402
from utils.paths import CONFIG_ROBOT, LOG_DIR, POSES_YAML  # noqa: E402

from rm75_control.core.exceptions import MotionError  # noqa: E402
from rm75_control.motion.canfd import send_velocity_canfd  # noqa: E402

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
AXIS_NAMES = ("wx", "wy", "wz")
ERR_4119 = "4119"
ERR_4119_HELP = (
    "4119 = 六维力外载校验失败（力传感器需重新重心标定）。"
    "示教器：系统信息→清除错误；配置→力传感器配置→六维力重心标定（标定中勿碰工具）。"
)

# Named presets — conservative → stronger. Run one at a time (no --skip-move).
PROFILES: dict[str, dict] = {
    "gentle": {
        "amp_deg_s": 5.0,
        "freqs_hz": [0.3],
        "segment_s": 10.0,
        "duration": 30.0,
        "ramp_s": 3.0,
        "frame_type": 1,
        "avoid_singularity": 0,
    },
    "baseline": {
        "amp_deg_s": 6.0,
        "freqs_hz": [0.5, 0.7],
        "segment_s": 10.0,
        "duration": 30.0,
        "ramp_s": 2.0,
        "frame_type": 1,
        "avoid_singularity": 0,
    },
    "single": {
        "amp_deg_s": 8.0,
        "freqs_hz": [0.35],
        "segment_s": 12.0,
        "duration": 36.0,
        "ramp_s": 3.0,
        "frame_type": 1,
        "avoid_singularity": 0,
    },
    "single_plus": {
        "amp_deg_s": 10.0,
        "freqs_hz": [0.3],
        "segment_s": 15.0,
        "duration": 45.0,
        "ramp_s": 3.0,
        "frame_type": 1,
        "avoid_singularity": 0,
    },
    "single_long": {
        "amp_deg_s": 10.0,
        "freqs_hz": [0.25],
        "segment_s": 20.0,
        "duration": 60.0,
        "ramp_s": 4.0,
        "frame_type": 1,
        "avoid_singularity": 0,
    },
    "single_raw": {
        "amp_deg_s": 10.0,
        "freqs_hz": [0.3],
        "segment_s": 15.0,
        "duration": 45.0,
        "ramp_s": 3.0,
        "frame_type": 1,
        "avoid_singularity": 0,
        "trajectory_mode": 0,
        "radio": 0,
    },
    "single_raw_long": {
        "amp_deg_s": 12.0,
        "freqs_hz": [0.25],
        "segment_s": 20.0,
        "duration": 60.0,
        "ramp_s": 4.0,
        "frame_type": 1,
        "avoid_singularity": 0,
        "trajectory_mode": 0,
        "radio": 0,
    },
    "tool_plus": {
        "amp_deg_s": 10.0,
        "freqs_hz": [0.3],
        "segment_s": 15.0,
        "duration": 45.0,
        "ramp_s": 3.0,
        "frame_type": 0,
        "avoid_singularity": 0,
    },
    "tool_raw_plus": {
        "amp_deg_s": 12.0,
        "freqs_hz": [0.28],
        "segment_s": 15.0,
        "duration": 45.0,
        "ramp_s": 3.0,
        "frame_type": 0,
        "avoid_singularity": 0,
        "trajectory_mode": 0,
        "radio": 0,
    },
    "tool_wy_solo": {
        "amp_deg_s": 12.0,
        "freqs_hz": [0.28],
        "segment_s": 45.0,
        "duration": 45.0,
        "ramp_s": 3.0,
        "frame_type": 0,
        "avoid_singularity": 0,
        "single_axis": 1,
        "trajectory_mode": 0,
        "radio": 0,
    },
    "base_wy_solo": {
        "amp_deg_s": 12.0,
        "freqs_hz": [0.28],
        "segment_s": 45.0,
        "duration": 45.0,
        "ramp_s": 3.0,
        "frame_type": 1,
        "avoid_singularity": 0,
        "single_axis": 1,
        "trajectory_mode": 0,
        "radio": 0,
    },
    "base_wy_first": {
        "amp_deg_s": 12.0,
        "freqs_hz": [0.28],
        "segment_s": 15.0,
        "duration": 45.0,
        "ramp_s": 3.0,
        "frame_type": 1,
        "avoid_singularity": 0,
        "axis_order": (1, 2, 0),
        "trajectory_mode": 0,
        "radio": 0,
    },
    "base_low_freq": {
        "amp_deg_s": 12.0,
        "freqs_hz": [0.18],
        "freqs_hz_per_seg": [[0.18], [0.22], [0.20]],
        "segment_s": 15.0,
        "duration": 45.0,
        "ramp_s": 3.0,
        "frame_type": 1,
        "avoid_singularity": 0,
        "trajectory_mode": 0,
        "radio": 0,
    },
    "slow": {
        "amp_deg_s": 10.0,
        "freqs_hz": [0.25],
        "segment_s": 15.0,
        "duration": 45.0,
        "ramp_s": 3.0,
        "frame_type": 1,
        "avoid_singularity": 0,
    },
    "tool": {
        "amp_deg_s": 8.0,
        "freqs_hz": [0.4],
        "segment_s": 12.0,
        "duration": 36.0,
        "ramp_s": 3.0,
        "frame_type": 0,
        "avoid_singularity": 0,
    },
    "strong": {
        "amp_deg_s": 12.0,
        "freqs_hz": [0.4, 0.55],
        "segment_s": 12.0,
        "duration": 36.0,
        "ramp_s": 2.5,
        "frame_type": 1,
        "avoid_singularity": 0,
    },
}
for _burst_name in BURST_PROFILES:
    PROFILES[_burst_name] = {**BURST_PROFILES[_burst_name], "duration": 45.0}
SWEEP_ORDER = ("gentle", "slow", "tool")
I_SWEEP_ORDER = (DEFAULT_BURST_PROFILE,)


def read_system_err(robot) -> list[str]:
    ret, st = robot.rm_get_current_arm_state()
    if ret != 0:
        return [f"get_state:{ret}"]
    err = st.get("err", {})
    n = int(err.get("err_len", 0))
    return list(err.get("err", []))[:n]


def ensure_arm_ready(robot, *, clear_err: bool) -> list[str]:
    codes = read_system_err(robot)
    if not codes or codes == ["0"]:
        return []
    if clear_err:
        robot.rm_clear_system_err()
        time.sleep(0.3)
        codes = read_system_err(robot)
    if codes and codes != ["0"]:
        return codes
    return []


def unwrap_delta(prev: float, curr: float) -> float:
    d = curr - prev
    if d > math.pi:
        d -= 2 * math.pi
    elif d < -math.pi:
        d += 2 * math.pi
    return d


def vel_burst_cmd(
    t_s: float,
    *,
    amp_rad_s: float,
    freqs_hz: list[float],
    segment_s: float,
    ramp_s: float,
    axis_order: tuple[int, int, int] = (0, 1, 2),
    amp_rad_s_per_seg: list[float] | None = None,
    freqs_hz_per_seg: list[list[float]] | None = None,
    single_axis: int | None = None,
) -> tuple[np.ndarray, int]:
    """Return (6D vel cmd, axis_idx 0=wx 1=wy 2=wz being excited)."""
    if single_axis is not None:
        seg_slot = 0
        axis_idx = int(single_axis)
        t_loc = t_s
    else:
        seg_slot = int(t_s // segment_s) % 3
        axis_idx = axis_order[seg_slot]
        t_loc = t_s - seg_slot * segment_s

    axis = 3 + axis_idx
    amp = amp_rad_s
    if amp_rad_s_per_seg is not None and single_axis is None:
        amp = amp_rad_s_per_seg[seg_slot]
    freqs = freqs_hz
    if freqs_hz_per_seg is not None and single_axis is None:
        freqs = freqs_hz_per_seg[seg_slot]

    ramp_global = min(1.0, t_s / ramp_s) if ramp_s > 0 else 1.0
    ramp_seg = min(1.0, t_loc / min(ramp_s, segment_s * 0.2)) if ramp_s > 0 else 1.0
    env = ramp_global * ramp_seg

    vel = np.zeros(6, dtype=float)
    for k, f in enumerate(freqs):
        ph = seg_slot * 1.4 + k * 0.85
        vel[axis] += amp * math.sin(2.0 * math.pi * f * t_loc + ph)
    vel *= env
    return vel, axis_idx


def prepare_movev_session(bot) -> dict:
    """Clear stale force / CANFD modes before rm_set_movev_canfd_init."""
    diag = bot.prepare_for_force_stream(settle_s=1.0)
    diag["delete_traj"] = bot.robot.rm_set_arm_delete_trajectory()
    diag["slow_stop"] = bot.robot.rm_set_arm_slow_stop()
    time.sleep(0.5)
    traj = bot.robot.rm_get_arm_current_trajectory()
    diag["trajectory_type"] = traj.get("trajectory_type", -1)
    ret, st = bot.robot.rm_get_current_arm_state()
    if ret == 0:
        diag["err"] = st.get("err", {})
    return diag


def init_velocity_canfd(robot, *, frame_type: int, avoid_singularity: int, dt_ms: float) -> int:
    ret = robot.rm_set_movev_canfd_init(avoid_singularity, frame_type, int(dt_ms))
    if ret != 0:
        raise RuntimeError(f"rm_set_movev_canfd_init failed: {ret}")
    return ret


def send_vel_checked(robot, vel, *, follow, trajectory_mode, radio) -> int:
    try:
        send_velocity_canfd(
            robot, vel,
            follow=follow, trajectory_mode=trajectory_mode, radio=radio,
        )
        return 0
    except MotionError as e:
        msg = str(e)
        if "code " in msg:
            return int(msg.rsplit("code ", 1)[-1].rstrip(")"))
        return -1


def ramp_down_velocity(
    robot,
    start_vel: np.ndarray,
    *,
    ramp_s: float,
    dt_ms: float,
    follow: bool,
    trajectory_mode: int,
    radio: int,
    next_tick: float | None = None,
) -> float:
    """Cosine fade of last cmd to zero — endpoint vel & accel both ~0 (no jerk snap)."""
    start_vel = np.asarray(start_vel, dtype=float)
    if ramp_s <= 0 or float(np.max(np.abs(start_vel))) < 1e-9:
        send_vel_checked(
            robot, [0.0] * 6,
            follow=follow, trajectory_mode=trajectory_mode, radio=radio,
        )
        return next_tick if next_tick is not None else time.monotonic()

    dt_s = dt_ms / 1000.0
    n = max(2, int(ramp_s / dt_s) + 1)
    if next_tick is None:
        next_tick = time.monotonic()
    for i in range(n):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
        next_tick += dt_s
        scale = 0.5 * (1.0 + math.cos(math.pi * i / (n - 1)))
        send_vel_checked(
            robot, (start_vel * scale).tolist(),
            follow=follow, trajectory_mode=trajectory_mode, radio=radio,
        )

    for _ in range(3):
        send_vel_checked(
            robot, [0.0] * 6,
            follow=follow, trajectory_mode=trajectory_mode, radio=radio,
        )
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
        next_tick += dt_s
    return next_tick


def motion_stats(pose_log: np.ndarray, q_log: np.ndarray, t_log: np.ndarray) -> dict:
    if len(t_log) < 2:
        return {}
    dt = np.diff(t_log)
    dt = np.maximum(dt, 1e-6)

    deul = np.diff(pose_log[:, 3:6], axis=0)
    for j in range(3):
        d = deul[:, j]
        deul[:, j] = np.where(d > math.pi, d - 2 * math.pi, d)
        deul[:, j] = np.where(d < -math.pi, d + 2 * math.pi, d)
    omega_eul = np.abs(deul / dt[:, None])

    dq = np.diff(q_log, axis=0)
    omega_j = np.abs(dq / dt[:, None])
    omega_j_max = np.max(omega_j, axis=1)

    pos_drift_mm = float(np.max(np.linalg.norm(pose_log[:, :3] - pose_log[0:1, :3], axis=1)) * 1000)
    return {
        "euler_deg_s_max": tuple(float(np.max(omega_eul[:, j]) * RAD2DEG) for j in range(3)),
        "joint_deg_s_max": tuple(float(np.max(omega_j[:, j])) for j in range(7)),
        "joint_deg_s_p90": float(np.percentile(omega_j_max, 90)),
        "pos_drift_mm": pos_drift_mm,
    }


def run_stream(
    bot,
    *,
    duration_s: float,
    dt_ms: float,
    log_every: int,
    follow: bool,
    trajectory_mode: int,
    radio: int,
    vel_fn,
) -> tuple[int, dict, np.ndarray, np.ndarray, np.ndarray]:
    dt_s = dt_ms / 1000.0
    n_cmd = int(duration_s / dt_s) + 1
    n_log = (n_cmd + log_every - 1) // log_every
    t_log = np.zeros(n_log)
    pose_log = np.zeros((n_log, 6))
    q_log = np.zeros((n_log, 7))
    vel_cmd_log = np.zeros((n_log, 6))

    next_tick = time.monotonic()
    log_i = 0
    err_count = 0

    for i in range(n_cmd):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
        next_tick += dt_s
        t_cmd = i * dt_s

        vel_cmd = np.asarray(vel_fn(t_cmd), dtype=float)
        ret = send_vel_checked(
            bot.robot, vel_cmd.tolist(),
            follow=follow, trajectory_mode=trajectory_mode, radio=radio,
        )
        if ret != 0:
            err_count += 1

        if i % log_every == 0:
            ret_s, st = bot.robot.rm_get_current_arm_state()
            if ret_s == 0:
                pose_log[log_i] = st["pose"][:6]
                q_log[log_i] = st["joint"][:7]
            t_log[log_i] = t_cmd
            vel_cmd_log[log_i] = vel_cmd
            log_i += 1

    stats = motion_stats(pose_log[:log_i], q_log[:log_i], t_log[:log_i])
    stats["movev_errors"] = err_count
    return log_i, stats, t_log[:log_i], pose_log[:log_i], q_log[:log_i]


def pose_to_rm_pose(pose: list[float]):
    from Robotic_Arm.rm_ctypes_wrap import rm_euler_t, rm_pose_t, rm_position_t

    po = rm_pose_t()
    po.position = rm_position_t(*pose[:3])
    po.euler = rm_euler_t(*pose[3:6])
    return po


def read_tool_pose(robot) -> list[float] | None:
    ret, state = robot.rm_get_current_arm_state()
    if ret != 0:
        return None
    return list(robot.rm_algo_end2tool(pose_to_rm_pose(list(state["pose"]))))


def run_diag(bot, args, pose0: np.ndarray) -> int:
    """Quick checks: linear movev, angular, sin_y-style position loop."""
    trajectory_mode, radio = resolve_stream_mode(args)
    dur = 5.0
    prep = prepare_movev_session(bot) if not args.no_prepare else {}
    if prep:
        print(f"  prep: {prep}")
        if ERR_4119 in str(prep.get("err", {})):
            print(f"  !! {ERR_4119_HELP}", file=sys.stderr)
    print(f"\n=== diagnostics ({dur:.0f}s each) ===")
    print(
        f"  stream: follow={args.follow} traj={trajectory_mode} radio={radio} "
        f"dt={args.dt_ms}ms"
    )
    print(f"  pose0 euler deg: {[round(v * RAD2DEG, 2) for v in pose0[3:6]]}\n")

    cases = [
        ("A tool vy=1cm/s open-loop", 0, 0, lambda _t: [0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
        ("B tool wy=12deg/s open-loop", 0, 0, lambda _t: [0.0, 0.0, 0.0, 0.0, 12.0 * DEG2RAD, 0.0]),
        ("C tool wz=6deg/s open-loop", 0, 0, lambda _t: [0.0, 0.0, 0.0, 0.0, 0.0, 6.0 * DEG2RAD]),
        ("D base wy=12deg/s avoid=0", 1, 0, lambda _t: [0.0, 0.0, 0.0, 0.0, 12.0 * DEG2RAD, 0.0]),
        ("E base wz=6deg/s avoid=0", 1, 0, lambda _t: [0.0, 0.0, 0.0, 0.0, 0.0, 6.0 * DEG2RAD]),
    ]

    for label, frame_type, avoid, vel_fn in cases:
        if not args.no_prepare:
            prepare_movev_session(bot)
        init_ret = init_velocity_canfd(
            bot.robot,
            frame_type=frame_type,
            avoid_singularity=avoid,
            dt_ms=args.dt_ms,
        )
        n, stats, _, pose_log, _ = run_stream(
            bot,
            duration_s=dur,
            dt_ms=args.dt_ms,
            log_every=args.log_every,
            follow=args.follow,
            trajectory_mode=trajectory_mode,
            radio=radio,
            vel_fn=vel_fn,
        )
        send_vel_checked(
            bot.robot, [0.0] * 6,
            follow=args.follow, trajectory_mode=trajectory_mode, radio=radio,
        )
        if n < 2:
            print(f"  {label}: no samples (init={init_ret})")
            continue
        eul = stats.get("euler_deg_s_max", (0, 0, 0))
        jmax = max(stats.get("joint_deg_s_max", (0,) * 7))
        y_drift = float(np.max(np.abs(pose_log[:, 1] - pose_log[0, 1])) * 1000)
        z_drift = float((pose_log[-1, 2] - pose_log[0, 2]) * 1000)
        print(
            f"  {label}: init={init_ret}  drift_xyz={stats.get('pos_drift_mm', 0):.1f}mm  "
            f"dy={y_drift:.1f}mm  dz_end={z_drift:+.1f}mm  euler_max={max(eul):.2f} deg/s  "
            f"joint_max={jmax:.2f} deg/s  movev_err={stats.get('movev_errors', 0)}"
        )

    # D: sin_y-style tool-Y with position closure (known-good pattern in this repo)
    print("\n  D sin_y-style tool Y (position loop, 20mm amp)...")
    if not args.no_prepare:
        prepare_movev_session(bot)
    init_ret = init_velocity_canfd(bot.robot, frame_type=0, avoid_singularity=0, dt_ms=args.dt_ms)
    from rm75_control.control.cartesian_velocity import (
        CartesianVelocityController,
        CartesianVelocityStreamConfig,
        CartesianVelocityTracker,
        CartesianVelocityTrackerConfig,
    )

    ret, st = bot.robot.rm_get_current_arm_state()
    if ret != 0:
        print(f"  D: get state failed {ret}")
        return 1
    pose0_tool = read_tool_pose(bot.robot)
    if pose0_tool is None:
        print("  D: read_tool_pose failed")
        return 1
    y0 = pose0_tool[1]
    amp_m = 0.02
    omega = 2.0 * math.pi / 4.0

    def y_ref_at(t: float, y_ref0: float, amplitude_m: float, om: float) -> float:
        return y_ref0 + amplitude_m * math.sin(om * t)

    def vy_at(t: float, amplitude_m: float, om: float) -> float:
        return amplitude_m * om * math.cos(om * t)

    vel_ctrl = CartesianVelocityController(
        bot.robot,
        tracker=CartesianVelocityTracker(
            CartesianVelocityTrackerConfig.for_motion_axes((1,), kp=0.8, ref_speed_peak_m_s=amp_m * omega)
        ),
        config=CartesianVelocityStreamConfig(
            follow=args.follow, trajectory_mode=trajectory_mode, radio=radio, period_ms=args.dt_ms,
        ),
    )
    dt_s = args.dt_ms / 1000.0
    n_cmd = int(dur / dt_s) + 1
    pose_start = np.array(pose0_tool[:3])
    last_fb = list(pose0_tool)
    t_start = time.monotonic()
    for i in range(n_cmd):
        tick = t_start + i * dt_s
        now = time.monotonic()
        if now < tick:
            time.sleep(tick - now)
        t_s = i * dt_s
        ref_pose = list(pose0_tool)
        ref_pose[1] = y_ref_at(t_s, y0, amp_m, omega)
        ref_vel = [0.0, vy_at(t_s, amp_m, omega), 0.0, 0.0, 0.0, 0.0]
        vel_ctrl.step(ref_pose=ref_pose, ref_vel=ref_vel, fb_pose=last_fb, dt_s=dt_s)
        if i % args.log_every == 0:
            fb = read_tool_pose(bot.robot)
            if fb is not None:
                last_fb = fb
    vel_ctrl.send_velocity([0.0] * 6)
    y_drift = abs(last_fb[1] - y0) * 1000
    xyz_drift = float(np.linalg.norm(np.array(last_fb[:3]) - pose_start) * 1000)
    print(
        f"  D sin_y-style: init={init_ret}  tool_y_drift={y_drift:.1f}mm  "
        f"tool_xyz_drift={xyz_drift:.1f}mm"
    )

    print(
        "\nInterpretation:"
        "\n  A linear tool vy: open-loop drift expected in 5s"
        "\n  B–E constant angular: euler_max vs cmd shows axis tracking; "
        "burst uses sin+segments — compare joint_fb on wy/wz segments (~11=good)"
        "\n  sin_y-style D: position-closed tool Y (reference for linear tracking)"
        "\n  if all angular joint_max << cmd: clear err / restart controller, then burst"
    )
    return 0


def apply_profile(args: argparse.Namespace) -> None:
    if args.profile is None:
        return
    if args.profile not in PROFILES:
        raise SystemExit(f"Unknown profile {args.profile!r}; choose from {list(PROFILES)}")
    p = PROFILES[args.profile]
    args.amp_deg_s = p["amp_deg_s"]
    args.freqs_hz = list(p["freqs_hz"])
    args.segment_s = p["segment_s"]
    args.duration = p["duration"]
    args.ramp_s = p["ramp_s"]
    args.frame_type = p["frame_type"]
    args.avoid_singularity = int(p.get("avoid_singularity", 1))
    if "axis_order" in p:
        args.axis_order = tuple(p["axis_order"])
    if "single_axis" in p:
        args.single_axis = p["single_axis"]
    if "amp_deg_s_per_seg" in p:
        args.amp_deg_s_per_seg = list(p["amp_deg_s_per_seg"])
    if "freqs_hz_per_seg" in p:
        args.freqs_hz_per_seg = [list(row) for row in p["freqs_hz_per_seg"]]
    if "ramp_down_s" in p:
        args.ramp_down_s = float(p["ramp_down_s"])
    if "trajectory_mode" in p:
        args.trajectory_mode = int(p["trajectory_mode"])
    if "radio" in p:
        args.radio = int(p["radio"])
    if args.out == LOG_DIR / "test_pose_d_vel_burst.npz":
        if args.profile == DEFAULT_BURST_PROFILE:
            args.out = LOG_DIR / "test_pose_d_vel_burst.npz"
        else:
            args.out = LOG_DIR / f"test_pose_d_vel_burst_{args.profile}.npz"


def print_kinematics_summary(out: Path) -> None:
    """Quick regressor-relevant stats from saved NPZ."""
    if not out.exists():
        return
    from utils import regressor as fid

    d = np.load(out, allow_pickle=True)
    pose, t, seg = d["pose"], d["t"], d["segment"]
    pose0 = d["pose0"]
    cfg = fid.FrameConfig.from_yaml(PKG.parents[1] / "configs" / "force_sensor.yaml")
    omega_s, alpha_s, _, _ = fid.kinematics_sensor(pose, t, cfg, fc=2.0)
    an = np.linalg.norm(alpha_s, axis=1) * RAD2DEG
    dp = pose[:, :3] - pose0[:3]
    drift_mm = float(np.max(np.linalg.norm(dp, axis=1)) * 1000)
    deul = pose[:, 3:6] - pose0[3:6]
    for j in range(3):
        delta = deul[:, j]
        deul[:, j] = np.where(delta > np.pi, delta - 2 * np.pi, delta)
        deul[:, j] = np.where(delta < -np.pi, delta + 2 * np.pi, delta)
    eul_swing = tuple(float(np.max(np.abs(deul[:, j])) * RAD2DEG) for j in range(3))
    print(f"\n--- kinematics summary ({out.name}) ---")
    print(
        f"  TCP drift: {drift_mm:.0f} mm  euler swing deg: "
        f"rx={eul_swing[0]:.1f} ry={eul_swing[1]:.1f} rz={eul_swing[2]:.1f}"
    )
    print(f"  |α| p90={np.percentile(an, 90):.1f} max={np.max(an):.1f} deg/s²")
    vel_fb = d["vel_fb_rad_s"]
    vel_cmd = d["vel_cmd"]
    qdot = d["qdot_fb_deg_s"] if "qdot_fb_deg_s" in d.files else None
    for s, name in enumerate(AXIS_NAMES):
        m = seg == s
        if not np.any(m):
            continue
        tr = 100.0 * np.max(np.abs(vel_fb[m, s])) / max(
            np.max(np.abs(vel_cmd[m, 3 + s])), 1e-9
        )
        os_p90 = float(np.percentile(np.linalg.norm(omega_s[m], axis=1) * RAD2DEG, 90))
        as_p90 = float(np.percentile(an[m], 90))
        line = (
            f"  {name}: |α_s| p90={as_p90:.1f}  |ω_s| p90={os_p90:.1f}  "
            f"euler_track≈{tr:.0f}%"
        )
        if qdot is not None:
            jq = float(np.percentile(np.max(np.abs(qdot[m]), axis=1), 90))
            line += f"  joint_p90={jq:.1f} deg/s"
        print(line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pose D rm_movev_canfd angular velocity burst test")
    p.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_BURST_PROFILE,
        choices=list(PROFILES),
        help=f"preset (default {DEFAULT_BURST_PROFILE}); collection uses pose_d_vel_burst only",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help=f"run profiles {SWEEP_ORDER} in sequence (always move to pose d)",
    )
    p.add_argument(
        "--sweep-i",
        action="store_true",
        help=f"I-ID sweep: {I_SWEEP_ORDER}",
    )
    p.add_argument(
        "--axis-order",
        type=int,
        nargs=3,
        default=None,
        metavar=("AX0", "AX1", "AX2"),
        help="segment axis order 0=wx 1=wy 2=wz (default wx,wy,wz)",
    )
    p.add_argument(
        "--single-axis",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="excite one angular axis for full duration (0=wx 1=wy 2=wz)",
    )
    p.add_argument("--diag", action="store_true", help="run linear/angular constant-vel checks first")
    p.add_argument("--duration", type=float, default=30.0, help="burst duration (s)")
    p.add_argument("--segment-s", type=float, default=10.0, help="seconds per wx/wy/wz axis")
    p.add_argument("--amp-deg-s", type=float, default=8.0, help="peak angular speed per tone (deg/s)")
    p.add_argument("--freqs-hz", type=float, nargs="+", default=argparse.SUPPRESS)
    p.add_argument("--ramp-s", type=float, default=2.0)
    p.add_argument(
        "--ramp-down-s",
        type=float,
        default=4.0,
        help="cosine fade-out after burst (s, default 4)",
    )
    p.add_argument("--dt-ms", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=10, help="log every N control ticks (~100ms at dt=10ms; use 10 for |α| summary)")
    p.add_argument("--move-speed", type=int, default=15)
    p.add_argument("--settle-timeout-s", type=float, default=15.0)
    p.add_argument("--frame-type", type=int, default=1, help="0 tool, 1 work/base")
    p.add_argument("--avoid-singularity", type=int, default=argparse.SUPPRESS)
    p.add_argument("--follow", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trajectory-mode", type=int, default=-1, help="-1: auto (1 if follow else 0)")
    p.add_argument("--radio", type=int, default=-1, help="-1: auto (40 if follow+mode1 else 0)")
    p.add_argument("--skip-move", action="store_true", help="assume already at pose d")
    p.add_argument("--clear-err", action="store_true", help="call rm_clear_system_err before test")
    p.add_argument("--no-prepare", action="store_true", help="skip prepare_movev_session (like sin_y)")
    p.add_argument(
        "--out",
        type=Path,
        default=LOG_DIR / "test_pose_d_vel_burst.npz",
        help="NPZ output path",
    )
    return p.parse_args()


def resolve_stream_mode(args: argparse.Namespace) -> tuple[int, int]:
    follow = args.follow
    if args.trajectory_mode == -1:
        trajectory_mode = 1 if follow else 0
    else:
        trajectory_mode = args.trajectory_mode
    if args.radio == -1:
        radio = 40 if follow and trajectory_mode == 1 else 0
    else:
        radio = args.radio
    if not follow and trajectory_mode != 0:
        trajectory_mode = 0
        radio = 0
    return trajectory_mode, radio


def run_burst_once(args: argparse.Namespace) -> int:
    cli_freqs = getattr(args, "freqs_hz", None)
    cli_avoid = getattr(args, "avoid_singularity", None)
    apply_profile(args)
    if cli_freqs is not None:
        args.freqs_hz = list(cli_freqs)
    if cli_avoid is not None:
        args.avoid_singularity = int(cli_avoid)
    elif not hasattr(args, "avoid_singularity"):
        args.avoid_singularity = 1
    if getattr(args, "axis_order", None) is None:
        args.axis_order = (0, 1, 2)
    if not hasattr(args, "amp_deg_s_per_seg"):
        args.amp_deg_s_per_seg = None
    if not hasattr(args, "freqs_hz_per_seg"):
        args.freqs_hz_per_seg = None
    if getattr(args, "ramp_down_s", None) is None:
        args.ramp_down_s = 4.0
    trajectory_mode, radio = resolve_stream_mode(args)
    dt_s = args.dt_ms / 1000.0
    amp_rad_s = args.amp_deg_s * DEG2RAD
    amp_rad_s_per_seg = (
        [a * DEG2RAD for a in args.amp_deg_s_per_seg] if args.amp_deg_s_per_seg else None
    )
    freqs = list(args.freqs_hz)
    use_shared_burst = (
        args.profile in BURST_PROFILES
        and args.single_axis is None
        and not args.freqs_hz_per_seg
        and not args.amp_deg_s_per_seg
    )
    burst_vb = load_velocity_burst({"profile": args.profile}) if use_shared_burst else None

    data = ex.load_poses_yaml(POSES_YAML)
    rec = ex.get_slot_record(data, "d")
    if rec is None:
        print(f"pose d missing in {POSES_YAML}", file=sys.stderr)
        return 1
    q_tgt = np.asarray(rec["q_deg"], dtype=float)
    pose_ref = np.asarray(rec["pose_base"], dtype=float)

    from rm75_control import RobotSession

    with RobotSession(config=CONFIG_ROBOT) as bot:
        if not args.skip_move:
            print(f"\nMove to pose d ({rec.get('label', 'pose_d')})")
            move_j(bot.robot, q_tgt, speed=args.move_speed)
            pose0, q0 = wait_settle(bot.robot, q_tgt, timeout_s=args.settle_timeout_s)
        else:
            ret, st = bot.robot.rm_get_current_arm_state()
            if ret != 0:
                print(f"get state failed: {ret}", file=sys.stderr)
                return 1
            pose0 = np.asarray(st["pose"][:6], dtype=float)
            q0 = np.asarray(st["joint"][:7], dtype=float)

        dpos, deul = ex.pose_drift_mm_deg(pose0, pose_ref)
        print(f"  settled pose drift vs yaml: {dpos:.1f} mm, {deul:.2f} deg")

        err_codes = ensure_arm_ready(bot.robot, clear_err=args.clear_err)
        if err_codes:
            print(f"  !! system err: {err_codes}", file=sys.stderr)
            if ERR_4119 in err_codes:
                print(f"  {ERR_4119_HELP}", file=sys.stderr)
            print("  motion will likely be ignored until cleared.", file=sys.stderr)

        if args.diag:
            return run_diag(bot, args, pose0)

        if not args.no_prepare:
            prep = prepare_movev_session(bot)
            print(f"  movev prep: {prep}")
            if ERR_4119 in str(prep.get("err", {})):
                print(f"  {ERR_4119_HELP}", file=sys.stderr)

        n_cmd = int(args.duration / dt_s) + 1
        n_log = (n_cmd + args.log_every - 1) // args.log_every
        t_log = np.zeros(n_log)
        pose_log = np.zeros((n_log, 6))
        q_log = np.zeros((n_log, 7))
        f_log = np.zeros((n_log, 6))
        vel_cmd_log = np.zeros((n_log, 6))
        vel_fb_log = np.zeros((n_log, 3))
        qdot_fb_log = np.zeros((n_log, 7))
        seg_log = np.zeros(n_log, dtype=np.int8)

        prof = f" profile={args.profile}" if args.profile else ""
        print(
            f"\nPose D velocity burst{prof} | frame={args.frame_type} avoid={args.avoid_singularity} "
            f"follow={args.follow} traj={trajectory_mode} radio={radio}"
        )
        n_tones = max(1, len(freqs))
        cmd_peak_est = args.amp_deg_s * n_tones
        print(
            f"  amp={args.amp_deg_s:.1f} deg/s/tone  freqs={freqs}  "
            f"segment={args.segment_s:.0f}s  duration={args.duration:.0f}s"
        )
        if args.single_axis is not None:
            print(f"  single_axis={AXIS_NAMES[args.single_axis]} (full duration)")
        elif args.axis_order != (0, 1, 2):
            print(f"  axis_order={[AXIS_NAMES[i] for i in args.axis_order]}")
        if args.freqs_hz_per_seg:
            print(f"  freqs_per_seg={args.freqs_hz_per_seg}")
        print(
            f"  cmd peak ≈ {cmd_peak_est:.1f} deg/s ({n_tones} tone(s))  "
            f"ramp_down={args.ramp_down_s:.1f}s"
        )
        if use_shared_burst:
            print(f"  vel_cmd: utils.excitation.vel_burst_cmd (ramp_down={burst_vb.ramp_down_s:.1f}s)")
        if args.frame_type == 0:
            print(
                "  NOTE: frame=tool — collection preset pose_d_vel_burst uses frame=base (1)."
            )

        init_velocity_canfd(
            bot.robot,
            frame_type=burst_vb.frame_type if use_shared_burst else args.frame_type,
            avoid_singularity=burst_vb.avoid_singularity if use_shared_burst else args.avoid_singularity,
            dt_ms=args.dt_ms,
        )
        next_tick = time.monotonic()
        if use_shared_burst:
            next_tick = ex.settle_movev_after_init(
                bot.robot, vb=burst_vb, dt_ms=args.dt_ms, next_tick=next_tick,
            )
        else:
            next_tick = ex.settle_movev_stream(
                bot.robot,
                dt_ms=args.dt_ms,
                follow=args.follow,
                trajectory_mode=trajectory_mode,
                radio=radio,
                next_tick=next_tick,
            )
        print("Streaming… (Ctrl+C to abort)\n")

        log_i = 0
        prev_euler = pose0[3:6].copy()
        prev_q = q0.copy()
        prev_log_t = 0.0
        win_cmd = 0.0
        win_fb_eul = 0.0
        win_fb_q = 0.0
        win_tcp_dz = 0.0
        win_n = 0
        active_axis = 0
        last_report = time.monotonic()
        movev_errors = 0
        last_vel_cmd = np.zeros(6, dtype=float)
        ramped_down = False

        try:
            for i in range(n_cmd):
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(min(0.002, next_tick - now))
                next_tick += dt_s
                t_cmd = i * dt_s

                vel_cmd, seg = (
                    ex.vel_burst_cmd(t_cmd, burst_vb)
                    if use_shared_burst
                    else vel_burst_cmd(
                        t_cmd,
                        amp_rad_s=amp_rad_s,
                        freqs_hz=freqs,
                        segment_s=args.segment_s,
                        ramp_s=args.ramp_s,
                        axis_order=args.axis_order,
                        amp_rad_s_per_seg=amp_rad_s_per_seg,
                        freqs_hz_per_seg=args.freqs_hz_per_seg,
                        single_axis=args.single_axis,
                    )
                )
                active_axis = seg
                last_vel_cmd = vel_cmd.copy()
                ret = send_vel_checked(
                    bot.robot, vel_cmd.tolist(),
                    follow=args.follow, trajectory_mode=trajectory_mode, radio=radio,
                )
                if ret != 0:
                    movev_errors += 1

                if i % args.log_every == 0:
                    ret_s, st = bot.robot.rm_get_current_arm_state()
                    if ret_s == 0:
                        pose = np.asarray(st["pose"][:6], dtype=float)
                        q = np.asarray(st["joint"][:7], dtype=float)
                        dt_log = max(t_cmd - prev_log_t, dt_s)
                        omega_fb = np.array([
                            unwrap_delta(prev_euler[j], pose[3 + j]) / dt_log for j in range(3)
                        ])
                        qdot_fb = (q - prev_q) / dt_log
                        prev_euler = pose[3:6].copy()
                        prev_q = q.copy()
                        prev_log_t = t_cmd

                        pose_log[log_i] = pose
                        q_log[log_i] = q
                        vel_fb_log[log_i] = omega_fb
                        qdot_fb_log[log_i] = qdot_fb
                        win_tcp_dz = max(win_tcp_dz, (pose[2] - pose0[2]) * 1000)
                    else:
                        omega_fb = np.zeros(3)
                        qdot_fb = np.zeros(7)

                    if log_i % 5 == 0:
                        ret_f, fd = bot.robot.rm_get_force_data()
                        if ret_f == 0:
                            f_log[log_i] = np.asarray(fd["force_data"][:6], dtype=float)

                    t_log[log_i] = t_cmd
                    vel_cmd_log[log_i] = vel_cmd
                    seg_log[log_i] = seg
                    log_i += 1

                    ax = 3 + seg
                    win_cmd = max(win_cmd, abs(vel_cmd[ax]))
                    win_fb_eul = max(win_fb_eul, abs(omega_fb[seg]))
                    win_fb_q = max(win_fb_q, float(np.max(np.abs(qdot_fb))))
                    win_n += 1

                if time.monotonic() - last_report >= 1.0 and win_n > 0:
                    last_report = time.monotonic()
                    ratio_e = (win_fb_eul / win_cmd * 100.0) if win_cmd > 1e-6 else 0.0
                    print(
                        f"  t={t_cmd:5.1f}s  {AXIS_NAMES[active_axis]}  "
                        f"cmd={win_cmd * RAD2DEG:5.1f}  euler_fb={win_fb_eul * RAD2DEG:5.1f}  "
                        f"joint_fb={win_fb_q:5.1f} deg/s  tcp_dz={win_tcp_dz:+.1f}mm  "
                        f"track={ratio_e:4.0f}%",
                        flush=True,
                    )
                    win_cmd = win_fb_eul = win_fb_q = win_tcp_dz = 0.0
                    win_n = 0

            ramp_down_s = burst_vb.ramp_down_s if use_shared_burst else args.ramp_down_s
            print(f"\nRamping down {ramp_down_s:.1f}s…", flush=True)
            if use_shared_burst:
                next_tick = ex.ramp_down_velocity(
                    bot.robot, last_vel_cmd, vb=burst_vb, dt_ms=args.dt_ms, next_tick=next_tick,
                )
            else:
                next_tick = ramp_down_velocity(
                    bot.robot,
                    last_vel_cmd,
                    ramp_s=args.ramp_down_s,
                    dt_ms=args.dt_ms,
                    follow=args.follow,
                    trajectory_mode=trajectory_mode,
                    radio=radio,
                    next_tick=next_tick,
                )
            ramped_down = True

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            try:
                if not ramped_down:
                    try:
                        if use_shared_burst:
                            ex.ramp_down_velocity(
                                bot.robot, last_vel_cmd, vb=burst_vb, dt_ms=args.dt_ms,
                            )
                        else:
                            ramp_down_velocity(
                                bot.robot,
                                last_vel_cmd,
                                ramp_s=args.ramp_down_s,
                                dt_ms=args.dt_ms,
                                follow=args.follow,
                                trajectory_mode=trajectory_mode,
                                radio=radio,
                            )
                    except KeyboardInterrupt:
                        send_vel_checked(
                            bot.robot, [0.0] * 6,
                            follow=args.follow,
                            trajectory_mode=trajectory_mode,
                            radio=radio,
                        )
                        print("  ramp skipped (second Ctrl+C)", flush=True)
                bot.robot.rm_set_arm_slow_stop()
            except KeyboardInterrupt:
                pass
            except Exception:
                pass

    if log_i < 2:
        print("Too few samples logged.", file=sys.stderr)
        return 1

    stats = motion_stats(pose_log[:log_i], q_log[:log_i], t_log[:log_i])
    print(f"\n--- summary (movev errors: {movev_errors}) ---")
    print(f"  position drift: {stats.get('pos_drift_mm', 0):.2f} mm")
    print(f"  euler fb max (deg/s): {[round(v, 2) for v in stats.get('euler_deg_s_max', ())]}")
    print(f"  joint fb max (deg/s): {[round(v, 2) for v in stats.get('joint_deg_s_max', ())]}")
    for seg in range(3):
        mask = seg_log[:log_i] == seg
        if not np.any(mask):
            continue
        cmd_peak = float(np.max(np.abs(vel_cmd_log[:log_i][mask, 3 + seg])))
        fb_peak = float(np.max(np.abs(vel_fb_log[:log_i][mask, seg])))
        tr = (fb_peak / cmd_peak * 100.0) if cmd_peak > 1e-6 else 0.0
        print(
            f"  {AXIS_NAMES[seg]}: cmd_peak={cmd_peak * RAD2DEG:.1f}  "
            f"euler_fb_peak={fb_peak * RAD2DEG:.1f} deg/s  track={tr:.0f}%"
        )

    if stats.get("joint_deg_s_p90", 0) < 1.0 and movev_errors == 0:
        print(
            "\n>>> Arm barely moved (joint p90 < 1 deg/s). NOT 'parameter too large' — "
            "likely base-frame angular movev is ignored. Run --diag to isolate."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        t=t_log[:log_i],
        pose=pose_log[:log_i],
        q_deg=q_log[:log_i],
        force_raw=f_log[:log_i],
        vel_cmd=vel_cmd_log[:log_i],
        vel_fb_rad_s=vel_fb_log[:log_i],
        qdot_fb_deg_s=qdot_fb_log[:log_i],
        segment=seg_log[:log_i],
        pose0=pose0,
        q0_deg=q0,
        method="movev_angular_burst",
        amp_deg_s=args.amp_deg_s,
        freqs_hz=np.array(freqs),
        segment_s=args.segment_s,
        duration_s=args.duration,
        ramp_down_s=args.ramp_down_s,
        dt_ms=args.dt_ms,
        follow=args.follow,
        frame_type=args.frame_type,
        avoid_singularity=args.avoid_singularity,
        trajectory_mode=trajectory_mode,
        radio=radio,
        movev_errors=movev_errors,
        profile=args.profile or "",
        axis_order=np.array(args.axis_order, dtype=np.int8),
        single_axis=-1 if args.single_axis is None else int(args.single_axis),
    )
    print(f"\nSaved {args.out} ({log_i} samples)")
    print_kinematics_summary(args.out)
    return 0


def main() -> int:
    args = parse_args()
    if args.sweep and args.sweep_i:
        raise SystemExit("--sweep and --sweep-i are mutually exclusive")
    sweep_names = I_SWEEP_ORDER if args.sweep_i else (SWEEP_ORDER if args.sweep else None)
    if sweep_names:
        if args.diag:
            raise SystemExit("--sweep/--sweep-i cannot combine with --diag")
        rc = 0
        for name in sweep_names:
            print(f"\n{'=' * 60}\nPROFILE: {name}\n{'=' * 60}")
            sweep_args = argparse.Namespace(**vars(args))
            sweep_args.profile = name
            sweep_args.skip_move = False
            sweep_args.sweep = False
            sweep_args.sweep_i = False
            try:
                rc = run_burst_once(sweep_args) or rc
            except MotionError as exc:
                print(f"PROFILE {name} failed: {exc}", file=sys.stderr)
                rc = 1
            if name != sweep_names[-1]:
                print("Pausing 3s before next profile…", flush=True)
                time.sleep(3.0)
        label = "I-ID sweep" if args.sweep_i else "Sweep"
        print(f"\n{'=' * 60}\n{label} done. Compare logs:")
        for name in sweep_names:
            p = LOG_DIR / f"test_pose_d_vel_burst_{name}.npz"
            if p.exists():
                print_kinematics_summary(p)
        return rc
    return run_burst_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
