"""CANFD pose and velocity streaming + unified handoff (traj0 only)."""

from __future__ import annotations

import time
from typing import Any, Protocol, Sequence

import numpy as np

from rm75_control.core.exceptions import MotionError

Pose6 = Sequence[float]
Vel6 = Sequence[float]

MAX_LINEAR_V_M_S = 0.25
MAX_ANGULAR_V_RAD_S = 0.6

TRAJ0_MODE = 0
TRAJ0_RADIO = 0

LATCHED_ERR_CODES = frozenset({"4119", "1003", "4099", "0x1003"})


class PoseCanfdClient(Protocol):
    def rm_movep_canfd(
        self,
        pose: list[float],
        follow: bool,
        trajectory_mode: int = 0,
        radio: int = 0,
    ) -> int:
        ...


def send_pose_canfd(
    robot: PoseCanfdClient,
    pose: Pose6,
    *,
    follow: bool = True,
    trajectory_mode: int = 0,
    radio: int = 0,
) -> None:
    del trajectory_mode, radio
    if len(pose) not in (6, 7):
        raise ValueError(f"pose must have 6 (euler) or 7 (quat) elements, got {len(pose)}")

    ret = robot.rm_movep_canfd(
        list(pose),
        follow,
        TRAJ0_MODE,
        TRAJ0_RADIO,
    )
    if ret != 0:
        raise MotionError(f"rm_movep_canfd failed with code {ret}")


class VelocityCanfdClient(Protocol):
    def rm_movev_canfd(
        self,
        cartesian_velocity: list[float],
        follow: bool,
        trajectory_mode: int = 0,
        radio: int = 0,
    ) -> int:
        ...


def clamp_cartesian_velocity(vel: Vel6) -> list[float]:
    out = list(vel)
    for i in range(3):
        out[i] = max(-MAX_LINEAR_V_M_S, min(MAX_LINEAR_V_M_S, out[i]))
    for i in range(3, 6):
        out[i] = max(-MAX_ANGULAR_V_RAD_S, min(MAX_ANGULAR_V_RAD_S, out[i]))
    return out


def send_velocity_canfd(
    robot: VelocityCanfdClient,
    cartesian_velocity: Vel6,
    *,
    follow: bool = False,
    trajectory_mode: int = 0,
    radio: int = 0,
) -> None:
    del trajectory_mode, radio
    ret = robot.rm_movev_canfd(
        clamp_cartesian_velocity(cartesian_velocity),
        follow,
        TRAJ0_MODE,
        TRAJ0_RADIO,
    )
    if ret != 0:
        raise MotionError(f"rm_movev_canfd failed with code {ret}")


def _wait_planning_idle(robot, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        traj = robot.rm_get_arm_current_trajectory()
        if traj.get("return_code") == 0 and traj.get("trajectory_type", 0) == 0:
            return True
        time.sleep(0.05)
    return False


def read_system_err(robot) -> list[str]:
    ret, st = robot.rm_get_current_arm_state()
    if ret != 0:
        return [f"get_state:{ret}"]
    err = st.get("err", {})
    n = int(err.get("err_len", 0))
    return [str(c) for c in list(err.get("err", []))[:n]]


def _should_clear_err(codes: list[str]) -> bool:
    if not codes or codes == ["0"]:
        return False
    for c in codes:
        if c in LATCHED_ERR_CODES:
            return True
        if "1003" in c or "4119" in c:
            return True
    return False


def _format_system_err(codes: list[str]) -> str:
    """Realman returns ['0'] when no latched error — not a fault."""
    if not codes:
        return "none"
    if len(codes) == 1 and str(codes[0]) in ("0", ""):
        return "none"
    return str(codes)


def print_handoff_diag(diag: dict[str, Any], *, prefix: str = "  handoff") -> None:
    parts = [
        f"slow_stop={diag.get('slow_stop')}",
        f"pause={diag.get('pause')}",
        f"delete={diag.get('delete_traj')}",
        f"idle={diag.get('planning_idle')}",
        f"traj={diag.get('trajectory_type')}",
    ]
    if diag.get("q_drift_deg") is not None:
        parts.append(f"q_drift={diag['q_drift_deg']:.2f}°")
    if diag.get("resync_done"):
        parts.append("resync=1")
    err = _format_system_err(diag.get("system_err") or [])
    parts.append(f"err={err}")
    if diag.get("quiescent_max_step_mm") is not None:
        parts.append(f"quiesce={diag.get('quiescent_frames')}f/{diag['quiescent_max_step_mm']:.2f}mm")
    print(f"{prefix}: " + "  ".join(parts), flush=True)


def exit_canfd_session(
    robot,
    *,
    q_resync: np.ndarray | None = None,
    settle_sleep_s: float = 0.3,
    move_speed: int = 15,
    settle_timeout_s: float = 15.0,
    resync_threshold_deg: float = 1.0,
    print_diag: bool = False,
) -> dict[str, Any]:
    """
    Unified CANFD exit: slow_stop → pause → delete → idle → sleep → resync → clear err.

    Call before move_j, between slots, and before enter_movev_session.
    """
    diag: dict[str, Any] = {}
    diag["slow_stop"] = robot.rm_set_arm_slow_stop()
    time.sleep(0.1)
    diag["pause"] = robot.rm_set_arm_pause()
    diag["delete_traj"] = robot.rm_set_arm_delete_trajectory()
    diag["planning_idle"] = _wait_planning_idle(robot)
    time.sleep(settle_sleep_s)

    if q_resync is not None:
        q_tgt = np.asarray(q_resync, dtype=float)
        ret, st = robot.rm_get_current_arm_state()
        if ret == 0:
            q_act = np.asarray(st["joint"][:7], dtype=float)
            q_drift = float(np.max(np.abs(q_act - q_tgt)))
            diag["q_drift_deg"] = q_drift
            if q_drift > resync_threshold_deg:
                from rm75_control.force.compensation.collection import move_j, wait_settle

                move_j(robot, q_tgt, speed=move_speed)
                wait_settle(robot, q_tgt, timeout_s=settle_timeout_s)
                diag["resync_done"] = True

    diag["system_err_before"] = read_system_err(robot)
    if _should_clear_err(diag["system_err_before"]):
        try:
            diag["clear_system_err"] = robot.rm_clear_system_err()
        except Exception:
            diag["clear_system_err"] = -999
        time.sleep(0.2)

    traj = robot.rm_get_arm_current_trajectory()
    diag["trajectory_type"] = traj.get("trajectory_type", -1)
    diag["system_err"] = read_system_err(robot)

    if print_diag:
        print_handoff_diag(diag)
    return diag


def settle_movev_after_init(
    robot,
    *,
    dt_ms: float,
    follow: bool,
    n_frames: int = 30,
    next_tick: float | None = None,
) -> float:
    """
    Zero-velocity frames after rm_set_movev_canfd_init — always low follow.

    rm_set_movev_canfd_init captures the current joint state as the IK
    reference.  If the arm is still micro-vibrating (which is common <200ms
    after move_j), with follow=True the high-bandwidth servo sees the residual
    error and issues a large corrective velocity → visible twitch/snap.

    Sending ALL settle frames with follow=False (低跟随, gentler servo) keeps
    the correction velocity small regardless of when init was called.  The
    actual scan commands use follow=True from the first tick of the main loop,
    by which point the arm is already confirmed quiescent.
    """
    dt_s = dt_ms / 1000.0
    if next_tick is None:
        next_tick = time.monotonic()
    zero = [0.0] * 6
    for _ in range(n_frames):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
        next_tick += dt_s
        send_velocity_canfd(robot, zero, follow=False)
    return next_tick


def wait_movev_quiescent(
    robot,
    *,
    dt_ms: float,
    follow: bool,
    settle_mm: float = 0.3,
    need_consecutive: int = 5,
    max_frames: int = 200,
    next_tick: float | None = None,
) -> tuple[np.ndarray | None, float, int, float]:
    """
    Stream zero velocity until TCP motion < settle_mm for need_consecutive ticks.

    Uses follow=False (低跟随) to match settle_movev_after_init — quiescence is
    measured under the same gentle-servo conditions the arm will settle in.
    The actual session's follow mode takes effect from the first real command.

    Returns (last_pose, max_step_mm, frames_used, next_tick).
    """
    dt_s = dt_ms / 1000.0
    if next_tick is None:
        next_tick = time.monotonic()
    del follow  # low follow throughout; see docstring
    zero = [0.0] * 6
    prev_xyz: np.ndarray | None = None
    last_pose: np.ndarray | None = None
    quiet = 0
    max_step_mm = 0.0
    for k in range(max_frames):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
        next_tick += dt_s
        send_velocity_canfd(robot, zero, follow=False)
        ret, st = robot.rm_get_current_arm_state()
        if ret != 0:
            continue
        pose = np.asarray(st["pose"][:6], dtype=float)
        last_pose = pose
        if prev_xyz is not None:
            step_mm = float(np.linalg.norm((pose[:3] - prev_xyz) * 1000.0))
            max_step_mm = max(max_step_mm, step_mm)
            if step_mm < settle_mm:
                quiet += 1
                if quiet >= need_consecutive:
                    return last_pose, max_step_mm, k + 1, next_tick
            else:
                quiet = 0
        prev_xyz = pose[:3].copy()
    return last_pose, max_step_mm, max_frames, next_tick


def enter_movev_session(
    robot,
    *,
    frame_type: int,
    avoid_singularity: int,
    dt_ms: float,
    follow: bool,
    q_resync: np.ndarray | None = None,
    settle_frames: int = 50,
    quiescent_mm: float = 0.25,
    quiescent_consecutive: int = 10,
    move_speed: int = 15,
    settle_timeout_s: float = 15.0,
    pre_init_settle_s: float = 1.5,
    next_tick: float | None = None,
    print_diag: bool = False,
) -> tuple[float, dict[str, Any]]:
    """
    Full movev handoff: exit → (settle) → init → zero frames → quiescence.

    pre_init_settle_s: extra sleep between exit_canfd_session and
    rm_set_movev_canfd_init.  exit_canfd_session already sleeps 0.3s, so the
    total static time before init = 0.3 + pre_init_settle_s.  Default 1.5s
    gives 1.8s total — arm must be fully static before init captures FK.

    Returns (next_tick, diag).
    """
    diag = exit_canfd_session(
        robot,
        q_resync=q_resync,
        move_speed=move_speed,
        settle_timeout_s=settle_timeout_s,
        print_diag=False,
    )

    if pre_init_settle_s > 0.0:
        time.sleep(pre_init_settle_s)

    # rm_set_movev_canfd_init can transiently refuse if the controller's internal
    # CANFD state machine (joint-CANFD layer) hasn't fully reset yet — the
    # trajectory-planner idle check doesn't cover that layer.  Retry with a
    # short light-cleanup cycle between attempts.
    _INIT_RETRIES = 3
    _INIT_RETRY_SLEEP_S = 0.5
    ret = -1
    for attempt in range(_INIT_RETRIES):
        ret = robot.rm_set_movev_canfd_init(avoid_singularity, frame_type, int(dt_ms))
        if ret == 0:
            break
        if attempt < _INIT_RETRIES - 1:
            robot.rm_set_arm_pause()
            robot.rm_set_arm_delete_trajectory()
            _wait_planning_idle(robot, timeout_s=5.0)
            time.sleep(_INIT_RETRY_SLEEP_S)
    diag["movev_init"] = ret
    diag["movev_init_attempts"] = attempt + 1
    if ret != 0:
        raise RuntimeError(
            f"rm_set_movev_canfd_init failed after {_INIT_RETRIES} attempts: {ret}"
        )

    next_tick = settle_movev_after_init(
        robot, dt_ms=dt_ms, follow=follow,
        n_frames=settle_frames, next_tick=next_tick,
    )
    pose_quiet, max_step_mm, q_frames, next_tick = wait_movev_quiescent(
        robot, dt_ms=dt_ms, follow=follow,
        settle_mm=quiescent_mm, need_consecutive=quiescent_consecutive,
        next_tick=next_tick,
    )
    diag["quiescent_max_step_mm"] = max_step_mm
    diag["quiescent_frames"] = q_frames
    if pose_quiet is not None:
        diag["quiescent_pose"] = pose_quiet.tolist()

    if print_diag:
        print_handoff_diag(diag)
        attempts = diag.get("movev_init_attempts", 1)
        if attempts > 1:
            print(f"  movev_init: needed {attempts} attempts", flush=True)
        if max_step_mm > 0:
            print(
                f"  movev quiescence: {q_frames} frames (max step {max_step_mm:.2f}mm/tick)",
                flush=True,
            )
    return next_tick, diag
