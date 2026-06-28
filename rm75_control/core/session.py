"""Robot session: connect, mode switching, high-level entry point."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import yaml

from rm75_control.backend.realman import RealManBackend
from rm75_control.control.cartesian_pose import (
    CartesianPoseController,
    CartesianPoseStreamConfig,
)
from rm75_control.core.exceptions import ControlModeError, MotionError
from rm75_control.core.types import ControlMode
from rm75_control.force.scan import ForceScanConfig, ForceScanController

Pose6 = Sequence[float]


class RobotSession:
    """Top-level facade for init -> cartesian path -> reset workflows."""

    def __init__(
        self,
        ip: str | None = None,
        port: int | None = None,
        config: str | Path | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        self._config = self._load_config(config)
        robot_cfg = self._config.get("robot", {})
        self.ip = ip or robot_cfg.get("ip", "192.168.1.18")
        self.port = port or robot_cfg.get("port", 8080)
        self.thread_mode = robot_cfg.get("thread_mode", 2)
        self.dry_run = dry_run
        self.mode = ControlMode.IDLE
        self._backend: RealManBackend | None = None
        self._force_scan: ForceScanController | None = None

    @staticmethod
    def _load_config(config: str | Path | None) -> dict[str, Any]:
        if config is None:
            return {}
        path = Path(config)
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def backend(self) -> RealManBackend:
        if self._backend is None:
            raise RuntimeError("RobotSession is not connected")
        return self._backend

    @property
    def robot(self):
        return self.backend.robot

    def connect(self) -> None:
        print(f"Connecting to {self.ip}:{self.port}...", flush=True)
        self._backend = RealManBackend(
            self.ip,
            self.port,
            thread_mode=self.thread_mode,
        )
        if not self.dry_run:
            self._backend.connect()
        self.mode = ControlMode.IDLE
        print("Connected.", flush=True)

    def disconnect(self) -> None:
        if self._backend is not None and not self.dry_run:
            self.stop_all(hard=False)
            self._backend.disconnect()
        self._backend = None
        self.mode = ControlMode.IDLE

    def stop_motion(self, *, hard: bool = False) -> None:
        if self.dry_run or self._backend is None:
            self.mode = ControlMode.IDLE
            return
        if self.mode == ControlMode.FORCE_SCAN:
            raise ControlModeError(
                "In FORCE_SCAN mode; call stop_force_scan() before stop_motion()"
            )
        ret = (
            self.robot.rm_set_arm_stop()
            if hard
            else self.robot.rm_set_arm_slow_stop()
        )
        if ret != 0:
            raise MotionError(f"stop motion failed with code {ret}")
        self.mode = ControlMode.IDLE

    def stop_force_scan(self) -> None:
        if self._force_scan is not None:
            self._force_scan.stop()
            self._force_scan = None
        self.mode = ControlMode.IDLE

    def stop_all(self, *, hard: bool = False) -> None:
        """Stop force scan (if active) then stop planned/canfd motion."""
        self.stop_force_scan()
        if self._backend is not None and not self.dry_run:
            self.stop_motion(hard=hard)

    def _wait_planning_idle(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            traj = self.robot.rm_get_arm_current_trajectory()
            if traj.get("return_code") == 0 and traj.get("trajectory_type", 0) == 0:
                return True
            time.sleep(0.05)
        return False

    def prepare_for_force_stream(self, *, settle_s: float = 1.0) -> dict[str, Any]:
        """
        Exit planned force (rm_set_force_position) and stale CANFD force modes.

        rm_set_force_position + rm_start_force_position_move must never overlap;
        a failed mix can leave the controller rejecting stream start until recovery.
        """
        diag: dict[str, Any] = {}
        try:
            diag["stop_force_move"] = self.robot.rm_stop_force_position_move()
        except Exception:
            diag["stop_force_move"] = -999
        diag["stop_force"] = self.robot.rm_stop_force_position()
        traj = self.robot.rm_get_arm_current_trajectory()
        diag["trajectory_type"] = traj.get("trajectory_type", -1)
        if traj.get("trajectory_type", 0) != 0:
            diag["slow_stop"] = self.robot.rm_set_arm_slow_stop()
            diag["planning_idle"] = self._wait_planning_idle()
        if settle_s > 0.0:
            time.sleep(settle_s)
        return diag

    def recover_controller(
        self,
        *,
        settle_s: float = 1.0,
        clear_errors: bool = True,
        probe_force_stream: bool = False,
    ) -> dict[str, Any]:
        """
        Full controller cleanup before velocity CANFD (run every session start).

        Clears latched errors, exits force/plan modes, waits for planner idle.
        Optional force-stream probe unsticks rm_set_force_position conflicts.
        """
        diag: dict[str, Any] = {}
        if clear_errors:
            try:
                diag["clear_system_err"] = self.robot.rm_clear_system_err()
            except Exception:
                diag["clear_system_err"] = -999
            time.sleep(0.3)
            ret, st = self.robot.rm_get_current_arm_state()
            if ret == 0:
                err = st.get("err", {})
                diag["system_err"] = list(err.get("err", []))[: int(err.get("err_len", 0))]

        if not self.dry_run and self._backend is not None:
            self.stop_all(hard=False)
        diag.update(self.prepare_for_force_stream(settle_s=0.0))
        try:
            diag["delete_traj"] = self.robot.rm_set_arm_delete_trajectory()
        except Exception:
            diag["delete_traj"] = -999
        diag["slow_stop"] = self.robot.rm_set_arm_slow_stop()
        diag["planning_idle"] = self._wait_planning_idle(timeout_s=8.0)

        if probe_force_stream and not self.dry_run:
            try:
                ret = self.robot.rm_start_force_position_move()
                diag["force_stream_probe"] = ret
                if ret == 0:
                    self.robot.rm_stop_force_position_move()
                    time.sleep(0.3)
            except Exception:
                diag["force_stream_probe"] = -999

        if settle_s > 0.0:
            time.sleep(settle_s)

        traj = self.robot.rm_get_arm_current_trajectory()
        diag["trajectory_type_final"] = traj.get("trajectory_type", -1)
        ret, st = self.robot.rm_get_current_arm_state()
        if ret == 0:
            diag["pose_euler_deg"] = [
                round(float(v) * 180.0 / 3.141592653589793, 3) for v in st["pose"][3:6]
            ]
        self.mode = ControlMode.IDLE
        return diag

    def start_force_scan(self, **overrides: Any) -> ForceScanController:
        self.stop_force_scan()
        if self._backend is not None and not self.dry_run and self.mode != ControlMode.IDLE:
            self.stop_motion(hard=False)
        cfg = ForceScanConfig.from_config(self._config, **overrides)
        self._force_scan = ForceScanController(
            self.robot if not self.dry_run else _DryRunForceClient(),
            cfg,
            dry_run=self.dry_run,
        )
        last_err: MotionError | None = None
        for attempt in range(5):
            if self._backend is not None and not self.dry_run:
                diag = self.prepare_for_force_stream(settle_s=2.0 if attempt else 1.0)
                if attempt:
                    print(f"force stream recover attempt {attempt}: {diag}", flush=True)
            try:
                self._force_scan.start()
                last_err = None
                break
            except MotionError as exc:
                last_err = exc
        if last_err is not None:
            raise MotionError(
                f"{last_err}. Planned force (rm_set_force_position) and stream force "
                f"(rm_start_force_position_move) cannot be mixed. Run "
                f"python /media/camp/EXT_DRIVE/rm75_control/tmp/recover_force_stream.py "
                f"or stop/power-cycle on the teach pendant."
            ) from last_err
        self.mode = ControlMode.FORCE_SCAN
        return self._force_scan

    def _ensure_idle(self) -> None:
        if self.mode == ControlMode.FORCE_SCAN:
            raise ControlModeError(
                "In FORCE_SCAN mode; call stop_force_scan() first"
            )
        if self.mode not in (ControlMode.IDLE, ControlMode.PTP_PLANNED):
            raise ControlModeError(
                f"Cannot start motion while mode={self.mode.name}; call stop_motion() first"
            )

    def move_joints(
        self,
        joint: Sequence[float],
        *,
        velocity_percent: int | None = None,
        block: int = 1,
    ) -> None:
        if self.dry_run:
            self.mode = ControlMode.IDLE
            return

        self._ensure_idle()
        motion = self._config.get("motion", {})
        v = velocity_percent or motion.get("default_velocity_percent", 20)
        self.mode = ControlMode.PTP_PLANNED
        ret = self.robot.rm_movej(list(joint), v, 0, 0, block)
        self.mode = ControlMode.IDLE
        if ret != 0:
            raise MotionError(f"rm_movej failed with code {ret}")

    def move_cartesian_path(
        self,
        waypoints: Sequence[Pose6],
        *,
        use_ruckig: bool | None = None,
        follow: bool | None = None,
        trajectory_mode: int | None = None,
        radio: int | None = None,
        period_ms: float | None = None,
        steps_per_segment: int | None = None,
    ) -> None:
        """Cartesian pose CANFD. Native rm_movep_canfd params preserved; use_ruckig is the only extra switch."""
        self._ensure_idle()
        cfg = CartesianPoseStreamConfig.from_config(
            self._config,
            use_ruckig=use_ruckig,
            follow=follow,
            trajectory_mode=trajectory_mode,
            radio=radio,
            period_ms=period_ms,
            steps_per_segment=steps_per_segment,
        )
        controller = CartesianPoseController(
            self.robot if not self.dry_run else _DryRunCanfdClient(),
            cfg,
            dry_run=self.dry_run,
        )
        start_pose = None
        if not self.dry_run and self._backend is not None:
            try:
                start_pose = self._backend.get_tcp_pose()
            except Exception:
                start_pose = None
        self.mode = ControlMode.CARTESIAN_POSE_CANFD
        try:
            controller.run(waypoints, start_pose=start_pose)
        finally:
            self.mode = ControlMode.IDLE

    def run_init_path_reset(
        self,
        home_joint: Sequence[float],
        path_waypoints: Sequence[Pose6],
        *,
        use_ruckig: bool | None = None,
        follow: bool | None = None,
        trajectory_mode: int | None = None,
        radio: int | None = None,
    ) -> None:
        self.move_joints(home_joint)
        self.move_cartesian_path(
            path_waypoints,
            use_ruckig=use_ruckig,
            follow=follow,
            trajectory_mode=trajectory_mode,
            radio=radio,
        )
        self.stop_motion(hard=False)
        self.move_joints(home_joint)

    def __enter__(self) -> RobotSession:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()


class _DryRunCanfdClient:
    def rm_movep_canfd(self, pose, follow, trajectory_mode=0, radio=0) -> int:
        return 0


class _DryRunForceClient:
    def rm_start_force_position_move(self) -> int:
        return 0

    def rm_stop_force_position_move(self) -> int:
        return 0

    def rm_force_position_move(self, param) -> int:
        return 0
