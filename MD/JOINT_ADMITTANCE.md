# Joint-Position QP-CLIK Inner Loop — Bring-up Guide

Cascaded controller for the RM75-F: the task-space **admittance outer loop**
(`rm75_control.control.hybrid_motion.AdmittanceController`) emits a 6D Cartesian
twist; the **joint-space inner loop** (`rm75_control.control.joint_admittance`)
turns it into absolute joint angles streamed via `rm_movej_canfd` — a single
interface, no MoveJ/MoveV mode switching once running.

## Package layout

```
rm75_control/control/joint_admittance/
├── model.py            # Pinocchio FK / 6x7 Jacobian / manipulability at `tcp`;
│                       # pose_error / pose_distance shared pose-math utilities
├── clik.py             # Phase 1: DLS (sigma-scheduled lambda) + nullspace CLIK
├── validation.py       # FK-vs-robot check (<1mm / <0.1deg) — run FIRST
├── config.py           # yaml -> JointIkConfig
├── reference.py        # MotionReferenceSources: HoldReference, CartesianMoveReference
│                       # (Cartesian-line smoothstep), JointSmoothMoveReference
│                       # (JOINT-space smoothstep -> MoveJ-like, no nullspace drift),
│                       # SinToolYReference (tool-Y sin)
├── loop.py             # JointIkController (inner, .update()=Cartesian CLIK/QP,
│                       # .update_joint()=joint-space PD+FF, bypasses IK entirely);
│                       # CartesianTrackOuterLoop (pure position tracking, no force),
│                       # JointSpaceMoveOuterLoop (feeds .update_joint()) + Admittance-
│                       # OuterLoop (force-position hybrid); Phase(mode="cartesian"|
│                       # "joint") + run_joint_admittance_phases (multi-phase on-robot
│                       # orchestration, one continuous stream)
├── tasks/nullspace_task.py    # joint centering + limit avoidance (Liegeois)
├── solver/
│   ├── constraint_mgr.py      # per-tick box bounds on qdot
│   └── qp_builder.py          # Phase 2: ProxQP (warm-start) / OSQP core
└── utils/
    ├── safety.py       # velocity/accel/position clamp + Watchdog
    ├── smoothing.py    # 1st/2nd-order low-pass, moving average
    └── friction.py     # optional Phase 3 stiction feed-forward (off by default)

assets/robots/rm75_6f/RM75-6F.urdf   # kinematics-only URDF (tcp = link_7 +0.220m Z)
configs/joint_admittance.yaml         # tuning
apps/joint_admittance/run_joint_admittance.py   # generic entry point (config-driven)
tmp/joint_admittance/d_sin_tool_y.py  # specific test: move to pose D, then tool-Y
                                       # sin scan there — see "Real-robot test" below
tests/test_joint_ik_offline.py        # offline closed-loop validation
```

## Dependencies

Installed in the `rm75` env: `pin` (Pinocchio 4.x), `proxsuite`, `osqp`, `scipy`.
`pip install -r requirements.txt` covers them.

## Bring-up order (do NOT skip step 1)

1. **FK validation (重中之重).** Prove the URDF matches the robot before any motion:
   ```
   source env.sh
   python -m rm75_control.control.joint_admittance.validation --ip 192.168.1.18
   ```
   - Default `--mode flange` is tool-agnostic (validates the arm chain).
   - `--mode tcp` requires the active Realman tool frame to be the +220 mm probe.
   - Drive fixed points: `--poses tmp/force_compensation/config/poses.yaml --move`.
   - Must report **PASS** (<1 mm / <0.1°).  If not, fix the URDF base rotation /
     TCP offset (`link_7_to_tcp` origin) — every Jacobian depends on it.

2. **Offline closed-loop sanity** (no robot):
   ```
   python tests/test_joint_ik_offline.py
   ```
   Confirms zero drift, reference tracking, nullspace-TCP orthogonality, limits.

3. **First on-robot run — CLIK, hold, no force, conservative limits.**
   In `configs/joint_admittance.yaml`: `inner.solver: clik`, `startup.reference: hold`,
   `startup.enable_force: false`, `inner.v_scale: 0.35`.
   ```
   python apps/joint_admittance/run_joint_admittance.py --config configs/joint_admittance.yaml --duration 10
   ```
   Expect the arm to hold pose smoothly. Watch for `[WATCHDOG FIRED]` and jitter.

4. **Enable force** once `tmp/force_compensation/logs/force_id_phi.json` exists
   (run `force_calibrate.py`): set `startup.enable_force: true`, `force.desired_z_n`.
   Bring the probe into contact; verify constant-force hold on tool-Z.

5. **Real-robot test: Cartesian move to pose D, then force-position sin scan there**
   (both through this cascade, no MoveV/MoveJ switch at the handoff):
   ```
   source env.sh
   # dry run first (no robot connection):
   python tmp/joint_admittance/d_sin_tool_y.py --dry-run

   # move to pose D only (proves pure Cartesian trajectory tracking):
   python tmp/joint_admittance/d_sin_tool_y.py --scan-duration 0

   # move to D, then 30s of tool-Y sin scan holding 3N on tool-Z
   # (requires a calibrated tmp/force_compensation/logs/force_id_phi.json):
   python tmp/joint_admittance/d_sin_tool_y.py --enable-force --desired-z 3.0 --scan-duration 30

   # tune the move / sweep:
   python tmp/joint_admittance/d_sin_tool_y.py --move-duration 10 \
       --y-pp-cm 10 --max-vel-cm-s 2 --enable-force --desired-z 3.0 --scan-duration 60
   ```
   Phase 1 (default) does **not** force a Cartesian straight line and does **not**
   call vendor IK.  It solves ``q_target = pose_ik.solve_pose_ik(pose D)`` once
   using our Pinocchio DLS solver, smoothsteps ``q(t)`` in joint space, and
   executes through the **live CLIK/QP inner loop** with nullspace ``q_ref``
   tracking (``Phase.q_ref_provider`` + ``nullspace.k_joint_ref``) so redundant
   DOF follows the planned joint path while nullspace remains available for
   centering and future obstacle-avoidance gradients.  Use ``--legacy-cartesian-move``
   only when you explicitly need a forced Cartesian line (comparison/debug). Phase 1
   ends early once within 2 mm / 1° of D. Phase 2 switches (in-place, same inner loop,
   same stream) to ``AdmittanceOuterLoop`` + ``SinToolYReference`` for the tool-Y sweep
   with tool-Z force hold. Ctrl+C stops cleanly.

6. **Add any other trajectory** by swapping in a different ``MotionReferenceSource``.
   Point-to-point repositioning: ``JointSmoothMoveReference`` + ``pose_ik.solve_pose_ik``
   + ``CartesianTrackOuterLoop`` + ``Phase.q_ref_provider`` (NOT ``update_joint``).
   Obstacle avoidance: add gradients via ``CompositeNullspaceTask`` in
   ``tasks/nullspace_task.py``.

7. **Upgrade to QP** (hard inequality limits): `inner.solver: qp`.  ProxQP is
   warm-started (full inner tick ~0.25 ms, well inside the 10 ms budget).  OSQP is
   the fallback (`inner.qp.backend: osqp`).

## Key tuning knobs (`configs/joint_admittance.yaml`)

| Key | Meaning |
|-----|---------|
| `inner.clik.k_task` | Cartesian error feedback gain (1/s). Higher = tighter tracking, less lag. |
| `inner.clik.sigma_thresh` / `lambda_max` | DLS damping schedule: `lambda=0` above `sigma_thresh`, ramps to `lambda_max` at singularity. |
| `inner.nullspace.k_center` / `k_limit` | Redundancy: pull to mid-range / repel from limits. |
| `inner.nullspace.k_joint_ref` | During joint-path moves: track `q_ref(t)` in nullspace (pins redundant DOF). |
| `inner.v_scale` / `a_max` | Fraction of URDF joint speed / accel clamp. Start low, raise gradually. |
| `inner.smooth_cutoff_hz` | q_cmd low-pass cutoff. Implemented as two cascaded 1st-order stages (unconditionally stable at any cutoff/dt). Ramp-tracking lag falls ~1/cutoff; jerk reduction falls as you raise it. 20 Hz is a safe start at 100 Hz control; raise toward 30–40 Hz once bring-up confirms no current-spike faults. (An earlier explicit-Euler 2nd-order discretization was only conditionally stable and blew up above ~17 Hz at 100 Hz control — fixed; see `tests/test_joint_ik_offline.py::test_smoothing_filter_stable_at_high_cutoff`.) |
| `inner.qp.reg` / `task_weight` | QP regularization / per-axis task weighting. |
| `startup.watchdog_timeout_s` | Loop-stall trip time -> hold at last q. |
| `startup.realtime` | Attempt SCHED_FIFO (needs CAP_SYS_NICE / root). |

## Safety notes

- `rm_movej_canfd` faults on discontinuities: the inner loop always low-passes and
  clamps velocity/acceleration/position before sending. Keep `v_scale` conservative.
- The Watchdog holds at the last commanded joint state (or slow-stops) if the loop
  stalls. Keep the loop body free of blocking I/O (state reads are async).
- j7 limit is `±6.28` in the URDF (looks like a placeholder). Confirm against the
  official RM75 manual before relaxing `v_scale` or trusting QP box bounds at j7.
- For lowest jitter: PREEMPT_RT kernel + `startup.realtime: true`. Without it,
  absolute `perf_counter` scheduling still bounds drift.
