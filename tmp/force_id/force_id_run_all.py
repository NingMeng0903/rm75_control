#!/usr/bin/env python3
"""
Full force-ID pipeline: collect A→B→C→D→A → OLS fit → validate at pose A.

  1. force_id_multi_pose.py   abc 30s cartesian, d joint+Stage-2 burst, return a
  2. force_id_fit.py          merge npz → force_id_phi.json
  3. force_id_validate.py     30s ORIENT compensation check at home (pose a)

Run:
  source env.sh
  python tmp/force_id/force_id_run_all.py --yes
  python tmp/force_id/force_id_run_all.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd: list[str], *, dry_run: bool) -> int:
    line = " ".join(cmd)
    print(f"\n>>> {line}\n", flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect + fit + validate (full ID pipeline)")
    parser.add_argument("--yes", action="store_true", help="skip interactive confirms")
    parser.add_argument("--dry-run", action="store_true", help="preview multi_pose only")
    parser.add_argument("--validate-s", type=float, default=30.0, help="validation duration at pose a")
    parser.add_argument("--skip-validate", action="store_true")
    # forwarded to multi_pose
    parser.add_argument("--duration-abc", type=float, default=30.0)
    parser.add_argument("--duration-d-joint", type=float, default=30.0)
    parser.add_argument("--duration-d-burst", type=float, default=45.0)
    parser.add_argument("--bc-max-deg", type=float, default=28.0)
    parser.add_argument("--max-orient-deg", type=float, default=18.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    collect = [
        PY,
        str(SCRIPT_DIR / "force_id_multi_pose.py"),
        "--duration-abc",
        str(args.duration_abc),
        "--duration-d-joint",
        str(args.duration_d_joint),
        "--duration-d-burst",
        str(args.duration_d_burst),
        "--bc-max-deg",
        str(args.bc_max_deg),
        "--max-orient-deg",
        str(args.max_orient_deg),
        "--scale",
        str(args.scale),
        "--log-every",
        str(args.log_every),
    ]
    if args.yes:
        collect.append("--yes")
    if args.dry_run:
        collect.append("--dry-run")

    rc = run(collect, dry_run=False)
    if rc != 0:
        return rc
    if args.dry_run:
        print("\n(dry-run stops before fit/validate)")
        return 0

    rc = run([PY, str(SCRIPT_DIR / "force_id_fit.py")], dry_run=False)
    if rc != 0:
        return rc

    if args.skip_validate:
        print("\nDone (validate skipped). φ → tmp/force_id/logs/force_id_phi.json")
        return 0

    validate = [
        PY,
        str(SCRIPT_DIR / "force_id_validate.py"),
        "--duration",
        str(args.validate_s),
    ]
    rc = run(validate, dry_run=False)
    if rc != 0:
        return rc

    print("\n=== Pipeline complete ===")
    print("  logs: tmp/force_id/logs/force_id_pose_{a,b,c,d}.npz")
    print("  φ:    tmp/force_id/logs/force_id_phi.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
