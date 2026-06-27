#!/usr/bin/env python3
"""
One-shot force compensation calibration: collect A→B→C→D→A, then identify φ.

  source env.sh
  python tmp/force_compensation/force_calibrate.py
  python tmp/force_compensation/force_calibrate.py --dry-run
  python tmp/force_compensation/force_calibrate.py --save-pose d --pose-label pose_d_large
  python tmp/force_compensation/force_calibrate.py --identify-only
  python tmp/force_compensation/force_calibrate.py --collect-only

Config: tmp/force_compensation/config/force_id.yaml, config/poses.yaml
Output: logs/force_id_pose_{a,b,c,d}.npz, logs/force_id_phi.json (overwrite each run)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from utils import collection, identification  # noqa: E402
from utils.paths import CONFIG_ID  # noqa: E402


def _collect_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    if args.config != CONFIG_ID:
        argv += ["--config", str(args.config)]
    if args.dry_run:
        argv.append("--dry-run")
    if args.save_pose:
        argv += ["--save-pose", args.save_pose]
    if args.pose_label:
        argv += ["--pose-label", args.pose_label]
    return argv


def _identify_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    if args.config != CONFIG_ID:
        argv += ["--id-config", str(args.config)]
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect + identify force compensation φ")
    parser.add_argument("--config", type=Path, default=CONFIG_ID, help="config/force_id.yaml")
    parser.add_argument("--dry-run", action="store_true", help="preview collection only")
    parser.add_argument("--save-pose", type=str, default=None, metavar="SLOT")
    parser.add_argument("--pose-label", type=str, default=None)
    parser.add_argument("--collect-only", action="store_true", help="skip identification")
    parser.add_argument("--identify-only", action="store_true", help="skip collection, fit existing npz")
    args = parser.parse_args()

    if args.identify_only and (args.collect_only or args.dry_run or args.save_pose):
        parser.error("--identify-only cannot combine with --collect-only, --dry-run, or --save-pose")

    collect_argv = _collect_argv(args)

    if args.save_pose:
        return collection.main(collect_argv)

    if args.identify_only:
        return identification.main(_identify_argv(args))

    if args.dry_run:
        rc = collection.main(collect_argv)
        if rc != 0:
            return rc
        print("\n(dry-run: identification would run after collection)")
        return 0

    rc = collection.main(collect_argv)
    if rc != 0 or args.collect_only:
        return rc

    return identification.main(_identify_argv(args))


if __name__ == "__main__":
    raise SystemExit(main())
