#!/usr/bin/env python3
"""Transfer pipeline: calibrators → Eyecandies dev → expert upgrade → one-shot M2.

Implements the ordered path:
  1) freeze_domain_calibrators (isotonic on validation)
  2) Eyecandies development run (gate decision + calibration transfer)
  3) MVTec PatchCore v2 expert upgrade (optional)
  4) M2 external one-shot confirmatory (transfer v1 config)

Usage:
  PYTHONPATH=src python src/scripts/scenario_c/run_transfer_development_pipeline.py --dry-run
  PYTHONPATH=src python src/scripts/scenario_c/run_transfer_development_pipeline.py --skip-experts
  PYTHONPATH=src python src/scripts/scenario_c/run_transfer_development_pipeline.py --m2-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _run(cmd: list[str], root: Path, label: str, *, dry_run: bool) -> int:
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}")
    if dry_run:
        return 0
    env = __import__("os").environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    return subprocess.call(cmd, cwd=root, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-calibrators", action="store_true")
    parser.add_argument("--skip-eyecandies", action="store_true")
    parser.add_argument("--skip-experts", action="store_true")
    parser.add_argument("--skip-m2", action="store_true")
    parser.add_argument("--m2-only", action="store_true", help="Run only M2 confirmatory (transfer v1)")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    root = _repo_root()
    py = sys.executable
    rc = 0

    steps: list[tuple[str, list[str]]] = []
    lock_path = root / "elara_master_c/models/calibrators/calibrator_lock_v1.json"
    need_calibrators = not args.skip_calibrators and (
        not args.m2_only or not lock_path.is_file()
    )

    if not args.m2_only:
        if need_calibrators:
            steps.append(
                (
                    "T2 — freeze domain calibrators",
                    [py, "src/scripts/scenario_c/freeze_domain_calibrators.py"],
                )
            )
        if not args.skip_eyecandies:
            eyec_csv = root / "experiments/fusion/eyecandies_inputs.csv"
            if eyec_csv.is_file():
                steps.append(
                    (
                        "Dev — Eyecandies transfer v1",
                        [
                            py,
                            "src/scripts/run_breakthrough_experiment.py",
                            "--config",
                            "configs/attention_eyecandies_transfer_dev_v1.yaml",
                            "--output",
                            "experiments/fusion/eyecandies_transfer_dev_v1_seed42.json",
                            "--seed",
                            "42",
                        ],
                    )
                )
            else:
                print(f"WARN: skip Eyecandies dev — missing {eyec_csv}")
        if not args.skip_experts:
            steps.append(
                (
                    "Experts — MVTec PatchCore v2",
                    [py, "src/scripts/scenario_c/upgrade_mvtec_experts.py"],
                )
            )

    if need_calibrators and args.m2_only:
        steps.insert(
            0,
            (
                "T2 — freeze domain calibrators (required for transfer v1)",
                [py, "src/scripts/scenario_c/freeze_domain_calibrators.py", "--only", "m2_external_3d_adam"],
            ),
        )

    if not args.skip_m2:
        m2_inputs = root / "experiments/fusion/m2_external_3d_adam_sealed_inputs.csv"
        if not m2_inputs.is_file():
            print(f"ERROR: sealed M2 inputs missing: {m2_inputs}", file=sys.stderr)
            print("Run: python src/scripts/scenario_c/seal_m2_external_3d_adam.py", file=sys.stderr)
            return 1
        if not lock_path.is_file() and not args.dry_run:
            print(f"WARN: calibrator lock missing; run freeze_domain_calibrators first ({lock_path})")
        cmd = [
            py,
            "src/scripts/scenario_c/run_m2_external_confirmatory.py",
            "--transfer-v1",
            "--seeds",
            *[str(s) for s in args.seeds],
        ]
        steps.append(("M2 — one-shot external confirmatory (transfer v1)", cmd))

    for label, cmd in steps:
        rc = _run(cmd, root, label, dry_run=args.dry_run) or rc

    if not args.dry_run and not args.skip_m2:
        paired_cmd = [
            py,
            "src/scripts/scenario_c/run_m2_external_paired_inference.py",
            "--experiment-id",
            "M2-EXTERNAL-3D-ADAM-TRANSFER-V1",
            "--out-json",
            "experiments/fusion/m2_external_3d_adam_transfer_v1_paired_inference.json",
            "--out-csv",
            "experiments/fusion/m2_external_3d_adam_transfer_v1_paired_inference.csv",
        ]
        rc = _run(paired_cmd, root, "M2 — paired inference (transfer v1)", dry_run=False) or rc

    return rc


if __name__ == "__main__":
    sys.exit(main())
