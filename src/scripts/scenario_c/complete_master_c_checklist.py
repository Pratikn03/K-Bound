#!/usr/bin/env python3
"""Run all automatable Master Scenario C checklist steps (infrastructure + optional training)."""

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


def _run(cmd: list[str], root: Path, *, label: str) -> int:
    print(f"\n=== {label} ===")
    print("$", " ".join(cmd))
    env = __import__("os").environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    return subprocess.call(cmd, cwd=root, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infra-only", action="store_true", help="Skip training (fast)")
    parser.add_argument("--train-seeds", type=int, default=5, help="Seeds for fusion retrain (use 1 for smoke)")
    parser.add_argument("--skip-mvtec-upgrade", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    py = sys.executable
    rc = 0

    steps = [
        ([py, "src/scripts/scenario_c/validate_master_c_governance.py"], "T0 governance"),
        ([py, "src/scripts/scenario_c/generate_split_hashes.py"], "T0 split hashes"),
        ([py, "src/scripts/scenario_c/freeze_domain_calibrators.py"], "T2 calibrator freeze"),
        ([py, "src/scripts/scenario_c/freeze_strongest_baselines.py"], "T3 strongest baseline freeze"),
        ([py, "src/scripts/scenario_c/qualify_upstream_experts.py"], "T1 Gate A (canonical)"),
    ]
    if not args.skip_mvtec_upgrade:
        steps.append(
            ([py, "src/scripts/scenario_c/upgrade_mvtec_experts.py"], "T1 MVTec expert upgrade v2")
        )
    steps.append(([py, "-m", "pytest", "tests/test_master_c_leakage_splits.py", "-q"], "T0 leakage tests"))

    if not args.infra_only:
        seed_arg = ["--seed", "42"] if args.train_seeds == 1 else []
        for cfg, out, meta in (
            (
                "configs/attention_real_fusion.yaml",
                "experiments/fusion/master_c_real_domain_results.json",
                "M0-ELARA-BENCH-LA",
            ),
            (
                "configs/attention_mvtec3d_patchcore_supervised_paired.yaml",
                "experiments/fusion/master_c_mvtec_supervised_paired_results.json",
                "M1-MVTEC-SUPERVISED-PAIRED",
            ),
        ):
            cmd = [
                py,
                "src/scripts/run_breakthrough_experiment.py",
                "--config",
                cfg,
                "--output",
                out,
                "--archive-root",
                "elara_master_c/predictions/development",
            ] + seed_arg
            steps.append((cmd, f"T3 fusion retrain {meta}"))

    steps.append(([py, "src/scripts/scenario_c/audit_checklist_progress.py"], "Final checklist audit"))

    for cmd, label in steps:
        rc = _run(cmd, root, label=label) or rc

    return rc


if __name__ == "__main__":
    sys.exit(main())
