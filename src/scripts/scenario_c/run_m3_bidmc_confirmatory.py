#!/usr/bin/env python3
"""M3 BIDMC confirmatory fusion (5 seeds) from local raw healthcare data."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _run(cmd: list[str], root: Path, label: str) -> int:
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}")
    env = __import__("os").environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    return subprocess.call(cmd, cwd=root, env=env)


def _merge(runs: list[dict]) -> dict:
    table = []
    for run in runs:
        table.extend(run.get("table_1_clean_performance") or [])
    base = runs[-1].copy()
    base["table_1_clean_performance"] = table
    base["n_seeds"] = len(table)
    base["benchmark"] = "BIDMC patient-stratified"
    base["protocol"] = "M3_bidmc_confirmatory"
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--skip-prep", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    py = sys.executable
    if not args.skip_prep:
        rc = _run(
            [py, "src/scripts/prepare_bidmc_healthcare_fusion_benchmark.py"],
            root,
            "Prepare BIDMC fusion CSV",
        )
        if rc != 0:
            return rc

    merged_out = root / "experiments/fusion/m3_bidmc_confirmatory_results.json"
    runs: list[dict] = []
    rc = 0
    for seed in args.seeds:
        seed_out = root / f"experiments/fusion/m3_bidmc_confirmatory_seed{seed}.json"
        cmd = [
            py,
            "src/scripts/run_breakthrough_experiment.py",
            "--config",
            "configs/attention_m3_bidmc_confirmatory.yaml",
            "--output",
            str(seed_out),
            "--archive-root",
            "elara_master_c/predictions/confirmation",
            "--seed",
            str(seed),
        ]
        rc = _run(cmd, root, f"M3 BIDMC seed={seed}") or rc
        if seed_out.is_file():
            runs.append(json.loads(seed_out.read_text(encoding="utf-8")))

    if not runs:
        return 1
    merged_out.write_text(json.dumps(_merge(runs), indent=2), encoding="utf-8")
    if not args.skip_stats:
        rc = _run([py, "src/scripts/scenario_c/confirmatory_statistics.py", "--write-report"], root, "Stats") or rc
        rc = _run([py, "src/scripts/scenario_c/audit_checklist_progress.py"], root, "Checklist") or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
