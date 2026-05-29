#!/usr/bin/env python3
"""One-shot M2 external confirmatory fusion (3D-ADAM, 5 seeds) + statistics."""

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


def _merge_seed_results(runs: list[dict]) -> dict:
    table = []
    for run in runs:
        for row in run.get("table_1_clean_performance") or []:
            table.append(row)
    base = runs[-1].copy()
    base["table_1_clean_performance"] = table
    base["n_seeds"] = len(table)
    methods = ["rga_boosted_fusion", "static_attention", "craf_attention", "sar_score_adapter"]
    summary: dict = {}
    for method in methods:
        vals = [
            float((row.get(method) or {}).get("roc_auc"))
            for row in table
            if isinstance((row.get(method) or {}).get("roc_auc"), (int, float))
        ]
        if vals:
            import statistics

            summary[method] = {
                "roc_auc": {
                    "mean": statistics.mean(vals),
                    "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
            }
    base["clean_metric_summary"] = summary
    base["benchmark"] = "3D-ADAM category-held-out"
    base["protocol"] = "M2_external_one_shot_audit"
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--skip-stats", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    py = sys.executable
    seal = root / "research_lock/M2_EXTERNAL_SEALED_v1.yaml"
    if not seal.is_file():
        print("ERROR: M2_EXTERNAL_SEALED_v1.yaml missing", file=sys.stderr)
        return 1
    inputs = root / "experiments/fusion/m2_external_3d_adam_sealed_inputs.csv"
    if not inputs.is_file():
        print("ERROR: sealed inputs missing — run seal_m2_external_3d_adam.py", file=sys.stderr)
        return 1

    merged_out = root / "experiments/fusion/m2_external_3d_adam_confirmatory_results.json"
    all_results: list[dict] = []
    rc = 0
    for seed in args.seeds:
        seed_out = root / f"experiments/fusion/m2_external_3d_adam_confirmatory_seed{seed}.json"
        cmd = [
            py,
            "src/scripts/run_breakthrough_experiment.py",
            "--config",
            "configs/attention_m2_external_3d_adam_sealed.yaml",
            "--output",
            str(seed_out),
            "--archive-root",
            "elara_master_c/predictions/confirmation",
            "--seed",
            str(seed),
        ]
        rc = _run(cmd, root, f"M2 external confirmatory seed={seed}") or rc
        if seed_out.is_file():
            all_results.append(json.loads(seed_out.read_text(encoding="utf-8")))

    if not all_results:
        print("ERROR: no seed results produced", file=sys.stderr)
        return 1

    merged = _merge_seed_results(all_results)
    merged_out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Merged external M2 results -> {merged_out}")

    if not args.skip_stats:
        rc = _run([py, "src/scripts/scenario_c/confirmatory_statistics.py"], root, "Confirmatory statistics") or rc
        rc = _run([py, "src/scripts/scenario_c/audit_checklist_progress.py"], root, "Checklist audit") or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
