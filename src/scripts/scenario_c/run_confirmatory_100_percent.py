#!/usr/bin/env python3
"""Execute remaining Master Scenario C confirmatory steps toward 100%."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
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


def seal_m2_prepare(root: Path, py: str) -> int:
    cmd = [
        py,
        "src/scripts/prepare_mvtec3d_fusion_benchmark.py",
        "--dataset-root",
        "data/raw/mvtec3d",
        "--feature-mode",
        "patchcore",
        "--embedding-dim",
        "32",
        "--patchcore-k",
        "5",
        "--patchcore-coreset-size",
        "2048",
        "--train-categories",
        "foam",
        "peach",
        "rope",
        "tire",
        "--heldout-val-fraction",
        "0.15",
        "--heldout-val-seed",
        "20260528",
        "--output",
        "experiments/fusion/m2_confirmatory_sealed_inputs.csv",
        "--metadata",
        "experiments/fusion/m2_confirmatory_sealed_metadata.json",
    ]
    rc = _run(cmd, root, "D3 seal + prepare M2 inverted held-out")
    if rc != 0:
        return rc
    _run([py, "src/scripts/scenario_c/generate_split_hashes.py"], root, "Refresh split hashes")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-m2-prepare", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    root = _repo_root()
    py = sys.executable
    rc = 0

    if not args.skip_m2_prepare:
        rc = seal_m2_prepare(root, py) or rc

    # One-shot M2 confirmatory fusion (multi-seed for CI only; no test-driven selection)
    m2_out = root / "experiments/fusion/m2_confirmatory_sealed_results.json"
    if not m2_out.is_file() or args.seeds:
        all_results: list[dict] = []
        for seed in args.seeds:
            seed_out = root / f"experiments/fusion/m2_confirmatory_sealed_seed{seed}.json"
            cmd = [
                py,
                "src/scripts/run_breakthrough_experiment.py",
                "--config",
                "configs/attention_m2_confirmatory_sealed.yaml",
                "--output",
                str(seed_out),
                "--archive-root",
                "elara_master_c/predictions/confirmation",
                "--seed",
                str(seed),
            ]
            rc = _run(cmd, root, f"M2 confirmatory fusion seed={seed}") or rc
            if seed_out.is_file():
                all_results.append(json.loads(seed_out.read_text(encoding="utf-8")))

        if all_results:
            merged = _merge_seed_results(all_results)
            m2_out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            print(f"Merged M2 results -> {m2_out}")

    # T5 multi-seed on M1 if needed
    m1_out = root / "experiments/fusion/m1_confirmatory_t5_results.json"
    if not m1_out.is_file():
        all_m1: list[dict] = []
        for seed in args.seeds:
            p = root / f"experiments/fusion/m1_confirmatory_seed{seed}.json"
            cmd = [
                py,
                "src/scripts/run_breakthrough_experiment.py",
                "--config",
                "configs/attention_mvtec3d_patchcore_supervised_paired.yaml",
                "--output",
                str(p),
                "--seed",
                str(seed),
            ]
            rc = _run(cmd, root, f"M1 T5 fusion seed={seed}") or rc
            if p.is_file():
                all_m1.append(json.loads(p.read_text(encoding="utf-8")))
        if all_m1:
            m1_out.write_text(json.dumps(_merge_seed_results(all_m1), indent=2), encoding="utf-8")
            # also refresh master_c copy for stats
            (root / "experiments/fusion/master_c_mvtec_supervised_paired_results.json").write_text(
                m1_out.read_text(encoding="utf-8")
            )

    steps = [
        ([py, "src/scripts/scenario_c/confirmatory_statistics.py"], "Confirmatory statistics"),
        ([py, "src/scripts/scenario_c/audit_gate_decision_rule_e2e.py"], "T6 GDR re-audit"),
        ([py, "src/scripts/scenario_c/audit_checklist_progress.py"], "Checklist audit"),
    ]
    for cmd, label in steps:
        rc = _run(cmd, root, label) or rc
    return rc


def _merge_seed_results(runs: list[dict]) -> dict:
    """Merge per-seed breakthrough JSONs into one payload with table_1_clean_performance."""
    table = []
    for run in runs:
        for row in run.get("table_1_clean_performance") or []:
            table.append(row)
    base = runs[-1].copy()
    base["table_1_clean_performance"] = table
    base["n_seeds"] = len(table)
    # recompute clean_metric_summary means for key methods
    methods = ["rga_boosted_fusion", "static_attention", "craf_attention", "sar_score_adapter", "tent_score_adapter"]
    summary: dict = {}
    for method in methods:
        vals = []
        for row in table:
            m = row.get(method) or {}
            v = m.get("roc_auc")
            if isinstance(v, (int, float)):
                vals.append(float(v))
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
    return base


if __name__ == "__main__":
    sys.exit(main())
