#!/usr/bin/env python3
"""Dispatch Master Scenario C training stages to existing repo scripts.

Usage:
  PYTHONPATH=src python src/scripts/scenario_c/run_training_stage.py --stage T0
  PYTHONPATH=src python src/scripts/scenario_c/run_training_stage.py --stage T1 --only elara_bench_la
  PYTHONPATH=src python src/scripts/scenario_c/run_training_stage.py --stage T4 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

STAGE_ALIASES = {
    "T0": "T0_governance",
    "T1": "T1_data_splits",
    "T2": "T2_calibration",
    "T3": "T3_static_baselines",
    "T4": "T4_base_rga",
    "T5": "T5_rga_plus",
    "T6": "T6_monitor_abstention",
    "T7": "T7_confirmatory_eval",
}

# Subset of T1 prepare scripts keyed by dataset id
T1_DATASET_SCRIPTS: dict[str, str] = {
    "elara_bench_la": "src/scripts/prepare_real_fusion_benchmark.py",
    "realfusion_la": "src/scripts/prepare_realfusion_la_benchmark.py",
    "mvtec_3d_ad": "src/scripts/prepare_mvtec3d_fusion_benchmark.py",
    "mvtec_loco": "src/scripts/prepare_mvtec_loco_fusion_benchmark.py",
    "visa": "src/scripts/prepare_visa_fusion_benchmark.py",
    "unsw_nb15": "src/scripts/prepare_unsw_paired_fusion_benchmark.py",
    "eyecandies": "src/scripts/prepare_real3d_fusion_benchmark.py",
}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("Could not locate repo root")


def _load_registry(root: Path) -> dict:
    path = root / "elara_master_c/configs/training_stage_registry.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run(cmd: list[str], root: Path, dry_run: bool) -> int:
    print("$", " ".join(cmd))
    if dry_run:
        return 0
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    return subprocess.call(cmd, cwd=root, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, help="T0–T7 or full key e.g. T4_base_rga")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", type=str, default=None, help="T1: dataset key from T1_DATASET_SCRIPTS")
    args = parser.parse_args()

    root = _repo_root()
    reg = _load_registry(root)
    stage_key = STAGE_ALIASES.get(args.stage.upper(), args.stage)
    stages = reg.get("stages", {})
    if stage_key not in stages:
        print(f"Unknown stage {args.stage!r}. Valid: {list(STAGE_ALIASES.keys())}")
        return 2

    stage = stages[stage_key]
    if not stage.get("training_allowed", True) and stage_key != "T0_governance":
        blocked = stage.get("blocked_until")
        if blocked and stage_key == "T7_confirmatory_eval":
            print("T7 blocked until:", blocked)
            return 3

    py = sys.executable
    manifest = {
        "stage": stage_key,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "commands": [],
    }
    rc = 0

    if stage_key == "T0_governance":
        cmd = [py, "src/scripts/scenario_c/validate_master_c_governance.py"]
        manifest["commands"].append(cmd)
        rc = _run(cmd, root, args.dry_run)
    elif stage_key == "T1_data_splits":
        if args.only:
            rel = T1_DATASET_SCRIPTS.get(args.only)
            if not rel:
                print(f"Unknown --only {args.only!r}. Keys: {list(T1_DATASET_SCRIPTS)}")
                return 2
            scripts = [rel]
        else:
            scripts = stage.get("scripts", [])
        for rel in scripts:
            script = root / rel
            if not script.is_file():
                print(f"Skip missing: {rel}")
                continue
            cmd = [py, str(rel)]
            manifest["commands"].append(cmd)
            rc = _run(cmd, root, args.dry_run) or rc
        if not args.dry_run and scripts:
            qual = [py, "src/scripts/scenario_c/qualify_upstream_experts.py", "--export-parquet"]
            manifest["commands"].append(qual)
            rc = _run(qual, root, False) or rc
    elif stage_key in ("T3_static_baselines", "T4_base_rga"):
        cmd = [py, "src/scripts/run_breakthrough_experiment.py"]
        manifest["commands"].append(cmd)
        if not args.dry_run:
            print("Note: run_breakthrough_experiment.py may take long; use project configs as documented.")
        rc = _run(cmd, root, args.dry_run)
    elif stage_key == "T5_rga_plus":
        cmd = [py, "src/scripts/run_phase2_powered_audited_pilot.py"]
        manifest["commands"].append(cmd)
        rc = _run(cmd, root, args.dry_run)
    elif stage_key == "T6_monitor_abstention":
        for rel in ("src/scripts/audit_gate_decision_rule_e2e.py",):
            cmd = [py, rel]
            manifest["commands"].append(cmd)
            rc = _run(cmd, root, args.dry_run) or rc
    elif stage_key == "T7_confirmatory_eval":
        print("T7 requires frozen models + new M2 dataset (D3). Run validate only.")
        cmd = [py, "src/scripts/scenario_c/validate_master_c_governance.py"]
        rc = _run(cmd, root, args.dry_run)
    else:
        for rel in stage.get("scripts", []):
            cmd = [py, rel]
            manifest["commands"].append(cmd)
            rc = _run(cmd, root, args.dry_run) or rc

    manifest["exit_code"] = rc
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    out_dir = root / "elara_master_c/audits/stage_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stage_key}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    if not args.dry_run:
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest: {out_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
