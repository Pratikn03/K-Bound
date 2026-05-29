#!/usr/bin/env python3
"""T0 governance validator for Master Scenario C.

Checks that research_lock contracts, training fixes, and stage registry exist.
Writes elara_master_c/audits/t0_governance_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "research_lock").is_dir() and (parent / "src").is_dir():
            return parent
    raise RuntimeError("Could not locate AutoML_Flagship_V8 repo root")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Master Scenario C T0 governance")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    root = _repo_root()
    checks: list[Check] = []

    required_lock = [
        "research_lock/SCENARIO_C_CLAIM_CONTRACT.md",
        "research_lock/dataset_registry_v1.yaml",
        "research_lock/dataset_registry_v2.yaml",
        "research_lock/frozen_test_sets_v2.yaml",
        "research_lock/protocol_registry_v1.yaml",
        "research_lock/model_registry_v1.yaml",
        "research_lock/primary_endpoints_v1.yaml",
        "research_lock/claim_matrix_v1.csv",
        "research_lock/statistical_policy_v1.md",
        "research_lock/DECISIONS_v1.md",
        "research_lock/family_d_failure_record.md",
    ]
    for rel in required_lock:
        p = root / rel
        checks.append(
            Check(
                name=f"exists:{rel}",
                passed=p.is_file(),
                detail=str(p) if p.is_file() else "missing",
            )
        )

    master_c = [
        "elara_master_c/configs/training_stage_registry.yaml",
        "elara_master_c/configs/hyperparameter_search_space_v1.yaml",
        "elara_master_c/configs/expert_registry.yaml",
        "elara_master_c/configs/split_registry.yaml",
        "elara_master_c/configs/baseline_registry.yaml",
        "elara_master_c/configs/endpoint_registry.yaml",
        "elara_master_c/configs/run_manifest.template.json",
        "audits/training_truth_audit/12_training_critical_fixes.md",
        "src/uais/fusion/attention/training_loop.py",
        "src/elara/evaluation/prediction_archive.py",
    ]
    for rel in master_c:
        p = root / rel
        checks.append(
            Check(
                name=f"exists:{rel}",
                passed=p.exists(),
                detail="ok" if p.exists() else "missing",
            )
        )

    # Training defaults: restore_best_weights + pr_auc in primary configs
    for rel in (
        "configs/attention_real_fusion.yaml",
        "configs/attention_mvtec3d_patchcore_supervised_paired.yaml",
    ):
        p = root / rel
        if not p.is_file():
            checks.append(Check(f"config:{rel}", False, "missing"))
            continue
        text = p.read_text(encoding="utf-8")
        has_restore = "restore_best_weights: true" in text or "restore_best_weights: true\n" in text
        has_pr = "early_stopping_metric: pr_auc" in text
        checks.append(
            Check(
                f"config_training_fixes:{rel}",
                has_restore and has_pr,
                f"restore_best_weights={has_restore}, early_stopping_metric_pr_auc={has_pr}",
            )
        )

    # D1 Policy B: v2 must list eyecandies as development
    v2 = root / "research_lock/dataset_registry_v2.yaml"
    if v2.is_file():
        t = v2.read_text(encoding="utf-8")
        policy_b = "eyecandies:" in t and "role: development" in t
        checks.append(
            Check(
                "decision_D1_eyecandies_development",
                policy_b,
                "dataset_registry_v2 reflects Policy B" if policy_b else "eyecandies not development in v2",
            )
        )

    # Open decisions block confirmatory
    decisions = (root / "research_lock/DECISIONS_v1.md").read_text(encoding="utf-8")
    d3_open = "D3" in decisions and "OPEN" in decisions
    checks.append(
        Check(
            "decision_D3_final_m2_pending",
            d3_open,
            "Final untouched M2 still required before confirmatory T7" if d3_open else "unexpected D3 state",
        )
    )

    passed = all(c.passed for c in checks)
    report = {
        "repo_root": str(root),
        "passed": passed,
        "checks": [asdict(c) for c in checks],
        "pass_condition_questions": {
            "development_only": "elara_bench_la, mvtec_3d_ad, eyecandies (post-D1), visa, loco, unsw",
            "final_confirmation": "m2_new_untouched_transfer (NOT_ACQUIRED)",
            "validation_tuning_allowed": "all development sets; not final_unseen_audit",
            "primary_metric": "see primary_endpoints_v1.yaml per family",
            "baseline_to_beat": "strongest frozen comparator (Phase 5; not yet frozen)",
            "sealed_until_final": "m2_new_untouched_transfer, m3, m4",
        },
    }

    out = args.json_out or root / "elara_master_c/audits/t0_governance_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"T0 governance: {'PASS' if passed else 'FAIL'} ({sum(c.passed for c in checks)}/{len(checks)} checks)")
    print(f"Report: {out}")
    if not passed:
        for c in checks:
            if not c.passed:
                print(f"  FAIL {c.name}: {c.detail}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
