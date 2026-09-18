#!/usr/bin/env python3
"""Run Protocol B KGA evaluation independently on each arm of the Task 4 matched SAR study.

Protocol B specifications:
- alpha = 0.10
- exact-rank rule (k = ceil((n_cal + 1)(1 - alpha)))
- tau = 0.0 (decision threshold)
- 3-way cell-outcome-disjoint cross-fitting (estimator fit, residual calibration, score)

Outputs:
- arm{1..5}_kga.json
- TASK4_MATCHED_SAR_KGA_SUMMARY.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

import kga
from kga.crossfit import controlled_grid_crossfit, controlled_grid_sample_id


def evaluate_arm_kga(
    arm_key: str,
    arm_file: Path,
    output_dir: Path,
    alpha: float = 0.10,
    n_folds: int = 5,
    random_state: int = 0,
) -> dict[str, Any]:
    """Run Protocol B disjoint cross-fitting on one arm."""
    recs = [json.loads(line) for line in arm_file.read_text().splitlines() if line.strip()]
    n = len(recs)
    if n == 0:
        raise ValueError(f"No records found in {arm_file}")

    arm_name = recs[0]["arm_name"]
    arm_id = recs[0]["arm_id"]

    sample_ids = [
        controlled_grid_sample_id(
            track="imagenetc_matched_sar",
            candidate=arm_name,
            seed=r["seed"],
            condition=r["condition_id"],
        )
        for r in recs
    ]

    B = np.array([r["candidate_benefit"] for r in recs], dtype=float)
    a0 = np.array([r["frozen_accuracy"] for r in recs], dtype=float)
    aa = np.array([r["candidate_accuracy"] for r in recs], dtype=float)

    # Feature matrix Z
    if recs[0].get("Z") is not None:
        Z = np.array([r["Z"] for r in recs], dtype=float)
        feature_names = recs[0].get("Z_names", [f"feat_{i}" for i in range(Z.shape[1])])
    else:
        # Fallback to standard features if Z wasn't stored
        Z = np.column_stack([a0, [r["parameter_update_norm"] for r in recs]])
        feature_names = ["frozen_acc", "upd_norm"]

    # Run canonical controlled-grid cross-fit
    crossfit_result = controlled_grid_crossfit(
        Z,
        B,
        sample_ids=sample_ids,
        alpha=alpha,
        n_folds=n_folds,
        random_state=random_state,
    )

    preds = crossfit_result.prediction
    radii = crossfit_result.radius
    actions = crossfit_result.action

    # Compute routed outcomes
    # Action rule: ADAPT -> deploy aa; FREEZE / ABSTAIN -> deploy a0
    a_deployed = np.where(actions == "ADAPT", aa, a0)
    a_oracle = np.maximum(aa, a0)
    oracle_regret = a_oracle - a_deployed

    # Metrics
    # FA_u: unconditional false adaptation (action == ADAPT and B < 0)
    # FF_u: unconditional false freeze (action != ADAPT and B > 0)
    fa_u = float(np.mean((actions == "ADAPT") & (B < 0)))
    ff_u = float(np.mean((actions != "ADAPT") & (B > 0)))

    adapt_count = int(np.sum(actions == "ADAPT"))
    freeze_count = int(np.sum(actions == "FREEZE"))
    abstain_count = int(np.sum(actions == "ABSTAIN"))

    # Always adapt baseline
    aa_regret = float(np.mean(a_oracle - aa))
    # Always freeze baseline
    af_regret = float(np.mean(a_oracle - a0))

    arm_result = {
        "arm_id": arm_id,
        "arm_name": arm_name,
        "n_conditions": n,
        "alpha": alpha,
        "protocol": crossfit_result.protocol,
        "routing_metrics": {
            "mean_frozen_accuracy": float(np.mean(a0)),
            "mean_candidate_accuracy": float(np.mean(aa)),
            "mean_candidate_benefit": float(np.mean(B)),
            "mean_oracle_accuracy": float(np.mean(a_oracle)),
            "mean_kga_accuracy": float(np.mean(a_deployed)),
            "mean_kga_oracle_regret": float(np.mean(oracle_regret)),
            "always_adapt_oracle_regret": aa_regret,
            "always_freeze_oracle_regret": af_regret,
            "unconditional_false_adapt_rate": fa_u,
            "unconditional_false_freeze_rate": ff_u,
            "decision_counts": {
                "ADAPT": adapt_count,
                "FREEZE": freeze_count,
                "ABSTAIN": abstain_count,
            },
            "decision_fractions": {
                "ADAPT": adapt_count / n,
                "FREEZE": freeze_count / n,
                "ABSTAIN": abstain_count / n,
            },
        },
        "records": [
            {
                "condition_id": recs[i]["condition_id"],
                "frozen_accuracy": float(a0[i]),
                "candidate_accuracy": float(aa[i]),
                "candidate_benefit": float(B[i]),
                "kga_prediction": None if math.isnan(preds[i]) else float(preds[i]),
                "kga_radius": None if math.isinf(radii[i]) else float(radii[i]),
                "kga_action": str(actions[i]),
                "oracle_action": "ADAPT" if B[i] > 0 else "FREEZE",
                "deployed_accuracy": float(a_deployed[i]),
                "oracle_regret": float(oracle_regret[i]),
            }
            for i in range(n)
        ],
    }

    arm_out = output_dir / f"{arm_key}_kga.json"
    arm_out.write_text(json.dumps(arm_result, indent=2) + "\n")
    print(f"[{arm_key}] ({arm_name}): KGA Acc={np.mean(a_deployed):.4f}, Regret={np.mean(oracle_regret):.4f}, Actions: ADAPT={adapt_count}, FREEZE={freeze_count}, ABSTAIN={abstain_count}")
    return arm_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/kbound/results/task4_matched_sar_v1"))
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    raw_dir = args.results_dir / "raw"
    arm_files = {
        "arm1": raw_dir / "arm1_bn_only.jsonl",
        "arm2": raw_dir / "arm2_reference_sar.jsonl",
        "arm3": raw_dir / "arm3_rate_only.jsonl",
        "arm4": raw_dir / "arm4_mask_only.jsonl",
        "arm5": raw_dir / "arm5_combined.jsonl",
    }

    results = {}
    for arm_key, fpath in arm_files.items():
        if not fpath.exists():
            print(f"Skipping {arm_key}: {fpath} does not exist")
            continue
        results[arm_key] = evaluate_arm_kga(
            arm_key,
            fpath,
            output_dir=args.results_dir,
            alpha=args.alpha,
            n_folds=args.n_folds,
        )

    summary_file = args.results_dir / "TASK4_MATCHED_SAR_KGA_SUMMARY.json"
    summary_file.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nUnified KGA summary written to: {summary_file}")


if __name__ == "__main__":
    main()
