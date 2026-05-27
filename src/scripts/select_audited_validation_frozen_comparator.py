"""Phase 1.C — emit the validation-frozen primary comparator per cell.

For every (benchmark, protocol) cell the audited reanalysis primary
comparator is the baseline with the highest seed-averaged
validation-fold ROC-AUC. Validation ROC-AUCs are extracted from the
router's `candidate_validation_roc_auc` map at the `base:<method>`
keys; this is the same val fold the router uses to choose its own
candidate, so no leakage occurs.

Forbidden:
  - Selecting the comparator by reading the test fold.
  - Calling any existing-cell comparator "pre-declared".

Emit output:
  experiments/audit/audited_comparator_selection.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

# Same benchmark list as the master comparison emit script.
BENCHMARKS = [
    ("MVTec 3D-AD", "PatchCore canonical", "A", "experiments/fusion/mvtec3d_patchcore_results.json"),
    ("MVTec 3D-AD", "PatchCore supervised", "A", "experiments/fusion/mvtec3d_patchcore_supervised_paired_results.json"),
    ("MVTec 3D-AD", "PatchCore held-out", "A", "experiments/fusion/mvtec3d_patchcore_heldout_results.json"),
    ("MVTec LOCO-AD", "PatchCore canonical", "A", "experiments/fusion/mvtec_loco_patchcore_results.json"),
    (
        "MVTec LOCO-AD",
        "PatchCore supervised",
        "A",
        "experiments/fusion/mvtec_loco_patchcore_supervised_paired_results.json",
    ),
    ("Real3D-AD", "PCA shape + depth supervised", "C", "experiments/fusion/real3d_supervised_paired_results.json"),
    ("VisA", "RGB+edge canonical", "A", "experiments/fusion/visa_fusion_results.json"),
    ("VisA", "RGB+edge supervised", "A", "experiments/fusion/visa_supervised_paired_results.json"),
    ("VisA", "RGB+random noise-floor", "C", "experiments/fusion/visa_supervised_paired_noise_floor_results.json"),
    ("UNSW-NB15", "flow/conn/context", "A", "experiments/fusion/unsw_paired_results.json"),
    ("UNSW-NB15", "held-out attack categories", "C", "experiments/fusion/unsw_heldout_attack_results.json"),
]

# Candidate non-RGA comparator pool (matches the runner's baseline_predictions roster).
BASELINE_CANDIDATES = (
    "random_forest",
    "early_fusion_mlp",
    "late_fusion_ensemble",
    "confidence_weighted_mean",
    "tent_score_adapter",
    "ttt_pseudo_label_adapter",
    "eata_score_adapter",
    "sar_score_adapter",
)


def per_seed_validation_scores(payload: dict) -> list[dict]:
    """Per-seed val ROC-AUC for every baseline candidate."""
    rows = []
    for seed_row in payload.get("table_1_clean_performance", []) or []:
        seed = int(seed_row.get("seed", -1))
        router = seed_row.get("rga_meta_router") or {}
        val_map = router.get("candidate_validation_roc_auc") or {}
        baseline_val: dict[str, float] = {}
        for name in BASELINE_CANDIDATES:
            v = val_map.get(f"base:{name}")
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                baseline_val[name] = float(v)
        # Per-seed test ROC-AUC per baseline.
        baseline_test: dict[str, float] = {}
        for name in BASELINE_CANDIDATES:
            b = seed_row.get(name) or {}
            if isinstance(b, dict):
                roc = b.get("roc_auc")
                if isinstance(roc, (int, float)) and math.isfinite(float(roc)):
                    baseline_test[name] = float(roc)
        rows.append({"seed": seed, "val": baseline_val, "test": baseline_test})
    return rows


def aggregate_per_cell(per_seed: list[dict]) -> dict:
    """Seed-averaged validation ranking → frozen primary comparator."""
    if not per_seed:
        return {"selected_comparator": None}
    val_means: dict[str, list[float]] = {}
    test_means: dict[str, list[float]] = {}
    for r in per_seed:
        for name, v in r["val"].items():
            val_means.setdefault(name, []).append(v)
        for name, v in r["test"].items():
            test_means.setdefault(name, []).append(v)

    val_mean = {n: sum(vs) / len(vs) for n, vs in val_means.items() if vs}
    test_mean = {n: sum(vs) / len(vs) for n, vs in test_means.items() if vs}
    if not val_mean:
        return {"selected_comparator": None}

    # Validation-frozen selection
    chosen = max(val_mean.items(), key=lambda kv: (kv[1], kv[0]))[0]  # deterministic tie-break by name
    chosen_val = val_mean[chosen]
    chosen_test = test_mean.get(chosen)

    # Post-hoc test winner (for audit comparison only)
    posthoc = max(test_mean.items(), key=lambda kv: (kv[1], kv[0]))[0] if test_mean else None
    posthoc_auc = test_mean.get(posthoc) if posthoc else None

    return {
        "candidate_comparators": list(val_mean.keys()),
        "validation_scores": val_mean,
        "test_scores": test_mean,
        "selected_comparator": chosen,
        "selected_comparator_validation_auc": chosen_val,
        "selected_comparator_test_auc": chosen_test,
        "posthoc_test_best_baseline": posthoc,
        "posthoc_test_best_auc": posthoc_auc,
        "does_selected_match_test_best": chosen == posthoc,
        "selection_used_test_metrics": False,
        "analysis_status": "locked_audited_reanalysis",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/audit/audited_comparator_selection.csv"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    rows: list[dict] = []
    for benchmark, protocol, family, rel_path in BENCHMARKS:
        p = args.repo_root / rel_path
        if not p.exists():
            rows.append(
                {
                    "benchmark": benchmark,
                    "protocol": protocol,
                    "analysis_family": family,
                    "candidate_comparators": "",
                    "validation_scores": "",
                    "selected_comparator": "",
                    "selected_comparator_validation_auc": "",
                    "selected_comparator_test_auc": "",
                    "posthoc_test_best_baseline": "",
                    "posthoc_test_best_auc": "",
                    "does_selected_match_test_best": "",
                    "selection_used_test_metrics": False,
                    "analysis_status": "pending — JSON missing",
                }
            )
            continue
        payload = json.loads(p.read_text())
        per_seed = per_seed_validation_scores(payload)
        agg = aggregate_per_cell(per_seed)
        rows.append(
            {
                "benchmark": benchmark,
                "protocol": protocol,
                "analysis_family": family,
                "candidate_comparators": "|".join(agg.get("candidate_comparators", []) or []),
                "validation_scores": json.dumps(agg.get("validation_scores") or {}, sort_keys=True),
                "selected_comparator": agg.get("selected_comparator"),
                "selected_comparator_validation_auc": agg.get("selected_comparator_validation_auc"),
                "selected_comparator_test_auc": agg.get("selected_comparator_test_auc"),
                "posthoc_test_best_baseline": agg.get("posthoc_test_best_baseline"),
                "posthoc_test_best_auc": agg.get("posthoc_test_best_auc"),
                "does_selected_match_test_best": agg.get("does_selected_match_test_best"),
                "selection_used_test_metrics": False,
                "analysis_status": agg.get("analysis_status", "locked_audited_reanalysis"),
            }
        )

    fields = [
        "benchmark",
        "protocol",
        "analysis_family",
        "candidate_comparators",
        "validation_scores",
        "selected_comparator",
        "selected_comparator_validation_auc",
        "selected_comparator_test_auc",
        "posthoc_test_best_baseline",
        "posthoc_test_best_auc",
        "does_selected_match_test_best",
        "selection_used_test_metrics",
        "analysis_status",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {args.output} with {len(rows)} rows")
    print()
    print("Validation-frozen primary comparator per cell:")
    for r in rows:
        sel = r["selected_comparator"]
        sel_auc = r["selected_comparator_test_auc"]
        ph = r["posthoc_test_best_baseline"]
        ph_auc = r["posthoc_test_best_auc"]
        sel_auc_s = f"{sel_auc:.4f}" if isinstance(sel_auc, (int, float)) else "--"
        ph_auc_s = f"{ph_auc:.4f}" if isinstance(ph_auc, (int, float)) else "--"
        match = r["does_selected_match_test_best"]
        print(
            f"  {r['benchmark']:<14s} {r['protocol']:<30s} val-frozen={sel} ({sel_auc_s}) | posthoc-best={ph} ({ph_auc_s}) | match={match}"
        )


if __name__ == "__main__":
    main()
