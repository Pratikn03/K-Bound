"""Phase 1.B — emit validation-frozen RGA+ headline selection.

Reads every (benchmark, protocol) results JSON, computes the
validation-frozen RGA+ head per cell from already-logged
`candidate_validation_roc_auc` fields, and emits:

  experiments/audit/rga_plus_validation_frozen_selection.csv

with one row per cell. The CSV is the single source of truth for the
RGA+ headline in the locked audited reanalysis; the master comparison
emit script (Phase 1.C / D) reads from this artifact instead of
computing `max(rga_router_test, rga_boost_test)`.

Tie-break rule (deterministic, documented):
  If |val_router - val_boost| < 1e-12, select 'boost' because it is
  the direct supervised reliability-feature extension and reflects the
  intended deployment configuration. Otherwise select the head with
  the higher validation ROC-AUC.

The emit output is a CSV consumed by the master comparison table; no
code-path elsewhere is permitted to call `max(router_test, boost_test)`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

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

TIE_BREAK_RULE = (
    "If |val_router - val_boost| < 1e-12, select 'boost'. Otherwise select the head "
    "with higher validation ROC-AUC. This rule is documented and deterministic."
)


def _selected_validation_auc(method_block: dict) -> float | None:
    """Per-seed validation ROC-AUC of the head's *selected* candidate.

    The runner's ReliabilityBoostedFusion and meta-router both log a
    `candidate_validation_roc_auc` map and a `selected_candidate`
    string; the selected candidate's value is the validation ROC-AUC
    of the head as deployed for that seed.
    """
    candidates = method_block.get("candidate_validation_roc_auc") or {}
    selected_name = method_block.get("selected_candidate")
    if isinstance(selected_name, str) and selected_name in candidates:
        v = candidates[selected_name]
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
    # Fall back to max(candidate val ROC-AUC) — also a valid post-hoc readout but only if no `selected`.
    finite = [float(v) for v in candidates.values() if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return max(finite) if finite else None


def per_seed_validation_frozen_selection(payload: dict) -> list[dict]:
    """One row per seed with the val-frozen choice."""
    out = []
    for seed_row in payload.get("table_1_clean_performance", []) or []:
        seed = int(seed_row.get("seed", -1))
        router = seed_row.get("rga_meta_router") or {}
        boost = seed_row.get("rga_boosted_fusion") or {}
        val_router = _selected_validation_auc(router)
        val_boost = _selected_validation_auc(boost)
        # Tie-break
        if val_router is None and val_boost is None:
            chosen = None
        elif val_router is None:
            chosen = "boost"
        elif val_boost is None:
            chosen = "router"
        elif abs(val_router - val_boost) < 1e-12:
            chosen = "boost"
        elif val_router > val_boost:
            chosen = "router"
        else:
            chosen = "boost"
        chosen_test = None
        chosen_val = None
        if chosen == "router":
            chosen_test = router.get("roc_auc") if isinstance(router.get("roc_auc"), (int, float)) else None
            chosen_val = val_router
        elif chosen == "boost":
            chosen_test = boost.get("roc_auc") if isinstance(boost.get("roc_auc"), (int, float)) else None
            chosen_val = val_boost
        out.append(
            {
                "seed": seed,
                "validation_roc_auc_router": val_router,
                "validation_roc_auc_boost": val_boost,
                "chosen_head": chosen,
                "chosen_validation_roc_auc": chosen_val,
                "chosen_test_roc_auc": chosen_test,
                "router_test_roc_auc": router.get("roc_auc"),
                "boost_test_roc_auc": boost.get("roc_auc"),
            }
        )
    return out


def ensemble_selection(per_seed: list[dict]) -> dict:
    """Cell-level val-frozen selection.

    Policy: ONE head per cell, chosen by the seed-mean validation ROC-AUC.
    Per-seed mixing is forbidden because it makes the ensemble auc a
    mixture of router+boost predictions.

    Tie-break: if |val_router_mean - val_boost_mean| < 1e-12, select 'boost'.
    """
    if not per_seed:
        return {"chosen_head": None}
    counts = {"router": 0, "boost": 0}
    for r in per_seed:
        c = r.get("chosen_head")
        if c in counts:
            counts[c] += 1

    def _mean(key):
        vals = [
            r.get(key) for r in per_seed if isinstance(r.get(key), (int, float)) and math.isfinite(float(r.get(key)))
        ]
        return sum(vals) / len(vals) if vals else None

    val_router_mean = _mean("validation_roc_auc_router")
    val_boost_mean = _mean("validation_roc_auc_boost")
    router_test_mean = _mean("router_test_roc_auc")
    boost_test_mean = _mean("boost_test_roc_auc")

    # Cell-level val-frozen choice (tie-break to boost).
    if val_router_mean is None and val_boost_mean is None:
        chosen = None
    elif val_router_mean is None:
        chosen = "boost"
    elif val_boost_mean is None:
        chosen = "router"
    elif abs(val_router_mean - val_boost_mean) < 1e-12:
        chosen = "boost"
    elif val_router_mean > val_boost_mean:
        chosen = "router"
    else:
        chosen = "boost"

    if chosen == "router":
        chosen_val_mean = val_router_mean
        chosen_test_mean = router_test_mean
    elif chosen == "boost":
        chosen_val_mean = val_boost_mean
        chosen_test_mean = boost_test_mean
    else:
        chosen_val_mean = None
        chosen_test_mean = None

    return {
        "chosen_head": chosen,
        "n_seeds": len(per_seed),
        "n_seed_choose_router": counts["router"],
        "n_seed_choose_boost": counts["boost"],
        "router_validation_auc_mean": val_router_mean,
        "boost_validation_auc_mean": val_boost_mean,
        "router_test_auc_mean": router_test_mean,
        "boost_test_auc_mean": boost_test_mean,
        "chosen_validation_auc_mean": chosen_val_mean,
        "chosen_test_auc_mean": chosen_test_mean,
        "tie_break_rule": TIE_BREAK_RULE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/audit/rga_plus_validation_frozen_selection.csv"),
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
                    "seed_or_ensemble": "ensemble",
                    "router_validation_auc": None,
                    "boost_validation_auc": None,
                    "selected_head": None,
                    "selected_validation_auc": None,
                    "selected_test_auc": None,
                    "old_test_max_head": None,
                    "old_test_max_auc": None,
                    "delta_old_max_minus_corrected_headline": None,
                    "selection_used_test_metrics": False,
                    "claim_status": "pending — JSON missing",
                    "n_seeds": 0,
                    "n_seed_choose_router": 0,
                    "n_seed_choose_boost": 0,
                }
            )
            continue
        payload = json.loads(p.read_text())
        per_seed = per_seed_validation_frozen_selection(payload)
        ens = ensemble_selection(per_seed)
        # Old test-max behaviour for delta
        cs = payload.get("clean_metric_summary", {})

        def _m(name, _cs=cs):
            v = _cs.get(name, {}).get("roc_auc")
            if isinstance(v, dict):
                return v.get("mean")
            return v if isinstance(v, (int, float)) else None

        router_test = _m("rga_meta_router")
        boost_test = _m("rga_boosted_fusion")
        if router_test is not None and (boost_test is None or router_test >= boost_test):
            old_max_head = "router"
            old_max_auc = router_test
        else:
            old_max_head = "boost"
            old_max_auc = boost_test
        corrected_head = ens.get("chosen_head")
        corrected_auc = ens.get("chosen_test_auc_mean")
        delta = (
            old_max_auc - corrected_auc
            if (isinstance(old_max_auc, (int, float)) and isinstance(corrected_auc, (int, float)))
            else None
        )
        # Ensemble row
        rows.append(
            {
                "benchmark": benchmark,
                "protocol": protocol,
                "analysis_family": family,
                "seed_or_ensemble": "ensemble",
                "router_validation_auc": ens.get("router_validation_auc_mean"),
                "boost_validation_auc": ens.get("boost_validation_auc_mean"),
                "selected_head": corrected_head,
                "selected_validation_auc": ens.get("chosen_validation_auc_mean"),
                "selected_test_auc": corrected_auc,
                "old_test_max_head": old_max_head,
                "old_test_max_auc": old_max_auc,
                "delta_old_max_minus_corrected_headline": delta,
                "selection_used_test_metrics": False,
                "claim_status": "locked_audited_reanalysis",
                "n_seeds": ens.get("n_seeds"),
                "n_seed_choose_router": ens.get("n_seed_choose_router"),
                "n_seed_choose_boost": ens.get("n_seed_choose_boost"),
            }
        )
        # Per-seed rows
        for r in per_seed:
            rows.append(
                {
                    "benchmark": benchmark,
                    "protocol": protocol,
                    "analysis_family": family,
                    "seed_or_ensemble": f"seed:{r['seed']}",
                    "router_validation_auc": r["validation_roc_auc_router"],
                    "boost_validation_auc": r["validation_roc_auc_boost"],
                    "selected_head": r["chosen_head"],
                    "selected_validation_auc": r["chosen_validation_roc_auc"],
                    "selected_test_auc": r["chosen_test_roc_auc"],
                    "old_test_max_head": None,
                    "old_test_max_auc": None,
                    "delta_old_max_minus_corrected_headline": None,
                    "selection_used_test_metrics": False,
                    "claim_status": "locked_audited_reanalysis",
                    "n_seeds": 1,
                    "n_seed_choose_router": int(r["chosen_head"] == "router"),
                    "n_seed_choose_boost": int(r["chosen_head"] == "boost"),
                }
            )

    fields = [
        "benchmark",
        "protocol",
        "analysis_family",
        "seed_or_ensemble",
        "router_validation_auc",
        "boost_validation_auc",
        "selected_head",
        "selected_validation_auc",
        "selected_test_auc",
        "old_test_max_head",
        "old_test_max_auc",
        "delta_old_max_minus_corrected_headline",
        "selection_used_test_metrics",
        "claim_status",
        "n_seeds",
        "n_seed_choose_router",
        "n_seed_choose_boost",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {args.output} with {len(rows)} rows")

    # Print ensemble summary for stdout
    print()
    print("Ensemble val-frozen selection per cell:")
    for r in rows:
        if r["seed_or_ensemble"] != "ensemble":
            continue
        sel = r["selected_head"]
        sel_auc = r["selected_test_auc"]
        old = r["old_test_max_head"]
        old_auc = r["old_test_max_auc"]
        delta = r["delta_old_max_minus_corrected_headline"]
        sel_auc_s = f"{sel_auc:.4f}" if isinstance(sel_auc, (int, float)) else "--"
        old_auc_s = f"{old_auc:.4f}" if isinstance(old_auc, (int, float)) else "--"
        delta_s = f"{delta:+.4f}" if isinstance(delta, (int, float)) else "--"
        print(
            f"  {r['benchmark']:<14s} {r['protocol']:<30s} val-frozen={sel} ({sel_auc_s}) | old_max={old} ({old_auc_s}) | Δ={delta_s}"
        )


if __name__ == "__main__":
    main()
