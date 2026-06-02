#!/usr/bin/env python3
"""Run D15 natural headroom/degradation audit on prepared Real-IAD D3 inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uais.fusion.attention.baselines import SARScoreAdapter  # noqa: E402
from uais.fusion.attention.headroom_routing import (  # noqa: E402
    confidence_weighted_mean,
    headroom_subset_masks,
    select_headroom_thresholds,
    stress_score_from_reliability,
)
from uais.fusion.attention.positive_transfer import paired_auc_bootstrap  # noqa: E402


def _pivot(csv_path: Path, split: str) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    s = df[df["split"] == split].copy()
    if s.empty:
        raise ValueError(f"split not found in {csv_path}: {split}")
    domains = sorted(s["domain"].drop_duplicates())
    ids = sorted(set.intersection(*[set(s[s["domain"] == d]["sample_id"]) for d in domains]))
    per_domain = [s[s["domain"] == d].set_index("sample_id").loc[ids] for d in domains]
    scores = np.column_stack([d["score"].to_numpy(dtype=float) for d in per_domain])
    confidence = np.column_stack([d["confidence"].to_numpy(dtype=float) for d in per_domain])
    reliability = np.column_stack([d["quality_reliability"].to_numpy(dtype=float) for d in per_domain])
    embedding_cols = sorted([c for c in s.columns if c.startswith("embedding_")], key=lambda c: int(c.split("_")[1]))
    feature_blocks = []
    for d in per_domain:
        block = np.column_stack(
            [
                d["score"].to_numpy(dtype=float),
                d["confidence"].to_numpy(dtype=float),
                d["quality_reliability"].to_numpy(dtype=float),
                d["quality_distance"].to_numpy(dtype=float),
                d[embedding_cols].to_numpy(dtype=float),
            ]
        )
        feature_blocks.append(block)
    features = np.stack(feature_blocks, axis=1)
    y = per_domain[0]["label"].to_numpy(dtype=int)
    return {
        "ids": ids,
        "domains": domains,
        "scores": scores,
        "confidence": confidence,
        "reliability": reliability,
        "features": features,
        "masks": np.zeros((len(ids), len(domains)), dtype=bool),
        "y": y,
    }


def _safe_stat(labels: np.ndarray, candidate: np.ndarray, comparator: np.ndarray, mask: np.ndarray, *, seed: int, n_iter: int) -> dict[str, Any]:
    n = int(mask.sum())
    if n < 20 or len(np.unique(labels[mask])) < 2:
        return {"valid": False, "reason": "degenerate_or_small_subset", "n": n, "delta": 0.0, "ci95": [0.0, 0.0]}
    return paired_auc_bootstrap(labels[mask], candidate[mask], comparator[mask], n_iter=n_iter, seed=seed)


def _ci_low(stat: dict[str, Any]) -> float:
    ci = stat.get("ci95") or [0.0, 0.0]
    return float(ci[0])


def _artifact_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _current_dataset_status(status: str, holdout_status: str) -> str:
    """Report the dataset's current evidence role after an audit has been archived."""
    if status in {"OFFICIAL_PASS", "OFFICIAL_FAIL", "INVALID_OFFICIAL_OPENED_CATEGORY"}:
        return "OPENED_AFTER_D16_OFFICIAL_ATTEMPT"
    if holdout_status == "OPENED_DEVELOPMENT_ONLY":
        return "OPENED_DEVELOPMENT_ONLY"
    return "PENDING_OR_UNEVALUATED"


def _d16_category_integrity(categories: list[str]) -> dict[str, Any]:
    opened = ["common_mode_filter"]
    present = sorted(set(categories) & set(opened))
    return {
        "development_opened_categories": opened,
        "opened_categories_present": present,
        "confirmatory_category_valid": not present,
    }


def _fusion_candidates(scores: np.ndarray, confidence: np.ndarray, reliability: np.ndarray) -> dict[str, np.ndarray]:
    rel = np.clip(reliability, 0.05, 1.0)
    inv_rel = 1.0 - np.clip(reliability, 0.0, 1.0) + 0.05
    disagreement = scores.max(axis=1) - scores.min(axis=1)
    return {
        "confidence_weighted_mean": confidence_weighted_mean(scores, confidence),
        "quality_reliability_weighted_mean": (scores * rel).sum(axis=1) / rel.sum(axis=1),
        "inverse_reliability_weighted_mean": (scores * inv_rel).sum(axis=1) / inv_rel.sum(axis=1),
        "max_score": scores.max(axis=1),
        "min_score": scores.min(axis=1),
        "score_disagreement": disagreement,
        "max_plus_disagreement": np.clip(scores.max(axis=1) + 0.1 * disagreement, 0.0, 1.0),
    }


def _auc_or_none(labels: np.ndarray, values: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) < 20 or len(np.unique(labels[mask])) < 2:
        return None
    return float(roc_auc_score(labels[mask], values[mask]))


def _select_stress_method(
    labels: np.ndarray,
    candidates: dict[str, np.ndarray],
    stress_mask: np.ndarray,
) -> dict[str, Any]:
    aucs = {name: _auc_or_none(labels, values, stress_mask) for name, values in candidates.items()}
    valid = {name: auc for name, auc in aucs.items() if auc is not None}
    if not valid:
        return {
            "selected_method": "confidence_weighted_mean",
            "reason": "degenerate_validation_stress_subset",
            "used_split": "validation",
            "used_test_labels": False,
            "validation_stress_auc": aucs,
            "validation_stress_delta_vs_cw": 0.0,
        }
    cw_auc = float(valid.get("confidence_weighted_mean", 0.5))
    selected, selected_auc = max(valid.items(), key=lambda item: (item[1], item[0]))
    delta = float(selected_auc - cw_auc)
    if delta <= 0.0:
        selected = "confidence_weighted_mean"
        selected_auc = cw_auc
        delta = 0.0
    return {
        "selected_method": selected,
        "reason": "validation_stress_auc_argmax",
        "used_split": "validation",
        "used_test_labels": False,
        "validation_stress_auc": aucs,
        "validation_stress_delta_vs_cw": delta,
        "selected_validation_stress_auc": float(selected_auc),
    }


def run_audit(
    *,
    csv_path: Path,
    output: Path,
    holdout_status: str,
    bootstrap_iter: int,
) -> dict[str, Any]:
    source_df = pd.read_csv(csv_path, usecols=["category"])
    categories = sorted(source_df["category"].drop_duplicates().astype(str).tolist())
    category_integrity = _d16_category_integrity(categories)
    val = _pivot(csv_path, "validation")
    test = _pivot(csv_path, "test")
    thresholds = select_headroom_thresholds(val["reliability"], stress_quantile=0.70, clean_quantile=0.30)

    val_stress_scores = stress_score_from_reliability(val["reliability"])
    val_subsets = headroom_subset_masks(
        val_stress_scores,
        stress_threshold=thresholds.stress_threshold,
        clean_threshold=thresholds.clean_threshold,
    )
    val_candidates = _fusion_candidates(val["scores"], val["confidence"], val["reliability"])
    selector = _select_stress_method(val["y"], val_candidates, val_subsets["stress"])

    test_candidates = _fusion_candidates(test["scores"], test["confidence"], test["reliability"])
    cw = test_candidates["confidence_weighted_mean"]
    equal_mean = test["scores"].mean(axis=1)
    max_score = test_candidates["max_score"]
    sar = SARScoreAdapter(random_seed=42, adaptation_steps=30).fit(
        val["features"],
        val["masks"],
        val["y"],
    ).predict_proba(test["features"], test["masks"])

    stress_scores = stress_score_from_reliability(test["reliability"])
    subsets = headroom_subset_masks(
        stress_scores,
        stress_threshold=thresholds.stress_threshold,
        clean_threshold=thresholds.clean_threshold,
    )
    selected_method = str(selector["selected_method"])
    candidate = cw.copy()
    candidate[subsets["stress"]] = test_candidates[selected_method][subsets["stress"]]
    gate_active = subsets["stress"] & (selected_method != "confidence_weighted_mean")
    stats = {
        "stress_vs_cw": _safe_stat(test["y"], candidate, cw, subsets["stress"], seed=0, n_iter=bootstrap_iter),
        "stress_vs_sar": _safe_stat(test["y"], candidate, sar, subsets["stress"], seed=1, n_iter=bootstrap_iter),
        "clean_vs_cw": _safe_stat(test["y"], candidate, cw, subsets["clean"], seed=2, n_iter=bootstrap_iter),
        "clean_vs_sar": _safe_stat(test["y"], candidate, sar, subsets["clean"], seed=3, n_iter=bootstrap_iter),
        "all_vs_cw": _safe_stat(test["y"], candidate, cw, subsets["all"], seed=4, n_iter=bootstrap_iter),
        "all_vs_sar": _safe_stat(test["y"], candidate, sar, subsets["all"], seed=5, n_iter=bootstrap_iter),
        "all_vs_equal_mean": _safe_stat(test["y"], candidate, equal_mean, subsets["all"], seed=6, n_iter=bootstrap_iter),
        "all_vs_max": _safe_stat(test["y"], candidate, max_score, subsets["all"], seed=7, n_iter=bootstrap_iter),
    }
    clean_default_rate = float((~gate_active[subsets["clean"]]).mean()) if subsets["clean"].any() else 0.0
    primary_pass = bool(
        holdout_status == "FRESH_OR_UNOPENED"
        and category_integrity["confirmatory_category_valid"]
        and stats["stress_vs_cw"].get("valid", False)
        and float(stats["stress_vs_cw"].get("delta", 0.0)) >= 0.005
        and _ci_low(stats["stress_vs_cw"]) > 0.0
        and stats["clean_vs_cw"].get("valid", False)
        and _ci_low(stats["clean_vs_cw"]) >= -0.005
        and clean_default_rate >= 0.80
    )
    official_attempt = bool(
        holdout_status == "FRESH_OR_UNOPENED"
        and category_integrity["confirmatory_category_valid"]
    )
    if primary_pass:
        status = "OFFICIAL_PASS"
    elif official_attempt:
        status = "OFFICIAL_FAIL"
    elif holdout_status == "FRESH_OR_UNOPENED":
        status = "INVALID_OFFICIAL_OPENED_CATEGORY"
    else:
        status = "DEVELOPMENT_ONLY"
    result = {
        "protocol": "D16_NATURAL_DEGRADATION_SELECTOR_PROTOCOL_v1",
        "supersedes_protocol": "D15_NATURAL_DEGRADATION_PROTOCOL_v1",
        "dataset_id": "realiad_d3_natural_degradation_v1",
        "status": status,
        "holdout_status": holdout_status,
        "initial_holdout_status": holdout_status,
        "current_dataset_status": _current_dataset_status(status, holdout_status),
        "evidence_role": "real_natural_degradation_evidence",
        "freshness_note": (
            "The initial holdout status records whether the archived D16 attempt was eligible. "
            "After that attempt, this Real-IAD D3 dataset is opened evidence, not a reusable fresh holdout."
        ),
        "categories": categories,
        "category_integrity": category_integrity,
        "domains": test["domains"],
        "thresholds": thresholds.as_dict(),
        "selection": {
            "threshold_source": "validation_quality_reliability_only",
            "method_selector": selector,
            "used_test_labels": False,
            "stress_quantile": 0.70,
            "clean_quantile": 0.30,
        },
        "subsets": {
            "n_test": int(len(test["y"])),
            "n_stress": int(subsets["stress"].sum()),
            "n_clean": int(subsets["clean"].sum()),
            "clean_default_rate": clean_default_rate,
            "stress_gate_active_rate": float(gate_active[subsets["stress"]].mean()) if subsets["stress"].any() else 0.0,
        },
        "stats": stats,
        "gate_s_natural_degradation_confirmed": primary_pass,
        "secondary_sar_note": "SAR is reported as secondary; D15 does not rewrite strict clean Gate E.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pred_path = output.with_name(output.stem + "_predictions.csv")
    pd.DataFrame(
        {
            "sample_id": test["ids"],
            "label": test["y"],
            "candidate": candidate,
            "cw": cw,
            "sar": sar,
            "equal_mean": equal_mean,
            "max_score": max_score,
            "stress_score": stress_scores,
            "gate_active": gate_active,
            "subset_stress": subsets["stress"],
            "subset_clean": subsets["clean"],
        }
    ).to_csv(pred_path, index=False)
    result["prediction_archive"] = _artifact_reference(pred_path)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "experiments/fusion/realiad_d3_headroom_inputs.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/fusion/realiad_d3_headroom_audit_result.json")
    parser.add_argument("--holdout-status", choices=["FRESH_OR_UNOPENED", "OPENED_DEVELOPMENT_ONLY"], required=True)
    parser.add_argument("--bootstrap-iter", type=int, default=10000)
    args = parser.parse_args()
    result = run_audit(
        csv_path=args.csv,
        output=args.output,
        holdout_status=args.holdout_status,
        bootstrap_iter=args.bootstrap_iter,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
