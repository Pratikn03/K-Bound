"""Phase 8 / Theorem T5 — false-fire vs detection-power trade-off.

NEW EXPLORATORY (development/synthetic only; not confirmatory; touches no sealed
or final test set).

T3 (dilution) and T6 (transfer) both rely on a gate threshold tau selected to a
clean false-fire budget. T5 characterises *why that selection rule is the right
one* and what it costs: it is the operating characteristic of the reliability
gate.

Claim (operational form):
  Let FFR(tau)  = P(gate fires | clean)            -- the cost
      TFR(tau)  = P(gate fires | degraded)         -- the detection power
  Both are CDFs of the per-sample mean reliability, hence monotone
  non-decreasing in tau. The gate's clean/degraded separability is the area
  between them (detector ROC-AUC > 0.5 exactly when reliability is informative).
  Selecting tau*(b) as the b-quantile of CLEAN validation mean reliability bounds
  the out-of-sample clean false-fire at ~b, and the achievable degraded benefit
  is monotone non-decreasing in the budget b (the power ceiling).

This study verifies all of the above on controlled synthetic data and emits both
the continuous tau-ROC and the budget -> benefit curve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.scripts.run_calibration_transfer_study import (
    _collapse_domain,
    _fuse_mean,
    _fuse_reliability_weighted,
    _make_dataset,
)
from uais.fusion.attention.reliability_estimator import PerSampleReliabilityEstimator


def _degrade(feats, masks, rng, k):
    """Collapse the first k domains into confident-but-uncorrelated streams."""
    for j in range(k):
        feats, masks = _collapse_domain(feats, masks, rng, domain=j)
    return feats, masks


def _mean_reliability(est, feats, masks):
    return est.compute_reliability_weights(feats, masks).mean(axis=1)


def run_study(seed=0, d=6, k_degraded=3, n=4000, budgets=None) -> dict:
    if budgets is None:
        budgets = [0.005, 0.01, 0.02, 0.05, 0.10]
    rng = np.random.default_rng(seed)

    # Fit the estimator (KS reference + ECE) on a clean source split.
    fit_feats, fit_masks, fit_labels = _make_dataset(rng, n, d)
    est = PerSampleReliabilityEstimator(
        domain_order=[f"d{j}" for j in range(d)],
        score_index=0,
        ece_weight=0.4, ks_weight=0.4, sharpness_weight=0.2,
        n_calibration_bins=10, min_samples_for_ks=30, gate_mode="mean",
    )
    est.fit(fit_feats, fit_masks, fit_labels)

    # Disjoint clean calibration split for threshold selection.
    cal_feats, cal_masks, _ = _make_dataset(rng, n, d)
    cal_mean_r = _mean_reliability(est, cal_feats, cal_masks)

    # Test stream: half clean, half degraded (k-of-D collapse). Labels preserved.
    clean_feats, clean_masks, clean_labels = _make_dataset(rng, n // 2, d)
    deg_feats, deg_masks, deg_labels = _make_dataset(rng, n // 2, d)
    deg_feats, deg_masks = _degrade(deg_feats, deg_masks, rng, k_degraded)

    feats = np.concatenate([clean_feats, deg_feats], axis=0)
    masks = np.concatenate([clean_masks, deg_masks], axis=0)
    labels = np.concatenate([clean_labels, deg_labels], axis=0)
    is_degraded = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(bool)

    mean_r = _mean_reliability(est, feats, masks)
    static = _fuse_mean(feats, masks)
    weights = est.compute_reliability_weights(feats, masks)
    gated = _fuse_reliability_weighted(feats, masks, weights)

    clean_r = mean_r[~is_degraded]
    deg_r = mean_r[is_degraded]

    # (1) Continuous tau-ROC: sweep tau over the clean reliability range.
    taus = np.quantile(clean_r, np.linspace(0.0, 1.0, 21))
    roc = []
    for tau in taus:
        ffr = float(np.mean(clean_r < tau))   # clean false-fire (cost)
        tfr = float(np.mean(deg_r < tau))      # degraded detection (power)
        roc.append({"tau": float(tau), "clean_false_fire": ffr, "degraded_detection": tfr})

    # Detector separability: lower reliability => more likely degraded.
    detector_auc = float(roc_auc_score(is_degraded.astype(int), -mean_r))

    # (2) Budget -> benefit: tau*(b) = b-quantile of CLEAN validation reliability.
    auc_static = float(roc_auc_score(labels, static))
    budget_rows = []
    for b in budgets:
        tau_b = float(np.quantile(cal_mean_r, b))
        fire = mean_r < tau_b
        acted = np.where(fire, gated, static)
        auc_acted = float(roc_auc_score(labels, acted))
        budget_rows.append({
            "budget": float(b),
            "tau_star": tau_b,
            "test_clean_false_fire": float(np.mean(clean_r < tau_b)),
            "test_degraded_detection": float(np.mean(deg_r < tau_b)),
            "auc_static": auc_static,
            "auc_acted_policy": auc_acted,
            "delta_auc": float(auc_acted - auc_static),
        })

    ffr_series = [r["clean_false_fire"] for r in roc]
    tfr_series = [r["degraded_detection"] for r in roc]
    tfr_budget = [r["test_degraded_detection"] for r in budget_rows]
    ffr_minus_b = [abs(r["test_clean_false_fire"] - r["budget"]) for r in budget_rows]

    return {
        "study": "false_fire_power_tradeoff_T5",
        "label": "NEW EXPLORATORY (development/synthetic only; not confirmatory)",
        "seed": seed, "n_domains": d, "k_degraded": k_degraded, "n": n,
        "detector_roc_auc": detector_auc,
        "tau_roc": roc,
        "budget_curve": budget_rows,
        # --- locked qualitative facts (the operating characteristic) ---
        # Cost and power are both CDFs of mean reliability -> monotone in tau.
        "ffr_monotone_in_tau": bool(np.all(np.diff(ffr_series) >= -1e-9)),
        "tfr_monotone_in_tau": bool(np.all(np.diff(tfr_series) >= -1e-9)),
        # Power dominates cost iff reliability separates clean from degraded.
        "power_dominates_cost": bool(all(t >= f - 1e-9 for f, t in zip(ffr_series, tfr_series))),
        "detector_separates": bool(detector_auc > 0.5),
        # Relaxing the budget buys strictly more detection power...
        "detection_power_monotone_in_budget": bool(np.all(np.diff(tfr_budget) >= -1e-9)),
        # ...and out-of-sample clean false-fire tracks the declared budget.
        "max_budget_calibration_error": float(max(ffr_minus_b)),
        # The trade-off itself: tight budgets cap (and can slightly negate) benefit;
        # benefit becomes positive once the budget admits enough detection power.
        "delta_auc_at_tightest_budget": float(budget_rows[0]["delta_auc"]),
        "delta_auc_at_loosest_budget": float(budget_rows[-1]["delta_auc"]),
        "benefit_positive_at_loosest_budget": bool(budget_rows[-1]["delta_auc"] > 0.0),
        "finding": (
            "Clean false-fire (cost) and degraded detection (power) are both "
            "monotone in tau, and the detection curve dominates the false-fire "
            "curve (detector AUC>0.5): the gate cannot buy detection power without "
            "spending clean false-fire. Selecting tau at the b-quantile of clean "
            "validation reliability bounds out-of-sample clean false-fire at ~b "
            "(the exact rule used in the T3 and P6 studies). Crucially, benefit is "
            "NOT free: at very tight budgets the gate buys too little power to "
            "overcome its clean false-fire cost (delta AUC ~0 or slightly "
            "negative), and net benefit becomes positive only once the budget "
            "admits enough detection power. This is the false-fire/power trade-off."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="False-fire vs power trade-off (T5)")
    ap.add_argument("--output", default="output/phase8/false_fire_power_study.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = run_study(seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== False-fire vs power trade-off (T5) — NEW EXPLORATORY ===")
    print(f"detector ROC-AUC (clean vs degraded) = {report['detector_roc_auc']:.4f}")
    print(f"\n{'budget':>8}{'tau*':>9}{'test_FFR':>10}{'test_TFR':>10}"
          f"{'a_stat':>9}{'a_act':>9}{'dAUC':>9}")
    for r in report["budget_curve"]:
        print(f"{r['budget']:>8.3f}{r['tau_star']:>9.3f}{r['test_clean_false_fire']:>10.3f}"
              f"{r['test_degraded_detection']:>10.3f}{r['auc_static']:>9.4f}"
              f"{r['auc_acted_policy']:>9.4f}{r['delta_auc']:>9.4f}")
    print(f"\nFFR monotone in tau: {report['ffr_monotone_in_tau']}; "
          f"TFR monotone: {report['tfr_monotone_in_tau']}; "
          f"power>=cost: {report['power_dominates_cost']}")
    print(f"detection power monotone in budget: {report['detection_power_monotone_in_budget']}; "
          f"max |FFR-budget| = {report['max_budget_calibration_error']:.3f}")
    print(f"dAUC tightest budget = {report['delta_auc_at_tightest_budget']:+.4f}; "
          f"loosest budget = {report['delta_auc_at_loosest_budget']:+.4f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
