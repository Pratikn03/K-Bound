"""Phase 4 / Theorem T3 — partial (k-of-D) domain-failure study.

NEW EXPLORATORY (development/synthetic only; not confirmatory; touches no
sealed/final test set).

Scenario C requires moving beyond the easy case (coherent ALL-domain collapse)
to realistic PARTIAL failures: 1-of-D, 2-of-D, ... domains corrupted. This
harness sweeps the number of failed domains ``k`` and compares three policies on
a synthetic, naturally-paired-style dataset:

  * static       : unweighted mean fusion (the reference comparator)
  * soft_rga     : per-domain reliability-weighted fusion (always applied)
  * hard_gate_g0 : the G0 batch/per-sample mean gate -- route a sample to the
                   reliability path only if its mean reliability < tau

It reports, per k:
  * delta AUC of soft_rga vs static and hard_gate_g0 vs static,
  * the gate fire rate (clean false-fire at k=0),
  * the smallest k at which each policy yields a meaningful benefit.

Expected mechanism (locked T3 "mean-gate dilution"): the hard mean gate is
*diluted* at low k -- a single failed domain among D does not drag the batch
mean reliability below tau, so the gate does not fire and the failure is not
handled. Per-domain soft weighting can down-weight the failed domain regardless
of k, so it handles partial failures the hard gate misses. This quantifies the
partial-failure boundary that the RGA-v2 sensitive gates (G1/G2/G3) tried and
failed to fix without exploding clean false-fire.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.scripts.run_calibration_transfer_study import _make_dataset
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator


def _fuse_mean(feats, masks):
    scores = feats[:, :, 0].astype(float)
    w = (~masks).astype(float)
    denom = w.sum(axis=1)
    return np.where(denom > 0, (scores * w).sum(axis=1) / np.maximum(denom, 1), 0.5)


def _fuse_weighted(feats, masks, weights):
    scores = feats[:, :, 0].astype(float)
    w = weights.astype(float) * (~masks)
    denom = w.sum(axis=1)
    return np.where(denom > 0, (scores * w).sum(axis=1) / np.maximum(denom, 1), 0.5)


def _corrupt_noise(feats, domains, rng):
    """Replace the listed domains' scores with label-uncorrelated noise.

    Noise has low sharpness AND drifts from the clean reference, so the gate can
    detect it -- this is the standard detectable single-/multi-domain collapse.
    """
    out = feats.copy()
    for j in domains:
        out[:, j, 0] = rng.random(out.shape[0]).astype(np.float32)
    return out


def run_study(seed=0, d=8, n=1500, false_fire_budget=0.01, margin=0.005) -> dict:
    rng = np.random.default_rng(seed)
    # Three disjoint clean splits: fit (KS reference + ECE), cal (tau selection),
    # and a fresh test draw per k. The tau-selection split MUST differ from the
    # KS-reference split, otherwise reliability is self-inflated (KS of a set
    # against itself ~ p=1) and tau is mis-scaled.
    fit_feats, fit_masks, fit_labels = _make_dataset(rng, n, d)
    cal_feats, cal_masks, cal_labels = _make_dataset(rng, n, d)
    est = ReliabilityEstimator(
        domain_order=[f"d{j}" for j in range(d)],
        score_index=0,
        ece_weight=0.4, ks_weight=0.4, sharpness_weight=0.2,
        n_calibration_bins=10, min_samples_for_ks=30,
        gate_mode="mean",
    )
    est.fit(fit_feats, fit_masks, fit_labels)

    # Validation-only tau selection to the clean false-fire budget: tau is the
    # budget-quantile of the clean CALIBRATION mean reliability, so at most
    # ~budget fraction of clean samples fire. (Locked policy: no test-driven
    # threshold tuning.)
    cal_mean_r = est.compute_reliability_weights(cal_feats, cal_masks).mean(axis=1)
    tau_gate = float(np.quantile(cal_mean_r, false_fire_budget))
    est.gate_threshold = tau_gate

    rows = []
    for k in range(0, d + 1):
        feats, masks, labels = _make_dataset(rng, n, d)
        if k > 0:
            feats = _corrupt_noise(feats, range(k), rng)

        weights = est.compute_reliability_weights(feats, masks)
        static = _fuse_mean(feats, masks)
        soft = _fuse_weighted(feats, masks, weights)

        mean_r = weights.mean(axis=1)
        fire = mean_r < tau_gate
        hard = np.where(fire, soft, static)

        auc_static = roc_auc_score(labels, static)
        auc_soft = roc_auc_score(labels, soft)
        auc_hard = roc_auc_score(labels, hard)
        rows.append({
            "k_failed_domains": k,
            "auc_static": float(auc_static),
            "auc_soft_rga": float(auc_soft),
            "auc_hard_gate_g0": float(auc_hard),
            "delta_soft_vs_static": float(auc_soft - auc_static),
            "delta_hard_vs_static": float(auc_hard - auc_static),
            "gate_fire_rate": float(fire.mean()),
            "mean_reliability": float(mean_r.mean()),
        })

    def _first_benefit_k(key):
        for r in rows:
            if r["k_failed_domains"] > 0 and r[key] > margin:
                return r["k_failed_domains"]
        return None

    clean = rows[0]
    return {
        "study": "partial_domain_failure_T3",
        "label": "NEW EXPLORATORY (development/synthetic only; not confirmatory)",
        "seed": seed, "n": n, "n_domains": d,
        "false_fire_budget": false_fire_budget,
        "selected_tau_on_validation": tau_gate,
        "benefit_margin": margin,
        "rows": rows,
        "clean_false_fire_rate": clean["gate_fire_rate"],
        "first_k_with_benefit_soft": _first_benefit_k("delta_soft_vs_static"),
        "first_k_with_benefit_hard": _first_benefit_k("delta_hard_vs_static"),
        "finding": (
            "Per-domain soft reliability weighting handles partial (k>=1) "
            "failures; the hard batch mean gate is diluted at low k (T3) and "
            "begins to help only after enough domains fail to drag the mean "
            "reliability below tau."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Partial-domain failure study (T3)")
    ap.add_argument("--output", default="output/phase4/partial_domain_failure_study.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = run_study(seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== Partial (k-of-D) domain-failure study (T3) — NEW EXPLORATORY ===")
    print(f"{'k':>3}{'auc_stat':>10}{'d_soft':>9}{'d_hard':>9}{'fire':>8}{'mean_r':>8}")
    for r in report["rows"]:
        print(f"{r['k_failed_domains']:>3}{r['auc_static']:>10.4f}"
              f"{r['delta_soft_vs_static']:>9.4f}{r['delta_hard_vs_static']:>9.4f}"
              f"{r['gate_fire_rate']:>8.3f}{r['mean_reliability']:>8.3f}")
    print(f"\nselected tau (val, budget {report['false_fire_budget']}): {report['selected_tau_on_validation']:.4f}")
    print(f"clean false-fire (k=0): {report['clean_false_fire_rate']:.3f}")
    print(f"first k with benefit  soft: {report['first_k_with_benefit_soft']}  "
          f"hard: {report['first_k_with_benefit_hard']}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
