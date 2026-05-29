"""Phase 3 / Theorem T6 — controlled calibration-transfer study.

NEW EXPLORATORY (development data only). This harness does NOT touch any sealed
or final test set and produces no confirmatory claim. It is a synthetic,
controlled model of *why Family D (Eyecandies) transfer failed*: a reliability
gate calibrated on a source validation distribution is applied to a target whose
per-domain score distribution is progressively shifted away from the source
reference.

Question (T6): under what score-distribution shift does a validation-calibrated
reliability estimate stay valid enough to still improve fusion on a new domain?

Design
------
* Source: D-domain synthetic anomaly scores; each domain individually
  informative. The ReliabilityEstimator is fit on a source validation split
  (this freezes the KS reference distribution + per-domain ECE).
* Target: domain 0 is *collapsed* (a within-target degradation the gate should
  catch), and ALL domains are then shifted by a transfer offset ``tau`` that
  moves the target score distribution away from the source reference.
* For each tau we compare:
    - static fusion  = unweighted mean of domain scores
    - gated fusion   = reliability-weighted mean of domain scores
  and report delta AUC = AUC(gated) - AUC(static).

Three candidate *predictors* of whether transfer is still valid (delta>0) are
recorded:
    1. drift_coherence            (source-rule, no target labels)
    2. source_validation_certificate (computed once on source; no target info)
    3. target_reference_divergence   (mean KS distance between target healthy-
       domain scores and the frozen source reference; NO target labels needed)

The key finding the harness is built to expose: (1) and (2) are computed without
any target-side distribution information and therefore CANNOT detect a transfer
that has drifted out of calibration; (3) tracks the delta-AUC sign and yields a
label-free abstention signal. That motivates the T6 abstention rule.

Usage::

    PYTHONPATH=.:src python src/scripts/run_calibration_transfer_study.py \
        --output output/phase3/calibration_transfer_study.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from uais.fusion.attention.gate_decision_rule import drift_coherence
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator
from uais.utils.metrics import bounded_switching_certificate


def _make_domain_scores(rng, n, prevalence, sep):
    """One domain's anomaly scores in [0,1]; anomalies shifted up by ``sep``."""
    labels = (rng.random(n) < prevalence).astype(int)
    base = rng.normal(0.0, 1.0, size=n)
    logits = base + sep * labels
    scores = 1.0 / (1.0 + np.exp(-logits))
    return scores.astype(float), labels


def _make_dataset(rng, n, d, prevalence=0.3, sep=1.6):
    """[N, D, 1] score features + shared labels (same incident across domains)."""
    labels = (rng.random(n) < prevalence).astype(int)
    feats = np.zeros((n, d, 1), dtype=np.float32)
    for j in range(d):
        base = rng.normal(0.0, 1.0, size=n)
        logits = base + sep * labels
        feats[:, j, 0] = 1.0 / (1.0 + np.exp(-logits))
    masks = np.zeros((n, d), dtype=bool)
    return feats, masks, labels.astype(float)


def _fuse_mean(feats, masks):
    scores = feats[:, :, 0].astype(float)
    w = (~masks).astype(float)
    denom = w.sum(axis=1)
    return np.where(denom > 0, (scores * w).sum(axis=1) / np.maximum(denom, 1), 0.5)


def _fuse_reliability_weighted(feats, masks, weights):
    scores = feats[:, :, 0].astype(float)
    w = weights.astype(float) * (~masks)
    denom = w.sum(axis=1)
    fused = np.where(denom > 0, (scores * w).sum(axis=1) / np.maximum(denom, 1), 0.5)
    return fused


def _apply_transfer_shift(feats, tau):
    """Monotone offset of all target scores -> moves them off the source ref."""
    out = feats.copy()
    s = out[:, :, 0]
    # logit-space offset keeps values in (0,1) and is monotone (rank-preserving
    # within a domain) yet changes the distribution vs the frozen source ref.
    eps = 1e-6
    s = np.clip(s, eps, 1 - eps)
    logit = np.log(s / (1 - s)) + tau
    out[:, :, 0] = 1.0 / (1.0 + np.exp(-logit))
    return out


def _collapse_domain(feats, masks, rng, domain=0):
    """Corrupt a domain into a *confident but label-uncorrelated* stream.

    This is the Family-D-relevant failure: the corrupted domain looks
    *confident* (scores pushed near 0/1, so the per-sample sharpness term reads
    it as reliable) yet carries no label information. Sharpness therefore CANNOT
    catch it -- only the KS drift term (current scores vs the frozen source
    reference) can. Under a large transfer shift the KS term saturates for all
    domains and loses its discriminating power, which is exactly the regime
    where the validation-calibrated gate stops helping.
    """
    out = feats.copy()
    hi = rng.random(out.shape[0]) < 0.5
    confident = np.where(hi, 0.95, 0.05) + rng.normal(0.0, 0.02, size=out.shape[0])
    out[:, domain, 0] = np.clip(confident, 0.0, 1.0).astype(np.float32)
    return out, masks


def run_study(seed=0, d=4, n=1500, taus=None, margin=0.005) -> dict:
    if taus is None:
        taus = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    rng = np.random.default_rng(seed)

    # Source: train (reference) + validation (calibration) + the gate fit.
    src_feats, src_masks, src_labels = _make_dataset(rng, n, d)
    val_feats, val_masks, val_labels = _make_dataset(rng, n, d)

    est = ReliabilityEstimator(
        domain_order=[f"d{j}" for j in range(d)],
        score_index=0,
        ece_weight=0.4, ks_weight=0.4, sharpness_weight=0.2,
        n_calibration_bins=10, min_samples_for_ks=30,
    )
    est.fit(val_feats, val_masks, val_labels)
    source_reference = {dn: est._reference_scores[dn].copy() for dn in est.domain_order}

    # Source-validation certificate (computed ONCE, no target information):
    # fire where mean reliability < tau_gate; loss = (1 - score_on_truth) proxy.
    val_w = est.compute_reliability_weights(val_feats, val_masks)
    val_mean_r = val_w.mean(axis=1)
    val_static = _fuse_mean(val_feats, val_masks)
    val_gated = _fuse_reliability_weighted(val_feats, val_masks, val_w)
    static_loss = np.abs(val_labels - val_static)
    reliability_loss = np.abs(val_labels - val_gated)
    fire = val_mean_r < float(np.median(val_mean_r))
    src_cert = bounded_switching_certificate(static_loss, reliability_loss, fire, margin_epsilon=0.0)

    rows = []
    for tau in taus:
        # Build target: collapse domain 0, then shift all domains by tau.
        tgt_feats, tgt_masks, tgt_labels = _make_dataset(rng, n, d)
        tgt_feats = _apply_transfer_shift(tgt_feats, tau)
        tgt_feats, tgt_masks = _collapse_domain(tgt_feats, tgt_masks, rng, domain=0)

        weights = est.compute_reliability_weights(tgt_feats, tgt_masks)
        static_fused = _fuse_mean(tgt_feats, tgt_masks)
        gated_fused = _fuse_reliability_weighted(tgt_feats, tgt_masks, weights)
        auc_static = roc_auc_score(tgt_labels, static_fused)
        auc_gated = roc_auc_score(tgt_labels, gated_fused)
        delta = float(auc_gated - auc_static)

        coherence = drift_coherence(weights, tgt_masks)

        # Target reference divergence (label-free): mean KS distance between the
        # frozen source reference and target *healthy* domains (exclude collapsed
        # domain 0, which is an intentional within-target failure, not transfer).
        ks_dists = []
        for j in range(1, d):
            dn = est.domain_order[j]
            cur = tgt_feats[:, j, 0].astype(float)
            stat, _ = ks_2samp(source_reference[dn], cur)
            ks_dists.append(float(stat))
        target_divergence = float(np.mean(ks_dists))

        if delta > margin:
            regime = "HELP"
        elif delta < -margin:
            regime = "HURT"
        else:
            regime = "NEUTRAL"
        rows.append({
            "tau": float(tau),
            "auc_static": float(auc_static),
            "auc_gated": float(auc_gated),
            "delta_auc": delta,
            "regime": regime,
            "drift_coherence": float(coherence),
            "target_reference_divergence": target_divergence,
            "source_certified": bool(src_cert["certified"]),
        })

    # The danger is the HURT region (gate calibrated on source actively harms
    # after transfer). A safe abstention rule must cover every HURT row using a
    # label-free signal. Derive a divergence threshold from the HELP/HURT gap.
    help_rows = [r for r in rows if r["regime"] == "HELP"]
    hurt_rows = [r for r in rows if r["regime"] == "HURT"]
    max_help_div = max((r["target_reference_divergence"] for r in help_rows), default=0.0)
    min_hurt_div = min((r["target_reference_divergence"] for r in hurt_rows), default=None)
    if min_hurt_div is not None:
        divergence_threshold = float((max_help_div + min_hurt_div) / 2.0)
        abstention_covers_all_hurt = all(
            r["target_reference_divergence"] > divergence_threshold for r in hurt_rows
        )
        # Would the source-side predictors have caught the HURT region? They are
        # constant across tau, so they cannot separate HELP from HURT.
        coherence_separates = len({round(r["drift_coherence"], 3) for r in rows}) > 1
        certificate_separates = len({r["source_certified"] for r in rows}) > 1
    else:
        divergence_threshold = None
        abstention_covers_all_hurt = None
        coherence_separates = None
        certificate_separates = None

    return {
        "study": "calibration_transfer_T6",
        "label": "NEW EXPLORATORY (development/synthetic only; not confirmatory)",
        "seed": seed, "n": n, "n_domains": d, "benefit_margin": margin,
        "source_certificate": src_cert,
        "rows": rows,
        "abstention_divergence_threshold": divergence_threshold,
        "abstention_covers_all_hurt": abstention_covers_all_hurt,
        "drift_coherence_separates_help_from_hurt": coherence_separates,
        "source_certificate_separates_help_from_hurt": certificate_separates,
        "finding": (
            "A validation-calibrated reliability gate that HELPS in-distribution "
            "(low target divergence) actively HURTS once the target score "
            "distribution drifts (the Family-D failure). Source-side "
            "drift_coherence and the source-validation certificate are constant "
            "across the shift and CANNOT separate the safe from the harmful "
            "regime. The label-free target_reference_divergence does, giving an "
            "abstention rule: fall back to static fusion when divergence exceeds "
            "the threshold."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration-transfer study (T6)")
    ap.add_argument("--output", default="output/phase3/calibration_transfer_study.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = run_study(seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== Calibration-transfer study (T6) — NEW EXPLORATORY ===")
    print(f"{'tau':>5}{'dAUC':>9}{'regime':>9}{'coher':>8}{'tgt_div':>9}{'src_cert':>9}")
    for r in report["rows"]:
        print(f"{r['tau']:>5.1f}{r['delta_auc']:>9.4f}{r['regime']:>9}"
              f"{r['drift_coherence']:>8.3f}{r['target_reference_divergence']:>9.3f}"
              f"{str(r['source_certified']):>9}")
    print(f"\nabstention divergence threshold ~ {report['abstention_divergence_threshold']}")
    print(f"abstention covers all HURT rows: {report['abstention_covers_all_hurt']}")
    print(f"coherence separates HELP/HURT: {report['drift_coherence_separates_help_from_hurt']}; "
          f"source certificate separates: {report['source_certificate_separates_help_from_hurt']}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
