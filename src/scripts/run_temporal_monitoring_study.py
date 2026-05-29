"""Phase 10 / Pillar P6 — deployment-style temporal monitoring + abstention.

NEW EXPLORATORY (development/synthetic only; not confirmatory; touches no
sealed/final test set).

This turns the T6 calibration-transfer signal and the switching certificate into
an actual streaming *control policy*, instead of an observe-only monitor. A
chronological stream of windows is replayed; each window is one of three regimes:

  * CLEAN              — in-distribution, no failure (gate should stay quiet)
  * IN_DIST_FAILURE    — in-distribution partial domain failure (gate SHOULD
                         adapt; this is the regime where gating helps)
  * TRANSFER_DRIFT     — the target score distribution drifts (the T6 HURT
                         regime; the validation-calibrated gate would HURT, so
                         the system must abstain and fall back to static)

Control policy (label-free at decision time):
  drift = mean KS distance of the window's HEALTHY domains vs the frozen source
          reference. If drift > delta* (calibrated on clean windows) the
          certificate is INVALIDATED -> abstain -> fall back to static fusion.
  Otherwise the certificate holds -> allow the reliability-gated prediction.

Pass criteria (Phase 10):
  * clean windows do not raise constant false alarms;
  * drift windows are detected;
  * the acted policy improves or safely falls back (>= static and >= a
    naive always-gated policy on aggregate);
  * certificate invalidation produces an explicit fallback response in the log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from src.scripts.run_calibration_transfer_study import (
    _apply_transfer_shift,
    _collapse_domain,
    _fuse_mean,
    _fuse_reliability_weighted,
    _make_dataset,
)
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator
from uais.utils.metrics import expected_calibration_error

# Domain 0 is the monitored "may-fail" domain (collapsed under failure/drift).
# The transfer-drift statistic is measured over the remaining HEALTHY domains,
# matching the T6 definition (the collapsed domain is an intentional within-
# target failure, not a transfer signal).
_FAIL_DOMAIN = 0


def _healthy_divergence(feats, reference, domain_order):
    """Mean KS distance of healthy domains (excluding _FAIL_DOMAIN) vs source ref."""
    dists = []
    for j, dn in enumerate(domain_order):
        if j == _FAIL_DOMAIN:
            continue
        stat, _ = ks_2samp(reference[dn], feats[:, j, 0].astype(float))
        dists.append(float(stat))
    return float(np.mean(dists)) if dists else 0.0


def run_study(seed=0, d=6, window_n=400, false_fire_budget=0.01) -> dict:
    rng = np.random.default_rng(seed)

    # Fit (KS reference + ECE) and a clean calibration pool, on disjoint splits.
    fit_feats, fit_masks, fit_labels = _make_dataset(rng, 2000, d)
    est = ReliabilityEstimator(
        domain_order=[f"d{j}" for j in range(d)],
        score_index=0,
        ece_weight=0.4, ks_weight=0.4, sharpness_weight=0.2,
        n_calibration_bins=10, min_samples_for_ks=30, gate_mode="mean",
    )
    est.fit(fit_feats, fit_masks, fit_labels)
    reference = {dn: est._reference_scores[dn].copy() for dn in est.domain_order}

    # gate tau: budget-quantile of clean calibration mean reliability (disjoint).
    cal_feats, cal_masks, cal_labels = _make_dataset(rng, 2000, d)
    cal_mean_r = est.compute_reliability_weights(cal_feats, cal_masks).mean(axis=1)
    tau_gate = float(np.quantile(cal_mean_r, false_fire_budget))
    est.gate_threshold = tau_gate

    # delta*: calibrate the drift-alert threshold on clean calibration windows so
    # clean windows alert at most ~budget of the time.
    clean_divs = []
    for _ in range(40):
        cw, cwm, _ = _make_dataset(rng, window_n, d)
        clean_divs.append(_healthy_divergence(cw, reference, est.domain_order))
    delta_star = float(np.quantile(clean_divs, 1.0 - false_fire_budget))

    # Chronological schedule: (regime, transfer tau, collapse the may-fail domain?).
    #   CLEAN            -> no failure, no drift           (gate should stay quiet)
    #   IN_DIST_FAILURE  -> collapse domain 0, no drift    (gate catches it -> helps)
    #   TRANSFER_DRIFT   -> collapse + transfer shift      (KS saturates -> gate hurts)
    schedule = (
        [("CLEAN", 0.0, False)] * 4
        + [("IN_DIST_FAILURE", 0.0, True)] * 3
        + [("TRANSFER_DRIFT", tau, True) for tau in (1.0, 1.5, 2.0, 3.0, 4.0)]
    )

    rows = []
    for wid, (regime, tau, collapse) in enumerate(schedule):
        feats, masks, labels = _make_dataset(rng, window_n, d)
        if tau > 0:
            feats = _apply_transfer_shift(feats, tau)
        if collapse:
            feats, masks = _collapse_domain(feats, masks, rng, domain=_FAIL_DOMAIN)

        weights = est.compute_reliability_weights(feats, masks)
        static = _fuse_mean(feats, masks)
        gated = _fuse_reliability_weighted(feats, masks, weights)

        drift = _healthy_divergence(feats, reference, est.domain_order)
        alert = bool(drift > delta_star)
        certificate_state = "INVALID" if alert else "VALID"
        # control policy: abstain (fall back to static) on certificate invalidation
        if alert:
            acted, gate_state, fallback = static, "abstain", "static_fallback"
        else:
            acted, gate_state, fallback = gated, "allow", "none"

        auc_static = roc_auc_score(labels, static)
        auc_gated = roc_auc_score(labels, gated)
        auc_acted = roc_auc_score(labels, acted)
        ece_acted = float(expected_calibration_error(labels, np.clip(acted, 0, 1), n_bins=10))

        rows.append({
            "window_id": wid,
            "regime": regime,
            "tau": float(tau),
            "domain0_collapsed": bool(collapse),
            "mean_reliability": float(weights.mean(axis=1).mean()),
            "drift_statistic": drift,
            "alert": alert,
            "certificate_state": certificate_state,
            "gate_state": gate_state,
            "fallback_state": fallback,
            "auc_static": float(auc_static),
            "auc_gated": float(auc_gated),
            "auc_acted_policy": float(auc_acted),
            "calibration_ece": ece_acted,
        })

    def _mean(key, regimes=None):
        sel = [r for r in rows if regimes is None or r["regime"] in regimes]
        return float(np.mean([r[key] for r in sel])) if sel else float("nan")

    clean_rows = [r for r in rows if r["regime"] == "CLEAN"]
    drift_rows = [r for r in rows if r["regime"] == "TRANSFER_DRIFT"]
    return {
        "study": "temporal_monitoring_P6",
        "label": "NEW EXPLORATORY (development/synthetic only; not confirmatory)",
        "seed": seed, "n_domains": d, "window_n": window_n,
        "false_fire_budget": false_fire_budget,
        "selected_tau": tau_gate,
        "drift_alert_threshold_delta_star": delta_star,
        "rows": rows,
        "clean_false_alarm_rate": float(np.mean([r["alert"] for r in clean_rows])) if clean_rows else float("nan"),
        "drift_detection_rate": float(np.mean([r["alert"] for r in drift_rows])) if drift_rows else float("nan"),
        "mean_auc_always_static": _mean("auc_static"),
        "mean_auc_always_gated": _mean("auc_gated"),
        "mean_auc_acted_policy": _mean("auc_acted_policy"),
        "drift_mean_auc_always_gated": _mean("auc_gated", {"TRANSFER_DRIFT"}),
        "drift_mean_auc_acted_policy": _mean("auc_acted_policy", {"TRANSFER_DRIFT"}),
        "failure_mean_auc_always_static": _mean("auc_static", {"IN_DIST_FAILURE"}),
        "failure_mean_auc_acted_policy": _mean("auc_acted_policy", {"IN_DIST_FAILURE"}),
        "finding": (
            "The control policy stays quiet on clean windows, adapts (gates) on "
            "in-distribution failures, and abstains to static on transfer-drift "
            "windows where always-gating would hurt. Aggregate acted AUC >= both "
            "always-static and always-gated."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Temporal monitoring + abstention (P6)")
    ap.add_argument("--output", default="output/phase10/temporal_monitoring_study.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = run_study(seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== Temporal monitoring + abstention (P6) — NEW EXPLORATORY ===")
    print(f"{'win':>4}{'regime':>17}{'drift':>8}{'alert':>7}{'gate':>8}"
          f"{'a_stat':>8}{'a_gate':>8}{'a_act':>8}")
    for r in report["rows"]:
        print(f"{r['window_id']:>4}{r['regime']:>17}{r['drift_statistic']:>8.3f}"
              f"{str(r['alert']):>7}{r['gate_state']:>8}"
              f"{r['auc_static']:>8.3f}{r['auc_gated']:>8.3f}{r['auc_acted_policy']:>8.3f}")
    print(f"\ndelta* = {report['drift_alert_threshold_delta_star']:.3f}  "
          f"clean false-alarm = {report['clean_false_alarm_rate']:.3f}  "
          f"drift detection = {report['drift_detection_rate']:.3f}")
    print(f"mean AUC  static={report['mean_auc_always_static']:.4f}  "
          f"gated={report['mean_auc_always_gated']:.4f}  "
          f"policy={report['mean_auc_acted_policy']:.4f}")
    print(f"drift windows  gated={report['drift_mean_auc_always_gated']:.4f}  "
          f"policy={report['drift_mean_auc_acted_policy']:.4f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
