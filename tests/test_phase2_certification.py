"""Phase 2.G — risk-dominance + switching-certificate smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from elara.certification import (  # noqa: E402
    estimate_risk_dominance,
    fired_subset_certificate,
    paired_bootstrap_lcb,
)


def test_risk_dominance_returns_finite_terms():
    rng = np.random.default_rng(0)
    n = 200
    # Build a synthetic scenario where the gated path has clearly lower loss
    # in the degraded fold and slightly higher loss in the clean fold.
    y_clean = (rng.random(n) < 0.3).astype(int)
    y_deg = (rng.random(n) < 0.3).astype(int)
    static_clean = np.clip(y_clean + rng.normal(0, 0.2, n), 0, 1)
    gated_clean = np.clip(y_clean + rng.normal(0, 0.25, n), 0, 1)  # slightly worse
    static_deg = np.clip(1 - y_deg + rng.normal(0, 0.2, n), 0, 1)   # static is wrong under degradation
    gated_deg = np.clip(y_deg + rng.normal(0, 0.2, n), 0, 1)
    fired_clean = rng.random(n) < 0.05
    fired_deg = rng.random(n) < 0.95
    out = estimate_risk_dominance(
        gate_id="G0", scenario_id="synthetic",
        clean_static_scores=static_clean, clean_gated_scores=gated_clean,
        clean_gate_fired=fired_clean, clean_labels=y_clean,
        degraded_static_scores=static_deg, degraded_gated_scores=gated_deg,
        degraded_gate_fired=fired_deg, degraded_labels=y_deg,
    )
    assert np.isfinite(out.q0)
    assert np.isfinite(out.q1)
    assert np.isfinite(out.delta_0)
    assert np.isfinite(out.delta_1)
    assert out.q0 < out.q1  # gate should fire more under degradation by construction


def test_paired_bootstrap_lcb_basic():
    arr = np.array([0.1, 0.2, 0.15, 0.18, 0.22, 0.05, 0.13])
    mean, lcb = paired_bootstrap_lcb(arr, alpha=0.05, n_iter=2000, seed=0)
    assert lcb <= mean
    assert lcb > -1.0 and lcb < 1.0


def test_fired_subset_certificate_positive_when_gated_dominates():
    rng = np.random.default_rng(1)
    n = 500
    y = (rng.random(n) < 0.3).astype(int)
    static = np.clip(1 - y + rng.normal(0, 0.1, n), 0, 1)  # wrong polarity
    gated = np.clip(y + rng.normal(0, 0.1, n), 0, 1)        # correct
    fired = np.ones(n, dtype=bool)
    cert = fired_subset_certificate(
        gate_id="G_dominant",
        scenario_id="synthetic_degraded",
        static_scores=static,
        gated_scores=gated,
        labels=y,
        gate_fired=fired,
        alpha=0.05,
        n_iter=2000,
    )
    assert cert.mean_paired_benefit > 0
    assert cert.bootstrap_lcb > 0
    assert cert.certified is True


def test_fired_subset_certificate_negative_when_no_fired_samples():
    n = 50
    rng = np.random.default_rng(2)
    cert = fired_subset_certificate(
        gate_id="G_silent",
        scenario_id="synthetic_clean",
        static_scores=rng.random(n),
        gated_scores=rng.random(n),
        labels=(rng.random(n) < 0.3).astype(int),
        gate_fired=np.zeros(n, dtype=bool),
    )
    assert cert.n_fired_samples == 0
    assert cert.certified is False
