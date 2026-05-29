"""Tests for the coherence-certified gate decision rule.

These tests encode the cross-benchmark contrast as unit cases: a coherent
collapse should allow switching, while legitimate heterogeneity should not,
and a failed switching certificate must veto an otherwise-coherent batch.
"""

from __future__ import annotations

import numpy as np

from uais.fusion.attention.gate_decision_rule import (
    decide_switch,
    drift_coherence,
    per_sample_mean_reliability,
)


def _no_mask(n: int, d: int) -> np.ndarray:
    return np.zeros((n, d), dtype=bool)


def test_per_sample_mean_reliability_ignores_missing_domains():
    weights = np.array([[0.8, 0.2], [0.6, 0.4]], dtype=float)
    masks = np.array([[False, True], [False, False]], dtype=bool)
    r = per_sample_mean_reliability(weights, masks)
    assert np.isclose(r[0], 0.8)  # second domain masked out
    assert np.isclose(r[1], 0.5)


def test_per_sample_mean_reliability_nan_when_all_missing():
    weights = np.array([[0.8, 0.2]], dtype=float)
    masks = np.array([[True, True]], dtype=bool)
    r = per_sample_mean_reliability(weights, masks)
    assert np.isnan(r[0])


def test_coherent_collapse_has_high_coherence():
    # Every sample's reliability collapses to the same low value.
    weights = np.full((64, 2), 0.2, dtype=float)
    coh = drift_coherence(weights, _no_mask(64, 2))
    assert coh > 0.9


def test_heterogeneous_mixture_has_low_coherence():
    # Half the batch is reliable, half collapsed -> dispersed -> low coherence.
    rng = np.random.default_rng(0)
    high = rng.uniform(0.85, 0.95, size=(32, 2))
    low = rng.uniform(0.05, 0.15, size=(32, 2))
    weights = np.vstack([high, low])
    coh = drift_coherence(weights, _no_mask(64, 2))
    # A maximally-dispersed 50/50 split sits near 0.2; well below the 0.5
    # switch gate, and far below the coherent-collapse case (>0.9).
    assert coh < 0.3


def test_switch_allowed_under_coherent_collapse():
    weights = np.full((50, 2), 0.2, dtype=float)
    decision = decide_switch(weights, _no_mask(50, 2), tau=0.66, coherence_min=0.5)
    assert decision.switch_allowed is True
    assert decision.coherence_ok is True
    # All samples are below tau and coherent -> all fire.
    assert decision.fire_rate == 1.0
    assert decision.decisions.sum() == 50


def test_switch_suppressed_under_heterogeneity():
    rng = np.random.default_rng(1)
    weights = np.vstack(
        [rng.uniform(0.85, 0.95, size=(25, 2)), rng.uniform(0.05, 0.15, size=(25, 2))]
    )
    decision = decide_switch(weights, _no_mask(50, 2), tau=0.66, coherence_min=0.5)
    # Even though half the samples are below tau, dispersion vetoes the switch.
    assert decision.coherence_ok is False
    assert decision.switch_allowed is False
    assert decision.fire_rate == 0.0
    assert decision.decisions.sum() == 0


def test_certificate_veto_blocks_coherent_switch():
    # Coherent collapse, but the calibration certificate fails (reliability
    # path is worse than static on the fired set) -> switch must be vetoed.
    weights = np.full((40, 2), 0.2, dtype=float)
    static_loss = np.full(40, 0.3)
    reliability_loss = np.full(40, 0.5)  # worse than static
    fire = np.ones(40, dtype=bool)
    decision = decide_switch(
        weights,
        _no_mask(40, 2),
        tau=0.66,
        coherence_min=0.5,
        calibration_static_loss=static_loss,
        calibration_reliability_loss=reliability_loss,
        calibration_fire_decisions=fire,
    )
    assert decision.coherence_ok is True
    assert decision.certified is False
    assert decision.switch_allowed is False
    assert decision.fire_rate == 0.0


def test_certificate_allows_switch_when_reliability_helps():
    weights = np.full((40, 2), 0.2, dtype=float)
    static_loss = np.full(40, 0.5)
    reliability_loss = np.full(40, 0.3)  # better than static on fired set
    fire = np.ones(40, dtype=bool)
    decision = decide_switch(
        weights,
        _no_mask(40, 2),
        tau=0.66,
        coherence_min=0.5,
        calibration_static_loss=static_loss,
        calibration_reliability_loss=reliability_loss,
        calibration_fire_decisions=fire,
        margin_epsilon=0.05,
    )
    assert decision.certified is True
    assert decision.switch_allowed is True
    assert decision.fire_rate == 1.0
