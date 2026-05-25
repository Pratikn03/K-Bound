"""Phase 2.2B — G3 top-q gate implementation correctness."""

from __future__ import annotations

import numpy as np
import pytest

from uais.fusion.attention.reliability_estimator import ReliabilityEstimator


def _estimator(top_q=1, top_q_threshold=0.5):
    return ReliabilityEstimator(
        domain_order=["a", "b", "c", "d"],
        score_index=0,
        gate_mode="top_q",
        top_q=top_q,
        top_q_threshold=top_q_threshold,
    )


def test_top_q_with_q1_equals_minimum_gate_when_thresholds_match():
    """G3 with q=1 reduces to the G1 minimum gate."""
    est = _estimator(top_q=1, top_q_threshold=0.4)
    weights = np.array([
        [0.9, 0.9, 0.9, 0.9],   # min=0.9 >= 0.4 → no fire
        [0.3, 0.9, 0.9, 0.9],   # min=0.3 < 0.4  → fire
    ])
    masks = np.zeros((2, 4), dtype=bool)
    fired = est.gate_decisions(weights, masks)
    assert fired.tolist() == [False, True]


def test_top_q_with_q2_does_not_fire_on_single_weak_domain():
    """With q=2, a single weak domain (rest strong) should NOT fire."""
    est = _estimator(top_q=2, top_q_threshold=0.4)
    weights = np.array([
        [0.3, 0.9, 0.9, 0.9],   # 2nd-smallest = 0.9 >= 0.4 → no fire
        [0.3, 0.3, 0.9, 0.9],   # 2nd-smallest = 0.3 <  0.4 → fire
    ])
    masks = np.zeros((2, 4), dtype=bool)
    fired = est.gate_decisions(weights, masks)
    assert fired.tolist() == [False, True]


def test_top_q_respects_per_sample_masks():
    """If fewer than q domains are present, gate stays closed (conservative)."""
    est = _estimator(top_q=2, top_q_threshold=0.4)
    weights = np.array([
        [0.3, 0.9, 0.9, 0.9],   # only domain 0 present → n_p=1 < q=2 → no fire
    ])
    masks = np.array([[False, True, True, True]])
    fired = est.gate_decisions(weights, masks)
    assert fired.tolist() == [False]


def test_top_q_rejects_q_less_than_1():
    est = _estimator()
    with pytest.raises(ValueError):
        # construction-time guard
        ReliabilityEstimator(
            domain_order=["a", "b"], score_index=0, gate_mode="top_q", top_q=0,
        )


def test_invalid_gate_mode_rejected():
    with pytest.raises(ValueError, match="gate_mode"):
        ReliabilityEstimator(
            domain_order=["a", "b"], score_index=0, gate_mode="not_a_gate",
        )
