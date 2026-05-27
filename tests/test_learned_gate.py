"""Tests for LearnedReliabilityGate (alternative to heuristic τ threshold)."""

from __future__ import annotations

import numpy as np
import pytest

from uais.fusion.attention.learned_gate import (
    LearnedGateConfig,
    LearnedReliabilityGate,
)


def _rng_weights(n: int, d: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=(n, d)).astype(np.float32)


def test_unfitted_decide_raises():
    gate = LearnedReliabilityGate()
    with pytest.raises(RuntimeError):
        gate.decide(_rng_weights(4, 3), np.zeros((4, 3), dtype=bool))


def test_fit_from_episodes_learns_separator():
    rng = np.random.default_rng(7)
    n, d = 200, 3
    # Episode 1: low reliability → should fire
    w_low = rng.uniform(0.0, 0.4, size=(n, d)).astype(np.float32)
    # Episode 2: high reliability → should not fire
    w_high = rng.uniform(0.6, 1.0, size=(n, d)).astype(np.float32)
    masks = np.zeros((n, d), dtype=bool)
    episodes = [
        {"weights": w_low, "masks": masks, "labels": np.ones(n, dtype=int)},
        {"weights": w_high, "masks": masks, "labels": np.zeros(n, dtype=int)},
    ]
    gate = LearnedReliabilityGate().fit_from_episodes(episodes)

    # Test on held-out samples drawn from each regime
    test_low = rng.uniform(0.0, 0.4, size=(50, d)).astype(np.float32)
    test_high = rng.uniform(0.6, 1.0, size=(50, d)).astype(np.float32)
    decisions_low = gate.decide(test_low, np.zeros((50, d), dtype=bool))
    decisions_high = gate.decide(test_high, np.zeros((50, d), dtype=bool))
    # Should mostly fire on low-reliability inputs, not on high-reliability inputs
    assert decisions_low.mean() > 0.8
    assert decisions_high.mean() < 0.2


def test_decision_probabilities_shape_and_range():
    rng = np.random.default_rng(11)
    n, d = 100, 4
    masks = np.zeros((n, d), dtype=bool)
    episodes = [
        {
            "weights": rng.uniform(0.0, 0.4, size=(n, d)).astype(np.float32),
            "masks": masks,
            "labels": np.ones(n, dtype=int),
        },
        {
            "weights": rng.uniform(0.6, 1.0, size=(n, d)).astype(np.float32),
            "masks": masks,
            "labels": np.zeros(n, dtype=int),
        },
    ]
    gate = LearnedReliabilityGate().fit_from_episodes(episodes)
    probs = gate.decision_probabilities(_rng_weights(20, d, seed=99), np.zeros((20, d), dtype=bool))
    assert probs.shape == (20,)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_scalar_feature_mode():
    rng = np.random.default_rng(13)
    n, d = 150, 3
    w_low = rng.uniform(0.0, 0.4, size=(n, d)).astype(np.float32)
    w_high = rng.uniform(0.6, 1.0, size=(n, d)).astype(np.float32)
    masks = np.zeros((n, d), dtype=bool)
    gate = LearnedReliabilityGate(LearnedGateConfig(feature_mode="scalar"))
    gate.fit_from_episodes(
        [
            {"weights": w_low, "masks": masks, "labels": np.ones(n, dtype=int)},
            {"weights": w_high, "masks": masks, "labels": np.zeros(n, dtype=int)},
        ]
    )
    test_w = np.array([[0.1, 0.1, 0.1], [0.8, 0.8, 0.8]], dtype=np.float32)
    test_m = np.zeros_like(test_w, dtype=bool)
    decisions = gate.decide(test_w, test_m)
    assert decisions[0]  # low reliability → fire
    assert not decisions[1]  # high reliability → static


def test_degenerate_fit_constant_decision():
    """If only one class is present in training, return majority constant."""
    n, d = 20, 3
    w = np.full((n, d), 0.5, dtype=np.float32)
    masks = np.zeros((n, d), dtype=bool)
    episode = {"weights": w, "masks": masks, "labels": np.zeros(n, dtype=int)}
    gate = LearnedReliabilityGate().fit_from_episodes([episode])
    # All-zero training labels → constant False decision
    decisions = gate.decide(w, masks)
    assert decisions.shape == (n,)
    assert (~decisions).all()
    assert gate.train_label_balance_ == 0.0


def test_mask_indicator_changes_feature_dim():
    rng = np.random.default_rng(17)
    n, d = 80, 3
    masks = np.zeros((n, d), dtype=bool)
    masks[: n // 2, 0] = True
    episode = {
        "weights": rng.uniform(0.0, 1.0, size=(n, d)).astype(np.float32),
        "masks": masks,
        "labels": rng.integers(0, 2, size=n),
    }
    gate_with = LearnedReliabilityGate(LearnedGateConfig(use_mask_indicators=True))
    gate_without = LearnedReliabilityGate(LearnedGateConfig(use_mask_indicators=False))
    gate_with.fit_from_episodes([episode])
    gate_without.fit_from_episodes([episode])
    # Internal coefficient count must differ
    assert gate_with._clf.coef_.shape[1] != gate_without._clf.coef_.shape[1]


def test_empty_episodes_raises():
    with pytest.raises(ValueError):
        LearnedReliabilityGate().fit_from_episodes([])
