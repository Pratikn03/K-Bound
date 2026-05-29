"""Tests for transfer pipeline: frozen calibrators + gate-decision CRAF path."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import QuantileTransformer

from uais.fusion.attention.frozen_calibrators import (
    FrozenCalibratorBundle,
    apply_calibrators_to_features,
)
from uais.fusion.attention.fusion_inference import (
    GateDecisionCalibration,
    build_gate_decision_calibration,
    decide_switch_batch,
)
from uais.fusion.attention.gate_decision_rule import drift_coherence
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator


def test_apply_calibrators_monotone_on_scores():
    features = np.array([[[0.2, 0.5], [0.8, 0.5]]], dtype=np.float32)
    masks = np.array([[False, False]], dtype=bool)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    bundle = FrozenCalibratorBundle(
        dataset_key="toy",
        domain_order=["a", "b"],
        score_index=0,
        models={"a": iso, "b": iso},
    )
    out = apply_calibrators_to_features(features, masks, ["a", "b"], bundle, score_index=0)
    assert out[0, 0, 0] == pytest.approx(0.2, rel=1e-3)
    assert out[0, 1, 0] == pytest.approx(0.8, rel=1e-3)


def test_quantile_calibrator_single_class_validation():
    qt = QuantileTransformer(n_quantiles=10, output_distribution="uniform", random_state=0)
    raw = np.linspace(0.01, 0.99, 50)
    qt.fit(raw.reshape(-1, 1))
    bundle = FrozenCalibratorBundle(
        dataset_key="toy_q",
        domain_order=["a"],
        score_index=0,
        models={"a": qt},
    )
    features = np.array([[[0.1, 0.0]]], dtype=np.float32)
    masks = np.array([[False]], dtype=bool)
    out = apply_calibrators_to_features(features, masks, ["a"], bundle, score_index=0)
    assert 0.0 <= out[0, 0, 0] <= 1.0


def test_heterogeneous_batch_blocks_gate_decision_switch():
    # Per-sample mean reliabilities dispersed (0.2 vs 0.8) → low coherence.
    weights = np.array([[0.05, 0.05], [0.95, 0.95], [0.35, 0.35]], dtype=float)
    masks = np.zeros((3, 2), dtype=bool)
    assert drift_coherence(weights, masks) < 0.5
    cal = GateDecisionCalibration(
        np.array([0.5, 0.5, 0.5]),
        np.array([0.4, 0.4, 0.4]),
        np.array([True, True, True]),
    )
    decision = decide_switch_batch(weights, masks, tau=0.66, calibration=cal, coherence_min=0.5)
    assert not decision.switch_allowed
    assert not decision.decisions.any()


def test_predict_craf_with_gate_decision_rule_smoke():
    from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
    from src.scripts.run_breakthrough_experiment import _predict_craf_with_stats

    torch.manual_seed(0)
    n, d, f = 32, 2, 4
    features = np.random.rand(n, d, f).astype(np.float32)
    masks = np.zeros((n, d), dtype=bool)
    labels = (features[:, 0, 0] > 0.5).astype(int)
    model = AttentionFusionModel(
        input_dim=f,
        num_domains=d,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        use_confidence=False,
        use_attention=True,
        use_domain_embeddings=False,
        use_positional_embeddings=False,
        use_missing_embedding=True,
    )
    estimator = ReliabilityEstimator(
        domain_order=["a", "b"],
        score_index=0,
        gate_threshold=0.66,
    )
    estimator.fit(features[:16], masks[:16], labels[:16])
    device = torch.device("cpu")
    cal = build_gate_decision_calibration(
        model,
        estimator,
        features[:16],
        masks[:16],
        labels[:16],
        device,
        tau=0.66,
    )
    gate_cfg = {"enabled": True, "coherence_min": 0.5, "margin_epsilon": 0.0, "tau": 0.66}
    probs, stats = _predict_craf_with_stats(
        model,
        estimator,
        features[16:],
        masks[16:],
        device,
        clean_gate_threshold=0.66,
        gate_decision_rule_cfg=gate_cfg,
        gate_decision_calibration=cal,
    )
    assert probs.shape == (16,)
    assert stats["gate_decision_rule"] is True
