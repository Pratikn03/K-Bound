"""Unit tests for ELARA deploy policy routing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from uais.fusion.attention.baselines import SARScoreAdapter
from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
from uais.fusion.attention.elara_deploy_policy import (
    DeployPolicySpec,
    ElaraDeployArtifacts,
    predict_elara_deploy,
    select_validation_fallback,
)
from uais.fusion.attention.fusion_inference import GateDecisionCalibration
from uais.fusion.attention.gate_decision_rule import drift_coherence
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator


def _toy_setup(n_train: int = 24, n_eval: int = 16):
    torch.manual_seed(0)
    n, d, f = n_train + n_eval, 2, 4
    features = np.random.rand(n, d, f).astype(np.float32)
    masks = np.zeros((n, d), dtype=bool)
    labels = (features[:, 0, 0] > 0.5).astype(int)
    train_feat, train_mask, train_labels = features[:n_train], masks[:n_train], labels[:n_train]
    eval_feat, eval_mask = features[n_train:], masks[n_train:]

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
    estimator = ReliabilityEstimator(domain_order=["a", "b"], score_index=0, gate_threshold=0.66)
    estimator.fit(train_feat, train_mask, train_labels[:n_train])
    sar = SARScoreAdapter(random_seed=0, adaptation_steps=0).fit(train_feat, train_mask, train_labels)
    return model, estimator, sar, eval_feat, eval_mask


def test_heterogeneous_batch_uses_sar_fallback():
    model, estimator, sar, eval_feat, eval_mask = _toy_setup()
    weights = np.array([[0.05, 0.05], [0.95, 0.95]], dtype=float)
    masks = np.zeros((2, 2), dtype=bool)
    assert drift_coherence(weights, masks) < 0.5

    def _dispersed_weights(feat, mask):
        return weights[: feat.shape[0]]

    estimator.compute_reliability_weights = _dispersed_weights  # type: ignore[method-assign]

    artifacts = ElaraDeployArtifacts(
        model=model,
        estimator=estimator,
        sar=sar,
        device=torch.device("cpu"),
        gate_decision_calibration=GateDecisionCalibration(
            np.array([0.5, 0.5]),
            np.array([0.4, 0.4]),
            np.array([True, True]),
        ),
        gate_decision_rule_cfg={"enabled": True, "coherence_min": 0.5, "tau": 0.66, "margin_epsilon": 0.0},
        batch_size=8,
    )
    small_feat = eval_feat[:2]
    small_mask = eval_mask[:2]
    probs, stats = predict_elara_deploy(artifacts, small_feat, small_mask)
    sar_only = sar.predict_proba(small_feat, small_mask)
    assert probs.shape == sar_only.shape
    np.testing.assert_allclose(probs, sar_only, rtol=1e-5, atol=1e-5)
    assert stats["sar_fallback_rate"] == pytest.approx(1.0)


def test_select_validation_fallback_picks_rga_when_better():
    labels = np.array([0, 0, 1, 1], dtype=int)
    sar_probs = np.array([0.2, 0.3, 0.4, 0.5], dtype=float)
    rga_probs = np.array([0.1, 0.2, 0.9, 0.95], dtype=float)
    choice, _, _ = select_validation_fallback(labels, sar_probs, rga_probs)
    assert choice == "rga_boosted_fusion"


def test_deploy_policy_spec_from_mapping():
    spec = DeployPolicySpec.from_mapping(
        {
            "policy_id": "TEST",
            "routing": {"mode": "gdr_sar_fallback", "gate_decision_rule": {"enabled": True, "tau": 0.5}},
        }
    )
    assert spec.policy_id == "TEST"
    assert spec.gate_decision_rule["tau"] == 0.5
