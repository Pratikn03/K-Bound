from __future__ import annotations

import numpy as np
import torch

from uais.fusion.attention.cross_modal_attention import AttentionFusionModel
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator


def test_gate_stats_reports_sample_weighted_adaptation():
    from src.scripts.run_breakthrough_experiment import _gate_decision_stats

    weights = np.array(
        [
            [0.9, 0.2, 0.0],
            [0.8, 0.1, 0.4],
        ],
        dtype=np.float32,
    )
    masks = np.array(
        [
            [False, False, True],
            [False, True, False],
        ],
        dtype=bool,
    )

    stats = _gate_decision_stats(weights, masks, threshold=0.7)

    assert stats["adapted"] is True
    assert stats["n_samples"] == 2
    assert stats["n_present"] == 4
    assert np.isclose(stats["mean_reliability"], 0.575)


def test_gate_stats_hybrid_mode_catches_minimum_reliability_failure():
    from src.scripts.run_breakthrough_experiment import _gate_decision_stats

    weights = np.array([[0.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    masks = np.zeros_like(weights, dtype=bool)

    mean_stats = _gate_decision_stats(weights, masks, threshold=0.66, gate_mode="mean")
    hybrid_stats = _gate_decision_stats(
        weights,
        masks,
        threshold=0.66,
        gate_mode="hybrid",
        min_gate_threshold=0.34,
    )

    assert mean_stats["adapted"] is False
    assert hybrid_stats["adapted"] is True
    assert np.isclose(hybrid_stats["min_reliability"], 0.0)


def test_k_domain_corruption_conditions_cover_requested_cardinalities():
    from src.scripts.run_breakthrough_experiment import _k_domain_corruption_conditions

    domain_order = ["d0", "d1", "d2", "d3"]
    features = np.ones((2, 4, 2), dtype=np.float32)
    masks = np.zeros((2, 4), dtype=bool)

    conditions = _k_domain_corruption_conditions(
        features,
        masks,
        domain_order,
        score_index=0,
        attack_name="zero_attack",
        k_values=[0, 1, 2, 4],
        sigma=0.1,
        seed=123,
    )

    counts = [condition["failed_domain_count"] for condition in conditions]
    assert counts.count(0) == 1
    assert counts.count(1) == 4
    assert counts.count(2) == 6
    assert counts.count(4) == 1

    d0_d1 = next(condition for condition in conditions if condition["failed_domains"] == "d0,d1")
    np.testing.assert_allclose(d0_d1["features"][:, :2, 0], 0.0)
    np.testing.assert_allclose(d0_d1["features"][:, 2:, 0], 1.0)


def test_reliability_component_weights_disable_and_renormalize():
    from src.scripts.run_breakthrough_experiment import _component_weights

    cfg = {"ece_weight": 0.45, "ks_weight": 0.35, "sharpness_weight": 0.20}

    weights = _component_weights(cfg, disabled=("ece",))

    assert weights == {
        "ece_weight": 0.0,
        "ks_weight": 0.35 / 0.55,
        "sharpness_weight": 0.20 / 0.55,
    }


def test_tau_sweep_rows_aggregate_with_gate_metrics():
    from uais.utils.result_aggregation import aggregate_stress_rows

    rows = [
        {
            "seed": 1,
            "condition": "clean",
            "tau": 0.5,
            "static_auc": 0.80,
            "craf_auc": 0.82,
            "adaptation_rate": 0.25,
            "mean_reliability": 0.70,
        },
        {
            "seed": 2,
            "condition": "clean",
            "tau": 0.5,
            "static_auc": 0.84,
            "craf_auc": 0.85,
            "adaptation_rate": 0.50,
            "mean_reliability": 0.72,
        },
    ]

    summary = aggregate_stress_rows(
        rows,
        group_keys=("condition", "tau"),
        metric_keys=("static_auc", "craf_auc", "adaptation_rate", "mean_reliability"),
    )

    clean = summary[0]
    assert clean["n_seeds"] == 2
    assert np.isclose(clean["static_auc"], 0.82)
    assert np.isclose(clean["craf_auc"], 0.835)
    assert np.isclose(clean["delta_auc"], 0.015)
    assert np.isclose(clean["adaptation_rate"], 0.375)


def _make_rga_fixture(seed: int = 42):
    """Tiny fitted model + estimator for gating-mode tests."""
    from src.scripts.run_breakthrough_experiment import _predict_craf_with_stats

    rng = np.random.default_rng(seed)
    domain_order = ["a", "b"]
    score_index = 0
    n_val, n_test, D, F = 80, 32, 2, 2

    val_feat = rng.normal(size=(n_val, D, F)).astype(np.float32)
    val_feat[..., score_index] = np.clip(val_feat[..., score_index] * 0.2 + 0.5, 0.05, 0.95)
    val_mask = np.zeros((n_val, D), dtype=bool)
    val_labels = (val_feat[:, score_index, score_index] > 0.5).astype(np.float32)

    test_feat = rng.normal(size=(n_test, D, F)).astype(np.float32)
    test_feat[..., score_index] = np.clip(test_feat[..., score_index] * 0.2 + 0.5, 0.05, 0.95)
    test_mask = np.zeros((n_test, D), dtype=bool)

    model = AttentionFusionModel(
        num_domains=D,
        input_dim=F,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        use_confidence=False,
        use_input_confidence=False,
    )
    model.eval()

    estimator = ReliabilityEstimator(
        domain_order=domain_order,
        score_index=score_index,
        ece_weight=0.45,
        ks_weight=0.35,
        sharpness_weight=0.20,
        gate_threshold=0.66,
    )
    estimator.fit(val_feat, val_mask, val_labels)
    return model, estimator, test_feat, test_mask, _predict_craf_with_stats


def test_per_sample_gating_high_threshold_matches_full_reliability_path():
    """At τ=1.01 (impossible to satisfy), every sample is gated regardless of mode."""
    model, estimator, test_feat, test_mask, predict = _make_rga_fixture()
    probs_batch, stats_batch = predict(
        model,
        estimator,
        test_feat,
        test_mask,
        torch.device("cpu"),
        clean_gate_threshold=1.01,
        per_sample_gating=False,
    )
    probs_persample, stats_persample = predict(
        model,
        estimator,
        test_feat,
        test_mask,
        torch.device("cpu"),
        clean_gate_threshold=1.01,
        per_sample_gating=True,
    )
    # All samples adapted in both modes
    assert stats_batch["adaptation_rate"] == 1.0
    assert stats_persample["adaptation_rate"] == 1.0
    # Outputs identical when every sample is below threshold
    np.testing.assert_allclose(probs_batch, probs_persample, atol=1e-6)


def test_per_sample_gating_low_threshold_matches_static_path():
    """At τ=0.0, no sample is below threshold, both modes return static path."""
    model, estimator, test_feat, test_mask, predict = _make_rga_fixture()
    probs_batch, stats_batch = predict(
        model,
        estimator,
        test_feat,
        test_mask,
        torch.device("cpu"),
        clean_gate_threshold=0.0,
        per_sample_gating=False,
    )
    probs_persample, stats_persample = predict(
        model,
        estimator,
        test_feat,
        test_mask,
        torch.device("cpu"),
        clean_gate_threshold=0.0,
        per_sample_gating=True,
    )
    assert stats_batch["adaptation_rate"] == 0.0
    assert stats_persample["adaptation_rate"] == 0.0
    np.testing.assert_allclose(probs_batch, probs_persample, atol=1e-6)


def test_per_sample_gating_reports_flag_in_stats():
    model, estimator, test_feat, test_mask, predict = _make_rga_fixture()
    _, stats = predict(
        model,
        estimator,
        test_feat,
        test_mask,
        torch.device("cpu"),
        per_sample_gating=True,
    )
    assert stats["per_sample_gating"] is True
    _, stats2 = predict(
        model,
        estimator,
        test_feat,
        test_mask,
        torch.device("cpu"),
        per_sample_gating=False,
    )
    assert stats2["per_sample_gating"] is False


def test_per_sample_gating_adapts_only_below_threshold_samples():
    """With a mid-range threshold, per-sample mode should gate only some samples."""
    model, estimator, test_feat, test_mask, predict = _make_rga_fixture()
    # Choose τ so the gate splits the batch
    craf_w = estimator.compute_reliability_weights(test_feat, test_mask)
    mean_r = craf_w.mean(axis=1)
    tau = float(np.median(mean_r))  # half above, half below

    _, stats = predict(
        model,
        estimator,
        test_feat,
        test_mask,
        torch.device("cpu"),
        clean_gate_threshold=tau,
        per_sample_gating=True,
    )
    expected_rate = float((mean_r < tau).mean())
    assert abs(stats["adaptation_rate"] - expected_rate) <= 1.0 / len(mean_r)
