"""Tests for ISSUE 3 (score-blend bypasses attention) and ISSUE 4 (tautological
one-class supervision target) fixes. All new behaviour is opt-in; the defaults
reproduce the legacy paths.
"""

from __future__ import annotations

import numpy as np
import torch

from src.scripts.run_breakthrough_experiment import (
    _build_model,
    _dropout_score_input,
    _predict_craf_with_stats,
    _pseudo_targets_from_domain_scores,
)
from uais.fusion.attention.reliability_estimator import ReliabilityEstimator


# --------------------------------------------------------------------------
# ISSUE 4: less-tautological pseudo targets + score-input dropout
# --------------------------------------------------------------------------

def _toy_scores():
    # 3 samples, 3 domains, 2 features; score column is index 0.
    feats = np.zeros((3, 3, 2), dtype=np.float32)
    feats[0, :, 0] = [0.1, 0.5, 0.9]
    feats[1, :, 0] = [0.2, 0.2, 0.8]
    feats[2, :, 0] = [0.4, 0.4, 0.4]
    masks = np.zeros((3, 3), dtype=bool)
    return feats, masks


def test_max_aggregation_is_copyable_but_quantile_is_not():
    feats, masks = _toy_scores()
    max_t = _pseudo_targets_from_domain_scores(feats, masks, 0, aggregation="max")
    assert np.allclose(max_t, [0.9, 0.8, 0.4])  # exactly the strongest domain score
    q_t = _pseudo_targets_from_domain_scores(feats, masks, 0, aggregation="quantile", quantile=0.5)
    # The 0.5 quantile (median) requires integrating domains; differs from max.
    assert np.allclose(q_t, [0.5, 0.2, 0.4])
    assert not np.allclose(max_t, q_t)


def test_mean_and_trimmed_mean_aggregations():
    feats, masks = _toy_scores()
    mean_t = _pseudo_targets_from_domain_scores(feats, masks, 0, aggregation="mean")
    assert np.allclose(mean_t, [0.5, 0.4, 0.4])
    trimmed = _pseudo_targets_from_domain_scores(feats, masks, 0, aggregation="trimmed_mean", trim_frac=0.0)
    assert np.allclose(trimmed, mean_t)


def test_pseudo_targets_handle_all_missing_row():
    feats, masks = _toy_scores()
    masks[1, :] = True  # fully missing row -> neutral 0.5
    out = _pseudo_targets_from_domain_scores(feats, masks, 0, aggregation="max")
    assert np.isclose(out[1], 0.5)


def test_score_input_dropout_p0_is_identity():
    feats, masks = _toy_scores()
    t = torch.tensor(feats)
    m = torch.tensor(masks)
    out = _dropout_score_input(t, m, 0, p=0.0)
    assert torch.equal(out, t)


def test_score_input_dropout_p1_neutralizes_present_scores_only():
    feats, masks = _toy_scores()
    masks[0, 0] = True  # one missing domain
    t = torch.tensor(feats)
    m = torch.tensor(masks)
    out = _dropout_score_input(t, m, 0, p=1.0)
    # Present-domain score columns become 0.5; missing domain untouched.
    assert out[0, 0, 0].item() == feats[0, 0, 0]  # masked -> unchanged
    assert out[0, 1, 0].item() == 0.5
    assert out[0, 2, 0].item() == 0.5
    # Non-score feature column (index 1) is never touched.
    assert torch.equal(out[:, :, 1], t[:, :, 1])


# --------------------------------------------------------------------------
# ISSUE 3: score_blend_alpha mixes the attention model back in
# --------------------------------------------------------------------------

def _fitted_model_and_estimator(seed=0):
    rng = np.random.default_rng(seed)
    n, d, f = 80, 3, 4
    features = rng.random((n, d, f)).astype(np.float32)
    labels = (rng.random(n) < 0.4).astype(np.float32)
    features[labels == 1, :, 0] = np.clip(features[labels == 1, :, 0] + 0.3, 0, 1)
    masks = np.zeros((n, d), dtype=bool)
    cfg = {
        "model": {"embed_dim": 8, "num_heads": 2, "num_layers": 1,
                  "use_confidence": False, "use_input_confidence": False},
    }
    device = torch.device("cpu")
    model = _build_model(cfg, d, f, None, device)
    model.eval()
    est = ReliabilityEstimator(
        domain_order=["d0", "d1", "d2"], score_index=0,
        ece_weight=0.4, ks_weight=0.4, sharpness_weight=0.2,
        n_calibration_bins=5, min_samples_for_ks=10,
    )
    est.fit(features, masks, labels)
    return model, est, features, masks, device


def test_score_blend_alpha_recorded_in_stats():
    model, est, features, masks, device = _fitted_model_and_estimator()
    _, stats = _predict_craf_with_stats(
        model, est, features, masks, device,
        clean_gate_threshold=2.0,  # force the gate to fire for all samples
        per_sample_gating=True, score_blend_on_gate=True, score_index=0,
        score_blend_alpha=0.5,
    )
    assert stats["score_blend_on_gate"] is True
    assert np.isclose(stats["score_blend_alpha"], 0.5)


def test_score_blend_alpha_changes_output():
    model, est, features, masks, device = _fitted_model_and_estimator()
    common = dict(
        clean_gate_threshold=2.0, per_sample_gating=True,
        score_blend_on_gate=True, score_index=0,
    )
    pure_blend, _ = _predict_craf_with_stats(model, est, features, masks, device, score_blend_alpha=1.0, **common)
    with_attn, _ = _predict_craf_with_stats(model, est, features, masks, device, score_blend_alpha=0.0, **common)
    # alpha=1.0 is the pure score blend; alpha=0.0 routes through the attention
    # fusion -> the two must differ, proving the model is no longer bypassed.
    assert not np.allclose(pure_blend, with_attn)
