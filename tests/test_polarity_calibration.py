"""Tests for the post-hoc polarity-calibration helper in the breakthrough runner."""

from __future__ import annotations

import numpy as np
import torch

from scripts.run_breakthrough_experiment import _calibrate_polarity


class _InversePolarityModel(torch.nn.Module):
    """Toy model that outputs HIGH probability when input score is LOW.

    Mimics the pathology observed under canonical one-class training:
    the supervised head learned an inverse-polarity solution.
    """

    domain_encoders = torch.nn.ModuleList()
    fusion = None  # populated below for runtime checks

    def __init__(self, score_index: int = 0):
        super().__init__()
        self.score_index = int(score_index)

    def forward(self, feat_t, key_padding_mask=None):
        # Negative scores -> positive logits (inverse polarity).
        if feat_t.ndim == 3:
            avg_score = feat_t[:, :, self.score_index].mean(dim=1)
        else:
            avg_score = feat_t[:, self.score_index]
        logits = -10.0 * (avg_score - 0.5)
        # Returned tuple matches the existing _predict_static interface.
        logits = logits.view(-1, 1)
        attn_dummy = torch.zeros(feat_t.shape[0], 1)
        conf_dummy = torch.zeros(feat_t.shape[0], 1)
        return logits, attn_dummy, conf_dummy


class _CorrectPolarityModel(torch.nn.Module):
    """Toy model that outputs HIGH probability when input score is HIGH."""

    domain_encoders = torch.nn.ModuleList()
    fusion = None

    def __init__(self, score_index: int = 0):
        super().__init__()
        self.score_index = int(score_index)

    def forward(self, feat_t, key_padding_mask=None):
        if feat_t.ndim == 3:
            avg_score = feat_t[:, :, self.score_index].mean(dim=1)
        else:
            avg_score = feat_t[:, self.score_index]
        logits = 10.0 * (avg_score - 0.5)
        logits = logits.view(-1, 1)
        attn_dummy = torch.zeros(feat_t.shape[0], 1)
        conf_dummy = torch.zeros(feat_t.shape[0], 1)
        return logits, attn_dummy, conf_dummy


def _make_val(n: int = 60, n_domains: int = 2, n_features: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.uniform(0.0, 1.0, size=(n, n_domains, n_features)).astype(np.float32)
    # Real val under canonical one-class is all-normal (label=0).
    labels = np.zeros(n, dtype=int)
    masks = np.zeros((n, n_domains), dtype=bool)
    return features, masks, labels


def test_polarity_calibration_flags_inverse_model():
    features, masks, labels = _make_val(seed=1)
    model = _InversePolarityModel(score_index=0)
    info = _calibrate_polarity(
        model,
        features,
        masks,
        labels,
        score_index=0,
        device=torch.device("cpu"),
        random_seed=1,
    )
    assert info["flip_required"] is True
    assert info["calibration_auroc"] < 0.5
    assert info["n_synthetic"] > 0
    assert info["n_calibration"] > info["n_synthetic"]


def test_polarity_calibration_does_not_flag_correct_model():
    features, masks, labels = _make_val(seed=2)
    model = _CorrectPolarityModel(score_index=0)
    info = _calibrate_polarity(
        model,
        features,
        masks,
        labels,
        score_index=0,
        device=torch.device("cpu"),
        random_seed=2,
    )
    assert info["flip_required"] is False
    assert info["calibration_auroc"] >= 0.5


def test_polarity_calibration_handles_tiny_val():
    features = np.zeros((2, 2, 4), dtype=np.float32)
    masks = np.zeros((2, 2), dtype=bool)
    labels = np.zeros(2, dtype=int)
    info = _calibrate_polarity(
        _CorrectPolarityModel(),
        features,
        masks,
        labels,
        score_index=0,
        device=torch.device("cpu"),
        random_seed=0,
    )
    assert info["flip_required"] is False
    assert info["n_calibration"] >= 2
