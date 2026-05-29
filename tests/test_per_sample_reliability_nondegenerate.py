"""ISSUE 2 regression guard: per-sample reliability must not be degenerate.

The batch-level ``ReliabilityEstimator`` broadcasts one scalar per domain to
every row, so when all samples share the same present-mask the per-sample mean
reliability is identical across samples (within-batch std == 0) and
"per-sample gating" degenerates to all-or-nothing. The
``PerSampleReliabilityEstimator`` must instead produce reliability that varies
sample-by-sample (within-batch std > 0), which is what makes per-sample gating
and per-sample attention reweighting meaningful.
"""

from __future__ import annotations

import numpy as np

from uais.fusion.attention.reliability_estimator import (
    PerSampleReliabilityEstimator,
    ReliabilityEstimator,
)


def _synthetic_split():
    rng = np.random.default_rng(7)
    n, d, f = 400, 2, 3
    features = rng.random((n, d, f)).astype(np.float32)
    labels = (rng.random(n) < 0.4).astype(np.float32)
    # Make score column (index 0) carry signal so calibration/ECE are defined.
    features[labels == 1, :, 0] = np.clip(features[labels == 1, :, 0] + 0.3, 0, 1)
    masks = np.zeros((n, d), dtype=bool)  # all present -> isolates the estimator effect
    idx = np.arange(n)
    return features, masks, labels, idx[:200], idx[200:]


def _per_sample_mean_reliability(weights, masks):
    present = ~masks
    counts = present.sum(axis=1)
    sums = np.where(present, weights, 0.0).sum(axis=1)
    return sums / np.maximum(counts, 1)


def _common_kwargs():
    return dict(
        domain_order=["d0", "d1"],
        score_index=0,
        ece_weight=0.4,
        ks_weight=0.4,
        sharpness_weight=0.2,
        n_calibration_bins=5,
        min_samples_for_ks=10,
    )


def test_batch_estimator_is_degenerate_per_sample():
    features, masks, labels, val_idx, test_idx = _synthetic_split()
    est = ReliabilityEstimator(**_common_kwargs())
    est.fit(features[val_idx], masks[val_idx], labels[val_idx])
    weights = est.compute_reliability_weights(features[test_idx], masks[test_idx])
    r = _per_sample_mean_reliability(weights, masks[test_idx])
    # All rows share the same present-mask, so batch-level reliability is a
    # constant across samples -> per-sample gating cannot discriminate.
    assert float(np.std(r)) < 1e-6


def test_per_sample_estimator_varies_across_samples():
    features, masks, labels, val_idx, test_idx = _synthetic_split()
    est = PerSampleReliabilityEstimator(**_common_kwargs())
    est.fit(features[val_idx], masks[val_idx], labels[val_idx])
    weights = est.compute_reliability_weights(features[test_idx], masks[test_idx])
    r = _per_sample_mean_reliability(weights, masks[test_idx])
    # Genuine per-sample reliability -> meaningful dispersion across samples.
    assert float(np.std(r)) > 1e-3
    assert weights.shape == (len(test_idx), 2)
