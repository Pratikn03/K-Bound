"""Tests for the cross-modal patch-interaction detector (Level-4 lever).

The defining property of feature-level (vs score-level) fusion: it detects a
defect that is *marginally normal in each modality but jointly anomalous*. These
tests pin that behavior on synthetic patch features (no backbone needed) and the
alignment contract.
"""

from __future__ import annotations

import numpy as np

from uais.fusion.attention.cross_modal_patchcore import (
    combine_cross_modal_patches,
    score_cross_modal_from_features,
)

N_PATCHES = 4
C = 4
A = np.array([0.0, 0.0, 0.0, 0.0])
B = np.array([5.0, 5.0, 5.0, 5.0])


def _img_patches(anchor, n_img, seed):
    rng = np.random.default_rng(seed)
    return (anchor + rng.normal(0, 0.05, size=(n_img * N_PATCHES, C))).astype(np.float32)


def test_combine_shapes_and_alignment():
    rgb = np.zeros((3 * N_PATCHES, 2))
    depth = np.ones((3 * N_PATCHES, 5))
    joint = combine_cross_modal_patches([rgb, depth], N_PATCHES)
    assert joint.shape == (3 * N_PATCHES, 7)          # C concatenated
    # first cols from rgb (0), last from depth (1)
    assert (joint[:, :2] == 0).all() and (joint[:, 2:] == 1).all()


def test_misaligned_grids_raise():
    try:
        combine_cross_modal_patches([np.zeros((5, 2)), np.zeros((5, 2))], N_PATCHES)
    except ValueError:
        return
    raise AssertionError("expected ValueError on rows not divisible by n_patches")


def test_cross_modal_anomaly_detected():
    # Train normal joint distribution = {(A,A), (B,B)} (rgb and depth co-vary).
    rgb_train = np.concatenate([_img_patches(A, 10, 1), _img_patches(B, 10, 2)])
    depth_train = np.concatenate([_img_patches(A, 10, 3), _img_patches(B, 10, 4)])

    # Normal eval = (A,A); cross-modal anomaly = (A,B): each marginal is a normal
    # value, but the JOINT (A,B) was never seen in training.
    rgb_eval = np.concatenate([_img_patches(A, 5, 11), _img_patches(A, 5, 13)])
    depth_eval = np.concatenate([_img_patches(A, 5, 12), _img_patches(B, 5, 14)])

    scores = score_cross_modal_from_features(
        [rgb_train, depth_train], [rgb_eval, depth_eval], N_PATCHES, coreset_size=10_000)
    normal_scores, anomaly_scores = scores[:5], scores[5:]
    assert anomaly_scores.min() > normal_scores.max()  # unambiguous separation


def test_marginals_alone_would_miss_it():
    # Sanity: the anomaly's rgb marginal (A) and depth marginal (B) are each
    # individually present in training, so a per-modality detector sees nothing.
    rgb_train = np.concatenate([_img_patches(A, 10, 1), _img_patches(B, 10, 2)])
    rgb_anom = _img_patches(A, 5, 13)            # rgb = A -> normal marginally
    # nearest rgb-only distance from anomaly to train should be ~0 (A is in train)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(rgb_train).query(rgb_anom, k=1)
    assert d.max() < 0.5                          # marginally normal in RGB
