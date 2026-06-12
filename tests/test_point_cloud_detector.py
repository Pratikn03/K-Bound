"""Tests for the rotation-invariant point-cloud detector (D18 XYZ fix).

The PCA relief projection (pcd_to_geometry_image) inverts on thin/symmetric
objects because PCA eigenvector signs are arbitrary. The covariance-feature
detector is rotation- and sign-invariant by construction. These tests pin:
  - it flags a geometric bump as more anomalous than a flat surface,
  - its features are invariant to rotation of the cloud,
  - it degrades gracefully on degenerate input.
"""

from __future__ import annotations

import numpy as np

from uais.fusion.attention.realiad_3d_detector import (
    point_cloud_covariance_features,
    score_one_class_point_cloud,
)


def _flat_plane(n: int = 3000, noise: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1, 1, size=(n, 2))
    z = rng.normal(0, noise, size=n)
    return np.column_stack([xy, z]).astype(np.float32)


def _bumped_plane(n: int = 3000, height: float = 0.4, seed: int = 0) -> np.ndarray:
    p = _flat_plane(n, seed=seed)
    r2 = (p[:, 0] ** 2 + p[:, 1] ** 2)
    p[:, 2] += height * np.exp(-r2 / 0.02)  # localized Gaussian bump = the defect
    return p


def _random_rotation(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_features_shape_and_none_guard():
    feats = point_cloud_covariance_features(_flat_plane(), scales=(8, 16))
    assert feats is not None
    assert feats.shape[1] == 8 * 2
    assert point_cloud_covariance_features(np.zeros((10, 3))) is None  # too small


def test_rotation_invariance():
    cloud = _bumped_plane(seed=1)
    f0 = point_cloud_covariance_features(cloud, n_sample=3000, seed=3)
    f1 = point_cloud_covariance_features(cloud @ _random_rotation(2).T, n_sample=3000, seed=3)
    # Eigenvalue-based features + k-NN distances are invariant under rotation;
    # same seed -> same subsample -> features match to numerical tolerance.
    assert np.allclose(f0, f1, atol=1e-4)


def test_bump_scores_higher_than_flat():
    train = [_flat_plane(seed=s) for s in range(8)]          # all-normal bank
    flats = [_flat_plane(seed=100 + s) for s in range(5)]
    bumps = [_bumped_plane(seed=200 + s) for s in range(5)]
    scores = score_one_class_point_cloud(train, flats + bumps, scales=(8, 16, 32),
                                         n_sample=3000)
    flat_scores, bump_scores = scores[:5], scores[5:]
    assert bump_scores.mean() > flat_scores.mean()
    # And the separation should be unambiguous (every bump above every flat).
    assert bump_scores.min() > flat_scores.max()


def test_score_handles_empty_bank():
    scores = score_one_class_point_cloud([], [_flat_plane()], scales=(8,))
    assert scores.shape == (1,)
