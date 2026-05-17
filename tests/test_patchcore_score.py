import numpy as np

from uais.fusion.attention.m3dm_features import (
    normal_reference_distance_score,
    patchcore_knn_score,
)


def test_patchcore_separates_unimodal_anomalies_from_normals():
    rng = np.random.default_rng(0)
    normal_features = rng.normal(0.0, 0.05, size=(80, 16)).astype(np.float32)
    anomaly_features = rng.normal(5.0, 0.05, size=(20, 16)).astype(np.float32)
    features = np.vstack([normal_features, anomaly_features])
    fit_mask = np.zeros(features.shape[0], dtype=bool)
    fit_mask[:80] = True

    scores = patchcore_knn_score(features, fit_mask, k=3)
    assert scores.shape == (100,)
    # Median of normals must sit well below the gate; all anomalies must clip to 1.
    assert float(np.median(scores[:80])) < 0.5
    assert float(np.min(scores[80:])) > 0.9


def test_patchcore_separates_multimodal_anomalies_better_than_mahalanobis():
    """When normals form two clusters, the single-centroid Mahalanobis variant
    gives anomalies that sit between the clusters a low score; the kNN-to-bank
    variant correctly flags them as far from any normal mode."""
    rng = np.random.default_rng(1)
    cluster_a = rng.normal(loc=-3.0, scale=0.1, size=(60, 8)).astype(np.float32)
    cluster_b = rng.normal(loc=+3.0, scale=0.1, size=(60, 8)).astype(np.float32)
    in_between_anomalies = rng.normal(loc=0.0, scale=0.1, size=(10, 8)).astype(np.float32)

    features = np.vstack([cluster_a, cluster_b, in_between_anomalies])
    fit_mask = np.zeros(features.shape[0], dtype=bool)
    fit_mask[:120] = True

    mahalanobis_scores = normal_reference_distance_score(features, fit_mask)
    knn_scores = patchcore_knn_score(features, fit_mask, k=3)

    # The between-cluster anomalies are far from both normal modes but close to
    # the global centroid. Mahalanobis must rank them low; kNN must rank them high.
    anomaly_knn = float(np.mean(knn_scores[120:]))
    anomaly_mahalanobis = float(np.mean(mahalanobis_scores[120:]))
    normal_knn = float(np.mean(knn_scores[:120]))
    assert anomaly_knn > anomaly_mahalanobis
    assert anomaly_knn - normal_knn > 0.4


def test_patchcore_handles_empty_fit_mask():
    features = np.zeros((10, 4), dtype=np.float32)
    fit_mask = np.zeros(10, dtype=bool)
    scores = patchcore_knn_score(features, fit_mask, k=3)
    np.testing.assert_array_equal(scores, np.zeros(10, dtype=np.float32))


def test_patchcore_coreset_subsamples_bank():
    rng = np.random.default_rng(2)
    normal_features = rng.normal(0.0, 1.0, size=(500, 32)).astype(np.float32)
    anomaly_features = rng.normal(10.0, 1.0, size=(20, 32)).astype(np.float32)
    features = np.vstack([normal_features, anomaly_features])
    fit_mask = np.zeros(features.shape[0], dtype=bool)
    fit_mask[:500] = True

    full = patchcore_knn_score(features, fit_mask, k=5)
    coreset = patchcore_knn_score(features, fit_mask, k=5, coreset_size=50, random_state=0)

    assert coreset.shape == full.shape
    # Both must still produce higher anomaly scores than normal scores.
    assert float(np.mean(coreset[500:])) > float(np.mean(coreset[:500])) + 0.2
