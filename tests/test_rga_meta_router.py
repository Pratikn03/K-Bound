from __future__ import annotations

import numpy as np

from uais.fusion.attention.meta_router import fit_rga_meta_router


def test_rga_meta_router_learns_from_validation_predictions():
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    val_predictions = {
        "craf_attention": np.array([0.20, 0.30, 0.70, 0.80, 0.25, 0.75]),
        "static_attention": np.array([0.40, 0.45, 0.55, 0.60, 0.50, 0.52]),
        "tent_score_adapter": np.array([0.10, 0.35, 0.65, 0.90, 0.30, 0.70]),
    }
    test_predictions = {
        "craf_attention": np.array([0.15, 0.85]),
        "static_attention": np.array([0.45, 0.55]),
        "tent_score_adapter": np.array([0.20, 0.80]),
    }

    router = fit_rga_meta_router(val_predictions, labels, random_seed=0)
    probs = router.predict_proba(test_predictions)

    assert probs.shape == (2,)
    assert np.all((0.0 <= probs) & (probs <= 1.0))
    assert probs[1] > probs[0]
    assert router.selected_candidate in router.candidate_scores


def test_rga_meta_router_falls_back_on_single_class_validation():
    labels = np.zeros(4, dtype=np.int64)
    val_predictions = {
        "craf_attention": np.array([0.2, 0.3, 0.4, 0.5]),
        "static_attention": np.array([0.1, 0.2, 0.3, 0.4]),
    }

    router = fit_rga_meta_router(val_predictions, labels, random_seed=0)
    probs = router.predict_proba(
        {
            "craf_attention": np.array([0.7, 0.8]),
            "static_attention": np.array([0.6, 0.9]),
        }
    )

    assert probs.shape == (2,)
    assert router.selected_candidate == "base:craf_attention"


def test_rga_meta_router_preserves_tiny_probability_ranking():
    labels = np.zeros(4, dtype=np.int64)
    val_predictions = {
        "craf_attention": np.array([1e-8, 2e-8, 3e-8, 4e-8]),
        "static_attention": np.array([0.1, 0.2, 0.3, 0.4]),
    }

    router = fit_rga_meta_router(val_predictions, labels, random_seed=0)
    probs = router.predict_proba(
        {
            "craf_attention": np.array([4e-8, 8e-8]),
            "static_attention": np.array([0.6, 0.9]),
        }
    )

    assert probs[1] > probs[0]


def test_rga_meta_router_honors_selection_metric():
    labels = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    val_predictions = {
        "craf_attention": np.array([0.2, 0.3, 0.4, 0.9, 0.8, 0.7]),
        "rga_boosted_fusion": np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05]),
    }

    router = fit_rga_meta_router(val_predictions, labels, random_seed=0, selection_metric="pr_auc")

    assert router.selected_candidate == "base:rga_boosted_fusion"
    assert router.candidate_metric_scores["base:rga_boosted_fusion"]["pr_auc"] == 1.0
