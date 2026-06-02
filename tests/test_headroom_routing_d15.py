from __future__ import annotations

import numpy as np

from uais.fusion.attention.headroom_routing import (
    confidence_weighted_mean,
    headroom_routed_score,
    headroom_subset_masks,
    select_headroom_thresholds,
    stress_score_from_reliability,
)


def test_stress_score_increases_with_reliability_disagreement():
    clean = np.array([[0.95, 0.94, 0.93]])
    stressed = np.array([[0.95, 0.30, 0.92]])

    assert stress_score_from_reliability(stressed)[0] > stress_score_from_reliability(clean)[0]


def test_threshold_selection_is_validation_only():
    val_reliability = np.array(
        [
            [0.95, 0.94, 0.93],
            [0.90, 0.88, 0.89],
            [0.95, 0.50, 0.90],
            [0.96, 0.20, 0.90],
        ]
    )

    thresholds = select_headroom_thresholds(val_reliability, stress_quantile=0.75, clean_quantile=0.25)

    assert thresholds.selection_split == "validation"
    assert thresholds.used_test_labels is False
    assert thresholds.stress_threshold > thresholds.clean_threshold


def test_headroom_router_defaults_clean_and_routes_stress():
    scores = np.array(
        [
            [0.2, 0.4, 0.6],
            [0.2, 0.9, 0.6],
        ]
    )
    score_conf = np.ones_like(scores)
    reliability = np.array(
        [
            [0.95, 0.94, 0.93],
            [0.95, 0.10, 0.90],
        ]
    )
    routed, gate_active, stress = headroom_routed_score(
        scores,
        score_conf,
        reliability,
        stress_threshold=0.70,
    )
    cw = confidence_weighted_mean(scores, score_conf)

    assert gate_active.tolist() == [False, True]
    assert routed[0] == cw[0]
    assert routed[1] != cw[1]
    masks = headroom_subset_masks(stress, stress_threshold=0.70, clean_threshold=0.20)
    assert masks["stress"].tolist() == [False, True]
    assert masks["clean"].tolist() == [True, False]


def test_headroom_router_can_activate_at_validation_threshold_one():
    scores = np.array([[0.2, 0.8]])
    score_conf = np.ones_like(scores)
    reliability = np.array([[0.0, 0.0]])

    _, gate_active, stress = headroom_routed_score(
        scores,
        score_conf,
        reliability,
        stress_threshold=1.0,
    )

    assert stress.tolist() == [1.0]
    assert gate_active.tolist() == [True]


def test_confidence_weighted_mean_is_finite_with_bad_inputs():
    scores = np.array([[np.nan, np.inf, -np.inf]])
    out = confidence_weighted_mean(scores)

    assert np.isfinite(out).all()
    assert out.shape == (1,)
