from __future__ import annotations

import numpy as np

from uais.fusion.attention.positive_transfer import (
    candidate_scores,
    coendpoint_pass,
    paired_auc_bootstrap,
    select_candidate_on_validation,
)


def test_candidate_scores_are_finite_and_shape_stable():
    rgb = np.array([0.0, 0.2, 0.8, 1.0])
    depth = np.array([1.0, 0.8, 0.2, 0.0])
    sar = np.array([0.1, 0.3, 0.7, 0.9])

    scores = candidate_scores(rgb, depth, sar=sar)

    assert set(scores) >= {"cw", "rank_cw", "product", "max", "softor"}
    for value in scores.values():
        assert np.isfinite(value).all()
        assert value.shape == rgb.shape


def test_selector_uses_validation_only_and_records_no_test_metrics():
    val_y = np.array([0, 0, 1, 1])
    val_rgb = np.array([0.1, 0.2, 0.8, 0.9])
    val_depth = np.array([0.1, 0.2, 0.8, 0.9])

    result = select_candidate_on_validation(val_y, val_rgb, val_depth)

    assert result.selected_rule in {"cw", "product", "rank_cw", "softor", "max"}
    assert result.used_test_labels is False
    assert result.selection_split == "validation"


def test_bootstrap_ci_crossing_zero_does_not_pass():
    y = np.array([0, 0, 1, 1])
    same = np.array([0.1, 0.2, 0.8, 0.9])
    stat = paired_auc_bootstrap(y, same, same, n_iter=300, seed=0)

    assert stat["delta"] == 0.0
    assert coendpoint_pass(
        stat,
        minimum_practical_delta=0.001,
        ci_low_must_be_gt=0.0,
    ) is False


def test_negative_delta_never_passes():
    y = np.array([0, 0, 0, 1, 1, 1])
    bad = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    good = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    stat = paired_auc_bootstrap(y, bad, good, n_iter=300, seed=0)

    assert stat["delta"] < 0
    assert coendpoint_pass(
        stat,
        minimum_practical_delta=0.001,
        ci_low_must_be_gt=0.0,
    ) is False


def test_both_sar_and_cw_endpoints_are_required():
    sar_stat = {"delta": 0.02, "ci95": [0.01, 0.03], "valid": True}
    cw_stat = {"delta": 0.0, "ci95": [-0.01, 0.01], "valid": True}

    assert coendpoint_pass(
        sar_stat,
        minimum_practical_delta=0.010,
        ci_low_must_be_gt=0.0,
    ) is True
    assert coendpoint_pass(
        cw_stat,
        minimum_practical_delta=0.005,
        ci_low_must_be_gt=0.0,
    ) is False

