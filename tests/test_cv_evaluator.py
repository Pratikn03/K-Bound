"""Tests for the unified 5-fold CV evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from uais.fusion.attention.baselines import (
    ConfidenceWeightedMean,
    RandomForestFusion,
)
from uais.fusion.attention.cv_evaluator import (
    BaselineSpec,
    CVConfig,
    cross_validate_baselines,
    pairwise_delong_from_predictions,
)
from uais.fusion.attention.unsupervised_baselines import (
    BGMMAnomalyDetector,
    BGMMConfig,
    IForestConfig,
    IsolationForestDetector,
)

N, D, F = 200, 3, 4


def _make_data(seed: int = 0, pos_rate: float = 0.3):
    rng = np.random.default_rng(seed)
    labels = (rng.random(N) < pos_rate).astype(int)
    features = rng.random((N, D, F)).astype(np.float32)
    for d in range(D):
        features[labels == 1, d, 0] = np.clip(features[labels == 1, d, 0] + 0.45, 0, 1)
    masks = (rng.random((N, D)) < 0.1).astype(bool)
    return features, masks, labels


def _basic_specs(seed: int = 42):
    return [
        BaselineSpec(
            name="confidence_weighted_mean",
            make=lambda: ConfidenceWeightedMean(),
            needs_3d=True,
            is_unsupervised=False,
        ),
        BaselineSpec(
            name="random_forest",
            make=lambda: RandomForestFusion(random_seed=seed),
            needs_3d=True,
            is_unsupervised=False,
        ),
        BaselineSpec(
            name="bgmm",
            make=lambda: BGMMAnomalyDetector(BGMMConfig(n_components=5, max_iter=50, random_state=seed)),
            needs_3d=True,
            is_unsupervised=True,
        ),
        BaselineSpec(
            name="isolation_forest",
            make=lambda: IsolationForestDetector(IForestConfig(n_estimators=50, random_state=seed)),
            needs_3d=True,
            is_unsupervised=True,
        ),
    ]


@pytest.fixture(scope="module")
def cv_results():
    features, masks, labels = _make_data()
    specs = _basic_specs()
    cfg = CVConfig(n_splits=3, bootstrap_resamples=100, random_state=42)
    return cross_validate_baselines(features, masks, labels, specs, cfg)


def test_cv_returns_results_for_every_spec(cv_results):
    expected = {"confidence_weighted_mean", "random_forest", "bgmm", "isolation_forest"}
    assert set(cv_results["per_baseline"].keys()) == expected


def test_cv_per_fold_metrics_have_required_keys(cv_results):
    for name, data in cv_results["per_baseline"].items():
        for fold in data["per_fold"]:
            if "error" in fold:
                continue
            for key in ("roc_auc", "pr_auc", "f1", "precision", "recall", "mcc"):
                assert key in fold, f"Missing {key} in {name} per-fold metrics"


def test_cv_aggregate_has_mean_std_ci(cv_results):
    for _name, data in cv_results["per_baseline"].items():
        agg = data["aggregate"]
        if "roc_auc" not in agg:
            continue
        roc = agg["roc_auc"]
        assert "mean" in roc and "std" in roc and "lo" in roc and "hi" in roc


def test_cv_bootstrap_ci_present(cv_results):
    for _name, data in cv_results["per_baseline"].items():
        if "bootstrap_ci" not in data:
            continue
        roc_ci = data["bootstrap_ci"].get("roc_auc")
        if roc_ci is None:
            continue
        assert "lo" in roc_ci and "hi" in roc_ci
        if np.isfinite(roc_ci["lo"]) and np.isfinite(roc_ci["hi"]):
            assert roc_ci["lo"] <= roc_ci["hi"]


def test_cv_contamination_check_clean(cv_results):
    """StratifiedKFold should produce zero-contamination splits."""
    contam = cv_results["contamination_check"]
    assert contam["max_duplicates"] == 0


def test_cv_test_oversampling_ok(cv_results):
    """No-duplicate-rows in test sets should always hold for clean data."""
    assert cv_results["test_oversampling_check"] == "ok"


def test_cv_hyperparameters_recorded(cv_results):
    """Every baseline must record its hyperparameter dict for reproducibility."""
    for _name, data in cv_results["per_baseline"].items():
        # Some may be empty dict for sklearn defaults, but key must exist
        assert "hyperparameters" in data


def test_pairwise_delong_returns_floats():
    """Manual DeLong call across two compatible prediction arrays."""
    rng = np.random.default_rng(7)
    y_true = (rng.random(80) < 0.4).astype(int)
    y_a = y_true + rng.normal(0, 0.2, 80)
    y_b = y_true + rng.normal(0, 0.3, 80)
    preds = {"a": (y_true, y_a), "b": (y_true, y_b)}
    result = pairwise_delong_from_predictions(preds)
    # If delong is available, expect one pair; otherwise empty
    if result:
        assert "a__vs__b" in result
        assert isinstance(result["a__vs__b"], float)
