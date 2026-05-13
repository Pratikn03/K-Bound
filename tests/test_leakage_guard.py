"""Tests for leakage detection and protocol enforcement."""

from __future__ import annotations

import numpy as np
import pytest

from uais.fusion.attention.leakage_guard import (
    PipelineGuard,
    assert_no_oversampling_in_test,
    assert_normal_only_training,
    assert_split_before_preprocess,
    check_label_overlap,
    check_train_test_contamination,
    flag_suspicious_metrics,
)


def _data(seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((50, 10)).astype(np.float32)


def test_no_contamination_on_disjoint_data():
    a = _data(0)
    b = _data(1)
    r = check_train_test_contamination(a, b)
    assert r["n_duplicate_rows"] == 0
    assert r["fraction_test_in_train"] == 0.0


def test_contamination_detects_duplicates():
    a = _data(0)
    b = np.concatenate([a[:5], _data(2)[:5]], axis=0)
    r = check_train_test_contamination(a, b)
    assert r["n_duplicate_rows"] == 5


def test_label_overlap_detects_conflicts():
    a = _data(0)
    b = a[:3].copy()
    train_y = np.zeros(len(a), dtype=int)
    test_y = np.array([1, 0, 1])
    r = check_label_overlap(a, train_y, b, test_y)
    assert r["conflicting_labels"] == 2


def test_assert_no_oversampling_in_test_clean():
    X = _data()
    assert_no_oversampling_in_test(X)


def test_assert_no_oversampling_in_test_dupes():
    X = _data()
    X_dupe = np.concatenate([X, X[:5]], axis=0)
    with pytest.raises(AssertionError, match="duplicate rows"):
        assert_no_oversampling_in_test(X_dupe)


def test_assert_normal_only_training_clean():
    y = np.zeros(20, dtype=int)
    assert_normal_only_training(y)


def test_assert_normal_only_training_with_anomalies():
    y = np.array([0, 0, 1, 0, 1])
    with pytest.raises(AssertionError, match="positive labels"):
        assert_normal_only_training(y)


def test_flag_suspicious_metrics_perfect_auc():
    flags = flag_suspicious_metrics({"roc_auc": 1.0, "f1": 0.5})
    assert len(flags) == 1
    assert "roc_auc=1.0" in flags[0]


def test_flag_suspicious_metrics_realistic():
    flags = flag_suspicious_metrics({"roc_auc": 0.85, "f1": 0.7})
    assert flags == []


def test_pipeline_guard_passes_when_test_unchanged():
    train_X = _data(0)
    test_X = _data(1)
    guard = PipelineGuard()
    guard.record_split(train_X, test_X)
    # Modify train_X — should be allowed
    train_X = train_X * 2.0
    # Don't touch test_X
    guard.assert_test_unmodified(test_X)


def test_pipeline_guard_catches_test_mutation():
    train_X = _data(0)
    test_X = _data(1)
    guard = PipelineGuard()
    guard.record_split(train_X, test_X)
    # Pretend someone scaled test_X
    test_X_modified = test_X * 2.0
    with pytest.raises(AssertionError, match="modified after split"):
        guard.assert_test_unmodified(test_X_modified)


def test_context_manager_normal_path():
    train_X = _data(0)
    test_X = _data(1)
    with assert_split_before_preprocess(train_X, test_X) as guard:
        # Preprocess train only
        train_X_scaled = train_X * 10.0
        assert guard.n_unique_in_test() == 50


def test_context_manager_catches_violation():
    train_X = _data(0)
    test_X = _data(1)
    with pytest.raises(AssertionError, match="modified after split"):
        with assert_split_before_preprocess(train_X, test_X):
            test_X *= 10.0
