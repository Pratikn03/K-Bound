"""Data-leakage detection and protocol enforcement.

Addresses reviewer comments:
  - "Unrealistic perfect performance (AUC = 1.000) ... strongly suggests
     methodological issues such as data leakage, overfitting, or memorization."
  - "Data leakage due to oversampling ... performed before the train/test split,
     which can cause duplicated samples to appear in the test set."
  - "Data splitting must occur before any preprocessing."

Public API
----------
check_train_test_contamination(train_X, test_X)
    Hash-based duplicate detection between train and test rows.
    Returns (n_duplicates, fraction).

check_label_overlap(train_y, test_y, train_X, test_X)
    Detect rows where features match exactly across train/test but with
    different labels — sign of corrupted preprocessing or label noise.

assert_no_oversampling_in_test(test_X, test_y)
    Verify no duplicate rows in the test set (oversampling should never
    touch test data).

assert_normal_only_training(train_y)
    Verify training labels for unsupervised models are all 0.

flag_suspicious_metrics(metrics_dict, auc_thr=0.99)
    Emit a list of warnings for metrics above plausible thresholds.

PipelineGuard
    Context manager that records pre-split hashes and asserts no test row
    was modified after split time.
"""

from __future__ import annotations

import hashlib
import warnings
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def _row_hashes(X: np.ndarray) -> np.ndarray:
    """Return a [N] array of stable per-row hashes (8-byte hex strings)."""
    X = np.ascontiguousarray(X)
    return np.array(
        [hashlib.blake2b(row.tobytes(), digest_size=8).hexdigest()
         for row in X.reshape(X.shape[0], -1)],
        dtype=object,
    )


# ---------------------------------------------------------------------------
# Contamination checks
# ---------------------------------------------------------------------------

def check_train_test_contamination(
    train_X: np.ndarray, test_X: np.ndarray
) -> Dict[str, float]:
    """Detect rows that appear in BOTH train and test (verbatim duplicates)."""
    th = set(_row_hashes(train_X).tolist())
    test_hashes = _row_hashes(test_X)
    dupes = sum(1 for h in test_hashes if h in th)
    return {
        "n_train": int(train_X.shape[0]),
        "n_test": int(test_X.shape[0]),
        "n_duplicate_rows": int(dupes),
        "fraction_test_in_train": float(dupes / max(test_X.shape[0], 1)),
    }


def check_label_overlap(
    train_X: np.ndarray, train_y: np.ndarray,
    test_X: np.ndarray, test_y: np.ndarray,
) -> Dict[str, int]:
    """Find samples whose features match across splits but with differing labels.

    A non-zero count indicates corrupted preprocessing or label noise.
    """
    train_y = np.asarray(train_y)
    test_y = np.asarray(test_y)
    h_train = _row_hashes(train_X)
    h_test = _row_hashes(test_X)
    train_map: Dict[str, int] = {h: int(y) for h, y in zip(h_train, train_y)}
    conflicts = 0
    for h, y in zip(h_test, test_y):
        if h in train_map and train_map[h] != int(y):
            conflicts += 1
    return {"conflicting_labels": int(conflicts)}


def assert_no_oversampling_in_test(test_X: np.ndarray) -> None:
    """Raise if the test split contains duplicate rows (sign of resampling)."""
    th = _row_hashes(test_X)
    _, counts = np.unique(th, return_counts=True)
    dup_count = int((counts > 1).sum())
    if dup_count > 0:
        raise AssertionError(
            f"Test set contains {dup_count} duplicate rows. "
            "Oversampling must only be applied to training data — "
            "re-split before resampling."
        )


def assert_normal_only_training(train_y: np.ndarray) -> None:
    """Enforce the unsupervised-model protocol: train labels must all be 0."""
    train_y = np.asarray(train_y)
    if (train_y != 0).any():
        n_pos = int((train_y == 1).sum())
        raise AssertionError(
            f"Unsupervised anomaly model received {n_pos} positive labels "
            f"during fit().  Train only on normal (label==0) samples; reserve "
            f"anomalies for testing."
        )


# ---------------------------------------------------------------------------
# Result-level warnings
# ---------------------------------------------------------------------------

def flag_suspicious_metrics(
    metrics_dict: Dict[str, float],
    auc_threshold: float = 0.99,
    f1_threshold: float = 0.99,
) -> List[str]:
    """Return a list of warning strings if any metric looks unrealistically high.

    Addresses reviewer concern: 'A perfect AUC strongly suggests ... data
    leakage, overfitting, or memorization.'
    """
    warnings_list: List[str] = []
    for key, thr in (("roc_auc", auc_threshold), ("pr_auc", auc_threshold),
                      ("f1", f1_threshold), ("accuracy", f1_threshold)):
        v = metrics_dict.get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v >= thr:
            warnings_list.append(
                f"{key}={v:.4f} ≥ {thr}: investigate possible leakage, "
                f"overfitting, or duplicate samples between train/test."
            )
    return warnings_list


# ---------------------------------------------------------------------------
# PipelineGuard — assert split happens before preprocessing
# ---------------------------------------------------------------------------

class PipelineGuard:
    """Records pre-split hashes and verifies the test set is never preprocessed.

    Usage
    -----
    >>> guard = PipelineGuard()
    >>> guard.record_split(train_X, test_X)
    >>> # ... preprocess train_X however you like ...
    >>> # ... do NOT touch test_X ...
    >>> guard.assert_test_unmodified(test_X)
    """

    def __init__(self) -> None:
        self._train_hashes_at_split: List[str] = []
        self._test_hashes_at_split: List[str] = []

    def record_split(self, train_X: np.ndarray, test_X: np.ndarray) -> None:
        self._train_hashes_at_split = _row_hashes(train_X).tolist()
        self._test_hashes_at_split = _row_hashes(test_X).tolist()

    def assert_test_unmodified(self, test_X: np.ndarray) -> None:
        cur = _row_hashes(test_X).tolist()
        if cur != self._test_hashes_at_split:
            n_diff = sum(1 for a, b in zip(cur, self._test_hashes_at_split) if a != b)
            raise AssertionError(
                f"PipelineGuard: test set was modified after split "
                f"({n_diff} rows changed). Do not preprocess test data."
            )

    def n_unique_in_test(self) -> int:
        return len(set(self._test_hashes_at_split))


@contextmanager
def assert_split_before_preprocess(train_X: np.ndarray, test_X: np.ndarray):
    """Context manager wrapper around PipelineGuard for cleaner usage."""
    guard = PipelineGuard()
    guard.record_split(train_X, test_X)
    try:
        yield guard
    finally:
        guard.assert_test_unmodified(test_X)


__all__ = [
    "check_train_test_contamination",
    "check_label_overlap",
    "assert_no_oversampling_in_test",
    "assert_normal_only_training",
    "flag_suspicious_metrics",
    "PipelineGuard",
    "assert_split_before_preprocess",
]
