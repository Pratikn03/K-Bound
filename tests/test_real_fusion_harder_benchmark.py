from __future__ import annotations

import numpy as np


def test_subsample_train_indices_keeps_classes_and_is_deterministic():
    from src.scripts.prepare_real_fusion_benchmark import _subsample_train_indices

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    train_idx = np.arange(len(y))

    first = _subsample_train_indices(train_idx, y, fraction=0.25, seed=7)
    second = _subsample_train_indices(train_idx, y, fraction=0.25, seed=7)

    np.testing.assert_array_equal(first, second)
    assert set(y[first]) == {0, 1}
    assert len(first) == 2


def test_subsample_train_indices_fraction_one_returns_all_indices():
    from src.scripts.prepare_real_fusion_benchmark import _subsample_train_indices

    y = np.array([0, 1, 0, 1])
    train_idx = np.array([3, 2, 1, 0])

    result = _subsample_train_indices(train_idx, y, fraction=1.0, seed=3)

    np.testing.assert_array_equal(result, train_idx)
