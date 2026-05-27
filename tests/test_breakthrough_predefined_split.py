import numpy as np
import pandas as pd
import pytest


def test_split_uses_predefined_split_values():
    from scripts.run_breakthrough_experiment import _split

    labels = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    split_values = np.array(["train", "train", "validation", "validation", "test", "test"])

    train_idx, val_idx, test_idx = _split(
        labels,
        {
            "train_split_values": ["train"],
            "val_split_values": ["validation"],
            "test_split_values": ["test"],
        },
        split_values=split_values,
    )

    np.testing.assert_array_equal(train_idx, np.array([0, 1]))
    np.testing.assert_array_equal(val_idx, np.array([2, 3]))
    np.testing.assert_array_equal(test_idx, np.array([4, 5]))


def test_split_rejects_unassigned_predefined_values():
    from scripts.run_breakthrough_experiment import _split

    labels = np.array([0, 1, 0], dtype=float)
    split_values = np.array(["train", "validation", "holdout"])

    with pytest.raises(ValueError, match="not assigned"):
        _split(
            labels,
            {
                "train_split_values": ["train"],
                "val_split_values": ["validation"],
                "test_split_values": ["test"],
            },
            split_values=split_values,
        )


def test_sample_split_values_are_loaded_in_tensor_order(tmp_path):
    from scripts.run_breakthrough_experiment import _load_data

    path = tmp_path / "fusion.csv"
    rows = [
        {"sample_id": "b", "domain": "d1", "label": 1, "score": 0.9, "fusion_split": "test"},
        {"sample_id": "a", "domain": "d1", "label": 0, "score": 0.1, "fusion_split": "train"},
        {"sample_id": "c", "domain": "d1", "label": 0, "score": 0.2, "fusion_split": "validation"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)

    (
        _features,
        _masks,
        _labels,
        sample_ids,
        _domain_order,
        _feature_columns,
        _confidence_index,
        _score_index,
        sample_splits,
        _sample_categories,
    ) = _load_data(
        {
            "data": {
                "path": str(path),
                "id_column": "sample_id",
                "domain_column": "domain",
                "label_column": "label",
                "score_column": "score",
                "confidence_column": "confidence",
                "embedding_prefix": "embedding_",
            },
            "model": {"domain_order": ["d1"], "use_input_confidence": False},
            "training": {"split_column": "fusion_split"},
        }
    )

    assert sample_ids == ["a", "b", "c"]
    np.testing.assert_array_equal(sample_splits, np.array(["train", "test", "validation"]))


def test_sample_categories_are_loaded_when_configured(tmp_path):
    from scripts.run_breakthrough_experiment import _load_data

    path = tmp_path / "fusion.csv"
    rows = [
        {"sample_id": "b", "domain": "d1", "label": 1, "score": 0.9, "fusion_split": "test", "category": "cookie"},
        {"sample_id": "a", "domain": "d1", "label": 0, "score": 0.1, "fusion_split": "train", "category": "bagel"},
        {
            "sample_id": "c",
            "domain": "d1",
            "label": 0,
            "score": 0.2,
            "fusion_split": "validation",
            "category": "cable_gland",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)

    (
        _features,
        _masks,
        _labels,
        sample_ids,
        _domain_order,
        _feature_columns,
        _confidence_index,
        _score_index,
        _sample_splits,
        sample_categories,
    ) = _load_data(
        {
            "data": {
                "path": str(path),
                "id_column": "sample_id",
                "domain_column": "domain",
                "label_column": "label",
                "score_column": "score",
                "confidence_column": "confidence",
                "embedding_prefix": "embedding_",
                "category_column": "category",
            },
            "model": {"domain_order": ["d1"], "use_input_confidence": False},
            "training": {"split_column": "fusion_split"},
        }
    )

    assert sample_ids == ["a", "b", "c"]
    np.testing.assert_array_equal(sample_categories, np.array(["bagel", "cookie", "cable_gland"]))


def test_aggregate_category_aware_reports_misfire_delta():
    from scripts.run_breakthrough_experiment import _aggregate_category_aware

    rows = [
        {
            "global_adapt_rate": 0.80,
            "category_aware_adapt_rate": 0.40,
            "global_mean_reliability": 0.40,
            "category_aware_mean_reliability": 0.70,
        },
        {
            "global_adapt_rate": 0.82,
            "category_aware_adapt_rate": 0.38,
            "global_mean_reliability": 0.41,
            "category_aware_mean_reliability": 0.71,
        },
        {
            "global_adapt_rate": 0.78,
            "category_aware_adapt_rate": 0.42,
            "global_mean_reliability": 0.39,
            "category_aware_mean_reliability": 0.69,
        },
    ]
    summary = _aggregate_category_aware(rows)

    assert summary["n_seeds"] == 3
    assert summary["global_adapt_rate"] == pytest.approx(0.80, abs=1e-6)
    assert summary["category_aware_adapt_rate"] == pytest.approx(0.40, abs=1e-6)
    assert summary["misfire_reduction_absolute"] == pytest.approx(0.40, abs=1e-6)
    assert summary["category_aware_mean_reliability"] > summary["global_mean_reliability"]
