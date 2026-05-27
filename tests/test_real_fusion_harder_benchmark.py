from __future__ import annotations

import numpy as np
import pandas as pd


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


def _domain_frame(n_per_label: int = 30) -> pd.DataFrame:
    rows = []
    for label in [0, 1]:
        for idx in range(n_per_label):
            source_id = label * 1000 + idx
            row = {
                "domain_source_id": source_id,
                "label": label,
                "score": 0.2 if label == 0 else 0.8,
                "confidence": 0.6,
            }
            for emb_idx in range(8):
                row[f"embedding_{emb_idx}"] = float(idx + emb_idx) / 100.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_split_safe_fusion_rows_keeps_source_rows_in_one_split():
    from src.scripts.prepare_real_fusion_benchmark import DOMAIN_ORDER, _build_split_safe_fusion_rows

    domain_frames = {domain: _domain_frame() for domain in DOMAIN_ORDER}

    fusion_df, split_counts = _build_split_safe_fusion_rows(
        domain_frames,
        n_samples=90,
        positive_fraction=0.5,
        missing_probability=0.0,
        seed=11,
        split_fractions={"train": 0.6, "validation": 0.2, "test": 0.2},
    )

    assert set(fusion_df["fusion_split"]) == {"train", "validation", "test"}
    assert fusion_df.groupby("sample_id")["fusion_split"].nunique().eq(1).all()
    assert set(split_counts) == {"train", "validation", "test"}

    key_splits = fusion_df.groupby(["domain", "source_row", "label"])["fusion_split"].nunique()
    assert key_splits.max() == 1
