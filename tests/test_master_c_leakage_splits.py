"""Master Scenario C — split integrity and leakage guards on fusion CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uais.fusion.attention.leakage_guard import check_train_test_contamination


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "csv_rel,split_col",
    [
        ("experiments/fusion/real_domain_fusion_inputs.csv", "fusion_split"),
        ("experiments/fusion/mvtec3d_patchcore_inputs.csv", "split"),
    ],
)
def test_no_sample_id_leakage_between_train_and_test(csv_rel: str, split_col: str):
    root = _root()
    path = root / csv_rel
    if not path.is_file():
        pytest.skip(f"missing {csv_rel}")
    df = pd.read_csv(path, usecols=["sample_id", split_col])
    train_ids = set(df[df[split_col] == "train"]["sample_id"].astype(str))
    test_ids = set(df[df[split_col] == "test"]["sample_id"].astype(str))
    overlap = train_ids & test_ids
    assert not overlap, f"{len(overlap)} sample_ids in both train and test"


def test_split_hash_manifest_exists():
    root = _root()
    manifest = root / "elara_master_c/data/splits/split_hashes/manifest.json"
    if not manifest.is_file():
        pytest.skip("run generate_split_hashes.py first")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(data.get("datasets", {})) >= 2


def test_disjoint_feature_rows_no_duplicates():
    root = _root()
    path = root / "experiments/fusion/mvtec3d_patchcore_inputs.csv"
    if not path.is_file():
        pytest.skip("missing mvtec csv")
    df = pd.read_csv(path)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    # Long format: check embedding columns if present
    emb_cols = [c for c in df.columns if c.startswith("embedding_")]
    if not emb_cols:
        pytest.skip("no embeddings")
    a = train[emb_cols].to_numpy(dtype=np.float32)
    b = test[emb_cols].to_numpy(dtype=np.float32)
    r = check_train_test_contamination(a[: min(500, len(a))], b[: min(500, len(b))])
    assert r["fraction_test_in_train"] < 0.05
