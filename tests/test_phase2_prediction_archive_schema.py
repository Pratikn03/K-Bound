"""Phase 2.B — schema tests for the prediction-archive contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from elara.evaluation.prediction_archive import (  # noqa: E402
    INDEX_COLUMNS,
    PREDICTION_ARCHIVE_SCHEMA,
    PredictionArchive,
)


@pytest.fixture
def archive(tmp_path: Path) -> PredictionArchive:
    return PredictionArchive(root=tmp_path / "predictions")


def _make_frame(archive: PredictionArchive, *, seed: int = 42, split: str = "test", n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return archive.build_frame(
        sample_ids=[f"s{i}" for i in range(n)],
        labels=(rng.random(n) < 0.3).astype(int),
        raw_scores=rng.random(n).astype(float),
        method="rga_meta_router",
        method_variant="ensemble-of-3",
        benchmark="MVTec 3D-AD",
        protocol="PatchCore supervised-paired",
        analysis_family="A",
        pairing_strength="independent_modalities",
        split=split,
        seed=seed,
        selection_rule="validation-only",
        selection_used_test_metrics=False,
        selected_head_or_comparator_status="validation-frozen RGA+ head (router)",
        gate_mode="mean",
        gate_fired=np.zeros(n, dtype=bool),
        mean_reliability=np.ones(n) * 0.7,
        min_reliability=np.ones(n) * 0.6,
        config_hash="testhash",
    )


def test_schema_has_required_columns(archive):
    df = _make_frame(archive)
    assert list(df.columns) == list(PREDICTION_ARCHIVE_SCHEMA)


def test_index_has_required_columns(archive):
    df = _make_frame(archive)
    entry = archive.write(
        experiment_id="A-POWERED-1",
        benchmark="MVTec 3D-AD",
        protocol="PatchCore supervised-paired",
        seed=42,
        method="rga_meta_router",
        split="test",
        frame=df,
        config={"toy": True},
    )
    archive.append_index(entry)
    loaded = archive.load_index()
    assert list(loaded.columns) == list(INDEX_COLUMNS)
    assert len(loaded) == 1


def test_write_rejects_test_set_selection_flag(archive):
    df = _make_frame(archive)
    with pytest.raises(ValueError):
        archive.write(
            experiment_id="A-POWERED-1",
            benchmark="MVTec 3D-AD",
            protocol="PatchCore supervised-paired",
            seed=42,
            method="rga_meta_router",
            split="test",
            frame=df,
            selection_used_test_metrics=True,
        )


def test_write_rejects_schema_mismatch(archive):
    df = _make_frame(archive)
    df = df.drop(columns=["raw_score"])
    with pytest.raises(ValueError):
        archive.write(
            experiment_id="A-POWERED-1",
            benchmark="MVTec 3D-AD",
            protocol="PatchCore supervised-paired",
            seed=42,
            method="rga_meta_router",
            split="test",
            frame=df,
        )


def test_immutability_appends_rerun_suffix(archive):
    df = _make_frame(archive)
    e1 = archive.write(
        experiment_id="A-POWERED-1",
        benchmark="MVTec 3D-AD",
        protocol="PatchCore supervised-paired",
        seed=42,
        method="rga_meta_router",
        split="test",
        frame=df,
    )
    e2 = archive.write(
        experiment_id="A-POWERED-1",
        benchmark="MVTec 3D-AD",
        protocol="PatchCore supervised-paired",
        seed=42,
        method="rga_meta_router",
        split="test",
        frame=df,
    )
    assert e1.artifact_path != e2.artifact_path
    assert "rerun_1" in e2.artifact_path
