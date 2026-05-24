"""Phase 2.B — selection-leakage gate for the prediction archive."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from elara.evaluation.prediction_archive import PredictionArchive  # noqa: E402


@pytest.fixture
def archive(tmp_path: Path) -> PredictionArchive:
    return PredictionArchive(root=tmp_path / "predictions")


def test_test_split_predictions_never_carry_test_selection_flag(archive):
    """Build a test-split archive and check that selection_used_test_metrics
    is False for every test row."""
    df = archive.build_frame(
        sample_ids=[f"t{i}" for i in range(10)],
        labels=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        raw_scores=np.linspace(0.1, 0.9, 10),
        method="rga_boosted_fusion",
        method_variant=None,
        benchmark="VisA",
        protocol="RGB+edge supervised-paired",
        analysis_family="A",
        pairing_strength="derived_view_proxy",
        split="test",
        seed=42,
        selection_rule="validation-only",
        selection_used_test_metrics=False,
        selected_head_or_comparator_status="validation-frozen RGA+ head (boost)",
    )
    entry = archive.write(
        experiment_id="A-POWERED-4",
        benchmark="VisA",
        protocol="RGB+edge supervised-paired",
        seed=42,
        method="rga_boosted_fusion",
        split="test",
        frame=df,
    )
    archive.append_index(entry)
    artifact = pd.read_parquet(entry.artifact_path) if entry.artifact_path.endswith(".parquet") else pd.read_csv(entry.artifact_path)
    assert (artifact["selection_used_test_metrics"].astype(str).str.lower().isin({"false", "0"})).all()
    assert entry.usable_for_inference is True


def test_validation_only_selection_verified_column_default_true(archive):
    df = archive.build_frame(
        sample_ids=["a", "b", "c"],
        labels=np.array([0, 1, 0]),
        raw_scores=np.array([0.1, 0.7, 0.3]),
        method="rga_meta_router",
        method_variant=None,
        benchmark="UNSW-NB15",
        protocol="flow/conn/context",
        analysis_family="A",
        pairing_strength="naturally_structured_views",
        split="validation",
        seed=42,
        selection_rule="validation-only",
        selection_used_test_metrics=False,
        selected_head_or_comparator_status="validation-frozen RGA+ head (router)",
    )
    entry = archive.write(
        experiment_id="A-POWERED-5",
        benchmark="UNSW-NB15",
        protocol="flow/conn/context",
        seed=42,
        method="rga_meta_router",
        split="validation",
        frame=df,
    )
    assert entry.validation_only_selection_verified is True
