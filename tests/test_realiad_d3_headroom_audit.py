from __future__ import annotations

from pathlib import Path

import numpy as np

from src.scripts.scenario_c.run_realiad_d3_headroom_audit import (
    _artifact_reference,
    _current_dataset_status,
    _d16_category_integrity,
    _select_stress_method,
)


def test_artifact_reference_accepts_relative_repo_paths() -> None:
    assert _artifact_reference(Path("experiments/fusion/example_predictions.csv")) == (
        "experiments/fusion/example_predictions.csv"
    )


def test_select_stress_method_uses_validation_auc_argmax() -> None:
    labels = np.array([0] * 10 + [1] * 10)
    stress = np.ones(20, dtype=bool)
    candidates = {
        "confidence_weighted_mean": np.array([0.3] * 10 + [0.2] * 10),
        "max_score": np.array([0.1] * 10 + [0.9] * 10),
        "score_disagreement": np.zeros(20),
    }

    selected = _select_stress_method(labels, candidates, stress)

    assert selected["selected_method"] == "max_score"
    assert selected["used_split"] == "validation"
    assert selected["used_test_labels"] is False
    assert selected["validation_stress_delta_vs_cw"] == 1.0


def test_select_stress_method_falls_back_when_validation_stress_degenerate() -> None:
    labels = np.array([0] * 20)
    stress = np.ones(20, dtype=bool)
    candidates = {
        "confidence_weighted_mean": np.linspace(0.0, 1.0, 20),
        "max_score": np.linspace(1.0, 0.0, 20),
    }

    selected = _select_stress_method(labels, candidates, stress)

    assert selected["selected_method"] == "confidence_weighted_mean"
    assert selected["reason"] == "degenerate_validation_stress_subset"


def test_d16_category_integrity_blocks_opened_smoke_category() -> None:
    blocked = _d16_category_integrity(["common_mode_filter", "audio_jack_socket"])
    clean = _d16_category_integrity(["audio_jack_socket"])

    assert blocked["confirmatory_category_valid"] is False
    assert blocked["opened_categories_present"] == ["common_mode_filter"]
    assert clean["confirmatory_category_valid"] is True


def test_current_dataset_status_separates_initial_holdout_from_opened_evidence() -> None:
    assert _current_dataset_status("OFFICIAL_FAIL", "FRESH_OR_UNOPENED") == (
        "OPENED_AFTER_D16_OFFICIAL_ATTEMPT"
    )
    assert _current_dataset_status("OFFICIAL_PASS", "FRESH_OR_UNOPENED") == (
        "OPENED_AFTER_D16_OFFICIAL_ATTEMPT"
    )
    assert _current_dataset_status("DEVELOPMENT_ONLY", "OPENED_DEVELOPMENT_ONLY") == (
        "OPENED_DEVELOPMENT_ONLY"
    )
