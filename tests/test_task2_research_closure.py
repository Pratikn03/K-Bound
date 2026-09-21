"""Regression tests for the Task 2 formal and population closure gates."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from docs.research.kbound.scripts import run_population_locked_panel as population_runner
from docs.research.kbound.scripts.run_population_locked_panel import (
    generate_locked_panel,
    outcome_blind_perturbation,
    run_locked_population,
    write_panel,
)

ROOT = Path(__file__).resolve().parents[1]


def test_population_writer_preserves_existing_evidence(tmp_path):
    panel = generate_locked_panel(seed=117)
    write_panel(tmp_path, panel, run_locked_population(panel))
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    replacement = generate_locked_panel(seed=118)
    with pytest.raises(FileExistsError, match="(empty|fresh|existing)"):
        write_panel(tmp_path, replacement, run_locked_population(replacement))
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original


def test_population_writer_does_not_overwrite_concurrent_claim(tmp_path, monkeypatch):
    panel = generate_locked_panel(seed=117)
    result = run_locked_population(panel)
    claim = tmp_path / "POPULATION_LOCK_MANIFEST.json"
    real_run = population_runner.run_locked_population

    def concurrent_writer(*args, **kwargs):
        computed = real_run(*args, **kwargs)
        if not claim.exists():
            claim.write_text("another run owns this directory\n")
        return computed

    monkeypatch.setattr(population_runner, "run_locked_population", concurrent_writer)
    with pytest.raises(FileExistsError):
        write_panel(tmp_path, panel, result)
    assert claim.read_text() == "another run owns this directory\n"
    assert not (tmp_path / "POPULATION_PROVENANCE_RECEIPT.json").exists()


def test_population_writer_rejects_result_from_different_panel(tmp_path):
    panel = generate_locked_panel(seed=117)
    other_result = run_locked_population(generate_locked_panel(seed=118))
    with pytest.raises(ValueError, match="(match|bound|differ)"):
        write_panel(tmp_path, panel, other_result)
    assert not list(tmp_path.iterdir())


def test_population_writer_checks_outcome_blindness_at_actual_risk_budgets(tmp_path):
    panel = generate_locked_panel(seed=117)
    result = run_locked_population(panel, alpha_cell=0.10, delta_sampling=0.02, alpha_population=0.12)
    write_panel(tmp_path, panel, result)
    probe = json.loads((tmp_path / "OUTCOME_BLIND_PERTURBATION_RECEIPT.json").read_text())
    assert probe["before_sha256"] == result["decision_path_sha256"]
    assert probe["after_sha256"] == result["decision_path_sha256"]


def test_insufficient_calibration_saves_explicit_unbounded_intervals(tmp_path):
    panel = generate_locked_panel(seed=117, n_fit=24, n_cal=1, n_score=12, m=64)
    result = run_locked_population(panel)
    assert all(row["action"] == "ABSTAIN" and math.isinf(row["population_radius"]) for row in result["decisions"])
    receipt = write_panel(tmp_path, panel, result)
    saved = json.loads(
        (tmp_path / "population_panel_output.json").read_text(),
        parse_constant=lambda value: pytest.fail(f"nonstandard JSON constant: {value}"),
    )
    assert receipt["status"] == "PASS"
    assert all(
        row["cell_radius"] == "+inf"
        and row["population_radius"] == "+inf"
        and row["threshold"] == {"lower": "-inf", "upper": "+inf"}
        for row in saved["decisions"]
    )


@pytest.mark.parametrize(
    "left,right",
    [
        ("fit_episodes", "calibration_episodes"),
        ("fit_episodes", "score_episodes"),
        ("calibration_episodes", "score_episodes"),
    ],
)
def test_population_runner_rejects_episode_reuse_across_roles(left, right):
    panel = generate_locked_panel(seed=117, n_fit=24, n_cal=39, n_score=12, m=64)
    old_id = panel[right][0]["episode_id"]
    panel[right][0]["episode_id"] = panel[left][0]["episode_id"]
    if right == "score_episodes":
        for sidecar in ("sealed_score_outcomes", "sealed_population_benefits"):
            panel[sidecar][panel[right][0]["episode_id"]] = panel[sidecar].pop(old_id)
    with pytest.raises(ValueError, match="episode.*(overlap|duplicate)"):
        run_locked_population(panel)


def test_population_runner_rejects_duplicate_calibration_episodes():
    panel = generate_locked_panel(seed=117, n_fit=24, n_cal=39, n_score=12, m=64)
    panel["calibration_episodes"][1] = copy.deepcopy(panel["calibration_episodes"][0])
    with pytest.raises(ValueError, match="episode.*(overlap|duplicate)"):
        run_locked_population(panel)


def test_outcome_exclusion_probe_uses_a_valid_changed_benefit():
    panel = generate_locked_panel(seed=117, n_fit=24, n_cal=39, n_score=12, m=64)
    panel["sealed_score_outcomes"][sorted(panel["sealed_score_outcomes"])[0]] = 1.0
    receipt = outcome_blind_perturbation(panel)
    assert -1 <= receipt["perturbed_outcome"] <= 1
    assert receipt["perturbed_outcome"] != receipt["original_outcome"]
    assert receipt["decision_path_unchanged"] is True


def test_locked_population_is_episode_scoped_and_provenance_bound() -> None:
    panel = generate_locked_panel(seed=20260911, n_fit=24, n_cal=39, n_score=12, m=64)
    provenance = panel["provenance"]
    assert provenance["episode_level_state"] == "one Z state sampled once and held fixed per episode"
    assert provenance["exchangeability_assumption"]
    assert provenance["score_outcomes_sealed_before_decision"] is True
    assert all("benefit" not in episode for episode in panel["score_episodes"])
    result = run_locked_population(panel)
    assert result["protocol"]["status"] == "PASS"
    assert len(result["decisions"]) == 12
    assert all(
        {"prediction", "cell_radius", "population_radius", "threshold", "action"} <= row.keys()
        for row in result["decisions"]
    )


def test_scored_outcome_perturbation_cannot_change_full_decision_path() -> None:
    panel = generate_locked_panel(seed=20260911, n_fit=24, n_cal=39, n_score=12, m=64)
    before = run_locked_population(panel)
    changed = copy.deepcopy(panel)
    key = sorted(changed["sealed_score_outcomes"])[0]
    changed["sealed_score_outcomes"][key] = -1.0 if changed["sealed_score_outcomes"][key] >= 0 else 1.0
    after = run_locked_population(changed)
    before_path = [
        (
            row["episode_id"],
            row["prediction"],
            row["cell_radius"],
            row["population_radius"],
            row["threshold"],
            row["action"],
        )
        for row in before["decisions"]
    ]
    after_path = [
        (
            row["episode_id"],
            row["prediction"],
            row["cell_radius"],
            row["population_radius"],
            row["threshold"],
            row["action"],
        )
        for row in after["decisions"]
    ]
    assert after_path == before_path


def test_theorem_map_registers_corrected_channel_and_ratio_rate_extensions() -> None:
    theorem_map = (ROOT / "docs/research/kbound/formal/KBound/TheoremMap.lean").read_text(encoding="utf-8")
    assert "#check KBound.historical_one_bit_decoder_iff_fibre_consistent" in theorem_map
    assert "#check KBound.h_ratio_rate_transfer" in theorem_map


def test_all_synthetic_score_outcomes_and_population_truths_are_decision_excluded():
    """Catch score-sidecar use anywhere in refitting, calibration or decisions.

    This is a whole synthetic score-partition test, not evidence that real
    environmental groups were held out or exchangeably sampled.
    """
    panel = generate_locked_panel(seed=117, n_fit=24, n_cal=39, n_score=12, m=64)
    before = run_locked_population(panel)
    changed = copy.deepcopy(panel)
    for name in ("sealed_score_outcomes", "sealed_population_benefits"):
        for key, value in changed[name].items():
            changed[name][key] = -1.0 if value >= 0 else 1.0
    after = run_locked_population(changed)
    assert before["scored_outcomes"] != after["scored_outcomes"]
    assert before["decisions"] == after["decisions"]
    assert before["decision_path_sha256"] == after["decision_path_sha256"]
