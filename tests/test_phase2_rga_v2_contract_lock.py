"""Phase 2.2B — RGA-v2 contract lock invariants."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "phase2" / "rga_v2_gate_contract.yaml"


def _contract():
    return yaml.safe_load(CONTRACT.read_text())["contract"]


def test_contract_locks_g0_g1_g2_g3_g4():
    ids = {g["id"] for g in _contract()["candidate_gates"]}
    for required in ("G0", "G1", "G2", "G3", "G4"):
        assert required in ids


def test_g0_baseline_tau_is_066():
    g0 = next(g for g in _contract()["candidate_gates"] if g["id"] == "G0")
    assert g0["tau_mean"] == 0.66
    assert g0["validation_tuning_allowed"] is False


def test_clean_false_fire_budget_rule_is_locked():
    rule = _contract()["clean_false_fire_budget"]["rule"]
    assert "max(0.010" in rule
    assert "base_G0_clean_activation_rate + 0.005" in rule
    assert _contract()["clean_false_fire_budget"]["overrideable"] is False


def test_promotion_criteria_includes_all_six():
    crit_ids = {c["id"] for c in _contract()["promotion_criteria_all_required"]}
    for required in ("C1", "C2", "C3", "C4", "C5", "C6"):
        assert required in crit_ids


def test_g3_top_q_search_grid_locked():
    g3 = next(g for g in _contract()["candidate_gates"] if g["id"] == "G3")
    assert g3["q_search_grid"] == [1, 2]
    assert g3["tau_q_search_grid"] == [0.30, 0.34, 0.40, 0.50, 0.60, 0.66]


def test_g3_top_q_implemented_in_estimator():
    """G3 requires top_q gate mode in ReliabilityEstimator."""
    from uais.fusion.attention.reliability_estimator import ReliabilityEstimator

    assert "top_q" in ReliabilityEstimator._VALID_GATE_MODES
