"""Literal, development-only pilot endpoint checks; no real outcomes."""

import math
from dataclasses import replace

import pytest

from experiments.kbound.domainnet.pilot_analysis import ScoredCell, source_readiness, summarize_check


def cell(index=0, positive=True, abstain=False):
    correct = (0,) * 128
    wrong32 = (1,) * 32 + (0,) * 96
    return ScoredCell(
        cell_id=f"DEV_check:{index}",
        labels=correct,
        frozen_predictions=wrong32 if positive else correct,
        candidate_predictions=correct if positive else wrong32,
        delta_hat=0.0 if abstain else (0.25 if positive else -0.25),
        epsilon=1.0 if abstain else 0.01,
        action="ABSTAIN" if abstain else ("ADAPT" if positive else "FREEZE"),
        no_radius_action="ABSTAIN" if abstain else ("ADAPT" if positive else "FREEZE"),
    )


def panel(abstain=False):
    return [cell(i, i < 10, abstain) for i in range(20)]


def test_literal_balanced_go_has_both_gaps_and_leave_one_cell_support():
    result = summarize_check(panel())
    assert result["status"] == "GO_FURTHER_DEVELOPMENT"
    assert result["helpful_count"] == result["harmful_count"] == 10
    assert result["tie_count"] == 0
    assert result["oracle_headroom"] == 0.125
    assert result["kga"]["mean_regret"] == 0.0
    assert result["kga"]["always_adapt_minus_kga_regret"] == 0.125
    assert result["kga"]["always_freeze_minus_kga_regret"] == 0.125
    assert result["minimum_leave_one_out_gaps"] == pytest.approx({"vs_adapt": 2.25 / 19, "vs_freeze": 2.25 / 19})
    assert result["cell_interval_inclusion"] == 1.0
    assert result["sample_weighted"]["n_images"] == 2560
    assert result["sample_weighted"]["frozen_accuracy"] == 0.875
    assert result["sample_weighted"]["served_accuracy"] == 1.0
    assert result["macro_recall"]["present_classes"] == 1
    assert result["macro_recall"]["absent_classes"] == 125
    assert result["macro_recall"]["served"] == 1.0
    assert result["eligible_for_confirmatory"] is False
    assert result["population_certificate"] is False


def test_all_abstain_is_not_a_success_and_cost_is_exposed():
    result = summarize_check(panel(abstain=True))
    assert result["status"] == "INCONCLUSIVE_GATE"
    assert result["kga"]["commitment_rate"] == 0.0
    assert result["kga"]["false_adapt_rate_conditional"] is None
    assert result["kga"]["false_freeze_rate_conditional"] is None
    assert result["kga"]["mean_abstain_opportunity_regret"] == 0.125
    assert result["sample_weighted"]["served_accuracy"] == 0.875


def test_one_sided_no_opportunity_and_incomplete_geometry():
    assert summarize_check([cell(i) for i in range(20)])["status"] == "NO_DEMONSTRATED_OPPORTUNITY"
    assert summarize_check(panel()[:19])["status"] == "INCONCLUSIVE_DATA_GEOMETRY"


def test_tie_is_false_for_both_strict_claims_but_zero_regret():
    rows = panel()
    rows[0] = replace(rows[0], frozen_predictions=rows[0].candidate_predictions)
    rows[10] = replace(rows[10], candidate_predictions=rows[10].frozen_predictions)
    result = summarize_check(rows)
    assert result["tie_count"] == 2
    assert result["kga"]["false_adapt_count"] == result["kga"]["false_freeze_count"] == 1
    assert result["kga"]["mean_regret"] == 0.0
    assert result["cell_interval_inclusion"] == 0.9


def test_unavailable_radius_is_not_invented_coverage_or_certified_freeze():
    rows = [replace(r, epsilon=None, action="ABSTAIN") for r in panel()]
    result = summarize_check(rows)
    assert result["cell_interval_inclusion"] is None
    assert result["unavailable_intervals"] == 20
    assert result["status"] == "INCONCLUSIVE_GATE"
    assert result["kga"]["freeze_count"] == 0
    assert result["no_radius"]["mean_regret"] == 0.0


@pytest.mark.parametrize(
    "mutation",
    [
        {"cell_id": ""},
        {"cell_id": "DEV_fit:0"},
        {"labels": ()},
        {"labels": (True,) * 128},
        {"labels": (126,) * 128},
        {"candidate_predictions": (0,) * 127},
        {"frozen_predictions": (-1,) * 128},
        {"delta_hat": True},
        {"delta_hat": math.inf},
        {"epsilon": math.nan},
        {"epsilon": -1.0},
        {"epsilon": True},
        {"action": "FREEZE"},
        {"no_radius_action": "FREEZE"},
        {"labels": [0] * 128},
    ],
)
def test_malformed_or_inconsistent_records_rejected(mutation):
    with pytest.raises((TypeError, ValueError)):
        summarize_check([replace(cell(), **mutation)])


def test_duplicate_ids_and_undersized_scored_windows_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        summarize_check([cell(), cell()])
    row = replace(cell(), labels=(0,), frozen_predictions=(1,), candidate_predictions=(0,))
    with pytest.raises(ValueError, match="128"):
        summarize_check([row])


def test_source_readiness_uses_final_epoch_not_best_epoch():
    rows = [
        {
            "epoch": i,
            "processed_count": 16811,
            "source_monitor_count": 1892,
            "mean_loss": 3.0 if i <= 5 else 2.0,
            "source_monitor_accuracy": 0.1,
        }
        for i in range(1, 21)
    ]
    assert source_readiness(rows)["ready"] is True
    rows[-1]["source_monitor_accuracy"] = 0.04
    assert source_readiness(rows)["status"] == "INCONCLUSIVE_SOURCE_LEARNING"
    rows[-1]["source_monitor_accuracy"] = 0.1
    for row in rows:
        row["mean_loss"] = 2.0
    assert source_readiness(rows)["ready"] is False
    with pytest.raises(ValueError):
        source_readiness(rows[:-1])
    rows[-1]["processed_count"] = 16810
    with pytest.raises(ValueError):
        source_readiness(rows)
