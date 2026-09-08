import math

import pytest

from kga.evaluation import evaluate_actions


def test_hand_derived_record_contributions_and_summary() -> None:
    report = evaluate_actions(
        ["ADAPT", "ADAPT", "FREEZE", "FREEZE", "ABSTAIN"],
        [0.5, -0.25, 0.4, -0.1, 0.3],
    )

    expected_numeric_records = (
        (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
        (-0.25, 0.25, 0.25, 0.0, 0.0, 0.25, 0.0),
        (0.4, 0.4, 0.0, 0.4, 0.0, 0.0, 0.4),
        (-0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0),
        (0.3, 0.3, 0.0, 0.0, 0.3, 0.0, 0.3),
    )
    for record, expected in zip(report.records, expected_numeric_records, strict=True):
        assert (
            record.delta,
            record.regret,
            record.harmful_adapt_regret,
            record.missed_benefit_freeze_regret,
            record.abstain_opportunity_regret,
            record.always_adapt_regret,
            record.always_freeze_regret,
        ) == pytest.approx(expected)
    assert tuple(record.action for record in report.records) == ("ADAPT", "ADAPT", "FREEZE", "FREEZE", "ABSTAIN")
    assert report.summary.n == 5
    assert report.summary.adapt_count == 2
    assert report.summary.freeze_count == 2
    assert report.summary.abstain_count == 1
    assert report.summary.adapt_rate == pytest.approx(0.4)
    assert report.summary.freeze_rate == pytest.approx(0.4)
    assert report.summary.abstain_rate == pytest.approx(0.2)
    assert report.summary.commitment_count == 4
    assert report.summary.commitment_rate == pytest.approx(0.8)
    assert report.summary.false_adapt_count == 1
    assert report.summary.false_freeze_count == 1
    assert report.summary.false_adapt_rate_all == pytest.approx(0.2)
    assert report.summary.false_freeze_rate_all == pytest.approx(0.2)
    assert report.summary.false_adapt_rate_conditional == pytest.approx(0.5)
    assert report.summary.false_freeze_rate_conditional == pytest.approx(0.5)
    assert report.summary.directional_accuracy_commitments == pytest.approx(0.5)
    assert report.summary.mean_regret == pytest.approx(0.19)
    assert report.summary.mean_regret_commitments == pytest.approx(0.1625)
    assert report.summary.mean_harmful_adapt_regret == pytest.approx(0.05)
    assert report.summary.mean_missed_benefit_freeze_regret == pytest.approx(0.08)
    assert report.summary.mean_abstain_opportunity_regret == pytest.approx(0.06)
    assert report.summary.mean_always_adapt_regret == pytest.approx(0.07)
    assert report.summary.mean_always_freeze_regret == pytest.approx(0.24)
    assert report.summary.always_adapt_minus_kga_regret == pytest.approx(-0.12)
    assert report.summary.always_freeze_minus_kga_regret == pytest.approx(0.05)
    assert "equally weighted" in report.summary.description.lower()
    assert "not an inferential independent sample size" in report.summary.description


def test_zero_is_an_error_for_both_committed_actions() -> None:
    report = evaluate_actions(["ADAPT", "FREEZE", "ABSTAIN"], [0.0, 0.0, 0.0])
    assert report.summary.false_adapt_count == 1
    assert report.summary.false_freeze_count == 1
    assert report.summary.directional_accuracy_commitments == 0.0
    assert report.summary.mean_regret == 0.0


@pytest.mark.parametrize(
    ("actions", "expected_fa", "expected_ff", "expected_commitment_regret"),
    [
        (["ABSTAIN", "ABSTAIN"], None, None, None),
        (["FREEZE", "ABSTAIN"], None, 1.0, 0.2),
        (["ADAPT", "ABSTAIN"], 0.0, None, 0.0),
    ],
)
def test_zero_denominator_conditional_metrics_are_none(
    actions: list[str],
    expected_fa: float | None,
    expected_ff: float | None,
    expected_commitment_regret: float | None,
) -> None:
    report = evaluate_actions(actions, [0.2, 0.1])
    assert report.summary.false_adapt_rate_conditional == expected_fa
    assert report.summary.false_freeze_rate_conditional == expected_ff
    assert report.summary.mean_regret_commitments == expected_commitment_regret


def test_regret_identity_and_positive_baseline_gap_favors_kga() -> None:
    report = evaluate_actions(["ADAPT", "FREEZE", "ABSTAIN"], [0.6, -0.4, 0.0])
    components = (
        report.summary.mean_harmful_adapt_regret
        + report.summary.mean_missed_benefit_freeze_regret
        + report.summary.mean_abstain_opportunity_regret
    )
    assert report.summary.mean_regret == pytest.approx(components)
    assert report.summary.always_freeze_minus_kga_regret > 0
    assert report.summary.always_adapt_minus_kga_regret > 0


def test_summary_is_permutation_invariant() -> None:
    first = evaluate_actions(["ADAPT", "FREEZE", "ABSTAIN"], [-0.2, 0.6, 0.1])
    second = evaluate_actions(["ABSTAIN", "ADAPT", "FREEZE"], [0.1, -0.2, 0.6])
    assert first.summary == second.summary


def test_inputs_are_copied_and_outputs_are_immutable() -> None:
    actions = ["ADAPT"]
    deltas = [0.25]
    report = evaluate_actions(actions, deltas)
    actions[0] = "FREEZE"
    deltas[0] = -1.0
    assert report.records[0].action == "ADAPT"
    assert report.records[0].delta == 0.25
    with pytest.raises((AttributeError, TypeError)):
        report.records[0].delta = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    ("actions", "deltas"),
    [
        ([], []),
        (["ADAPT"], []),
        (["adapt"], [0.1]),
        (["UNKNOWN"], [0.1]),
        ([True], [0.1]),
        (["ADAPT"], [True]),
        (["ADAPT"], [math.nan]),
        (["ADAPT"], [math.inf]),
        (["ADAPT"], [-math.inf]),
        (["ADAPT"], [1.01]),
        (["ADAPT"], [-1.01]),
        ([["ADAPT"]], [0.1]),
        (["ADAPT"], [[0.1]]),
        ("ADAPT", [0.1]),
    ],
)
def test_invalid_inputs_are_rejected(actions: object, deltas: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluate_actions(actions, deltas)  # type: ignore[arg-type]
