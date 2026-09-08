"""Adversarial unit and domain tests for the population frontier."""

from __future__ import annotations

import math

import pytest

from kga.frontier import assess_frontier, frontier_action, frontier_sensitivity
from kga.policy import Decision


def test_positive_frontier_clips_centered_correctness_before_conversion() -> None:
    assessment = assess_frontier(M=0.4, beta=0.3)

    assert assessment.margin == pytest.approx(0.4)
    assert assessment.residual_budget == pytest.approx(0.3)
    assert assessment.lower_centered_correctness == pytest.approx(0.1)
    assert assessment.upper_centered_correctness == pytest.approx(0.5)
    assert assessment.action is Decision.ADAPT

    benefit = assessment.population_benefit_interval(mass=0.2)
    assert benefit.mass == pytest.approx(0.2)
    assert benefit.lower == pytest.approx(0.04)
    assert benefit.upper == pytest.approx(0.2)


def test_negative_frontier_clips_centered_correctness_before_conversion() -> None:
    assessment = assess_frontier(M=-0.4, beta=0.3)

    assert assessment.lower_centered_correctness == pytest.approx(-0.5)
    assert assessment.upper_centered_correctness == pytest.approx(-0.1)
    assert assessment.action is Decision.FREEZE

    benefit = assessment.population_benefit_interval(mass=0.2)
    assert benefit.lower == pytest.approx(-0.2)
    assert benefit.upper == pytest.approx(-0.04)


def test_large_residual_budget_clips_to_full_domain_and_abstains() -> None:
    assessment = assess_frontier(M=0.4, beta=1.0)

    assert assessment.lower_centered_correctness == -0.5
    assert assessment.upper_centered_correctness == 0.5
    assert assessment.action is Decision.ABSTAIN


@pytest.mark.parametrize(
    ("margin", "residual_budget"),
    [(0.1, 0.1), (-0.1, 0.1), (0.0, 0.0)],
)
def test_zero_touching_interval_abstains(margin: float, residual_budget: float) -> None:
    assert frontier_action(margin, residual_budget) is Decision.ABSTAIN


@pytest.mark.parametrize(
    "margin",
    [-0.5000001, 0.5000001, math.nan, math.inf, -math.inf],
)
def test_margin_outside_centered_correctness_domain_is_rejected(
    margin: float,
) -> None:
    with pytest.raises(ValueError, match=r"M must"):
        assess_frontier(M=margin, beta=0.1)


@pytest.mark.parametrize(
    "residual_budget",
    [None, -1e-12, math.nan, math.inf, -math.inf],
)
def test_invalid_residual_budget_is_rejected(
    residual_budget: float | None,
) -> None:
    with pytest.raises(ValueError, match=r"beta"):
        assess_frontier(M=0.0, beta=residual_budget)


@pytest.mark.parametrize(
    "mass",
    [0.0, -1e-12, 1.0000001, math.nan, math.inf, -math.inf],
)
def test_invalid_population_mass_is_rejected(mass: float) -> None:
    assessment = assess_frontier(M=0.25, beta=0.05)

    with pytest.raises(ValueError, match=r"mass"):
        assessment.population_benefit_interval(mass=mass)


def test_population_mass_is_explicit_and_keyword_only() -> None:
    assessment = assess_frontier(M=0.25, beta=0.05)

    with pytest.raises(TypeError):
        assessment.population_benefit_interval()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        assessment.population_benefit_interval(0.5)  # type: ignore[misc]


def test_unit_mass_recovers_the_full_population_benefit_interval() -> None:
    assessment = assess_frontier(M=0.25, beta=0.05)

    benefit = assessment.population_benefit_interval(mass=1.0)
    assert benefit.lower == pytest.approx(0.4)
    assert benefit.upper == pytest.approx(0.6)


def test_sensitivity_preserves_declared_order_and_strict_actions() -> None:
    assessments = frontier_sensitivity(0.2, [0.0, 0.1, 0.2, 0.3])

    assert [item.residual_budget for item in assessments] == [0.0, 0.1, 0.2, 0.3]
    assert [item.action for item in assessments] == [
        Decision.ADAPT,
        Decision.ADAPT,
        Decision.ABSTAIN,
        Decision.ABSTAIN,
    ]
    assert all(-0.5 <= item.lower_centered_correctness <= 0.5 for item in assessments)
    assert all(-0.5 <= item.upper_centered_correctness <= 0.5 for item in assessments)


def test_sensitivity_requires_at_least_one_declared_budget() -> None:
    with pytest.raises(ValueError, match=r"nonempty"):
        frontier_sensitivity(0.2, [])


@pytest.mark.parametrize("value", [True, False, "0.2", [], None])
def test_frontier_margin_does_not_coerce_nonnumeric_metadata(value) -> None:
    with pytest.raises(ValueError, match="M"):
        assess_frontier(value, 0.1)


@pytest.mark.parametrize("value", [True, False, "0.2", []])
def test_frontier_residual_budget_does_not_coerce_metadata(value) -> None:
    with pytest.raises(ValueError, match="beta"):
        assess_frontier(0.2, value)


@pytest.mark.parametrize("value", [True, False, "0.2", [], None])
def test_population_mass_does_not_coerce_metadata(value) -> None:
    with pytest.raises(ValueError, match="mass"):
        assess_frontier(0.2, 0.1).population_benefit_interval(mass=value)
