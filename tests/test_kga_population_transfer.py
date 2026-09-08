"""Adversarial tests for conditional cell-to-population interval composition."""

from __future__ import annotations

import math

import pytest

from kga.policy import Decision
from kga.population_transfer import (
    compose_conditional_population_interval,
    hoeffding_paired_accuracy_radius,
)


def test_hoeffding_paired_accuracy_uses_fixed_width_two() -> None:
    # Hand-calculated from width 2, n=100, and delta=.05.
    assert hoeffding_paired_accuracy_radius(n=100, delta=0.05) == pytest.approx(0.2716203031481239)


def test_hoeffding_paired_accuracy_radius_is_clipped_at_two() -> None:
    assert hoeffding_paired_accuracy_radius(n=1, delta=1.0e-300) == 2.0


def test_hoeffding_handles_the_smallest_positive_float_without_overflow() -> None:
    assert hoeffding_paired_accuracy_radius(n=1, delta=math.nextafter(0.0, 1.0)) == 2.0


@pytest.mark.parametrize("n", [True, False, 0, -1, 1.0])
def test_hoeffding_rejects_nonpositive_or_noninteger_sample_sizes(n: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        hoeffding_paired_accuracy_radius(n=n, delta=0.05)  # type: ignore[arg-type]


@pytest.mark.parametrize("delta", [True, False, 0.0, 1.0, -0.1, math.inf, math.nan])
def test_hoeffding_rejects_invalid_sampling_error_budgets(delta: object) -> None:
    with pytest.raises(ValueError, match="delta"):
        hoeffding_paired_accuracy_radius(n=100, delta=delta)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("delta_hat", "expected_action", "expected_interval"),
    [
        (0.40, Decision.ADAPT, (0.20, 0.60)),
        (-0.40, Decision.FREEZE, (-0.60, -0.20)),
        (0.20, Decision.ABSTAIN, (0.00, 0.40)),
        (-0.20, Decision.ABSTAIN, (-0.40, 0.00)),
    ],
)
def test_conditional_composition_adds_radii_and_uses_strict_actions(
    delta_hat: float,
    expected_action: Decision,
    expected_interval: tuple[float, float],
) -> None:
    result = compose_conditional_population_interval(
        delta_hat=delta_hat,
        epsilon=0.10,
        r_samp=0.10,
        alpha_cell=0.02,
        delta_sampling=0.02,
        alpha_population=0.05,
    )

    assert result.population_radius == pytest.approx(0.20)
    assert (result.population_lower, result.population_upper) == pytest.approx(expected_interval)
    assert result.action is expected_action
    assert result.alpha_spent == pytest.approx(0.04)


def test_infinite_existing_cell_radius_forces_abstention() -> None:
    result = compose_conditional_population_interval(
        delta_hat=0.90,
        epsilon=math.inf,
        r_samp=0.01,
        alpha_cell=0.02,
        delta_sampling=0.02,
        alpha_population=0.05,
    )

    assert math.isinf(result.population_radius)
    assert result.population_lower == -math.inf
    assert result.population_upper == math.inf
    assert result.action is Decision.ABSTAIN


def test_conditional_composition_rejects_overspent_population_budget() -> None:
    with pytest.raises(ValueError, match=r"alpha_cell \+ delta_sampling"):
        compose_conditional_population_interval(
            delta_hat=0.20,
            epsilon=0.05,
            r_samp=0.05,
            alpha_cell=0.04,
            delta_sampling=0.02,
            alpha_population=0.05,
        )


def test_conditional_composition_rejects_one_ulp_budget_overspend() -> None:
    with pytest.raises(ValueError, match=r"alpha_cell \+ delta_sampling"):
        compose_conditional_population_interval(
            delta_hat=0.20,
            epsilon=0.05,
            r_samp=0.05,
            alpha_cell=0.0,
            delta_sampling=math.nextafter(0.05, math.inf),
            alpha_population=0.05,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha_cell", True),
        ("alpha_cell", math.nan),
        ("alpha_cell", -0.01),
        ("delta_sampling", "0.01"),
        ("delta_sampling", math.inf),
        ("delta_sampling", 1.0),
        ("alpha_population", False),
        ("alpha_population", 0.0),
        ("alpha_population", 1.0),
    ],
)
def test_conditional_composition_rejects_invalid_error_budgets(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "delta_hat": 0.20,
        "epsilon": 0.05,
        "r_samp": 0.05,
        "alpha_cell": 0.01,
        "delta_sampling": 0.01,
        "alpha_population": 0.05,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        compose_conditional_population_interval(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("delta_hat", True),
        ("delta_hat", math.inf),
        ("epsilon", False),
        ("epsilon", math.nan),
        ("epsilon", -0.01),
        ("r_samp", True),
        ("r_samp", math.inf),
        ("r_samp", -0.01),
    ],
)
def test_conditional_composition_rejects_invalid_numeric_inputs(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "delta_hat": 0.20,
        "epsilon": 0.05,
        "r_samp": 0.05,
        "alpha_cell": 0.01,
        "delta_sampling": 0.01,
        "alpha_population": 0.05,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        compose_conditional_population_interval(**arguments)  # type: ignore[arg-type]
