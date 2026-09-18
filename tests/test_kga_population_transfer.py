"""tests.test_kga_population_transfer -- Tests for kga.population_transfer.

Verifies the mathematical and interface invariants of the cell-to-population bridge:
    * Hoeffding paired accuracy radius calculation and reference value
    * Numeric trichotomy actions (ADAPT, FREEZE, ABSTAIN)
    * Infeasible conformal rank / infinite uncertainty handling
    * Input validation (types, bounds, budgets, overspending)
    * Monotonicity under sampling radius expansion
    * Zero-effect boundary behavior
"""

from __future__ import annotations

import math
import pytest

from kga.policy import Decision
from kga.population_transfer import (
    ConditionalPopulationInterval,
    compose_conditional_population_interval,
    hoeffding_paired_accuracy_radius,
)


def test_hoeffding_paired_accuracy_radius_reference():
    """Verify reference value: n=100, delta=0.05 gives ~0.2716203031481239."""
    b = hoeffding_paired_accuracy_radius(n=100, delta=0.05)
    expected = 0.2716203031481239
    assert b == pytest.approx(expected, abs=1e-12)


def test_hoeffding_paired_accuracy_radius_capped_at_two():
    """Verify that small n does not exceed maximum possible 0-1 difference of 2."""
    b = hoeffding_paired_accuracy_radius(n=1, delta=0.01)
    assert b == 2.0


@pytest.mark.parametrize(
    "estimate, expected",
    [
        (0.4, Decision.ADAPT),
        (-0.4, Decision.FREEZE),
        (0.2, Decision.ABSTAIN),
        (-0.2, Decision.ABSTAIN),
        (0.0, Decision.ABSTAIN),
    ],
)
def test_population_interval_signs(estimate, expected):
    """Verify standard trichotomy decisions under symmetric radius 0.2."""
    result = compose_conditional_population_interval(
        delta_hat=estimate,
        epsilon=0.1,
        r_samp=0.1,
        alpha_cell=0.02,
        delta_sampling=0.02,
        alpha_population=0.05,
    )
    assert result.population_radius == pytest.approx(0.2)
    assert result.action is expected


def test_infinite_uncertainty_abstains():
    """Verify that infinite conformal radius returns ABSTAIN and infinite endpoints."""
    result = compose_conditional_population_interval(
        delta_hat=0.8,
        epsilon=float("inf"),
        r_samp=0.1,
        alpha_cell=0.05,
        delta_sampling=0.05,
        alpha_population=0.10,
    )
    assert math.isinf(result.population_radius)
    assert result.action is Decision.ABSTAIN
    assert result.interval_lower == float("-inf")
    assert result.interval_upper == float("inf")


def test_expansion_monotonicity():
    """Adding nonnegative sampling radius b cannot create a new strict commitment."""
    # When cell-level is ambiguous (ABSTAIN), population-level cannot commit
    res_cell = compose_conditional_population_interval(
        delta_hat=0.05,
        epsilon=0.10,
        r_samp=0.0,
        alpha_cell=0.05,
        delta_sampling=0.01,
        alpha_population=0.10,
    )
    assert res_cell.action is Decision.ABSTAIN

    res_pop = compose_conditional_population_interval(
        delta_hat=0.05,
        epsilon=0.10,
        r_samp=0.05,
        alpha_cell=0.05,
        delta_sampling=0.01,
        alpha_population=0.10,
    )
    assert res_pop.action is Decision.ABSTAIN

    # When cell-level commits (ADAPT), larger b can only retain ADAPT or revert to ABSTAIN
    res_adapt_small = compose_conditional_population_interval(
        delta_hat=0.25,
        epsilon=0.10,
        r_samp=0.05,
        alpha_cell=0.05,
        delta_sampling=0.01,
        alpha_population=0.10,
    )
    assert res_adapt_small.action is Decision.ADAPT

    res_adapt_large = compose_conditional_population_interval(
        delta_hat=0.25,
        epsilon=0.10,
        r_samp=0.20,
        alpha_cell=0.05,
        delta_sampling=0.01,
        alpha_population=0.10,
    )
    assert res_adapt_large.action is Decision.ABSTAIN


def test_budget_overspend_raises():
    """Reject budget allocation where alpha_cell + delta_sampling > alpha_population."""
    with pytest.raises(ValueError, match="Budget overspent"):
        compose_conditional_population_interval(
            delta_hat=0.4,
            epsilon=0.1,
            r_samp=0.1,
            alpha_cell=0.06,
            delta_sampling=0.05,
            alpha_population=0.10,
        )


def test_invalid_inputs_rejected():
    """Reject invalid sample counts, negative radii, and out-of-bound budgets."""
    with pytest.raises(TypeError):
        hoeffding_paired_accuracy_radius(n=True, delta=0.05)
    with pytest.raises(TypeError):
        hoeffding_paired_accuracy_radius(n=100.5, delta=0.05)
    with pytest.raises(ValueError):
        hoeffding_paired_accuracy_radius(n=0, delta=0.05)
    with pytest.raises(ValueError):
        hoeffding_paired_accuracy_radius(n=100, delta=1.5)

    with pytest.raises(ValueError):
        compose_conditional_population_interval(
            delta_hat=0.4,
            epsilon=-0.1,
            r_samp=0.1,
            alpha_cell=0.05,
            delta_sampling=0.05,
            alpha_population=0.10,
        )
    with pytest.raises(ValueError):
        compose_conditional_population_interval(
            delta_hat=0.4,
            epsilon=0.1,
            r_samp=-0.1,
            alpha_cell=0.05,
            delta_sampling=0.05,
            alpha_population=0.10,
        )
