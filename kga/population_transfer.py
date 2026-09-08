"""Conditional interval composition from a cell target to a population target.

This module performs one mathematical composition only.  It does not validate
an experimental design, establish exchangeability, or issue a population
deployment certificate.  Callers must supply both the existing cell radius
``epsilon`` and a separately justified sampling radius ``r_samp``.

For a new episode, ``U_e`` may construct the adapted candidate and ``V_e`` is
the unlabeled decision window.  The scored sample ``E_e`` must be fresh IID
*after* the candidate and decision are fixed for the Hoeffding helper below to
apply.  Historical labelled residual-calibration environments are denoted
``C``; they are not ``V_e``.  Merely making ``U_e``, ``V_e``, ``E_e``, and
``C`` disjoint does not prove that ``E_e`` is IID or that cell coverage
transfers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from kga.policy import Decision


@dataclass(frozen=True)
class ConditionalPopulationInterval:
    """Result of conditional interval composition, not a design certificate."""

    delta_hat: float
    cell_radius: float
    sampling_radius: float
    population_radius: float
    population_lower: float
    population_upper: float
    action: Decision
    alpha_cell: float
    delta_sampling: float
    alpha_population: float
    alpha_spent: float

    @property
    def population_interval(self) -> tuple[float, float]:
        """Return the composed population-target interval endpoints."""

        return self.population_lower, self.population_upper


def _finite_real(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite numeric value, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _component_budget(value: Any, *, field: str) -> float:
    result = _finite_real(value, field=field)
    if not 0.0 <= result < 1.0:
        raise ValueError(f"{field} must lie in [0, 1)")
    return result


def _population_budget(value: Any) -> float:
    result = _finite_real(value, field="alpha_population")
    if not 0.0 < result < 1.0:
        raise ValueError("alpha_population must lie in (0, 1)")
    return result


def hoeffding_paired_accuracy_radius(*, n: int, delta: float) -> float:
    """Return a two-sided Hoeffding radius for paired accuracy benefit.

    Each paired correctness difference lies in ``[-1, 1]``, so its fixed
    support width is 2.  For a fresh IID scored sample ``E_e`` of size ``n``
    after the candidate and decision are fixed, the returned radius is

    ``min(2, sqrt(2 * log(2 / delta) / n))``.

    ``delta`` is a required sampling-error budget.  It is not estimated from
    outcomes, and this function accepts no observations.  Disjoint unlabeled
    ``U_e``/``V_e`` and scored ``E_e`` windows alone do not establish IID
    sampling.
    """

    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be a positive integer")
    delta_value = _finite_real(delta, field="delta")
    if not 0.0 < delta_value < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    width = 2.0
    log_two_over_delta = math.log(2.0) - math.log(delta_value)
    radius = width * math.sqrt(log_two_over_delta / (2.0 * int(n)))
    return min(width, radius)


def compose_conditional_population_interval(
    *,
    delta_hat: float,
    epsilon: float,
    r_samp: float,
    alpha_cell: float,
    delta_sampling: float,
    alpha_population: float,
) -> ConditionalPopulationInterval:
    """Conditionally compose cell and sampling radii by a union bound.

    Given separately justified events

    ``|delta_hat - delta_cell| <= epsilon`` with failure ``alpha_cell`` and
    ``|delta_cell - delta_population| <= r_samp`` with failure
    ``delta_sampling``, the triangle inequality gives radius
    ``epsilon + r_samp`` with total failure at most their sum.  Independence
    between those two events is not required.

    This function checks only numeric inputs and the declared budget relation
    ``alpha_cell + delta_sampling <= alpha_population``.  It cannot establish
    either premise, IID sampling, metric alignment, or deployment validity.
    ``r_samp`` has no default and is never inferred from target outcomes.

    An existing ``epsilon=+inf`` remains maximally uncertain and therefore
    forces ``ABSTAIN``.  Otherwise actions use strict interval signs.
    """

    estimate = _finite_real(delta_hat, field="delta_hat")
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise ValueError("epsilon must be a non-negative numeric value")
    cell_radius = float(epsilon)
    if math.isnan(cell_radius) or cell_radius < 0.0 or cell_radius == -math.inf:
        raise ValueError("epsilon must be non-negative or +inf")
    sampling_radius = _finite_real(r_samp, field="r_samp")
    if sampling_radius < 0.0:
        raise ValueError("r_samp must be non-negative")

    cell_budget = _component_budget(alpha_cell, field="alpha_cell")
    sampling_budget = _component_budget(delta_sampling, field="delta_sampling")
    population_budget = _population_budget(alpha_population)
    spent = math.fsum((cell_budget, sampling_budget))
    if spent > population_budget:
        raise ValueError("alpha_cell + delta_sampling must be <= alpha_population")

    population_radius = cell_radius + sampling_radius
    if math.isinf(population_radius):
        lower = -math.inf
        upper = math.inf
        action = Decision.ABSTAIN
    else:
        lower = estimate - population_radius
        upper = estimate + population_radius
        if lower > 0.0:
            action = Decision.ADAPT
        elif upper < 0.0:
            action = Decision.FREEZE
        else:
            action = Decision.ABSTAIN

    return ConditionalPopulationInterval(
        delta_hat=estimate,
        cell_radius=cell_radius,
        sampling_radius=sampling_radius,
        population_radius=population_radius,
        population_lower=lower,
        population_upper=upper,
        action=action,
        alpha_cell=cell_budget,
        delta_sampling=sampling_budget,
        alpha_population=population_budget,
        alpha_spent=spent,
    )
