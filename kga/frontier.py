"""Population strict-commitment frontier for a declared target class.

This module is intentionally separate from :mod:`kga.policy`. The population
frontier consumes an observable population centered-correctness margin ``M``
and an externally declared residual budget ``beta``. Empirical KGA consumes
``Delta_hat`` and ``epsilon``. Neither pair estimates or aliases the other.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

from kga.policy import Decision


@dataclass(frozen=True)
class PopulationBenefitInterval:
    """Population-benefit interval after an explicit mass conversion.

    This is a deterministic change of units, not a confidence interval. The
    caller remains responsible for how ``M``, ``beta``, and ``mass`` apply to
    the declared population.
    """

    mass: float
    lower: float
    upper: float


@dataclass(frozen=True)
class FrontierAssessment:
    """Auditable centered-correctness result of one frontier evaluation."""

    margin: float
    residual_budget: float
    lower_centered_correctness: float
    upper_centered_correctness: float
    action: Decision

    def population_benefit_interval(self, *, mass: float) -> PopulationBenefitInterval:
        """Convert to population-benefit units using a declared positive mass.

        If ``mass`` is the population probability of the relevant disagreement
        event, the population accuracy-difference interval is
        ``2 * mass * [lower_centered_correctness, upper_centered_correctness]``.
        Requiring the mass explicitly prevents a conditional margin from being
        mislabeled as an unconditional population benefit.
        """
        if isinstance(mass, bool) or not isinstance(mass, Real):
            raise ValueError("mass must be a real number, not boolean or text")
        population_mass = float(mass)
        if not math.isfinite(population_mass) or not 0 < population_mass <= 1:
            raise ValueError(f"mass must be finite and lie in (0, 1], got {population_mass}")

        scale = 2.0 * population_mass
        return PopulationBenefitInterval(
            mass=population_mass,
            lower=scale * self.lower_centered_correctness,
            upper=scale * self.upper_centered_correctness,
        )


def assess_frontier(M: float, beta: float | None) -> FrontierAssessment:
    """Evaluate the maximal sound strict action over ``|gamma| <= beta``.

    ``M`` is a centered-correctness margin and therefore must lie in
    ``[-0.5, 0.5]``. ``beta`` bounds the declared residual ``gamma``; it is not
    automatically distribution drift. ``None`` is not interpreted as zero:
    when no credible budget exists, the function fails closed instead of
    manufacturing a commitment certificate.
    """
    if isinstance(M, bool) or not isinstance(M, Real):
        raise ValueError("M must be a real number, not boolean or text")
    margin = float(M)
    if beta is None:
        raise ValueError("beta is required and must be externally specified")
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise ValueError("beta must be a real number, not boolean or text")
    residual_budget = float(beta)
    if not math.isfinite(margin) or not -0.5 <= margin <= 0.5:
        raise ValueError(f"M must be finite and lie in [-0.5, 0.5], got {margin}")
    if not math.isfinite(residual_budget) or residual_budget < 0:
        raise ValueError(f"beta must be finite and nonnegative, got {residual_budget}")

    lower = max(-0.5, margin - residual_budget)
    upper = min(0.5, margin + residual_budget)
    if lower > 0:
        action = Decision.ADAPT
    elif upper < 0:
        action = Decision.FREEZE
    else:
        action = Decision.ABSTAIN
    return FrontierAssessment(margin, residual_budget, lower, upper, action)


def frontier_action(M: float, beta: float | None) -> Decision:
    """Return ADAPT/FREEZE/ABSTAIN under the population frontier."""
    return assess_frontier(M, beta).action


def frontier_sensitivity(M: float, beta_values: Iterable[float]) -> list[FrontierAssessment]:
    """Evaluate declared residual budgets without selecting one from outcomes."""
    values = list(beta_values)
    if not values:
        raise ValueError("beta_values must be nonempty")
    return [assess_frontier(M, beta) for beta in values]
