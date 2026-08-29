from __future__ import annotations

import inspect

import pytest

from kga.frontier import assess_frontier, frontier_action, frontier_sensitivity
from kga.policy import Decision, decide_kga


@pytest.mark.parametrize(
    ("margin", "budget", "expected"),
    [
        (0.2, 0.1, Decision.ADAPT),
        (-0.2, 0.1, Decision.FREEZE),
        (0.1, 0.1, Decision.ABSTAIN),
        (-0.1, 0.1, Decision.ABSTAIN),
        (0.0, 0.0, Decision.ABSTAIN),
    ],
)
def test_frontier_strict_commitment_rule(margin: float, budget: float, expected: Decision) -> None:
    assert frontier_action(margin, budget) is expected


def test_frontier_fails_closed_without_credible_budget() -> None:
    with pytest.raises(ValueError, match="externally specified"):
        assess_frontier(0.2, None)


def test_population_and_empirical_apis_cannot_substitute_units() -> None:
    frontier_parameters = inspect.signature(frontier_action).parameters
    kga_parameters = inspect.signature(decide_kga).parameters
    assert set(frontier_parameters) == {"M", "beta"}
    assert "epsilon" not in frontier_parameters
    assert "beta" not in kga_parameters
    assert "M" not in kga_parameters
    with pytest.raises(TypeError):
        frontier_action(M=0.2, epsilon=0.1)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        decide_kga([0.2], [0.1], beta=0.1)  # type: ignore[call-arg]


def test_sensitivity_preserves_external_budget_order() -> None:
    assessments = frontier_sensitivity(0.2, [0.0, 0.1, 0.2, 0.3])
    assert [row.action for row in assessments] == [
        Decision.ADAPT,
        Decision.ADAPT,
        Decision.ABSTAIN,
        Decision.ABSTAIN,
    ]
