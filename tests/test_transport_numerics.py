"""Exact arithmetic oracles and adversarial numerical-scope checks."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from kga import paired_transport as transport
from kga import transport_numerics as numerics


def exact_tail(k, n, p):
    return sum((Fraction(math.comb(n, j)) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1)), Fraction(0))


def test_directed_tail_comparisons_agree_with_exact_rational_polynomials():
    for n in range(1, 14):
        for k in range(1, n + 1):
            for numerator in (1, 3, 7, 11, 15):
                p = Fraction(numerator, 16)
                for budget in (Fraction(1, 40), Fraction(1, 2)):
                    proof = numerics.binomial_tail_certificate(k, n, p, budget)
                    assert proof["leq"] == (exact_tail(k, n, p) <= budget)
                    if "tail_lower" in proof:
                        assert Fraction(Decimal(proof["tail_lower"])) <= exact_tail(k, n, p)
                        assert exact_tail(k, n, p) <= Fraction(Decimal(proof["tail_upper"]))


def test_cp_endpoints_and_coverage_are_verified_without_float_tolerances():
    alpha = Fraction(1, 20)
    for n in range(1, 13):
        intervals = [transport.binomial_interval(x, n, alpha) for x in range(n + 1)]
        for x, (lo, hi) in enumerate(intervals):
            if x:
                assert exact_tail(x, n, Fraction(lo)) <= alpha / 2
            if x < n:
                assert exact_tail(n - x, n, 1 - Fraction(hi)) <= alpha / 2
        for numerator in range(17):
            truth = Fraction(numerator, 16)
            coverage = sum(
                (
                    Fraction(math.comb(n, x)) * truth**x * (1 - truth) ** (n - x)
                    for x, (lo, hi) in enumerate(intervals)
                    if Fraction(lo) <= truth <= Fraction(hi)
                ),
                Fraction(0),
            )
            assert coverage >= 1 - alpha


@pytest.mark.parametrize("proposal", [float("nan"), 0.0, 1.0, 0.99, -0.2, 1.2])
def test_corrupt_quantile_proposal_can_only_widen_valid_endpoints(monkeypatch, proposal):
    monkeypatch.setattr(numerics.beta_distribution, "ppf", lambda *args: proposal)
    for x in range(9):
        lo, hi, certificate = numerics.binomial_interval_certificate(x, 8, Fraction(1, 20))
        assert certificate["outward_verified"]
        if x:
            assert exact_tail(x, 8, Fraction(lo)) <= Fraction(1, 40)
        if x < 8:
            assert exact_tail(8 - x, 8, 1 - Fraction(hi)) <= Fraction(1, 40)


def test_probability_underflow_and_resource_limit_return_valid_full_intervals():
    smallest = float(np.nextafter(0.0, 1.0))
    lo, hi, proof = numerics.binomial_interval_certificate(1, 2, smallest)
    assert (lo, hi) == (0.0, 1.0)
    assert proof["alpha_exact"] == numerics.fraction_record(Fraction(smallest))
    lo, hi, proof = numerics.binomial_interval_certificate(3, numerics.MAX_BINOMIAL_TRIALS + 1, 0.05)
    assert (lo, hi) == (0.0, 1.0)
    assert proof["status"] == "resource_limit_full_interval"
    assert transport.binomial_interval(0, 0, 0.05) == (0.0, 1.0)


def test_bonferroni_allocates_exact_rational_budget_even_below_float_range():
    bands = transport.confidence_bounds([[1, 1], [0, 0]], [1, 0], alpha=float(np.nextafter(0.0, 1.0)))
    certificate = bands.numerical_certificate
    family = certificate["family_alpha_exact"]
    per = certificate["per_interval_alpha_exact"]
    assert Fraction(int(per["numerator"]), int(per["denominator"])) * bands.interval_count == Fraction(
        int(family["numerator"]), int(family["denominator"])
    )
    assert bands.per_interval_alpha == 0.0  # display rounds down, exact budget does not vanish
    assert certificate["all_endpoints_outward_verified"]


def test_count_totals_do_not_overflow_int64_before_resource_fallback():
    counts = np.full((2049, 2), 2**52, dtype=np.int64)
    bands = transport.confidence_bounds(counts, np.full(2049, 2**52, dtype=np.int64))
    assert bands.source_class_counts == (2049 * 2**52,) * 2
    assert bands.target_count == 2049 * 2**52
    assert np.all(bands.source_lower == 0) and np.all(bands.source_upper == 1)


def decode(values):
    return [Fraction(int(x["numerator"]), int(x["denominator"])) for x in values]


def test_exact_primal_refuses_tolerance_feasible_but_empty_problem():
    # The contradiction is below the default numerical LP feasibility tolerance.
    delta = 2.0**-40
    solution, proof = numerics.exact_primal_certificate([[1.0], [-1.0]], [0.5, -(0.5 + delta)], [], [], [0.5])
    assert solution is None
    assert proof["status"] == "exact_primal_reconstruction_failed"


def test_exact_primal_recovers_basis_and_checks_every_constraint():
    # 3*x0+x1=1 and x1=0 require the non-binary rational 1/3.
    solution, proof = numerics.exact_primal_certificate([[0.0, 1.0]], [0.0], [[3.0, 1.0]], [1.0], [1 / 3, 0])
    assert solution == [Fraction(1, 3), Fraction(0)]
    assert decode(proof["solution"]) == solution
    assert proof["all_constraints_checked_exactly"]
    solution, proof = numerics.exact_primal_certificate(np.empty((0, 181)), [], np.empty((0, 181)), [], np.zeros(181))
    assert solution is None and proof["status"] == "exact_primal_resource_or_input_limit"


def test_binary64_point_boxes_are_not_silently_normalized_or_widened():
    a = [[0.2, 0.2], [0.8, 0.8]]
    # .2+.8 is >1 as exact binary64 rationals, so no normalized source column exists.
    assert sum(map(Fraction, [0.2, 0.8])) > 1
    with pytest.raises(ValueError, match="simplex"):
        transport.TransportSpec(a, a, [0.2, 0.8], [0.2, 0.8], [[-1.0, 1.0], [-1.0, 1.0]], assumption_contract="test")
    exact_float = transport.TransportSpec(
        [[0.25, 0.25], [0.75, 0.75]],
        [[0.25, 0.25], [0.75, 0.75]],
        [0.25, 0.75],
        [0.25, 0.75],
        [[-1.0, 1.0], [-1.0, 1.0]],
        assumption_contract="test",
    )
    assert not exact_float.numerical_input_metadata["outward_enclosures"]["source_lower"]


def test_explicit_rational_probability_inputs_get_documented_outer_enclosures():
    a = [[Fraction(1, 5), Fraction(1, 5)], [Decimal(".8"), Decimal(".8")]]
    q = [Decimal(".2"), Fraction(4, 5)]
    spec = transport.TransportSpec(a, a, q, q, [[-1, 1], [-1, 1]], rho=Fraction(1, 3), assumption_contract="test")
    assert spec.numerical_input_metadata["outward_enclosures"]["source_lower"]
    for index in np.ndindex((2, 2)):
        truth = Fraction(a[index[0]][index[1]])
        assert Fraction(spec.source_lower[index]) <= truth <= Fraction(spec.source_upper[index])
    assert Fraction(spec.rho) >= Fraction(1, 3)
    result = transport.solve_benefit(spec)
    assert result.verified
    for witness in result.witnesses.values():
        assert witness["exact_primal"]["all_constraints_checked_exactly"]
    with pytest.raises(ValueError, match="Contrast"):
        transport.TransportSpec([[1, 1]], [[1, 1]], [1], [1], [[Fraction(1, 3), 0]])
