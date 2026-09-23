"""Outward statistical bounds and exact primal checks for paired transport.

SciPy estimates only propose quantiles/LP solutions. Certification uses directed
rational enclosures or exact Fraction arithmetic on the supplied binary64 LP.
No sampling assumption is established by these numerical checks.
"""

from __future__ import annotations

import math
from decimal import MAX_EMAX, MIN_EMIN, ROUND_CEILING, ROUND_FLOOR, Context, Decimal
from fractions import Fraction
from typing import Any

import numpy as np
from scipy import sparse
from scipy.stats import beta as beta_distribution

MAX_BINOMIAL_TRIALS = 100_000
DYADIC_BITS = 44
DECIMAL_PRECISION = 80
MAX_EXACT_LP_DIMENSION = 180


def rational(value: Any) -> Fraction:
    """Interpret floats as exact binary64, and Fraction/Decimal as exact values."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Boolean is not a probability or coefficient")
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("Nonfinite number")
        return Fraction.from_float(float(value))
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, (Fraction, Decimal)):
        return Fraction(value)
    raise TypeError("Use a finite float, integer, Fraction or Decimal")


def fraction_record(value: Fraction) -> dict[str, str]:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def outward_float(value: Fraction, upper: bool) -> float:
    out = float(value)
    represented = Fraction.from_float(out)
    if (upper and represented < value) or (not upper and represented > value):
        out = float(np.nextafter(out, np.inf if upper else -np.inf))
    return out


def probability_array(value, ndim, *, upper):
    raw = np.asarray(value, dtype=object)
    if raw.ndim != ndim or not raw.size:
        raise ValueError("Probability bounds must be nonempty arrays of the declared dimension")
    result = np.empty(raw.shape, dtype=float)
    changed = []
    for index in np.ndindex(raw.shape):
        exact = rational(raw[index])
        if not 0 <= exact <= 1:
            raise ValueError("Probability bounds must lie in [0,1]")
        result[index] = outward_float(exact, upper)
        if Fraction.from_float(result[index]) != exact:
            changed.append({"index": list(index), "input": fraction_record(exact), "enclosed_by": result[index]})
    result.setflags(write=False)
    return result, changed


def _power(context: Context, value: Decimal, exponent: int) -> Decimal:
    """Integer powers using only correctly rounded nonnegative multiplications."""
    result = Decimal(1)
    while exponent:
        if exponent & 1:
            result = context.multiply(result, value)
        exponent >>= 1
        if exponent:
            value = context.multiply(value, value)
    return result


def _exact_tail_leq(k: int, n: int, p: Fraction, threshold: Fraction) -> bool:
    if k == 0:
        return Fraction(1) <= threshold
    if p == 0:
        return True
    if p == 1:
        return Fraction(1) <= threshold
    a = int(p.numerator)
    den = int(p.denominator)
    b = den - a
    term: int = math.comb(n, k) * pow(a, k) * pow(b, n - k)
    total: int = term
    for j in range(k, n):
        term = term * (n - j) * a // ((j + 1) * b)
        total += term
    lhs: int = total * int(threshold.denominator)
    rhs: int = int(threshold.numerator) * pow(den, n)
    return lhs <= rhs


def binomial_tail_certificate(k: int, n: int, p: Fraction, threshold: Fraction) -> dict:
    """Certify P_{Bin(n,p)}(X>=k) <= threshold or its strict reverse.

    Every arithmetic enclosure is directed. After term j, future term ratios
    decrease, so t_j*r/(1-r), for an upper bound r<1 on the next ratio, bounds
    all remaining mass. Comparing an outward bound with the exact threshold
    yields a proof; uncertainty never counts as a successful comparison.
    """
    if not 0 <= k <= n or not 0 <= p <= 1 or not 0 < threshold < 1:
        raise ValueError("Invalid tail comparison")
    if p == 0 or p == 1 or k == 0:
        tail = int(k == 0 or p == 1)
        return {
            "leq": Fraction(tail) <= threshold,
            "method": "exact_endpoint",
            "tail_lower": str(tail),
            "tail_upper": str(tail),
        }
    if n > MAX_BINOMIAL_TRIALS:
        return {"leq": None, "method": "resource_limit"}
    down = Context(prec=DECIMAL_PRECISION, rounding=ROUND_FLOOR, Emin=MIN_EMIN, Emax=MAX_EMAX)
    up = Context(prec=DECIMAL_PRECISION, rounding=ROUND_CEILING, Emin=MIN_EMIN, Emax=MAX_EMAX)
    q = 1 - p
    pl = down.divide(Decimal(p.numerator), Decimal(p.denominator))
    pu = up.divide(Decimal(p.numerator), Decimal(p.denominator))
    ql = down.divide(Decimal(q.numerator), Decimal(q.denominator))
    qu = up.divide(Decimal(q.numerator), Decimal(q.denominator))
    coefficient = Decimal(math.comb(n, k))
    tl = down.multiply(down.multiply(coefficient, _power(down, pl, k)), _power(down, ql, n - k))
    tu = up.multiply(up.multiply(coefficient, _power(up, pu, k)), _power(up, qu, n - k))
    sl, su = tl, tu
    for j in range(k, n + 1):
        tail_upper = su
        if j < n:
            rl = down.divide(down.multiply(Decimal(n - j), pl), up.multiply(Decimal(j + 1), qu))
            ru = up.divide(up.multiply(Decimal(n - j), pu), down.multiply(Decimal(j + 1), ql))
            if ru < 1:
                remainder = up.divide(up.multiply(tu, ru), down.subtract(Decimal(1), ru))
                tail_upper = up.add(su, remainder)
            else:
                tail_upper = Decimal(1)
        if Fraction(sl) > threshold:
            return {
                "leq": False,
                "method": "directed_decimal_polynomial",
                "tail_lower": str(sl),
                "tail_upper": str(tail_upper),
                "terms": j - k + 1,
            }
        if Fraction(tail_upper) <= threshold:
            return {
                "leq": True,
                "method": "directed_decimal_polynomial",
                "tail_lower": str(sl),
                "tail_upper": str(tail_upper),
                "terms": j - k + 1,
            }
        if j < n:
            tl, tu = down.multiply(tl, rl), up.multiply(tu, ru)
            sl, su = down.add(sl, tl), up.add(su, tu)
    if n <= 256:
        return {"leq": _exact_tail_leq(k, n, p, threshold), "method": "exact_integer_polynomial"}
    return {"leq": None, "method": "precision_limit", "tail_lower": str(sl), "tail_upper": str(su)}


def _lower_endpoint(k, n, threshold):
    if k == 0:
        return 0.0, {"method": "endpoint_zero", "outward_verified": True}
    # This proposal is untrusted. A corrupted/underflowed quantile can only
    # widen the returned interval: the tail inequality is always checked.
    proposal = float(beta_distribution.ppf(float(threshold), k, n - k + 1))
    if not math.isfinite(proposal) or not 0 <= proposal <= 1:
        proposal = 0.0
    denominator = 1 << DYADIC_BITS
    numerator = min(denominator, max(0, math.floor(proposal * denominator)))
    for _ in range(16):
        point = Fraction(numerator, denominator)
        evidence = binomial_tail_certificate(k, n, point, threshold)
        if evidence["leq"] is True:
            return float(point), {"outward_verified": True, "point": fraction_record(point), "tail": evidence}
        if numerator == 0:
            break
        numerator -= 1
    return 0.0, {"outward_verified": True, "method": "conservative_zero_fallback"}


def binomial_interval_certificate(successes, trials, alpha):
    for number in (successes, trials):
        if isinstance(number, (bool, np.bool_)) or not isinstance(number, (int, np.integer)):
            raise TypeError("Binomial observations must be integers")
    x, n = int(successes), int(trials)
    budget = rational(alpha)
    if not 0 <= x <= n or not 0 < budget < 1:
        raise ValueError("Invalid binomial observations or error budget")
    meta = {
        "method": "outward_clopper_pearson_by_certified_binomial_tail",
        "alpha_exact": fraction_record(budget),
        "dyadic_bits": DYADIC_BITS,
        "decimal_precision": DECIMAL_PRECISION,
        "max_trials": MAX_BINOMIAL_TRIALS,
        "coverage_scope": "conditional_on_binomial_sampling_law",
    }
    if n == 0 or n > MAX_BINOMIAL_TRIALS:
        meta.update(
            {
                "status": "empty_sample_full_interval" if n == 0 else "resource_limit_full_interval",
                "outward_verified": True,
            }
        )
        return 0.0, 1.0, meta
    lower, low_certificate = _lower_endpoint(x, n, budget / 2)
    reflected, upper_certificate = _lower_endpoint(n - x, n, budget / 2)
    # Both are on the same dyadic grid (<=44 bits), so 1-reflected is exact.
    upper = 1.0 - reflected
    meta.update(
        {
            "status": "outward_verified",
            "outward_verified": True,
            "lower": low_certificate,
            "upper_reflected": upper_certificate,
        }
    )
    return lower, upper, meta


def _fraction_rows(matrix, rhs):
    matrix = sparse.csr_matrix(matrix)
    for i, bound in enumerate(rhs):
        yield (
            {
                int(matrix.indices[j]): rational(matrix.data[j])
                for j in range(matrix.indptr[i], matrix.indptr[i + 1])
                if matrix.data[j]
            },
            rational(bound),
        )


def _feasible(x, inequalities, equalities):
    if any(value < 0 or value > 1 for value in x):
        return False

    def dot(row):
        return sum((value * x[j] for j, value in row.items()), Fraction(0))

    return all(dot(row) == rhs for row, rhs in equalities) and all(dot(row) <= rhs for row, rhs in inequalities)


def exact_primal_certificate(aub, bub, aeq, beq, proposed, *, max_dimension=MAX_EXACT_LP_DIMENSION):
    """Recover and verify an exact rational feasible point, or refuse to certify.

    Numerical active rows select a candidate basis only. Rational elimination and
    checks of EVERY supplied constraint are the authority. Failure is not proof
    that the polytope is empty; it requires abstention.
    """
    approximate = np.asarray(proposed, dtype=float)
    n = approximate.size
    if approximate.ndim != 1 or not np.all(np.isfinite(approximate)) or n > max_dimension:
        return None, {"status": "exact_primal_resource_or_input_limit"}
    inequalities = list(_fraction_rows(aub, bub))
    equalities = list(_fraction_rows(aeq, beq))
    simple = [rational(v).limit_denominator(1 << 24) for v in approximate]
    if _feasible(simple, inequalities, equalities):
        solution, method = simple, "rational_reconstruction"
    else:
        candidates = list(equalities)
        active = []
        for j, v in enumerate(approximate):
            for endpoint in (0, 1):
                distance = abs(v - endpoint)
                if distance <= 1e-7:
                    active.append((distance, {j: Fraction(1)}, Fraction(endpoint)))
        distances = np.abs(np.asarray(aub @ approximate - bub))
        for distance, (row, rhs) in zip(distances, inequalities):
            if distance <= 1e-7:
                active.append((float(distance), row, rhs))
        candidates.extend((row, rhs) for _, row, rhs in sorted(active, key=lambda item: item[0]))
        basis = {}
        for row, rhs in candidates:
            row = dict(row)
            for pivot in sorted(basis):
                coefficient = row.pop(pivot, Fraction(0))
                if not coefficient:
                    continue
                old, old_rhs = basis[pivot]
                rhs -= coefficient * old_rhs
                for j, value in old.items():
                    if j != pivot:
                        row[j] = row.get(j, Fraction(0)) - coefficient * value
                        if not row[j]:
                            del row[j]
            if not row:
                if rhs:
                    return None, {
                        "status": "exact_primal_reconstruction_failed",
                        "reason": "selected active equations are inconsistent",
                    }
                continue
            pivot = min(row)
            divisor = row[pivot]
            basis[pivot] = ({j: value / divisor for j, value in row.items()}, rhs / divisor)
            if len(basis) == n:
                break
        solution = list(simple)
        for pivot in sorted(basis, reverse=True):
            row, rhs = basis[pivot]
            solution[pivot] = rhs - sum((value * solution[j] for j, value in row.items() if j != pivot), Fraction(0))
        if not _feasible(solution, inequalities, equalities):
            return None, {
                "status": "exact_primal_reconstruction_failed",
                "reason": "rational candidate violates a supplied constraint",
            }
        method = "exact_active_basis_elimination"
    return solution, {
        "status": "exact_primal_feasible",
        "method": method,
        "coefficient_semantics": "exact_binary64_rationals_of_supplied_LP",
        "all_constraints_checked_exactly": True,
        "solution": [fraction_record(v) for v in solution],
    }
