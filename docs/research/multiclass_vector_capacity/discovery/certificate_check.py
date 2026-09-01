"""Independent exact checker for MVC v1 primal/dual interval certificates.

Deliberately does not import the discovery solver, its rank routine, or its
decision function. This checks concrete rational instances, not general theorems,
scientific availability of structural moments, or the solver's rank diagnostics.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path


class InvalidCertificate(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidCertificate(message)


def _q(value: object) -> Fraction:
    _require(type(value) in (str, int), "Certificates require rational strings or integers")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise InvalidCertificate("Invalid rational") from exc


def _vector(value: object, length: int, description: str) -> list[Fraction]:
    _require(isinstance(value, list) and len(value) == length, f"Wrong {description} length")
    return [_q(item) for item in value]


def _inner(x: list[Fraction], y: list[Fraction]) -> Fraction:
    _require(len(x) == len(y), "Inner-product shape mismatch")
    answer = Fraction(0)
    for i in range(len(x)):
        answer += x[i] * y[i]
    return answer


def check_certificate(certificate: dict) -> dict:
    """Raise InvalidCertificate on malformed data or any false bound/action."""
    try:
        return _check(certificate)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise InvalidCertificate("Missing or malformed certificate field") from exc


def check_batch(data: object) -> list[dict]:
    items = data if isinstance(data, list) else [data]
    _require(bool(items), "An empty certificate batch is not a passed gate")
    return [check_certificate(item) for item in items]


def _check(certificate: dict) -> dict:
    _require(certificate["schema"] == "mvc.exact-fiber-certificate.v1", "Unknown schema")
    _require(certificate["orientation"] == "frozen_cost_minus_candidate_cost", "Wrong benefit orientation")
    _require(certificate["comparison_scope"] == "fixed_candidate_vs_frozen_only", "No multiadapter-selection theorem is checked")
    _require(certificate["deployment_selection"] == "NOT_IMPLEMENTED", "Pairwise certificate cannot promote a deployment selection")
    problem = certificate["problem"]
    k = problem["classes"]
    _require(type(k) is int and k >= 3, "K must be an integer >= 3")
    _require(isinstance(problem["name"], str) and bool(problem["name"].strip()), "Missing name")
    _require(isinstance(problem["masses"], list) and bool(problem["masses"]), "No strata")
    m = len(problem["masses"])
    d = m * k
    masses = _vector(problem["masses"], m, "mass")
    _require(all(x >= 0 for x in masses) and sum(masses) == 1, "Invalid stratum probabilities")
    _require(isinstance(problem["candidates"], list) and bool(problem["candidates"]), "No candidates")
    candidate = certificate["candidate"]
    _require(type(candidate) is int and 0 <= candidate < len(problem["candidates"]), "Invalid candidate index")

    def cost_matrix(raw: object) -> list[list[Fraction]]:
        _require(isinstance(raw, list) and len(raw) == m, "Wrong cost stratum count")
        matrix = [_vector(row, k, "class cost") for row in raw]
        _require(all(0 <= value <= 1 for row in matrix for value in row), "Cost outside [0,1]")
        return matrix

    frozen = cost_matrix(problem["frozen"])
    candidates = [cost_matrix(raw) for raw in problem["candidates"]]
    costs = candidates[candidate]
    restrictions = problem["restrictions"]
    _require(isinstance(restrictions, list), "Restrictions must be a list")
    a = [_vector(row, d, "observable row") for row in restrictions]
    values = _vector(problem["values"], len(a), "observable value")
    for key in ("justifications", "ablations"):
        _require(isinstance(problem[key], list) and len(problem[key]) == len(a), f"Missing {key}")
        _require(all(isinstance(value, str) and bool(value.strip()) for value in problem[key]), f"Empty {key}")
    _require(problem["structural_availability"] == "mathematical_toy_only_not_established_from_unlabeled_data", "This initial checker has no validated natural-data restriction protocol")

    # Independently reconstruct the product-simplex affine hull and cost contrast.
    h = [[Fraction(int(i // k == s)) for i in range(d)] for s in range(m)] + a
    rhs = [Fraction(1)] * m + values
    objective = [masses[s] * (frozen[s][y] - costs[s][y]) for s in range(m) for y in range(k)]
    lower, upper = _q(certificate["lower"]), _q(certificate["upper"])
    _require(lower <= upper, "Reversed interval")

    for which, bound in (("minimum", lower), ("maximum", upper)):
        witness = certificate[which]
        point = _vector(witness["point"], d, "primal point")
        dual = _vector(witness["dual"], len(h), "dual")
        _require(all(x >= 0 for x in point), "Negative conditional probability")
        _require(all(sum(point[s * k:(s + 1) * k]) == 1 for s in range(m)), "Conditional probabilities do not normalize")
        _require(all(_inner(row, point) == value for row, value in zip(a, values)), "Point violates observable equivalence")
        # Separate expected-cost sums also guard the interpretation of the objective.
        frozen_risk = sum(masses[s] * sum(frozen[s][y] * point[s * k + y] for y in range(k)) for s in range(m))
        candidate_risk = sum(masses[s] * sum(costs[s][y] * point[s * k + y] for y in range(k)) for s in range(m))
        _require(frozen_risk - candidate_risk == bound == _inner(objective, point), "Primal objective/orientation mismatch")
        for i in range(d):
            lhs = sum(h[row][i] * dual[row] for row in range(len(h)))
            if which == "minimum":
                _require(lhs <= objective[i], "Infeasible lower-bound dual")
            else:
                _require(lhs >= objective[i], "Infeasible upper-bound dual")
        _require(_inner(rhs, dual) == bound, "Nonzero primal-dual gap")

    expected = "ADAPT" if lower > 0 else ("FREEZE" if upper < 0 else "ABSTAIN")
    _require(certificate["action"] == expected, "Action violates strict interval frontier")
    _require(type(certificate["point_identified_at_realized_fiber"]) is bool, "Point-identification field must be Boolean")
    _require(certificate["point_identified_at_realized_fiber"] == (lower == upper), "Point-identification claim mismatch")
    return {"name": problem["name"], "candidate": candidate, "lower": str(lower), "upper": str(upper), "action": expected, "comparison_scope": "fixed_candidate_vs_frozen_only", "status": "EXACT_PRIMAL_DUAL_CHECKED"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args()
    data = json.loads(args.certificate.read_text())
    results = check_batch(data)
    print(json.dumps({"checked": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
