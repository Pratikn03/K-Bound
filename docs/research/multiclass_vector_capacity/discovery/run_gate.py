"""Generate or reproduce the bounded exact discovery corpus and local receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction as Q
from pathlib import Path

from certificate_check import check_certificate
from exact_oracle import EmptyFiber, certify, dot, minimum_allowed_moments, null_basis, rank
from finite_cases import bounded_grid, coordinate, named_cases

TRACK = Path(__file__).resolve().parents[1]


class GateFailure(ValueError):
    """A named witness or generated report no longer supports its stated claim."""


def require(condition: bool, message: str) -> None:
    # Never use assert for a scientific gate: python -O would remove it.
    if not condition:
        raise GateFailure(message)


def _bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def generate() -> tuple[list[dict], dict]:
    certificates = []
    rejected = []
    cases = named_cases() + bounded_grid()
    require(len({problem.name for problem in cases}) == len(cases), "Duplicate case names")
    for problem in cases:
        for candidate in range(len(problem.candidates)):
            try:
                certificate = certify(problem, candidate)
            except EmptyFiber as exc:
                if not problem.name.endswith("_inconsistent"):
                    raise
                rejected.append({"name": problem.name, "candidate": candidate, "reason": str(exc)})
                continue
            if problem.name.endswith("_inconsistent"):
                raise GateFailure("Inconsistent fiber was incorrectly certified")
            check_certificate(certificate)
            certificates.append(certificate)
    witness, supplement, boundary_face = named_cases()[:3]
    by_name = {(item["problem"]["name"], item["candidate"]): item for item in certificates}
    positive_certificate = by_name[(witness.name, 0)]
    face_certificate = by_name[(boundary_face.name, 0)]
    direction = (Q(0), Q(1), Q(-1))
    require(all(dot(row, direction) == 0 for row in witness.equalities), "T5 direction is not observable-null")
    contrast_change = dot(witness.contrast(0), direction)
    require(contrast_change == Q(-1, 2), "T5 witness contrast changed")
    require((positive_certificate["lower"], positive_certificate["upper"], positive_certificate["action"]) == ("1/5", "3/5", "ADAPT"), "T5 witness no longer matches the declared positive nonpoint interval")
    require(not positive_certificate["point_identified_at_realized_fiber"], "T5 witness became point identified")
    require(face_certificate["lower"] == face_certificate["upper"] == "0", "T2 boundary witness no longer has zero identified benefit")
    require(face_certificate["minimum"]["point"] == face_certificate["maximum"]["point"] == ["0", "0", "1"], "T2 boundary witness no longer matches the declared vertex")
    require(face_certificate["discovery_diagnostics_not_checked_by_lp_checker"]["vertices"] == 1, "T2 witness is no longer a singleton fiber")
    rank_increment = rank([*supplement.equalities, supplement.contrast(0)], supplement.dimension) - rank(supplement.equalities, supplement.dimension)
    admissible_minimum = minimum_allowed_moments(supplement, [coordinate(3, i) for i in range(3)])
    require((rank_increment, admissible_minimum) == (1, 2), "T3 primitive-moment counterexample no longer holds")
    # The two asymmetric worlds share q and the same (empty) restrictions.
    rare = named_cases()[6]
    worlds = [tuple([Q(1, 3)] * 3 + [Q(5, 16), Q(7, 16), Q(1, 4)]), tuple([Q(1, 3)] * 3 + [Q(3, 16), Q(9, 16), Q(1, 4)])]
    for eta in worlds:
        require(len(eta) == rare.dimension and all(x >= 0 for x in eta), "Rare-stratum world has invalid conditional probabilities")
        require(all(dot(row, eta) == value for row, value in zip(rare.equalities, rare.rhs)), "Rare-stratum world is outside the declared observable fiber")
    benefits = [dot(rare.contrast(0), eta) for eta in worlds]
    require(benefits == [Q(3, 512), Q(-3, 512)], "Rare-stratum world benefits changed")
    sources = [*sorted((TRACK / "discovery").glob("*.py")), TRACK / "protocols/initial_exact_suite.json"]
    report = {
        "schema": "mvc.initial-exact-gate.v1",
        "status": "BOUNDED_EXACT_DISCOVERY_PASSED_NOT_THEOREM_OR_PROMOTION_PASS",
        "arithmetic": "exact rational",
        "problems": len(cases),
        "certificates_independently_checked": len(certificates),
        "extrema_with_primal_and_dual_certificates": 2 * len(certificates),
        "expected_empty_fiber_rejections": rejected,
        "counterexamples": {
            "T5_without_crossing_condition": {"direction": list(map(str, direction)), "observable_null": True, "contrast_on_direction": str(contrast_change), "benefit_interval": [positive_certificate["lower"], positive_certificate["upper"]], "point_identified_at_realized_fiber": positive_certificate["point_identified_at_realized_fiber"], "conclusion": "Null-space uncertainty alone does not imply opposite strict decisions"},
            "T3_unrestricted_rank_as_admissible_minimum": {"unrestricted_rank_increment": rank_increment, "admissible_catalog": "three primitive class-probability coordinate moments", "admissible_minimum": admissible_minimum, "conclusion": "Rank increment is not generally the minimum under the specified restricted supplement catalog"},
            "T2_fixed_fiber_vs_uniform": {"unique_vertex": face_certificate["minimum"]["point"], "benefit_interval": [face_certificate["lower"], face_certificate["upper"]], "point_identified_at_realized_fiber": face_certificate["point_identified_at_realized_fiber"], "surviving_null_directions": [list(map(str, x)) for x in null_basis(boundary_face.equalities, 3)], "conclusion": "A realized simplex face can identify the benefit despite surviving affine contrast directions"},
        },
        "rare_stratum_candidate_family": {"worlds": [list(map(str, x)) for x in worlds], "benefits": list(map(str, benefits)), "status": "EXACT_WORLDS_ONLY_NO_LABEL_COMPLEXITY_THEOREM"},
        "source_sha256": {str(path.relative_to(TRACK)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "certificate_corpus_sha256": hashlib.sha256(_bytes(certificates)).hexdigest(),
        "fresh_checkout_gate": "NOT_RUN",
        "prospective_experiment": "NOT_RUN",
        "program_required_statistical_mutations": "NOT_IMPLEMENTED",
        "lean_status": "SEE_THEOREM_LEDGER_NOT_INFERRED_FROM_THIS_REPORT",
        "novelty_status": "UNRESOLVED_PARTIAL_LITERATURE_COLLISION",
        "existing_kbound_integration": "NOT_PERMITTED",
    }
    return certificates, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Recompute and compare existing generated files without writing")
    args = parser.parse_args()
    certificates, report = generate()
    artifacts = {
        TRACK / "artifacts/initial_exact_certificates.json": _bytes(certificates),
        TRACK / "reports/initial_exact_gate.json": _bytes(report),
    }
    for path, content in artifacts.items():
        if args.check:
            if not path.is_file() or path.read_bytes() != content:
                raise SystemExit(f"FAIL: stale or missing generated file: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    print(json.dumps({"status": "PASS", "mode": "check" if args.check else "generate", "problems": report["problems"], "certificates": len(certificates), "empty_fiber_candidate_rejections": len(report["expected_empty_fiber_rejections"]), "scope": "bounded exact discovery only"}))


if __name__ == "__main__":
    main()
