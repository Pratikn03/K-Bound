"""Exact finite-instance and deliberate-corruption tests; no theorem promotion."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from unittest.mock import patch
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path

TRACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK / "discovery"))

from certificate_check import InvalidCertificate, check_batch, check_certificate
from exact_oracle import ProtocolFailure, SearchLimit, certify, dot, minimum_allowed_moments, null_basis, rank, rational, strict_action, unique_solution, vertices
from finite_cases import bounded_grid, coordinate, named_cases
import run_gate


class LinearAlgebraTests(unittest.TestCase):
    def test_exact_rank_with_redundant_rows(self):
        self.assertEqual(rank(((1, 2, 3), (2, 4, 6)), 3), 1)

    def test_empty_rank(self):
        self.assertEqual(rank((), 3), 0)

    def test_rational_solution(self):
        self.assertEqual(unique_solution(((1, 1), (1, -1)), (Q(1), Q(1, 3)), 2), (Q(2, 3), Q(1, 3)))

    def test_inconsistent_solution(self):
        self.assertIsNone(unique_solution(((1, 1), (2, 2)), (Q(1), Q(3)), 2))

    def test_underdetermined_solution(self):
        self.assertIsNone(unique_solution(((1, 1),), (Q(1),), 2))

    def test_null_vectors(self):
        rows = ((Q(1), Q(1), Q(1)), (Q(1), Q(0), Q(0)))
        basis = null_basis(rows, 3)
        self.assertEqual(len(basis), 1)
        self.assertTrue(all(dot(row, direction) == 0 for row in rows for direction in basis))

    def test_simplex_affine_hull_mutation_is_detected(self):
        constant = (Q(1), Q(1), Q(1))
        self.assertEqual(rank((constant,), 3) - rank((), 3), 1)  # Incorrect omitted-hull answer.
        self.assertEqual(rank((constant, constant), 3) - rank((constant,), 3), 0)

    def test_dimension_errors_fail_loudly(self):
        with self.assertRaises(ProtocolFailure):
            rank(((1, 2),), 3)
        with self.assertRaises(ProtocolFailure):
            dot((Q(1),), ())


class InputTests(unittest.TestCase):
    def test_float_rejected(self):
        with self.assertRaises(ProtocolFailure):
            rational(0.5)

    def test_boolean_rejected(self):
        with self.assertRaises(ProtocolFailure):
            rational(True)

    def test_zero_denominator_rejected(self):
        with self.assertRaises(ProtocolFailure):
            rational("1/0")

    def test_negative_mass_rejected(self):
        with self.assertRaises(ProtocolFailure):
            replace(named_cases()[0], masses=(Q(-1),))

    def test_nonunit_mass_rejected(self):
        with self.assertRaises(ProtocolFailure):
            replace(named_cases()[0], masses=(Q(1, 2),))

    def test_cost_range_rejected(self):
        with self.assertRaises(ProtocolFailure):
            replace(named_cases()[0], frozen=((Q(0), Q(1), Q(2)),))

    def test_missing_justification_rejected(self):
        with self.assertRaises(ProtocolFailure):
            replace(named_cases()[0], justifications=())

    def test_missing_ablation_rejected(self):
        with self.assertRaises(ProtocolFailure):
            replace(named_cases()[0], ablations=("",))

    def test_cost_shape_rejected(self):
        with self.assertRaises(ProtocolFailure):
            replace(named_cases()[0], frozen=((Q(0), Q(1)),))

    def test_invalid_candidate_rejected(self):
        for candidate in (-1, 99, True):
            with self.assertRaises(ProtocolFailure):
                certify(named_cases()[0], candidate)

    def test_invalid_name_rejected_before_emission(self):
        for name in (42, "   ", True):
            with self.assertRaises(ProtocolFailure):
                replace(named_cases()[0], name=name)

    def test_invalid_enumeration_budget_rejected(self):
        for limit in (float("nan"), float("inf"), 2.0, -1, 0, True):
            with self.assertRaises(ProtocolFailure):
                certify(named_cases()[0], max_subsets=limit)


class FiberTests(unittest.TestCase):
    def test_nonpoint_identified_but_positive(self):
        certificate = certify(named_cases()[0])
        self.assertEqual((certificate["lower"], certificate["upper"], certificate["action"]), ("1/5", "3/5", "ADAPT"))
        self.assertFalse(certificate["point_identified_at_realized_fiber"])
        check_certificate(certificate)

    def test_null_direction_does_not_supply_zero_crossing(self):
        problem = named_cases()[0]
        direction = (Q(0), Q(1), Q(-1))
        self.assertTrue(all(dot(row, direction) == 0 for row in problem.equalities))
        self.assertEqual(dot(problem.contrast(0), direction), Q(-1, 2))
        self.assertTrue(all(dot(problem.contrast(0), eta) > 0 for eta in vertices(problem)))

    def test_structural_ablation_changes_sign_identification(self):
        problem = replace(named_cases()[0], restrictions=(), values=(), justifications=(), ablations=())
        certificate = certify(problem)
        self.assertEqual((certificate["lower"], certificate["upper"], certificate["action"]), ("-1", "1", "ABSTAIN"))

    def test_realized_face_distinct_from_uniform_affine(self):
        problem = named_cases()[2]
        certificate = certify(problem)
        self.assertTrue(certificate["point_identified_at_realized_fiber"])
        self.assertEqual(certificate["lower"], "0")
        self.assertEqual(certificate["discovery_diagnostics_not_checked_by_lp_checker"]["unrestricted_linear_rank_increment"], 1)

    def test_zero_lower_endpoint_abstains(self):
        self.assertEqual(certify(named_cases()[3])["action"], "ABSTAIN")

    def test_zero_upper_endpoint_abstains(self):
        self.assertEqual(certify(named_cases()[4])["action"], "ABSTAIN")

    def test_opposite_worlds_are_available_when_qualified(self):
        certificate = certify(named_cases()[5])
        self.assertEqual((certificate["lower"], certificate["upper"]), ("-1", "1"))
        check_certificate(certificate)

    def test_empty_affine_fiber_is_protocol_failure(self):
        problem = next(p for p in bounded_grid() if p.name.endswith("inconsistent"))
        with self.assertRaisesRegex(ProtocolFailure, "Empty fiber"):
            certify(problem)

    def test_nonnegative_infeasibility_is_protocol_failure(self):
        problem = replace(named_cases()[0], values=(Q(2),))
        with self.assertRaisesRegex(ProtocolFailure, "Empty fiber"):
            certify(problem)

    def test_search_limit_not_abstention_or_certificate(self):
        with self.assertRaises(SearchLimit):
            certify(named_cases()[6], max_subsets=1)

    def test_zero_mass_stratum_has_no_contrast(self):
        problem = next(p for p in bounded_grid() if p.masses == (Q(1), Q(0)) and p.name.endswith("none"))
        self.assertEqual(problem.contrast(0)[problem.classes:], (Q(0),) * problem.classes)
        check_certificate(certify(problem))

    def test_cost_swap_reverses_interval(self):
        problem = named_cases()[0]
        reverse = replace(problem, frozen=problem.candidates[0], candidates=(problem.frozen,))
        certificate = certify(reverse)
        self.assertEqual((certificate["lower"], certificate["upper"], certificate["action"]), ("-3/5", "-1/5", "FREEZE"))
        check_certificate(certificate)

    def test_strict_frontier(self):
        self.assertEqual(strict_action(Q(1, 10**6), Q(1)), "ADAPT")
        self.assertEqual(strict_action(Q(-1), Q(-1, 10**6)), "FREEZE")
        self.assertEqual(strict_action(Q(0), Q(0)), "ABSTAIN")
        with self.assertRaises(ProtocolFailure):
            strict_action(Q(1), Q(-1))


class SupplementTests(unittest.TestCase):
    def test_admissible_class_moments_require_two_not_rank_one(self):
        problem = named_cases()[1]
        catalog = [coordinate(3, i) for i in range(3)]
        self.assertEqual(minimum_allowed_moments(problem, catalog), 2)
        self.assertEqual(rank([*problem.equalities, problem.contrast(0)], 3) - rank(problem.equalities, 3), 1)

    def test_every_single_primitive_moment_insufficient(self):
        for i in range(3):
            self.assertIsNone(minimum_allowed_moments(named_cases()[1], [coordinate(3, i)]))

    def test_constant_benefit_needs_no_supplement(self):
        problem = replace(named_cases()[1], frozen=((Q(1, 2), Q(1, 2), Q(1, 2)),))
        self.assertEqual(minimum_allowed_moments(problem, ()), 0)

    def test_forbidden_answer_oracle_is_algebraically_sufficient_not_admissible(self):
        problem = named_cases()[1]
        # This is precisely why algebraic attainment cannot establish scientific admissibility.
        self.assertEqual(minimum_allowed_moments(problem, [problem.contrast(0)]), 1)

    def test_catalog_budget_and_shape(self):
        with self.assertRaises(SearchLimit):
            minimum_allowed_moments(named_cases()[1], [coordinate(3, 0)] * 17)
        with self.assertRaises(ProtocolFailure):
            minimum_allowed_moments(named_cases()[1], [(Q(1),)])


class MutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid = certify(named_cases()[0])

    def changed(self):
        return copy.deepcopy(self.valid)

    def test_benefit_orientation_mutation(self):
        value = self.changed()
        value["orientation"] = "candidate_cost_minus_frozen_cost"
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_non_strict_adapt_mutation(self):
        value = certify(named_cases()[3])
        value["action"] = "ADAPT"
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_non_strict_freeze_mutation(self):
        value = certify(named_cases()[4])
        value["action"] = "FREEZE"
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_simplex_normalization_mutation(self):
        value = self.changed()
        value["minimum"]["point"][1] = "1"
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_nonnegativity_mutation(self):
        value = self.changed()
        value["minimum"]["point"] = ["1/5", "1", "-1/5"]
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_observable_equivalence_mutation(self):
        value = self.changed()
        value["minimum"]["point"] = ["0", "4/5", "1/5"]
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_lower_dual_mutation(self):
        value = self.changed()
        value["minimum"]["dual"][0] = "99"
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_upper_dual_mutation(self):
        value = self.changed()
        value["maximum"]["dual"][0] = "-99"
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_wrong_primal_objective_mutation(self):
        value = self.changed()
        value["lower"] = "1/4"
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_point_identification_claim_mutation(self):
        value = self.changed()
        value["point_identified_at_realized_fiber"] = True
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_cost_orientation_semantic_mutation(self):
        value = self.changed()
        value["problem"]["frozen"], value["problem"]["candidates"][0] = value["problem"]["candidates"][0], value["problem"]["frozen"]
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_float_certificate_mutation(self):
        value = self.changed()
        value["lower"] = 0.2
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_missing_field_mutation(self):
        value = self.changed()
        del value["maximum"]["dual"]
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_invalid_cost_of_unselected_candidate_rejected(self):
        value = self.changed()
        value["problem"]["candidates"].append([["2", "0", "0"]])
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_unvalidated_scientific_promotion_rejected(self):
        value = self.changed()
        value["problem"]["structural_availability"] = "established_from_target_observables"
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_multiadapter_scope_promotion_rejected(self):
        value = self.changed()
        value["comparison_scope"] = "strictly_best_of_all_candidates"
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)

    def test_deployment_selection_promotion_rejected(self):
        value = self.changed()
        value["deployment_selection"] = "ADAPT"
        with self.assertRaises(InvalidCertificate):
            check_certificate(value)


class CorpusTests(unittest.TestCase):
    def test_empty_certificate_batch_rejected(self):
        with self.assertRaises(InvalidCertificate):
            check_batch([])

    def test_declared_grid_coverage(self):
        cases = named_cases() + bounded_grid()
        self.assertEqual(len(cases), 67)
        self.assertEqual({p.classes for p in cases}, {3, 4, 5})
        self.assertEqual({p.strata for p in cases}, {1, 2})
        self.assertEqual({len(p.candidates) for p in cases}, {1, 2})
        self.assertEqual(sum(p.name.endswith("inconsistent") for p in cases), 12)

    def test_generated_corpus_independently_checks(self):
        path = TRACK / "artifacts/initial_exact_certificates.json"
        self.assertTrue(path.is_file(), "Run discovery/run_gate.py first")
        corpus = json.loads(path.read_text())
        self.assertEqual(len(corpus), 103)
        for certificate in corpus:
            with self.subTest(name=certificate["problem"]["name"], candidate=certificate["candidate"]):
                check_certificate(certificate)

    def test_rare_stratum_worlds_same_observables_opposite_costs(self):
        problem = named_cases()[6]
        common = (Q(1, 3),) * 3
        plus = common + (Q(5, 16), Q(7, 16), Q(1, 4))
        minus = common + (Q(3, 16), Q(9, 16), Q(1, 4))
        for eta in (plus, minus):
            self.assertTrue(all(dot(row, eta) == value for row, value in zip(problem.equalities, problem.rhs)))
            self.assertTrue(all(x > 0 for x in eta))
        self.assertEqual(dot(problem.contrast(0), plus), Q(3, 512))
        self.assertEqual(dot(problem.contrast(0), minus), Q(-3, 512))

    def test_abstention_breaks_unqualified_error_lower_bound(self):
        adapt_probability, freeze_probability = Q(0), Q(0)
        self.assertEqual(max(adapt_probability, freeze_probability), 0)
        self.assertLess(max(adapt_probability, freeze_probability), Q(1, 2))
        # This finite arithmetic observation is not a promoted minimax theorem.


class ReportMutationTests(unittest.TestCase):
    def expect_named_mutation_failure(self, index, **changes):
        cases = named_cases()
        cases[index] = replace(cases[index], **changes)
        with patch.object(run_gate, "named_cases", return_value=cases), patch.object(run_gate, "bounded_grid", return_value=[]):
            with self.assertRaises(run_gate.GateFailure):
                run_gate.generate()

    def test_no_hardcoded_positive_interval_after_cost_mutation(self):
        self.expect_named_mutation_failure(0, candidates=(((Q(1, 2), Q(0), Q(0)),),))

    def test_no_hardcoded_vertex_after_observable_mutation(self):
        self.expect_named_mutation_failure(2, values=(Q(1, 2),))

    def test_rare_worlds_must_be_feasible(self):
        self.expect_named_mutation_failure(
            6, restrictions=(coordinate(6, 3),), values=(Q(1, 2),),
            justifications=("Deliberately incompatible extra row",), ablations=("Remove row",),
        )

    def test_optimized_python_cannot_remove_scientific_gate(self):
        # compile(..., optimize=1) applies the same assert removal as python -O.
        namespace = {"__name__": "optimized_run_gate_probe", "__file__": run_gate.__file__}
        exec(compile(Path(run_gate.__file__).read_text(), run_gate.__file__, "exec", optimize=1), namespace)
        cases = named_cases()
        cases[0] = replace(cases[0], frozen=((Q(1, 2),) * 3,), candidates=(((Q(0),) * 3,),))
        namespace["named_cases"] = lambda: cases
        namespace["bounded_grid"] = lambda: []
        with self.assertRaises(namespace["GateFailure"]):
            namespace["generate"]()


if __name__ == "__main__":
    unittest.main()
