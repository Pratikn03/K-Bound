"""The release must distinguish real kernel checks from a static name inventory."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from docs.research.kbound.formal import formal_audit as audit


def test_audit_floor_capstones_are_in_the_kernel_audit_contract() -> None:
    """Omitting the new module must not silently remove its kernel checks."""
    expected = {
        "AuditFloor": {
            "measurable_audit_floor",
            "fibrewise_randomized_audit_floor",
            "constant_fibreRadius_valid",
            "constant_fibreRadius_audit_valid",
            "audit_floor_frontier_inert",
            "fibreRadius_eq_of_bound_and_witness",
        },
        "AuditFloorCorrectness": {
            "correctness_fibreRadius_eq_beta",
            "correctness_target_fibreRadius_eq_beta",
        },
    }
    for module, names in expected.items():
        assert names <= set(audit.FOUNDATION_THEOREMS.get(module, []))
        for name in names:
            assert audit.VERIFIED_THEOREMS.count(name) == 1


def test_registered_capstones_are_unique_and_imported() -> None:
    assert len(audit.LEGACY_CORE_THEOREMS) == 65
    assert len(audit.VERIFIED_THEOREMS) == len(set(audit.VERIFIED_THEOREMS))
    assert audit.theorem_map_checks() == []
    assert audit.scan_for_forbidden_tokens() == []
    entry = (audit.ROOT / "KBound.lean").read_text(encoding="utf-8")
    for module in audit.FOUNDATION_THEOREMS:
        prefix = "KBound" if module in {"PaperDecisionAlgebra", "WeightedHelpful", "FeatureRank"} else "KBound.Probability"
        assert f"import {prefix}.{module}\n" in entry
        assert (audit.ROOT / Path(*prefix.split(".")) / f"{module}.lean").is_file()


def test_probability_interfaces_remain_in_the_kernel_audit_contract() -> None:
    """Removing imports and registry entries together must not hide either gap."""
    expected = {
        "RandomizedActionLaw": {
            "abstention_lower_bound", "common_law_forces_abstention",
            "randomized_rules_force_abstention", "zero_errors_full_abstention",
        },
        "ExtendedRadiusCertificate": {
            "action_top", "action_finite", "action_ofReal", "measurableSet_coverage",
            "false_direction_subset_failure", "directional_error_probability",
            "measurableSet_falseDirection",
        },
    }
    for module, names in expected.items():
        qualified = {f"{module}.{name}" for name in names}
        assert qualified <= set(audit.FOUNDATION_THEOREMS.get(module, []))
        assert all(audit.VERIFIED_THEOREMS.count(name) == 1 for name in qualified)
    assert len(audit.VERIFIED_THEOREMS) == 315


def test_joint_kernel_score_is_exhaustively_registered() -> None:
    source = audit.strip_lean_comments((audit.ROOT / "KBound/Probability/JointKernelScore.lean").read_text())
    declared = {f"JointKernelScore.{name}" for name in re.findall(
        r"^theorem\s+(\w+)", source, re.MULTILINE)}
    assert len(declared) == 8
    assert set(audit.FOUNDATION_THEOREMS["JointKernelScore"]) == declared
    assert all(audit.VERIFIED_THEOREMS.count(name) == 1 for name in declared)


def test_actual_world_frontier_is_exhaustively_registered() -> None:
    expected = {
        "fieldWorld_fst", "fieldWorld_residual", "actual_frontier_adapt_iff",
        "actual_frontier_freeze_iff", "actual_identified_interval", "actual_class_nonempty",
        "actual_closed_band_zero_target", "actual_zero_margin_budget",
        "actual_strict_direction_iff", "actual_pointwise_maximal_rule",
    }
    qualified = {f"ActualWorldFrontier.{name}" for name in expected}
    assert set(audit.FOUNDATION_THEOREMS.get("ActualWorldFrontier", [])) == qualified
    source = audit.strip_lean_comments((audit.ROOT / "KBound/Probability/ActualWorldFrontier.lean").read_text())
    declared = set(re.findall(r"^theorem\s+(\w+)", source, re.MULTILINE))
    assert declared == expected
    assert all(audit.VERIFIED_THEOREMS.count(name) == 1 for name in qualified)


def test_actual_world_evidence_is_exhaustively_registered() -> None:
    expected = {
        "augmentedLaw_eq", "iidObservation_realization", "augmentedLaw_iid",
        "actual_open_band_field_witnesses", "actual_open_band_matched_targets",
        "actual_closed_band_matched_zero", "actual_augmented_fibre_strict_direction_iff",
        "actual_closed_band_abstention",
    }
    qualified = {f"ActualWorldEvidence.{name}" for name in expected}
    assert set(audit.FOUNDATION_THEOREMS.get("ActualWorldEvidence", [])) == qualified
    source = audit.strip_lean_comments((audit.ROOT / "KBound/Probability/ActualWorldEvidence.lean").read_text())
    declared = set(re.findall(r"^theorem\s+(\w+)", source, re.MULTILINE))
    assert declared == expected
    assert all(audit.VERIFIED_THEOREMS.count(name) == 1 for name in qualified)


def test_actual_world_subclass_is_exhaustively_registered() -> None:
    expected = {
        "zeroWorld_residual", "zeroWorld_benefit", "zeroWorld_admissible_iff",
        "tiltWorld_residual", "tiltWorld_benefit", "tiltWorld_admissible",
        "full_class_tilt_witnesses", "subclass_closed_band_obstructions",
        "subclass_strict_direction_iff", "subclass_adapt_iff_of_nonempty",
        "subclass_freeze_iff_of_nonempty", "subclass_pointwise_maximal_rule",
        "subclass_augmented_fibre_strict_direction_iff", "subclass_closed_band_abstention",
    }
    qualified = {f"ActualWorldSubclass.{name}" for name in expected}
    assert set(audit.FOUNDATION_THEOREMS.get("ActualWorldSubclass", [])) == qualified
    source = audit.strip_lean_comments((audit.ROOT / "KBound/Probability/ActualWorldSubclass.lean").read_text())
    declared = set(re.findall(r"^theorem\s+(\w+)", source, re.MULTILINE))
    assert declared == expected
    assert all(audit.VERIFIED_THEOREMS.count(name) == 1 for name in qualified)


def test_actual_fibre_radius_is_exhaustively_registered() -> None:
    expected = {
        "actual_abs_residual_attained", "actual_class_radius",
        "actual_augmented_fibre_eq_class", "actual_augmented_fibre_radius",
    }
    qualified = {f"ActualFibreRadius.{name}" for name in expected}
    assert set(audit.FOUNDATION_THEOREMS.get("ActualFibreRadius", [])) == qualified
    source = audit.strip_lean_comments((audit.ROOT / "KBound/Probability/ActualFibreRadius.lean").read_text())
    assert set(re.findall(r"^theorem\s+(\w+)", source, re.MULTILINE)) == expected
    assert all(audit.VERIFIED_THEOREMS.count(name) == 1 for name in qualified)


def test_feature_rank_is_exhaustively_registered() -> None:
    expected = {
        "encode_relations", "encode_injective", "mem_range_iff",
        "schema_finrank", "rowspan_finrank_le",
    }
    qualified = {f"FeatureRank.{name}" for name in expected}
    assert set(audit.FOUNDATION_THEOREMS.get("FeatureRank", [])) == qualified
    source = audit.strip_lean_comments((audit.ROOT / "KBound/FeatureRank.lean").read_text())
    assert set(re.findall(r"^theorem\s+(\w+)", source, re.MULTILINE)) == expected
    assert all(audit.VERIFIED_THEOREMS.count(name) == 1 for name in qualified)


@pytest.mark.parametrize("module,names", [
    ("ConditionalExposure", {
        "event_ratio_le_min", "false_adapt_ratio_le_min", "conditional_false_adapt_le_min",
    }),
    ("IntegrableProxy", {
        "expectation_decomposition", "positive_of_residual_bound", "negative_of_residual_bound",
    }),
    ("InformationRefinement", {
        "zero_disagreement_risks_eq", "zero_disagreement_populationBenefit", "fibreRadius_mono",
        "observable_fibre_refinement", "observable_fibreRadius_mono", "deterministic_observable_fibre_eq",
    }),
])
def test_report_probability_bridges_cannot_disappear_from_kernel_audit(module, names) -> None:
    """Dropping a report bridge from the registry must not leave a green audit."""
    qualified = {f"{module}.{name}" for name in names}
    assert set(audit.FOUNDATION_THEOREMS.get(module, [])) == qualified
    assert all(audit.VERIFIED_THEOREMS.count(name) == 1 for name in qualified)
    assert audit.theorem_map_checks() == []


def test_weighted_helpful_is_exhaustively_registered() -> None:
    expected = {"helpful_weighted_sum", "helpful_weighted_mean", "helpful_weighted_regret_mean"}
    qualified = {f"WeightedHelpful.{name}" for name in expected}
    assert set(audit.FOUNDATION_THEOREMS.get("WeightedHelpful", [])) == qualified
    source = audit.strip_lean_comments((audit.ROOT / "KBound/WeightedHelpful.lean").read_text())
    assert set(re.findall(r"^theorem\s+(\w+)", source, re.MULTILINE)) == expected
    assert all(audit.VERIFIED_THEOREMS.count(name) == 1 for name in qualified)


def test_risk_alignment_is_exhaustively_registered() -> None:
    expected = {
        "strict_implies_aligned", "aligned_of_nonnegative", "not_strict_of_zero",
        "actual_positive_boundary_nonnegative", "actual_boundary_risk_aligned_not_strict",
    }
    qualified = {f"RiskAlignment.{name}" for name in expected}
    assert set(audit.FOUNDATION_THEOREMS.get("RiskAlignment", [])) == qualified
    source = audit.strip_lean_comments((audit.ROOT / "KBound/Probability/RiskAlignment.lean").read_text())
    assert set(re.findall(r"^theorem\s+(\w+)", source, re.MULTILINE)) == expected
    assert all(audit.VERIFIED_THEOREMS.count(name) == 1 for name in qualified)


def test_evidence_transport_is_exhaustively_registered() -> None:
    source = audit.strip_lean_comments((audit.ROOT / "KBound/Probability/EvidenceTransport.lean").read_text())
    declared = {f"EvidenceTransport.{name}" for name in re.findall(
        r"^theorem\s+(\w+)", source, re.MULTILINE)}
    assert len(declared) == 8
    assert set(audit.FOUNDATION_THEOREMS["EvidenceTransport"]) == declared
    assert all(audit.VERIFIED_THEOREMS.count(name) == 1 for name in declared)


def test_literal_conformal_construction_is_exhaustively_registered() -> None:
    source = audit.strip_lean_comments((audit.ROOT / "KBound/Probability/ExactConformal.lean").read_text())
    declared = {f"ExactConformal.{name}" for name in re.findall(
        r"^theorem\s+(\w+)", source, re.MULTILINE)}
    assert len(declared) == 17
    assert set(audit.FOUNDATION_THEOREMS["ExactConformal"]) == declared
    assert all(audit.VERIFIED_THEOREMS.count(name) == 1 for name in declared)


def test_reviewed_paper_proofs_are_exhaustively_registered() -> None:
    """Dropping a reviewed theorem/helper or its module must fail the inventory."""
    expected_counts = {
        "JointTargetReduction": 9, "PaperCounterexamples": 17,
        "DependentSignFlip": 20, "IndependentSignFlip": 1,
        "PaperDecisionAlgebra": 13,
    }
    for module, count in expected_counts.items():
        prefix = Path("KBound") if module in {"PaperDecisionAlgebra", "WeightedHelpful", "FeatureRank"} else Path("KBound/Probability")
        source = audit.strip_lean_comments((audit.ROOT / prefix / f"{module}.lean").read_text())
        declared = {f"{module}.{name}" for name in re.findall(
            r"^(?:theorem|lemma)\s+(\w+)", source, re.MULTILINE)}
        assert len(declared) == count
        assert set(audit.FOUNDATION_THEOREMS[module]) == declared
        assert all(audit.VERIFIED_THEOREMS.count(name) == 1 for name in declared)


def test_five_foundations_do_not_silently_close_the_false_sixth_extension() -> None:
    assert len(audit.FOUNDATION_LAYERS) == 6
    assert [row["status"] for row in audit.FOUNDATION_LAYERS].count("MECHANIZED_WITH_EXPLICIT_ASSUMPTIONS") == 5
    assert audit.FOUNDATION_LAYERS[-1]["status"] == "PARTIAL_COUNTEREXAMPLE_FOUND"
    assert audit.OPEN_RESEARCH_FRONTIER
    assert "orbit-selection sufficiency" in audit.OPEN_RESEARCH_FRONTIER[0]["current"]
    assert "bool_decoder_iff_constant_on_fibres" in audit.VERIFIED_THEOREMS


def test_nested_comment_mask_preserves_lines_and_does_not_hide_active_hole() -> None:
    source = "/- outer\n /- sorry -/ admit\n-/\ntheorem bad : True := by sorry\n"
    masked = audit.strip_lean_comments(source)
    assert masked.count("\n") == source.count("\n")
    assert masked.splitlines()[3] == "theorem bad : True := by sorry"
    assert "sorry" not in "\n".join(masked.splitlines()[:3])


def test_string_and_line_comment_are_not_proof_commands() -> None:
    source = 'def s := "sorry \\" admit /-" -- axiom\ntheorem ok : True := by trivial\n'
    masked = audit.strip_lean_comments(source)
    assert all(token not in masked for token in ("sorry", "admit", "axiom"))
    assert "theorem ok" in masked
    assert source.count("\n") == masked.count("\n")


@pytest.mark.parametrize("literal", ["'\"'", "'\\\"'"])
def test_double_quote_character_cannot_hide_the_next_declaration(literal: str) -> None:
    source = f"def quote : Char := {literal}\ntheorem probe : True := by sorry\n"
    assert "by sorry" in audit.strip_lean_comments(source)


@pytest.mark.parametrize("prefix", ["s!", "m!"])
def test_interpolated_lean_code_is_not_masked_as_literal_text(prefix: str) -> None:
    source = prefix + '"{(by /- harmless comment -/ sorry : Nat)}"'
    masked = audit.strip_lean_comments(source)
    assert "sorry" in masked
    assert "harmless" not in masked


def test_nested_interpolation_and_strings_preserve_active_code() -> None:
    source = 's!"{let s := "not code /-"; s!"{(by sorry : Nat)}"}"'
    assert "sorry" in audit.strip_lean_comments(source)


def test_axiom_audit_accepts_only_the_expected_standard_dependencies() -> None:
    output = (
        "'KBound.a' depends on axioms: [propext,\n Classical.choice, Quot.sound]\n"
        "'KBound.b' does not depend on any axioms\n"
    )
    result = audit.parse_axiom_audit(output, ["a", "b"])
    assert result["ok"]
    assert result["dependencies"]["KBound.b"] == []


@pytest.mark.parametrize("forbidden", ["sorryAx", "myCustomAxiom", "Lean.ofReduceBool"])
def test_axiom_audit_rejects_untrusted_transitive_dependency(forbidden: str) -> None:
    result = audit.parse_axiom_audit(f"'KBound.a' depends on axioms: [propext, {forbidden}]\n", ["a"])
    assert not result["ok"]
    assert result["forbidden_dependencies"] == {"KBound.a": [forbidden]}


@pytest.mark.parametrize(
    "output",
    [
        "",
        "'KBound.a_longer' does not depend on any axioms\n",
        "'KBound.a' does not depend on any axioms\n" * 2,
        "'KBound.a' depends on axioms: unexpected format\n",
    ],
)
def test_missing_misnamed_duplicate_or_malformed_axiom_output_fails(output: str) -> None:
    assert not audit.parse_axiom_audit(output, ["a"])["ok"]


def test_static_scan_cannot_be_reported_as_kernel_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = tmp_path / "static.json"
    monkeypatch.setattr(sys, "argv", ["formal_audit.py", "--json-out", str(report)])
    assert audit.main() == 0
    payload = json.loads(report.read_text())
    assert payload["status"] == "STATIC_PASS"
    assert payload["verified_theorem_count"] == 0
    assert payload["kernel_axiom_audit"] is None


def test_strict_core_requires_actual_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = tmp_path / "strict.json"
    monkeypatch.setattr(sys, "argv", ["formal_audit.py", "--strict-core", "--json-out", str(report)])
    assert audit.main() == 1
    payload = json.loads(report.read_text())
    assert payload["status"] == "FAIL"
    assert "requires --build" in payload["strict_blockers"][0]


def test_successful_build_without_axiom_evidence_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = tmp_path / "missing-axioms.json"
    monkeypatch.setattr(sys, "argv", ["formal_audit.py", "--build", "--json-out", str(report)])
    monkeypatch.setattr(audit, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, ""))
    assert audit.main() == 1
    payload = json.loads(report.read_text())
    assert payload["build_ok"] is True
    assert payload["verified_theorem_count"] == 0
    assert payload["kernel_axiom_audit"]["missing_declarations"]


def test_failed_build_still_writes_failure_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = tmp_path / "failed-build.json"
    monkeypatch.setattr(sys, "argv", ["formal_audit.py", "--build", "--json-out", str(report)])
    monkeypatch.setattr(audit, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, "build failed"))
    assert audit.main() == 1
    assert json.loads(report.read_text())["build_ok"] is False


@pytest.mark.parametrize("quotes", [("'", "'"), ("`", "`"), ("‘", "’")])
def test_unregistered_lean_proof_hole_warning_fails_even_with_good_capstones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quotes: tuple[str, str]
) -> None:
    report = tmp_path / "unregistered-hole.json"
    monkeypatch.setattr(sys, "argv", ["formal_audit.py", "--build", "--json-out", str(report)])
    monkeypatch.setattr(
        audit,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, f"warning: Extra.lean:2:0: declaration uses {quotes[0]}sorry{quotes[1]}\n"
        ),
    )
    monkeypatch.setattr(audit, "inspect_kernel_axioms", lambda: {"ok": True})
    assert audit.main() == 1
    assert json.loads(report.read_text())["compiler_proof_hole_warnings"]
