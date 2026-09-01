#!/usr/bin/env python3
"""Audit the K-Bound Lean formalization status."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

LEGACY_CORE_THEOREMS = [
    "cert_false_adapt_sound",
    "cert_false_freeze_sound",
    "measure_false_adapt_le_alpha",
    "measure_false_freeze_le_alpha",
    "measure_false_adapt_le_alpha_of_measurable",
    "card_high_strictRank_le",
    "card_low_strictRank_ge",
    "uniformIndex_miss_eq",
    "uniformIndex_miss_le",
    "uniformIndex_coverage_ge",
    "uniformIndex_false_adapt_le",
    "uniformIndex_false_freeze_le",
    "gate_regret_identity",
    "abstention_mass_ge_one_sub_two_alpha_arith",
    "matched_opposite_worlds_force_abstain",
    "lecam_regret_floor_two_point",
    "lecam_testing_two_point",
    "frontier_identifiable_positive",
    "frontier_identifiable_negative",
    "frontier_decision_adapt",
    "frontier_decision_freeze",
    "frontier_decision_abstain",
    "frontier_band_zero_witness",
    "frontier_open_band_opposite_witnesses",
    "frontier_positive_boundary_zero_strict",
    "frontier_negative_boundary_zero_strict",
    "binary_sign_reduction",
    "multiclass_sign_reduction",
    "multiclass_harm_iff_nonpos",
    "multiclass_benefit_pos_of_pa_gt",
    "multiclass_routing_harm_equiv",
    "single_candidate_false_adapt_sound",
    "one_sided_commit_when_radius_small",
    "two_sided_sign_certified",
    "finite_uniform_rank_coverage_add_miss",
    "finite_uniform_rank_miss_le_alpha",
    "exchangeable_conformal_miss_le_alpha",
    "exchangeable_cert_false_adapt_sound",
    "bettingFactor_le_one",
    "betting_wealth_step_le",
    "binary_benefit_neg_accuracy",
    "binary_sign_flip_on_accuracy_complement",
    "multiclass_benefit_swap_pa_p0",
    "lecam_tv_identity",
    "lecam_single_error_ge_one_sub_tv",
    "rate_implies_commit",
    "rate_conformal_miss",
    # Historical finite reductions, not substitutes for the probability layer.
    "uniformIndexLaw_miss_le_alpha",
    "uniformIndexLaw_false_adapt_le",
    "betting_wealth_supermartingale_step",
    "ville_bound_false_adapt",
    "lecam_tv_two_point_measure",
    "lecam_testing_error_ge_one_sub_tv_measure",
    "hoeffding_radius_le",
    "rate_commit_from_concentration",
    "evidence_swap_involution",
    "swap_flips_benefit_preserves_evidence",
    # Wave 7 finite measurable target-law and distributional frontier closure
    "finiteEvidence_measurable",
    "finite_target_laws_matched_evidence",
    "positiveTargetLaw_benefit",
    "negativeTargetLaw_benefit",
    "finite_target_world_pair",
    "rich_closed_band_forces_abstain",
    "frontierDecision_uniformly_sound",
    "distributional_frontier_maximal",
]

# Explicit capstones, not a count of every supporting lemma in the project.
# All names are checked by Lean and their transitive axioms are inspected below.
FOUNDATION_THEOREMS = {
    "MeasureConformal": [
        "exchangeable_scoreLaw_miss_le",
        "exchangeable_scores_rank_miss_le",
        "exchangeable_scores_rank_coverage_ge",
        "calibrationThreshold_rank_le",
        "exchangeable_calibration_threshold_miss_le",
        "exchangeable_calibration_threshold_coverage_ge",
        "exchangeable_residual_coverage_ge",
        "exchangeable_residual_false_adapt_le",
        "exchangeable_residual_false_freeze_le",
        "exchangeable_residual_either_error_le",
    ],
    "FilteredVille": [
        "filtered_optional_stopping_le",
        "filtered_ville_finite",
        "filtered_ville",
        "filtered_ville_alpha",
        "dominated_eprocess_ville",
        "eprocess_finite_time_crossing",
        "filtered_betting_supermartingale",
        "filtered_betting_anytime",
        "predictable_betting_wealth_bounds",
        "predictable_betting_wealth_adapted",
        "bounded_predictable_betting_anytime",
    ],
    "InformationBound": ["binary_bretagnolle_huber"],
    "GeneralLeCam": [
        "measurableTotalVariation_eq_abs_sup",
        "measurableTotalVariation_symm",
        "measurableTotalVariation_map_le",
        "general_lecam_testing_error_ge",
        "exists_lecam_optimal_test",
        "general_lecam_inf_testing_error",
        "general_lecam_worst_case_error_ge",
        "general_lecam_regret_floor",
        "general_lecam_iid_testing_identity",
        "binary_partition_kl_le",
        "binary_partition_support",
        "klDiv_map_measurableEquiv",
        "klDiv_prod_add",
        "klDiv_iidObservationLaw",
        "general_bretagnolle_huber_finite",
        "general_bretagnolle_huber",
        "general_lecam_exponential_regret_floor",
        "general_lecam_iid_exponential_regret_floor",
    ],
    "Concentration": [
        "subgaussian_abs_tail",
        "bounded_independent_sum_tail",
        "unit_interval_mean_tail",
        "unit_interval_hoeffding_coverage",
        "common_mean_hoeffding_coverage",
        "paired_benefit_hoeffding_coverage",
        "adapted_subgaussian_sum_tail",
        "conditional_hoeffding_of_bounded_zero_mean",
        "bounded_martingale_difference_tail",
    ],
    "MeasureSwap": [
        "measurable_predictionSwap",
        "predictionSwap_law_involutive",
        "predictionSwap_preserves_evidence",
        "predictionSwap_preserves_channel",
        "predictionSwap_negates_populationBenefit",
        "evidence_definable_opposite_target",
    ],
    "MeasureTarget": [
        "targetLabelKernel_isMarkov",
        "joint_target_probability",
        "target_label_free_law",
        "measurable_label_kernel_freedom",
        "measurable_label_kernel_freedom_subtype",
        "constructed_target_population_benefit",
        "constant_target_population_benefit",
        "disagreementMean_bounds",
        "measurable_target_benefit_reduction",
        "measurable_correctness_identified_interval",
        "measurable_target_frontier_attainment",
    ],
    "MeasureFrontier": [
        "correctnessFieldTarget_properties",
        "correctnessFieldTarget_benefit",
        "measurable_frontier_class_nonempty",
        "measurable_frontier_adapt_iff",
        "measurable_frontier_freeze_iff",
        "measurable_closed_band_zero_target",
        "measurable_open_band_opposite_targets",
    ],
    "ChannelCounterexample": [
        "OrbitFibreCounterexample.selected_exactly_one",
        "OrbitFibreCounterexample.orbit_selection_not_fibre_orientation",
        "OrbitFibreCounterexample.no_evidence_decoder",
        "bool_decoder_iff_constant_on_fibres",
    ],
}
VERIFIED_THEOREMS = LEGACY_CORE_THEOREMS + [
    name for names in FOUNDATION_THEOREMS.values() for name in names
]
ALLOWED_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})

# Status is a statement of encoded scope, not a claim that empirical assumptions
# hold. A successful --build is still required to verify this source revision.
FOUNDATION_LAYERS: list[dict[str, str]] = [
    {
        "item": "measure-theoretic split-conformal exchangeability",
        "status": "MECHANIZED_WITH_EXPLICIT_ASSUMPTIONS",
        "scope": "measurable exchangeable score laws, ties, calibration thresholds, and one-shot residual coverage/error bounds",
        "limits": "does not establish benchmark exchangeability, batch-to-population transfer, or simultaneous repeated-use coverage",
    },
    {
        "item": "filtered e-process and Ville optional-stopping layer",
        "status": "MECHANIZED_WITH_EXPLICIT_ASSUMPTIONS",
        "scope": "filtered nonnegative supermartingales, bounded optional stopping, countable-time Ville, domination, and constructed predictable betting wealth",
        "limits": "requires the declared filtration, integrability and conditional null; no unbounded stopping-time expectation equality is asserted",
    },
    {
        "item": "general KL/TV Le Cam probability layer",
        "status": "MECHANIZED_WITH_EXPLICIT_ASSUMPTIONS",
        "scope": "arbitrary probability measures, randomized measurable tests, exact TV testing identity, KL/Bretagnolle-Huber bound, and finite iid products",
        "limits": "KL infinity is handled explicitly; no independence or KL bound is inferred for an empirical dataset",
    },
    {
        "item": "concentration and martingale probability layer",
        "status": "MECHANIZED_WITH_EXPLICIT_ASSUMPTIONS",
        "scope": "independent bounded Hoeffding tails/coverage, paired-benefit scaling, conditional Hoeffding, and adapted martingale-difference tails",
        "limits": "not a nonlinear evidence-ratio rate, empirical-Bernstein theorem, or automatic concentration of cross-fitted benchmark cells",
    },
    {
        "item": "general measurable target-law richness construction",
        "status": "MECHANIZED_WITH_EXPLICIT_ASSUMPTIONS",
        "scope": "actual label kernels and joint population risks on arbitrary measurable input spaces; full correctness-field class supported on the two predicted labels on disagreement, preserved evidence laws, and exact clipped strict frontier without RichAt",
        "limits": "requires measurable predictors/kernels, feasible margins and positive disagreement mass; arbitrary restricted deployment subclasses and unrestricted multiclass kernels are not automatically covered",
    },
    {
        "item": "full one-bit channel dichotomy over the historical declared model class",
        "status": "PARTIAL_COUNTEREXAMPLE_FOUND",
        "scope": "general measurable label-swap involution, invariant label-free channels, opposite population risks, evidence-definable-class impossibility, an orbit/fibre counterexample, and set-theoretic decoder factorization",
        "limits": "one representative per swap orbit is insufficient without sign consistency on the entire evidence fibre; historical H/ratio-rate extensions are not verified",
    },
]

FOUNDATIONAL_PROBABILITY_LIMITS: list[dict[str, str]] = []

# This is a substantive open/incorrect extension, not a placeholder to remove
# merely because the maintained, narrower theorem spine now compiles.
OPEN_RESEARCH_FRONTIER: list[dict[str, str]] = [
    {
        "item": "full historical one-bit channel sufficiency and H/ratio-rate extension",
        "current": "measurable swap/impossibility proved; orbit-selection sufficiency has an evidence-fibre counterexample",
        "needed": "a corrected fibre-consistent orientation theorem and separately justified nondegenerate H/ratio-rate model; excluded from the compact paper",
    }
]

CLOSURE_RECORD = {
    "revision": "measurable-foundations-2026-08-31",
    "date": "2026-08-31",
    "scope": (
        "Legacy finite/algebraic spine plus general exchangeable residual coverage, "
        "filtered Ville/predictable betting, KL/TV finite-product testing, concentration, "
        "and measurable target-law frontier. The historical full one-bit/H extension "
        "is not closed and is not claimed by the compact paper."
    ),
    "new_mechanized_modules": [
        f"KBound/Probability/{module}.lean" for module in FOUNDATION_THEOREMS
    ],
    "foundation_layers": FOUNDATION_LAYERS,
    "open_research_frontier": OPEN_RESEARCH_FRONTIER,
}


def run(
    cmd: list[str], *, check: bool = True, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        input=input_text,
    )


def strip_lean_comments(text: str) -> str:
    """Mask comments/literal text, retaining code in interpolated strings.

    A single regular expression mishandles Lean's nested block comments and can
    either hide a proof hole or report one on the wrong line. This lexical guard
    complements, and never replaces, the transitive kernel-axiom inspection.
    """
    masked = list(text)
    char_literal = re.compile(r"'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^\\'\n])'")

    def blank(start: int, end: int) -> None:
        for offset in range(start, end):
            if text[offset] != "\n":
                masked[offset] = " "

    def string(start: int, interpolated: bool) -> int:
        blank(start, start + 1)
        i = start + 1
        while i < len(text):
            if text[i] == "\\":
                end = min(i + 2, len(text))
                blank(i, end)
                i = end
            elif text[i] == '"':
                blank(i, i + 1)
                return i + 1
            elif interpolated and text[i : i + 2] in {"{{", "}}"}:
                blank(i, i + 2)
                i += 2
            elif interpolated and text[i] == "{":
                # The interpolation body is Lean code, including nested braces,
                # comments and strings. In particular, never hide `by sorry`.
                i = code(i + 1, in_interpolation=True)
            else:
                blank(i, i + 1)
                i += 1
        return i

    def code(start: int, *, in_interpolation: bool = False) -> int:
        i, braces = start, 0
        while i < len(text):
            pair = text[i : i + 2]
            if pair == "/-":
                begin, depth = i, 1
                i += 2
                while i < len(text) and depth:
                    if text[i : i + 2] == "/-":
                        depth += 1
                        i += 2
                    elif text[i : i + 2] == "-/":
                        depth -= 1
                        i += 2
                    else:
                        i += 1
                blank(begin, i)
            elif pair == "--":
                end = text.find("\n", i)
                end = len(text) if end == -1 else end
                blank(i, end)
                i = end
            elif text[i] == "'" and (match := char_literal.match(text, i)):
                # `Char := '"'` must not open a multiline string.
                blank(i, match.end())
                i = match.end()
            elif text[i] == '"':
                i = string(i, text[:i].rstrip().endswith("!"))
            elif in_interpolation and text[i] == "}":
                if braces == 0:
                    return i + 1
                braces -= 1
                i += 1
            else:
                if in_interpolation and text[i] == "{":
                    braces += 1
                i += 1
        return i

    code(0)
    return "".join(masked)


def lean_source_paths() -> list[Path]:
    paths = sorted(p for p in (ROOT / "KBound").rglob("*.lean") if p.is_file() and not p.name.startswith("._"))
    for extra in (ROOT / "KBound.lean", ROOT / "lakefile.lean"):
        if extra.is_file() and not extra.name.startswith("._"):
            paths.append(extra)
    return paths


def scan_for_forbidden_tokens() -> list[dict[str, str | int]]:
    forbidden = re.compile(r"\b(sorry|admit|axiom|unsafe|native_decide)\b")
    hits: list[dict[str, str | int]] = []
    for path in lean_source_paths():
        try:
            stripped = strip_lean_comments(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            hits.append({"path": str(path.relative_to(ROOT)), "line": 0, "token": "non_utf8"})
            continue
        for lineno, line in enumerate(stripped.splitlines(), start=1):
            match = forbidden.search(line)
            if match:
                hits.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "line": lineno,
                        "token": match.group(1),
                    }
                )
    return hits


def theorem_map_checks() -> list[str]:
    theorem_map = ROOT / "KBound" / "TheoremMap.lean"
    text = strip_lean_comments(theorem_map.read_text(encoding="utf-8"))
    checked = set(re.findall(r"^\s*#check\s+KBound\.([\w.]+)\s*$", text, re.MULTILINE))
    missing = []
    for name in VERIFIED_THEOREMS:
        if name not in checked:
            missing.append(name)
    return missing


def parse_axiom_audit(output: str, names: list[str]) -> dict[str, object]:
    """Fail closed on missing/duplicate declarations or nonstandard axioms."""
    pattern = re.compile(
        r"'([^'\n]+)'\s+(?:depends on axioms:\s*\[([^\]]*)\]|"
        r"(does not depend on any axioms))"
    )
    dependencies: dict[str, list[str]] = {}
    duplicates: list[str] = []
    for match in pattern.finditer(output):
        name = match.group(1)
        if name in dependencies:
            duplicates.append(name)
        dependencies[name] = sorted(
            part.strip() for part in (match.group(2) or "").split(",") if part.strip()
        )
    expected = {f"KBound.{name}" for name in names}
    missing = sorted(expected - dependencies.keys())
    unexpected_names = sorted(dependencies.keys() - expected)
    forbidden = {
        name: sorted(set(axioms) - ALLOWED_AXIOMS)
        for name, axioms in dependencies.items()
        if set(axioms) - ALLOWED_AXIOMS
    }
    return {
        "ok": not (missing or duplicates or unexpected_names or forbidden),
        "allowed_axioms": sorted(ALLOWED_AXIOMS),
        "dependencies": dependencies,
        "missing_declarations": missing,
        "duplicate_declarations": duplicates,
        "unexpected_declarations": unexpected_names,
        "forbidden_dependencies": forbidden,
    }


def inspect_kernel_axioms() -> dict[str, object]:
    source = "import KBound\n" + "".join(
        f"#print axioms KBound.{name}\n" for name in VERIFIED_THEOREMS
    )
    proc = run(["lake", "env", "lean", "--stdin"], check=False, input_text=source)
    result = parse_axiom_audit(proc.stdout, VERIFIED_THEOREMS)
    result["returncode"] = proc.returncode
    result["ok"] = result["ok"] and proc.returncode == 0
    if not result["ok"]:
        result["output_tail"] = "\n".join(proc.stdout.strip().splitlines()[-30:])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--strict-core",
        action="store_true",
        help="Require --build and verify the declared spine, probability capstones and allowed axioms.",
    )
    parser.add_argument(
        "--strict-100",
        action="store_true",
        help=(
            "Legacy alias for --strict-core. This does not mean full foundational "
            "Mathlib probability/martingale/KL mechanization."
        ),
    )
    parser.add_argument(
        "--full-foundations",
        action="store_true",
        help="Audit all six layers; fails while the disclosed historical one-bit extension remains open.",
    )
    args = parser.parse_args()
    strict_core = args.strict_core or args.strict_100

    build_ok = None
    build_tail = ""
    axioms = None
    proof_hole_warnings: list[str] = []
    if args.build:
        try:
            proc = run(["lake", "build", "KBound"], check=False)
            build_ok = proc.returncode == 0
            build_tail = "\n".join(proc.stdout.strip().splitlines()[-20:])
            # Also reject Lean's own proof-hole warnings, including declarations
            # outside the capstone registry and warnings replayed from the cache.
            proof_hole_warnings = [
                line for line in proc.stdout.splitlines()
                if re.search(r"declaration uses ['‘`]sorry['’`]", line)
            ]
            if build_ok:
                axioms = inspect_kernel_axioms()
        except OSError as exc:
            build_ok = False
            build_tail = str(exc)

    forbidden_hits = scan_for_forbidden_tokens()
    missing_checks = theorem_map_checks()
    strict_blockers = []
    if strict_core and not args.build:
        strict_blockers.append("--strict-core requires --build; a source scan is not kernel verification")
    if len(VERIFIED_THEOREMS) != len(set(VERIFIED_THEOREMS)):
        strict_blockers.append("duplicate names in the declared theorem registry")
    foundation_blockers = [row["item"] for row in FOUNDATIONAL_PROBABILITY_LIMITS]
    if args.full_foundations:
        strict_blockers.extend(foundation_blockers)
        strict_blockers.extend([row["item"] for row in OPEN_RESEARCH_FRONTIER])

    kernel_ok = bool(build_ok and axioms and axioms["ok"])
    ok = (
        not forbidden_hits
        and not missing_checks
        and not proof_hole_warnings
        and (not args.build or kernel_ok)
        and not strict_blockers
    )

    report = {
        "status": ("PASS" if args.build else "STATIC_PASS") if ok else "FAIL",
        "build_checked": bool(args.build),
        "build_ok": build_ok,
        "declared_theorem_count": len(VERIFIED_THEOREMS),
        "legacy_core_theorem_count": len(LEGACY_CORE_THEOREMS),
        "verified_theorem_count": len(VERIFIED_THEOREMS) if kernel_ok else 0,
        "kernel_axiom_audit": axioms,
        "compiler_proof_hole_warnings": proof_hole_warnings,
        "closure_record": CLOSURE_RECORD,
        "forbidden_token_hits": forbidden_hits,
        "missing_theorem_map_checks": missing_checks,
        "strict_core_checked": bool(strict_core),
        "legacy_strict_100_alias_used": bool(args.strict_100),
        "full_foundations_checked": bool(args.full_foundations),
        "foundational_probability_limits": FOUNDATIONAL_PROBABILITY_LIMITS,
        "foundation_layers": FOUNDATION_LAYERS,
        "open_research_frontier": OPEN_RESEARCH_FRONTIER,
        "full_foundations_scope_complete": not (FOUNDATIONAL_PROBABILITY_LIMITS or OPEN_RESEARCH_FRONTIER),
        "strict_blockers": strict_blockers,
        "build_output_tail": build_tail,
    }
    if args.json_out is not None:
        out = args.json_out if args.json_out.is_absolute() else ROOT / args.json_out
        out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Formal audit: {report['status']}")
    if args.build:
        print("Lean build: PASS" if build_ok else "Lean build: FAIL")
        print("Kernel axiom audit: PASS" if kernel_ok else "Kernel axiom audit: FAIL")
        print(f"Kernel-checked registered declarations: {report['verified_theorem_count']}")
    else:
        print(f"Declared theorem-map checks (kernel build not requested): {len(VERIFIED_THEOREMS)}")
    print("Forbidden proof-hole scan: PASS" if not forbidden_hits else "Forbidden proof-hole scan: FAIL")
    if args.strict_100:
        print("NOTE: --strict-100 is a legacy alias for strict-core scope, not full Mathlib probability foundations.")
    print("Scope: five probability/construction layers plus a partial sixth layer; not full historical closure.")
    if args.full_foundations:
        print(
            "Full foundational probability gate: FAIL"
            if strict_blockers
            else "Full foundational probability gate: PASS"
        )
    if strict_blockers:
        print("Strict/full-foundation blockers:")
        for item in strict_blockers:
            print(f"- {item}")
    if forbidden_hits:
        for hit in forbidden_hits:
            print(f"FORBIDDEN {hit['path']}:{hit['line']} {hit['token']}")
    if missing_checks:
        for name in missing_checks:
            print(f"MISSING #check {name}")
    for warning in proof_hole_warnings:
        print(f"PROOF-HOLE WARNING {warning}")
    if args.build and build_tail:
        print(build_tail)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
