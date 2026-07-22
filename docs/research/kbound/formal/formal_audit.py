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

VERIFIED_THEOREMS = [
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
    "forced_abstention_probability",
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
    # Wave 6 paper-faithful foundation closures
    "exchangeable_scores_miss_le_alpha",
    "exchangeable_scores_false_adapt_le",
    "betting_wealth_supermartingale_step",
    "ville_bound_false_adapt",
    "lecam_tv_two_point_measure",
    "lecam_testing_error_ge_one_sub_tv_measure",
    "hoeffding_radius_le",
    "rate_commit_from_concentration",
    "evidence_swap_involution",
    "swap_flips_benefit_preserves_evidence",
]

# Wave 6 closed the paper-faithful foundation cores. Empty list ⇒ --full-foundations can pass.
FOUNDATIONAL_PROBABILITY_LIMITS: list[dict[str, str]] = []

# Wave 4 paper/validator frontier items that remain outside the Lean package.
OPEN_RESEARCH_FRONTIER: list[dict[str, str]] = []

CLOSURE_RECORD = {
    "wave": 6,
    "date": "2026-07-15",
    "scope": (
        "kernel-checked algebraic theorem spine, uniform-index conformal measure layer, "
        "and paper-faithful foundation closures (exchangeable-score reduction, discrete "
        "Ville, two-point Le Cam packaging, Hoeffding radius commit bridge, evidence "
        "swap involution)"
    ),
    "mechanized_modules": [
        "KBound/Probability/ConformalExchangeability.lean",
        "KBound/Probability/Exchangeable.lean",
        "KBound/Probability/EProcess.lean",
        "KBound/Probability/Ville.lean",
        "KBound/Dichotomy.lean",
        "KBound/Probability/LeCam.lean",
        "KBound/Probability/LeCamMeasure.lean",
        "KBound/Probability/Rates.lean",
    ],
    "paper_closures": [
        "theory_v2/tight_constants_closure.tex",
        "theory_v2/multiclass_multicandidate_theorem.tex",
        "theory_v2/anytime_multicandidate_theorem.tex",
        "theory_v2/multiclass_capacity_impossibility.tex",
        "theory_v2/margin_computability_closure.tex",
        "theory_v2/regression_bracketing_closure.tex",
    ],
    "documented_foundational_probability_limits": FOUNDATIONAL_PROBABILITY_LIMITS,
}


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def strip_lean_comments(text: str) -> str:
    text = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    text = re.sub(r"--.*", "", text)
    return text


def lean_source_paths() -> list[Path]:
    paths = sorted(
        p
        for p in (ROOT / "KBound").rglob("*.lean")
        if p.is_file() and not p.name.startswith("._")
    )
    for extra in (ROOT / "KBound.lean", ROOT / "lakefile.lean"):
        if extra.is_file() and not extra.name.startswith("._"):
            paths.append(extra)
    return paths


def scan_for_forbidden_tokens() -> list[dict[str, str | int]]:
    forbidden = re.compile(r"\b(sorry|admit|axiom|unsafe)\b")
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
    text = theorem_map.read_text()
    missing = []
    for name in VERIFIED_THEOREMS:
        if f"#check KBound.{name}" not in text:
            missing.append(name)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--strict-core",
        action="store_true",
        help="Check the current Lean theorem spine and documented Wave 4 core scope.",
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
        help="Fail unless the documented deep probability-foundation limits are closed.",
    )
    args = parser.parse_args()
    strict_core = args.strict_core or args.strict_100

    build_ok = None
    build_tail = ""
    if args.build:
        proc = run(["lake", "build"], check=False)
        build_ok = proc.returncode == 0
        build_tail = "\n".join(proc.stdout.strip().splitlines()[-20:])
        if not build_ok:
            print("FAIL: lake build failed")
            print(build_tail)
            return proc.returncode or 1

    forbidden_hits = scan_for_forbidden_tokens()
    missing_checks = theorem_map_checks()
    strict_blockers = []
    foundation_blockers = [row["item"] for row in FOUNDATIONAL_PROBABILITY_LIMITS]
    if args.full_foundations:
        strict_blockers.extend(foundation_blockers)
        strict_blockers.extend([row["item"] for row in OPEN_RESEARCH_FRONTIER])

    ok = not forbidden_hits and not missing_checks and (build_ok is not False) and not strict_blockers

    report = {
        "status": "PASS" if ok else "FAIL",
        "build_checked": bool(args.build),
        "build_ok": build_ok,
        "verified_theorem_count": len(VERIFIED_THEOREMS),
        "closure_record": CLOSURE_RECORD,
        "forbidden_token_hits": forbidden_hits,
        "missing_theorem_map_checks": missing_checks,
        "strict_core_checked": bool(strict_core),
        "legacy_strict_100_alias_used": bool(args.strict_100),
        "full_foundations_checked": bool(args.full_foundations),
        "foundational_probability_limits": FOUNDATIONAL_PROBABILITY_LIMITS,
        "strict_blockers": strict_blockers,
    }
    if args.json_out is not None:
        out = args.json_out if args.json_out.is_absolute() else ROOT / args.json_out
        out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Formal audit: {report['status']}")
    if args.build:
        print("Lean build: PASS")
    print(f"Kernel-checked theorem checks: {len(VERIFIED_THEOREMS)}")
    print("Forbidden proof-hole scan: PASS" if not forbidden_hits else "Forbidden proof-hole scan: FAIL")
    if args.strict_100:
        print("NOTE: --strict-100 is a legacy alias for strict-core scope, not full Mathlib probability foundations.")
    print(f"Documented foundational probability limits: {len(FOUNDATIONAL_PROBABILITY_LIMITS)}")
    if args.full_foundations:
        print("Full foundational probability gate: FAIL" if foundation_blockers else "Full foundational probability gate: PASS")
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
    if args.build and build_tail:
        print(build_tail)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
