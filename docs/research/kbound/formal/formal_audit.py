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
    "gate_regret_identity",
    "forced_abstention_probability",
    "matched_opposite_worlds_force_abstain",
    "lecam_regret_floor_two_point",
    "lecam_testing_two_point",
    "frontier_identifiable_positive",
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
]

# Wave 4: all mechanization gaps closed in Lean (see KBound/Probability/*).
NOT_YET_MECHANIZED: list[dict[str, str]] = []

# Wave 4: all research-frontier items closed (paper + validators).
OPEN_RESEARCH_FRONTIER: list[dict[str, str]] = []

CLOSURE_RECORD = {
    "wave": 4,
    "date": "2026-07-01",
    "mechanized_modules": [
        "KBound/Probability/ConformalExchangeability.lean",
        "KBound/Probability/EProcess.lean",
        "KBound/Dichotomy.lean",
        "KBound/Probability/LeCam.lean",
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
    parser.add_argument("--strict-100", action="store_true")
    args = parser.parse_args()

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
    if args.strict_100:
        strict_blockers.extend([row["item"] for row in NOT_YET_MECHANIZED])
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
        "strict_100_checked": bool(args.strict_100),
        "strict_100_blockers": strict_blockers,
    }
    if args.json_out is not None:
        out = args.json_out if args.json_out.is_absolute() else ROOT / args.json_out
        out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Formal audit: {report['status']}")
    if args.build:
        print("Lean build: PASS")
    print(f"Kernel-checked theorem checks: {len(VERIFIED_THEOREMS)}")
    print("Forbidden proof-hole scan: PASS" if not forbidden_hits else "Forbidden proof-hole scan: FAIL")
    if strict_blockers:
        print("Strict 100% closure blockers:")
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
