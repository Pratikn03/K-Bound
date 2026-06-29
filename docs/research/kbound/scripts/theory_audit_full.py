#!/usr/bin/env python3
"""Full K-Bound theory audit: paper label → proof → validator → artifact → claim.

Usage:
  python3 docs/research/kbound/scripts/theory_audit_full.py
  python3 docs/research/kbound/scripts/theory_audit_full.py --run-validators  # ~6 min
  python3 docs/research/kbound/scripts/theory_audit_full.py --write-report

Exit 1 if any CLOSED spine theorem lacks validator, artifact, or (when --run) fails.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
KBOUND = ROOT / "docs" / "research" / "kbound"

# status: closed | extension | open | withdrawn
THEOREM_REGISTRY: list[dict[str, Any]] = [
    {
        "labels": ["thm:headline", "thm:frontier"],
        "name": "Benefit-sign frontier (identifiable iff margin > β)",
        "status": "closed",
        "proof": "paper/sections/main_theory_5.tex",
        "paper": "kbound_short.tex",
        "validators": [
            "experiments/kbound/theory_validation/val_frontier.py",
            "experiments/kbound/theory_validation/val_benefit_frontier.py",
        ],
        "artifacts": [
            "experiments/kbound/theory_validation/results_frontier.json",
            "experiments/kbound/theory_validation/results_benefit_frontier.json",
        ],
        "code": "docs/research/kbound/kbound_pkg/kbound/certificate.py",
        "claim_ids": ["KB-CLAIM-001"],
    },
    {
        "labels": ["thm:disagree", "lem:reduction"],
        "name": "Disagreement-region reduction + multiclass extension",
        "status": "closed",
        "proof": "paper/sections/main_theory_5.tex",
        "paper": "kbound_short.tex",
        "validators": [
            "experiments/kbound/theory_validation/val_thm2_regret.py",
            "experiments/kbound/theory_validation/val_thm5_multiclass.py",
        ],
        "artifacts": [
            "experiments/kbound/theory_validation/results_thm2_regret.json",
            "experiments/kbound/theory_validation/results_thm5_multiclass.json",
        ],
        "code": "docs/research/kbound/scripts/analysis_F.py",
        "claim_ids": ["KB-CLAIM-001"],
    },
    {
        "labels": ["thm:imp", "thm:imp-quant"],
        "name": "Two-point impossibility + Le Cam lower bound",
        "status": "closed",
        "proof": "paper/sections/main_theory_5.tex",
        "paper": "kbound_short.tex",
        "validators": [
            "experiments/kbound/theory_validation/val_thm1_lecam.py",
            "experiments/kbound/theory_validation/val_thm2_lecam_finite_n.py",
        ],
        "artifacts": [
            "experiments/kbound/theory_validation/results_thm1_lecam.json",
            "experiments/kbound/theory_validation/results_thm2_lecam_finite_n.json",
            "docs/research/kbound/results/witness/witness_clean.json",
        ],
        "code": "docs/research/kbound/scripts/theory_extensions_validation.py",
        "claim_ids": ["KB-CLAIM-002"],
    },
    {
        "labels": ["thm:cert", "thm:gate"],
        "name": "Adapt/freeze/abstain certificate + regret identity",
        "status": "closed",
        "proof": "paper/sections/main_theory_5.tex",
        "paper": "kbound_short.tex",
        "validators": [
            "experiments/kbound/theory_validation/val_thm3_evalue.py",
            "experiments/kbound/theory_validation/val_thm2_regret.py",
        ],
        "artifacts": [
            "experiments/kbound/theory_validation/results_thm3_evalue_alpha005.json",
        ],
        "code": "docs/research/kbound/kbound_pkg/kbound/certificate.py",
        "claim_ids": ["KB-CLAIM-003"],
    },
    {
        "labels": ["thm:conj1-dichotomy", "conj:gen"],
        "name": "One-bit dichotomy; conj:gen resolved negatively",
        "status": "closed",
        "proof": "paper/sections/main_theory_5.tex",
        "paper": "kbound_short.tex",
        "validators": [
            "experiments/kbound/theory_validation/val_knowability_dichotomy.py",
            "docs/research/kbound/theory_v2/val_conj1_closure.py",
            "experiments/kbound/theory_validation/val_conj1_genpos.py",
        ],
        "artifacts": [
            "experiments/kbound/theory_validation/results_knowability_dichotomy.json",
            "docs/research/kbound/theory_v2/conj1_closure_results.json",
            "experiments/kbound/theory_validation/results_conj1_genpos.json",
        ],
        "code": None,
        "claim_ids": ["KB-CLAIM-025"],
    },
    {
        "labels": ["thm:cmono-weakest"],
        "name": "Weakest one-bit class under General Position (conditional)",
        "status": "closed",
        "proof": "paper/sections/weakest_class.tex",
        "paper": "kbound.tex appendix",
        "validators": ["experiments/kbound/theory_validation/val_conj1_genpos.py"],
        "artifacts": ["experiments/kbound/theory_validation/results_conj1_genpos.json"],
        "code": None,
        "claim_ids": ["KB-CLAIM-028"],
    },
    {
        "labels": ["thm:uncond-weakest"],
        "name": "Unconditional weakest one-bit dominance polytopes",
        "status": "closed",
        "proof": "paper/sections/weakest_class.tex; theory_v2/UNCONDITIONAL_WEAKEST_CLASS_ATTEMPT.md",
        "paper": "kbound.tex appendix",
        "validators": ["docs/research/kbound/theory_v2/val_unconditional_weakest.py"],
        "artifacts": ["docs/research/kbound/theory_v2/unconditional_weakest_results.json"],
        "code": None,
        "claim_ids": ["KB-CLAIM-029"],
    },
    {
        "labels": ["thm:anytime"],
        "name": "Anytime-valid streaming false-adapt certificate",
        "status": "extension",
        "proof": "theory_v2/sequential_anytime_theorem.tex",
        "paper": "kbound.tex App. theory extensions",
        "validators": ["docs/research/kbound/theory_v2/val_sequential_anytime.py"],
        "artifacts": ["docs/research/kbound/theory_v2/val_sequential_anytime_results.json"],
        "code": "docs/research/kbound/kbound_pkg/kbound/eprocess.py",
        "claim_ids": ["KB-CLAIM-031"],
    },
    {
        "labels": ["thm:multicand", "thm:routing"],
        "name": "Multicandidate family-wise false-adapt (Bonferroni)",
        "status": "extension",
        "proof": "theory_v2/multicandidate_theorem.tex",
        "paper": "kbound.tex App. theory extensions",
        "validators": ["docs/research/kbound/theory_v2/val_multicandidate.py"],
        "artifacts": ["docs/research/kbound/theory_v2/val_multicandidate_results.json"],
        "code": "docs/research/kbound/kbound_pkg/kbound/router.py",
        "claim_ids": ["KB-CLAIM-032"],
    },
    {
        "labels": ["thm:ev-rate", "thm:agl"],
        "name": "Evidence-channel rate (matching m^{-1/2})",
        "status": "extension",
        "proof": "paper/sections/main_theory_5.tex",
        "paper": "kbound.tex",
        "validators": ["experiments/kbound/theory_validation/val_agl.py"],
        "artifacts": ["experiments/kbound/theory_validation/results_agl.json"],
        "code": None,
        "claim_ids": [],
    },
    {
        "labels": ["conj:gen-capacity"],
        "name": "General knowability–capacity without full R1/R2",
        "status": "open",
        "proof": "paper/sections/knowability_capacity_general.tex",
        "paper": "appendix (partial results only)",
        "validators": [
            "experiments/kbound/theory_validation/val_knowability_capacity.py",
            "experiments/kbound/theory_validation/val_knowability_capacity_general.py",
            "docs/research/kbound/theory_v2/val_multiclass_capacity.py",
        ],
        "artifacts": [
            "experiments/kbound/theory_validation/results_knowability_capacity.json",
            "experiments/kbound/theory_validation/results_knowability_capacity_general.json",
            "docs/research/kbound/theory_v2/results_multiclass_capacity.json",
        ],
        "code": None,
        "claim_ids": [],
        "note": "Exploratory validators; full removal of R1/R2 not closed.",
    },
    {
        "labels": ["thm:minimax-opt"],
        "name": "Certificate minimax optimality",
        "status": "open",
        "proof": "theory_v2/minimax_optimality_theorem.tex",
        "paper": "not in main spine",
        "validators": ["docs/research/kbound/theory_v2/val_minimax_optimality.py"],
        "artifacts": ["docs/research/kbound/theory_v2/minimax_optimality_results.json"],
        "code": None,
        "claim_ids": [],
        "note": "Wave 2 draft; not claimed solved in PROJECT_STATUS.",
    },
]

DOC_DRIFT = [
    {
        "path": "docs/research/kbound/manuscript/",
        "issue": "Parallel book manuscript still marks conj:gen as open (resolved negatively in live kbound.tex).",
        "action": "Do not cite manuscript/ for submission status; use kbound_short.tex + PROJECT_STATUS.",
    },
    {
        "path": "docs/research/kbound/COMPLETION_STATUS_2026-06-19.md",
        "issue": "Superseded; banner added but file retained.",
        "action": "Use PROJECT_STATUS_AND_OPEN_PROBLEMS.md only.",
    },
]

OPEN_NO_VALIDATOR = [
    {"labels": ["conj:dich-compute"], "note": "Constructive measurability; partial via val_knowability_dichotomy only."},
    {"labels": ["thm:reg-iff"], "note": "Regression bracketing open; val_thm9prime_drift is partial probe."},
]


def _exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def _run_validator(script: str) -> tuple[bool, str]:
    sp = ROOT / script
    if not sp.exists():
        return False, "missing script"
    proc = subprocess.run(
        [sys.executable, str(sp)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    tail = (proc.stdout or proc.stderr or "").strip().splitlines()
    msg = tail[-1] if tail else f"exit {proc.returncode}"
    return proc.returncode == 0, msg


def audit(*, run_validators: bool = False) -> dict[str, Any]:
    ledger_path = KBOUND / "claim_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"claims": []}
    claims_by_id = {c["claim_id"]: c for c in ledger.get("claims", [])}

    rows = []
    issues: list[str] = []
    validators_run: dict[str, str] = {}

    for thm in THEOREM_REGISTRY:
        row = dict(thm)
        row["validators_ok"] = {v: _exists(v) for v in thm["validators"]}
        row["artifacts_ok"] = {a: _exists(a) for a in thm["artifacts"]}
        row["proof_ok"] = _exists(thm["proof"].split(";")[0].strip())
        row["code_ok"] = _exists(thm["code"]) if thm.get("code") else None

        if thm["status"] in ("closed", "extension"):
            for v, ok in row["validators_ok"].items():
                if not ok:
                    issues.append(f"{thm['labels'][0]}: missing validator {v}")
            for a, ok in row["artifacts_ok"].items():
                if not ok:
                    issues.append(f"{thm['labels'][0]}: missing artifact {a}")

        for cid in thm.get("claim_ids", []):
            c = claims_by_id.get(cid)
            row.setdefault("claim_status", {})[cid] = c["status"] if c else "MISSING_FROM_LEDGER"
            if not c:
                issues.append(f"{cid} referenced by {thm['labels'][0]} but not in claim_ledger")
            elif c["status"] not in ("supported", "no-harm"):
                issues.append(f"{cid} status={c['status']} for closed theorem {thm['labels'][0]}")

        if run_validators:
            for v in thm["validators"]:
                if v not in validators_run:
                    ok, msg = _run_validator(v)
                    validators_run[v] = "ok" if ok else f"FAIL: {msg}"
                    if not ok:
                        issues.append(f"validator failed: {v} ({msg})")

        rows.append(row)

    # Cross-check theorem-type claims in ledger
    for c in ledger.get("claims", []):
        if c.get("claim_type") != "theorem" or c["status"] == "withdrawn":
            continue
        for art in c.get("supporting_artifacts", []):
            if not _exists(art):
                issues.append(f"{c['claim_id']}: missing artifact {art}")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "theorems": rows,
        "open_frontier": OPEN_NO_VALIDATOR,
        "doc_drift_warnings": DOC_DRIFT,
        "validator_run": validators_run or None,
        "issues": issues,
        "all_ok": len(issues) == 0,
        "validator_count": len(list((ROOT / "experiments/kbound/theory_validation").glob("val_*.py")))
        + len(list((KBOUND / "theory_v2").glob("val_*.py"))),
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# K-Bound Full Theory Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Verdict: **{'PASS' if report['all_ok'] else 'ISSUES FOUND'}**",
        "",
        "## Closed spine (paper claims)",
        "",
        "| Theorem | Status | Validators | Artifacts | Claims |",
        "|---------|--------|------------|-----------|--------|",
    ]
    for thm in report["theorems"]:
        if thm["status"] not in ("closed", "extension"):
            continue
        v_ok = all(thm["validators_ok"].values())
        a_ok = all(thm["artifacts_ok"].values())
        mark = "✓" if v_ok and a_ok else "✗"
        lines.append(
            f"| {mark} `{thm['labels'][0]}` | {thm['status']} | "
            f"{len(thm['validators'])}/{sum(thm['validators_ok'].values())} | "
            f"{len(thm['artifacts'])}/{sum(thm['artifacts_ok'].values())} | "
            f"{', '.join(thm.get('claim_ids') or []) or '—'} |"
        )
    lines.extend(["", "## Open frontier (not claimed closed)", ""])
    for o in report["open_frontier"]:
        lines.append(f"- `{o['labels'][0]}`: {o['note']}")
    for thm in report["theorems"]:
        if thm["status"] == "open":
            lines.append(f"- `{thm['labels'][0]}`: {thm.get('note', 'see PROJECT_STATUS')}")
    if report.get("doc_drift_warnings"):
        lines.extend(["", "## Documentation drift (not failures)", ""])
        for d in report["doc_drift_warnings"]:
            lines.append(f"- **{d['path']}**: {d['issue']} → {d['action']}")
    if report["issues"]:
        lines.extend(["", "## Issues", ""])
        for i in report["issues"]:
            lines.append(f"- {i}")
    if report.get("validator_run"):
        fails = [k for k, v in report["validator_run"].items() if not v.startswith("ok")]
        lines.extend([
            "",
            "## Validator execution",
            f"- Ran {len(report['validator_run'])} scripts; failures: {len(fails)}",
        ])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-validators", action="store_true", help="execute all validators (~6 min)")
    ap.add_argument("--write-report", action="store_true", help="write reports/THEORY_AUDIT_FULL.md")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = audit(run_validators=args.run_validators)
    if args.write_report or not args.json:
        out = KBOUND / "reports" / "THEORY_AUDIT_FULL.md"
        write_report(report, out)
        print(f"Wrote {out.relative_to(ROOT)}")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Verdict: {'PASS' if report['all_ok'] else 'FAIL'} ({len(report['issues'])} issues)")
        for i in report["issues"][:20]:
            print(f"  - {i}")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
