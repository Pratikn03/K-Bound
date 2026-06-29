#!/usr/bin/env python3
"""K-Bound complete picture: theory labels → proof/validator → empirical artifacts.

Used by notebooks/00_KBound_Master_Guide.ipynb and CLI:
  bash docs/research/kbound/scripts/kbound_tour.sh
  python3 docs/research/kbound/scripts/kbound_tour.py --json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]  # AutoML_Flagship_V8
KBOUND = ROOT / "docs" / "research" / "kbound"


THEORY_SPINE: list[dict[str, Any]] = [
    {
        "label": "thm:frontier / thm:headline",
        "statement": "Benefit sign identifiable iff observable margin |M| > β.",
        "paper": "kbound_short.tex §Theory; paper/sections/main_theory_5.tex",
        "proof": "paper/sections/main_theory_5.tex",
        "validators": [
            "experiments/kbound/theory_validation/val_frontier.py",
            "experiments/kbound/theory_validation/val_benefit_frontier.py",
        ],
        "artifacts": [
            "experiments/kbound/theory_validation/results_frontier.json",
        ],
        "code": "docs/research/kbound/kbound_pkg/kbound/certificate.py",
        "claim_ids": ["KB-CLAIM-001"],
    },
    {
        "label": "thm:imp / thm:conj1-dichotomy",
        "statement": "Matched evidence, opposite benefit → minimax committal error ½; one-bit minimal.",
        "paper": "kbound_short.tex",
        "proof": "paper/sections/main_theory_5.tex",
        "validators": [
            "experiments/kbound/theory_validation/val_thm1_lecam.py",
            "experiments/kbound/theory_validation/val_knowability_dichotomy.py",
        ],
        "artifacts": [
            "experiments/kbound/theory_validation/results_thm1_lecam.json",
            "docs/research/kbound/results/witness/witness_clean.json",
        ],
        "code": "docs/research/kbound/scripts/theory_extensions_validation.py",
        "claim_ids": ["KB-CLAIM-002", "KB-CLAIM-025"],
    },
    {
        "label": "thm:certificate / thm:anytime",
        "statement": "Split conformal + e-value: false-adapt FA_u ≤ α (time-uniform under optional stopping).",
        "paper": "kbound_short.tex guarantee box",
        "proof": "paper/sections/main_theory_5.tex; App. theory extensions",
        "validators": [
            "experiments/kbound/theory_validation/val_thm3_evalue.py",
            "docs/research/kbound/theory_v2/val_sequential_anytime.py",
            "docs/research/kbound/theory_v2/val_multicandidate.py",
        ],
        "artifacts": [
            "experiments/kbound/theory_validation/results_thm3_evalue_alpha005.json",
        ],
        "code": "docs/research/kbound/kbound_pkg/kbound/certificate.py",
        "claim_ids": ["KB-CLAIM-003"],
    },
    {
        "label": "thm:uncond-weakest",
        "statement": "Unconditional weakest one-bit classes = dominance polytopes W*.",
        "paper": "kbound.tex App.",
        "proof": "theory_v2/UNCONDITIONAL_WEAKEST_CLASS_ATTEMPT.md",
        "validators": ["docs/research/kbound/theory_v2/val_unconditional_weakest.py"],
        "artifacts": ["docs/research/kbound/theory_v2/unconditional_weakest_results.json"],
        "code": None,
        "claim_ids": [],
    },
]

EMPIRICAL_CLAIMS: list[dict[str, Any]] = [
    {
        "claim_id": "KB-CLAIM-010",
        "headline": "CIFAR stress grid beats-both (Tent, EATA)",
        "artifact": "experiments/kbound/results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json",
        "runner": "docs/research/kbound/scripts/kbtrain.sh cifar10c",
        "notebook": "notebooks/05_TTA_CIFAR_and_Online.ipynb",
    },
    {
        "claim_id": "KB-CLAIM-026",
        "headline": "Mixed head-to-head vs POEM/AETTA (WIN)",
        "artifact": "experiments/kbound/results/mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_tent_primary.json",
        "runner": "experiments/kbound/poem_aetta/run_all_headtohead.sh",
        "notebook": "notebooks/00_KBound_Master_Guide.ipynb",
    },
    {
        "claim_id": "KB-CLAIM-024",
        "headline": "Cross-protocol mixed OOF aggregate beats-both",
        "artifact": "experiments/kbound/results/mixed_protocol_oof_v2/mixed_protocol_oof_v2_result.json",
        "runner": "docs/research/kbound/scripts/mixed_stream_kbound.py",
        "notebook": "notebooks/03_Harmful_Mixed_Rigor.ipynb",
    },
    {
        "claim_id": "KB-CLAIM-020",
        "headline": "Office-Home no-harm OOF",
        "artifact": "docs/research/kbound/results_source.json",
        "runner": "docs/research/kbound/scripts/kbtrain.sh protocol-m-v2",
        "notebook": "notebooks/06_Evidence_and_Drift.ipynb",
    },
    {
        "claim_id": "KB-CLAIM-021",
        "headline": "iWildCam no-harm OOF",
        "artifact": "docs/research/kbound/results_source.json",
        "runner": "docs/research/kbound/scripts/kbtrain.sh protocol-h-v2",
        "notebook": "notebooks/06_Evidence_and_Drift.ipynb",
    },
    {
        "claim_id": "KB-CLAIM-011",
        "headline": "ImageNet-C SAR harmful point beats-both",
        "artifact": "docs/research/kbound/results_source.json",
        "runner": "docs/research/kbound/scripts/kbtrain.sh",
        "notebook": "notebooks/05_TTA_CIFAR_and_Online.ipynb",
    },
    {
        "claim_id": "KB-CLAIM-027",
        "headline": "Controlled multimodal D33",
        "artifact": "experiments/kbound/results/controlled_multimodal_d33/results.json",
        "runner": "experiments/kbound/controlled_multimodal_d33.py",
        "notebook": "notebooks/08_ELARA_Multimodal_Instantiation.ipynb",
    },
]

NOTEBOOK_CURRICULUM = [
    ("00_KBound_Master_Guide.ipynb", "START HERE — full theory→code→results map (2026-06)"),
    ("00_KBound_Reproduction.ipynb", "Legacy repro on 123-task ELARA archive (partially stale)"),
    ("01_Problem_and_Theory.ipynb", "Theorem validators + impossibility figures"),
    ("02_Knowability_Trichotomy.ipynb", "Adapt/freeze/abstain trichotomy demos"),
    ("03_Harmful_Mixed_Rigor.ipynb", "Harmful + mixed regimes"),
    ("04_Regression_and_Witness.ipynb", "Regression witness constructions"),
    ("05_TTA_CIFAR_and_Online.ipynb", "CIFAR TTA stress grid"),
    ("06_Evidence_and_Drift.ipynb", "Natural-shift evidence channels"),
    ("07_Certificate_and_Calibration.ipynb", "Certificate + conformal calibration"),
    ("08_ELARA_Multimodal_Instantiation.ipynb", "Multimodal / fusion instantiation"),
    ("09_Conclusions_and_Reproducibility.ipynb", "Artifact inventory + rebuild commands"),
]


def _exists(rel: str | None) -> bool:
    if not rel:
        return True
    return (ROOT / rel).exists()


def _load_json(rel: str) -> dict | None:
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text())


def build_report(*, run_validators: bool = False) -> dict[str, Any]:
    theory_rows = []
    for row in THEORY_SPINE:
        arts = {a: _exists(a) for a in row["artifacts"]}
        vals = {v: _exists(v) for v in row["validators"]}
        theory_rows.append({**row, "artifacts_ok": arts, "validators_ok": vals})

    emp_rows = []
    for row in EMPIRICAL_CLAIMS:
        art = row["artifact"]
        detail: dict[str, Any] = {"exists": _exists(art)}
        data = _load_json(art)
        if data and row["claim_id"] == "KB-CLAIM-026":
            detail["verdict"] = data.get("headtohead", {}).get("VERDICT")
            detail["kga_regret"] = data.get("policy_mean_regret", {}).get("kga")
        if data and row["claim_id"] == "KB-CLAIM-024":
            detail["beats_both_robust"] = data.get("beats_both_robust")
            detail["n"] = data.get("n_conditions") or data.get("n")
        if data and row["claim_id"] == "KB-CLAIM-010":
            tent = (data.get("candidates") or {}).get("tent") or {}
            detail["tent_beats_both"] = tent.get("beats_both_robust") or tent.get("beats_both")
        emp_rows.append({**row, "detail": detail})

    ledger_path = KBOUND / "claim_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    claims = {c["claim_id"]: c["status"] for c in ledger.get("claims", [])}

    validator_run: dict[str, str] = {}
    if run_validators:
        for row in THEORY_SPINE:
            for script in row["validators"]:
                sp = ROOT / script
                if not sp.exists():
                    validator_run[script] = "missing"
                    continue
                proc = subprocess.run(
                    [sys.executable, str(sp)],
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                )
                validator_run[script] = "ok" if proc.returncode == 0 else f"fail:{proc.returncode}"

    return {
        "repo_root": str(ROOT),
        "canonical_status": "docs/research/kbound/PROJECT_STATUS_AND_OPEN_PROBLEMS.md",
        "papers": {
            "short": "docs/research/kbound/kbound_short.pdf",
            "long": "docs/research/kbound/kbound.pdf",
        },
        "reproduce": "docs/research/kbound/scripts/reproduce_submission.sh",
        "theory_spine": theory_rows,
        "empirical_headlines": emp_rows,
        "claim_ledger_status": claims,
        "notebook_curriculum": NOTEBOOK_CURRICULUM,
        "validator_run": validator_run or None,
    }


def print_human(report: dict[str, Any]) -> None:
    print("=" * 72)
    print("K-BOUND COMPLETE PICTURE")
    print(f"Repo: {report['repo_root']}")
    print(f"Status doc: {report['canonical_status']}")
    print("=" * 72)

    print("\n## THEORY SPINE (paper label → proof → validator → code)\n")
    for row in report["theory_spine"]:
        ok = all(row["artifacts_ok"].values()) if row["artifacts_ok"] else False
        mark = "✓" if ok else "○"
        print(f"{mark} {row['label']}")
        print(f"   {row['statement']}")
        print(f"   Proof: {row['proof']}")
        for v in row["validators"]:
            vm = "✓" if row["validators_ok"].get(v) else "✗"
            print(f"   {vm} validator: {v}")
        if row["code"]:
            print(f"   Code: {row['code']}")
        if row["claim_ids"]:
            print(f"   Claims: {', '.join(row['claim_ids'])}")
        print()

    print("## EMPIRICAL HEADLINES (claim → artifact → how to rerun)\n")
    for row in report["empirical_headlines"]:
        ex = row["detail"]["exists"]
        mark = "✓" if ex else "✗"
        print(f"{mark} {row['claim_id']}: {row['headline']}")
        print(f"   Artifact: {row['artifact']}")
        if row["detail"]:
            extra = {k: v for k, v in row["detail"].items() if k != "exists"}
            if extra:
                print(f"   Loaded: {extra}")
        print(f"   Rerun: {row['runner']}")
        print(f"   Notebook: {row['notebook']}")
        print()

    print("## NOTEBOOK CURRICULUM (10 K-Bound notebooks + master guide)\n")
    for nb, desc in report["notebook_curriculum"]:
        print(f"  {nb}")
        print(f"    {desc}")
    print()
    print("Reproduce all checks: bash docs/research/kbound/scripts/reproduce_submission.sh")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit JSON report")
    ap.add_argument("--run-validators", action="store_true", help="execute theory validators (slow)")
    args = ap.parse_args()
    report = build_report(run_validators=args.run_validators)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human(report)
    missing = [
        r["artifact"]
        for r in report["empirical_headlines"]
        if not r["detail"]["exists"]
    ]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
