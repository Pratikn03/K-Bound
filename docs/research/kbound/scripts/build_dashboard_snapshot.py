#!/usr/bin/env python3
"""Build the researcher-facing K-Bound dashboard snapshot.

The paper's generated result manifest is the only source for promoted benchmark
numbers. Physical-study files are read only from the active edge result tree.
Archived or legacy ELARA outputs are intentionally ignored.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
KBOUND = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
MANIFEST = KBOUND / "paper" / "generated" / "kbound_result_manifest.json"
EDGE = KBOUND / "edge"
EDGE_RESULTS = REPO / "experiments" / "kbound" / "results" / "edge_real_phone_v1"
OUT = KBOUND / "dashboard" / "data" / "snapshot.json"


def load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regret_row(
    manifest: dict[str, Any],
    key: str,
    name: str,
    status: str,
    framing: str,
) -> dict[str, Any]:
    track = manifest["tracks"][key]
    kga, adapt, freeze = track["regret"]
    return {
        "name": name,
        "status": status,
        "artifact": track.get("source", rel(MANIFEST)),
        "framing": framing,
        "freeze": freeze,
        "adapt": adapt,
        "kga": kga,
        "oracle": 0.0,
        "regret_kga": kga,
        "regret_adapt": adapt,
        "regret_freeze": freeze,
        "false_adapt": track.get("false_adapt", track.get("false_adapt_unconditional")),
        "point_beats_both": track.get("point_beats_both"),
        "ci_robust_beats_both": track.get("ci_robust_beats_both"),
        # Retained for the existing dashboard schema.  It is intentionally tied
        # to the explicit inferential flag, never to a substring in prose (for
        # example, "not a beats-both result").
        "beats_both_artifact": track.get("ci_robust_beats_both") is True,
    }


def session_progress() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw = EDGE / "artifacts_real" / "raw"
    checklists = EDGE / "artifacts_real" / "checklists"
    for number in range(1, 11):
        sid = f"S{number:02d}"
        checklist = checklists / f"{sid}_checklist.csv"
        expected = 0
        if checklist.is_file():
            with checklist.open(encoding="utf-8") as fh:
                expected = sum(1 for _ in csv.DictReader(fh))
        captured = len(list((raw / sid).glob("*.mp4"))) if (raw / sid).is_dir() else 0
        rows.append(
            {
                "session": sid,
                "expected_clips": expected,
                "captured_clips": captured,
                "complete": expected > 0 and captured >= expected,
            }
        )
    return rows


def edge_status() -> dict[str, Any]:
    lock = load(EDGE / "artifacts_real" / "protocol_lock.json") or {}
    model = load(EDGE_RESULTS / "model_card.json") or {}
    heldout = load(EDGE_RESULTS / "heldout_metrics.json") or {}
    replication = load(EDGE_RESULTS / "replication_metrics.json") or {}
    audit = load(EDGE_RESULTS / "anti_leakage_audit.json") or {}
    inventory = load(EDGE_RESULTS / "recording_inventory.json") or {}
    gate = load(EDGE_RESULTS / "publication_gate.json") or {}

    progress = session_progress()
    all_sessions = bool(progress) and all(row["complete"] for row in progress)
    clips = inventory.get("clips") or []
    physical_only = bool(clips) and all(row.get("capture_mode") == "physical" for row in clips)
    audit_pass = bool(audit.get("checks")) and all(
        bool(row.get("passed")) for row in audit.get("checks", [])
    )
    metrics = model.get("metrics") or {}
    source_gate = (
        float(metrics.get("val_balanced_acc", 0.0)) >= 0.80
        and float(metrics.get("val_macro_f1", 0.0)) >= 0.80
    )
    heldout_present = bool(heldout.get("n_windows"))
    replication_present = bool(replication.get("n_windows"))
    publication_pass = bool(gate.get("passed")) and all(
        (all_sessions, physical_only, audit_pass, source_gate, heldout_present, replication_present)
    )

    phases = [
        {
            "id": "protocol",
            "label": "Protocol lock",
            "status": "verified" if lock else "pending",
            "detail": "Configuration and hash are frozen before capture.",
            "artifact": rel(EDGE / "artifacts_real" / "protocol_lock.json"),
        },
        {
            "id": "capture",
            "label": "Fresh physical sessions S01-S10",
            "status": "verified" if all_sessions and physical_only else "pending",
            "detail": "Raw clips must be physical, complete, hashed, and split by day/device.",
            "artifact": rel(EDGE / "PHYSICAL_STUDY_RUNBOOK.md"),
        },
        {
            "id": "source",
            "label": "Source-model quality gate",
            "status": "verified" if source_gate else "pending",
            "detail": "S02 balanced accuracy and macro-F1 must both be at least 0.80.",
            "artifact": rel(EDGE_RESULTS / "model_card.json"),
        },
        {
            "id": "heldout",
            "label": "Held-out Phone A replay",
            "status": "verified" if heldout_present and audit_pass else "pending",
            "detail": "S07-S08 are opened only after development and conformal calibration are sealed.",
            "artifact": rel(EDGE_RESULTS / "heldout_metrics.json"),
        },
        {
            "id": "replication",
            "label": "Phone B replication and publication gate",
            "status": "verified" if publication_pass else "pending",
            "detail": "S09-S10, strict anti-leakage audit, report, and table export.",
            "artifact": rel(EDGE_RESULTS / "publication_gate.json"),
        },
    ]

    held_metrics = heldout.get("kga_full_metrics") or {}
    bootstrap = (heldout.get("bootstrap_results") or {}).get("kga_full") or {}
    development_metrics = None
    if model or heldout:
        development_metrics = {
            "note": (
                "Diagnostic only until publication_gate.json passes. Browser previews, pilots, "
                "mock captures, and chance-level replays are not physical-study evidence."
            ),
            "phone_a_balanced_acc": (bootstrap.get("balanced_acc") or {}).get("val"),
            "phone_a_macro_f1": (bootstrap.get("macro_f1") or {}).get("val"),
            "kga_abstain_rate": held_metrics.get("abstain_rate"),
            "latency_ms_mean": held_metrics.get("latency_ms_mean"),
            "latency_ms_p95": held_metrics.get("latency_ms_p95"),
        }

    checks = [
        ("Fresh physical S01-S10 captures", all_sessions and physical_only, "no mock or pilot clips"),
        ("Source-model gate", source_gate, "balanced accuracy and macro-F1 >= 0.80"),
        ("Strict anti-leakage audit", audit_pass, "all eight checks pass"),
        ("Phone A held-out replay", heldout_present, "S07-S08 metrics exist"),
        ("Phone B replication", replication_present, "S09-S10 metrics exist"),
        ("Publication gate", bool(gate.get("passed")), "machine-readable final gate passes"),
    ]
    return {
        "study_status": "verified" if publication_pass else "pending",
        "study_label": "Physical study complete" if publication_pass else "Pre-registered / awaiting fresh physical sessions",
        "phases": phases,
        "session_progress": progress,
        "development_metrics": development_metrics,
        "unblock": {
            "all_pass": publication_pass,
            "gate_thresholds": {"balanced_acc": 0.80, "macro_f1": 0.80},
            "current": {
                "sessions_complete": all_sessions,
                "physical_only": physical_only,
                "source_gate": source_gate,
                "audit_pass": audit_pass,
            },
            "gaps": [
                {"check": label, "passed": passed, "detail": detail}
                for label, passed, detail in checks
            ],
            "commands": {
                "preflight": "python docs/research/kbound/edge/scripts/preflight_r2.py",
                "full_pipeline": "bash docs/research/kbound/edge/scripts/run_edge_publication_pipeline.sh",
                "retrain_source": "bash docs/research/kbound/edge/scripts/run_edge_source_gate.sh",
                "replay_heldout": (
                    "python docs/research/kbound/edge/scripts/06_replay_heldout.py "
                    "--config docs/research/kbound/edge/configs/edge_real_phone_v1.yaml"
                ),
                "refresh_dashboard": "bash docs/research/kbound/scripts/build_dashboard.sh",
            },
        },
        "protocol_hash": lock.get("protocol_hash"),
        "audit_pass": audit_pass,
    }


def build_snapshot() -> dict[str, Any]:
    manifest = load(MANIFEST)
    if not manifest:
        raise FileNotFoundError(f"Canonical result manifest is missing: {MANIFEST}")

    controlled = [
        regret_row(
            manifest,
            "cifar10c_tent",
            "CIFAR-10-C stress / Tent",
            "conditional",
            (
                "Point estimate is below both fixed policies and ordinary six-family intervals "
                "are positive, but p-values from retrospective Holm adjustment over the six "
                "prospectively named contrasts are both 0.09375."
            ),
        ),
        regret_row(
            manifest,
            "cifar10c_eata",
            "CIFAR-10-C stress / EATA",
            "diagnostic",
            (
                "Point estimate is below both fixed policies, but the ordinary family interval "
                "against always-adapt crosses zero and the retrospective Holm gate over the "
                "six prospectively named contrasts fails."
            ),
        ),
        regret_row(
            manifest,
            "imagenetc_sar",
            "ImageNet-C / SAR",
            "no_harm",
            (
                "Pooled point estimate is below both fixed policies, with one false adaptation "
                "in 135 cells; the freeze-side seed interval touches zero."
            ),
        ),
    ]
    constructed = regret_row(
        manifest,
        "three_source_oof",
        "Constructed three-source OOF stream",
        "conditional",
        "Researcher-constructed heterogeneous stream; routing evidence, not unseen-domain transfer.",
    )

    natural = [
        regret_row(manifest, "officehome_M_v2", "Office-Home M v2", "no_harm", "No-harm; ties the safer fixed policy within the declared criterion."),
        regret_row(manifest, "rxrx1_J", "RxRx1 J", "no_harm", "Locked no-harm result; always-freeze is already optimal."),
    ]

    c101 = manifest["tracks"]["cifar10_1_K"]
    camelyon = regret_row(
        manifest,
        "camelyon17_ood",
        "Camelyon17 OOD",
        "diagnostic",
        "Opened all-helpful OOD diagnostic; KGA ties always-adapt on all 18 conditions.",
    )
    camelyon["note"] = "Not prospective, not beats-both, and not a non-vacuous safety result."
    boundary = [
        {
            "name": "iWildCam H v2",
            "status": "withheld",
            "artifact": "docs/research/kbound/claim_ledger.json#KB-CLAIM-021",
            "framing": "Numerical and action evidence is not release-eligible.",
            "note": "The archived scorer violates the official WILDS metric contract; a population-sealed official-metric rerun is required.",
        },
        camelyon,
        {
            "name": "CIFAR-10.1",
            "status": "diagnostic",
            "artifact": rel(MANIFEST),
            "framing": "Transfer bar fails; not promoted as a policy win.",
            "regret_kga": c101["regret"][0],
            "false_adapt": c101["false_adapt_unconditional"],
            "note": "Consistent with weak evidence, low margin, estimator inadequacy, or calibration failure.",
        },
        {
            "name": "ImageNet-R Protocol D",
            "status": "diagnostic",
            "artifact": rel(MANIFEST),
            "framing": "Three of four planned seeds complete; no stable CI-robust beats-both result.",
            "note": "A diagnostic null, not proof of structural non-identifiability.",
        },
        {
            "name": "PACS",
            "status": "pending",
            "artifact": rel(MANIFEST),
            "framing": "One of three planned seeds complete; breadth evidence remains incomplete.",
        },
        {
            **constructed,
            "name": "Constructed three-source OOF stream",
            "status": "conditional",
            "note": "CI beats both, but the stream is researcher-constructed and does not establish transfer.",
        },
    ]

    theory_ledger = [
        {
            "id": "T1",
            "name": "Interior matched-evidence impossibility",
            "status": "verified",
            "artifact": "docs/research/kbound/paper/sections/theory_core_main.tex",
            "implication": "For beta > 0 and |M| < beta, evidence-identical target laws can have opposite nonzero benefit.",
            "evidence": "Paper proof; Lean covers supporting algebra, not the full target-law construction.",
        },
        {
            "id": "P1",
            "name": "Closed-band abstention",
            "status": "verified",
            "artifact": "docs/research/kbound/kbound_short.tex",
            "implication": "Under strict-action semantics, abstention is maximal on |M| <= beta.",
            "evidence": "Boundary distinguishes zero-versus-strict ambiguity from the interior construction.",
        },
        {
            "id": "T2",
            "name": "Strict-commitment frontier",
            "status": "verified",
            "artifact": "docs/research/kbound/formal/KBound/Frontier.lean",
            "implication": "A uniform strict action is supportable exactly outside the declared drift band.",
            "evidence": "Lean checks the sufficiency spine; richness/necessity assumptions remain paper-level.",
        },
        {
            "id": "T3",
            "name": "Marginal false-adapt certificate",
            "status": "conditional",
            "artifact": "docs/research/kbound/formal/KBound/Certificate.lean",
            "implication": "Interval coverage controls FA_u, not conditional FA_c.",
            "evidence": "Coverage is the premise; exchangeability or shift correction is external support for coverage.",
        },
        {
            "id": "M1",
            "name": "Multiclass bridge",
            "status": "conditional",
            "artifact": "docs/research/kbound/kbound_short.tex",
            "implication": "Delta = P(D)(p_a-p_0); empirical KGA estimates Delta directly.",
            "evidence": "The converse frontier requires declared-class richness.",
        },
    ]

    release_date = manifest.get("regenerated_utc")
    if not isinstance(release_date, str) or len(release_date) != 10:
        raise ValueError("canonical result manifest must provide YYYY-MM-DD regenerated_utc")
    reconciliation = manifest.get("reconciliation_source") or {}
    canonical_sha = reconciliation.get("canonical_panel_sha256")
    current_policy = reconciliation.get("current_policy_family_sensitivity") or {}
    if not isinstance(canonical_sha, str) or len(canonical_sha) != 64:
        raise ValueError("canonical result manifest is missing canonical_panel_sha256")
    if canonical_sha != sha256(REPO / reconciliation["canonical_panel"]):
        raise ValueError("canonical result manifest has a stale canonical-panel binding")
    if current_policy.get("artifact_sha256") != sha256(REPO / current_policy["artifact"]):
        raise ValueError("canonical result manifest has a stale current-policy artifact binding")

    return {
        "meta": {
            "build_id": release_date.replace("-", "") + "T000000Z",
            "generated_at": release_date + "T00:00:00Z",
            "commit": None,
            "canonical_panel_sha256": canonical_sha,
            "current_policy_sha256": current_policy["artifact_sha256"],
            "paper": "docs/research/kbound/kbound_short_final_draft.pdf",
            "paper_pages": 22,
        },
        "research_status": {
            "theory": "verified",
            "controlled": "conditional",
            "natural_shifts": "diagnostic",
            "edge_study": "pending",
        },
        "evidence_strip": {
            "proven_theorems": {"value": "3 core", "sub": "plus multiclass/regression bridges"},
            "theorem_validators": {"value": "Lean partial", "sub": "kernel-checked spine; external assumptions disclosed"},
            "controlled_beats_both": {
                "value": "3 point-estimate tracks",
                "sub": "No current track has a promoted CI-robust or preregistered cluster win",
            },
            "natural_shift_no_harm": {
                "value": "2 ties",
                "sub": "Office-Home and RxRx1; Camelyon is diagnostic and iWildCam is withheld",
            },
            "open_theory": {"value": "foundations", "sub": "full probability mechanization remains incomplete"},
            "reproducibility": {"value": "manifest-backed", "sub": "one promoted-number source"},
        },
        "regime_map": [
            {
                "id": "mixed",
                "title": "Mixed and detectable",
                "action": "Route by certificate",
                "status": "verified",
                "examples": "CIFAR-10-C stress; ImageNet-C SAR",
                "artifact": rel(MANIFEST),
            },
            {
                "id": "one-sided",
                "title": "Natural one-sided shifts",
                "action": "Prevent damage; often tie",
                "status": "no_harm",
                "examples": "Office-Home, RxRx1",
                "artifact": rel(MANIFEST),
            },
            {
                "id": "withheld",
                "title": "Withheld evidence",
                "action": "Rerun under the official metric contract",
                "status": "withheld",
                "examples": "iWildCam",
                "artifact": "docs/research/kbound/claim_ledger.json#KB-CLAIM-021",
            },
            {
                "id": "weak",
                "title": "Weak or non-transferable evidence",
                "action": "Abstain or report diagnostic failure",
                "status": "diagnostic",
                "examples": "CIFAR-10.1, ImageNet-R, incomplete PACS",
                "artifact": rel(MANIFEST),
            },
        ],
        "theory_ledger": theory_ledger,
        "headline_controlled": controlled,
        "evidence_board": {
            "controlled_wins": controlled,
            "helpful_dominated": [],
            "natural_shift_no_harm": natural,
            "boundary_negative": boundary,
        },
        "edge_validation": edge_status(),
        "safety": {
            "metrics": [
                {
                    "label": "FA_u",
                    "value": "P(adapt and Delta <= 0)",
                    "meaning": "The marginal false-adapt quantity controlled under valid interval coverage.",
                },
                {
                    "label": "FA_c",
                    "value": "P(Delta <= 0 | adapt)",
                    "meaning": "Descriptive unless separately proved; not theorem-controlled here.",
                },
                {
                    "label": "Abstain",
                    "value": "retain f0",
                    "meaning": "Do not commit the update; prediction continues from the frozen fallback.",
                },
            ],
            "prose": {
                "false_adapt": "The empirical certificate is Delta_hat +/- epsilon; epsilon is not beta.",
                "abstain": "Empirical abstention may reflect structural ambiguity, finite data, model inadequacy, transfer failure, or conservative width.",
                "unknowable": "A benchmark null alone does not establish structural non-identifiability.",
                "certificate_scope": "Every claim is limited to its saved protocol, calibration unit, adapter, and artifact lineage.",
            },
        },
        "reproduce": {
            "primary": "bash docs/research/kbound/scripts/reproduce_submission.sh",
            "gpu": "bash docs/research/kbound/scripts/kbtrain.sh smoke-all",
            "validators": "cd docs/research/kbound/formal && bash build.sh",
            "dashboard": "bash docs/research/kbound/scripts/build_dashboard.sh",
            "runtime_estimate": "Cached audit is CPU-friendly; dataset refreshes require the documented data and accelerator.",
            "inputs": [rel(MANIFEST), "research_lock/", "experiments/kbound/results/"],
            "outputs": [rel(OUT), "docs/research/kbound/kbound_short_final_draft.pdf"],
        },
        "provenance": {
            "snapshot_path": rel(OUT),
            "manifest": rel(MANIFEST),
            "headline_lock": "research_lock/",
            "edge_protocol_lock": rel(EDGE / "artifacts_real" / "protocol_lock.json"),
            "commit": None,
            "canonical_panel_sha256": canonical_sha,
            "current_policy_sha256": current_policy["artifact_sha256"],
            "local_clips_note": "Raw physical clips remain local; manifests and hashes are release artifacts after privacy review.",
        },
    }


def main() -> int:
    snapshot = build_snapshot()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"[dashboard] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
