#!/usr/bin/env python3
"""Build the researcher-facing K-Bound dashboard snapshot.

The paper's generated result manifest is the only source for promoted benchmark
numbers. Physical-study files are read only from the active edge result tree.
Archived or legacy ELARA outputs are intentionally ignored.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from typing import Any

SCRIPT = Path(__file__).resolve()
KBOUND = SCRIPT.parents[1]
REPO = SCRIPT.parents[4]
MANIFEST = KBOUND / "paper" / "generated" / "kbound_result_manifest.json"
CANONICAL_PANEL = REPO / "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
CURRENT_POLICY = REPO / "experiments/kbound/results/reconciled_panels_v1/current_policy_cluster_inference.json"
FORMAL_REGISTRY = KBOUND / "formal" / "formal_audit.py"
EDGE = KBOUND / "edge"
EDGE_RESULTS = REPO / "experiments" / "kbound" / "results" / "edge_real_phone_v1"
OUT = KBOUND / "dashboard" / "data" / "snapshot.json"
SHORT_PDF = KBOUND / "kbound_short_final_draft.pdf"
THEORY_SOURCES = (
    KBOUND / "kbound_submission_body.tex",
    KBOUND / "paper" / "sections" / "theory_core_main.tex",
    KBOUND / "paper" / "sections" / "theory_certificate.tex",
)


def require_resident_file(path: Path) -> os.stat_result:
    """Do not treat a missing, empty, or cloud-only input as verified content."""
    if not path.is_file():
        raise FileNotFoundError(f"Required presentation input is missing: {path}")
    if stat.S_ISLNK(os.lstat(path).st_mode):
        raise ValueError(f"Required presentation input must not be a symlink: {path}")
    info = path.stat()
    if info.st_size <= 0:
        raise ValueError(f"Required presentation input is empty: {path}")
    if getattr(info, "st_blocks", 1) == 0 or getattr(info, "st_flags", 0) & getattr(stat, "SF_DATALESS", 0x40000000):
        raise ValueError(f"Required presentation input is not locally resident: {path}")
    return info


def resident_bytes(path: Path) -> bytes:
    """Read one resident authority, rejecting a concurrent partial refresh."""
    before = require_resident_file(path)
    content = path.read_bytes()
    after = require_resident_file(path)
    if len(content) != before.st_size or (
        before.st_ino, before.st_mtime_ns, before.st_size
    ) != (after.st_ino, after.st_mtime_ns, after.st_size):
        raise ValueError(f"Presentation input changed during read: {path}")
    return content


def resident_json(path: Path) -> tuple[dict[str, Any], str]:
    content = resident_bytes(path)

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"Non-finite JSON value in required presentation input {path}: {value}")

    value = json.loads(content, parse_constant=reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"Required presentation input must be a JSON object: {path}")
    return value, hashlib.sha256(content).hexdigest()


def pdf_page_count(path: Path) -> int:
    """Read the built PDF with Poppler; never guess or retain a stale count."""
    require_resident_file(path)
    tool = shutil.which("pdfinfo")
    if tool is None:
        raise RuntimeError("pdfinfo is required to verify the built PDF page count")
    try:
        result = subprocess.run(
            [tool, str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError(f"Could not verify the built PDF page count: {path}") from exc
    counts = re.findall(r"^Pages:\s+([0-9]+)\s*$", result.stdout, flags=re.MULTILINE)
    if len(counts) != 1 or int(counts[0]) <= 0:
        raise ValueError(f"pdfinfo did not report one positive page count: {path}")
    return int(counts[0])


def theory_statement_counts() -> dict[str, int]:
    """Count maintained theorem-style statements, not definitions or remarks.

    These are paper statements under their stated assumptions. Their count is
    not a claim that all probability foundations have been mechanized in Lean.
    The three explicit sources are the maintained body and its two theory inputs;
    superseded drivers and historical bridge manuscripts are intentionally absent.
    """
    counts = {kind: 0 for kind in ("theorem", "lemma", "proposition", "corollary")}
    pattern = re.compile(r"\\begin\{(theorem|lemma|proposition|corollary)\}")
    for path in THEORY_SOURCES:
        info = require_resident_file(path)
        content = path.read_bytes()
        if len(content) != info.st_size:
            raise ValueError(f"Incomplete presentation-source read: {path}")
        source = re.sub(r"(?<!\\)%[^\n]*", "", content.decode("utf-8"))
        for kind in pattern.findall(source):
            counts[kind] += 1
    if counts["theorem"] == 0:
        raise ValueError("Maintained theory sources contain no numbered theorems")
    return counts


def presentation_metadata() -> tuple[int, dict[str, str]]:
    pages = pdf_page_count(SHORT_PDF)
    counts = theory_statement_counts()
    return pages, {
        "value": f"{counts['theorem']} theorems",
        "sub": f"{sum(counts.values())} numbered statements; stated assumptions apply",
    }


def refresh_presentation_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Refresh two presentation fields only; do not read edge or target data."""
    if not isinstance(snapshot, dict):
        raise ValueError("Presentation refresh requires an existing dashboard snapshot")
    meta = snapshot.get("meta")
    strip = snapshot.get("evidence_strip")
    if not isinstance(meta, dict) or meta.get("paper") != rel(SHORT_PDF):
        raise ValueError("Dashboard metadata does not identify the maintained short PDF")
    if not isinstance(strip, dict) or not isinstance(strip.get("proven_theorems"), dict):
        raise ValueError("Dashboard snapshot lacks the existing theory presentation field")
    pages, theory = presentation_metadata()
    refreshed = copy.deepcopy(snapshot)
    refreshed["meta"]["paper_pages"] = pages
    refreshed["evidence_strip"]["proven_theorems"] = theory
    return refreshed


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
    return hashlib.sha256(resident_bytes(path)).hexdigest()


def paper_authorities() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Load only the three named paper authorities, never manifest-selected data."""
    manifest, manifest_sha = resident_json(MANIFEST)
    reconciliation = manifest.get("reconciliation_source")
    if not isinstance(reconciliation, dict):
        raise ValueError("canonical result manifest is missing reconciliation_source")
    policy_binding = reconciliation.get("current_policy_family_sensitivity")
    if not isinstance(policy_binding, dict):
        raise ValueError("canonical result manifest is missing current-policy binding")
    if reconciliation.get("canonical_panel") != rel(CANONICAL_PANEL):
        raise ValueError("canonical result manifest has an unexpected canonical-panel path")
    if policy_binding.get("artifact") != rel(CURRENT_POLICY):
        raise ValueError("canonical result manifest has an unexpected current-policy artifact path")
    expected_canonical = reconciliation.get("canonical_panel_sha256")
    expected_policy = policy_binding.get("artifact_sha256")
    for name, value in (("canonical-panel", expected_canonical), ("current-policy", expected_policy)):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"canonical result manifest has an invalid {name} SHA256 binding")
    canonical, canonical_sha = resident_json(CANONICAL_PANEL)
    if canonical_sha != expected_canonical:
        raise ValueError("canonical result manifest has a stale canonical-panel binding")
    _, policy_sha = resident_json(CURRENT_POLICY)
    if policy_sha != expected_policy:
        raise ValueError("canonical result manifest has a stale current-policy artifact binding")
    return manifest, canonical, {
        "manifest_sha256": manifest_sha,
        "canonical_panel_sha256": canonical_sha,
        "current_policy_sha256": policy_sha,
    }


def registered_formal_scope() -> dict[str, Any]:
    """Read declared scope without importing the audit or claiming a new build."""
    content = resident_bytes(FORMAL_REGISTRY)
    wanted = {"LEGACY_CORE_THEOREMS", "FOUNDATION_THEOREMS", "FOUNDATION_LAYERS"}
    values: dict[str, Any] = {}
    for node in ast.parse(content, filename=str(FORMAL_REGISTRY)).body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        for name in wanted.intersection(names):
            if name in values:
                raise ValueError(f"Formal registry declares {name} more than once")
            values[name] = ast.literal_eval(node.value)
    if set(values) != wanted:
        raise ValueError("Formal registry is missing declared theorem/foundation scope")
    core = values["LEGACY_CORE_THEOREMS"]
    foundations = values["FOUNDATION_THEOREMS"]
    layers = values["FOUNDATION_LAYERS"]
    if not isinstance(core, list) or not isinstance(foundations, dict) or not isinstance(layers, list):
        raise ValueError("Formal registry has invalid declaration containers")
    if not all(isinstance(names, list) for names in foundations.values()):
        raise ValueError("Formal registry has invalid foundational theorem lists")
    extended = [name for names in foundations.values() for name in names]
    names = [*core, *extended]
    if not names or not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
        raise ValueError("Formal registry has missing or duplicate theorem names")
    if not all(isinstance(layer, dict) for layer in layers):
        raise ValueError("Formal registry has invalid foundational layer descriptions")
    positive = sum(layer.get("status") == "MECHANIZED_WITH_EXPLICIT_ASSUMPTIONS" for layer in layers)
    counterexamples = sum(layer.get("status") == "PARTIAL_COUNTEREXAMPLE_FOUND" for layer in layers)
    if len(layers) != 6 or positive != 5 or counterexamples != 1:
        raise ValueError("Dashboard wording requires the declared five-positive/partial-sixth foundation scope")
    return {
        "registry": rel(FORMAL_REGISTRY),
        "registry_sha256": hashlib.sha256(content).hexdigest(),
        "registered_lean_checks": len(names),
        "legacy_core_checks": len(core),
        "foundational_checks": len(extended),
        "positive_foundational_layers": positive,
        "counterexample_layers": counterexamples,
        "full_foundations_proof": False,
        "verification_note": "Registered source scope; this dashboard refresh does not run Lean or reverify kernel proofs.",
    }


def completed_diagnostic_rows(manifest: dict[str, Any], canonical: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe saved ImageNet-R/PACS coverage without promoting either result."""
    imagenetr = canonical["panels"]["imagenet_r"]["panel"]
    image_track = manifest["tracks"]["imagenet_r_D"]
    seeds = imagenetr.get("seeds")
    candidates = imagenetr.get("candidates")
    if (
        not isinstance(seeds, list) or not seeds
        or not all(type(seed) is int for seed in seeds) or len(set(seeds)) != len(seeds)
        or image_track.get("completed_seeds") != seeds
        or not isinstance(candidates, dict) or not candidates
        or type(imagenetr.get("candidate_count")) is not int
        or imagenetr["candidate_count"] != len(candidates)
        or not isinstance(image_track.get("per_backbone"), dict)
        or set(image_track["per_backbone"]) != set(candidates)
    ):
        raise ValueError("ImageNet-R manifest/canonical seed or backbone coverage is inconsistent")
    pacs = canonical["panels"]["pacs"]
    pacs_track = manifest["tracks"]["pacs"]
    pacs_seeds = pacs.get("seeds")
    if (
        not isinstance(pacs_seeds, list) or not pacs_seeds
        or not all(type(seed) is int for seed in pacs_seeds) or len(set(pacs_seeds)) != len(pacs_seeds)
        or type(pacs_track.get("completed_seeds")) is not int
        or pacs_track["completed_seeds"] != len(pacs_seeds)
        or pacs.get("aggregate_matches_seed_files") is not True
        or pacs.get("decision_replay_available") is not False
        or pacs_track.get("decision_replay_available") is not False
        or not isinstance(pacs.get("decision_replay_blocker"), str) or not pacs["decision_replay_blocker"]
    ):
        raise ValueError("PACS manifest/canonical aggregate or incomplete-replay scope is inconsistent")
    return [
        {
            "name": "ImageNet-R Protocol D",
            "status": "diagnostic",
            "artifact": rel(MANIFEST),
            "completed_seed_count": len(seeds),
            "backbone_count": len(candidates),
            "framing": f"{len(seeds)} completed run seeds across {len(candidates)} backbones; an architecture-panel diagnostic, not one deployable policy.",
            "note": "No natural-shift superiority is claimed; a benchmark null does not prove structural non-identifiability.",
        },
        {
            "name": "PACS",
            "status": "diagnostic",
            "artifact": rel(MANIFEST),
            "completed_seed_count": len(pacs_seeds),
            "decision_replay_available": False,
            "framing": f"{len(pacs_seeds)}-seed aggregate agrees with the archived seed summaries; cell-level decisions remain unreplayable.",
            "note": pacs["decision_replay_blocker"],
        },
    ]


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


def build_paper_projection() -> dict[str, Any]:
    """Build paper fields from resident authorities without inspecting edge data."""
    pages, theory = presentation_metadata()
    manifest, canonical, bindings = paper_authorities()
    formal_scope = registered_formal_scope()
    diagnostic_rows = completed_diagnostic_rows(manifest, canonical)

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
        *diagnostic_rows,
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
            "evidence": "Lean includes measurable target-law and testing constructions under their explicit assumptions; empirical population assumptions are not inferred.",
        },
        {
            "id": "P1",
            "name": "Closed-band abstention",
            "status": "verified",
            "artifact": "docs/research/kbound/kbound_submission_body.tex",
            "implication": "Under strict-action semantics, abstention is maximal on |M| <= beta.",
            "evidence": "Boundary distinguishes zero-versus-strict ambiguity from the interior construction.",
        },
        {
            "id": "T2",
            "name": "Strict-commitment frontier",
            "status": "verified",
            "artifact": "docs/research/kbound/formal/KBound/Probability/MeasureFrontier.lean",
            "implication": "A uniform strict action is supportable exactly outside the declared uncertainty band.",
            "evidence": "Lean covers the feasible measurable target-law class; arbitrary restricted deployment classes are not automatically covered.",
        },
        {
            "id": "T3",
            "name": "Marginal false-adapt certificate",
            "status": "conditional",
            "artifact": "docs/research/kbound/formal/KBound/Probability/MeasureConformal.lean",
            "implication": "Interval coverage controls FA_u, not conditional FA_c.",
            "evidence": "Lean derives one-shot residual coverage under exchangeable score laws; this does not verify benchmark exchangeability, population improvement, or repeated-use coverage.",
        },
        {
            "id": "M1",
            "name": "Multiclass bridge",
            "status": "conditional",
            "artifact": "docs/research/kbound/kbound_submission_body.tex",
            "implication": "Delta = P(D)(p_a-p_0); empirical KGA estimates Delta directly.",
            "evidence": "The converse frontier requires declared-class richness.",
        },
    ]

    release_date = manifest.get("regenerated_utc")
    if not isinstance(release_date, str) or len(release_date) != 10:
        raise ValueError("canonical result manifest must provide YYYY-MM-DD regenerated_utc")
    canonical_sha = bindings["canonical_panel_sha256"]
    policy_sha = bindings["current_policy_sha256"]

    return {
        "meta": {
            "build_id": release_date.replace("-", "") + "T000000Z",
            "generated_at": release_date + "T00:00:00Z",
            "commit": None,
            "canonical_panel_sha256": canonical_sha,
            "current_policy_sha256": policy_sha,
            "paper": rel(SHORT_PDF),
            "paper_pages": pages,
        },
        "research_status": {
            "theory": "verified",
            "controlled": "conditional",
            "natural_shifts": "diagnostic",
            "edge_study": "pending",
        },
        "evidence_strip": {
            "proven_theorems": theory,
            "theorem_validators": {
                "value": f"{formal_scope['registered_lean_checks']} registered Lean checks",
                "sub": f"{formal_scope['positive_foundational_layers']} positive foundational layers; stated assumptions apply",
            },
            "controlled_beats_both": {
                "value": "3 point-estimate tracks",
                "sub": "No current track has a promoted CI-robust or preregistered cluster win",
            },
            "natural_shift_no_harm": {
                "value": "2 ties",
                "sub": "Office-Home and RxRx1; Camelyon is diagnostic and iWildCam is withheld",
            },
            "open_theory": {
                "value": "historical sixth layer",
                "sub": "An orbit/fibre counterexample is proved; the broader sufficiency and H-rate extension is not.",
            },
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
                "examples": "CIFAR-10.1, ImageNet-R, PACS aggregate-only replay",
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
            "current_policy_sha256": policy_sha,
            "manifest_sha256": bindings["manifest_sha256"],
            "formal_scope": formal_scope,
            "local_clips_note": "Raw physical clips remain local; manifests and hashes are release artifacts after privacy review.",
        },
    }


def build_snapshot() -> dict[str, Any]:
    """The default full build still performs the active physical-edge checks."""
    snapshot = build_paper_projection()
    snapshot["edge_validation"] = edge_status()
    snapshot["provenance"]["refresh_mode"] = "full"
    snapshot["provenance"]["edge_validation_refresh"] = {
        "checked_this_run": True,
        "mode": "active_edge_inputs",
    }
    return snapshot


def validated_saved_edge(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Validate the saved JSON shape, not the underlying physical evidence."""
    if not isinstance(snapshot, dict):
        raise ValueError("Paper-only refresh requires an existing dashboard snapshot")
    meta = snapshot.get("meta")
    if not isinstance(meta, dict) or meta.get("paper") != rel(SHORT_PDF):
        raise ValueError("Dashboard metadata does not identify the maintained short PDF")
    edge = snapshot.get("edge_validation")
    if (
        not isinstance(edge, dict)
        or not isinstance(edge.get("study_status"), str)
        or edge["study_status"] not in {"pending", "verified"}
    ):
        raise ValueError("Paper-only refresh requires a valid saved edge_validation object")
    if not isinstance(edge.get("study_label"), str) or not edge["study_label"].strip():
        raise ValueError("Saved edge_validation lacks its study label")
    phases = edge.get("phases")
    if not isinstance(phases, list) or not phases or not all(
        isinstance(row, dict)
        and all(isinstance(row.get(key), str) and row[key] for key in ("id", "label", "status", "detail", "artifact"))
        and row["status"] in {"pending", "verified"}
        for row in phases
    ):
        raise ValueError("Saved edge_validation has invalid phases")
    progress = edge.get("session_progress")
    if not isinstance(progress, list) or not progress or not all(
        isinstance(row, dict) and isinstance(row.get("session"), str) and row["session"]
        and all(type(row.get(key)) is int and row[key] >= 0 for key in ("expected_clips", "captured_clips"))
        and type(row.get("complete")) is bool
        for row in progress
    ):
        raise ValueError("Saved edge_validation has invalid session progress")
    unblock = edge.get("unblock")
    if not isinstance(unblock, dict) or type(unblock.get("all_pass")) is not bool or type(edge.get("audit_pass")) is not bool:
        raise ValueError("Saved edge_validation lacks boolean gate/audit status")
    if (edge["study_status"] == "verified") != unblock["all_pass"] or (unblock["all_pass"] and not edge["audit_pass"]):
        raise ValueError("Saved edge_validation has contradictory study/gate status")
    current = unblock.get("current")
    thresholds = unblock.get("gate_thresholds")
    gaps = unblock.get("gaps")
    commands = unblock.get("commands")
    if (
        not isinstance(current, dict)
        or not all(type(current.get(key)) is bool for key in ("sessions_complete", "physical_only", "source_gate", "audit_pass"))
        or not isinstance(thresholds, dict)
        or not all(type(thresholds.get(key)) in (int, float) and 0 < thresholds[key] <= 1 for key in ("balanced_acc", "macro_f1"))
        or not isinstance(gaps, list)
        or not all(isinstance(row, dict) and type(row.get("passed")) is bool and isinstance(row.get("check"), str) and isinstance(row.get("detail"), str) for row in gaps)
        or not isinstance(commands, dict) or not commands
        or not all(isinstance(value, str) and value for value in commands.values())
        or "development_metrics" not in edge
        or (edge["development_metrics"] is not None and not isinstance(edge["development_metrics"], dict))
        or "protocol_hash" not in edge
        or (edge["protocol_hash"] is not None and not isinstance(edge["protocol_hash"], str))
    ):
        raise ValueError("Saved edge_validation has an invalid cached evidence structure")
    # Reject non-standard numeric values even in extra cached diagnostic fields.
    json.dumps(edge, allow_nan=False)
    return edge


def refresh_paper_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Refresh paper fields, preserving cached edge evidence without a recheck."""
    edge = validated_saved_edge(snapshot)
    refreshed = build_paper_projection()
    refreshed["edge_validation"] = copy.deepcopy(edge)
    refreshed["research_status"]["edge_study"] = edge["study_status"]
    refreshed["meta"]["refresh_mode"] = "paper-only"
    refreshed["provenance"]["refresh_mode"] = "paper-only"
    refreshed["provenance"]["edge_validation_refresh"] = {
        "checked_this_run": False,
        "mode": "preserved_not_rechecked",
        "preserved_edge_canonical_json_sha256": hashlib.sha256(
            json.dumps(edge, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "source_snapshot_canonical_json_sha256": hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "source_snapshot_generated_at": snapshot["meta"].get("generated_at"),
        "note": "Saved edge_validation copied unchanged; no physical-edge files, sessions, or publication gates were rechecked by this paper-only refresh.",
    }
    return refreshed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--metadata-only",
        action="store_true",
        help="refresh built-PDF pages and scoped statement counts; preserve all evidence/edge fields",
    )
    mode.add_argument(
        "--paper-only",
        action="store_true",
        help="refresh resident paper/benchmark authorities; preserve existing edge_validation without rechecking physical inputs",
    )
    args = parser.parse_args(argv)
    if args.metadata_only:
        require_resident_file(OUT)
        snapshot = refresh_presentation_metadata(load(OUT))
    elif args.paper_only:
        existing, _ = resident_json(OUT)
        snapshot = refresh_paper_snapshot(existing)
    else:
        snapshot = build_snapshot()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"[dashboard] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
