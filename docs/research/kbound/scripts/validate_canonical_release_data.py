#!/usr/bin/env python3
"""Fail-closed validation for K-Bound's canonical release-data surfaces."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
CURRENT_POLICY_REL = (
    "experiments/kbound/results/reconciled_panels_v1/"
    "current_policy_cluster_inference.json"
)
CANONICAL_REL = "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
SOURCE_MANIFEST_REL = "experiments/kbound/results/reconciled_panels_v1/source_manifest.json"

SURFACES = {
    "claim_ledger": "docs/research/kbound/claim_ledger.json",
    "table_manifest": "docs/research/kbound/paper/generated/kbound_result_manifest.json",
    "uniform_verdicts": "docs/research/kbound/paper/generated/uniform_verdicts.json",
    "decision_metrics": (
        "docs/research/kbound/paper/generated/empirical_audit/decision_metrics.json"
    ),
    "result_manifest": "docs/research/kbound/RESULT_MANIFEST.json",
    "results_source": "docs/research/kbound/results_source.json",
    "storage_manifest": "docs/research/kbound/STORAGE_MANIFEST.json",
    "dashboard_snapshot": "docs/research/kbound/dashboard/data/snapshot.json",
    "audit_summary": (
        "docs/research/kbound/audits/empirical_data_quality_2026_08_27/"
        "audit_summary.json"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> Any:
    def reject(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def validate_path_hash_bindings(
    problems: list[str],
    *,
    label: str,
    document: Any,
    relative_path: str,
    actual_hash: str,
    actual_bytes: int,
    required: bool,
) -> None:
    bindings = []
    for node in walk(document):
        if not isinstance(node, dict):
            continue
        if node.get("artifact") == relative_path or node.get("artifact_path") == relative_path:
            if "artifact_sha256" in node or "artifact_bytes" in node:
                bindings.append(node)
    if required and not bindings:
        problems.append(f"{label} has no hashed binding for {relative_path}")
    for node in bindings:
        if node.get("artifact_sha256") != actual_hash:
            problems.append(f"{label} has a stale SHA-256 binding for {relative_path}")
        if node.get("artifact_bytes") != actual_bytes:
            problems.append(f"{label} has a stale byte-count binding for {relative_path}")


def validate_named_hash_bindings(
    problems: list[str],
    *,
    label: str,
    document: Any,
    path_key: str,
    hash_key: str,
    relative_path: str,
    actual_hash: str,
    required: bool,
) -> None:
    bindings = [
        node
        for node in walk(document)
        if isinstance(node, dict) and node.get(path_key) == relative_path and hash_key in node
    ]
    if required and not bindings:
        problems.append(f"{label} has no {path_key}/{hash_key} binding for {relative_path}")
    for node in bindings:
        if node.get(hash_key) != actual_hash:
            problems.append(f"{label} has a stale {hash_key} binding for {relative_path}")


def validate() -> list[str]:
    problems: list[str] = []
    documents: dict[str, Any] = {}
    for label, relative_path in SURFACES.items():
        path = ROOT / relative_path
        if not path.is_file():
            problems.append(f"missing release-data surface: {relative_path}")
            continue
        try:
            documents[label] = strict_json(path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"invalid release-data surface {relative_path}: {exc}")
    if problems:
        return problems

    current_path = ROOT / CURRENT_POLICY_REL
    canonical_path = ROOT / CANONICAL_REL
    source_manifest_path = ROOT / SOURCE_MANIFEST_REL
    current_hash = sha256(current_path)
    current_bytes = current_path.stat().st_size
    canonical_hash = sha256(canonical_path)
    source_manifest_hash = sha256(source_manifest_path)

    current = strict_json(current_path)
    if current.get("schema") != "kbound-current-policy-cluster-inference-v2":
        problems.append("current-policy authority has the wrong schema")
    family = current.get("preregistered_six_comparison_holm") or {}
    if family.get("family_size") != 6 or family.get("alpha") != 0.05:
        problems.append("current-policy authority has the wrong preregistered Holm family")
    for candidate in ("tent", "eata", "sar"):
        row = (current.get("candidates") or {}).get(candidate) or {}
        gate = row.get("gate") or {}
        if gate.get("preregistered_six_comparison_cluster_sensitivity_pass") is not False:
            problems.append(f"{candidate} must not pass the preregistered six-comparison gate")

    for label in (
        "claim_ledger",
        "table_manifest",
        "uniform_verdicts",
        "decision_metrics",
        "result_manifest",
        "results_source",
    ):
        validate_path_hash_bindings(
            problems,
            label=label,
            document=documents[label],
            relative_path=CURRENT_POLICY_REL,
            actual_hash=current_hash,
            actual_bytes=current_bytes,
            required=True,
        )

    for label in (
        "claim_ledger",
        "table_manifest",
        "uniform_verdicts",
        "decision_metrics",
        "result_manifest",
    ):
        validate_named_hash_bindings(
            problems,
            label=label,
            document=documents[label],
            path_key="canonical_panel",
            hash_key="canonical_panel_sha256",
            relative_path=CANONICAL_REL,
            actual_hash=canonical_hash,
            required=True,
        )
        validate_named_hash_bindings(
            problems,
            label=label,
            document=documents[label],
            path_key="source_manifest",
            hash_key="source_manifest_sha256",
            relative_path=SOURCE_MANIFEST_REL,
            actual_hash=source_manifest_hash,
            required=True,
        )

    storage_rows = {
        row.get("expected_location"): row
        for row in documents["storage_manifest"].get("artifacts", [])
        if isinstance(row, dict)
    }
    storage_current = storage_rows.get(CURRENT_POLICY_REL) or {}
    if storage_current.get("sha256") != current_hash:
        problems.append("storage manifest has a stale current-policy SHA-256")
    if storage_current.get("size_bytes") != current_bytes:
        problems.append("storage manifest has a stale current-policy byte count")

    ledger_claims = {
        row.get("claim_id"): row
        for row in documents["claim_ledger"].get("claims", [])
        if isinstance(row, dict)
    }
    controlled_claim = ledger_claims.get("KB-CLAIM-010") or {}
    if controlled_claim.get("status") != "supported":
        problems.append("KB-CLAIM-010 must remain a bounded supported controlled claim")
    forbidden = set(controlled_claim.get("forbidden_wording") or [])
    if "current-policy cluster-robust win" not in forbidden:
        problems.append("KB-CLAIM-010 must forbid a current-policy cluster-robust win")
    if (ledger_claims.get("KB-CLAIM-021") or {}).get("status") != "withheld":
        problems.append("KB-CLAIM-021 iWildCam must remain withheld")

    table = documents["table_manifest"]
    tracks = table.get("tracks") or {}
    for key in ("cifar10c_tent", "cifar10c_eata", "cifar10c_sar", "imagenetc_sar"):
        if (tracks.get(key) or {}).get("ci_robust_beats_both") is not False:
            problems.append(f"{key} must not be promoted as CI-robust beats-both")
    iwild = tracks.get("iwildcam_H_v2") or {}
    if iwild.get("numeric_release_eligible") is not False:
        problems.append("iWildCam must remain numerically ineligible")
    if iwild.get("regret") is not None:
        problems.append("iWildCam release regret must remain null")
    if any(value is not None for value in (iwild.get("decision_counts") or {}).values()):
        problems.append("iWildCam release action counts must remain null")

    result_claim_ids = {
        row.get("claim_id") for row in documents["result_manifest"].get("results", [])
    }
    if "KB-CLAIM-021" in result_claim_ids:
        problems.append("withheld iWildCam must not appear in RESULT_MANIFEST results")
    compat_iwild = (documents["results_source"].get("tracks") or {}).get("iwildcam_H_v2") or {}
    if compat_iwild.get("regret") is not None:
        problems.append("results_source iWildCam regret must remain null")

    dashboard = documents["dashboard_snapshot"]
    if (dashboard.get("meta") or {}).get("current_policy_sha256") != current_hash:
        problems.append("dashboard snapshot has a stale current-policy identity")
    board = dashboard.get("evidence_board") or {}
    controlled = board.get("controlled_wins") or []
    if any(row.get("status") == "verified" for row in controlled):
        problems.append("dashboard must not label current controlled evidence verified")
    if any(row.get("ci_robust_beats_both") is not False for row in controlled):
        problems.append("dashboard controlled rows must preserve negative CI-robust statuses")
    natural_names = {row.get("name") for row in board.get("natural_shift_no_harm") or []}
    if "iWildCam H v2" in natural_names or "Camelyon17 OOD" in natural_names:
        problems.append("dashboard no-harm group contains a withheld or all-helpful diagnostic")

    audit = documents["audit_summary"]
    release_claim = (audit.get("release_decision") or {}).get(
        "controlled_cifar10c_tent_claim", ""
    )
    if "PREREGISTERED_SIX_COMPARISON_HOLM_FAILED" not in release_claim:
        problems.append("empirical audit does not preserve the failed preregistered Holm gate")
    if (audit.get("bottom_line") or {}).get(
        "controlled_cifar10c_preregistered_cluster_win"
    ) is not False:
        problems.append("empirical audit must record no preregistered cluster win")
    current_checksum_rows = [
        row
        for row in (audit.get("release_checksums") or {}).get("rows", [])
        if row.get("path") == CURRENT_POLICY_REL
    ]
    if len(current_checksum_rows) != 1 or current_checksum_rows[0].get(
        "actual_sha256"
    ) != current_hash:
        problems.append("empirical audit has a stale current-policy observed hash")
    return problems


def main() -> int:
    problems = validate()
    if problems:
        print("canonical release data: FAIL")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("canonical release data: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
