#!/usr/bin/env python3
"""Fail-closed validation for K-Bound's canonical release-data surfaces."""

from __future__ import annotations

import hashlib
import importlib.util
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
CCT20_REL = "docs/research/kbound/paper/generated/cct20_release_manifest.json"
CCT20_RECEIPT_REL = CCT20_REL + ".receipt.json"
SO2SAT_REL = (
    "experiments/kbound/results/so2sat_lcz42_prospective_v1/"
    "development_mps_bn_fix_v1/so2sat_candidate_selection.json"
)
SO2SAT_RECEIPT_REL = SO2SAT_REL + ".receipt.json"
SO2SAT_NUMBERS_REL = "docs/research/kbound/paper/generated/so2sat_numbers.tex"
SO2SAT_NUMBERS_BUILDER_REL = "docs/research/kbound/scripts/build_so2sat_numbers.py"
CURRENT_CLUSTER_SCHEMA = "kbound-current-policy-cluster-inference-v3"
FAMILY_FIELD = "retrospective_holm_over_six_prospectively_named_contrasts"
GATE_PASS_FIELD = "retrospective_six_contrast_cluster_sensitivity_pass"

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

    numbers_path = ROOT / SO2SAT_NUMBERS_REL
    builder_path = ROOT / SO2SAT_NUMBERS_BUILDER_REL
    if not numbers_path.is_file() or not builder_path.is_file():
        problems.append("missing generated So2Sat manuscript numbers or builder")
    else:
        try:
            spec = importlib.util.spec_from_file_location("kbound_so2sat_numbers", builder_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("cannot load So2Sat numbers builder")
            builder = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(builder)
            expected_numbers = builder.render_numbers_tex(builder.load_validated_numbers())
            if numbers_path.read_text(encoding="ascii") != expected_numbers:
                problems.append("generated So2Sat manuscript numbers are stale")
        except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
            problems.append(f"cannot validate generated So2Sat manuscript numbers: {exc}")
    for driver_rel in (
        "docs/research/kbound/kbound_submission.tex",
        "docs/research/kbound/kbound_tmlr.tex",
    ):
        driver = ROOT / driver_rel
        if not driver.is_file() or r"\input{paper/generated/so2sat_numbers.tex}" not in driver.read_text(
            encoding="utf-8"
        ):
            problems.append(f"maintained driver does not input So2Sat numbers: {driver_rel}")

    current = strict_json(current_path)
    if current.get("schema") != CURRENT_CLUSTER_SCHEMA:
        problems.append("current-policy authority has the wrong schema")
    family = current.get(FAMILY_FIELD) or {}
    if family.get("family_size") != 6 or family.get("alpha") != 0.05:
        problems.append(
            "current-policy authority has the wrong retrospective Holm family over the six "
            "prospectively named contrasts"
        )
    for candidate in ("tent", "eata", "sar"):
        row = (current.get("candidates") or {}).get(candidate) or {}
        gate = row.get("gate") or {}
        if gate.get(GATE_PASS_FIELD) is not False:
            problems.append(f"{candidate} must not pass the retrospective six-contrast gate")

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

    cct20_path = ROOT / CCT20_REL
    cct20_receipt_path = ROOT / CCT20_RECEIPT_REL
    so2sat_path = ROOT / SO2SAT_REL
    so2sat_receipt_path = ROOT / SO2SAT_RECEIPT_REL
    for path in (cct20_path, cct20_receipt_path, so2sat_path, so2sat_receipt_path):
        if not path.is_file():
            problems.append(f"missing separate release authority: {path.relative_to(ROOT)}")
    if any(not path.is_file() for path in (cct20_path, cct20_receipt_path, so2sat_path, so2sat_receipt_path)):
        return problems

    cct20_hash = sha256(cct20_path)
    so2sat_hash = sha256(so2sat_path)
    cct20 = strict_json(cct20_path)
    cct20_receipt = strict_json(cct20_receipt_path)
    so2sat = strict_json(so2sat_path)
    so2sat_receipt = strict_json(so2sat_receipt_path)
    if (
        cct20_receipt.get("artifact_sha256") != cct20_hash
        or cct20_receipt.get("artifact_bytes") != cct20_path.stat().st_size
    ):
        problems.append("CCT-20 release receipt does not bind the release manifest")
    if (
        so2sat_receipt.get("artifact_sha256") != so2sat_hash
        or so2sat_receipt.get("artifact_bytes") != so2sat_path.stat().st_size
    ):
        problems.append("So2Sat selection receipt does not bind the selection artifact")

    cct_claim = ledger_claims.get("KB-CLAIM-051") or {}
    if (
        cct_claim.get("status") != "no-harm"
        or cct_claim.get("verdict") != "SAFE_UTILITY_ONLY"
        or cct_claim.get("action_counts") != {"ADAPT": 0, "FREEZE": 44, "ABSTAIN": 1}
    ):
        problems.append("KB-CLAIM-051 must remain the bounded CCT-20 safe-utility-only result")
    so2sat_claim = ledger_claims.get("KB-CLAIM-052") or {}
    if (
        so2sat_claim.get("status") != "diagnostic"
        or so2sat_claim.get("verdict") != "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL"
        or (so2sat_claim.get("target_access") or {}).get("target_pixels_read") != 0
        or (so2sat_claim.get("target_access") or {}).get("target_labels_read") != 0
    ):
        problems.append("KB-CLAIM-052 must remain a no-target-access So2Sat development diagnostic")

    if (
        cct20.get("schema") != "kbound_cct20_release_manifest_v1"
        or cct20.get("status") != "RELEASE_COMPLETE"
        or (cct20.get("verdict") or {}).get("code") != "SAFE_UTILITY_ONLY"
        or (cct20.get("verdict") or {}).get("protocol_strong_success") is not False
        or (cct20.get("action_exposure") or {}).get("counts")
        != {"ABSTAIN": 1, "ADAPT": 0, "FREEZE": 44}
    ):
        problems.append("CCT-20 authority drifted from the safe-utility-only verdict")
    if (
        so2sat.get("schema") != "kbound_so2sat_adapter_candidate_selection_v1"
        or so2sat.get("status") != "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL"
        or so2sat.get("selected_candidate_id") is not None
        or so2sat.get("gate_cal_rows_read_before_selection") != 0
        or so2sat.get("target_inputs") != []
        or so2sat.get("target_pixels_read") != 0
        or so2sat.get("target_labels_read") != 0
    ):
        problems.append("So2Sat authority drifted from the no-candidate/no-target-access stop")

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
    if not {"KB-CLAIM-051", "KB-CLAIM-052"} <= result_claim_ids:
        problems.append("RESULT_MANIFEST must include CCT-20 and So2Sat release entries")
    result_rows = {
        row.get("claim_id"): row
        for row in documents["result_manifest"].get("results", [])
        if isinstance(row, dict)
    }
    cct_result = result_rows.get("KB-CLAIM-051") or {}
    cct_metrics = cct_result.get("metrics") or {}
    if (
        cct_result.get("source_artifact") != CCT20_REL
        or cct_metrics.get("artifact_sha256") != cct20_hash
        or cct_metrics.get("decision_counts") != {"ADAPT": 0, "FREEZE": 44, "ABSTAIN": 1}
        or cct_metrics.get("point_beats_both") is not False
        or cct_metrics.get("ci_robust_beats_both") is not False
    ):
        problems.append("RESULT_MANIFEST CCT-20 row is stale or overclaims")
    so2sat_result = result_rows.get("KB-CLAIM-052") or {}
    so2sat_metrics = so2sat_result.get("metrics") or {}
    if (
        so2sat_result.get("source_artifact") != SO2SAT_REL
        or so2sat_metrics.get("artifact_sha256") != so2sat_hash
        or so2sat_metrics.get("selected_candidate_id") is not None
        or so2sat_metrics.get("target_score") is not None
        or (so2sat_metrics.get("target_access") or {}).get("target_pixels_read") != 0
        or (so2sat_metrics.get("target_access") or {}).get("target_labels_read") != 0
    ):
        problems.append("RESULT_MANIFEST So2Sat row is stale or implies target access")

    for label in ("claim_ledger", "result_manifest"):
        authorities = (documents[label].get("reconciliation_source") or {}).get(
            "separate_receipt_linked_authorities"
        ) or {}
        if (
            (authorities.get("cct20") or {}).get("artifact_sha256") != cct20_hash
            or (authorities.get("so2sat_development") or {}).get("artifact_sha256")
            != so2sat_hash
        ):
            problems.append(f"{label} has stale separate CCT-20/So2Sat authorities")
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
    if "RETROSPECTIVE_HOLM_OVER_SIX_PROSPECTIVELY_NAMED_CONTRASTS_FAILED" not in release_claim:
        problems.append(
            "empirical audit does not preserve the failed retrospective Holm gate over the "
            "six prospectively named contrasts"
        )
    if (audit.get("bottom_line") or {}).get(
        "controlled_cifar10c_retrospective_six_contrast_holm_win"
    ) is not False:
        problems.append("empirical audit must record no retrospective six-contrast Holm win")
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
