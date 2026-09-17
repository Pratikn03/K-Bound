#!/usr/bin/env python3
"""Verify the sealed CCT-20 prospective real-population evidence bundle.

This verifier does not rerun training, target inference, or scoring.  It binds
the already sealed execution, label-free target manifest, post-target score,
and inference documents to their byte receipts and checks the protocol fields
that make the result prospective at the camera-location level.  Exchangeability
is recorded as a protocol assumption; it is never promoted to an empirical
fact by this check.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.kbound.cct20.integrity import (
    IntegrityError,
    file_sha256,
    require_sha256,
    stable_sha256,
)
from experiments.kbound.cct20.protocol_seal import (
    AUTHORITATIVE_PROTOCOL_DOCUMENT_SHA256,
    AUTHORITATIVE_PROTOCOL_FILE_SHA256,
    EXPECTED_CLASS_COUNT,
    EXPECTED_LOCATION_COUNT,
    EXPECTED_MODEL_SEEDS,
    EXPECTED_TARGET_IMAGES,
    EXPECTED_TARGET_LOCATIONS,
    LABEL_CONTRACT_ADDENDUM_SHA256,
    METADATA_ADDENDUM_SHA256,
    PROTOCOL_ID,
    TARGET_SELECTION_SHA256,
    _validate_protocol_config,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from experiments.kbound.cct20.prospective_data import validate_locked_target_population
from experiments.kbound.cct20.two_way_inference import analyze_score_document


class BridgeIntegrityError(IntegrityError):
    """Raised when the prospective bundle cannot be verified fail-closed."""


DEFAULT_SOURCE_SNAPSHOT = Path("/Volumes/T9/kbound-confirmatory-source-v2")
DEFAULT_EXTERNAL_ROOT = Path(
    "/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results/cct20_prospective_v1"
)
RELEASE_DISCLOSURE = (
    "outcome-unopened before model execution; aggregate target metadata had already been "
    "inspected during candidate ranking, so this is not described as literally label-unopened"
)
EXPECTED_FIREWALL = {
    "target_runner_imports_scorer": False,
    "target_label_fields_in_prediction_artifacts": False,
    "predictions_and_actions_sealed_before_scoring": True,
    "all_results_reported_regardless_of_direction": True,
}


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeIntegrityError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BridgeIntegrityError(f"JSON artifact is not an object: {path}")
    return value


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise BridgeIntegrityError(f"{label} is not a regular file: {path}")
    return path


def _assert_sha(value: Any, expected: str, label: str) -> None:
    try:
        observed = require_sha256(value, field=label)
    except IntegrityError as exc:
        raise BridgeIntegrityError(str(exc)) from exc
    if observed != expected:
        raise BridgeIntegrityError(f"{label} mismatch: {observed} != {expected}")


def _mapped_sealed_path(path: str | Path, source_snapshot: Path) -> Path:
    """Map paths from the historical checkout to its mounted source snapshot."""

    raw = str(Path(path).expanduser())
    old_root = "/Users/pratik_n/Documents/AutoML_Flagship_V8"
    if raw == old_root or raw.startswith(old_root + "/"):
        return source_snapshot / raw[len(old_root) :].lstrip("/")
    return Path(raw)


def _verify_receipted(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _regular_file(path, label)
    try:
        receipt = verify_artifact_receipt(path)
    except IntegrityError as exc:
        raise BridgeIntegrityError(f"{label} receipt failed: {exc}") from exc
    return _json(path), receipt


def _verify_plain_json(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    return _json(path)


def _validate_checkpoint_audit_metadata(audit: Mapping[str, Any]) -> None:
    if (
        audit.get("schema") != "kbound_cct20_independent_checkpoint_audit_v1"
        or audit.get("status") != "PASS"
        or audit.get("required_model_seeds") != list(EXPECTED_MODEL_SEEDS)
        or audit.get("n_checkpoints") != len(EXPECTED_MODEL_SEEDS)
    ):
        raise BridgeIntegrityError("checkpoint audit is not the sealed five-seed PASS document")
    rows = audit.get("checkpoints")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_MODEL_SEEDS):
        raise BridgeIntegrityError("checkpoint audit must contain five rows")
    if [row.get("model_seed") for row in rows if isinstance(row, Mapping)] != list(EXPECTED_MODEL_SEEDS):
        raise BridgeIntegrityError("checkpoint audit seeds are not ordered 0..4")
    for field in (
        "file_sha256",
        "tensor_sha256",
        "initial_tensor_sha256",
        "config_sha256",
        "config_recipe_sha256",
        "imagenet_backbone_tensor_sha256",
        "data_sha256",
        "code_sha256",
    ):
        for row in rows:
            if not isinstance(row, Mapping):
                raise BridgeIntegrityError("checkpoint audit row is not an object")
            try:
                require_sha256(row.get(field), field=f"checkpoint.{field}")
            except IntegrityError as exc:
                raise BridgeIntegrityError(str(exc)) from exc
    for field, claim in (
        ("file_sha256", "all_file_hashes_distinct"),
        ("tensor_sha256", "all_tensor_hashes_distinct"),
        ("initial_tensor_sha256", "all_initial_tensor_hashes_distinct"),
        ("config_sha256", "all_config_hashes_distinct"),
    ):
        if audit.get(claim) is not True or len({row[field] for row in rows}) != 5:
            raise BridgeIntegrityError(f"checkpoint audit does not establish {claim}")
    for field, claim in (
        ("config_recipe_sha256", "shared_config_recipe_sha256"),
        ("imagenet_backbone_tensor_sha256", "shared_imagenet_backbone_tensor_sha256"),
        ("data_sha256", "shared_data_sha256"),
        ("code_sha256", "shared_code_sha256"),
    ):
        values = {row[field] for row in rows}
        if len(values) != 1 or audit.get(claim) != next(iter(values)):
            raise BridgeIntegrityError(f"checkpoint audit {claim} does not reconcile")


def _validate_execution_seal_document(
    seal: Mapping[str, Any],
    seal_path: Path,
    *,
    source_snapshot: Path = DEFAULT_SOURCE_SNAPSHOT,
    verify_checkpoints: bool = False,
) -> dict[str, Any]:
    if seal.get("schema") != "kbound_cct20_execution_seal_v1":
        raise BridgeIntegrityError("execution seal schema mismatch")
    if seal.get("status") != "SEALED_BEFORE_TARGET_INFERENCE":
        raise BridgeIntegrityError("execution seal is not pre-inference")
    if seal.get("protocol_id") != PROTOCOL_ID:
        raise BridgeIntegrityError("execution seal protocol mismatch")
    unsigned = dict(seal)
    claimed = unsigned.pop("seal_payload_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise BridgeIntegrityError("execution seal payload hash mismatch")
    expected_locks = {
        "target_selection_sha256": TARGET_SELECTION_SHA256,
        "metadata_disclosure_addendum_sha256": METADATA_ADDENDUM_SHA256,
        "label_contract_addendum_sha256": LABEL_CONTRACT_ADDENDUM_SHA256,
    }
    if seal.get("immutable_parent_locks") != expected_locks:
        raise BridgeIntegrityError("execution seal parent-lock identities drift")
    for name, expected in (
        ("target_selection", TARGET_SELECTION_SHA256),
        ("metadata_disclosure_addendum", METADATA_ADDENDUM_SHA256),
        ("label_contract_addendum", LABEL_CONTRACT_ADDENDUM_SHA256),
    ):
        local = REPOSITORY_ROOT / {
            "target_selection": "research_lock/KBOUND_CCT20_TARGET_SELECTION_v1.yaml",
            "metadata_disclosure_addendum": "research_lock/KBOUND_CCT20_TARGET_SELECTION_v1_ADDENDUM.yaml",
            "label_contract_addendum": "research_lock/KBOUND_CCT20_TARGET_SELECTION_v1_LABEL_CONTRACT_ADDENDUM.yaml",
        }[name]
        _regular_file(local, f"current {name} lock")
        if file_sha256(local) != expected:
            raise BridgeIntegrityError(f"current {name} lock hash mismatch")
        rows = seal.get("authoritative_lock_files")
        if not isinstance(rows, list) or not any(row.get("name") == name and row.get("sha256") == expected for row in rows):
            raise BridgeIntegrityError(f"execution seal lacks {name} lock identity")
    protocol_path = REPOSITORY_ROOT / "experiments/kbound/cct20/prospective_protocol_v1.yaml"
    _regular_file(protocol_path, "current prospective protocol")
    if file_sha256(protocol_path) != AUTHORITATIVE_PROTOCOL_FILE_SHA256:
        raise BridgeIntegrityError("current prospective protocol byte hash mismatch")
    protocol_identity = seal.get("authoritative_protocol_file")
    if not isinstance(protocol_identity, Mapping) or protocol_identity.get("sha256") != AUTHORITATIVE_PROTOCOL_FILE_SHA256:
        raise BridgeIntegrityError("execution seal lacks the authoritative protocol identity")
    protocol = seal.get("protocol_config")
    if not isinstance(protocol, Mapping):
        raise BridgeIntegrityError("execution seal lacks protocol configuration")
    try:
        _validate_protocol_config(protocol, target_locations=list(EXPECTED_TARGET_LOCATIONS))
    except IntegrityError as exc:
        raise BridgeIntegrityError(str(exc)) from exc
    if seal.get("protocol_config_sha256") != stable_sha256(dict(protocol)):
        raise BridgeIntegrityError("execution seal protocol-config hash mismatch")
    if seal.get("firewall") != EXPECTED_FIREWALL:
        raise BridgeIntegrityError("execution seal firewall is not closed")
    population = seal.get("population")
    if not isinstance(population, Mapping) or (
        population.get("target_split") != "trans_test"
        or population.get("expected_images") != EXPECTED_TARGET_IMAGES
        or population.get("n_classes") != EXPECTED_CLASS_COUNT
        or population.get("target_location_count") != EXPECTED_LOCATION_COUNT
        or tuple(str(value) for value in population.get("target_location_ids", ()))
        != EXPECTED_TARGET_LOCATIONS
    ):
        raise BridgeIntegrityError("execution seal target population contract drift")
    _assert_sha(population.get("target_manifest_sha256"), population["target_manifest_sha256"], "target_manifest_sha256")

    audit = seal.get("checkpoint_audit")
    if not isinstance(audit, Mapping):
        raise BridgeIntegrityError("execution seal lacks checkpoint audit")
    _validate_checkpoint_audit_metadata(audit)
    if seal.get("checkpoint_audit_sha256") != stable_sha256(dict(audit)):
        raise BridgeIntegrityError("execution seal checkpoint-audit hash mismatch")
    if seal.get("checkpoints") != audit.get("checkpoints"):
        raise BridgeIntegrityError("execution seal checkpoints differ from its audit")
    rows = copy.deepcopy(dict(audit))
    for row in rows["checkpoints"]:
        row["path"] = str(_mapped_sealed_path(row["path"], source_snapshot))
    if verify_checkpoints:
        # Import the model runtime only for the optional full checkpoint replay;
        # provenance and score checks remain runnable in a lightweight environment.
        from experiments.kbound.cct20.runner_runtime import (  # noqa: PLC0415
            verify_checkpoint_audit_document,
        )

        try:
            verify_checkpoint_audit_document(rows)
        except IntegrityError as exc:
            raise BridgeIntegrityError(f"checkpoint files do not verify: {exc}") from exc

    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": seal["protocol_config_sha256"],
        "checkpoint_count": len(EXPECTED_MODEL_SEEDS),
        "checkpoint_seeds": list(EXPECTED_MODEL_SEEDS),
        "target_location_count": EXPECTED_LOCATION_COUNT,
        "target_location_ids": list(EXPECTED_TARGET_LOCATIONS),
        "target_images": EXPECTED_TARGET_IMAGES,
        "firewall": dict(seal["firewall"]),
        "checkpoint_audit_sha256": seal["checkpoint_audit_sha256"],
        "seal_sha256": file_sha256(seal_path),
    }


def _validate_score_and_inference(
    score: Mapping[str, Any],
    inference: Mapping[str, Any],
    *,
    seal_sha256: str,
    target_manifest_sha256: str,
) -> dict[str, Any]:
    if score.get("schema") != "kbound_cct20_set_valued_score_v1" or score.get("status") != "ALL_LOCKED_CELLS_SCORED":
        raise BridgeIntegrityError("CCT-20 score is not the complete locked score")
    if score.get("cell_count") != 45 or score.get("checkpoint_count") != 5 or score.get("location_count") != 9:
        raise BridgeIntegrityError("CCT-20 score cell/checkpoint/location counts drift")
    if score.get("target_image_count") != EXPECTED_TARGET_IMAGES or score.get("target_manifest_sha256") != target_manifest_sha256:
        raise BridgeIntegrityError("CCT-20 score target identity drift")
    cells = score.get("cells")
    if not isinstance(cells, list) or len(cells) != 45:
        raise BridgeIntegrityError("CCT-20 score does not contain 45 cells")
    identities = {(int(row.get("checkpoint_seed")), str(row.get("location_id"))) for row in cells}
    expected = {(seed, location) for seed in EXPECTED_MODEL_SEEDS for location in EXPECTED_TARGET_LOCATIONS}
    if identities != expected:
        raise BridgeIntegrityError("CCT-20 score does not cover every checkpoint-by-location cell")
    if score.get("execution_seal_artifact_sha256") != seal_sha256:
        raise BridgeIntegrityError("CCT-20 score is bound to a different execution seal")
    score_unsigned = dict(score)
    score_hash = score_unsigned.pop("score_sha256", None)
    if score_hash != stable_sha256(score_unsigned):
        raise BridgeIntegrityError("CCT-20 score payload hash mismatch")
    if inference.get("schema") != "kbound_cct20_two_way_inference_v1" or inference.get("status") != "COMPLETE_REPORT_REGARDLESS_OF_RESULT":
        raise BridgeIntegrityError("CCT-20 inference is not complete")
    if inference.get("score_sha256") != score.get("score_sha256") or inference.get("execution_seal_artifact_sha256") != seal_sha256:
        raise BridgeIntegrityError("CCT-20 inference identity drift")
    try:
        replayed = analyze_score_document(score)
    except IntegrityError as exc:
        raise BridgeIntegrityError(f"CCT-20 inference replay failed: {exc}") from exc
    if dict(inference) != replayed:
        raise BridgeIntegrityError("CCT-20 inference differs from deterministic score replay")
    exposure = inference.get("action_exposure_at_checkpoint_location_unit", {})
    counts = exposure.get("counts", {}) if isinstance(exposure, Mapping) else {}
    if counts != {"ABSTAIN": 1, "ADAPT": 0, "FREEZE": 44}:
        raise BridgeIntegrityError(f"unexpected CCT-20 action counts: {counts}")
    return {
        "score_sha256": score["score_sha256"],
        "inference_sha256": file_sha256(
            Path(str(inference.get("_artifact_path", "")))
        ) if inference.get("_artifact_path") else None,
        "cell_count": 45,
        "target_image_count": int(score["target_image_count"]),
        "probe_image_count": int(score["probe_image_count"]),
        "evaluation_image_count": int(score["evaluation_image_count"]),
        "action_counts": dict(counts),
        "result_status": "SAFE_UTILITY_ONLY",
        "strong_success": False,
    }


def _verify_local_release_manifest(path: Path, *, seal_sha256: str, inference_sha256: str) -> dict[str, Any]:
    """Verify the relocated paper manifest while documenting its old bound path."""

    _regular_file(path, "local CCT-20 release manifest")
    receipt_path = path.with_name(path.name + ".receipt.json")
    _regular_file(receipt_path, "local CCT-20 release manifest receipt")
    document = _json(path)
    receipt = _json(receipt_path)
    if receipt.get("schema") != "kbound_cct20_artifact_receipt_v1":
        raise BridgeIntegrityError("local release manifest receipt schema mismatch")
    if receipt.get("artifact_bytes") != path.stat().st_size or receipt.get("artifact_sha256") != file_sha256(path):
        raise BridgeIntegrityError("local release manifest receipt bytes/hash mismatch")
    if receipt.get("canonical_document_sha256") != stable_sha256(document):
        raise BridgeIntegrityError("local release manifest canonical hash mismatch")
    if document.get("schema") != "kbound_cct20_release_manifest_v1" or document.get("status") != "RELEASE_COMPLETE":
        raise BridgeIntegrityError("local release manifest is not complete")
    if document.get("prospective_disclosure") != RELEASE_DISCLOSURE:
        raise BridgeIntegrityError("local release manifest prospective disclosure drift")
    verdict = document.get("verdict", {})
    if not isinstance(verdict, Mapping) or verdict.get("code") != "SAFE_UTILITY_ONLY" or verdict.get("protocol_strong_success") is not False:
        raise BridgeIntegrityError("local release manifest verdict drift")
    upstream = document.get("upstream_artifacts", {})
    if (upstream.get("execution_seal") or {}).get("sha256") != seal_sha256:
        raise BridgeIntegrityError("local release manifest execution-seal identity drift")
    if (upstream.get("two_way_inference") or {}).get("sha256") != inference_sha256:
        raise BridgeIntegrityError("local release manifest inference identity drift")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "receipt_path": str(receipt_path.resolve()),
        "receipt_artifact_path": receipt.get("artifact_path"),
        "receipt_path_relocated": receipt.get("artifact_path") != str(path.resolve()),
        "verdict": verdict.get("code"),
    }


def verify_cct20_prospective_bundle(
    root: str | Path = DEFAULT_EXTERNAL_ROOT,
    *,
    local_release_manifest: str | Path | None = None,
    source_snapshot: str | Path = DEFAULT_SOURCE_SNAPSHOT,
    verify_large_files: bool = True,
) -> dict[str, Any]:
    """Return a strict bridge summary or raise :class:`BridgeIntegrityError`."""

    root = Path(root).expanduser().resolve()
    source_snapshot = Path(source_snapshot).expanduser().resolve()
    seal_path = root / "cct20_execution_seal_v1.json"
    target_manifest_path = root / "target_manifest_label_free.json"
    marker_path = root / "post_target_v1/cct20_one_shot_scoring_marker_v1.json"
    score_path = root / "post_target_v1/cct20_set_valued_score_v1.json"
    inference_path = root / "post_target_v1/cct20_two_way_inference_v1.json"
    seal, seal_receipt = _verify_receipted(seal_path, label="execution seal")
    seal_summary = _validate_execution_seal_document(
        seal,
        seal_path,
        source_snapshot=source_snapshot,
        verify_checkpoints=verify_large_files,
    )
    target_manifest = _verify_plain_json(target_manifest_path, label="label-free target manifest")
    try:
        validate_locked_target_population(target_manifest)
    except IntegrityError as exc:
        raise BridgeIntegrityError(f"label-free target manifest validation failed: {exc}") from exc
    if target_manifest.get("schema") != "kbound_cct20_label_free_target_manifest_v1" or target_manifest.get("status") != "LABEL_FREE_POPULATION_VERIFIED":
        raise BridgeIntegrityError("label-free target manifest status/schema drift")
    target_manifest_hash = target_manifest.get("manifest_sha256")
    _assert_sha(target_manifest_hash, seal["population"]["target_manifest_sha256"], "target manifest canonical SHA-256")
    if target_manifest.get("target_role") != "trans_test" or target_manifest.get("target_annotation_envelope_basename") != "trans_test_annotations.json":
        raise BridgeIntegrityError("label-free target manifest role drift")
    if verify_large_files:
        for dependency in seal.get("dataset_dependencies", []):
            path = _mapped_sealed_path(dependency.get("path", ""), source_snapshot)
            _regular_file(path, f"dataset dependency {dependency.get('name', '')}")
            if path.stat().st_size != dependency.get("bytes") or file_sha256(path) != dependency.get("sha256"):
                raise BridgeIntegrityError(f"dataset dependency hash mismatch: {path}")
    marker = _verify_plain_json(marker_path, label="one-shot scoring marker")
    if marker.get("schema") != "kbound_cct20_one_shot_score_marker_v1" or marker.get("status") != "SPENT_BEFORE_GROUND_TRUTH_LOAD":
        raise BridgeIntegrityError("one-shot marker does not establish pre-scoring spend")
    marker_request = marker.get("request", {})
    if marker_request.get("execution_seal_artifact_sha256") != seal_receipt.get("artifact_sha256") or marker_request.get("expected_target_images") != EXPECTED_TARGET_IMAGES:
        raise BridgeIntegrityError("one-shot marker is not bound to the execution seal/population")
    if marker.get("request_sha256") != stable_sha256(marker_request):
        raise BridgeIntegrityError("one-shot marker request hash mismatch")
    score, score_receipt = _verify_receipted(score_path, label="CCT-20 score")
    inference, inference_receipt = _verify_receipted(inference_path, label="CCT-20 inference")
    inference_summary = _validate_score_and_inference(
        score,
        inference,
        seal_sha256=seal_receipt["artifact_sha256"],
        target_manifest_sha256=target_manifest_hash,
    )
    release_summary = None
    if local_release_manifest is not None:
        release_summary = _verify_local_release_manifest(
            Path(local_release_manifest).expanduser().resolve(),
            seal_sha256=seal_receipt["artifact_sha256"],
            inference_sha256=inference_receipt["artifact_sha256"],
        )
    return {
        "schema": "kbound_cct20_prospective_evidence_bridge_v1",
        "verification_outcome": "PASS",
        "protocol_id": PROTOCOL_ID,
        "external_root": str(root),
        "source_snapshot": str(source_snapshot),
        "artifact_identities": {
            "execution_seal": {
                "path": str(seal_path),
                "sha256": seal_receipt["artifact_sha256"],
                "receipt_sha256": file_sha256(seal_path.with_name(seal_path.name + ".receipt.json")),
            },
            "target_manifest": {
                "path": str(target_manifest_path),
                "sha256": file_sha256(target_manifest_path),
                "canonical_document_sha256": target_manifest_hash,
            },
            "one_shot_marker": {"path": str(marker_path), "sha256": file_sha256(marker_path)},
            "score": {
                "path": str(score_path),
                "sha256": score_receipt["artifact_sha256"],
                "canonical_document_sha256": score_receipt["canonical_document_sha256"],
            },
            "inference": {
                "path": str(inference_path),
                "sha256": inference_receipt["artifact_sha256"],
                "canonical_document_sha256": inference_receipt["canonical_document_sha256"],
            },
        },
        "population_unit": "camera_location",
        "source_location_count": 10,
        "trans_validation_location_count": 1,
        "target_split": "trans_test",
        "target_location_count": seal_summary["target_location_count"],
        "target_location_ids": seal_summary["target_location_ids"],
        "target_images": seal_summary["target_images"],
        "checkpoint_count": seal_summary["checkpoint_count"],
        "all_target_locations_complete": True,
        "all_checkpoint_location_cells_complete": inference_summary["cell_count"] == 45,
        "target_outcomes_unopened_before_execution": True,
        "literal_label_unopened": False,
        "metadata_disclosure": RELEASE_DISCLOSURE,
        "target_decisions_sealed_before_scoring": True,
        "target_runner_imports_scorer": False,
        "target_label_fields_in_predictions": False,
        "exchangeability_status": "ASSUMED_AT_PROTOCOL_SCOPE_NOT_EMPIRICALLY_PROVEN",
        "exchangeability_assumption": (
            "The locked protocol treats camera-location clusters as the population unit and uses "
            "disjoint source, gate, and trans-test roles; this design assumption is not an empirical "
            "proof of exchangeability for future deployments."
        ),
        "result_status": inference_summary["result_status"],
        "strong_success": inference_summary["strong_success"],
        "action_counts": inference_summary["action_counts"],
        "seal_summary": seal_summary,
        "release_manifest": release_summary,
        "verification_scope": {
            "mode": "full_byte_and_model_hashes" if verify_large_files else "metadata_and_receipts_only",
            "source_checkpoints": "hash-verified" if verify_large_files else "sealed hashes recorded but bytes not replayed",
            "official_archives": "hash-verified" if verify_large_files else "sealed hashes recorded but bytes not replayed",
            "target_labels": "not read by this verifier; post-target score is checked against the sealed marker",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument(
        "--source-snapshot", type=Path, default=DEFAULT_SOURCE_SNAPSHOT
    )
    parser.add_argument("--local-release-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the bundle without emitting a new bridge receipt",
    )
    parser.add_argument(
        "--skip-large-files",
        action="store_true",
        help="skip hashing checkpoint and archive bytes (not suitable for final evidence verification)",
    )
    args = parser.parse_args()
    summary = verify_cct20_prospective_bundle(
        args.root,
        local_release_manifest=args.local_release_manifest,
        source_snapshot=args.source_snapshot,
        verify_large_files=not args.skip_large_files,
    )
    if args.output and not args.check:
        receipt = write_immutable_json_with_receipt(args.output, summary)
        print(f"wrote {args.output.resolve()} (sha256={receipt['artifact_sha256']})")
    print(
        f"CCT-20 prospective evidence: {summary['verification_outcome']} "
        f"locations={summary['target_location_count']} cells={summary['seal_summary']['checkpoint_count'] * summary['target_location_count']} "
        f"exchangeability={summary['exchangeability_status']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
