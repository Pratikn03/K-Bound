"""Validation for the create-only So2Sat target-boundary protocol amendment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .integrity import (
    IntegrityError,
    file_sha256,
    stable_sha256,
    strict_json_load,
    verify_artifact_receipt,
)
from .protocol import PROTOCOL_ID

AMENDMENT_BASENAME = "target_boundary_amendment_v1_1.json"
AMENDMENT_SCHEMA = "kbound_so2sat_target_boundary_amendment_v1_1"
AMENDMENT_ID = "KBOUND_SO2SAT_TARGET_BOUNDARY_AMENDMENT_v1.1"
BASE_PROTOCOL_BASENAME = "prospective_protocol_v1.json"


def _expected_boundary_contract() -> dict[str, Any]:
    return {
        "live_runner_imports_offline_scorer": False,
        "live_runner_target_outcome_interface": False,
        "offline_scorer_imports_live_inference_backend": False,
        "offline_scorer_imports_live_runner": False,
        "offline_scorer_process_starts_only_after_complete_target_bundle_receipt": True,
        "final_target_seal_extends_prior_precalibration_seal": True,
        "precalibration_execution_seal_required_before_gate_calibration": True,
        "precalibration_seal_binds_gate_algorithm_before_calibration_outcomes": True,
        "shared_receipt_and_artifact_validation_contracts_allowed": True,
        "testing_label_dataset_maximum_reveals_per_execution_seal": 1,
        "validation_probe_labels_opened": False,
        "validation_probe_labels_scored": False,
    }


def validate_target_boundary_amendment(document: Mapping[str, Any]) -> None:
    """Validate the one-field amendment and its immutable base binding."""

    if not isinstance(document, Mapping) or set(document) != {
        "schema",
        "status",
        "amendment_id",
        "base_protocol_binding",
        "supersession",
        "boundary_contract",
        "target_access_at_amendment_creation",
    }:
        raise IntegrityError("target-boundary amendment has unknown or missing fields")
    if (
        document.get("schema") != AMENDMENT_SCHEMA
        or document.get("amendment_id") != AMENDMENT_ID
        or document.get("status")
        != "SEALED_TARGET_BOUNDARY_AMENDMENT_BEFORE_ANY_TARGET_CONTAINER_DESERIALIZATION"
    ):
        raise IntegrityError("unknown or unsealed target-boundary amendment")
    directory = Path(__file__).resolve().parent
    base_path = directory / BASE_PROTOCOL_BASENAME
    base_receipt_path = base_path.with_name(base_path.name + ".receipt.json")
    base_document = strict_json_load(base_path)
    base_receipt = strict_json_load(base_receipt_path)
    expected_base = {
        "artifact_basename": BASE_PROTOCOL_BASENAME,
        "artifact_sha256": file_sha256(base_path),
        "canonical_document_sha256": stable_sha256(base_document),
        "protocol_id": PROTOCOL_ID,
    }
    if document.get("base_protocol_binding") != expected_base:
        raise IntegrityError("target-boundary amendment base protocol binding mismatch")
    if (
        not isinstance(base_receipt, Mapping)
        or base_receipt.get("artifact_sha256") != expected_base["artifact_sha256"]
        or base_receipt.get("canonical_document_sha256")
        != expected_base["canonical_document_sha256"]
    ):
        raise IntegrityError("base protocol receipt differs from the amendment binding")
    supersession = document.get("supersession")
    if not isinstance(supersession, Mapping) or supersession != {
        "all_other_base_protocol_fields_preserved": True,
        "new_value": True,
        "old_value": False,
        "reason": (
            "The target runner and offline scorer are now implemented as distinct sealed "
            "entry points. The live entry point remains structurally unable to request an "
            "outcome dataset; the offline entry point is authorized only after the complete "
            "prediction/action bundle and all receipts exist."
        ),
        "superseded_json_pointer": (
            "/target_label_firewall/scoring_implementation_in_this_package"
        ),
    }:
        raise IntegrityError("target-boundary amendment changes more than its declared field")
    if document.get("boundary_contract") != _expected_boundary_contract():
        raise IntegrityError("target-boundary amendment process contract drift")
    if document.get("target_access_at_amendment_creation") != {
        "hdf5_datasets_deserialized": 0,
        "target_container_paths_accepted_by_creation_step": False,
        "target_labels_opened": 0,
        "target_pixels_opened": 0,
    }:
        raise IntegrityError("target-boundary amendment does not record zero target access")


def load_target_boundary_amendment(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the exact versioned amendment and its portable byte receipt."""

    amendment_path = Path(path).expanduser().resolve()
    if amendment_path.name != AMENDMENT_BASENAME:
        raise IntegrityError(f"target-boundary amendment must be named {AMENDMENT_BASENAME}")
    receipt = verify_artifact_receipt(amendment_path)
    document = strict_json_load(amendment_path)
    if not isinstance(document, dict):
        raise IntegrityError("target-boundary amendment must be a JSON mapping")
    validate_target_boundary_amendment(document)
    return document, receipt
