"""Neutral contracts shared by So2Sat live inference and offline scoring.

This module contains only immutable protocol constants plus receipt/document
validation and replay helpers.  It imports neither the live target runner nor
the offline scorer, so both sides can verify the same evidence without either
process depending on the other.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import CANDIDATE_IDS, candidate_spec
from .development import (
    load_verified_checkpoints,
    validate_gate_authorization,
    validate_selection,
)
from .features import N_CLASSES, extract_label_free_features, validate_feature_document
from .gate import (
    CHECKPOINT_IDS,
    validate_action_document,
    validate_gate_document,
    validate_study_binding,
)
from .integrity import (
    IntegrityError,
    canonical_json_bytes,
    file_sha256,
    require_sha256,
    stable_sha256,
    strict_json_load,
    verify_artifact_receipt,
)
from .protocol import PROTOCOL_ID
from .source_acceptance import (
    load_verified_source_postrun_acceptance,
    source_postrun_acceptance_binding,
)
from .target_amendment import validate_target_boundary_amendment

EXECUTION_SEAL_SCHEMA = "kbound_so2sat_execution_seal_v1"
TARGET_CELL_SCHEMA = "kbound_so2sat_label_blind_target_cell_v1"
TARGET_BUNDLE_SCHEMA = "kbound_so2sat_complete_label_blind_target_bundle_v1"
LOGIT_ARCHIVE_MANIFEST_SCHEMA = "kbound_so2sat_replayable_logit_archive_manifest_v1"

TARGET_SPLITS = ("validation", "testing")
TARGET_CELL_COUNT = 50
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_026_082_801
INFERENCE_ALPHA = 0.05
PRODUCTION_MODE = "PRODUCTION"
TEST_ONLY_MODE = "TEST_ONLY"


def load_source_postrun_acceptance_pair(
    path: str | Path,
    *,
    strict_document: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Load the strict portable source acceptance pair and canonical binding."""

    if strict_document:
        document, receipt = load_verified_source_postrun_acceptance(path)
    else:
        document, receipt = load_receipted_document(path)
    return document, receipt, source_postrun_acceptance_binding(document, receipt)


def target_scorer_code_identity() -> dict[str, Any]:
    """Hash offline scoring code without importing the scorer process module."""

    directory = Path(__file__).resolve().parent
    names = (
        "integrity.py",
        "protocol.py",
        "metadata_manifest.py",
        "model.py",
        "source_data.py",
        "train_source.py",
        "source_acceptance.py",
        "source_preflight.py",
        "adapters.py",
        "features.py",
        "gate.py",
        "development.py",
        "target_amendment.py",
        "precalibration_seal.py",
        "target_contract.py",
        "target_scorer.py",
        "target_boundary_amendment_v1_1.json",
        "target_boundary_amendment_v1_1.json.receipt.json",
        "prospective_protocol_v1.json",
        "prospective_protocol_v1.json.receipt.json",
    )
    files = {name: file_sha256(directory / name) for name in names}
    return {"files_sha256": files, "code_identity_sha256": stable_sha256(files)}


def target_scorer_environment_identity() -> dict[str, Any]:
    """Return the software/runtime identity frozen for offline scoring."""

    versions: dict[str, str] = {}
    for package in ("h5py", "torch", "torchvision"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    document = {
        "schema": "kbound_so2sat_offline_scorer_environment_v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "numpy_version": str(np.__version__),
        "h5py_version": versions["h5py"],
        "torch_version": versions["torch"],
        "torchvision_version": versions["torchvision"],
    }
    document["environment_identity_sha256"] = stable_sha256(document)
    return document


def load_receipted_document(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a JSON mapping only after its immutable byte receipt verifies."""

    receipt = verify_artifact_receipt(path)
    document = strict_json_load(path)
    if not isinstance(document, dict):
        raise IntegrityError(f"receipt-verified artifact must be a JSON mapping: {path}")
    return document, receipt


def artifact_binding(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact byte and canonical-document hashes from a receipt."""

    return {
        "artifact_sha256": require_sha256(
            receipt.get("artifact_sha256"), field="receipt.artifact_sha256"
        ),
        "canonical_document_sha256": require_sha256(
            receipt.get("canonical_document_sha256"),
            field="receipt.canonical_document_sha256",
        ),
    }


def validate_self_hash(document: Mapping[str, Any], *, field: str) -> None:
    """Require ``field`` to hash the canonical document excluding itself."""

    claimed = require_sha256(document.get(field), field=field)
    unsigned = dict(document)
    unsigned.pop(field, None)
    if stable_sha256(unsigned) != claimed:
        raise IntegrityError(f"{field} does not match the document")


def normalize_target_data_identities(value: Any) -> dict[str, dict[str, Any]]:
    """Validate and normalize the opaque validation/testing container identities."""

    if not isinstance(value, Mapping) or set(value) != set(TARGET_SPLITS):
        raise IntegrityError("target data identities must contain validation and testing")
    output: dict[str, dict[str, Any]] = {}
    for split in TARGET_SPLITS:
        row = value[split]
        if not isinstance(row, Mapping) or set(row) != {"basename", "bytes", "sha256"}:
            raise IntegrityError(f"{split} target data identity schema drift")
        expected_basename = f"{split}.h5"
        count = row.get("bytes")
        if (
            row.get("basename") != expected_basename
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise IntegrityError(f"invalid {split} target data identity")
        output[split] = {
            "basename": expected_basename,
            "bytes": count,
            "sha256": require_sha256(row.get("sha256"), field=f"{split}.sha256"),
        }
    return output


def opaque_target_identities_from_paths(
    target_data_paths: Mapping[str, str | Path],
) -> dict[str, dict[str, Any]]:
    """Hash the two target containers as opaque bytes without opening HDF5."""

    if set(target_data_paths) != set(TARGET_SPLITS):
        raise IntegrityError("target identities require validation and testing paths")
    identities: dict[str, dict[str, Any]] = {}
    for split in TARGET_SPLITS:
        path = Path(target_data_paths[split]).expanduser().resolve()
        if not path.is_file() or path.name != f"{split}.h5":
            raise IntegrityError(f"target identity requires exact {split}.h5 basename")
        size = path.stat().st_size
        if size < 1:
            raise IntegrityError(f"target identity refuses an empty {split} container")
        identities[split] = {
            "basename": path.name,
            "bytes": size,
            "sha256": file_sha256(path),
        }
    return normalize_target_data_identities(identities)


def validate_selected_candidate(
    document: Mapping[str, Any], *, study_binding: Mapping[str, Any]
) -> None:
    """Validate the canonical development-only selection and require a winner."""

    validate_selection(document, study_binding=study_binding)
    candidate_id = document.get("selected_candidate_id")
    if candidate_id not in CANDIDATE_IDS:
        raise IntegrityError("no feasible adapter was selected; target execution is forbidden")
    expected_spec = candidate_spec(str(candidate_id))
    summaries = document.get("candidate_summaries")
    if (
        not isinstance(summaries, Mapping)
        or summaries[candidate_id]["candidate_config_sha256"]
        != expected_spec["candidate_config_sha256"]
    ):
        raise IntegrityError("selected adapter configuration differs from the frozen specification")


def selected_candidate_view(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the small canonical selection view embedded in target artifacts."""

    candidate_id = str(document["selected_candidate_id"])
    return {
        "selected_candidate_sha256": require_sha256(
            document.get("selection_sha256"), field="selection_sha256"
        ),
        "candidate_id": candidate_id,
        "candidate_spec": candidate_spec(candidate_id),
        "selected_gate_fit_bundle_sha256": require_sha256(
            document.get("selected_bundle_sha256"), field="selected_bundle_sha256"
        ),
    }


def validate_checkpoint_collection(
    document: Mapping[str, Any],
    *,
    collection_receipt: Mapping[str, Any],
    collection_path: str | Path,
    checkpoint_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Verify the canonical checkpoint collection, receipts, bytes, and tensors."""

    root = Path(checkpoint_dir).expanduser().resolve()
    expected_collection = root / "so2sat_source_checkpoint_collection.json"
    if Path(collection_path).expanduser().resolve() != expected_collection:
        raise IntegrityError("checkpoint collection must be the canonical file in checkpoint_dir")
    if (
        Path(collection_receipt.get("artifact_name", expected_collection.name)).name
        != expected_collection.name
    ):
        raise IntegrityError("checkpoint collection receipt names another artifact")
    loaded, verified = load_verified_checkpoints(root)
    if loaded != dict(document):
        raise IntegrityError(
            "provided checkpoint collection differs from the canonical verified collection"
        )
    artifact_binding(collection_receipt)
    return {
        checkpoint.checkpoint_id: {
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_path": str(checkpoint.checkpoint_path),
            "checkpoint_basename": checkpoint.checkpoint_path.name,
            "checkpoint_file_sha256": checkpoint.checkpoint_file_sha256,
            "checkpoint_tensor_sha256": checkpoint.checkpoint_tensor_sha256,
            "training_receipt_file_sha256": checkpoint.training_receipt_sha256,
        }
        for checkpoint in verified
    }


def validate_execution_seal(
    document: Mapping[str, Any],
    *,
    study_binding: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
    gate: Mapping[str, Any],
    gate_authorization: Mapping[str, Any],
    target_boundary_amendment: Mapping[str, Any],
    checkpoint_collection: Mapping[str, Any],
    precalibration_seal: Mapping[str, Any],
) -> None:
    """Validate the complete pre-target execution seal and its fixed contract."""

    validate_study_binding(study_binding)
    validate_selected_candidate(selected_candidate, study_binding=study_binding)
    selected = selected_candidate_view(selected_candidate)
    expected_keys = {
        "schema",
        "status",
        "execution_mode",
        "protocol_id",
        "study_binding_sha256",
        "manifest_sha256",
        "population_identity_sha256",
        "protocol_file_sha256",
        "protocol_document_sha256",
        "selected_candidate_sha256",
        "selected_candidate_artifact",
        "candidate_id",
        "candidate_config_sha256",
        "gate_artifact",
        "gate_sha256",
        "gate_authorization_artifact",
        "gate_authorization_sha256",
        "target_boundary_amendment_artifact",
        "target_boundary_amendment_sha256",
        "precalibration_seal_artifact",
        "precalibration_seal_sha256",
        "source_postrun_acceptance",
        "source_postrun_acceptance_artifact_sha256",
        "source_postrun_training_container",
        "source_hdf5_runtime_disclosure",
        "source_checkpoint_selection_disclosure",
        "source_initialization_clarification",
        "checkpoint_collection_artifact",
        "checkpoint_collection_document_sha256",
        "checkpoint_identities",
        "target_data_identities",
        "outcome_reveal_registry",
        "code_identity_sha256",
        "environment_identity_sha256",
        "scorer_code_identity_sha256",
        "scorer_environment_identity_sha256",
        "live_contract",
        "seal_creation_audit",
        "inference_contract",
        "execution_seal_sha256",
    }
    if not isinstance(document, Mapping) or set(document) != expected_keys:
        raise IntegrityError("execution seal has unknown or missing fields")
    mode = document.get("execution_mode")
    expected_status = (
        "SEALED_BEFORE_ANY_TARGET_PIXEL_ACCESS"
        if mode == PRODUCTION_MODE
        else "TEST_ONLY_SEALED_WITH_SYNTHETIC_OR_INJECTED_DEPENDENCIES"
        if mode == TEST_ONLY_MODE
        else None
    )
    if (
        document.get("schema") != EXECUTION_SEAL_SCHEMA
        or document.get("status") != expected_status
        or document.get("protocol_id") != PROTOCOL_ID
    ):
        raise IntegrityError("unknown or unsealed So2Sat execution seal")
    binding_map = {
        "study_binding_sha256": "binding_sha256",
        "manifest_sha256": "manifest_sha256",
        "population_identity_sha256": "population_identity_sha256",
        "protocol_file_sha256": "protocol_file_sha256",
        "protocol_document_sha256": "protocol_document_sha256",
    }
    for seal_field, binding_field in binding_map.items():
        if document.get(seal_field) != study_binding[binding_field]:
            raise IntegrityError(f"execution seal {seal_field} mismatch")
    if (
        document.get("selected_candidate_sha256")
        != selected["selected_candidate_sha256"]
        or document.get("candidate_id") != selected["candidate_id"]
        or document.get("candidate_config_sha256")
        != selected["candidate_spec"]["candidate_config_sha256"]
    ):
        raise IntegrityError("execution seal selected-candidate identity mismatch")
    validate_gate_document(gate)
    validate_gate_authorization(
        gate_authorization,
        selection=selected_candidate,
        gate=gate,
        study_binding=study_binding,
    )
    validate_target_boundary_amendment(target_boundary_amendment)
    if (
        document.get("gate_sha256") != gate["gate_sha256"]
        or document.get("gate_authorization_sha256")
        != gate_authorization["authorization_sha256"]
        or document.get("target_boundary_amendment_sha256")
        != stable_sha256(dict(target_boundary_amendment))
    ):
        raise IntegrityError("execution seal gate/authorization/amendment identity mismatch")
    if (
        document.get("precalibration_seal_sha256")
        != precalibration_seal.get("precalibration_seal_sha256")
        or document.get("execution_mode") != precalibration_seal.get("execution_mode")
        or document.get("outcome_reveal_registry")
        != precalibration_seal.get("outcome_reveal_registry")
        or document.get("target_data_identities")
        != precalibration_seal.get("target_data_identities")
        or document.get("source_postrun_acceptance")
        != precalibration_seal.get("source_postrun_acceptance")
        or document.get("source_postrun_acceptance_artifact_sha256")
        != precalibration_seal.get("source_postrun_acceptance_artifact_sha256")
        or document.get("source_postrun_training_container")
        != precalibration_seal.get("source_postrun_training_container")
        or document.get("source_hdf5_runtime_disclosure")
        != precalibration_seal.get("source_hdf5_runtime_disclosure")
        or document.get("source_checkpoint_selection_disclosure")
        != precalibration_seal.get("source_checkpoint_selection_disclosure")
        or document.get("source_initialization_clarification")
        != precalibration_seal.get("source_initialization_clarification")
    ):
        raise IntegrityError("execution seal does not extend the prior precalibration seal")
    source_acceptance = document.get("source_postrun_acceptance")
    if (
        source_acceptance != selected_candidate.get("source_postrun_acceptance")
        or gate_authorization.get("source_postrun_acceptance_artifact_sha256")
        != source_acceptance.get("source_postrun_acceptance_artifact_sha256")
        or gate_authorization.get(
            "source_postrun_acceptance_canonical_document_sha256"
        )
        != source_acceptance.get(
            "source_postrun_acceptance_canonical_document_sha256"
        )
    ):
        raise IntegrityError("execution seal source acceptance chain mismatch")
    for field in (
        "selected_candidate_artifact",
        "gate_artifact",
        "gate_authorization_artifact",
        "target_boundary_amendment_artifact",
        "precalibration_seal_artifact",
        "checkpoint_collection_artifact",
    ):
        value = document.get(field)
        if not isinstance(value, Mapping) or set(value) != {
            "artifact_sha256",
            "canonical_document_sha256",
        }:
            raise IntegrityError(f"execution seal {field} schema drift")
        artifact_binding(value)
    if document.get("checkpoint_collection_document_sha256") != stable_sha256(
        dict(checkpoint_collection)
    ):
        raise IntegrityError("execution seal checkpoint collection document mismatch")
    expected_checkpoints = {
        str(row["model_seed"]): {
            "checkpoint_file_sha256": row["checkpoint_file_sha256"],
            "checkpoint_tensor_sha256": row["checkpoint_tensor_sha256"],
        }
        for row in checkpoint_collection.get("checkpoints", [])
    }
    if document.get("checkpoint_identities") != dict(sorted(expected_checkpoints.items())):
        raise IntegrityError("execution seal checkpoint identities mismatch")
    normalize_target_data_identities(document.get("target_data_identities"))
    registry = document.get("outcome_reveal_registry")
    if (
        not isinstance(registry, Mapping)
        or set(registry)
        != {
            "registry_id",
            "identity_basename",
            "identity_artifact",
            "ledger_basename_template",
            "policy",
        }
        or not isinstance(registry.get("registry_id"), str)
        or len(registry["registry_id"]) != 64
        or registry.get("identity_basename")
        != "so2sat_outcome_reveal_registry.json"
        or not isinstance(registry.get("identity_artifact"), Mapping)
        or registry.get("ledger_basename_template")
        != "so2sat_target_outcome_reveal_{execution_seal_sha256}.json"
        or registry.get("policy")
        != "ONE_CREATE_ONLY_REVEAL_LEDGER_PER_EXECUTION_SEAL"
    ):
        raise IntegrityError("execution seal outcome reveal registry drift")
    artifact_binding(registry["identity_artifact"])
    require_sha256(document.get("code_identity_sha256"), field="code_identity_sha256")
    require_sha256(
        document.get("environment_identity_sha256"), field="environment_identity_sha256"
    )
    require_sha256(
        document.get("scorer_code_identity_sha256"),
        field="scorer_code_identity_sha256",
    )
    require_sha256(
        document.get("scorer_environment_identity_sha256"),
        field="scorer_environment_identity_sha256",
    )
    if document.get("live_contract") != {
        "target_modality": "sen2_10_band",
        "probe_split": "validation",
        "evaluation_split": "testing",
        "target_city_count": 10,
        "checkpoint_count": 5,
        "cell_count": TARGET_CELL_COUNT,
        "probe_labels_opened": False,
        "probe_labels_scored": False,
        "evaluation_labels_available_to_live_runner": False,
        "abstain_realized_action": "FREEZE",
    }:
        raise IntegrityError("execution seal live contract drift")
    if document.get("seal_creation_audit") != {
        "extends_precalibration_seal": True,
        "gate_calibration_complete": True,
        "target_container_hash_method": "opaque_raw_file_bytes_sha256",
        "target_hdf5_datasets_deserialized": 0,
        "target_pixels_opened": 0,
        "target_labels_opened": 0,
    }:
        raise IntegrityError("execution seal creation audit drift")
    if document.get("inference_contract") != {
        "effect": "fixed_policy_regret_minus_kga_regret_equals_kga_accuracy_minus_fixed_policy_accuracy",
        "positive_favors": "KGA",
        "estimand": "equal_target_city_macro_mean_of_equal_checkpoint_cell_accuracy_differences",
        "within_cell_weighting": "equal_weight_per_testing_sample",
        "target_city_weighting": "equal_weight_one_tenth_per_city",
        "source_checkpoint_weighting": "equal_weight_one_fifth_per_checkpoint",
        "cluster_unit": "target_city",
        "crossed_unit": "source_checkpoint",
        "bootstrap": "paired_two_way_city_by_checkpoint_resampling_with_replacement",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_level": 1.0 - INFERENCE_ALPHA,
        "randomization_test": "exact_two_sided_sign_flip_of_10_city_means_under_joint_sign_symmetry",
        "sign_flip_assumption": "joint_city_effect_sign_symmetry_under_the_null;_the_gate_is_deterministic_not_randomized",
        "multiplicity": "Holm_over_two_fixed_policy_comparisons",
        "minimum_realized_action_cell_fraction": 0.20,
        "minimum_realized_action_city_count": 2,
        "minimum_direct_decision_cell_fraction": 0.20,
        "minimum_direct_decision_city_count": 2,
        "report_all_outcomes_regardless_of_direction": True,
    }:
        raise IntegrityError("execution seal inference contract drift")
    validate_self_hash(document, field="execution_seal_sha256")


def tensor_sha256(array: np.ndarray) -> str:
    """Hash an ndarray with explicit dtype, shape, and C-order framing."""

    contiguous = np.ascontiguousarray(array)
    header = canonical_json_bytes(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape), "order": "C"}
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    raw = contiguous.tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def validate_target_cell(
    document: Mapping[str, Any],
    *,
    seal: Mapping[str, Any],
    gate_authorization: Mapping[str, Any],
    gate: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
) -> None:
    """Validate one immutable checkpoint-by-city target prediction cell."""

    expected_keys = {
        "schema",
        "status",
        "execution_mode",
        "execution_seal_sha256",
        "gate_authorization_sha256",
        "target_boundary_amendment_sha256",
        "precalibration_seal_sha256",
        "source_postrun_acceptance_artifact_sha256",
        "gate_sha256",
        "selected_candidate_sha256",
        "candidate_id",
        "manifest_sha256",
        "population_identity_sha256",
        "checkpoint_id",
        "checkpoint_file_sha256",
        "checkpoint_tensor_sha256",
        "city_id",
        "partition_sha256",
        "probe",
        "action",
        "action_artifact",
        "evaluation",
        "logit_archive",
        "target_data_identities",
        "cell_sha256",
    }
    if not isinstance(document, Mapping) or set(document) != expected_keys:
        raise IntegrityError("target cell has unknown or missing fields")
    expected_cell_status = (
        "SEALED_BEFORE_TARGET_OUTCOME_ACCESS"
        if seal["execution_mode"] == PRODUCTION_MODE
        else "TEST_ONLY_CELL_WITH_SYNTHETIC_OR_INJECTED_DEPENDENCIES"
    )
    if (
        document.get("schema") != TARGET_CELL_SCHEMA
        or document.get("status") != expected_cell_status
        or document.get("execution_mode") != seal["execution_mode"]
    ):
        raise IntegrityError("unknown or unsealed target cell")
    identity_fields = {
        "execution_seal_sha256": seal["execution_seal_sha256"],
        "gate_authorization_sha256": gate_authorization["authorization_sha256"],
        "target_boundary_amendment_sha256": seal[
            "target_boundary_amendment_sha256"
        ],
        "precalibration_seal_sha256": seal["precalibration_seal_sha256"],
        "source_postrun_acceptance_artifact_sha256": seal[
            "source_postrun_acceptance"
        ]["source_postrun_acceptance_artifact_sha256"],
        "gate_sha256": gate["gate_sha256"],
        "selected_candidate_sha256": selected_candidate["selected_candidate_sha256"],
        "candidate_id": selected_candidate["candidate_id"],
        "manifest_sha256": seal["manifest_sha256"],
        "population_identity_sha256": seal["population_identity_sha256"],
    }
    for field, expected in identity_fields.items():
        if document.get(field) != expected:
            raise IntegrityError(f"target cell {field} mismatch")
    checkpoint_id = document.get("checkpoint_id")
    city_id = document.get("city_id")
    if (
        checkpoint_id not in CHECKPOINT_IDS
        or city_id not in gate["study_binding"]["target_cities"]
    ):
        raise IntegrityError("target cell checkpoint/city is outside the sealed design")
    checkpoint = seal["checkpoint_identities"][checkpoint_id]
    if (
        document.get("checkpoint_file_sha256") != checkpoint["checkpoint_file_sha256"]
        or document.get("checkpoint_tensor_sha256")
        != checkpoint["checkpoint_tensor_sha256"]
    ):
        raise IntegrityError("target cell checkpoint identity mismatch")
    require_sha256(document.get("partition_sha256"), field="partition_sha256")
    probe = document.get("probe")
    evaluation = document.get("evaluation")
    if not isinstance(probe, Mapping) or set(probe) != {
        "official_split",
        "row_indices",
        "row_indices_sha256",
        "sample_count",
        "feature_document",
        "target_labels_opened",
        "target_labels_scored",
    }:
        raise IntegrityError("target cell probe schema drift")
    if not isinstance(evaluation, Mapping) or set(evaluation) != {
        "official_split",
        "row_indices",
        "row_indices_sha256",
        "sample_count",
        "frozen_logits_tensor_sha256",
        "adapted_logits_tensor_sha256",
        "frozen_prediction_class_ids",
        "adapted_prediction_class_ids",
        "frozen_predictions_sha256",
        "adapted_predictions_sha256",
        "target_labels_opened",
    }:
        raise IntegrityError("target cell evaluation schema drift")
    if (
        probe.get("official_split") != "validation"
        or probe.get("target_labels_opened") is not False
        or probe.get("target_labels_scored") is not False
        or evaluation.get("official_split") != "testing"
        or evaluation.get("target_labels_opened") is not False
    ):
        raise IntegrityError("target cell crosses the label firewall")
    for partition, name in ((probe, "probe"), (evaluation, "evaluation")):
        rows = partition.get("row_indices")
        if (
            not isinstance(rows, list)
            or not rows
            or any(
                isinstance(row, bool) or not isinstance(row, int) or row < 0
                for row in rows
            )
            or rows != sorted(set(rows))
            or partition.get("sample_count") != len(rows)
            or partition.get("row_indices_sha256") != stable_sha256(rows)
        ):
            raise IntegrityError(f"target cell {name} row partition is invalid")
    expected_partition_sha256 = stable_sha256(
        {
            "schema": "kbound_so2sat_target_city_partition_v1",
            "population_identity_sha256": seal["population_identity_sha256"],
            "city_id": city_id,
            "validation_row_indices": probe["row_indices"],
            "testing_row_indices": evaluation["row_indices"],
        }
    )
    if document.get("partition_sha256") != expected_partition_sha256:
        raise IntegrityError("target cell partition hash does not bind its exact row indices")
    feature = probe.get("feature_document")
    if not isinstance(feature, Mapping):
        raise IntegrityError("target cell lacks probe features")
    validate_feature_document(feature)
    for prefix in ("frozen", "adapted"):
        values = evaluation.get(f"{prefix}_prediction_class_ids")
        if (
            not isinstance(values, list)
            or len(values) != evaluation["sample_count"]
            or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
            or any(not 0 <= value < N_CLASSES for value in values)
        ):
            raise IntegrityError(f"target cell {prefix} predictions are invalid")
        array = np.asarray(values, dtype=np.int64)
        if evaluation.get(f"{prefix}_predictions_sha256") != tensor_sha256(array):
            raise IntegrityError(f"target cell {prefix} prediction hash mismatch")
        require_sha256(
            evaluation.get(f"{prefix}_logits_tensor_sha256"),
            field=f"{prefix}_logits_tensor_sha256",
        )
    action = document.get("action")
    if not isinstance(action, Mapping):
        raise IntegrityError("target cell lacks its sealed gate action")
    validate_action_document(action, gate=gate)
    if (
        action.get("city_id") != city_id
        or action.get("checkpoint_id") != checkpoint_id
        or action.get("partition_sha256") != document.get("partition_sha256")
        or action.get("feature_document") != feature
        or action.get("realized_action") not in {"ADAPT", "FREEZE"}
        or (
            action.get("decision") == "ABSTAIN"
            and action.get("realized_action") != "FREEZE"
        )
    ):
        raise IntegrityError("target cell action/feature/partition binding mismatch")
    action_artifact = document.get("action_artifact")
    if (
        not isinstance(action_artifact, Mapping)
        or set(action_artifact)
        != {
            "action_basename",
            "artifact_sha256",
            "canonical_document_sha256",
            "sealed_before_evaluation_pixel_access",
        }
        or not isinstance(action_artifact.get("action_basename"), str)
        or Path(action_artifact["action_basename"]).name
        != action_artifact["action_basename"]
        or action_artifact.get("sealed_before_evaluation_pixel_access") is not True
    ):
        raise IntegrityError("target cell lacks a pre-evaluation action artifact")
    require_sha256(
        action_artifact.get("artifact_sha256"), field="action_artifact_sha256"
    )
    require_sha256(
        action_artifact.get("canonical_document_sha256"),
        field="action_canonical_document_sha256",
    )
    archive = document.get("logit_archive")
    if not isinstance(archive, Mapping) or set(archive) != {
        "archive_basename",
        "archive_bytes",
        "archive_sha256",
        "manifest_basename",
        "manifest_sha256",
        "manifest_artifact",
    }:
        raise IntegrityError("target cell logit archive binding schema drift")
    for name in ("archive_basename", "manifest_basename"):
        value = archive.get(name)
        if not isinstance(value, str) or Path(value).name != value:
            raise IntegrityError("target cell logit archive uses a nonlocal basename")
    if (
        isinstance(archive.get("archive_bytes"), bool)
        or not isinstance(archive.get("archive_bytes"), int)
        or archive["archive_bytes"] < 1
    ):
        raise IntegrityError("target cell logit archive byte count is invalid")
    for name in ("archive_sha256", "manifest_sha256"):
        require_sha256(archive.get(name), field=f"logit_archive.{name}")
    if not isinstance(archive.get("manifest_artifact"), Mapping):
        raise IntegrityError("target cell logit archive lacks a manifest receipt binding")
    artifact_binding(archive["manifest_artifact"])
    if document.get("target_data_identities") != seal["target_data_identities"]:
        raise IntegrityError("target cell opaque target identities mismatch")
    validate_self_hash(document, field="cell_sha256")


def validate_complete_target_bundle_document(document: Mapping[str, Any]) -> None:
    """Validate the immutable inventory for all 10 x 5 target cells."""

    expected_keys = {
        "schema",
        "status",
        "execution_mode",
        "execution_seal_artifact",
        "execution_seal_sha256",
        "gate_authorization_artifact",
        "gate_authorization_sha256",
        "target_boundary_amendment_artifact",
        "target_boundary_amendment_sha256",
        "precalibration_seal_artifact",
        "precalibration_seal_sha256",
        "source_postrun_acceptance",
        "source_postrun_acceptance_artifact_sha256",
        "source_postrun_training_container",
        "source_hdf5_runtime_disclosure",
        "source_checkpoint_selection_disclosure",
        "source_initialization_clarification",
        "population_manifest_artifact",
        "manifest_sha256",
        "population_identity_sha256",
        "selected_candidate_artifact",
        "selected_candidate_sha256",
        "candidate_id",
        "gate_sha256",
        "checkpoint_collection_artifact",
        "target_data_identities",
        "target_cities",
        "checkpoint_ids",
        "cell_count",
        "cells",
        "access_audit",
        "probe_labels_opened",
        "probe_labels_scored",
        "evaluation_labels_opened",
        "complete_before_scoring",
        "bundle_sha256",
    }
    if not isinstance(document, Mapping) or set(document) != expected_keys:
        raise IntegrityError("target master bundle has unknown or missing fields")
    mode = document.get("execution_mode")
    expected_bundle_status = (
        "COMPLETE_50_CELLS_SEALED_BEFORE_TARGET_OUTCOME_ACCESS"
        if mode == PRODUCTION_MODE
        else "TEST_ONLY_COMPLETE_50_CELLS_WITH_SYNTHETIC_OR_INJECTED_DEPENDENCIES"
        if mode == TEST_ONLY_MODE
        else None
    )
    if (
        document.get("schema") != TARGET_BUNDLE_SCHEMA
        or document.get("status") != expected_bundle_status
        or document.get("checkpoint_ids") != list(CHECKPOINT_IDS)
        or document.get("cell_count") != TARGET_CELL_COUNT
        or document.get("probe_labels_opened") is not False
        or document.get("probe_labels_scored") is not False
        or document.get("evaluation_labels_opened") is not False
        or document.get("complete_before_scoring") is not True
    ):
        raise IntegrityError("target master bundle is incomplete or crossed the outcome boundary")
    cities = document.get("target_cities")
    if (
        not isinstance(cities, list)
        or len(cities) != 10
        or cities != sorted(set(cities))
    ):
        raise IntegrityError("target master bundle must contain ten sorted unique cities")
    for field in (
        "execution_seal_artifact",
        "gate_authorization_artifact",
        "target_boundary_amendment_artifact",
        "precalibration_seal_artifact",
        "population_manifest_artifact",
        "selected_candidate_artifact",
        "checkpoint_collection_artifact",
    ):
        value = document.get(field)
        if not isinstance(value, Mapping):
            raise IntegrityError(f"target master bundle lacks {field}")
        artifact_binding(value)
    for field in (
        "execution_seal_sha256",
        "gate_authorization_sha256",
        "target_boundary_amendment_sha256",
        "precalibration_seal_sha256",
        "manifest_sha256",
        "population_identity_sha256",
        "selected_candidate_sha256",
        "gate_sha256",
        "source_postrun_acceptance_artifact_sha256",
    ):
        require_sha256(document.get(field), field=field)
    if document.get("candidate_id") not in CANDIDATE_IDS:
        raise IntegrityError("target bundle candidate id is unknown")
    source_acceptance = document.get("source_postrun_acceptance")
    if (
        not isinstance(source_acceptance, Mapping)
        or set(source_acceptance)
        != {
            "source_postrun_acceptance_artifact_basename",
            "source_postrun_acceptance_artifact_sha256",
            "source_postrun_acceptance_canonical_document_sha256",
        }
        or source_acceptance.get("source_postrun_acceptance_artifact_basename")
        != "so2sat_source_postrun_acceptance.json"
    ):
        raise IntegrityError("target bundle source acceptance schema drift")
    for field in (
        "source_postrun_acceptance_artifact_sha256",
        "source_postrun_acceptance_canonical_document_sha256",
    ):
        require_sha256(source_acceptance.get(field), field=field)
    for field in (
        "source_postrun_training_container",
        "source_hdf5_runtime_disclosure",
        "source_checkpoint_selection_disclosure",
        "source_initialization_clarification",
    ):
        if not isinstance(document.get(field), Mapping):
            raise IntegrityError(f"target bundle {field} is invalid")
    normalize_target_data_identities(document.get("target_data_identities"))
    rows = document.get("cells")
    expected_cells = {
        (city, checkpoint) for city in cities for checkpoint in CHECKPOINT_IDS
    }
    if not isinstance(rows, list) or len(rows) != TARGET_CELL_COUNT:
        raise IntegrityError("target master bundle cell inventory is incomplete")
    observed: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "city_id",
            "checkpoint_id",
            "cell_basename",
            "cell_sha256",
            "artifact_sha256",
            "canonical_document_sha256",
            "action_sha256",
            "action_basename",
            "action_artifact_sha256",
            "action_canonical_document_sha256",
            "logit_archive_sha256",
            "logit_manifest_sha256",
        }:
            raise IntegrityError("target master bundle cell-row schema drift")
        key = (row.get("city_id"), row.get("checkpoint_id"))
        if key in observed:
            raise IntegrityError("target master bundle contains a duplicate cell")
        observed.add(key)  # type: ignore[arg-type]
        basename = row.get("cell_basename")
        if not isinstance(basename, str) or Path(basename).name != basename:
            raise IntegrityError("target cell inventory must use a local basename")
        action_basename = row.get("action_basename")
        if (
            not isinstance(action_basename, str)
            or Path(action_basename).name != action_basename
        ):
            raise IntegrityError("target action inventory must use a local basename")
        for field in (
            "cell_sha256",
            "artifact_sha256",
            "canonical_document_sha256",
            "action_sha256",
            "action_artifact_sha256",
            "action_canonical_document_sha256",
            "logit_archive_sha256",
            "logit_manifest_sha256",
        ):
            require_sha256(row.get(field), field=f"target_cell.{field}")
    if observed != expected_cells:
        raise IntegrityError("target master bundle does not cover exactly 10 cities x 5 checkpoints")
    audit = document.get("access_audit")
    if (
        not isinstance(audit, Mapping)
        or set(audit)
        != {
            "pixel_rows_read_exactly_once",
            "validation_pixel_rows",
            "testing_pixel_rows",
            "pixel_dataset",
            "target_outcome_dataset_accessed",
            "container_bytes_verified_before_and_after",
        }
        or audit.get("pixel_rows_read_exactly_once") is not True
        or audit.get("container_bytes_verified_before_and_after") is not True
        or audit.get("pixel_dataset") != "sen2"
        or audit.get("target_outcome_dataset_accessed") is not False
        or any(
            isinstance(audit.get(field), bool)
            or not isinstance(audit.get(field), int)
            or audit[field] < 1
            for field in ("validation_pixel_rows", "testing_pixel_rows")
        )
    ):
        raise IntegrityError("target master bundle lacks a clean live access audit")
    validate_self_hash(document, field="bundle_sha256")


def _verify_replayable_logit_archive(root: Path, cell: Mapping[str, Any]) -> None:
    binding = cell["logit_archive"]
    archive_path = root / binding["archive_basename"]
    manifest_path = root / binding["manifest_basename"]
    manifest, receipt = load_receipted_document(manifest_path)
    if (
        artifact_binding(receipt) != binding["manifest_artifact"]
        or manifest.get("logit_archive_manifest_sha256") != binding["manifest_sha256"]
    ):
        raise IntegrityError("logit archive manifest differs from the target cell")
    expected_status = (
        "REPLAYABLE_LOGITS_SEALED_BEFORE_TARGET_OUTCOME_ACCESS"
        if cell["execution_mode"] == PRODUCTION_MODE
        else "TEST_ONLY_REPLAYABLE_SYNTHETIC_LOGITS"
    )
    expected_manifest_keys = {
        "schema",
        "status",
        "execution_mode",
        "archive_basename",
        "archive_bytes",
        "archive_sha256",
        "compression",
        "arrays",
        "target_outcomes_present",
        "logit_archive_manifest_sha256",
    }
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema") != LOGIT_ARCHIVE_MANIFEST_SCHEMA
        or manifest.get("status") != expected_status
        or manifest.get("execution_mode") != cell["execution_mode"]
        or manifest.get("archive_basename") != archive_path.name
        or manifest.get("compression") != "numpy_savez_compressed"
        or manifest.get("target_outcomes_present") is not False
    ):
        raise IntegrityError("logit archive manifest schema/status drift")
    validate_self_hash(manifest, field="logit_archive_manifest_sha256")
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size != manifest.get("archive_bytes")
        or manifest.get("archive_bytes") != binding["archive_bytes"]
        or file_sha256(archive_path) != manifest.get("archive_sha256")
        or manifest.get("archive_sha256") != binding["archive_sha256"]
    ):
        raise IntegrityError("replayable logit archive bytes differ from their seal")
    expected_shapes = {
        "frozen_probe_logits": (cell["probe"]["sample_count"], N_CLASSES),
        "adapted_probe_logits": (cell["probe"]["sample_count"], N_CLASSES),
        "frozen_evaluation_logits": (
            cell["evaluation"]["sample_count"],
            N_CLASSES,
        ),
        "adapted_evaluation_logits": (
            cell["evaluation"]["sample_count"],
            N_CLASSES,
        ),
    }
    metadata = manifest.get("arrays")
    if not isinstance(metadata, Mapping) or set(metadata) != set(expected_shapes):
        raise IntegrityError("logit archive array metadata inventory drift")
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected_shapes):
                raise IntegrityError("logit archive array inventory drift")
            for name, shape in expected_shapes.items():
                array = np.asarray(archive[name])
                expected_metadata = {
                    "dtype": "float64",
                    "shape": list(shape),
                    "tensor_sha256": tensor_sha256(
                        np.ascontiguousarray(array, dtype=np.float64)
                    ),
                }
                if (
                    array.dtype != np.dtype("float64")
                    or array.shape != shape
                    or not np.isfinite(array).all()
                    or metadata[name] != expected_metadata
                ):
                    raise IntegrityError(f"logit archive {name} content drift")
                arrays[name] = np.ascontiguousarray(array, dtype=np.float64)
    except IntegrityError:
        raise
    except Exception as exc:
        raise IntegrityError(f"cannot safely replay sealed logit archive: {exc}") from exc
    feature_values = cell["probe"]["feature_document"]["features"]
    replayed_feature = extract_label_free_features(
        arrays["frozen_probe_logits"],
        arrays["adapted_probe_logits"],
        normalized_adapter_update_norm=feature_values[
            "normalized_adapter_update_norm"
        ],
        batchnorm_source_statistic_divergence=feature_values[
            "batchnorm_source_statistic_divergence"
        ],
    )
    if replayed_feature != cell["probe"]["feature_document"]:
        raise IntegrityError("sealed probe feature does not replay from the logit archive")
    for prefix in ("frozen", "adapted"):
        evaluation_logits = arrays[f"{prefix}_evaluation_logits"]
        if (
            tensor_sha256(evaluation_logits)
            != cell["evaluation"][f"{prefix}_logits_tensor_sha256"]
            or evaluation_logits.argmax(axis=1).astype(np.int64).tolist()
            != cell["evaluation"][f"{prefix}_prediction_class_ids"]
        ):
            raise IntegrityError(
                f"sealed {prefix} predictions do not replay from evaluation logits"
            )


def load_complete_target_bundle(
    path: str | Path,
    *,
    seal: Mapping[str, Any],
    gate_authorization: Mapping[str, Any],
    gate: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify the master, all cells, and replay archives before returning predictions."""

    master, _ = load_receipted_document(path)
    validate_complete_target_bundle_document(master)
    if (
        master["execution_mode"] != seal["execution_mode"]
        or master["execution_seal_sha256"] != seal["execution_seal_sha256"]
        or master["gate_authorization_sha256"]
        != gate_authorization["authorization_sha256"]
        or master["target_boundary_amendment_sha256"]
        != seal["target_boundary_amendment_sha256"]
        or master["precalibration_seal_sha256"]
        != seal["precalibration_seal_sha256"]
        or master["gate_sha256"] != gate["gate_sha256"]
        or master["selected_candidate_sha256"]
        != selected_candidate["selected_candidate_sha256"]
        or master["target_data_identities"] != seal["target_data_identities"]
        or master["source_postrun_acceptance"]
        != seal["source_postrun_acceptance"]
        or master["source_postrun_acceptance_artifact_sha256"]
        != seal["source_postrun_acceptance_artifact_sha256"]
        or master["source_postrun_training_container"]
        != seal["source_postrun_training_container"]
        or master["source_hdf5_runtime_disclosure"]
        != seal["source_hdf5_runtime_disclosure"]
        or master["source_checkpoint_selection_disclosure"]
        != seal["source_checkpoint_selection_disclosure"]
        or master["source_initialization_clarification"]
        != seal["source_initialization_clarification"]
    ):
        raise IntegrityError("target master bundle input identities differ from scoring inputs")
    root = Path(path).expanduser().resolve().parent
    cells: list[dict[str, Any]] = []
    testing_coverage_by_checkpoint: dict[str, list[int]] = {
        checkpoint: [] for checkpoint in CHECKPOINT_IDS
    }
    validation_coverage_by_checkpoint: dict[str, list[int]] = {
        checkpoint: [] for checkpoint in CHECKPOINT_IDS
    }
    validation_partition_by_city: dict[str, list[int]] = {}
    testing_partition_by_city: dict[str, list[int]] = {}
    for inventory in master["cells"]:
        cell_path = root / inventory["cell_basename"]
        cell, receipt = load_receipted_document(cell_path)
        validate_target_cell(
            cell,
            seal=seal,
            gate_authorization=gate_authorization,
            gate=gate,
            selected_candidate=selected_candidate,
        )
        _verify_replayable_logit_archive(root, cell)
        action_path = root / cell["action_artifact"]["action_basename"]
        action_document, action_receipt = load_receipted_document(action_path)
        if (
            action_document != cell["action"]
            or artifact_binding(action_receipt)
            != {
                "artifact_sha256": cell["action_artifact"]["artifact_sha256"],
                "canonical_document_sha256": cell["action_artifact"][
                    "canonical_document_sha256"
                ],
            }
        ):
            raise IntegrityError("pre-evaluation action artifact differs from target cell")
        if (
            cell["cell_sha256"] != inventory["cell_sha256"]
            or receipt["artifact_sha256"] != inventory["artifact_sha256"]
            or receipt["canonical_document_sha256"]
            != inventory["canonical_document_sha256"]
            or cell["action"]["action_sha256"] != inventory["action_sha256"]
            or cell["action_artifact"]["action_basename"]
            != inventory["action_basename"]
            or cell["action_artifact"]["artifact_sha256"]
            != inventory["action_artifact_sha256"]
            or cell["action_artifact"]["canonical_document_sha256"]
            != inventory["action_canonical_document_sha256"]
            or cell["logit_archive"]["archive_sha256"]
            != inventory["logit_archive_sha256"]
            or cell["logit_archive"]["manifest_sha256"]
            != inventory["logit_manifest_sha256"]
            or cell["city_id"] != inventory["city_id"]
            or cell["checkpoint_id"] != inventory["checkpoint_id"]
        ):
            raise IntegrityError("target cell artifact differs from master inventory")
        city = cell["city_id"]
        probe_rows = cell["probe"]["row_indices"]
        testing_rows = cell["evaluation"]["row_indices"]
        if (
            city in validation_partition_by_city
            and validation_partition_by_city[city] != probe_rows
        ):
            raise IntegrityError("target probe partition differs across checkpoints")
        if (
            city in testing_partition_by_city
            and testing_partition_by_city[city] != testing_rows
        ):
            raise IntegrityError("target evaluation partition differs across checkpoints")
        validation_partition_by_city[city] = probe_rows
        testing_partition_by_city[city] = testing_rows
        validation_coverage_by_checkpoint[cell["checkpoint_id"]].extend(probe_rows)
        testing_coverage_by_checkpoint[cell["checkpoint_id"]].extend(testing_rows)
        cells.append(cell)
    for split, coverage in (
        ("validation", validation_coverage_by_checkpoint),
        ("testing", testing_coverage_by_checkpoint),
    ):
        expected_rows = None
        for checkpoint, rows in coverage.items():
            if len(rows) != len(set(rows)):
                raise IntegrityError(f"{split} rows overlap for checkpoint {checkpoint}")
            ordered_rows = sorted(rows)
            if expected_rows is None:
                expected_rows = ordered_rows
            elif ordered_rows != expected_rows:
                raise IntegrityError(f"{split} population differs across checkpoints")
        if not expected_rows or expected_rows != list(range(max(expected_rows) + 1)):
            raise IntegrityError(
                f"target bundle {split} rows are not a complete zero-based population"
            )
    if set(validation_partition_by_city) != set(master["target_cities"]):
        raise IntegrityError("target bundle probe city partition is incomplete")
    cells.sort(key=lambda cell: (cell["city_id"], cell["checkpoint_id"]))
    return master, cells
