"""Create the So2Sat execution/configuration seal before gate calibration.

This create-only step runs immediately after gate-fit candidate selection.  It
binds the selected adapter, exact gate-fit evidence, source checkpoints and
normalizer, the deterministic gate algorithm and already-fit ridge parameters,
package/runtime identities, opaque target-container hashes, and the canonical
one-reveal registry.  It has no gate-calibration outcome interface and hashes
target containers only as raw bytes.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .adapters import candidate_spec
from .development import (
    GATE_FIT_ROLE,
    _runner_code_identity,
    validate_candidate_bundle,
    validate_development_environment_identity,
)
from .features import FEATURE_NAMES, feature_vector
from .gate import (
    CALIBRATION_ALPHA,
    CALIBRATION_CITY_COUNT,
    CHECKPOINT_IDS,
    CONFORMAL_RANK,
    RIDGE_PENALTY,
    load_study_binding,
    validate_study_binding,
)
from .integrity import (
    IntegrityError,
    file_sha256,
    require_sha256,
    stable_sha256,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from .source_acceptance import (
    source_postrun_acceptance_binding,
    verify_source_postrun_acceptance_bindings,
)
from .source_data import load_sealed_band_normalizer
from .target_amendment import (
    load_target_boundary_amendment,
    validate_target_boundary_amendment,
)
from .target_contract import (
    PRODUCTION_MODE,
    TEST_ONLY_MODE,
    artifact_binding,
    load_receipted_document,
    normalize_target_data_identities,
    opaque_target_identities_from_paths,
    target_scorer_environment_identity,
    validate_checkpoint_collection,
    validate_selected_candidate,
    validate_self_hash,
)

PRECALIBRATION_SEAL_SCHEMA = "kbound_so2sat_precalibration_execution_seal_v1"
REVEAL_REGISTRY_IDENTITY_SCHEMA = "kbound_so2sat_outcome_reveal_registry_v1"
REVEAL_REGISTRY_IDENTITY_BASENAME = "so2sat_outcome_reveal_registry.json"
_PRODUCTION_PRECALIBRATION_BUILD_AUTHORITY = object()


def _reveal_registry_id(
    study_binding: Mapping[str, Any], selection: Mapping[str, Any]
) -> str:
    return stable_sha256(
        {
            "schema": REVEAL_REGISTRY_IDENTITY_SCHEMA,
            "protocol_id": study_binding["protocol_id"],
            "study_binding_sha256": study_binding["binding_sha256"],
            "population_identity_sha256": study_binding["population_identity_sha256"],
            "selection_sha256": selection["selection_sha256"],
        }
    )


def ensure_reveal_registry_identity(
    root: str | Path,
    *,
    study_binding: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create once or verify the portable identity of the external reveal registry."""

    directory = Path(root).expanduser().resolve()
    if not directory.is_dir() or not os.access(directory, os.W_OK):
        raise IntegrityError("outcome reveal registry root must exist and be writable")
    path = directory / REVEAL_REGISTRY_IDENTITY_BASENAME
    expected = {
        "schema": REVEAL_REGISTRY_IDENTITY_SCHEMA,
        "status": "CREATE_ONLY_OUTCOME_REVEAL_REGISTRY",
        "protocol_id": study_binding["protocol_id"],
        "study_binding_sha256": study_binding["binding_sha256"],
        "population_identity_sha256": study_binding["population_identity_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "registry_id": _reveal_registry_id(study_binding, selection),
        "policy": "ONE_CREATE_ONLY_REVEAL_LEDGER_PER_EXECUTION_SEAL",
    }
    expected["registry_identity_sha256"] = stable_sha256(expected)
    if path.exists() or path.with_name(path.name + ".receipt.json").exists():
        document, receipt = load_receipted_document(path)
        if document != expected:
            raise IntegrityError("existing outcome reveal registry has another identity")
        return document, receipt
    receipt = write_immutable_json_with_receipt(path, expected)
    return expected, receipt


def validate_reveal_registry_directory(
    root: str | Path, sealed_registry: Mapping[str, Any]
) -> Path:
    """Resolve a local registry root only when its portable identity matches the seal."""

    directory = Path(root).expanduser().resolve()
    if not directory.is_dir() or not os.access(directory, os.W_OK):
        raise IntegrityError("sealed outcome reveal registry root is unavailable")
    path = directory / REVEAL_REGISTRY_IDENTITY_BASENAME
    document, receipt = load_receipted_document(path)
    if (
        sealed_registry.get("registry_id") != document.get("registry_id")
        or sealed_registry.get("identity_basename") != path.name
        or sealed_registry.get("identity_artifact") != artifact_binding(receipt)
        or sealed_registry.get("policy")
        != "ONE_CREATE_ONLY_REVEAL_LEDGER_PER_EXECUTION_SEAL"
        or sealed_registry.get("ledger_basename_template")
        != "so2sat_target_outcome_reveal_{execution_seal_sha256}.json"
    ):
        raise IntegrityError("local outcome reveal registry differs from the sealed identity")
    validate_self_hash(document, field="registry_identity_sha256")
    return directory


def gate_algorithm_contract() -> dict[str, Any]:
    """Return every gate degree of freedom fixed before gate calibration."""

    return {
        "feature_names": list(FEATURE_NAMES),
        "ridge_penalty": RIDGE_PENALTY,
        "ridge_intercept_unpenalized": True,
        "ridge_standardization": "gate_fit_population_sd; zero_sd_replaced_by_one",
        "calibration_alpha": CALIBRATION_ALPHA,
        "calibration_method": (
            "split_conformal_over_city_max_checkpoint_absolute_residual"
        ),
        "calibration_aggregation_within_city": (
            "maximum_absolute_residual_over_five_checkpoints"
        ),
        "calibration_independent_city_count": CALIBRATION_CITY_COUNT,
        "calibration_order_statistic_rank_one_based": CONFORMAL_RANK,
        "decision_rule": {
            "adapt": "lower > 0",
            "freeze": "upper < 0",
            "otherwise": "ABSTAIN",
            "abstain_realized_action": "FREEZE",
        },
        "support": {
            "primary": "finite_values_and_exact_feature_schema",
            "failure_action": "ABSTAIN",
            "abstain_realized_action": "FREEZE",
        },
    }


def frozen_gate_fit_model(fit_bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Fit the ridge component using gate-fit rows only, before calibration."""

    cells = fit_bundle.get("cells")
    if fit_bundle.get("role") != GATE_FIT_ROLE or not isinstance(cells, list):
        raise IntegrityError("precalibration seal requires the selected gate-fit bundle")
    ordered = sorted(cells, key=lambda row: (str(row["city_id"]), str(row["checkpoint_id"])))
    rows = [row.get("gate_row") for row in ordered]
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise IntegrityError("selected gate-fit bundle lacks gate rows")
    features = np.vstack([feature_vector(row["feature_document"]) for row in rows])
    benefits = np.asarray([row["observed_benefit"] for row in rows], dtype=np.float64)
    if features.shape != (45, len(FEATURE_NAMES)) or benefits.shape != (45,):
        raise IntegrityError("selected gate-fit ridge design must be 9 cities x 5 checkpoints")
    means = features.mean(axis=0)
    raw_scales = features.std(axis=0, ddof=0)
    scales = np.where(raw_scales > 0.0, raw_scales, 1.0)
    standardized = (features - means) / scales
    design = np.column_stack([np.ones(len(standardized)), standardized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_PENALTY
    penalty[0, 0] = 0.0
    try:
        solution = np.linalg.solve(design.T @ design + penalty, design.T @ benefits)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - ridge is regularized
        raise IntegrityError("precalibration ridge fit failed") from exc
    if not np.isfinite(solution).all():
        raise IntegrityError("precalibration ridge fit is non-finite")
    return {
        "penalty": RIDGE_PENALTY,
        "intercept_unpenalized": True,
        "standardization": "gate_fit_population_sd; zero_sd_replaced_by_one",
        "intercept": float(solution[0]),
        "coefficients": [float(value) for value in solution[1:]],
        "fit_means": [float(value) for value in means],
        "fit_scales": [float(value) for value in scales],
    }


def precalibration_code_identity() -> dict[str, Any]:
    """Hash all code/protocol files whose settings are frozen by this seal."""

    directory = Path(__file__).resolve().parent
    names = (
        "integrity.py",
        "protocol.py",
        "metadata_manifest.py",
        "label_firewall.py",
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
        "target_contract.py",
        "precalibration_seal.py",
        "target_inference.py",
        "target_runner.py",
        "target_scorer.py",
        "target_seal.py",
        "target_boundary_amendment_v1_1.json",
        "target_boundary_amendment_v1_1.json.receipt.json",
        "prospective_protocol_v1.json",
        "prospective_protocol_v1.json.receipt.json",
    )
    files = {name: file_sha256(directory / name) for name in names}
    return {"files_sha256": files, "code_identity_sha256": stable_sha256(files)}


def development_calibration_environment_identity(device: torch.device) -> dict[str, Any]:
    """Freeze the exact runtime expected to consume gate-calibration rows."""

    if device.type not in {"cpu", "mps"}:
        raise IntegrityError("precalibration environment supports only CPU or MPS")
    versions: dict[str, str] = {}
    for package in ("h5py", "torch", "torchvision"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    document = {
        "schema": "kbound_so2sat_gate_calibration_environment_v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "numpy_version": str(np.__version__),
        "torch_version": versions["torch"],
        "torchvision_version": versions["torchvision"],
        "h5py_version": versions["h5py"],
        "device_type": device.type,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    document["environment_identity_sha256"] = stable_sha256(document)
    return document


def _checkpoint_identities(collection: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    rows = collection.get("checkpoints")
    if not isinstance(rows, list) or len(rows) != len(CHECKPOINT_IDS):
        raise IntegrityError("precalibration seal requires five checkpoints")
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        checkpoint_id = str(row.get("model_seed"))
        if checkpoint_id not in CHECKPOINT_IDS or checkpoint_id in output:
            raise IntegrityError("precalibration checkpoint ids must be seeds 0--4")
        output[checkpoint_id] = {
            "checkpoint_file_sha256": require_sha256(
                row.get("checkpoint_file_sha256"), field="checkpoint_file_sha256"
            ),
            "checkpoint_tensor_sha256": require_sha256(
                row.get("checkpoint_tensor_sha256"), field="checkpoint_tensor_sha256"
            ),
        }
    if set(output) != set(CHECKPOINT_IDS):
        raise IntegrityError("precalibration checkpoint coverage is incomplete")
    return dict(sorted(output.items()))


def build_precalibration_seal(
    *,
    study_binding: Mapping[str, Any],
    manifest_receipt: Mapping[str, Any],
    selection: Mapping[str, Any],
    selection_receipt: Mapping[str, Any],
    fit_bundle: Mapping[str, Any],
    fit_bundle_receipt: Mapping[str, Any],
    target_boundary_amendment: Mapping[str, Any],
    target_boundary_amendment_receipt: Mapping[str, Any],
    checkpoint_collection: Mapping[str, Any],
    checkpoint_collection_receipt: Mapping[str, Any],
    normalizer_sha256: str,
    normalizer_receipt: Mapping[str, Any],
    source_postrun_acceptance: Mapping[str, Any],
    source_postrun_acceptance_receipt: Mapping[str, Any],
    target_data_identities: Mapping[str, Mapping[str, Any]],
    reveal_registry_identity: Mapping[str, Any],
    reveal_registry_identity_receipt: Mapping[str, Any],
    package_code_identity: Mapping[str, Any],
    development_environment_identity: Mapping[str, Any],
    target_environment_identity: Mapping[str, Any],
    scorer_environment_identity: Mapping[str, Any],
    execution_mode: str = TEST_ONLY_MODE,
    _production_authority: object | None = None,
) -> dict[str, Any]:
    """Build a pre-calibration seal; production creation requires private authority."""

    if execution_mode == PRODUCTION_MODE and (
        _production_authority is not _PRODUCTION_PRECALIBRATION_BUILD_AUTHORITY
    ):
        raise IntegrityError("PRODUCTION precalibration seals require canonical authority")
    if execution_mode not in {PRODUCTION_MODE, TEST_ONLY_MODE}:
        raise IntegrityError("precalibration seal execution mode is invalid")
    validate_study_binding(study_binding)
    validate_selected_candidate(selection, study_binding=study_binding)
    validate_candidate_bundle(fit_bundle, study_binding=study_binding)
    validate_target_boundary_amendment(target_boundary_amendment)
    selected = selection["selected_candidate_id"]
    if (
        fit_bundle.get("role") != GATE_FIT_ROLE
        or fit_bundle.get("bundle_sha256") != selection.get("selected_bundle_sha256")
        or fit_bundle.get("candidate_spec", {}).get("candidate_id") != selected
    ):
        raise IntegrityError("precalibration selection and gate-fit bundle differ")
    gate_fit_environment = fit_bundle.get("development_environment_identity")
    if not isinstance(gate_fit_environment, Mapping):
        raise IntegrityError("precalibration gate-fit bundle lacks its environment")
    validate_development_environment_identity(gate_fit_environment)
    if selection.get("gate_fit_environment_identity") != gate_fit_environment:
        raise IntegrityError("precalibration selection/gate-fit environment mismatch")
    normalizer_hash = require_sha256(normalizer_sha256, field="normalizer_sha256")
    source_acceptance_binding = source_postrun_acceptance_binding(
        source_postrun_acceptance, source_postrun_acceptance_receipt
    )
    if (
        checkpoint_collection.get("normalizer_sha256") != normalizer_hash
        or fit_bundle.get("normalizer_sha256") != normalizer_hash
        or fit_bundle.get("checkpoint_collection_canonical_sha256")
        != stable_sha256(dict(checkpoint_collection))
    ):
        raise IntegrityError("precalibration source collection/normalizer binding differs")
    postrun_source = source_postrun_acceptance.get("postrun_source_container")
    acceptance_collection = source_postrun_acceptance.get("checkpoint_collection")
    acceptance_population = source_postrun_acceptance.get("population_manifest")
    selection_source_acceptance = selection.get("source_postrun_acceptance")
    if (
        selection_source_acceptance != source_acceptance_binding
        or not isinstance(postrun_source, Mapping)
        or not isinstance(acceptance_collection, Mapping)
        or not isinstance(acceptance_population, Mapping)
        or postrun_source.get("source_container_identity_sha256")
        != fit_bundle.get("source_container_identity_sha256")
        or acceptance_collection.get("normalizer_sha256") != normalizer_hash
        or acceptance_population.get("manifest_sha256")
        != study_binding["manifest_sha256"]
        or acceptance_population.get("population_identity_sha256")
        != study_binding["population_identity_sha256"]
    ):
        raise IntegrityError("precalibration source post-run acceptance chain differs")
    expected_registry_id = _reveal_registry_id(study_binding, selection)
    if (
        reveal_registry_identity.get("registry_id") != expected_registry_id
        or reveal_registry_identity.get("schema") != REVEAL_REGISTRY_IDENTITY_SCHEMA
    ):
        raise IntegrityError("precalibration reveal registry identity mismatch")
    target_identities = normalize_target_data_identities(target_data_identities)
    candidate = candidate_spec(str(selected))
    document = {
        "schema": PRECALIBRATION_SEAL_SCHEMA,
        "status": (
            "SEALED_AFTER_GATE_FIT_SELECTION_BEFORE_GATE_CALIBRATION_OR_TARGET_PIXEL_ACCESS"
            if execution_mode == PRODUCTION_MODE
            else "TEST_ONLY_PRECALIBRATION_SEAL_WITH_SYNTHETIC_OR_INJECTED_INPUTS"
        ),
        "execution_mode": execution_mode,
        "protocol_id": study_binding["protocol_id"],
        "study_binding_sha256": study_binding["binding_sha256"],
        "manifest_sha256": study_binding["manifest_sha256"],
        "population_identity_sha256": study_binding["population_identity_sha256"],
        "protocol_file_sha256": study_binding["protocol_file_sha256"],
        "protocol_document_sha256": study_binding["protocol_document_sha256"],
        "population_manifest_artifact": artifact_binding(manifest_receipt),
        "selection_artifact": artifact_binding(selection_receipt),
        "selection_sha256": selection["selection_sha256"],
        "selected_gate_fit_bundle_artifact": artifact_binding(fit_bundle_receipt),
        "selected_gate_fit_bundle_sha256": fit_bundle["bundle_sha256"],
        "gate_fit_rows_sha256": fit_bundle["gate_rows_sha256"],
        "candidate_id": selected,
        "candidate_config_sha256": candidate["candidate_config_sha256"],
        "target_boundary_amendment_artifact": artifact_binding(
            target_boundary_amendment_receipt
        ),
        "target_boundary_amendment_sha256": stable_sha256(
            dict(target_boundary_amendment)
        ),
        "checkpoint_collection_artifact": artifact_binding(
            checkpoint_collection_receipt
        ),
        "checkpoint_collection_document_sha256": stable_sha256(
            dict(checkpoint_collection)
        ),
        "checkpoint_identities": _checkpoint_identities(checkpoint_collection),
        "source_container_identity_sha256": require_sha256(
            fit_bundle.get("source_container_identity_sha256"),
            field="source_container_identity_sha256",
        ),
        "source_data_identity_sha256": require_sha256(
            checkpoint_collection.get("data_identity_sha256"),
            field="source_data_identity_sha256",
        ),
        "normalizer_sha256": normalizer_hash,
        "normalizer_artifact": artifact_binding(normalizer_receipt),
        "source_postrun_acceptance": source_acceptance_binding,
        "source_postrun_acceptance_artifact_sha256": source_acceptance_binding[
            "source_postrun_acceptance_artifact_sha256"
        ],
        "source_postrun_training_container": dict(postrun_source),
        "source_hdf5_runtime_disclosure": dict(
            source_postrun_acceptance["source_hdf5_runtime_disclosure"]
        ),
        "source_checkpoint_selection_disclosure": dict(
            source_postrun_acceptance["source_checkpoint_selection_disclosure"]
        ),
        "source_initialization_clarification": dict(
            source_postrun_acceptance["source_initialization_clarification"]
        ),
        "target_data_identities": target_identities,
        "gate_algorithm_contract": gate_algorithm_contract(),
        "frozen_gate_fit_model": frozen_gate_fit_model(fit_bundle),
        "development_runner_code": dict(fit_bundle["runner_code"]),
        "gate_fit_development_environment_identity": dict(
            gate_fit_environment
        ),
        "package_code_identity": dict(package_code_identity),
        "development_calibration_environment_identity": dict(
            development_environment_identity
        ),
        "target_live_environment_identity": dict(target_environment_identity),
        "offline_scorer_environment_identity": dict(scorer_environment_identity),
        "outcome_reveal_registry": {
            "registry_id": expected_registry_id,
            "identity_basename": REVEAL_REGISTRY_IDENTITY_BASENAME,
            "identity_artifact": artifact_binding(reveal_registry_identity_receipt),
            "ledger_basename_template": (
                "so2sat_target_outcome_reveal_{execution_seal_sha256}.json"
            ),
            "policy": "ONE_CREATE_ONLY_REVEAL_LEDGER_PER_EXECUTION_SEAL",
        },
        "seal_creation_audit": {
            "created_after_gate_fit_selection": True,
            "created_before_gate_calibration": True,
            "gate_calibration_rows_opened": 0,
            "gate_calibration_labels_opened": 0,
            "target_container_hash_method": "opaque_raw_file_bytes_sha256",
            "target_hdf5_datasets_deserialized": 0,
            "target_pixels_opened": 0,
            "target_labels_opened": 0,
        },
    }
    document["precalibration_seal_sha256"] = stable_sha256(document)
    validate_precalibration_seal(
        document,
        study_binding=study_binding,
        selection=selection,
        fit_bundle=fit_bundle,
        target_boundary_amendment=target_boundary_amendment,
        checkpoint_collection=checkpoint_collection,
    )
    return document


def validate_precalibration_seal(
    document: Mapping[str, Any],
    *,
    study_binding: Mapping[str, Any],
    selection: Mapping[str, Any],
    fit_bundle: Mapping[str, Any],
    target_boundary_amendment: Mapping[str, Any],
    checkpoint_collection: Mapping[str, Any],
) -> None:
    """Validate the complete immutable pre-calibration configuration chain."""

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
        "population_manifest_artifact",
        "selection_artifact",
        "selection_sha256",
        "selected_gate_fit_bundle_artifact",
        "selected_gate_fit_bundle_sha256",
        "gate_fit_rows_sha256",
        "candidate_id",
        "candidate_config_sha256",
        "target_boundary_amendment_artifact",
        "target_boundary_amendment_sha256",
        "checkpoint_collection_artifact",
        "checkpoint_collection_document_sha256",
        "checkpoint_identities",
        "source_container_identity_sha256",
        "source_data_identity_sha256",
        "normalizer_sha256",
        "normalizer_artifact",
        "source_postrun_acceptance",
        "source_postrun_acceptance_artifact_sha256",
        "source_postrun_training_container",
        "source_hdf5_runtime_disclosure",
        "source_checkpoint_selection_disclosure",
        "source_initialization_clarification",
        "target_data_identities",
        "gate_algorithm_contract",
        "frozen_gate_fit_model",
        "development_runner_code",
        "gate_fit_development_environment_identity",
        "package_code_identity",
        "development_calibration_environment_identity",
        "target_live_environment_identity",
        "offline_scorer_environment_identity",
        "outcome_reveal_registry",
        "seal_creation_audit",
        "precalibration_seal_sha256",
    }
    if not isinstance(document, Mapping) or set(document) != expected_keys:
        raise IntegrityError("precalibration seal has unknown or missing fields")
    mode = document.get("execution_mode")
    expected_status = (
        "SEALED_AFTER_GATE_FIT_SELECTION_BEFORE_GATE_CALIBRATION_OR_TARGET_PIXEL_ACCESS"
        if mode == PRODUCTION_MODE
        else "TEST_ONLY_PRECALIBRATION_SEAL_WITH_SYNTHETIC_OR_INJECTED_INPUTS"
        if mode == TEST_ONLY_MODE
        else None
    )
    if document.get("schema") != PRECALIBRATION_SEAL_SCHEMA or document.get(
        "status"
    ) != expected_status:
        raise IntegrityError("unknown or unsealed precalibration artifact")
    validate_study_binding(study_binding)
    validate_selected_candidate(selection, study_binding=study_binding)
    validate_candidate_bundle(fit_bundle, study_binding=study_binding)
    validate_target_boundary_amendment(target_boundary_amendment)
    bindings = {
        "protocol_id": "protocol_id",
        "study_binding_sha256": "binding_sha256",
        "manifest_sha256": "manifest_sha256",
        "population_identity_sha256": "population_identity_sha256",
        "protocol_file_sha256": "protocol_file_sha256",
        "protocol_document_sha256": "protocol_document_sha256",
    }
    if any(document[field] != study_binding[source] for field, source in bindings.items()):
        raise IntegrityError("precalibration study binding differs")
    candidate = candidate_spec(str(selection["selected_candidate_id"]))
    if (
        document.get("selection_sha256") != selection["selection_sha256"]
        or document.get("selected_gate_fit_bundle_sha256")
        != fit_bundle["bundle_sha256"]
        or document.get("gate_fit_rows_sha256") != fit_bundle["gate_rows_sha256"]
        or document.get("candidate_id") != selection["selected_candidate_id"]
        or document.get("candidate_config_sha256")
        != candidate["candidate_config_sha256"]
        or document.get("target_boundary_amendment_sha256")
        != stable_sha256(dict(target_boundary_amendment))
        or document.get("checkpoint_collection_document_sha256")
        != stable_sha256(dict(checkpoint_collection))
        or document.get("checkpoint_identities")
        != _checkpoint_identities(checkpoint_collection)
        or document.get("normalizer_sha256")
        != checkpoint_collection["normalizer_sha256"]
        or document.get("source_postrun_acceptance")
        != selection.get("source_postrun_acceptance")
        or document.get("source_postrun_acceptance_artifact_sha256")
        != selection.get("source_postrun_acceptance", {}).get(
            "source_postrun_acceptance_artifact_sha256"
        )
    ):
        raise IntegrityError("precalibration candidate/source identity mismatch")
    for field in (
        "population_manifest_artifact",
        "selection_artifact",
        "selected_gate_fit_bundle_artifact",
        "target_boundary_amendment_artifact",
        "checkpoint_collection_artifact",
        "normalizer_artifact",
    ):
        value = document.get(field)
        if not isinstance(value, Mapping):
            raise IntegrityError(f"precalibration {field} is invalid")
        artifact_binding(value)
    normalize_target_data_identities(document.get("target_data_identities"))
    postrun_source = document.get("source_postrun_training_container")
    hdf5_disclosure = document.get("source_hdf5_runtime_disclosure")
    disclosure = document.get("source_checkpoint_selection_disclosure")
    initialization = document.get("source_initialization_clarification")
    if (
        not isinstance(postrun_source, Mapping)
        or set(postrun_source)
        != {
            "basename",
            "bytes",
            "sha256",
            "source_container_identity_sha256",
            "matches_source_preflight",
            "stable_during_hash",
            "hdf5_datasets_opened",
        }
        or postrun_source.get("basename") != "training.h5"
        or postrun_source.get("source_container_identity_sha256")
        != document.get("source_container_identity_sha256")
        or postrun_source.get("matches_source_preflight") is not True
        or postrun_source.get("stable_during_hash") is not True
        or postrun_source.get("hdf5_datasets_opened") is not False
        or not isinstance(hdf5_disclosure, Mapping)
        or hdf5_disclosure.get(
            "source_preflight_explicit_h5py_version_recorded"
        )
        is not False
        or hdf5_disclosure.get(
            "postrun_acceptance_h5py_version_is_retroactive_source_runtime_proof"
        )
        is not False
        or not isinstance(
            hdf5_disclosure.get("postrun_acceptance_h5py_version"), str
        )
        or not isinstance(disclosure, Mapping)
        or disclosure.get("source_monitor_supported_class_count") != 15
        or disclosure.get("source_monitor_absent_class_ids") != [0, 6]
        or disclosure.get("is_17_class_macro_recall") is not False
        or disclosure.get("development_target_endpoint") != "top1_accuracy"
        or not isinstance(initialization, Mapping)
        or initialization.get("exact_initial_tensor_hashes_authoritative")
        is not True
        or initialization.get("numerical_artifacts_changed_by_clarification")
        is not False
    ):
        raise IntegrityError("precalibration source acceptance disclosure drift")
    byte_count = postrun_source.get("bytes")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 1
    ):
        raise IntegrityError("precalibration post-run training byte count is invalid")
    require_sha256(postrun_source.get("sha256"), field="postrun_training.sha256")
    for field in (
        "source_postrun_acceptance_artifact_sha256",
        "source_postrun_acceptance_canonical_document_sha256",
    ):
        require_sha256(document["source_postrun_acceptance"].get(field), field=field)
    if document.get("gate_algorithm_contract") != gate_algorithm_contract():
        raise IntegrityError("precalibration gate algorithm contract drift")
    if document.get("frozen_gate_fit_model") != frozen_gate_fit_model(fit_bundle):
        raise IntegrityError("precalibration frozen ridge fit changed")
    if document.get("development_runner_code") != fit_bundle["runner_code"]:
        raise IntegrityError("precalibration development code differs from gate-fit")
    gate_fit_environment = document.get(
        "gate_fit_development_environment_identity"
    )
    if not isinstance(gate_fit_environment, Mapping):
        raise IntegrityError("precalibration gate-fit environment is invalid")
    validate_development_environment_identity(gate_fit_environment)
    if (
        gate_fit_environment != fit_bundle.get("development_environment_identity")
        or gate_fit_environment != selection.get("gate_fit_environment_identity")
    ):
        raise IntegrityError("precalibration gate-fit environment binding drift")
    package = document.get("package_code_identity")
    if (
        not isinstance(package, Mapping)
        or stable_sha256(package.get("files_sha256"))
        != package.get("code_identity_sha256")
    ):
        raise IntegrityError("precalibration package code identity is invalid")
    for field in (
        "development_calibration_environment_identity",
        "target_live_environment_identity",
        "offline_scorer_environment_identity",
    ):
        identity = document.get(field)
        if not isinstance(identity, Mapping):
            raise IntegrityError(f"precalibration {field} is invalid")
        validate_self_hash(identity, field="environment_identity_sha256")
    registry = document.get("outcome_reveal_registry")
    if not isinstance(registry, Mapping) or set(registry) != {
        "registry_id",
        "identity_basename",
        "identity_artifact",
        "ledger_basename_template",
        "policy",
    }:
        raise IntegrityError("precalibration reveal registry schema drift")
    if (
        registry.get("registry_id") != _reveal_registry_id(study_binding, selection)
        or registry.get("identity_basename") != REVEAL_REGISTRY_IDENTITY_BASENAME
        or not isinstance(registry.get("identity_artifact"), Mapping)
        or registry.get("ledger_basename_template")
        != "so2sat_target_outcome_reveal_{execution_seal_sha256}.json"
        or registry.get("policy")
        != "ONE_CREATE_ONLY_REVEAL_LEDGER_PER_EXECUTION_SEAL"
    ):
        raise IntegrityError("precalibration reveal registry is invalid")
    artifact_binding(registry["identity_artifact"])
    if document.get("seal_creation_audit") != {
        "created_after_gate_fit_selection": True,
        "created_before_gate_calibration": True,
        "gate_calibration_rows_opened": 0,
        "gate_calibration_labels_opened": 0,
        "target_container_hash_method": "opaque_raw_file_bytes_sha256",
        "target_hdf5_datasets_deserialized": 0,
        "target_pixels_opened": 0,
        "target_labels_opened": 0,
    }:
        raise IntegrityError("precalibration seal creation audit drift")
    validate_self_hash(document, field="precalibration_seal_sha256")


def load_precalibration_seal_with_receipt(
    path: str | Path,
    *,
    study_binding: Mapping[str, Any],
    selection: Mapping[str, Any],
    fit_bundle: Mapping[str, Any],
    target_boundary_amendment: Mapping[str, Any],
    checkpoint_collection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, receipt = load_receipted_document(path)
    validate_precalibration_seal(
        document,
        study_binding=study_binding,
        selection=selection,
        fit_bundle=fit_bundle,
        target_boundary_amendment=target_boundary_amendment,
        checkpoint_collection=checkpoint_collection,
    )
    return document, receipt


def create_production_precalibration_seal(
    *,
    population_manifest_path: str | Path,
    source_postrun_acceptance_path: str | Path,
    source_preflight_path: str | Path,
    training_data_path: str | Path,
    selected_candidate_path: str | Path,
    selected_gate_fit_bundle_path: str | Path,
    target_boundary_amendment_path: str | Path,
    checkpoint_collection_path: str | Path,
    checkpoint_dir: str | Path,
    normalizer_path: str | Path,
    target_data_paths: Mapping[str, str | Path],
    reveal_registry_dir: str | Path,
    output_path: str | Path,
    calibration_device_name: str,
    target_device_name: str,
) -> Path:
    """Create the production pre-calibration seal without opening target HDF5."""

    if calibration_device_name not in {"cpu", "mps"} or target_device_name not in {
        "cpu",
        "mps",
    }:
        raise IntegrityError("precalibration devices must be exactly CPU or MPS")
    if (
        "mps" in {calibration_device_name, target_device_name}
        and not torch.backends.mps.is_available()
    ):
        raise IntegrityError("MPS precalibration seal requested but MPS is unavailable")
    manifest, manifest_receipt = load_receipted_document(population_manifest_path)
    from .metadata_manifest import validate_population_manifest

    validate_population_manifest(manifest)
    source_acceptance, source_acceptance_receipt = (
        verify_source_postrun_acceptance_bindings(
            source_postrun_acceptance_path,
            population_manifest_path=population_manifest_path,
            source_preflight_path=source_preflight_path,
            training_data_path=training_data_path,
            checkpoint_dir=checkpoint_dir,
        )
    )
    binding = load_study_binding(population_manifest_path)
    selection, selection_receipt = load_receipted_document(selected_candidate_path)
    validate_selected_candidate(selection, study_binding=binding)
    fit_bundle, fit_receipt = load_receipted_document(selected_gate_fit_bundle_path)
    validate_candidate_bundle(fit_bundle, study_binding=binding)
    amendment, amendment_receipt = load_target_boundary_amendment(
        target_boundary_amendment_path
    )
    collection, collection_receipt = load_receipted_document(
        checkpoint_collection_path
    )
    validate_checkpoint_collection(
        collection,
        collection_receipt=collection_receipt,
        collection_path=checkpoint_collection_path,
        checkpoint_dir=checkpoint_dir,
    )
    normalizer = load_sealed_band_normalizer(normalizer_path)
    normalizer_receipt = verify_artifact_receipt(normalizer_path)
    if normalizer.normalizer_sha256 != collection.get("normalizer_sha256"):
        raise IntegrityError("precalibration normalizer differs from checkpoints")
    development_code = _runner_code_identity()
    if fit_bundle.get("runner_code") != development_code:
        raise IntegrityError("development code changed after gate-fit selection")
    package_code = precalibration_code_identity()
    target_identities = opaque_target_identities_from_paths(target_data_paths)
    registry_identity, registry_identity_receipt = ensure_reveal_registry_identity(
        reveal_registry_dir,
        study_binding=binding,
        selection=selection,
    )
    from .target_inference import target_runtime_environment_identity

    target_environment = target_runtime_environment_identity(
        torch.device(target_device_name)
    )
    document = build_precalibration_seal(
        study_binding=binding,
        manifest_receipt=manifest_receipt,
        selection=selection,
        selection_receipt=selection_receipt,
        fit_bundle=fit_bundle,
        fit_bundle_receipt=fit_receipt,
        target_boundary_amendment=amendment,
        target_boundary_amendment_receipt=amendment_receipt,
        checkpoint_collection=collection,
        checkpoint_collection_receipt=collection_receipt,
        normalizer_sha256=normalizer.normalizer_sha256,
        normalizer_receipt=normalizer_receipt,
        source_postrun_acceptance=source_acceptance,
        source_postrun_acceptance_receipt=source_acceptance_receipt,
        target_data_identities=target_identities,
        reveal_registry_identity=registry_identity,
        reveal_registry_identity_receipt=registry_identity_receipt,
        package_code_identity=package_code,
        development_environment_identity=development_calibration_environment_identity(
            torch.device(calibration_device_name)
        ),
        target_environment_identity=target_environment,
        scorer_environment_identity=target_scorer_environment_identity(),
        execution_mode=PRODUCTION_MODE,
        _production_authority=_PRODUCTION_PRECALIBRATION_BUILD_AUTHORITY,
    )
    if precalibration_code_identity() != package_code:
        raise IntegrityError("package code changed while creating precalibration seal")
    destination = Path(output_path).expanduser().resolve()
    write_immutable_json_with_receipt(destination, document)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-manifest", required=True)
    parser.add_argument("--source-postrun-acceptance", required=True)
    parser.add_argument("--source-preflight", required=True)
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--selected-candidate", required=True)
    parser.add_argument("--selected-gate-fit-bundle", required=True)
    parser.add_argument("--target-boundary-amendment", required=True)
    parser.add_argument("--checkpoint-collection", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--testing-data", required=True)
    parser.add_argument("--reveal-registry-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--target-device", choices=("cpu", "mps"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = create_production_precalibration_seal(
        population_manifest_path=arguments.population_manifest,
        source_postrun_acceptance_path=arguments.source_postrun_acceptance,
        source_preflight_path=arguments.source_preflight,
        training_data_path=arguments.training_data,
        selected_candidate_path=arguments.selected_candidate,
        selected_gate_fit_bundle_path=arguments.selected_gate_fit_bundle,
        target_boundary_amendment_path=arguments.target_boundary_amendment,
        checkpoint_collection_path=arguments.checkpoint_collection,
        checkpoint_dir=arguments.checkpoint_dir,
        normalizer_path=arguments.normalizer,
        target_data_paths={
            "validation": arguments.validation_data,
            "testing": arguments.testing_data,
        },
        reveal_registry_dir=arguments.reveal_registry_dir,
        output_path=arguments.output,
        calibration_device_name=arguments.calibration_device,
        target_device_name=arguments.target_device,
    )
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI integration
    raise SystemExit(main())
