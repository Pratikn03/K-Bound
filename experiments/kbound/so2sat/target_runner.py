"""Label-blind live target execution for the prospective So2Sat study.

This module is the *live* side of a two-process boundary.  It can receive target
pixels only through :class:`LabelFreeTargetLoader`; it has no outcome-array
name, label argument, or scoring function.  Each checkpoint-by-city cell is
written with a create-only receipt.  A master bundle is written last, only
after all 50 cells and the loader access audit are complete.

The production entry point constructs exact concrete geo, pixel-loader, and
PyTorch executor types.  Dependency injection exists only behind a TEST_ONLY
helper whose artifacts cannot become confirmatory evidence.  Executors receive
pixels, never target paths or outcome handles.
"""

from __future__ import annotations

import argparse
import copy
import io
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .adapters import CANDIDATE_IDS
from .development import (
    load_gate_authorization_with_receipt,
    validate_gate_authorization,
)
from .features import N_CLASSES, extract_label_free_features
from .gate import (
    CHECKPOINT_IDS,
    apply_gate,
    load_gate_with_receipt,
    trace_identity_sha256,
    validate_gate_document,
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
from .label_firewall import LabelFreeTargetLoader, PixelSample, VerifiedGeoIndex
from .metadata_manifest import validate_population_manifest
from .precalibration_seal import (
    load_precalibration_seal_with_receipt,
    validate_precalibration_seal,
)
from .protocol import PROTOCOL_ID, require_production_target_action_unit_alignment
from .target_amendment import (
    load_target_boundary_amendment,
    validate_target_boundary_amendment,
)
from .target_contract import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXECUTION_SEAL_SCHEMA,
    INFERENCE_ALPHA,
    LOGIT_ARCHIVE_MANIFEST_SCHEMA,
    PRODUCTION_MODE,
    TARGET_BUNDLE_SCHEMA,
    TARGET_CELL_COUNT,
    TARGET_CELL_SCHEMA,
    TARGET_SPLITS,
    TEST_ONLY_MODE,
    load_source_postrun_acceptance_pair,
    validate_checkpoint_collection,
    validate_complete_target_bundle_document,
    validate_execution_seal,
    validate_selected_candidate,
    validate_target_cell,
)
from .target_contract import (
    artifact_binding as _artifact_binding,
)
from .target_contract import (
    load_receipted_document as _receipt_document,
)
from .target_contract import (
    normalize_target_data_identities as _target_identities,
)
from .target_contract import (
    selected_candidate_view as _selected_candidate_view,
)
from .target_contract import (
    tensor_sha256 as _tensor_sha256,
)

_PRODUCTION_SEAL_BUILD_AUTHORITY = object()
_PRODUCTION_RUN_AUTHORITY = object()


def build_execution_seal(
    *,
    study_binding: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
    selected_candidate_receipt: Mapping[str, Any],
    selected_gate_fit_bundle: Mapping[str, Any],
    gate: Mapping[str, Any],
    gate_receipt: Mapping[str, Any],
    gate_authorization: Mapping[str, Any],
    gate_authorization_receipt: Mapping[str, Any],
    target_boundary_amendment: Mapping[str, Any],
    target_boundary_amendment_receipt: Mapping[str, Any],
    checkpoint_collection: Mapping[str, Any],
    checkpoint_collection_receipt: Mapping[str, Any],
    precalibration_seal: Mapping[str, Any],
    precalibration_seal_receipt: Mapping[str, Any],
    target_data_identities: Mapping[str, Mapping[str, Any]],
    code_identity_sha256: str,
    environment_identity_sha256: str,
    scorer_code_identity_sha256: str,
    scorer_environment_identity_sha256: str,
    execution_mode: str = TEST_ONLY_MODE,
    _production_authority: object | None = None,
) -> dict[str, Any]:
    """Build the exact seal that must predate every target-pixel read."""

    if (
        execution_mode == PRODUCTION_MODE
        and _production_authority is not _PRODUCTION_SEAL_BUILD_AUTHORITY
    ):
        raise IntegrityError(
            "PRODUCTION execution seals require the canonical target-seal authority"
        )
    validate_study_binding(study_binding)
    validate_selected_candidate(selected_candidate, study_binding=study_binding)
    validate_gate_document(gate)
    validate_gate_authorization(
        gate_authorization,
        selection=selected_candidate,
        gate=gate,
        study_binding=study_binding,
    )
    validate_target_boundary_amendment(target_boundary_amendment)
    validate_precalibration_seal(
        precalibration_seal,
        study_binding=study_binding,
        selection=selected_candidate,
        fit_bundle=selected_gate_fit_bundle,
        target_boundary_amendment=target_boundary_amendment,
        checkpoint_collection=checkpoint_collection,
    )
    if gate.get("study_binding") != dict(study_binding):
        raise IntegrityError("execution seal gate study binding mismatch")
    if execution_mode not in {PRODUCTION_MODE, TEST_ONLY_MODE}:
        raise IntegrityError("execution seal mode must be PRODUCTION or TEST_ONLY")
    selected = _selected_candidate_view(selected_candidate)
    rows = checkpoint_collection.get("checkpoints")
    if not isinstance(rows, list) or len(rows) != 5:
        raise IntegrityError("execution seal requires the complete checkpoint collection")
    checkpoint_identities: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise IntegrityError("checkpoint collection row must be a mapping")
        checkpoint_id = str(row.get("model_seed"))
        if checkpoint_id not in CHECKPOINT_IDS or checkpoint_id in checkpoint_identities:
            raise IntegrityError("execution seal checkpoint ids must be seeds 0--4")
        checkpoint_identities[checkpoint_id] = {
            "checkpoint_file_sha256": require_sha256(
                row.get("checkpoint_file_sha256"), field="checkpoint_file_sha256"
            ),
            "checkpoint_tensor_sha256": require_sha256(
                row.get("checkpoint_tensor_sha256"), field="checkpoint_tensor_sha256"
            ),
        }
    if set(checkpoint_identities) != set(CHECKPOINT_IDS):
        raise IntegrityError("execution seal checkpoint coverage is incomplete")
    document = {
        "schema": EXECUTION_SEAL_SCHEMA,
        "status": (
            "SEALED_BEFORE_ANY_TARGET_PIXEL_ACCESS"
            if execution_mode == PRODUCTION_MODE
            else "TEST_ONLY_SEALED_WITH_SYNTHETIC_OR_INJECTED_DEPENDENCIES"
        ),
        "execution_mode": execution_mode,
        "protocol_id": PROTOCOL_ID,
        "study_binding_sha256": study_binding["binding_sha256"],
        "manifest_sha256": study_binding["manifest_sha256"],
        "population_identity_sha256": study_binding["population_identity_sha256"],
        "protocol_file_sha256": study_binding["protocol_file_sha256"],
        "protocol_document_sha256": study_binding["protocol_document_sha256"],
        "selected_candidate_sha256": selected["selected_candidate_sha256"],
        "selected_candidate_artifact": _artifact_binding(selected_candidate_receipt),
        "candidate_id": selected["candidate_id"],
        "candidate_config_sha256": selected["candidate_spec"][
            "candidate_config_sha256"
        ],
        "gate_artifact": _artifact_binding(gate_receipt),
        "gate_sha256": gate["gate_sha256"],
        "gate_authorization_artifact": _artifact_binding(gate_authorization_receipt),
        "gate_authorization_sha256": gate_authorization["authorization_sha256"],
        "target_boundary_amendment_artifact": _artifact_binding(
            target_boundary_amendment_receipt
        ),
        "target_boundary_amendment_sha256": stable_sha256(
            dict(target_boundary_amendment)
        ),
        "precalibration_seal_artifact": _artifact_binding(
            precalibration_seal_receipt
        ),
        "precalibration_seal_sha256": precalibration_seal[
            "precalibration_seal_sha256"
        ],
        "source_postrun_acceptance": copy.deepcopy(
            dict(precalibration_seal["source_postrun_acceptance"])
        ),
        "source_postrun_acceptance_artifact_sha256": precalibration_seal[
            "source_postrun_acceptance_artifact_sha256"
        ],
        "source_postrun_training_container": copy.deepcopy(
            dict(precalibration_seal["source_postrun_training_container"])
        ),
        "source_hdf5_runtime_disclosure": copy.deepcopy(
            dict(precalibration_seal["source_hdf5_runtime_disclosure"])
        ),
        "source_checkpoint_selection_disclosure": copy.deepcopy(
            dict(precalibration_seal["source_checkpoint_selection_disclosure"])
        ),
        "source_initialization_clarification": copy.deepcopy(
            dict(precalibration_seal["source_initialization_clarification"])
        ),
        "checkpoint_collection_artifact": _artifact_binding(checkpoint_collection_receipt),
        "checkpoint_collection_document_sha256": stable_sha256(dict(checkpoint_collection)),
        "checkpoint_identities": dict(sorted(checkpoint_identities.items())),
        "target_data_identities": _target_identities(target_data_identities),
        "outcome_reveal_registry": copy.deepcopy(
            dict(precalibration_seal["outcome_reveal_registry"])
        ),
        "code_identity_sha256": require_sha256(
            code_identity_sha256, field="code_identity_sha256"
        ),
        "environment_identity_sha256": require_sha256(
            environment_identity_sha256, field="environment_identity_sha256"
        ),
        "scorer_code_identity_sha256": require_sha256(
            scorer_code_identity_sha256, field="scorer_code_identity_sha256"
        ),
        "scorer_environment_identity_sha256": require_sha256(
            scorer_environment_identity_sha256,
            field="scorer_environment_identity_sha256",
        ),
        "live_contract": {
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
        },
        "seal_creation_audit": {
            "extends_precalibration_seal": True,
            "gate_calibration_complete": True,
            "target_container_hash_method": "opaque_raw_file_bytes_sha256",
            "target_hdf5_datasets_deserialized": 0,
            "target_pixels_opened": 0,
            "target_labels_opened": 0,
        },
        "inference_contract": {
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
        },
    }
    document["execution_seal_sha256"] = stable_sha256(document)
    validate_execution_seal(
        document,
        study_binding=study_binding,
        selected_candidate=selected_candidate,
        gate=gate,
        gate_authorization=gate_authorization,
        target_boundary_amendment=target_boundary_amendment,
        checkpoint_collection=checkpoint_collection,
        precalibration_seal=precalibration_seal,
    )
    return document


@dataclass(frozen=True)
class CellComputation:
    """Compatibility aggregate used only by direct executor replay tests."""

    frozen_probe_logits: Any
    adapted_probe_logits: Any
    frozen_evaluation_logits: Any
    adapted_evaluation_logits: Any
    normalized_adapter_update_norm: float
    batchnorm_source_statistic_divergence: float


@dataclass(frozen=True)
class ProbeComputation:
    """Probe-only values and opaque model state produced before testing pixels."""

    frozen_probe_logits: Any
    adapted_probe_logits: Any
    normalized_adapter_update_norm: float
    batchnorm_source_statistic_divergence: float
    opaque_evaluation_state: Any


@dataclass(frozen=True)
class EvaluationComputation:
    """Fixed-policy testing logits produced only after the action receipt exists."""

    frozen_evaluation_logits: Any
    adapted_evaluation_logits: Any


class CellExecutor(Protocol):
    candidate_id: str
    normalizer_sha256: str
    code_identity_sha256: str
    environment_identity_sha256: str

    def prepare_probe(
        self,
        checkpoint: Mapping[str, Any],
        candidate_spec: Mapping[str, Any],
        probe_samples: Sequence[PixelSample],
    ) -> ProbeComputation: ...

    def evaluate_after_action(
        self,
        probe_computation: ProbeComputation,
        evaluation_samples: Sequence[PixelSample],
    ) -> EvaluationComputation: ...


def _logits(value: Any, *, rows: int, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError(f"{field} must be a numeric logit matrix") from exc
    if array.shape != (rows, N_CLASSES) or not np.isfinite(array).all():
        raise IntegrityError(
            f"{field} must have finite shape {(rows, N_CLASSES)}, found {array.shape}"
        )
    return np.ascontiguousarray(array)


def _exclusive_binary_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise IntegrityError(f"refusing to overwrite immutable logit archive: {path}") from exc
    finally:
        if descriptor is not None:  # pragma: no cover - exceptional cleanup
            os.close(descriptor)


def _write_logit_archive(
    path: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    execution_mode: str,
) -> dict[str, Any]:
    expected_names = {
        "frozen_probe_logits",
        "adapted_probe_logits",
        "frozen_evaluation_logits",
        "adapted_evaluation_logits",
    }
    if set(arrays) != expected_names:
        raise IntegrityError("logit archive array inventory drift")
    normalized: dict[str, np.ndarray] = {}
    for name in sorted(expected_names):
        array = np.ascontiguousarray(arrays[name], dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != N_CLASSES or not np.isfinite(array).all():
            raise IntegrityError(f"logit archive {name} is invalid")
        normalized[name] = array
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **normalized)
    payload = buffer.getvalue()
    _exclusive_binary_write(path, payload)
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = {
        "schema": LOGIT_ARCHIVE_MANIFEST_SCHEMA,
        "status": (
            "REPLAYABLE_LOGITS_SEALED_BEFORE_TARGET_OUTCOME_ACCESS"
            if execution_mode == PRODUCTION_MODE
            else "TEST_ONLY_REPLAYABLE_SYNTHETIC_LOGITS"
        ),
        "execution_mode": execution_mode,
        "archive_basename": path.name,
        "archive_bytes": len(payload),
        "archive_sha256": file_sha256(path),
        "compression": "numpy_savez_compressed",
        "arrays": {
            name: {
                "dtype": "float64",
                "shape": list(array.shape),
                "tensor_sha256": _tensor_sha256(array),
            }
            for name, array in sorted(normalized.items())
        },
        "target_outcomes_present": False,
    }
    manifest["logit_archive_manifest_sha256"] = stable_sha256(manifest)
    receipt = write_immutable_json_with_receipt(manifest_path, manifest)
    return {
        "archive_basename": path.name,
        "archive_bytes": len(payload),
        "archive_sha256": manifest["archive_sha256"],
        "manifest_basename": manifest_path.name,
        "manifest_sha256": manifest["logit_archive_manifest_sha256"],
        "manifest_artifact": _artifact_binding(receipt),
    }


def _freeze_sample_pixels(samples: Sequence[PixelSample]) -> None:
    for sample in samples:
        pixels = sample.pixels
        if isinstance(pixels, np.ndarray):
            pixels.setflags(write=False)


def _partition(
    geo_index: VerifiedGeoIndex,
    manifest: Mapping[str, Any],
    *,
    split: str,
    cities: Sequence[str],
) -> dict[str, list[Any]]:
    split_document = manifest.get("splits", {}).get(split, {})
    count = split_document.get("observed_samples")
    city_counts = split_document.get("city_counts")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or not isinstance(city_counts, Mapping)
        or set(city_counts) != set(cities)
    ):
        raise IntegrityError(f"population manifest {split} target partition is invalid")
    expected_role = "target_probe" if split == "validation" else "target_evaluation"
    grouped = {city: [] for city in cities}
    records = list(geo_index.iter_records(split))
    if len(records) != count:
        raise IntegrityError(f"safe {split} metadata population count drift")
    for row_index, record in enumerate(records):
        if (
            record.row_index != row_index
            or record.official_split != split
            or record.city_id not in grouped
            or record.city_role != "target"
            or record.sample_role != expected_role
        ):
            raise IntegrityError(f"safe {split} metadata violates target role contract")
        grouped[record.city_id].append(record)
    if {city: len(rows) for city, rows in grouped.items()} != dict(city_counts):
        raise IntegrityError(f"safe {split} metadata differs from sealed city counts")
    if sorted(record.row_index for rows in grouped.values() for record in rows) != list(
        range(count)
    ):
        raise IntegrityError(f"safe {split} metadata does not partition every row exactly once")
    return grouped


def _prepare_action_context(
    *,
    seal: Mapping[str, Any],
    gate: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    city_id: str,
    probe_samples: Sequence[PixelSample],
    evaluation_records: Sequence[Any],
    computation: ProbeComputation,
) -> dict[str, Any]:
    """Compute the gate action using probe evidence and safe geo rows only."""

    checkpoint_id = checkpoint["checkpoint_id"]
    probe_rows = [sample.metadata.row_index for sample in probe_samples]
    evaluation_rows = [record.row_index for record in evaluation_records]
    if probe_rows != sorted(set(probe_rows)) or evaluation_rows != sorted(set(evaluation_rows)):
        raise IntegrityError("target cell row indices must be unique and sorted")
    if any(sample.metadata.city_id != city_id for sample in probe_samples) or any(
        record.city_id != city_id for record in evaluation_records
    ):
        raise IntegrityError("target cell samples cross city boundaries")
    frozen_probe = _logits(
        computation.frozen_probe_logits, rows=len(probe_rows), field="frozen_probe_logits"
    )
    adapted_probe = _logits(
        computation.adapted_probe_logits, rows=len(probe_rows), field="adapted_probe_logits"
    )
    feature = extract_label_free_features(
        frozen_probe,
        adapted_probe,
        normalized_adapter_update_norm=computation.normalized_adapter_update_norm,
        batchnorm_source_statistic_divergence=computation.batchnorm_source_statistic_divergence,
    )
    partition_sha256 = stable_sha256(
        {
            "schema": "kbound_so2sat_target_city_partition_v1",
            "population_identity_sha256": seal["population_identity_sha256"],
            "city_id": city_id,
            "validation_row_indices": probe_rows,
            "testing_row_indices": evaluation_rows,
        }
    )
    trace_id = f"target_probe:{city_id}:{checkpoint_id}"
    binding = gate["study_binding"]
    trace_sha256 = trace_identity_sha256(
        role="target_probe",
        city_id=city_id,
        checkpoint_id=checkpoint_id,
        checkpoint_tensor_sha256=checkpoint["checkpoint_tensor_sha256"],
        checkpoint_file_sha256=checkpoint["checkpoint_file_sha256"],
        trace_id=trace_id,
        partition_sha256=partition_sha256,
        feature_sha256=feature["feature_sha256"],
        manifest_sha256=binding["manifest_sha256"],
        population_identity_sha256=binding["population_identity_sha256"],
        protocol_file_sha256=binding["protocol_file_sha256"],
        protocol_document_sha256=binding["protocol_document_sha256"],
    )
    action = apply_gate(
        gate,
        feature,
        city_id=city_id,
        checkpoint_id=checkpoint_id,
        checkpoint_tensor_sha256=checkpoint["checkpoint_tensor_sha256"],
        checkpoint_file_sha256=checkpoint["checkpoint_file_sha256"],
        trace_id=trace_id,
        trace_sha256=trace_sha256,
        partition_sha256=partition_sha256,
    )
    return {
        "checkpoint_id": checkpoint_id,
        "probe_rows": probe_rows,
        "evaluation_rows": evaluation_rows,
        "frozen_probe_logits": frozen_probe,
        "adapted_probe_logits": adapted_probe,
        "feature": feature,
        "partition_sha256": partition_sha256,
        "action": action,
    }


def _cell_document(
    *,
    seal: Mapping[str, Any],
    gate_authorization: Mapping[str, Any],
    gate: Mapping[str, Any],
    selected: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    city_id: str,
    evaluation_samples: Sequence[PixelSample],
    action_context: Mapping[str, Any],
    action_artifact: Mapping[str, Any],
    computation: EvaluationComputation,
    logit_archive_path: Path,
) -> dict[str, Any]:
    """Seal predictions after the immutable probe-derived action already exists."""

    checkpoint_id = checkpoint["checkpoint_id"]
    probe_rows = list(action_context["probe_rows"])
    evaluation_rows = [sample.metadata.row_index for sample in evaluation_samples]
    if evaluation_rows != action_context["evaluation_rows"] or any(
        sample.metadata.city_id != city_id for sample in evaluation_samples
    ):
        raise IntegrityError("testing pixels differ from the pre-action safe partition")
    frozen_probe = np.asarray(action_context["frozen_probe_logits"], dtype=np.float64)
    adapted_probe = np.asarray(action_context["adapted_probe_logits"], dtype=np.float64)
    feature = action_context["feature"]
    partition_sha256 = action_context["partition_sha256"]
    action = action_context["action"]
    frozen_evaluation = _logits(
        computation.frozen_evaluation_logits,
        rows=len(evaluation_rows),
        field="frozen_evaluation_logits",
    )
    adapted_evaluation = _logits(
        computation.adapted_evaluation_logits,
        rows=len(evaluation_rows),
        field="adapted_evaluation_logits",
    )
    frozen_predictions = frozen_evaluation.argmax(axis=1).astype(np.int64)
    adapted_predictions = adapted_evaluation.argmax(axis=1).astype(np.int64)
    logit_archive = _write_logit_archive(
        logit_archive_path,
        arrays={
            "frozen_probe_logits": frozen_probe,
            "adapted_probe_logits": adapted_probe,
            "frozen_evaluation_logits": frozen_evaluation,
            "adapted_evaluation_logits": adapted_evaluation,
        },
        execution_mode=seal["execution_mode"],
    )
    document = {
        "schema": TARGET_CELL_SCHEMA,
        "status": (
            "SEALED_BEFORE_TARGET_OUTCOME_ACCESS"
            if seal["execution_mode"] == PRODUCTION_MODE
            else "TEST_ONLY_CELL_WITH_SYNTHETIC_OR_INJECTED_DEPENDENCIES"
        ),
        "execution_mode": seal["execution_mode"],
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
        "selected_candidate_sha256": selected["selected_candidate_sha256"],
        "candidate_id": selected["candidate_id"],
        "manifest_sha256": seal["manifest_sha256"],
        "population_identity_sha256": seal["population_identity_sha256"],
        "checkpoint_id": checkpoint_id,
        "checkpoint_file_sha256": checkpoint["checkpoint_file_sha256"],
        "checkpoint_tensor_sha256": checkpoint["checkpoint_tensor_sha256"],
        "city_id": city_id,
        "partition_sha256": partition_sha256,
        "probe": {
            "official_split": "validation",
            "row_indices": probe_rows,
            "row_indices_sha256": stable_sha256(probe_rows),
            "sample_count": len(probe_rows),
            "feature_document": feature,
            "target_labels_opened": False,
            "target_labels_scored": False,
        },
        "action": action,
        "action_artifact": copy.deepcopy(dict(action_artifact)),
        "evaluation": {
            "official_split": "testing",
            "row_indices": evaluation_rows,
            "row_indices_sha256": stable_sha256(evaluation_rows),
            "sample_count": len(evaluation_rows),
            "frozen_logits_tensor_sha256": _tensor_sha256(frozen_evaluation),
            "adapted_logits_tensor_sha256": _tensor_sha256(adapted_evaluation),
            "frozen_prediction_class_ids": frozen_predictions.tolist(),
            "adapted_prediction_class_ids": adapted_predictions.tolist(),
            "frozen_predictions_sha256": _tensor_sha256(frozen_predictions),
            "adapted_predictions_sha256": _tensor_sha256(adapted_predictions),
            "target_labels_opened": False,
        },
        "logit_archive": logit_archive,
        "target_data_identities": copy.deepcopy(dict(seal["target_data_identities"])),
    }
    document["cell_sha256"] = stable_sha256(document)
    validate_target_cell(
        document,
        seal=seal,
        gate_authorization=gate_authorization,
        gate=gate,
        selected_candidate=selected,
    )
    return document


def _verify_loader_audit(loader: LabelFreeTargetLoader, manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(loader.access_log)
    expected_total = sum(int(manifest["splits"][split]["observed_samples"]) for split in TARGET_SPLITS)
    if len(rows) != expected_total:
        raise IntegrityError(
            f"label-free loader accessed {len(rows)} target rows, expected {expected_total}"
        )
    keys = [(row.get("split"), row.get("row_index")) for row in rows]
    expected = [
        (split, row_index)
        for split in TARGET_SPLITS
        for row_index in range(int(manifest["splits"][split]["observed_samples"]))
    ]
    if sorted(keys) != sorted(expected) or len(set(keys)) != len(keys):
        raise IntegrityError("label-free loader did not access each target pixel row exactly once")
    if any(
        row.get("dataset") != "sen2" or row.get("target_outcome_dataset_accessed") is not False
        for row in rows
    ):
        raise IntegrityError("live target loader audit indicates a firewall violation")
    return {
        "pixel_rows_read_exactly_once": True,
        "validation_pixel_rows": int(manifest["splits"]["validation"]["observed_samples"]),
        "testing_pixel_rows": int(manifest["splits"]["testing"]["observed_samples"]),
        "pixel_dataset": "sen2",
        "target_outcome_dataset_accessed": False,
        "container_bytes_verified_before_and_after": True,
    }


def _run_label_blind_target_core(
    *,
    execution_seal_path: str | Path,
    population_manifest_path: str | Path,
    source_postrun_acceptance_path: str | Path,
    selected_candidate_path: str | Path,
    selected_gate_fit_bundle_path: str | Path,
    selected_gate_cal_bundle_path: str | Path,
    precalibration_seal_path: str | Path,
    gate_path: str | Path,
    gate_authorization_path: str | Path,
    target_boundary_amendment_path: str | Path,
    checkpoint_collection_path: str | Path,
    checkpoint_dir: str | Path,
    geo_index: VerifiedGeoIndex,
    target_loader: LabelFreeTargetLoader,
    cell_executor: CellExecutor,
    output_dir: str | Path,
    population_manifest_validator: Callable[[Mapping[str, Any]], None],
    expected_execution_mode: str,
) -> Path:
    """Run and seal all 50 label-blind target cells.

    Every input JSON artifact must have a valid byte receipt.  No target pixel
    is read until all identities, checkpoints, target partitions, and the empty
    output destination have passed validation.
    """

    if expected_execution_mode == PRODUCTION_MODE:
        from .target_inference import TorchTargetCellExecutor

        if (
            population_manifest_validator is not validate_population_manifest
            or type(geo_index) is not VerifiedGeoIndex
            or type(target_loader) is not LabelFreeTargetLoader
            or type(cell_executor) is not TorchTargetCellExecutor
            or getattr(target_loader, "_geo_index", None) is not geo_index
            or geo_index.uses_canonical_h5_factory is not True
            or target_loader.uses_canonical_h5_factory is not True
        ):
            raise IntegrityError(
                "production target core rejects injected validators, geo indexes, "
                "loaders, or executors"
            )
        require_production_target_action_unit_alignment()
    elif expected_execution_mode != TEST_ONLY_MODE:
        raise IntegrityError("target runner execution mode is invalid")

    manifest, manifest_receipt = _receipt_document(population_manifest_path)
    population_manifest_validator(manifest)
    source_acceptance, _, source_acceptance_binding = (
        load_source_postrun_acceptance_pair(
            source_postrun_acceptance_path,
            strict_document=expected_execution_mode == PRODUCTION_MODE,
        )
    )
    gate = load_gate_with_receipt(gate_path)
    gate_receipt = verify_artifact_receipt(gate_path)
    validate_gate_document(gate)
    binding = gate["study_binding"]
    if (
        manifest.get("manifest_sha256") != binding["manifest_sha256"]
        or manifest.get("population_identity_sha256") != binding["population_identity_sha256"]
        or _artifact_binding(manifest_receipt)["artifact_sha256"]
        != binding["manifest_artifact_sha256"]
        or _artifact_binding(manifest_receipt)["canonical_document_sha256"]
        != binding["manifest_canonical_document_sha256"]
    ):
        raise IntegrityError("gate and receipt-verified population manifest differ")
    selection, selected_receipt = _receipt_document(selected_candidate_path)
    validate_selected_candidate(selection, study_binding=binding)
    if selection["source_postrun_acceptance"] != source_acceptance_binding:
        raise IntegrityError("target runner selection binds another source acceptance")
    selected = _selected_candidate_view(selection)
    fit_bundle, fit_bundle_receipt = _receipt_document(selected_gate_fit_bundle_path)
    gate_authorization, authorized_selection, authorized_gate = (
        load_gate_authorization_with_receipt(
            gate_authorization_path,
            selection_path=selected_candidate_path,
            gate_path=gate_path,
            population_manifest_path=population_manifest_path,
            fit_bundle_path=selected_gate_fit_bundle_path,
            calibration_bundle_path=selected_gate_cal_bundle_path,
        )
    )
    gate_authorization_receipt = verify_artifact_receipt(gate_authorization_path)
    if authorized_selection != selection or authorized_gate != gate:
        raise IntegrityError("gate authorization loader returned a different selection or gate")
    amendment, amendment_receipt = load_target_boundary_amendment(
        target_boundary_amendment_path
    )
    collection, collection_receipt = _receipt_document(checkpoint_collection_path)
    checkpoints = validate_checkpoint_collection(
        collection,
        collection_receipt=collection_receipt,
        collection_path=checkpoint_collection_path,
        checkpoint_dir=checkpoint_dir,
    )
    if (
        gate_authorization["checkpoint_collection_canonical_sha256"]
        != stable_sha256(collection)
        or gate_authorization["normalizer_sha256"] != collection["normalizer_sha256"]
    ):
        raise IntegrityError("gate authorization differs from the verified checkpoint collection")
    precalibration_seal, precalibration_seal_receipt = (
        load_precalibration_seal_with_receipt(
            precalibration_seal_path,
            study_binding=binding,
            selection=selection,
            fit_bundle=fit_bundle,
            target_boundary_amendment=amendment,
            checkpoint_collection=collection,
        )
    )
    seal, seal_receipt = _receipt_document(execution_seal_path)
    validate_execution_seal(
        seal,
        study_binding=binding,
        selected_candidate=selection,
        gate=gate,
        gate_authorization=gate_authorization,
        target_boundary_amendment=amendment,
        checkpoint_collection=collection,
        precalibration_seal=precalibration_seal,
    )
    if seal["execution_mode"] != expected_execution_mode:
        raise IntegrityError(
            f"target runner expected {expected_execution_mode}, found {seal['execution_mode']}"
        )
    if seal["selected_candidate_artifact"] != _artifact_binding(selected_receipt):
        raise IntegrityError("execution seal does not bind the selected-candidate artifact bytes")
    if seal["checkpoint_collection_artifact"] != _artifact_binding(collection_receipt):
        raise IntegrityError("execution seal does not bind the checkpoint collection artifact bytes")
    if (
        seal["gate_artifact"] != _artifact_binding(gate_receipt)
        or seal["gate_authorization_artifact"]
        != _artifact_binding(gate_authorization_receipt)
        or seal["target_boundary_amendment_artifact"]
        != _artifact_binding(amendment_receipt)
        or seal["precalibration_seal_artifact"]
        != _artifact_binding(precalibration_seal_receipt)
    ):
        raise IntegrityError(
            "execution seal does not bind the gate/amendment/precalibration artifact bytes"
        )
    if (
        precalibration_seal["selected_gate_fit_bundle_artifact"]
        != _artifact_binding(fit_bundle_receipt)
    ):
        raise IntegrityError("precalibration seal does not bind selected gate-fit bytes")
    if (
        seal["source_postrun_acceptance"] != source_acceptance_binding
        or seal["source_postrun_training_container"]
        != source_acceptance["postrun_source_container"]
        or seal["source_hdf5_runtime_disclosure"]
        != source_acceptance["source_hdf5_runtime_disclosure"]
        or seal["source_checkpoint_selection_disclosure"]
        != source_acceptance["source_checkpoint_selection_disclosure"]
        or seal["source_initialization_clarification"]
        != source_acceptance["source_initialization_clarification"]
    ):
        raise IntegrityError("target runner source acceptance provenance drift")
    for checkpoint_id in CHECKPOINT_IDS:
        for field in ("checkpoint_file_sha256", "checkpoint_tensor_sha256"):
            if checkpoints[checkpoint_id][field] != seal["checkpoint_identities"][checkpoint_id][field]:
                raise IntegrityError("verified source checkpoint differs from execution seal")
            if checkpoints[checkpoint_id][field] != gate["development_provenance"][
                f"{field}_by_id"
            ][checkpoint_id]:
                raise IntegrityError("verified source checkpoint differs from calibrated gate")
    if geo_index.population_identity_sha256 != seal["population_identity_sha256"]:
        raise IntegrityError("target geographic index differs from execution seal")
    if target_loader.access_log:
        raise IntegrityError("live runner requires a fresh target loader with an empty access log")
    if (
        getattr(cell_executor, "candidate_id", None) != selected["candidate_id"]
        or getattr(cell_executor, "normalizer_sha256", None)
        != gate_authorization["normalizer_sha256"]
        or getattr(cell_executor, "code_identity_sha256", None)
        != seal["code_identity_sha256"]
        or getattr(cell_executor, "environment_identity_sha256", None)
        != seal["environment_identity_sha256"]
    ):
        raise IntegrityError(
            "target cell executor candidate, normalizer, code, or environment identity "
            "differs from the seals"
        )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    master_path = destination / "so2sat_target_bundle.json"
    expected_cell_paths = {
        (city, checkpoint): destination / f"target_{city}_checkpoint{checkpoint}.json"
        for city in binding["target_cities"]
        for checkpoint in CHECKPOINT_IDS
    }
    expected_logit_paths = {
        (city, checkpoint): destination
        / f"target_{city}_checkpoint{checkpoint}.logits.npz"
        for city in binding["target_cities"]
        for checkpoint in CHECKPOINT_IDS
    }
    expected_action_paths = {
        (city, checkpoint): destination
        / f"target_{city}_checkpoint{checkpoint}.action.json"
        for city in binding["target_cities"]
        for checkpoint in CHECKPOINT_IDS
    }
    reserved = [master_path, master_path.with_name(master_path.name + ".receipt.json")]
    for path in expected_cell_paths.values():
        reserved.extend([path, path.with_name(path.name + ".receipt.json")])
    for path in expected_action_paths.values():
        reserved.extend([path, path.with_name(path.name + ".receipt.json")])
    for path in expected_logit_paths.values():
        manifest_path = path.with_name(path.name + ".manifest.json")
        reserved.extend(
            [
                path,
                manifest_path,
                manifest_path.with_name(manifest_path.name + ".receipt.json"),
            ]
        )
    if any(path.exists() for path in reserved):
        raise IntegrityError("target output contains a prior or partial immutable bundle")

    container_verification = target_loader.verify_containers()
    observed_containers = {
        row["split"]: {
            "basename": row["basename"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in container_verification.get("containers", [])
    }
    if observed_containers != seal["target_data_identities"]:
        raise IntegrityError("live target containers differ from the execution seal")

    cities = list(binding["target_cities"])
    validation_by_city = _partition(geo_index, manifest, split="validation", cities=cities)
    testing_by_city = _partition(geo_index, manifest, split="testing", cities=cities)
    cell_rows: list[dict[str, Any]] = []
    for city_id in cities:
        probe_samples = target_loader.read_verified_many(
            "validation", validation_by_city[city_id]
        )
        _freeze_sample_pixels(probe_samples)
        prepared: dict[str, tuple[ProbeComputation, dict[str, Any], dict[str, Any]]] = {}
        for checkpoint_id in CHECKPOINT_IDS:
            probe_computation = cell_executor.prepare_probe(
                checkpoints[checkpoint_id],
                selected["candidate_spec"],
                probe_samples,
            )
            if not isinstance(probe_computation, ProbeComputation):
                raise IntegrityError("cell executor must return ProbeComputation")
            action_context = _prepare_action_context(
                seal=seal,
                gate=gate,
                checkpoint=checkpoints[checkpoint_id],
                city_id=city_id,
                probe_samples=probe_samples,
                evaluation_records=testing_by_city[city_id],
                computation=probe_computation,
            )
            action_path = expected_action_paths[(city_id, checkpoint_id)]
            action_receipt = write_immutable_json_with_receipt(
                action_path, action_context["action"]
            )
            action_artifact = {
                "action_basename": action_path.name,
                "artifact_sha256": action_receipt["artifact_sha256"],
                "canonical_document_sha256": action_receipt[
                    "canonical_document_sha256"
                ],
                "sealed_before_evaluation_pixel_access": True,
            }
            prepared[checkpoint_id] = (
                probe_computation,
                action_context,
                action_artifact,
            )
        # Testing pixels for this city are unreachable until all five action
        # documents and receipts for the city have been durably created.
        if any(
            not expected_action_paths[(city_id, checkpoint_id)].is_file()
            or not expected_action_paths[(city_id, checkpoint_id)]
            .with_name(
                expected_action_paths[(city_id, checkpoint_id)].name
                + ".receipt.json"
            )
            .is_file()
            for checkpoint_id in CHECKPOINT_IDS
        ):
            raise IntegrityError("city actions were not sealed before testing pixel access")
        evaluation_samples = target_loader.read_verified_many(
            "testing", testing_by_city[city_id]
        )
        _freeze_sample_pixels(evaluation_samples)
        for checkpoint_id in CHECKPOINT_IDS:
            probe_computation, action_context, action_artifact = prepared[checkpoint_id]
            evaluation_computation = cell_executor.evaluate_after_action(
                probe_computation, evaluation_samples
            )
            if not isinstance(evaluation_computation, EvaluationComputation):
                raise IntegrityError("cell executor must return EvaluationComputation")
            cell = _cell_document(
                seal=seal,
                gate_authorization=gate_authorization,
                gate=gate,
                selected=selected,
                checkpoint=checkpoints[checkpoint_id],
                city_id=city_id,
                evaluation_samples=evaluation_samples,
                action_context=action_context,
                action_artifact=action_artifact,
                computation=evaluation_computation,
                logit_archive_path=expected_logit_paths[(city_id, checkpoint_id)],
            )
            path = expected_cell_paths[(city_id, checkpoint_id)]
            receipt = write_immutable_json_with_receipt(path, cell)
            cell_rows.append(
                {
                    "city_id": city_id,
                    "checkpoint_id": checkpoint_id,
                    "cell_basename": path.name,
                    "cell_sha256": cell["cell_sha256"],
                    "artifact_sha256": receipt["artifact_sha256"],
                    "canonical_document_sha256": receipt["canonical_document_sha256"],
                    "action_sha256": cell["action"]["action_sha256"],
                    "action_basename": action_artifact["action_basename"],
                    "action_artifact_sha256": action_artifact["artifact_sha256"],
                    "action_canonical_document_sha256": action_artifact[
                        "canonical_document_sha256"
                    ],
                    "logit_archive_sha256": cell["logit_archive"]["archive_sha256"],
                    "logit_manifest_sha256": cell["logit_archive"]["manifest_sha256"],
                }
            )

    postrun_container_verification = target_loader.verify_containers()
    postrun_containers = {
        row["split"]: {
            "basename": row["basename"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in postrun_container_verification.get("containers", [])
    }
    if postrun_containers != observed_containers:
        raise IntegrityError("target containers changed during label-blind inference")
    access_audit = _verify_loader_audit(target_loader, manifest)
    cell_rows.sort(key=lambda row: (row["city_id"], row["checkpoint_id"]))
    master = {
        "schema": TARGET_BUNDLE_SCHEMA,
        "status": (
            "COMPLETE_50_CELLS_SEALED_BEFORE_TARGET_OUTCOME_ACCESS"
            if seal["execution_mode"] == PRODUCTION_MODE
            else "TEST_ONLY_COMPLETE_50_CELLS_WITH_SYNTHETIC_OR_INJECTED_DEPENDENCIES"
        ),
        "execution_mode": seal["execution_mode"],
        "execution_seal_artifact": _artifact_binding(seal_receipt),
        "execution_seal_sha256": seal["execution_seal_sha256"],
        "gate_authorization_artifact": _artifact_binding(gate_authorization_receipt),
        "gate_authorization_sha256": gate_authorization["authorization_sha256"],
        "target_boundary_amendment_artifact": _artifact_binding(amendment_receipt),
        "target_boundary_amendment_sha256": seal[
            "target_boundary_amendment_sha256"
        ],
        "precalibration_seal_artifact": _artifact_binding(
            precalibration_seal_receipt
        ),
        "precalibration_seal_sha256": precalibration_seal[
            "precalibration_seal_sha256"
        ],
        "source_postrun_acceptance": copy.deepcopy(
            dict(seal["source_postrun_acceptance"])
        ),
        "source_postrun_acceptance_artifact_sha256": seal[
            "source_postrun_acceptance_artifact_sha256"
        ],
        "source_postrun_training_container": copy.deepcopy(
            dict(seal["source_postrun_training_container"])
        ),
        "source_hdf5_runtime_disclosure": copy.deepcopy(
            dict(seal["source_hdf5_runtime_disclosure"])
        ),
        "source_checkpoint_selection_disclosure": copy.deepcopy(
            dict(seal["source_checkpoint_selection_disclosure"])
        ),
        "source_initialization_clarification": copy.deepcopy(
            dict(seal["source_initialization_clarification"])
        ),
        "population_manifest_artifact": _artifact_binding(manifest_receipt),
        "manifest_sha256": seal["manifest_sha256"],
        "population_identity_sha256": seal["population_identity_sha256"],
        "selected_candidate_artifact": _artifact_binding(selected_receipt),
        "selected_candidate_sha256": selected["selected_candidate_sha256"],
        "candidate_id": selected["candidate_id"],
        "gate_sha256": gate["gate_sha256"],
        "checkpoint_collection_artifact": _artifact_binding(collection_receipt),
        "target_data_identities": copy.deepcopy(dict(seal["target_data_identities"])),
        "target_cities": cities,
        "checkpoint_ids": list(CHECKPOINT_IDS),
        "cell_count": len(cell_rows),
        "cells": cell_rows,
        "access_audit": access_audit,
        "probe_labels_opened": False,
        "probe_labels_scored": False,
        "evaluation_labels_opened": False,
        "complete_before_scoring": True,
    }
    master["bundle_sha256"] = stable_sha256(master)
    validate_complete_target_bundle_document(master)
    write_immutable_json_with_receipt(master_path, master)
    return master_path


def run_label_blind_target(
    *,
    execution_seal_path: str | Path,
    population_manifest_path: str | Path,
    source_postrun_acceptance_path: str | Path,
    selected_candidate_path: str | Path,
    selected_gate_fit_bundle_path: str | Path,
    selected_gate_cal_bundle_path: str | Path,
    precalibration_seal_path: str | Path,
    gate_path: str | Path,
    gate_authorization_path: str | Path,
    target_boundary_amendment_path: str | Path,
    checkpoint_collection_path: str | Path,
    checkpoint_dir: str | Path,
    geo_index: VerifiedGeoIndex,
    target_loader: LabelFreeTargetLoader,
    cell_executor: CellExecutor,
    output_dir: str | Path,
    _production_authority: object | None = None,
) -> Path:
    """Production-only live runner; injected substitutes are rejected."""

    from .target_inference import TorchTargetCellExecutor

    if (
        type(geo_index) is not VerifiedGeoIndex
        or type(target_loader) is not LabelFreeTargetLoader
        or type(cell_executor) is not TorchTargetCellExecutor
        or getattr(target_loader, "_geo_index", None) is not geo_index
        or geo_index.uses_canonical_h5_factory is not True
        or target_loader.uses_canonical_h5_factory is not True
        or _production_authority is not _PRODUCTION_RUN_AUTHORITY
    ):
        raise IntegrityError(
            "production target runner requires the exact verified geo index, label "
            "firewall loader, and Torch target executor implementations"
        )
    return _run_label_blind_target_core(
        execution_seal_path=execution_seal_path,
        population_manifest_path=population_manifest_path,
        source_postrun_acceptance_path=source_postrun_acceptance_path,
        selected_candidate_path=selected_candidate_path,
        selected_gate_fit_bundle_path=selected_gate_fit_bundle_path,
        selected_gate_cal_bundle_path=selected_gate_cal_bundle_path,
        precalibration_seal_path=precalibration_seal_path,
        gate_path=gate_path,
        gate_authorization_path=gate_authorization_path,
        target_boundary_amendment_path=target_boundary_amendment_path,
        checkpoint_collection_path=checkpoint_collection_path,
        checkpoint_dir=checkpoint_dir,
        geo_index=geo_index,
        target_loader=target_loader,
        cell_executor=cell_executor,
        output_dir=output_dir,
        population_manifest_validator=validate_population_manifest,
        expected_execution_mode=PRODUCTION_MODE,
    )


def _run_label_blind_target_for_test(
    *,
    population_manifest_validator: Callable[[Mapping[str, Any]], None],
    **kwargs: Any,
) -> Path:
    """Test-only injected path; it accepts and emits only TEST_ONLY artifacts."""

    return _run_label_blind_target_core(
        **kwargs,
        population_manifest_validator=population_manifest_validator,
        expected_execution_mode=TEST_ONLY_MODE,
    )


def run_production_target(
    *,
    execution_seal_path: str | Path,
    population_manifest_path: str | Path,
    source_postrun_acceptance_path: str | Path,
    selected_candidate_path: str | Path,
    selected_gate_fit_bundle_path: str | Path,
    selected_gate_cal_bundle_path: str | Path,
    precalibration_seal_path: str | Path,
    gate_path: str | Path,
    gate_authorization_path: str | Path,
    target_boundary_amendment_path: str | Path,
    checkpoint_collection_path: str | Path,
    checkpoint_dir: str | Path,
    normalizer_path: str | Path,
    geo_paths: Mapping[str, str | Path],
    target_data_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    device_name: str,
) -> Path:
    """Construct the production label firewall/executor and run the live side.

    The only co-located target-container interface created here is
    :class:`LabelFreeTargetLoader`, whose dataset name is fixed to ``sen2``.
    The concrete executor receives ``PixelSample`` objects and never receives
    an HDF5 path, handle, dataset name, or target outcome.
    """

    if device_name not in {"cpu", "mps"}:
        raise IntegrityError("production target device must be exactly 'cpu' or 'mps'")
    # Imports are deliberately local: target_inference implements the concrete
    # executor and imports CellComputation from this module.
    import torch

    from .target_inference import TorchTargetCellExecutor

    if device_name == "mps" and not torch.backends.mps.is_available():
        raise IntegrityError("MPS target execution was requested but is unavailable")
    manifest, _ = _receipt_document(population_manifest_path)
    validate_population_manifest(manifest)
    seal, _ = _receipt_document(execution_seal_path)
    identities = _target_identities(seal.get("target_data_identities"))
    selection, _ = _receipt_document(selected_candidate_path)
    candidate_id = selection.get("selected_candidate_id")
    if candidate_id not in CANDIDATE_IDS:
        raise IntegrityError("production target execution requires a selected adapter")
    geo_index = VerifiedGeoIndex(manifest, geo_paths)
    target_loader = LabelFreeTargetLoader(
        geo_index,
        target_data_paths,
        identities,
        modality="sen2_10_band",
    )
    executor = TorchTargetCellExecutor(
        candidate_id=str(candidate_id),
        normalizer_path=normalizer_path,
        device=torch.device(device_name),
    )
    return run_label_blind_target(
        execution_seal_path=execution_seal_path,
        population_manifest_path=population_manifest_path,
        source_postrun_acceptance_path=source_postrun_acceptance_path,
        selected_candidate_path=selected_candidate_path,
        selected_gate_fit_bundle_path=selected_gate_fit_bundle_path,
        selected_gate_cal_bundle_path=selected_gate_cal_bundle_path,
        precalibration_seal_path=precalibration_seal_path,
        gate_path=gate_path,
        gate_authorization_path=gate_authorization_path,
        target_boundary_amendment_path=target_boundary_amendment_path,
        checkpoint_collection_path=checkpoint_collection_path,
        checkpoint_dir=checkpoint_dir,
        geo_index=geo_index,
        target_loader=target_loader,
        cell_executor=executor,
        output_dir=output_dir,
        _production_authority=_PRODUCTION_RUN_AUTHORITY,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-seal", required=True)
    parser.add_argument("--population-manifest", required=True)
    parser.add_argument("--source-postrun-acceptance", required=True)
    parser.add_argument("--selected-candidate", required=True)
    parser.add_argument("--selected-gate-fit-bundle", required=True)
    parser.add_argument("--selected-gate-cal-bundle", required=True)
    parser.add_argument("--precalibration-seal", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--gate-authorization", required=True)
    parser.add_argument("--target-boundary-amendment", required=True)
    parser.add_argument("--checkpoint-collection", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--training-geo", required=True)
    parser.add_argument("--validation-geo", required=True)
    parser.add_argument("--testing-geo", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--testing-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = run_production_target(
        execution_seal_path=arguments.execution_seal,
        population_manifest_path=arguments.population_manifest,
        source_postrun_acceptance_path=arguments.source_postrun_acceptance,
        selected_candidate_path=arguments.selected_candidate,
        selected_gate_fit_bundle_path=arguments.selected_gate_fit_bundle,
        selected_gate_cal_bundle_path=arguments.selected_gate_cal_bundle,
        precalibration_seal_path=arguments.precalibration_seal,
        gate_path=arguments.gate,
        gate_authorization_path=arguments.gate_authorization,
        target_boundary_amendment_path=arguments.target_boundary_amendment,
        checkpoint_collection_path=arguments.checkpoint_collection,
        checkpoint_dir=arguments.checkpoint_dir,
        normalizer_path=arguments.normalizer,
        geo_paths={
            "training": arguments.training_geo,
            "validation": arguments.validation_geo,
            "testing": arguments.testing_geo,
        },
        target_data_paths={
            "validation": arguments.validation_data,
            "testing": arguments.testing_data,
        },
        output_dir=arguments.output_dir,
        device_name=arguments.device,
    )
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI integration
    raise SystemExit(main())
