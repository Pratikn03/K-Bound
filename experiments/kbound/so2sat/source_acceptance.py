#!/usr/bin/env python3
"""Create the source-only post-run acceptance artifact for So2Sat.

This command is intentionally representationally unable to receive validation
or testing paths.  It verifies the label-free population manifest, the five
immutable source checkpoint/receipt pairs, their collection, and the
source-only normalizer before hashing the raw bytes of ``training.h5`` exactly
once.  It never opens an HDF5 dataset.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .integrity import (
    ARTIFACT_RECEIPT_SCHEMA_V2,
    IntegrityError,
    file_sha256,
    require_sha256,
    stable_sha256,
    strict_json_load,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from .metadata_manifest import SCHEMA as POPULATION_MANIFEST_SCHEMA
from .metadata_manifest import validate_population_manifest
from .model import ARCHITECTURE_ID, CANONICAL_MODEL_SEEDS, NUM_CLASSES
from .source_data import (
    SOURCE_MONITOR_ROLE,
    BandNormalizer,
    load_sealed_band_normalizer,
)
from .source_preflight import SCHEMA as SOURCE_PREFLIGHT_SCHEMA
from .source_preflight import require_source_training_path
from .train_source import (
    COLLECTION_SCHEMA,
    TRAINING_RECEIPT_SCHEMA,
    verify_complete_source_result,
)

SOURCE_PREFLIGHT_RECEIPT_SCHEMA = "kbound_so2sat_source_data_preflight_receipt_v1"
SOURCE_POSTRUN_ACCEPTANCE_SCHEMA = "kbound_so2sat_source_postrun_acceptance_v1"
SOURCE_POSTRUN_ACCEPTANCE_STATUS = "SOURCE_POSTRUN_ACCEPTED"
SOURCE_POSTRUN_ACCEPTANCE_BASENAME = "so2sat_source_postrun_acceptance.json"
TARGET_SEAL_BINDING_FIELD = "source_postrun_acceptance_artifact_sha256"
TARGET_SEAL_BINDING_VALUE_SOURCE = "portable_receipt.artifact_sha256"
SOURCE_NORMALIZER_BASENAME = "so2sat_sen2_source_normalizer.json"
SOURCE_COLLECTION_BASENAME = "so2sat_source_checkpoint_collection.json"
EXPECTED_SUPPORTED_CLASS_COUNT = 15
EXPECTED_ABSENT_SOURCE_MONITOR_CLASS_IDS = (0, 6)
ACCEPTANCE_CODE_BASENAMES = (
    "integrity.py",
    "protocol.py",
    "metadata_manifest.py",
    "label_firewall.py",
    "model.py",
    "source_data.py",
    "source_preflight.py",
    "train_source.py",
    "source_acceptance.py",
    "prospective_protocol_v1.json",
    "prospective_protocol_v1.json.receipt.json",
)


def _require_exact_mapping(
    value: Any,
    fields: set[str],
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise IntegrityError(f"{field} has unknown or missing fields")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise IntegrityError(f"{field} must be a positive integer")
    return value


def _require_portable_basename(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise IntegrityError(f"{field} must be one portable basename")
    return value


def _acceptance_code_identity() -> dict[str, Any]:
    directory = Path(__file__).resolve().parent
    files = {name: file_sha256(directory / name) for name in ACCEPTANCE_CODE_BASENAMES}
    return {
        "files_sha256": files,
        "code_identity_sha256": stable_sha256(files),
    }


def _acceptance_environment_identity() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in ("h5py", "numpy", "torch", "torchvision"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    if versions["h5py"] == "NOT_INSTALLED":
        raise IntegrityError(
            "source post-run acceptance requires h5py so its exact version can be sealed"
        )
    document = {
        "schema": "kbound_so2sat_source_postrun_acceptance_environment_v1",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable_basename": Path(sys.executable).name,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "package_versions": versions,
    }
    document["environment_identity_sha256"] = stable_sha256(document)
    return document


def _portable_artifact_identity(path: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_basename": path.name,
        "artifact_bytes": receipt["artifact_bytes"],
        "artifact_sha256": require_sha256(
            receipt.get("artifact_sha256"), field=f"{path.name}.artifact_sha256"
        ),
        "canonical_document_sha256": require_sha256(
            receipt.get("canonical_document_sha256"),
            field=f"{path.name}.canonical_document_sha256",
        ),
    }


def _validate_portable_artifact_identity(
    value: Any,
    *,
    field: str,
) -> dict[str, Any]:
    identity = _require_exact_mapping(
        value,
        {
            "artifact_basename",
            "artifact_bytes",
            "artifact_sha256",
            "canonical_document_sha256",
        },
        field=field,
    )
    return {
        "artifact_basename": _require_portable_basename(
            identity.get("artifact_basename"), field=f"{field}.artifact_basename"
        ),
        "artifact_bytes": _require_positive_int(
            identity.get("artifact_bytes"), field=f"{field}.artifact_bytes"
        ),
        "artifact_sha256": require_sha256(
            identity.get("artifact_sha256"), field=f"{field}.artifact_sha256"
        ),
        "canonical_document_sha256": require_sha256(
            identity.get("canonical_document_sha256"),
            field=f"{field}.canonical_document_sha256",
        ),
    }


def _load_population_manifest(
    path: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    artifact = Path(path).expanduser().resolve()
    receipt = verify_artifact_receipt(artifact)
    document = strict_json_load(artifact)
    if not isinstance(document, dict):
        raise IntegrityError("So2Sat population manifest must be a JSON mapping")
    validate_population_manifest(document)
    return artifact, document, receipt


def _load_source_preflight(path: str | os.PathLike[str]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    artifact = Path(path).expanduser().resolve()
    receipt = verify_artifact_receipt(
        artifact,
        receipt_schema=SOURCE_PREFLIGHT_RECEIPT_SCHEMA,
    )
    document = strict_json_load(artifact)
    if not isinstance(document, dict) or document.get("schema") != SOURCE_PREFLIGHT_SCHEMA:
        raise IntegrityError("unknown So2Sat source preflight schema")
    quality = document.get("quality_gate")
    if (
        document.get("status") not in {
            "SOURCE_DATA_PREFLIGHT_PASSED",
            "SOURCE_DATA_PREFLIGHT_PASSED_WITH_WARNINGS",
        }
        or not isinstance(quality, Mapping)
        or quality.get("ready_for_source_training") is not True
    ):
        raise IntegrityError("source preflight did not authorize source training")
    scope = document.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("official_image_split_opened") != "training"
        or scope.get("target_image_containers_opened") is not False
        or scope.get("target_outcome_arrays_opened") is not False
        or scope.get("target_outcome_arrays_counted") is not False
        or scope.get("target_outcome_arrays_hashed") is not False
    ):
        raise IntegrityError("source preflight scope crosses or obscures the target boundary")
    return artifact, document, receipt


def _source_monitor_support(preflight: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    datasets = preflight.get("datasets")
    labels = datasets.get("label") if isinstance(datasets, Mapping) else None
    role_counts = labels.get("class_counts_by_sample_role") if isinstance(labels, Mapping) else None
    monitor_counts = role_counts.get(SOURCE_MONITOR_ROLE) if isinstance(role_counts, Mapping) else None
    missing_by_role = (
        labels.get("missing_classes_by_sample_role") if isinstance(labels, Mapping) else None
    )
    missing = missing_by_role.get(SOURCE_MONITOR_ROLE) if isinstance(missing_by_role, Mapping) else None
    expected_keys = {str(class_id) for class_id in range(NUM_CLASSES)}
    if (
        not isinstance(monitor_counts, Mapping)
        or set(monitor_counts) != expected_keys
        or any(
            isinstance(monitor_counts[str(class_id)], bool)
            or not isinstance(monitor_counts[str(class_id)], int)
            or monitor_counts[str(class_id)] < 0
            for class_id in range(NUM_CLASSES)
        )
        or not isinstance(missing, list)
        or any(isinstance(class_id, bool) or not isinstance(class_id, int) for class_id in missing)
    ):
        raise IntegrityError("source preflight has an invalid source_monitor class profile")
    support = [int(monitor_counts[str(class_id)]) for class_id in range(NUM_CLASSES)]
    replay_missing = [class_id for class_id, count in enumerate(support) if count == 0]
    if missing != replay_missing:
        raise IntegrityError("source preflight source_monitor missing-class profile does not replay")
    if (
        sum(count > 0 for count in support) != EXPECTED_SUPPORTED_CLASS_COUNT
        or tuple(replay_missing) != EXPECTED_ABSENT_SOURCE_MONITOR_CLASS_IDS
    ):
        raise IntegrityError(
            "source acceptance requires the disclosed 15-class source_monitor support with absent IDs [0,6]"
        )
    return support, replay_missing


def _expected_source_container_identity(preflight: Mapping[str, Any]) -> dict[str, Any]:
    raw = preflight.get("training_container_identity")
    datasets = preflight.get("datasets")
    sen2 = datasets.get("sen2") if isinstance(datasets, Mapping) else None
    labels = datasets.get("label") if isinstance(datasets, Mapping) else None
    if not isinstance(raw, Mapping) or not isinstance(sen2, Mapping) or not isinstance(labels, Mapping):
        raise IntegrityError("source preflight lacks the source-container identity inputs")
    basename = raw.get("basename")
    byte_count = raw.get("bytes")
    if (
        basename != "training.h5"
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 1
    ):
        raise IntegrityError("source preflight training-container basename/bytes are invalid")
    return {
        "schema": "kbound_so2sat_source_container_identity_v1",
        "basename": basename,
        "bytes": byte_count,
        "file_sha256": require_sha256(raw.get("sha256"), field="training_container.sha256"),
        "sen2_shape": sen2.get("shape"),
        "sen2_dtype": sen2.get("dtype"),
        "label_shape": labels.get("shape"),
        "label_dtype": labels.get("dtype"),
        "accessible_official_split": "training",
        "target_split_paths": [],
    }


def _verify_checkpoint_collection(
    checkpoint_dir: Path,
    expected_source_identity: Mapping[str, Any],
    normalizer: BandNormalizer,
    monitor_support: list[int],
) -> tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    collection_path = checkpoint_dir / SOURCE_COLLECTION_BASENAME
    collection_receipt = verify_artifact_receipt(collection_path)
    collection = strict_json_load(collection_path)
    expected_collection_fields = {
        "schema",
        "status",
        "model_seeds",
        "all_checkpoint_tensor_hashes_distinct",
        "all_initial_tensor_hashes_distinct",
        "config_sha256",
        "data_identity_sha256",
        "normalizer_sha256",
        "source_rows_sha256",
        "checkpoints",
        "target_data_inputs",
    }
    if not isinstance(collection, dict) or set(collection) != expected_collection_fields:
        raise IntegrityError("source checkpoint collection has unknown or missing fields")
    if (
        collection.get("schema") != COLLECTION_SCHEMA
        or collection.get("status") != "FIVE_INDEPENDENT_SOURCE_CHECKPOINTS_VERIFIED"
        or collection.get("model_seeds") != list(CANONICAL_MODEL_SEEDS)
        or collection.get("all_checkpoint_tensor_hashes_distinct") is not True
        or collection.get("all_initial_tensor_hashes_distinct") is not True
        or collection.get("target_data_inputs") != []
    ):
        raise IntegrityError("source checkpoint collection contract drift")
    for field in (
        "config_sha256",
        "data_identity_sha256",
        "normalizer_sha256",
        "source_rows_sha256",
    ):
        require_sha256(collection.get(field), field=f"checkpoint_collection.{field}")
    rows = collection.get("checkpoints")
    if not isinstance(rows, list) or len(rows) != len(CANONICAL_MODEL_SEEDS):
        raise IntegrityError("source checkpoint collection does not contain five rows")

    verified_rows: list[dict[str, Any]] = []
    observed_collection_rows: list[dict[str, Any]] = []
    initial_hashes: list[str] = []
    tensor_hashes: list[str] = []
    for model_seed in CANONICAL_MODEL_SEEDS:
        result = verify_complete_source_result(
            checkpoint_dir,
            model_seed,
            expected_config_sha256=collection["config_sha256"],
            expected_data_identity_sha256=collection["data_identity_sha256"],
        )
        receipt_path = checkpoint_dir / f"so2sat_resnet18_seed{model_seed}.training.json"
        receipt_byte_receipt = verify_artifact_receipt(receipt_path)
        receipt = strict_json_load(receipt_path)
        if not isinstance(receipt, Mapping) or receipt.get("schema") != TRAINING_RECEIPT_SCHEMA:
            raise IntegrityError("unknown source training receipt during post-run acceptance")
        data = receipt.get("data")
        metrics = receipt.get("best_source_monitor")
        scientific = receipt.get("scientific_identity")
        runtime = scientific.get("runtime") if isinstance(scientific, Mapping) else None
        runtime_sha = scientific.get("runtime_sha256") if isinstance(scientific, Mapping) else None
        if (
            not isinstance(data, Mapping)
            or not isinstance(metrics, Mapping)
            or not isinstance(runtime, Mapping)
            or runtime_sha != stable_sha256(dict(runtime))
        ):
            raise IntegrityError("source training receipt lacks data/best metrics")
        explicit_h5py_version = runtime.get("h5py")
        if explicit_h5py_version is not None and (
            not isinstance(explicit_h5py_version, str) or not explicit_h5py_version
        ):
            raise IntegrityError("source training runtime has an invalid explicit h5py version")
        if data.get("source_container_identity") != dict(expected_source_identity):
            raise IntegrityError("source checkpoint was trained from another source container")
        if data.get("normalizer") != normalizer.document():
            raise IntegrityError("source checkpoint receipt and normalizer artifact differ")
        if metrics.get("class_support") != monitor_support:
            raise IntegrityError("source checkpoint selection support differs from the source preflight")
        if metrics.get("supported_class_count") != EXPECTED_SUPPORTED_CLASS_COUNT:
            raise IntegrityError("source checkpoint selection was not the disclosed 15-class metric")
        for field in ("config_sha256", "data_identity_sha256"):
            if receipt.get(field) != collection[field]:
                raise IntegrityError(f"source checkpoint receipt differs from collection {field}")
        if (
            data.get("normalizer", {}).get("normalizer_sha256") != collection["normalizer_sha256"]
            or data.get("source_rows_sha256") != collection["source_rows_sha256"]
        ):
            raise IntegrityError("source checkpoint receipt differs from collection normalization/rows")
        observed_collection_rows.append(result.collection_row())
        initial_hashes.append(result.initial_tensor_sha256)
        tensor_hashes.append(result.checkpoint_tensor_sha256)
        verified_rows.append(
            {
                "model_seed": model_seed,
                "checkpoint_basename": result.checkpoint_path.name,
                "checkpoint_file_sha256": result.checkpoint_file_sha256,
                "checkpoint_tensor_sha256": result.checkpoint_tensor_sha256,
                "training_receipt_artifact": _portable_artifact_identity(
                    receipt_path, receipt_byte_receipt
                ),
                "initial_tensor_sha256": result.initial_tensor_sha256,
                "best_epoch_zero_based": result.best_epoch,
                "best_source_monitor_macro_recall_supported_classes": (
                    result.best_source_monitor_macro_recall
                ),
                "best_source_monitor_top1_accuracy": result.best_source_monitor_accuracy,
                "source_monitor_supported_class_count": EXPECTED_SUPPORTED_CLASS_COUNT,
                "source_monitor_absent_class_ids": list(
                    EXPECTED_ABSENT_SOURCE_MONITOR_CLASS_IDS
                ),
                "scientific_identity_sha256": receipt["scientific_identity_sha256"],
                "training_runtime_sha256": runtime_sha,
                "training_runtime_h5py_version": explicit_h5py_version,
                "config_sha256": receipt["config_sha256"],
                "data_identity_sha256": receipt["data_identity_sha256"],
                "code_sha256": receipt["code_sha256"],
            }
        )
    if rows != observed_collection_rows:
        raise IntegrityError("source checkpoint collection rows do not replay from final artifacts")
    if len(set(initial_hashes)) != 5 or len(set(tensor_hashes)) != 5:
        raise IntegrityError("source checkpoint collection lacks five independent tensor identities")
    if (
        collection["normalizer_sha256"] != normalizer.normalizer_sha256
        or collection["source_rows_sha256"] != normalizer.source_rows_sha256
    ):
        raise IntegrityError("source checkpoint collection and normalizer identity differ")
    return collection_path, collection, collection_receipt, verified_rows


def _source_checkpoint_selection_disclosure() -> dict[str, Any]:
    return {
        "checkpoint_selection_primary": "macro_recall_over_supported_classes",
        "source_monitor_supported_class_count": EXPECTED_SUPPORTED_CLASS_COUNT,
        "source_monitor_absent_class_ids": list(EXPECTED_ABSENT_SOURCE_MONITOR_CLASS_IDS),
        "is_17_class_macro_recall": False,
        "development_target_endpoint": "top1_accuracy",
        "required_reporting": (
            "Checkpoint selection used macro recall over exactly 15 supported source_monitor "
            "classes; classes 0 and 6 were absent. Development and target benefits use top-1 accuracy."
        ),
    }


def _source_initialization_clarification(
    initial_hashes_by_model_seed: Mapping[str, str],
) -> dict[str, Any]:
    if set(initial_hashes_by_model_seed) != {
        str(model_seed) for model_seed in CANONICAL_MODEL_SEEDS
    }:
        raise IntegrityError("initial-tensor mapping does not cover model seeds 0--4")
    hashes = {
        str(model_seed): require_sha256(
            initial_hashes_by_model_seed[str(model_seed)],
            field=f"initial_tensor_sha256_by_model_seed.{model_seed}",
        )
        for model_seed in CANONICAL_MODEL_SEEDS
    }
    return {
        "architecture_id": ARCHITECTURE_ID,
        "legacy_architecture_spec_initialization_label": (
            "independent_torchvision_kaiming_per_model_seed"
        ),
        "residual_body_initialization": "torchvision_resnet18_constructor_initialization",
        "replacement_conv1_initialization": (
            "torch.nn.Conv2d_default_reset_parameters_kaiming_uniform_a_sqrt5"
        ),
        "replacement_fc_initialization": "torch.nn.Linear_default_reset_parameters",
        "exact_initial_tensor_hashes_authoritative": True,
        "initial_tensor_sha256_by_model_seed": hashes,
        "numerical_artifacts_changed_by_clarification": False,
    }


def _source_hdf5_runtime_disclosure(
    verified_checkpoints: list[Mapping[str, Any]],
    acceptance_environment_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Disclose the source-runtime gap without retroactively relabeling it."""

    runtime_sha_by_seed: dict[str, str] = {}
    h5py_version_by_seed: dict[str, str | None] = {}
    for expected_seed, row in zip(
        CANONICAL_MODEL_SEEDS, verified_checkpoints, strict=True
    ):
        if row.get("model_seed") != expected_seed:
            raise IntegrityError("source runtime disclosure checkpoint order drift")
        runtime_sha_by_seed[str(expected_seed)] = require_sha256(
            row.get("training_runtime_sha256"),
            field=f"verified_checkpoints[{expected_seed}].training_runtime_sha256",
        )
        h5py_version = row.get("training_runtime_h5py_version")
        if h5py_version is not None and (
            not isinstance(h5py_version, str) or not h5py_version
        ):
            raise IntegrityError("source runtime disclosure has an invalid h5py version")
        h5py_version_by_seed[str(expected_seed)] = h5py_version
    package_versions = acceptance_environment_identity.get("package_versions")
    acceptance_h5py = (
        package_versions.get("h5py") if isinstance(package_versions, Mapping) else None
    )
    if (
        not isinstance(acceptance_h5py, str)
        or not acceptance_h5py
        or acceptance_h5py == "NOT_INSTALLED"
    ):
        raise IntegrityError("source acceptance environment lacks an installed h5py version")
    return {
        "source_preflight_schema": SOURCE_PREFLIGHT_SCHEMA,
        "source_preflight_explicit_h5py_version_recorded": False,
        "source_training_scientific_identity_schema": (
            "kbound_so2sat_source_seed_identity_v1"
        ),
        "source_training_runtime_sha256_by_model_seed": runtime_sha_by_seed,
        "source_training_explicit_h5py_version_by_model_seed": h5py_version_by_seed,
        "all_source_training_receipts_explicitly_record_h5py_version": all(
            version is not None for version in h5py_version_by_seed.values()
        ),
        "postrun_acceptance_h5py_version": acceptance_h5py,
        "postrun_acceptance_h5py_version_is_retroactive_source_runtime_proof": False,
        "required_reporting": (
            "The sealed source-preflight artifact and source-training v1 receipts do not "
            "necessarily expose h5py as a named runtime field. Their artifact, code, and "
            "per-seed runtime hashes remain authoritative. The post-run acceptance seals "
            "its own h5py version; that later observation is not retroactive proof of the "
            "h5py version used by preflight or training."
        ),
    }


def create_source_postrun_acceptance(
    *,
    population_manifest: str | os.PathLike[str],
    training_data: str | os.PathLike[str],
    source_preflight: str | os.PathLike[str],
    checkpoint_dir: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify completed source artifacts, then hash source bytes once and seal acceptance."""

    output_path = Path(output).expanduser().resolve()
    if output_path.name != SOURCE_POSTRUN_ACCEPTANCE_BASENAME:
        raise IntegrityError(
            f"source post-run acceptance output must be named {SOURCE_POSTRUN_ACCEPTANCE_BASENAME!r}"
        )
    output_receipt_path = output_path.with_name(output_path.name + ".receipt.json")
    if output_path.exists() or output_receipt_path.exists():
        raise IntegrityError(
            f"refusing to overwrite source post-run acceptance pair for {output_path}"
        )
    acceptance_code_identity = _acceptance_code_identity()
    acceptance_environment_identity = _acceptance_environment_identity()
    manifest_path, manifest, manifest_receipt = _load_population_manifest(population_manifest)
    source_path = require_source_training_path(training_data)
    preflight_path, preflight, preflight_receipt = _load_source_preflight(source_preflight)
    manifest_identity = preflight.get("population_manifest_identity")
    expected_manifest_identity = {
        "basename": manifest_path.name,
        "bytes": manifest_path.stat().st_size,
        "file_sha256": manifest_receipt["artifact_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "population_identity_sha256": manifest["population_identity_sha256"],
        "receipt_artifact_sha256": manifest_receipt["artifact_sha256"],
    }
    if manifest_identity != expected_manifest_identity:
        raise IntegrityError("source preflight belongs to another population manifest")
    monitor_support, absent_class_ids = _source_monitor_support(preflight)
    if absent_class_ids != list(EXPECTED_ABSENT_SOURCE_MONITOR_CLASS_IDS):  # defensive clarity
        raise IntegrityError("source_monitor absent-class disclosure drift")
    expected_source_identity = _expected_source_container_identity(preflight)

    directory = Path(checkpoint_dir).expanduser().resolve()
    normalizer_path = directory / SOURCE_NORMALIZER_BASENAME
    normalizer_receipt = verify_artifact_receipt(normalizer_path)
    normalizer = load_sealed_band_normalizer(normalizer_path)
    expected_container_sha = stable_sha256(expected_source_identity)
    if normalizer.source_container_identity_sha256 != expected_container_sha:
        raise IntegrityError("source normalizer is not bound to the preflight source container")
    (
        collection_path,
        collection,
        collection_receipt,
        verified_checkpoints,
    ) = _verify_checkpoint_collection(
        directory,
        expected_source_identity,
        normalizer,
        monitor_support,
    )

    # The expensive source-file pass is deliberately last, after all five final
    # artifacts are known to exist.  This function invokes file_sha256 exactly
    # once for training.h5 and never opens it through h5py.
    stat_before = source_path.stat()
    if stat_before.st_size != expected_source_identity["bytes"]:
        raise IntegrityError("post-run source byte count differs from the source preflight")
    observed_source_sha = file_sha256(source_path)
    stat_after = source_path.stat()
    if (
        stat_after.st_size != stat_before.st_size
        or stat_after.st_mtime_ns != stat_before.st_mtime_ns
        or observed_source_sha != expected_source_identity["file_sha256"]
    ):
        raise IntegrityError("post-run source bytes changed or differ from the source preflight")
    if (
        _acceptance_code_identity() != acceptance_code_identity
        or _acceptance_environment_identity() != acceptance_environment_identity
    ):
        raise IntegrityError("source acceptance code/environment changed during verification")

    initial_hashes = {
        str(row["model_seed"]): row["initial_tensor_sha256"]
        for row in verified_checkpoints
    }
    document = {
        "schema": SOURCE_POSTRUN_ACCEPTANCE_SCHEMA,
        "status": SOURCE_POSTRUN_ACCEPTANCE_STATUS,
        "target_data_inputs": [],
        "acceptance_scope": {
            "official_image_split_hashed": "training",
            "source_hdf5_datasets_opened": False,
            "target_image_containers_opened": False,
            "target_outcome_arrays_opened": False,
            "training_h5_raw_byte_sha256_passes": 1,
            "five_checkpoint_receipt_pairs_verified": True,
        },
        "target_seal_binding": {
            "required_field": TARGET_SEAL_BINDING_FIELD,
            "value_source": TARGET_SEAL_BINDING_VALUE_SOURCE,
        },
        "acceptance_code_identity": acceptance_code_identity,
        "acceptance_environment_identity": acceptance_environment_identity,
        "population_manifest": {
            **_portable_artifact_identity(manifest_path, manifest_receipt),
            "schema": manifest["schema"],
            "status": manifest["status"],
            "manifest_sha256": manifest["manifest_sha256"],
            "population_identity_sha256": manifest["population_identity_sha256"],
        },
        "source_preflight": {
            **_portable_artifact_identity(preflight_path, preflight_receipt),
            "schema": preflight["schema"],
            "status": preflight["status"],
            "training_container_bytes": expected_source_identity["bytes"],
            "training_container_sha256": expected_source_identity["file_sha256"],
            "population_manifest_sha256": manifest["manifest_sha256"],
            "population_identity_sha256": manifest["population_identity_sha256"],
        },
        "postrun_source_container": {
            "basename": source_path.name,
            "bytes": stat_after.st_size,
            "sha256": observed_source_sha,
            "source_container_identity_sha256": expected_container_sha,
            "matches_source_preflight": True,
            "stable_during_hash": True,
            "hdf5_datasets_opened": False,
        },
        "source_normalizer": {
            **_portable_artifact_identity(normalizer_path, normalizer_receipt),
            "normalizer_sha256": normalizer.normalizer_sha256,
            "source_container_identity_sha256": normalizer.source_container_identity_sha256,
            "source_rows_sha256": normalizer.source_rows_sha256,
        },
        "checkpoint_collection": {
            **_portable_artifact_identity(collection_path, collection_receipt),
            "schema": collection["schema"],
            "status": collection["status"],
            "config_sha256": collection["config_sha256"],
            "data_identity_sha256": collection["data_identity_sha256"],
            "normalizer_sha256": collection["normalizer_sha256"],
            "source_rows_sha256": collection["source_rows_sha256"],
        },
        "verified_checkpoints": verified_checkpoints,
        "source_hdf5_runtime_disclosure": _source_hdf5_runtime_disclosure(
            verified_checkpoints,
            acceptance_environment_identity,
        ),
        "source_checkpoint_selection_disclosure": _source_checkpoint_selection_disclosure(),
        "source_initialization_clarification": _source_initialization_clarification(initial_hashes),
    }
    receipt = write_immutable_json_with_receipt(output_path, document)
    loaded_document, loaded_receipt = load_verified_source_postrun_acceptance(output_path)
    if loaded_document != document or loaded_receipt != receipt:  # pragma: no cover - defensive
        raise IntegrityError("source post-run acceptance failed immediate verification")
    return {"document": loaded_document, "artifact_receipt": loaded_receipt}


def load_verified_source_postrun_acceptance(
    path: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the portable pair and every self-contained acceptance invariant.

    This loader is appropriate for downstream target stages that receive only
    the immutable acceptance pair.  The development boundary must additionally
    call :func:`verify_source_postrun_acceptance_bindings`, which re-verifies
    every upstream path and the current raw ``training.h5`` bytes.
    """

    artifact = Path(path).expanduser().resolve()
    if artifact.name != SOURCE_POSTRUN_ACCEPTANCE_BASENAME:
        raise IntegrityError(
            f"source post-run acceptance must be named {SOURCE_POSTRUN_ACCEPTANCE_BASENAME!r}"
        )
    receipt = verify_artifact_receipt(artifact)
    document = strict_json_load(artifact)
    expected_root_fields = {
        "schema",
        "status",
        "target_data_inputs",
        "acceptance_scope",
        "target_seal_binding",
        "acceptance_code_identity",
        "acceptance_environment_identity",
        "population_manifest",
        "source_preflight",
        "postrun_source_container",
        "source_normalizer",
        "checkpoint_collection",
        "verified_checkpoints",
        "source_hdf5_runtime_disclosure",
        "source_checkpoint_selection_disclosure",
        "source_initialization_clarification",
    }
    if not isinstance(document, dict) or set(document) != expected_root_fields:
        raise IntegrityError("source post-run acceptance has unknown or missing fields")
    if (
        document.get("schema") != SOURCE_POSTRUN_ACCEPTANCE_SCHEMA
        or document.get("status") != SOURCE_POSTRUN_ACCEPTANCE_STATUS
        or document.get("target_data_inputs") != []
        or document.get("target_seal_binding")
        != {
            "required_field": TARGET_SEAL_BINDING_FIELD,
            "value_source": TARGET_SEAL_BINDING_VALUE_SOURCE,
        }
        or document.get("source_checkpoint_selection_disclosure")
        != _source_checkpoint_selection_disclosure()
    ):
        raise IntegrityError("source post-run acceptance contract drift")
    scope = _require_exact_mapping(
        document.get("acceptance_scope"),
        {
            "official_image_split_hashed",
            "source_hdf5_datasets_opened",
            "target_image_containers_opened",
            "target_outcome_arrays_opened",
            "training_h5_raw_byte_sha256_passes",
            "five_checkpoint_receipt_pairs_verified",
        },
        field="acceptance_scope",
    )
    if (
        not isinstance(scope, Mapping)
        or scope.get("official_image_split_hashed") != "training"
        or scope.get("source_hdf5_datasets_opened") is not False
        or scope.get("target_image_containers_opened") is not False
        or scope.get("target_outcome_arrays_opened") is not False
        or scope.get("training_h5_raw_byte_sha256_passes") != 1
        or scope.get("five_checkpoint_receipt_pairs_verified") is not True
    ):
        raise IntegrityError("source post-run acceptance scope drift")

    code_identity = _require_exact_mapping(
        document.get("acceptance_code_identity"),
        {"files_sha256", "code_identity_sha256"},
        field="acceptance_code_identity",
    )
    code_files = code_identity.get("files_sha256")
    if not isinstance(code_files, Mapping) or set(code_files) != set(ACCEPTANCE_CODE_BASENAMES):
        raise IntegrityError("source acceptance code-file coverage drift")
    for name, digest in code_files.items():
        require_sha256(digest, field=f"acceptance_code_identity.files_sha256.{name}")
    if code_identity.get("code_identity_sha256") != stable_sha256(dict(code_files)):
        raise IntegrityError("source acceptance aggregate code identity does not replay")

    environment = _require_exact_mapping(
        document.get("acceptance_environment_identity"),
        {
            "schema",
            "python_implementation",
            "python_version",
            "python_executable_basename",
            "platform_system",
            "platform_release",
            "platform_machine",
            "package_versions",
            "environment_identity_sha256",
        },
        field="acceptance_environment_identity",
    )
    package_versions = environment.get("package_versions")
    if (
        environment.get("schema")
        != "kbound_so2sat_source_postrun_acceptance_environment_v1"
        or not isinstance(package_versions, Mapping)
        or set(package_versions) != {"h5py", "numpy", "torch", "torchvision"}
        or any(not isinstance(value, str) or not value for value in package_versions.values())
        or package_versions.get("h5py") == "NOT_INSTALLED"
    ):
        raise IntegrityError("source acceptance environment identity drift")
    environment_unsigned = dict(environment)
    environment_sha = environment_unsigned.pop("environment_identity_sha256", None)
    if environment_sha != stable_sha256(environment_unsigned):
        raise IntegrityError("source acceptance environment identity does not replay")

    population = _require_exact_mapping(
        document.get("population_manifest"),
        {
            "artifact_basename",
            "artifact_bytes",
            "artifact_sha256",
            "canonical_document_sha256",
            "schema",
            "status",
            "manifest_sha256",
            "population_identity_sha256",
        },
        field="population_manifest",
    )
    _validate_portable_artifact_identity(
        {key: population[key] for key in (
            "artifact_basename",
            "artifact_bytes",
            "artifact_sha256",
            "canonical_document_sha256",
        )},
        field="population_manifest",
    )
    population_manifest_sha = require_sha256(
        population.get("manifest_sha256"), field="population_manifest.manifest_sha256"
    )
    population_identity_sha = require_sha256(
        population.get("population_identity_sha256"),
        field="population_manifest.population_identity_sha256",
    )
    if (
        population.get("schema") != POPULATION_MANIFEST_SCHEMA
        or population.get("status") != "LABEL_FREE_METADATA_POPULATION_VERIFIED"
    ):
        raise IntegrityError("source acceptance population-manifest contract drift")

    preflight = _require_exact_mapping(
        document.get("source_preflight"),
        {
            "artifact_basename",
            "artifact_bytes",
            "artifact_sha256",
            "canonical_document_sha256",
            "schema",
            "status",
            "training_container_bytes",
            "training_container_sha256",
            "population_manifest_sha256",
            "population_identity_sha256",
        },
        field="source_preflight",
    )
    _validate_portable_artifact_identity(
        {key: preflight[key] for key in (
            "artifact_basename",
            "artifact_bytes",
            "artifact_sha256",
            "canonical_document_sha256",
        )},
        field="source_preflight",
    )
    preflight_container_bytes = _require_positive_int(
        preflight.get("training_container_bytes"),
        field="source_preflight.training_container_bytes",
    )
    preflight_container_sha = require_sha256(
        preflight.get("training_container_sha256"),
        field="source_preflight.training_container_sha256",
    )
    if (
        preflight.get("schema") != SOURCE_PREFLIGHT_SCHEMA
        or preflight.get("status")
        not in {
            "SOURCE_DATA_PREFLIGHT_PASSED",
            "SOURCE_DATA_PREFLIGHT_PASSED_WITH_WARNINGS",
        }
        or preflight.get("population_manifest_sha256") != population_manifest_sha
        or preflight.get("population_identity_sha256") != population_identity_sha
    ):
        raise IntegrityError("source acceptance preflight/population binding drift")

    source = _require_exact_mapping(
        document.get("postrun_source_container"),
        {
            "basename",
            "bytes",
            "sha256",
            "source_container_identity_sha256",
            "matches_source_preflight",
            "stable_during_hash",
            "hdf5_datasets_opened",
        },
        field="postrun_source_container",
    )
    source_identity_sha = require_sha256(
        source.get("source_container_identity_sha256"),
        field="postrun_source_container.source_container_identity_sha256",
    )
    if (
        source.get("basename") != "training.h5"
        or source.get("bytes") != preflight_container_bytes
        or source.get("sha256") != preflight_container_sha
        or source.get("matches_source_preflight") is not True
        or source.get("stable_during_hash") is not True
        or source.get("hdf5_datasets_opened") is not False
    ):
        raise IntegrityError("post-run source container/preflight binding drift")

    normalizer = _require_exact_mapping(
        document.get("source_normalizer"),
        {
            "artifact_basename",
            "artifact_bytes",
            "artifact_sha256",
            "canonical_document_sha256",
            "normalizer_sha256",
            "source_container_identity_sha256",
            "source_rows_sha256",
        },
        field="source_normalizer",
    )
    normalizer_artifact = _validate_portable_artifact_identity(
        {key: normalizer[key] for key in (
            "artifact_basename",
            "artifact_bytes",
            "artifact_sha256",
            "canonical_document_sha256",
        )},
        field="source_normalizer",
    )
    normalizer_sha = require_sha256(
        normalizer.get("normalizer_sha256"), field="source_normalizer.normalizer_sha256"
    )
    source_rows_sha = require_sha256(
        normalizer.get("source_rows_sha256"), field="source_normalizer.source_rows_sha256"
    )
    if (
        normalizer_artifact["artifact_basename"] != SOURCE_NORMALIZER_BASENAME
        or normalizer.get("source_container_identity_sha256") != source_identity_sha
    ):
        raise IntegrityError("source normalizer/container binding drift")

    collection = _require_exact_mapping(
        document.get("checkpoint_collection"),
        {
            "artifact_basename",
            "artifact_bytes",
            "artifact_sha256",
            "canonical_document_sha256",
            "schema",
            "status",
            "config_sha256",
            "data_identity_sha256",
            "normalizer_sha256",
            "source_rows_sha256",
        },
        field="checkpoint_collection",
    )
    collection_artifact = _validate_portable_artifact_identity(
        {key: collection[key] for key in (
            "artifact_basename",
            "artifact_bytes",
            "artifact_sha256",
            "canonical_document_sha256",
        )},
        field="checkpoint_collection",
    )
    collection_config_sha = require_sha256(
        collection.get("config_sha256"), field="checkpoint_collection.config_sha256"
    )
    collection_data_sha = require_sha256(
        collection.get("data_identity_sha256"),
        field="checkpoint_collection.data_identity_sha256",
    )
    if (
        collection_artifact["artifact_basename"] != SOURCE_COLLECTION_BASENAME
        or collection.get("schema") != COLLECTION_SCHEMA
        or collection.get("status") != "FIVE_INDEPENDENT_SOURCE_CHECKPOINTS_VERIFIED"
        or collection.get("normalizer_sha256") != normalizer_sha
        or collection.get("source_rows_sha256") != source_rows_sha
    ):
        raise IntegrityError("source checkpoint collection/normalizer binding drift")

    checkpoints = document.get("verified_checkpoints")
    clarification = document.get("source_initialization_clarification")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(CANONICAL_MODEL_SEEDS):
        raise IntegrityError("source post-run acceptance lacks five checkpoint rows")
    checkpoint_fields = {
        "model_seed",
        "checkpoint_basename",
        "checkpoint_file_sha256",
        "checkpoint_tensor_sha256",
        "training_receipt_artifact",
        "initial_tensor_sha256",
        "best_epoch_zero_based",
        "best_source_monitor_macro_recall_supported_classes",
        "best_source_monitor_top1_accuracy",
        "source_monitor_supported_class_count",
        "source_monitor_absent_class_ids",
        "scientific_identity_sha256",
        "training_runtime_sha256",
        "training_runtime_h5py_version",
        "config_sha256",
        "data_identity_sha256",
        "code_sha256",
    }
    initial_by_seed: dict[str, str] = {}
    tensor_hashes: list[str] = []
    for expected_seed, row_value in zip(CANONICAL_MODEL_SEEDS, checkpoints, strict=True):
        row = _require_exact_mapping(
            row_value, checkpoint_fields, field=f"verified_checkpoints[{expected_seed}]"
        )
        expected_checkpoint_basename = f"so2sat_resnet18_seed{expected_seed}.pt"
        if row.get("model_seed") != expected_seed or row.get("checkpoint_basename") != expected_checkpoint_basename:
            raise IntegrityError("source post-run checkpoint seed/basename order drift")
        for field in (
            "checkpoint_file_sha256",
            "checkpoint_tensor_sha256",
            "initial_tensor_sha256",
            "scientific_identity_sha256",
            "training_runtime_sha256",
            "config_sha256",
            "data_identity_sha256",
            "code_sha256",
        ):
            require_sha256(row.get(field), field=f"verified_checkpoints[{expected_seed}].{field}")
        explicit_h5py_version = row.get("training_runtime_h5py_version")
        if explicit_h5py_version is not None and (
            not isinstance(explicit_h5py_version, str) or not explicit_h5py_version
        ):
            raise IntegrityError("source training runtime h5py-version disclosure drift")
        training_receipt_artifact = _validate_portable_artifact_identity(
            row.get("training_receipt_artifact"),
            field=f"verified_checkpoints[{expected_seed}].training_receipt_artifact",
        )
        if training_receipt_artifact["artifact_basename"] != (
            f"so2sat_resnet18_seed{expected_seed}.training.json"
        ):
            raise IntegrityError("source training-receipt basename drift")
        best_epoch = row.get("best_epoch_zero_based")
        macro = row.get("best_source_monitor_macro_recall_supported_classes")
        accuracy = row.get("best_source_monitor_top1_accuracy")
        if (
            isinstance(best_epoch, bool)
            or not isinstance(best_epoch, int)
            or best_epoch < 0
            or isinstance(macro, bool)
            or not isinstance(macro, (int, float))
            or not 0.0 <= float(macro) <= 1.0
            or isinstance(accuracy, bool)
            or not isinstance(accuracy, (int, float))
            or not 0.0 <= float(accuracy) <= 1.0
            or row.get("source_monitor_supported_class_count") != EXPECTED_SUPPORTED_CLASS_COUNT
            or row.get("source_monitor_absent_class_ids")
            != list(EXPECTED_ABSENT_SOURCE_MONITOR_CLASS_IDS)
            or row.get("config_sha256") != collection_config_sha
            or row.get("data_identity_sha256") != collection_data_sha
        ):
            raise IntegrityError("source post-run checkpoint metrics/identity drift")
        initial_by_seed[str(expected_seed)] = row["initial_tensor_sha256"]
        tensor_hashes.append(row["checkpoint_tensor_sha256"])
    if len(set(initial_by_seed.values())) != 5 or len(set(tensor_hashes)) != 5:
        raise IntegrityError("source post-run acceptance lacks five independent tensor identities")
    if (
        not isinstance(clarification, Mapping)
        or clarification != _source_initialization_clarification(initial_by_seed)
    ):
        raise IntegrityError("source post-run checkpoint/initialization clarification drift")
    if document.get("source_hdf5_runtime_disclosure") != _source_hdf5_runtime_disclosure(
        checkpoints,
        environment,
    ):
        raise IntegrityError("source HDF5 runtime disclosure drift")
    return document, receipt


def source_postrun_acceptance_binding(
    document: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, str]:
    """Return the portable downstream binding for an already verified pair."""

    if (
        document.get("schema") != SOURCE_POSTRUN_ACCEPTANCE_SCHEMA
        or document.get("status") != SOURCE_POSTRUN_ACCEPTANCE_STATUS
    ):
        raise IntegrityError("cannot bind an unverified source post-run acceptance")
    if set(receipt) != {
        "schema",
        "artifact_basename",
        "artifact_bytes",
        "artifact_sha256",
        "canonical_document_sha256",
    } or receipt.get("schema") != ARTIFACT_RECEIPT_SCHEMA_V2:
        raise IntegrityError("source post-run acceptance needs a portable v2 receipt")
    basename = _require_portable_basename(
        receipt.get("artifact_basename"), field="source acceptance receipt basename"
    )
    _require_positive_int(
        receipt.get("artifact_bytes"), field="source acceptance receipt artifact_bytes"
    )
    artifact_sha = require_sha256(
        receipt.get("artifact_sha256"), field="source acceptance artifact_sha256"
    )
    canonical_sha = require_sha256(
        receipt.get("canonical_document_sha256"),
        field="source acceptance canonical_document_sha256",
    )
    if (
        basename != SOURCE_POSTRUN_ACCEPTANCE_BASENAME
        or canonical_sha != stable_sha256(dict(document))
    ):
        raise IntegrityError("source post-run acceptance binding does not replay")
    return {
        "source_postrun_acceptance_artifact_basename": basename,
        TARGET_SEAL_BINDING_FIELD: artifact_sha,
        "source_postrun_acceptance_canonical_document_sha256": canonical_sha,
    }


def verify_source_postrun_acceptance_bindings(
    acceptance_path: str | os.PathLike[str],
    *,
    population_manifest_path: str | os.PathLike[str],
    source_preflight_path: str | os.PathLike[str],
    training_data_path: str | os.PathLike[str],
    checkpoint_dir: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-verify the complete source chain before any gate-fit data are opened."""

    document, receipt = load_verified_source_postrun_acceptance(acceptance_path)
    verifier_code_start = _acceptance_code_identity()
    if document["acceptance_code_identity"] != verifier_code_start:
        raise IntegrityError("current source acceptance code differs from its creator identity")
    if document["acceptance_environment_identity"] != _acceptance_environment_identity():
        raise IntegrityError(
            "current source acceptance environment differs from its creator identity"
        )
    manifest_path, manifest, manifest_receipt = _load_population_manifest(
        population_manifest_path
    )
    expected_population = {
        **_portable_artifact_identity(manifest_path, manifest_receipt),
        "schema": manifest["schema"],
        "status": manifest["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "population_identity_sha256": manifest["population_identity_sha256"],
    }
    if document["population_manifest"] != expected_population:
        raise IntegrityError("source acceptance belongs to another population manifest")

    preflight_path, preflight, preflight_receipt = _load_source_preflight(
        source_preflight_path
    )
    expected_source_identity = _expected_source_container_identity(preflight)
    expected_preflight = {
        **_portable_artifact_identity(preflight_path, preflight_receipt),
        "schema": preflight["schema"],
        "status": preflight["status"],
        "training_container_bytes": expected_source_identity["bytes"],
        "training_container_sha256": expected_source_identity["file_sha256"],
        "population_manifest_sha256": manifest["manifest_sha256"],
        "population_identity_sha256": manifest["population_identity_sha256"],
    }
    if document["source_preflight"] != expected_preflight:
        raise IntegrityError("source acceptance belongs to another source preflight")
    preflight_manifest = preflight.get("population_manifest_identity")
    if preflight_manifest != {
        "basename": manifest_path.name,
        "bytes": manifest_path.stat().st_size,
        "file_sha256": manifest_receipt["artifact_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "population_identity_sha256": manifest["population_identity_sha256"],
        "receipt_artifact_sha256": manifest_receipt["artifact_sha256"],
    }:
        raise IntegrityError("source preflight population-manifest identity drift")

    monitor_support, _ = _source_monitor_support(preflight)
    directory = Path(checkpoint_dir).expanduser().resolve()
    normalizer_path = directory / SOURCE_NORMALIZER_BASENAME
    normalizer_receipt = verify_artifact_receipt(normalizer_path)
    normalizer = load_sealed_band_normalizer(normalizer_path)
    expected_container_sha = stable_sha256(expected_source_identity)
    if normalizer.source_container_identity_sha256 != expected_container_sha:
        raise IntegrityError("source normalizer belongs to another source container")
    collection_path, collection, collection_receipt, verified_rows = (
        _verify_checkpoint_collection(
            directory,
            expected_source_identity,
            normalizer,
            monitor_support,
        )
    )
    expected_normalizer = {
        **_portable_artifact_identity(normalizer_path, normalizer_receipt),
        "normalizer_sha256": normalizer.normalizer_sha256,
        "source_container_identity_sha256": normalizer.source_container_identity_sha256,
        "source_rows_sha256": normalizer.source_rows_sha256,
    }
    expected_collection = {
        **_portable_artifact_identity(collection_path, collection_receipt),
        "schema": collection["schema"],
        "status": collection["status"],
        "config_sha256": collection["config_sha256"],
        "data_identity_sha256": collection["data_identity_sha256"],
        "normalizer_sha256": collection["normalizer_sha256"],
        "source_rows_sha256": collection["source_rows_sha256"],
    }
    if document["source_normalizer"] != expected_normalizer:
        raise IntegrityError("source acceptance normalizer binding drift")
    if document["checkpoint_collection"] != expected_collection:
        raise IntegrityError("source acceptance checkpoint-collection binding drift")
    if document["verified_checkpoints"] != verified_rows:
        raise IntegrityError("source acceptance checkpoint-pair bindings drift")

    # Raw bytes are verified before any HDF5 dataset constructor is called.
    source_path = require_source_training_path(training_data_path)
    stat_before = source_path.stat()
    observed_sha = file_sha256(source_path)
    stat_after = source_path.stat()
    expected_postrun_source = {
        "basename": source_path.name,
        "bytes": stat_after.st_size,
        "sha256": observed_sha,
        "source_container_identity_sha256": expected_container_sha,
        "matches_source_preflight": True,
        "stable_during_hash": True,
        "hdf5_datasets_opened": False,
    }
    if (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        or stat_after.st_size != expected_source_identity["bytes"]
        or observed_sha != expected_source_identity["file_sha256"]
        or document["postrun_source_container"] != expected_postrun_source
    ):
        raise IntegrityError("current training.h5 bytes differ from the accepted source chain")
    if _acceptance_code_identity() != verifier_code_start:
        raise IntegrityError("source acceptance verifier code changed during chain replay")
    source_postrun_acceptance_binding(document, receipt)
    return document, receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-manifest", required=True, type=Path)
    parser.add_argument("--training-data", required=True, type=Path)
    parser.add_argument("--source-preflight", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = create_source_postrun_acceptance(
        population_manifest=args.population_manifest,
        training_data=args.training_data,
        source_preflight=args.source_preflight,
        checkpoint_dir=args.checkpoint_dir,
        output=args.output,
    )
    receipt = result["artifact_receipt"]
    print(
        "So2Sat source post-run acceptance: PASS "
        f"artifact_sha256={receipt['artifact_sha256']} "
        f"target_seal_field={TARGET_SEAL_BINDING_FIELD}",
        flush=True,
    )


if __name__ == "__main__":
    main()
