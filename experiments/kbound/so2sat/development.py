#!/usr/bin/env python3
"""Development-only So2Sat adapter evaluation and candidate selection.

This runner accepts only the official training container and its separately
sealed geographic manifest.  Candidate selection consumes ``gate_fit`` cities
only.  ``gate_cal`` cities cannot be opened until a create-only selection
artifact proves that exactly one candidate passed the declared feasibility
gate.  No target split path is accepted anywhere in this module.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import math
import os
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .adapters import (
    ADAPTER_BATCH_SIZE,
    CANDIDATE_IDS,
    TENT_CANDIDATE_ID,
    adapt_on_probe,
    candidate_spec,
    fixed_model_logits,
    frozen_logits_and_bn_divergence,
    validate_adaptation_diagnostics,
    validate_candidate_spec,
)
from .features import extract_label_free_features, feature_vector, validate_feature_document
from .gate import (
    CHECKPOINT_IDS,
    fit_calibrate_ridge_gate,
    load_gate_with_receipt,
    load_study_binding,
    trace_identity_sha256,
    validate_gate_document,
    validate_study_binding,
    write_gate_with_receipt,
)
from .integrity import (
    IntegrityError,
    file_sha256,
    ordered_records_sha256,
    require_sha256,
    stable_sha256,
    strict_json_load,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from .label_firewall import VerifiedTrainingGeoIndex
from .metadata_manifest import GeoRecord
from .model import (
    ARCHITECTURE_ID,
    NUM_CLASSES,
    build_so2sat_resnet18,
    tensor_state_sha256,
)
from .source_acceptance import (
    source_postrun_acceptance_binding,
    verify_source_postrun_acceptance_bindings,
)
from .source_data import (
    BandNormalizer,
    H5SourceContainer,
    load_sealed_band_normalizer,
)
from .train_source import CHECKPOINT_SCHEMA, COLLECTION_SCHEMA, TRAINING_RECEIPT_SCHEMA

GATE_FIT_ROLE = "gate_fit"
GATE_CAL_ROLE = "gate_cal"
DEVELOPMENT_ROLES = (GATE_FIT_ROLE, GATE_CAL_ROLE)
ROLE_SAMPLE_NAMES = {
    GATE_FIT_ROLE: ("gate_fit_probe", "gate_fit_evaluation"),
    GATE_CAL_ROLE: ("gate_cal_probe", "gate_cal_evaluation"),
}

CANDIDATE_BUNDLE_SCHEMA = "kbound_so2sat_development_candidate_bundle_v1"
SELECTION_SCHEMA = "kbound_so2sat_adapter_candidate_selection_v1"
CELL_SCHEMA = "kbound_so2sat_development_adapter_cell_v1"
GATE_AUTHORIZATION_SCHEMA = "kbound_so2sat_gate_authorization_v1"
DEVELOPMENT_ENVIRONMENT_SCHEMA = "kbound_so2sat_development_environment_v1"
NO_FEASIBLE_CANDIDATE_EXIT_CODE = 20
_GATE_ROW_KEYS = {
    "role",
    "city_id",
    "checkpoint_id",
    "checkpoint_tensor_sha256",
    "checkpoint_file_sha256",
    "trace_id",
    "trace_sha256",
    "partition_sha256",
    "manifest_sha256",
    "population_identity_sha256",
    "protocol_file_sha256",
    "protocol_document_sha256",
    "feature_document",
    "observed_benefit",
}
_CELL_KEYS = {
    "schema",
    "status",
    "candidate_id",
    "candidate_config_sha256",
    "role",
    "city_id",
    "checkpoint_id",
    "probe_n",
    "evaluation_n",
    "frozen_evaluation_accuracy",
    "adapted_evaluation_accuracy",
    "observed_benefit",
    "adapter_diagnostics",
    "gate_row",
    "source_training_receipt_sha256",
    "source_normalizer_sha256",
    "source_container_identity_sha256",
    "runner_code_sha256",
    "probe_labels_read",
    "evaluation_label_read_passes",
    "target_pixels_read",
    "target_labels_read",
    "target_inputs",
    "cell_sha256",
}
_CANDIDATE_BUNDLE_KEYS = {
    "schema",
    "status",
    "role",
    "candidate_spec",
    "candidate_config_sha256",
    "study_binding",
    "checkpoint_collection_canonical_sha256",
    "source_container_identity_sha256",
    "normalizer_sha256",
    "runner_code",
    "development_environment_identity",
    "cells",
    "gate_rows_sha256",
    "candidate_feasibility",
    "candidate_selection_used_this_bundle",
    "target_pixels_read",
    "target_labels_read",
    "target_inputs",
    "bundle_sha256",
}

RIDGE_PENALTY = 10.0
MIN_CITY_MEAN_MAGNITUDE = 0.0025
MIN_HELPFUL_CITIES = 2
MIN_HARMFUL_CITIES = 2
MIN_ORACLE_ROUTING_GAP = 0.0025
MIN_LOCO_ROUTED_GAIN_OVER_BEST_FIXED = 0.0010
MIN_LOCO_SIGN_ACCURACY = 0.55
MIN_ACTION_CELLS_PER_POLICY = 9
MIN_ACTION_CITIES_PER_POLICY = 2


class NoFeasibleCandidateError(IntegrityError):
    """Raised before gate-calibration data access when no candidate qualifies."""


def development_environment_identity(device: torch.device) -> dict[str, Any]:
    """Return the self-hashed runtime identity for one development phase."""

    if device.type not in {"cpu", "mps"}:
        raise IntegrityError("development environment supports only CPU or MPS")
    versions: dict[str, str] = {}
    for package in ("h5py", "numpy", "torch", "torchvision"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    if any(value == "NOT_INSTALLED" for value in versions.values()):
        raise IntegrityError("development environment lacks a required package")
    document = {
        "schema": DEVELOPMENT_ENVIRONMENT_SCHEMA,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable_basename": Path(sys.executable).name,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "package_versions": versions,
        "device_type": device.type,
        "torch_deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    document["environment_identity_sha256"] = stable_sha256(document)
    validate_development_environment_identity(document)
    return document


def validate_development_environment_identity(document: Mapping[str, Any]) -> None:
    """Validate a recorded CPU/MPS development runtime without relabeling it."""

    expected_fields = {
        "schema",
        "python_implementation",
        "python_version",
        "python_executable_basename",
        "platform_system",
        "platform_release",
        "platform_machine",
        "package_versions",
        "device_type",
        "torch_deterministic_algorithms_enabled",
        "torch_num_threads",
        "torch_num_interop_threads",
        "mps_built",
        "mps_available",
        "environment_identity_sha256",
    }
    if not isinstance(document, Mapping) or set(document) != expected_fields:
        raise IntegrityError("development environment identity schema drift")
    versions = document.get("package_versions")
    if (
        document.get("schema") != DEVELOPMENT_ENVIRONMENT_SCHEMA
        or not isinstance(versions, Mapping)
        or set(versions) != {"h5py", "numpy", "torch", "torchvision"}
        or any(not isinstance(value, str) or not value or value == "NOT_INSTALLED" for value in versions.values())
        or document.get("device_type") not in {"cpu", "mps"}
        or not isinstance(document.get("torch_deterministic_algorithms_enabled"), bool)
        or not isinstance(document.get("mps_built"), bool)
        or not isinstance(document.get("mps_available"), bool)
    ):
        raise IntegrityError("development environment identity contract drift")
    if document.get("device_type") == "mps" and document.get("mps_available") is not True:
        raise IntegrityError("development environment records unavailable MPS execution")
    for field in ("torch_num_threads", "torch_num_interop_threads"):
        value = document.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise IntegrityError(f"development environment {field} is invalid")
    for field in (
        "python_implementation",
        "python_version",
        "python_executable_basename",
        "platform_system",
        "platform_release",
        "platform_machine",
    ):
        if not isinstance(document.get(field), str) or not document[field]:
            raise IntegrityError(f"development environment {field} is invalid")
    claimed = require_sha256(
        document.get("environment_identity_sha256"),
        field="development_environment_identity.environment_identity_sha256",
    )
    unsigned = dict(document)
    unsigned.pop("environment_identity_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("development environment identity SHA-256 mismatch")


@dataclass(frozen=True)
class DevelopmentRow:
    row_index: int
    sample_id: str
    city_id: str
    spatial_block_id: str
    sample_role: str

    def commitment(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "sample_id": self.sample_id,
            "city_id": self.city_id,
            "spatial_block_id": self.spatial_block_id,
            "sample_role": self.sample_role,
        }


@dataclass(frozen=True)
class CityPartition:
    role: str
    city_id: str
    probe_rows: tuple[DevelopmentRow, ...]
    evaluation_rows: tuple[DevelopmentRow, ...]
    partition_sha256: str

    def __post_init__(self) -> None:
        if self.role not in DEVELOPMENT_ROLES:
            raise IntegrityError("unknown development partition role")
        probe_role, evaluation_role = ROLE_SAMPLE_NAMES[self.role]
        if not self.probe_rows or not self.evaluation_rows:
            raise IntegrityError("development city requires nonempty probe and evaluation halves")
        if any(row.city_id != self.city_id for row in (*self.probe_rows, *self.evaluation_rows)):
            raise IntegrityError("development partition contains another city")
        if any(row.sample_role != probe_role for row in self.probe_rows):
            raise IntegrityError("development probe rows have the wrong role")
        if any(row.sample_role != evaluation_role for row in self.evaluation_rows):
            raise IntegrityError("development evaluation rows have the wrong role")
        probe_indices = [row.row_index for row in self.probe_rows]
        evaluation_indices = [row.row_index for row in self.evaluation_rows]
        if probe_indices != sorted(probe_indices) or len(probe_indices) != len(set(probe_indices)):
            raise IntegrityError("development probe row indices must be sorted and unique")
        if evaluation_indices != sorted(evaluation_indices) or len(evaluation_indices) != len(set(evaluation_indices)):
            raise IntegrityError("development evaluation row indices must be sorted and unique")
        if set(probe_indices) & set(evaluation_indices):
            raise IntegrityError("development probe/evaluation rows overlap")
        require_sha256(self.partition_sha256, field="partition_sha256")
        if self.partition_sha256 != _partition_hash(self.role, self.city_id, self.probe_rows, self.evaluation_rows):
            raise IntegrityError("development partition SHA-256 mismatch")


@dataclass(frozen=True)
class DevelopmentInventory:
    population_n: int
    manifest_sha256: str
    population_identity_sha256: str
    partitions: Mapping[str, Mapping[str, CityPartition]]

    def __post_init__(self) -> None:
        if isinstance(self.population_n, bool) or not isinstance(self.population_n, int) or self.population_n < 1:
            raise IntegrityError("development inventory population count is invalid")
        require_sha256(self.manifest_sha256, field="manifest_sha256")
        require_sha256(self.population_identity_sha256, field="population_identity_sha256")
        if set(self.partitions) != set(DEVELOPMENT_ROLES):
            raise IntegrityError("development inventory must contain gate_fit and gate_cal roles")


@dataclass(frozen=True)
class VerifiedCheckpoint:
    checkpoint_id: str
    model_seed: int
    checkpoint_path: Path
    checkpoint_file_sha256: str
    checkpoint_tensor_sha256: str
    checkpoint_payload: Mapping[str, Any]
    training_receipt_sha256: str

    def fresh_model(self, *, device: torch.device) -> nn.Module:
        state = self.checkpoint_payload.get("model_state")
        if not isinstance(state, Mapping):
            raise IntegrityError("verified source checkpoint lost its model state")
        if tensor_state_sha256(state) != self.checkpoint_tensor_sha256:
            raise IntegrityError("source checkpoint tensor state changed in memory")
        model = build_so2sat_resnet18()
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        return model


def _partition_hash(
    role: str,
    city_id: str,
    probe_rows: Sequence[DevelopmentRow],
    evaluation_rows: Sequence[DevelopmentRow],
) -> str:
    return stable_sha256(
        {
            "schema": "kbound_so2sat_development_spatial_partition_v1",
            "role": role,
            "city_id": city_id,
            "probe_role": ROLE_SAMPLE_NAMES[role][0],
            "evaluation_role": ROLE_SAMPLE_NAMES[role][1],
            "probe_n": len(probe_rows),
            "evaluation_n": len(evaluation_rows),
            "probe_rows_sha256": ordered_records_sha256(row.commitment() for row in probe_rows),
            "evaluation_rows_sha256": ordered_records_sha256(row.commitment() for row in evaluation_rows),
        }
    )


def build_development_inventory(
    geo_index: VerifiedTrainingGeoIndex,
    manifest: Mapping[str, Any],
    *,
    study_binding: Mapping[str, Any],
) -> DevelopmentInventory:
    """Reproduce all development roles from training-only geographic metadata."""

    validate_study_binding(study_binding)
    population_n = manifest.get("splits", {}).get("training", {}).get("observed_samples")
    if isinstance(population_n, bool) or not isinstance(population_n, int) or population_n < 1:
        raise IntegrityError("manifest has an invalid training population")
    if geo_index.population_identity_sha256 != study_binding["population_identity_sha256"]:
        raise IntegrityError("training geo index and gate study binding differ")
    expected_city_sets = {
        GATE_FIT_ROLE: set(study_binding["gate_fit_cities"]),
        GATE_CAL_ROLE: set(study_binding["gate_cal_cities"]),
    }
    collected: dict[str, dict[str, dict[str, list[DevelopmentRow]]]] = {
        role: {city: {"probe": [], "evaluation": []} for city in sorted(cities)}
        for role, cities in expected_city_sets.items()
    }
    observed_rows = 0
    for expected_index, record in enumerate(geo_index.iter_records()):
        if not isinstance(record, GeoRecord) or record.row_index != expected_index:
            raise IntegrityError("training geo index returned an invalid ordered record")
        observed_rows += 1
        for role in DEVELOPMENT_ROLES:
            probe_name, evaluation_name = ROLE_SAMPLE_NAMES[role]
            if record.sample_role not in {probe_name, evaluation_name}:
                continue
            if record.city_id not in expected_city_sets[role] or record.city_role != role:
                raise IntegrityError("development row city role differs from study binding")
            half = "probe" if record.sample_role == probe_name else "evaluation"
            collected[role][record.city_id][half].append(
                DevelopmentRow(
                    row_index=record.row_index,
                    sample_id=record.sample_id,
                    city_id=record.city_id,
                    spatial_block_id=record.spatial_block_id,
                    sample_role=record.sample_role,
                )
            )
            break
    if observed_rows != population_n:
        raise IntegrityError("training geo scan did not cover the sealed population")

    partitions: dict[str, dict[str, CityPartition]] = {}
    for role in DEVELOPMENT_ROLES:
        partitions[role] = {}
        for city in sorted(expected_city_sets[role]):
            probe = tuple(collected[role][city]["probe"])
            evaluation = tuple(collected[role][city]["evaluation"])
            partitions[role][city] = CityPartition(
                role=role,
                city_id=city,
                probe_rows=probe,
                evaluation_rows=evaluation,
                partition_sha256=_partition_hash(role, city, probe, evaluation),
            )
    return DevelopmentInventory(
        population_n=population_n,
        manifest_sha256=require_sha256(manifest.get("manifest_sha256"), field="manifest_sha256"),
        population_identity_sha256=require_sha256(
            manifest.get("population_identity_sha256"), field="population_identity_sha256"
        ),
        partitions=partitions,
    )


class DevelopmentData:
    """Role-gated access to development rows in ``training.h5`` only."""

    def __init__(
        self,
        training_data: str | os.PathLike[str],
        inventory: DevelopmentInventory,
        normalizer: BandNormalizer,
        *,
        authorized_role: str,
    ) -> None:
        if authorized_role not in DEVELOPMENT_ROLES:
            raise IntegrityError("development data authority has an unknown role")
        self.authorized_role = authorized_role
        self.inventory = inventory
        self.container = H5SourceContainer(training_data, expected_rows=inventory.population_n)
        self.normalizer = normalizer
        if normalizer.source_container_identity_sha256 != self.container.identity_sha256:
            raise IntegrityError("development container differs from source normalizer container")
        self._authorized: dict[str, set[int]] = {}
        for partition in inventory.partitions[authorized_role].values():
            self._authorized[f"{authorized_role}_probe:{partition.city_id}"] = {
                row.row_index for row in partition.probe_rows
            }
            self._authorized[f"{authorized_role}_evaluation:{partition.city_id}"] = {
                row.row_index for row in partition.evaluation_rows
            }

    def _validate_rows(
        self,
        partition: CityPartition,
        rows: Sequence[DevelopmentRow],
        *,
        half: str,
    ) -> tuple[int, ...]:
        if partition.role != self.authorized_role:
            raise IntegrityError(
                "development read attempted outside its phase-specific authority"
            )
        if half not in {"probe", "evaluation"}:
            raise IntegrityError("unknown development partition half")
        expected_rows = partition.probe_rows if half == "probe" else partition.evaluation_rows
        if tuple(rows) != expected_rows:
            raise IntegrityError("development read rows differ from the sealed city partition")
        indices = tuple(row.row_index for row in rows)
        authority = self._authorized[f"{partition.role}_{half}:{partition.city_id}"]
        if set(indices) != authority or len(indices) != len(authority):
            raise IntegrityError("development read exceeded its row authority")
        return indices

    def pixel_batches(
        self,
        partition: CityPartition,
        *,
        half: str,
        batch_size: int = ADAPTER_BATCH_SIZE,
    ) -> Iterable[torch.Tensor]:
        rows = partition.probe_rows if half == "probe" else partition.evaluation_rows
        indices = self._validate_rows(partition, rows, half=half)
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            pixels = self.container.read_pixels(selected)
            yield self._normalize_pixels(pixels)

    def evaluation_batches(
        self,
        partition: CityPartition,
        *,
        batch_size: int = ADAPTER_BATCH_SIZE,
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        rows = partition.evaluation_rows
        indices = self._validate_rows(partition, rows, half="evaluation")
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            pixels, labels = self.container.read_labeled_many(selected)
            yield self._normalize_pixels(pixels), _classes_from_one_hot(labels)

    def _normalize_pixels(self, pixels: np.ndarray) -> torch.Tensor:
        values = np.asarray(pixels, dtype=np.float32)
        if values.ndim != 4 or tuple(values.shape[1:]) != (32, 32, 10):
            raise IntegrityError("development Sentinel-2 batch shape drift")
        if not np.isfinite(values).all():
            raise IntegrityError("development Sentinel-2 pixels contain NaN or Infinity")
        tensor = torch.from_numpy(values).permute(0, 3, 1, 2).contiguous()
        mean = torch.tensor(self.normalizer.mean, dtype=tensor.dtype)[None, :, None, None]
        std = torch.tensor(self.normalizer.std, dtype=tensor.dtype)[None, :, None, None]
        return (tensor - mean) / std


def _classes_from_one_hot(labels: np.ndarray) -> torch.Tensor:
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != NUM_CLASSES:
        raise IntegrityError("development evaluation labels must have shape N x 17")
    if not np.isfinite(values).all():
        raise IntegrityError("development evaluation labels contain NaN or Infinity")
    zeros = np.isclose(values, 0.0, rtol=0.0, atol=1.0e-6)
    ones = np.isclose(values, 1.0, rtol=0.0, atol=1.0e-6)
    if not np.all(zeros | ones) or not np.all(ones.sum(axis=1) == 1):
        raise IntegrityError("development evaluation labels must be exactly one-hot")
    return torch.from_numpy(np.argmax(values, axis=1).astype(np.int64))


def load_verified_checkpoints(
    checkpoint_dir: str | os.PathLike[str],
) -> tuple[dict[str, Any], tuple[VerifiedCheckpoint, ...]]:
    """Verify the five-checkpoint collection and every checkpoint/receipt pair."""

    directory = Path(checkpoint_dir).expanduser().resolve()
    collection_path = directory / "so2sat_source_checkpoint_collection.json"
    verify_artifact_receipt(collection_path)
    collection = strict_json_load(collection_path)
    if not isinstance(collection, Mapping) or collection.get("schema") != COLLECTION_SCHEMA:
        raise IntegrityError("unknown source checkpoint collection")
    if collection.get("status") != "FIVE_INDEPENDENT_SOURCE_CHECKPOINTS_VERIFIED":
        raise IntegrityError("source checkpoint collection is not complete")
    if (
        collection.get("model_seeds") != [0, 1, 2, 3, 4]
        or collection.get("all_checkpoint_tensor_hashes_distinct") is not True
        or collection.get("all_initial_tensor_hashes_distinct") is not True
        or collection.get("target_data_inputs") != []
    ):
        raise IntegrityError("source checkpoint collection independence/access contract drift")
    for field in (
        "config_sha256",
        "data_identity_sha256",
        "normalizer_sha256",
        "source_rows_sha256",
    ):
        require_sha256(collection.get(field), field=f"checkpoint_collection.{field}")
    rows = collection.get("checkpoints")
    if not isinstance(rows, list) or len(rows) != len(CHECKPOINT_IDS):
        raise IntegrityError("source checkpoint collection must contain exactly five rows")
    verified: list[VerifiedCheckpoint] = []
    for expected_id, row in zip(CHECKPOINT_IDS, sorted(rows, key=lambda value: value["model_seed"]), strict=True):
        if not isinstance(row, Mapping) or row.get("model_seed") != int(expected_id):
            raise IntegrityError("source checkpoint collection seed order drift")
        checkpoint_basename = row.get("checkpoint_basename")
        receipt_basename = row.get("training_receipt_basename")
        if (
            not isinstance(checkpoint_basename, str)
            or Path(checkpoint_basename).name != checkpoint_basename
            or not isinstance(receipt_basename, str)
            or Path(receipt_basename).name != receipt_basename
        ):
            raise IntegrityError("source checkpoint collection contains a non-basename path")
        checkpoint_path = directory / checkpoint_basename
        receipt_path = directory / receipt_basename
        verify_artifact_receipt(receipt_path)
        receipt = strict_json_load(receipt_path)
        if not isinstance(receipt, Mapping) or receipt.get("schema") != TRAINING_RECEIPT_SCHEMA:
            raise IntegrityError("unknown source training receipt")
        if receipt.get("model_seed") != int(expected_id):
            raise IntegrityError("source training receipt seed mismatch")
        checkpoint_file_hash = file_sha256(checkpoint_path)
        if checkpoint_file_hash != row.get("checkpoint_file_sha256") or checkpoint_file_hash != receipt.get(
            "checkpoint_file_sha256"
        ):
            raise IntegrityError("source checkpoint byte hash mismatch")
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise IntegrityError(f"cannot load trusted source checkpoint {checkpoint_path}: {exc}") from exc
        if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
            raise IntegrityError("unknown source checkpoint payload")
        if payload.get("architecture_id") != ARCHITECTURE_ID or payload.get("model_seed") != int(expected_id):
            raise IntegrityError("source checkpoint architecture or seed drift")
        state = payload.get("model_state")
        if not isinstance(state, Mapping):
            raise IntegrityError("source checkpoint lacks model state")
        tensor_hash = tensor_state_sha256(state)
        if (
            tensor_hash != row.get("checkpoint_tensor_sha256")
            or tensor_hash != receipt.get("checkpoint_tensor_sha256")
            or tensor_hash != payload.get("checkpoint_tensor_sha256")
        ):
            raise IntegrityError("source checkpoint tensor identity mismatch")
        receipt_data = receipt.get("data")
        receipt_config = receipt.get("config")
        if not isinstance(receipt_data, Mapping) or not isinstance(receipt_config, Mapping):
            raise IntegrityError("source training receipt data/config schema drift")
        if (
            payload.get("target_data_inputs") != []
            or receipt_data.get("target_split_pixels_read") != 0
            or receipt_data.get("target_split_labels_read") != 0
            or receipt_data.get("other_role_label_rows_read") != 0
            or receipt_config.get("target_data_inputs") != []
            or receipt.get("optimization_data_role") != "source_train"
            or receipt.get("selection_data_role") != "source_monitor"
        ):
            raise IntegrityError("source checkpoint receipt discloses target access")
        if (
            payload.get("normalizer_sha256") != collection["normalizer_sha256"]
            or receipt.get("data_identity_sha256") != collection["data_identity_sha256"]
            or payload.get("data_identity_sha256") != collection["data_identity_sha256"]
            or payload.get("source_rows_sha256") != collection["source_rows_sha256"]
        ):
            raise IntegrityError("source checkpoint differs from the sealed collection identity")
        verified.append(
            VerifiedCheckpoint(
                checkpoint_id=expected_id,
                model_seed=int(expected_id),
                checkpoint_path=checkpoint_path,
                checkpoint_file_sha256=checkpoint_file_hash,
                checkpoint_tensor_sha256=tensor_hash,
                checkpoint_payload=payload,
                training_receipt_sha256=file_sha256(receipt_path),
            )
        )
    if (
        len({row.checkpoint_tensor_sha256 for row in verified}) != 5
        or len({row.checkpoint_file_sha256 for row in verified}) != 5
    ):
        raise IntegrityError("source checkpoints are not five independent identities")
    return dict(collection), tuple(verified)


def _fixed_evaluation(
    model: nn.Module,
    data: DevelopmentData,
    partition: CityPartition,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets: list[torch.Tensor] = []

    def images() -> Iterable[torch.Tensor]:
        for batch, labels in data.evaluation_batches(partition):
            targets.append(labels)
            yield batch

    logits = fixed_model_logits(model, images(), device=device)
    if not targets:
        raise IntegrityError("development evaluation produced no targets")
    target = torch.cat(targets, dim=0)
    if logits.shape[0] != target.shape[0]:
        raise IntegrityError("development evaluation logits/targets count mismatch")
    return logits, target


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.shape[0] != targets.shape[0] or logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise IntegrityError("accuracy inputs have incompatible shapes")
    return float(logits.argmax(dim=1).eq(targets).double().mean())


def _runner_code_identity() -> dict[str, Any]:
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
        "prospective_protocol_v1.json",
        "prospective_protocol_v1.json.receipt.json",
    )
    files = {name: file_sha256(directory / name) for name in names}
    return {"files_sha256": files, "code_sha256": stable_sha256(files)}


def run_development_cell(
    *,
    candidate_id: str,
    role: str,
    partition: CityPartition,
    checkpoint: VerifiedCheckpoint,
    data: DevelopmentData,
    study_binding: Mapping[str, Any],
    device: torch.device,
    code_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Reset, adapt on west/probe, and score both policies on east/evaluation."""

    if (
        role not in DEVELOPMENT_ROLES
        or partition.role != role
        or data.authorized_role != role
    ):
        raise IntegrityError("development cell role/partition mismatch")
    validate_study_binding(study_binding)
    spec = candidate_spec(candidate_id)
    source_model = checkpoint.fresh_model(device=device)
    frozen_probe, bn_divergence = frozen_logits_and_bn_divergence(
        source_model,
        data.pixel_batches(partition, half="probe"),
        device=device,
    )
    frozen_evaluation, frozen_targets = _fixed_evaluation(source_model, data, partition, device=device)
    adapted_model, diagnostics = adapt_on_probe(
        source_model,
        data.pixel_batches(partition, half="probe"),
        candidate_id=candidate_id,
        device=device,
        batchnorm_source_statistic_divergence=bn_divergence,
    )
    adapted_probe = fixed_model_logits(
        adapted_model,
        data.pixel_batches(partition, half="probe"),
        device=device,
    )
    adapted_evaluation, adapted_targets = _fixed_evaluation(adapted_model, data, partition, device=device)
    if not torch.equal(frozen_targets, adapted_targets):
        raise IntegrityError("frozen/adapted evaluation target streams differ")
    if frozen_probe.shape != adapted_probe.shape or frozen_probe.shape[0] != len(partition.probe_rows):
        raise IntegrityError("development probe logit coverage drift")
    if frozen_evaluation.shape != adapted_evaluation.shape or frozen_evaluation.shape[0] != len(
        partition.evaluation_rows
    ):
        raise IntegrityError("development evaluation logit coverage drift")
    frozen_accuracy = _accuracy(frozen_evaluation, frozen_targets)
    adapted_accuracy = _accuracy(adapted_evaluation, adapted_targets)
    benefit = adapted_accuracy - frozen_accuracy
    feature_document = extract_label_free_features(
        frozen_probe.numpy(),
        adapted_probe.numpy(),
        normalized_adapter_update_norm=diagnostics.normalized_adapter_update_norm,
        batchnorm_source_statistic_divergence=diagnostics.batchnorm_source_statistic_divergence,
    )
    trace_id = f"{candidate_id}:{role}:{partition.city_id}:checkpoint{checkpoint.checkpoint_id}"
    trace_sha256 = trace_identity_sha256(
        role=role,
        city_id=partition.city_id,
        checkpoint_id=checkpoint.checkpoint_id,
        checkpoint_tensor_sha256=checkpoint.checkpoint_tensor_sha256,
        checkpoint_file_sha256=checkpoint.checkpoint_file_sha256,
        trace_id=trace_id,
        partition_sha256=partition.partition_sha256,
        feature_sha256=feature_document["feature_sha256"],
        manifest_sha256=study_binding["manifest_sha256"],
        population_identity_sha256=study_binding["population_identity_sha256"],
        protocol_file_sha256=study_binding["protocol_file_sha256"],
        protocol_document_sha256=study_binding["protocol_document_sha256"],
    )
    gate_row = {
        "role": role,
        "city_id": partition.city_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "checkpoint_tensor_sha256": checkpoint.checkpoint_tensor_sha256,
        "checkpoint_file_sha256": checkpoint.checkpoint_file_sha256,
        "trace_id": trace_id,
        "trace_sha256": trace_sha256,
        "partition_sha256": partition.partition_sha256,
        "manifest_sha256": study_binding["manifest_sha256"],
        "population_identity_sha256": study_binding["population_identity_sha256"],
        "protocol_file_sha256": study_binding["protocol_file_sha256"],
        "protocol_document_sha256": study_binding["protocol_document_sha256"],
        "feature_document": feature_document,
        "observed_benefit": benefit,
    }
    cell: dict[str, Any] = {
        "schema": CELL_SCHEMA,
        "status": "DEVELOPMENT_ONLY_COMPLETE",
        "candidate_id": candidate_id,
        "candidate_config_sha256": spec["candidate_config_sha256"],
        "role": role,
        "city_id": partition.city_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "probe_n": len(partition.probe_rows),
        "evaluation_n": len(partition.evaluation_rows),
        "frozen_evaluation_accuracy": frozen_accuracy,
        "adapted_evaluation_accuracy": adapted_accuracy,
        "observed_benefit": benefit,
        "adapter_diagnostics": diagnostics.document(),
        "gate_row": gate_row,
        "source_training_receipt_sha256": checkpoint.training_receipt_sha256,
        "source_normalizer_sha256": data.normalizer.normalizer_sha256,
        "source_container_identity_sha256": data.container.identity_sha256,
        "runner_code_sha256": code_identity["code_sha256"],
        "probe_labels_read": 0,
        "evaluation_label_read_passes": 2,
        "target_pixels_read": 0,
        "target_labels_read": 0,
        "target_inputs": [],
    }
    cell["cell_sha256"] = stable_sha256(cell)
    return cell


def _fit_ridge_predict(
    train_features: np.ndarray,
    train_benefits: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    means = train_features.mean(axis=0)
    scales_raw = train_features.std(axis=0, ddof=0)
    scales = np.where(scales_raw > 0.0, scales_raw, 1.0)
    x_train = (train_features - means) / scales
    design = np.column_stack([np.ones(len(x_train)), x_train])
    penalty = np.eye(design.shape[1]) * RIDGE_PENALTY
    penalty[0, 0] = 0.0
    solution = np.linalg.solve(design.T @ design + penalty, design.T @ train_benefits)
    predicted = solution[0] + ((test_features - means) / scales) @ solution[1:]
    if not np.isfinite(predicted).all():
        raise IntegrityError("leave-one-city-out ridge prediction is non-finite")
    return predicted


def candidate_feasibility(
    cells: Sequence[Mapping[str, Any]],
    *,
    study_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the locked, city-aware gate-fit-only candidate feasibility rule."""

    validate_study_binding(study_binding)
    expected = {(city, checkpoint) for city in study_binding["gate_fit_cities"] for checkpoint in CHECKPOINT_IDS}
    ordered = sorted(cells, key=lambda row: (str(row["city_id"]), str(row["checkpoint_id"])))
    observed = [(row.get("city_id"), row.get("checkpoint_id")) for row in ordered]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise IntegrityError("candidate feasibility requires exactly 9 gate-fit cities x 5 checkpoints")
    if any(row.get("role") != GATE_FIT_ROLE for row in ordered):
        raise IntegrityError("candidate feasibility may use gate_fit rows only")
    benefits = np.asarray([float(row["observed_benefit"]) for row in ordered], dtype=np.float64)
    frozen = np.asarray([float(row["frozen_evaluation_accuracy"]) for row in ordered])
    adapted = np.asarray([float(row["adapted_evaluation_accuracy"]) for row in ordered])
    if not np.isfinite(benefits).all() or not np.allclose(adapted - frozen, benefits, atol=1e-12, rtol=0):
        raise IntegrityError("candidate cell benefits are invalid or inconsistent")
    features = np.vstack([feature_vector(row["gate_row"]["feature_document"]) for row in ordered])
    cities = np.asarray([row["city_id"] for row in ordered], dtype=object)

    city_mean_benefit = {city: float(benefits[cities == city].mean()) for city in study_binding["gate_fit_cities"]}
    helpful = sorted(city for city, value in city_mean_benefit.items() if value >= MIN_CITY_MEAN_MAGNITUDE)
    harmful = sorted(city for city, value in city_mean_benefit.items() if value <= -MIN_CITY_MEAN_MAGNITUDE)
    always_freeze = float(frozen.mean())
    always_adapt = float(adapted.mean())
    best_fixed = max(always_freeze, always_adapt)
    oracle = float(np.maximum(frozen, adapted).mean())
    oracle_gap = oracle - best_fixed

    loco_predictions = np.empty(len(ordered), dtype=np.float64)
    for held_out in study_binding["gate_fit_cities"]:
        test = cities == held_out
        train = ~test
        loco_predictions[test] = _fit_ridge_predict(features[train], benefits[train], features[test])
    actions = loco_predictions > 0.0
    routed = frozen + np.where(actions, benefits, 0.0)
    routed_accuracy = float(routed.mean())
    routed_gain = routed_accuracy - best_fixed
    sign_accuracy = float(np.mean(actions == (benefits > 0.0)))
    adapt_cells = int(actions.sum())
    freeze_cells = int((~actions).sum())
    adapt_cities = sorted(set(cities[actions].tolist()))
    freeze_cities = sorted(set(cities[~actions].tolist()))

    checks = {
        "at_least_two_helpful_cities": len(helpful) >= MIN_HELPFUL_CITIES,
        "at_least_two_harmful_cities": len(harmful) >= MIN_HARMFUL_CITIES,
        "nontrivial_oracle_routing_gap": oracle_gap >= MIN_ORACLE_ROUTING_GAP,
        "loco_routed_gain_over_best_fixed": routed_gain >= MIN_LOCO_ROUTED_GAIN_OVER_BEST_FIXED,
        "loco_sign_accuracy": sign_accuracy >= MIN_LOCO_SIGN_ACCURACY,
        "meaningful_adapt_cell_exposure": adapt_cells >= MIN_ACTION_CELLS_PER_POLICY,
        "meaningful_freeze_cell_exposure": freeze_cells >= MIN_ACTION_CELLS_PER_POLICY,
        "meaningful_adapt_city_exposure": len(adapt_cities) >= MIN_ACTION_CITIES_PER_POLICY,
        "meaningful_freeze_city_exposure": len(freeze_cities) >= MIN_ACTION_CITIES_PER_POLICY,
    }
    return {
        "schema": "kbound_so2sat_candidate_feasibility_v1",
        "data_role": "gate_fit_only",
        "city_count": 9,
        "checkpoint_count": 5,
        "cell_count": 45,
        "thresholds": {
            "minimum_absolute_city_mean_benefit": MIN_CITY_MEAN_MAGNITUDE,
            "minimum_helpful_cities": MIN_HELPFUL_CITIES,
            "minimum_harmful_cities": MIN_HARMFUL_CITIES,
            "minimum_oracle_routing_gap": MIN_ORACLE_ROUTING_GAP,
            "minimum_loco_routed_gain_over_best_fixed": MIN_LOCO_ROUTED_GAIN_OVER_BEST_FIXED,
            "minimum_loco_sign_accuracy": MIN_LOCO_SIGN_ACCURACY,
            "minimum_action_cells_per_policy": MIN_ACTION_CELLS_PER_POLICY,
            "minimum_action_cities_per_policy": MIN_ACTION_CITIES_PER_POLICY,
        },
        "city_mean_benefit": city_mean_benefit,
        "helpful_cities": helpful,
        "harmful_cities": harmful,
        "always_freeze_accuracy": always_freeze,
        "always_adapt_accuracy": always_adapt,
        "best_fixed_accuracy": best_fixed,
        "oracle_cell_router_accuracy": oracle,
        "oracle_routing_gap": oracle_gap,
        "loco_routed_accuracy": routed_accuracy,
        "loco_routed_gain_over_best_fixed": routed_gain,
        "loco_sign_accuracy": sign_accuracy,
        "loco_adapt_cells": adapt_cells,
        "loco_freeze_cells": freeze_cells,
        "loco_adapt_cities": adapt_cities,
        "loco_freeze_cities": freeze_cities,
        "checks": checks,
        "feasible": all(checks.values()),
        "ranking_key": [routed_gain, oracle_gap, sign_accuracy],
    }


def validate_candidate_feasibility(document: Mapping[str, Any]) -> None:
    """Replay all declared feasibility thresholds from a compact summary."""

    expected_keys = {
        "schema",
        "data_role",
        "city_count",
        "checkpoint_count",
        "cell_count",
        "thresholds",
        "city_mean_benefit",
        "helpful_cities",
        "harmful_cities",
        "always_freeze_accuracy",
        "always_adapt_accuracy",
        "best_fixed_accuracy",
        "oracle_cell_router_accuracy",
        "oracle_routing_gap",
        "loco_routed_accuracy",
        "loco_routed_gain_over_best_fixed",
        "loco_sign_accuracy",
        "loco_adapt_cells",
        "loco_freeze_cells",
        "loco_adapt_cities",
        "loco_freeze_cities",
        "checks",
        "feasible",
        "ranking_key",
    }
    if not isinstance(document, Mapping) or set(document) != expected_keys:
        raise IntegrityError("candidate feasibility summary schema drift")
    if (
        document.get("schema") != "kbound_so2sat_candidate_feasibility_v1"
        or document.get("data_role") != "gate_fit_only"
        or document.get("city_count") != 9
        or document.get("checkpoint_count") != 5
        or document.get("cell_count") != 45
    ):
        raise IntegrityError("candidate feasibility design drift")
    thresholds = {
        "minimum_absolute_city_mean_benefit": MIN_CITY_MEAN_MAGNITUDE,
        "minimum_helpful_cities": MIN_HELPFUL_CITIES,
        "minimum_harmful_cities": MIN_HARMFUL_CITIES,
        "minimum_oracle_routing_gap": MIN_ORACLE_ROUTING_GAP,
        "minimum_loco_routed_gain_over_best_fixed": MIN_LOCO_ROUTED_GAIN_OVER_BEST_FIXED,
        "minimum_loco_sign_accuracy": MIN_LOCO_SIGN_ACCURACY,
        "minimum_action_cells_per_policy": MIN_ACTION_CELLS_PER_POLICY,
        "minimum_action_cities_per_policy": MIN_ACTION_CITIES_PER_POLICY,
    }
    if document.get("thresholds") != thresholds:
        raise IntegrityError("candidate feasibility threshold drift")
    city_means = document.get("city_mean_benefit")
    if not isinstance(city_means, Mapping) or len(city_means) != 9:
        raise IntegrityError("candidate feasibility city effects are incomplete")
    normalized_city_means: dict[str, float] = {}
    for city, value in city_means.items():
        if not isinstance(city, str) or not city:
            raise IntegrityError("candidate feasibility city id is invalid")
        normalized = float(value)
        if not math.isfinite(normalized) or not -1.0 <= normalized <= 1.0:
            raise IntegrityError("candidate feasibility city benefit is invalid")
        normalized_city_means[city] = normalized
    expected_helpful = sorted(city for city, value in normalized_city_means.items() if value >= MIN_CITY_MEAN_MAGNITUDE)
    expected_harmful = sorted(
        city for city, value in normalized_city_means.items() if value <= -MIN_CITY_MEAN_MAGNITUDE
    )
    if document.get("helpful_cities") != expected_helpful or document.get("harmful_cities") != expected_harmful:
        raise IntegrityError("candidate feasibility mixed-effect city lists drift")
    numeric_fields = (
        "always_freeze_accuracy",
        "always_adapt_accuracy",
        "best_fixed_accuracy",
        "oracle_cell_router_accuracy",
        "oracle_routing_gap",
        "loco_routed_accuracy",
        "loco_routed_gain_over_best_fixed",
        "loco_sign_accuracy",
    )
    values: dict[str, float] = {}
    for field in numeric_fields:
        value = float(document.get(field))
        if not math.isfinite(value):
            raise IntegrityError(f"candidate feasibility {field} is non-finite")
        values[field] = value
    for field in (
        "always_freeze_accuracy",
        "always_adapt_accuracy",
        "best_fixed_accuracy",
        "oracle_cell_router_accuracy",
        "loco_routed_accuracy",
        "loco_sign_accuracy",
    ):
        if not 0.0 <= values[field] <= 1.0:
            raise IntegrityError(f"candidate feasibility {field} lies outside [0,1]")
    if not math.isclose(
        values["best_fixed_accuracy"],
        max(values["always_freeze_accuracy"], values["always_adapt_accuracy"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise IntegrityError("candidate feasibility best-fixed accuracy drift")
    if not math.isclose(
        values["oracle_routing_gap"],
        values["oracle_cell_router_accuracy"] - values["best_fixed_accuracy"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) or not math.isclose(
        values["loco_routed_gain_over_best_fixed"],
        values["loco_routed_accuracy"] - values["best_fixed_accuracy"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise IntegrityError("candidate feasibility routing-gain arithmetic drift")
    adapt_cells = document.get("loco_adapt_cells")
    freeze_cells = document.get("loco_freeze_cells")
    if (
        isinstance(adapt_cells, bool)
        or not isinstance(adapt_cells, int)
        or isinstance(freeze_cells, bool)
        or not isinstance(freeze_cells, int)
        or adapt_cells < 0
        or freeze_cells < 0
        or adapt_cells + freeze_cells != 45
    ):
        raise IntegrityError("candidate feasibility routed cell counts drift")
    action_cities: dict[str, list[str]] = {}
    for field in ("loco_adapt_cities", "loco_freeze_cities"):
        cities = document.get(field)
        if (
            not isinstance(cities, list)
            or cities != sorted(cities)
            or len(cities) != len(set(cities))
            or any(city not in normalized_city_means for city in cities)
        ):
            raise IntegrityError(f"candidate feasibility {field} is invalid")
        action_cities[field] = cities
    checks = {
        "at_least_two_helpful_cities": len(expected_helpful) >= MIN_HELPFUL_CITIES,
        "at_least_two_harmful_cities": len(expected_harmful) >= MIN_HARMFUL_CITIES,
        "nontrivial_oracle_routing_gap": values["oracle_routing_gap"] >= MIN_ORACLE_ROUTING_GAP,
        "loco_routed_gain_over_best_fixed": values["loco_routed_gain_over_best_fixed"]
        >= MIN_LOCO_ROUTED_GAIN_OVER_BEST_FIXED,
        "loco_sign_accuracy": values["loco_sign_accuracy"] >= MIN_LOCO_SIGN_ACCURACY,
        "meaningful_adapt_cell_exposure": adapt_cells >= MIN_ACTION_CELLS_PER_POLICY,
        "meaningful_freeze_cell_exposure": freeze_cells >= MIN_ACTION_CELLS_PER_POLICY,
        "meaningful_adapt_city_exposure": len(action_cities["loco_adapt_cities"]) >= MIN_ACTION_CITIES_PER_POLICY,
        "meaningful_freeze_city_exposure": len(action_cities["loco_freeze_cities"]) >= MIN_ACTION_CITIES_PER_POLICY,
    }
    if document.get("checks") != checks or document.get("feasible") is not all(checks.values()):
        raise IntegrityError("candidate feasibility Boolean decision drift")
    if document.get("ranking_key") != [
        values["loco_routed_gain_over_best_fixed"],
        values["oracle_routing_gap"],
        values["loco_sign_accuracy"],
    ]:
        raise IntegrityError("candidate feasibility ranking key drift")


def build_candidate_bundle(
    *,
    candidate_id: str,
    role: str,
    cells: Sequence[Mapping[str, Any]],
    study_binding: Mapping[str, Any],
    checkpoint_collection: Mapping[str, Any],
    source_container_identity_sha256: str,
    normalizer_sha256: str,
    code_identity: Mapping[str, Any],
    development_environment: Mapping[str, Any],
) -> dict[str, Any]:
    if role not in DEVELOPMENT_ROLES:
        raise IntegrityError("unknown candidate bundle role")
    validate_development_environment_identity(development_environment)
    spec = candidate_spec(candidate_id)
    expected_cities = study_binding["gate_fit_cities"] if role == GATE_FIT_ROLE else study_binding["gate_cal_cities"]
    expected = {(city, checkpoint) for city in expected_cities for checkpoint in CHECKPOINT_IDS}
    ordered = sorted(
        (copy.deepcopy(dict(cell)) for cell in cells), key=lambda row: (row["city_id"], row["checkpoint_id"])
    )
    observed = [(cell.get("city_id"), cell.get("checkpoint_id")) for cell in ordered]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise IntegrityError("candidate bundle does not cover its exact city/checkpoint grid")
    for cell in ordered:
        if cell.get("schema") != CELL_SCHEMA or cell.get("candidate_id") != candidate_id or cell.get("role") != role:
            raise IntegrityError("candidate bundle contains an incompatible cell")
        claimed = require_sha256(cell.get("cell_sha256"), field="cell_sha256")
        unsigned = dict(cell)
        unsigned.pop("cell_sha256", None)
        if claimed != stable_sha256(unsigned):
            raise IntegrityError("development cell SHA-256 mismatch")
    bundle: dict[str, Any] = {
        "schema": CANDIDATE_BUNDLE_SCHEMA,
        "status": "DEVELOPMENT_ONLY_COMPLETE",
        "role": role,
        "candidate_spec": spec,
        "candidate_config_sha256": spec["candidate_config_sha256"],
        "study_binding": copy.deepcopy(dict(study_binding)),
        "checkpoint_collection_canonical_sha256": stable_sha256(dict(checkpoint_collection)),
        "source_container_identity_sha256": require_sha256(
            source_container_identity_sha256, field="source_container_identity_sha256"
        ),
        "normalizer_sha256": require_sha256(normalizer_sha256, field="normalizer_sha256"),
        "runner_code": copy.deepcopy(dict(code_identity)),
        "development_environment_identity": copy.deepcopy(
            dict(development_environment)
        ),
        "cells": ordered,
        "gate_rows_sha256": stable_sha256([cell["gate_row"] for cell in ordered]),
        "candidate_feasibility": (
            candidate_feasibility(ordered, study_binding=study_binding) if role == GATE_FIT_ROLE else None
        ),
        "candidate_selection_used_this_bundle": role == GATE_FIT_ROLE,
        "target_pixels_read": 0,
        "target_labels_read": 0,
        "target_inputs": [],
    }
    bundle["bundle_sha256"] = stable_sha256(bundle)
    validate_candidate_bundle(bundle, study_binding=study_binding)
    return bundle


def validate_candidate_bundle(
    bundle: Mapping[str, Any],
    *,
    study_binding: Mapping[str, Any],
) -> None:
    if not isinstance(bundle, Mapping) or set(bundle) != _CANDIDATE_BUNDLE_KEYS:
        raise IntegrityError("candidate bundle has unknown or missing fields")
    specification = bundle.get("candidate_spec")
    if not isinstance(specification, Mapping):
        raise IntegrityError("candidate bundle lacks a candidate specification")
    candidate_id = specification.get("candidate_id")
    role = bundle.get("role")
    if candidate_id not in CANDIDATE_IDS or role not in DEVELOPMENT_ROLES:
        raise IntegrityError("unknown candidate bundle identity")
    if bundle.get("schema") != CANDIDATE_BUNDLE_SCHEMA or bundle.get("status") != "DEVELOPMENT_ONLY_COMPLETE":
        raise IntegrityError("unknown or incomplete candidate bundle")
    validate_candidate_spec(specification)
    if bundle.get("candidate_config_sha256") != specification["candidate_config_sha256"]:
        raise IntegrityError("candidate bundle configuration hash mismatch")
    if bundle.get("study_binding") != dict(study_binding):
        raise IntegrityError("candidate bundle study binding mismatch")
    if (
        bundle.get("target_pixels_read") != 0
        or bundle.get("target_labels_read") != 0
        or bundle.get("target_inputs") != []
    ):
        raise IntegrityError("candidate bundle discloses target access")
    for field in (
        "checkpoint_collection_canonical_sha256",
        "source_container_identity_sha256",
        "normalizer_sha256",
        "gate_rows_sha256",
    ):
        require_sha256(bundle.get(field), field=f"candidate_bundle.{field}")
    runner_code = bundle.get("runner_code")
    if (
        not isinstance(runner_code, Mapping)
        or set(runner_code) != {"files_sha256", "code_sha256"}
        or not isinstance(runner_code.get("files_sha256"), Mapping)
        or not runner_code["files_sha256"]
    ):
        raise IntegrityError("candidate bundle runner-code identity is invalid")
    for name, digest in runner_code["files_sha256"].items():
        if not isinstance(name, str) or not name:
            raise IntegrityError("candidate bundle runner-code filename is invalid")
        require_sha256(digest, field=f"runner_code.files_sha256.{name}")
    if require_sha256(runner_code.get("code_sha256"), field="runner_code.code_sha256") != stable_sha256(
        dict(runner_code["files_sha256"])
    ):
        raise IntegrityError("candidate bundle runner-code aggregate hash mismatch")
    environment = bundle.get("development_environment_identity")
    if not isinstance(environment, Mapping):
        raise IntegrityError("candidate bundle lacks a development environment identity")
    validate_development_environment_identity(environment)

    cells = bundle.get("cells")
    expected_cities = study_binding["gate_fit_cities"] if role == GATE_FIT_ROLE else study_binding["gate_cal_cities"]
    expected_cells = {(city, checkpoint) for city in expected_cities for checkpoint in CHECKPOINT_IDS}
    if not isinstance(cells, list):
        raise IntegrityError("candidate bundle cells must be a list")
    observed_cells: list[tuple[str, str]] = []
    checkpoint_identity_by_id: dict[str, tuple[str, str, str]] = {}
    partition_by_city: dict[str, str] = {}
    parameter_names: tuple[str, ...] | None = None
    trace_ids: list[str] = []
    trace_hashes: list[str] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping) or set(cell) != _CELL_KEYS or cell.get("schema") != CELL_SCHEMA:
            raise IntegrityError(f"candidate bundle cell {index} schema drift")
        if (
            cell.get("status") != "DEVELOPMENT_ONLY_COMPLETE"
            or cell.get("candidate_id") != candidate_id
            or cell.get("candidate_config_sha256") != bundle["candidate_config_sha256"]
            or cell.get("role") != role
        ):
            raise IntegrityError(f"candidate bundle cell {index} identity drift")
        city_id = cell.get("city_id")
        checkpoint_id = cell.get("checkpoint_id")
        if not isinstance(city_id, str) or checkpoint_id not in CHECKPOINT_IDS:
            raise IntegrityError(f"candidate bundle cell {index} city/checkpoint drift")
        observed_cells.append((city_id, checkpoint_id))
        for count_field in ("probe_n", "evaluation_n"):
            count = cell.get(count_field)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise IntegrityError(f"candidate bundle cell {index} {count_field} is invalid")
        try:
            frozen = float(cell.get("frozen_evaluation_accuracy"))
            adapted = float(cell.get("adapted_evaluation_accuracy"))
            benefit = float(cell.get("observed_benefit"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise IntegrityError(f"candidate bundle cell {index} has nonnumeric metrics") from exc
        if (
            not all(math.isfinite(value) for value in (frozen, adapted, benefit))
            or not 0.0 <= frozen <= 1.0
            or not 0.0 <= adapted <= 1.0
            or not math.isclose(adapted - frozen, benefit, rel_tol=0.0, abs_tol=1.0e-12)
        ):
            raise IntegrityError(f"candidate bundle cell {index} accuracy/benefit drift")
        if (
            cell.get("probe_labels_read") != 0
            or cell.get("evaluation_label_read_passes") != 2
            or cell.get("target_pixels_read") != 0
            or cell.get("target_labels_read") != 0
            or cell.get("target_inputs") != []
        ):
            raise IntegrityError(f"candidate bundle cell {index} access contract drift")
        diagnostics = validate_adaptation_diagnostics(cell.get("adapter_diagnostics"))
        expected_probe_batches = math.ceil(int(cell["probe_n"]) / ADAPTER_BATCH_SIZE)
        if diagnostics.candidate_id != candidate_id or diagnostics.probe_batches != expected_probe_batches:
            raise IntegrityError(f"candidate bundle cell {index} adapter diagnostics drift")
        if diagnostics.optimizer_updates > diagnostics.probe_batches:
            raise IntegrityError(f"candidate bundle cell {index} has too many adapter updates")
        if candidate_id == TENT_CANDIDATE_ID:
            if (
                diagnostics.optimizer_updates != diagnostics.probe_batches
                or diagnostics.reliable_examples != cell["probe_n"]
                or diagnostics.skipped_empty_reliable_batches != 0
                or diagnostics.model_recovery_resets != 0
            ):
                raise IntegrityError(f"candidate bundle cell {index} Tent diagnostics drift")
        elif (
            diagnostics.reliable_examples > cell["probe_n"]
            or diagnostics.skipped_empty_reliable_batches != diagnostics.probe_batches - diagnostics.optimizer_updates
            or diagnostics.model_recovery_resets > diagnostics.optimizer_updates
        ):
            raise IntegrityError(f"candidate bundle cell {index} SAR diagnostics drift")
        if parameter_names is None:
            parameter_names = diagnostics.selected_parameter_names
        elif parameter_names != diagnostics.selected_parameter_names:
            raise IntegrityError("candidate bundle adapter parameter scope changes across cells")
        for field in (
            "source_training_receipt_sha256",
            "source_normalizer_sha256",
            "source_container_identity_sha256",
            "runner_code_sha256",
        ):
            require_sha256(cell.get(field), field=f"candidate_bundle.cell.{field}")
        if (
            cell["source_normalizer_sha256"] != bundle["normalizer_sha256"]
            or cell["source_container_identity_sha256"] != bundle["source_container_identity_sha256"]
            or cell["runner_code_sha256"] != runner_code["code_sha256"]
        ):
            raise IntegrityError(f"candidate bundle cell {index} provenance drift")
        gate_row = cell.get("gate_row")
        if not isinstance(gate_row, Mapping) or set(gate_row) != _GATE_ROW_KEYS:
            raise IntegrityError(f"candidate bundle cell {index} lacks its gate row")
        if (
            gate_row.get("role") != role
            or gate_row.get("city_id") != city_id
            or gate_row.get("checkpoint_id") != checkpoint_id
            or not math.isclose(
                float(gate_row.get("observed_benefit")),
                benefit,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise IntegrityError(f"candidate bundle cell {index} gate-row identity drift")
        for field in (
            "manifest_sha256",
            "population_identity_sha256",
            "protocol_file_sha256",
            "protocol_document_sha256",
        ):
            if gate_row.get(field) != study_binding[field]:
                raise IntegrityError(f"candidate bundle cell {index} gate-row binding drift")
        feature = gate_row.get("feature_document")
        if not isinstance(feature, Mapping):
            raise IntegrityError(f"candidate bundle cell {index} lacks probe features")
        validate_feature_document(feature)
        expected_trace = trace_identity_sha256(
            role=role,
            city_id=city_id,
            checkpoint_id=checkpoint_id,
            checkpoint_tensor_sha256=gate_row.get("checkpoint_tensor_sha256"),
            checkpoint_file_sha256=gate_row.get("checkpoint_file_sha256"),
            trace_id=gate_row.get("trace_id"),
            partition_sha256=gate_row.get("partition_sha256"),
            feature_sha256=feature.get("feature_sha256"),
            manifest_sha256=study_binding["manifest_sha256"],
            population_identity_sha256=study_binding["population_identity_sha256"],
            protocol_file_sha256=study_binding["protocol_file_sha256"],
            protocol_document_sha256=study_binding["protocol_document_sha256"],
        )
        if gate_row.get("trace_sha256") != expected_trace:
            raise IntegrityError(f"candidate bundle cell {index} trace hash mismatch")
        trace_ids.append(str(gate_row["trace_id"]))
        trace_hashes.append(str(gate_row["trace_sha256"]))
        checkpoint_identity = (
            str(gate_row["checkpoint_tensor_sha256"]),
            str(gate_row["checkpoint_file_sha256"]),
            str(cell["source_training_receipt_sha256"]),
        )
        prior_checkpoint = checkpoint_identity_by_id.setdefault(str(checkpoint_id), checkpoint_identity)
        if prior_checkpoint != checkpoint_identity:
            raise IntegrityError("candidate bundle checkpoint identity changes across cities")
        partition_sha = str(gate_row["partition_sha256"])
        prior_partition = partition_by_city.setdefault(city_id, partition_sha)
        if prior_partition != partition_sha:
            raise IntegrityError("candidate bundle city partition changes across checkpoints")
        claimed_cell = require_sha256(cell.get("cell_sha256"), field="cell_sha256")
        unsigned_cell = dict(cell)
        unsigned_cell.pop("cell_sha256", None)
        if claimed_cell != stable_sha256(unsigned_cell):
            raise IntegrityError(f"candidate bundle cell {index} SHA-256 mismatch")
    if len(observed_cells) != len(set(observed_cells)) or set(observed_cells) != expected_cells:
        raise IntegrityError("candidate bundle cells do not cover the exact role grid")
    if (
        set(checkpoint_identity_by_id) != set(CHECKPOINT_IDS)
        or len({identity[0] for identity in checkpoint_identity_by_id.values()}) != 5
        or len({identity[1] for identity in checkpoint_identity_by_id.values()}) != 5
        or len(set(partition_by_city.values())) != len(partition_by_city)
        or len(trace_ids) != len(set(trace_ids))
        or len(trace_hashes) != len(set(trace_hashes))
    ):
        raise IntegrityError("candidate bundle checkpoint/partition/trace identities are not unique")
    if bundle.get("gate_rows_sha256") != stable_sha256([cell["gate_row"] for cell in cells]):
        raise IntegrityError("candidate bundle gate-row aggregate hash mismatch")
    expected_feasibility = candidate_feasibility(cells, study_binding=study_binding) if role == GATE_FIT_ROLE else None
    if expected_feasibility is not None:
        validate_candidate_feasibility(expected_feasibility)
    if bundle.get("candidate_feasibility") != expected_feasibility:
        raise IntegrityError("candidate bundle feasibility does not replay from its cells")
    if bundle.get("candidate_selection_used_this_bundle") is not (role == GATE_FIT_ROLE):
        raise IntegrityError("candidate bundle selection-use disclosure drift")
    claimed = require_sha256(bundle.get("bundle_sha256"), field="bundle_sha256")
    unsigned = dict(bundle)
    unsigned.pop("bundle_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("candidate bundle SHA-256 mismatch")


def select_candidate(
    bundles: Sequence[Mapping[str, Any]],
    *,
    study_binding: Mapping[str, Any],
    source_postrun_acceptance: Mapping[str, str],
) -> dict[str, Any]:
    """Choose at most one candidate using complete gate-fit bundles only."""

    if len(bundles) != len(CANDIDATE_IDS):
        raise IntegrityError("selection requires exactly the two frozen candidate bundles")
    by_id: dict[str, Mapping[str, Any]] = {}
    for bundle in bundles:
        validate_candidate_bundle(bundle, study_binding=study_binding)
        if bundle.get("role") != GATE_FIT_ROLE:
            raise IntegrityError("candidate selection may consume gate_fit bundles only")
        candidate_id = str(bundle["candidate_spec"]["candidate_id"])
        if candidate_id in by_id:
            raise IntegrityError("candidate selection received a duplicate candidate")
        by_id[candidate_id] = bundle
    if set(by_id) != set(CANDIDATE_IDS):
        raise IntegrityError("candidate selection candidate set drift")
    gate_fit_environment = by_id[CANDIDATE_IDS[0]].get(
        "development_environment_identity"
    )
    if not isinstance(gate_fit_environment, Mapping):
        raise IntegrityError("candidate selection lacks its gate-fit environment")
    validate_development_environment_identity(gate_fit_environment)
    if any(
        bundle.get("development_environment_identity") != gate_fit_environment
        for bundle in by_id.values()
    ):
        raise IntegrityError("candidate bundles were produced in different environments")
    expected_acceptance_fields = {
        "source_postrun_acceptance_artifact_basename",
        "source_postrun_acceptance_artifact_sha256",
        "source_postrun_acceptance_canonical_document_sha256",
    }
    if not isinstance(source_postrun_acceptance, Mapping) or set(
        source_postrun_acceptance
    ) != expected_acceptance_fields:
        raise IntegrityError("candidate selection lacks the source post-run acceptance")
    for field in expected_acceptance_fields - {
        "source_postrun_acceptance_artifact_basename"
    }:
        require_sha256(source_postrun_acceptance.get(field), field=field)
    if (
        source_postrun_acceptance.get(
            "source_postrun_acceptance_artifact_basename"
        )
        != "so2sat_source_postrun_acceptance.json"
    ):
        raise IntegrityError("candidate selection source acceptance basename drift")
    feasible = [
        candidate_id for candidate_id in CANDIDATE_IDS if bool(by_id[candidate_id]["candidate_feasibility"]["feasible"])
    ]
    ranked = sorted(
        feasible,
        key=lambda candidate_id: (
            -float(by_id[candidate_id]["candidate_feasibility"]["ranking_key"][0]),
            -float(by_id[candidate_id]["candidate_feasibility"]["ranking_key"][1]),
            -float(by_id[candidate_id]["candidate_feasibility"]["ranking_key"][2]),
            candidate_id,
        ),
    )
    selected = ranked[0] if ranked else None
    document: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "status": (
            "EXACTLY_ONE_CANDIDATE_SELECTED_BEFORE_GATE_CAL"
            if selected is not None
            else "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL"
        ),
        "study_binding": copy.deepcopy(dict(study_binding)),
        "selection_data_role": "gate_fit_only",
        "candidate_ids": list(CANDIDATE_IDS),
        "candidate_summaries": {
            candidate_id: {
                "bundle_sha256": by_id[candidate_id]["bundle_sha256"],
                "candidate_config_sha256": by_id[candidate_id]["candidate_config_sha256"],
                "development_environment_identity_sha256": gate_fit_environment[
                    "environment_identity_sha256"
                ],
                "feasibility": copy.deepcopy(by_id[candidate_id]["candidate_feasibility"]),
            }
            for candidate_id in CANDIDATE_IDS
        },
        "gate_fit_environment_identity": copy.deepcopy(
            dict(gate_fit_environment)
        ),
        "ranking_rule": [
            "descending_loco_routed_gain_over_best_fixed",
            "descending_oracle_routing_gap",
            "descending_loco_sign_accuracy",
            "ascending_candidate_id",
        ],
        "selected_candidate_id": selected,
        "selected_bundle_sha256": by_id[selected]["bundle_sha256"] if selected else None,
        "source_postrun_acceptance": copy.deepcopy(
            dict(source_postrun_acceptance)
        ),
        "source_postrun_acceptance_verified_before_gate_fit_access": True,
        "gate_cal_rows_read_before_selection": 0,
        "target_pixels_read": 0,
        "target_labels_read": 0,
        "target_inputs": [],
    }
    document["selection_sha256"] = stable_sha256(document)
    validate_selection(document, study_binding=study_binding)
    return document


def validate_selection(
    document: Mapping[str, Any],
    *,
    study_binding: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema",
        "status",
        "study_binding",
        "selection_data_role",
        "candidate_ids",
        "candidate_summaries",
        "gate_fit_environment_identity",
        "ranking_rule",
        "selected_candidate_id",
        "selected_bundle_sha256",
        "source_postrun_acceptance",
        "source_postrun_acceptance_verified_before_gate_fit_access",
        "gate_cal_rows_read_before_selection",
        "target_pixels_read",
        "target_labels_read",
        "target_inputs",
        "selection_sha256",
    }
    if not isinstance(document, Mapping) or set(document) != expected_keys:
        raise IntegrityError("candidate-selection artifact schema drift")
    if document.get("schema") != SELECTION_SCHEMA or document.get("study_binding") != dict(study_binding):
        raise IntegrityError("unknown candidate-selection artifact or binding")
    if (
        document.get("selection_data_role") != "gate_fit_only"
        or document.get("gate_cal_rows_read_before_selection") != 0
        or document.get("source_postrun_acceptance_verified_before_gate_fit_access")
        is not True
    ):
        raise IntegrityError("candidate selection crossed the gate-cal boundary")
    source_acceptance = document.get("source_postrun_acceptance")
    if not isinstance(source_acceptance, Mapping) or set(source_acceptance) != {
        "source_postrun_acceptance_artifact_basename",
        "source_postrun_acceptance_artifact_sha256",
        "source_postrun_acceptance_canonical_document_sha256",
    }:
        raise IntegrityError("candidate selection source acceptance schema drift")
    if (
        source_acceptance.get("source_postrun_acceptance_artifact_basename")
        != "so2sat_source_postrun_acceptance.json"
    ):
        raise IntegrityError("candidate selection source acceptance basename drift")
    for field in (
        "source_postrun_acceptance_artifact_sha256",
        "source_postrun_acceptance_canonical_document_sha256",
    ):
        require_sha256(source_acceptance.get(field), field=field)
    if document.get("candidate_ids") != list(CANDIDATE_IDS):
        raise IntegrityError("candidate-selection candidate set drift")
    gate_fit_environment = document.get("gate_fit_environment_identity")
    if not isinstance(gate_fit_environment, Mapping):
        raise IntegrityError("candidate selection lacks its gate-fit environment")
    validate_development_environment_identity(gate_fit_environment)
    if document.get("ranking_rule") != [
        "descending_loco_routed_gain_over_best_fixed",
        "descending_oracle_routing_gap",
        "descending_loco_sign_accuracy",
        "ascending_candidate_id",
    ]:
        raise IntegrityError("candidate-selection ranking rule drift")
    summaries = document.get("candidate_summaries")
    if not isinstance(summaries, Mapping) or set(summaries) != set(CANDIDATE_IDS):
        raise IntegrityError("candidate-selection summaries are incomplete")
    feasible_ids: list[str] = []
    for candidate_id in CANDIDATE_IDS:
        summary = summaries[candidate_id]
        if not isinstance(summary, Mapping) or set(summary) != {
            "bundle_sha256",
            "candidate_config_sha256",
            "development_environment_identity_sha256",
            "feasibility",
        }:
            raise IntegrityError("candidate-selection summary schema drift")
        require_sha256(summary.get("bundle_sha256"), field="summary.bundle_sha256")
        require_sha256(
            summary.get("candidate_config_sha256"),
            field="summary.candidate_config_sha256",
        )
        if summary.get("development_environment_identity_sha256") != (
            gate_fit_environment["environment_identity_sha256"]
        ):
            raise IntegrityError("candidate-selection environment binding drift")
        if (
            summary["candidate_config_sha256"]
            != candidate_spec(candidate_id, verify_official_sources=False)["candidate_config_sha256"]
        ):
            raise IntegrityError("candidate-selection configuration hash drift")
        feasibility = summary.get("feasibility")
        if not isinstance(feasibility, Mapping) or not isinstance(feasibility.get("feasible"), bool):
            raise IntegrityError("candidate-selection feasibility summary is invalid")
        validate_candidate_feasibility(feasibility)
        ranking_key = feasibility.get("ranking_key")
        if (
            not isinstance(ranking_key, list)
            or len(ranking_key) != 3
            or not all(math.isfinite(float(value)) for value in ranking_key)
        ):
            raise IntegrityError("candidate-selection feasibility ranking key is invalid")
        if feasibility["feasible"]:
            feasible_ids.append(candidate_id)
    deterministically_ranked = sorted(
        feasible_ids,
        key=lambda candidate_id: (
            -float(summaries[candidate_id]["feasibility"]["ranking_key"][0]),
            -float(summaries[candidate_id]["feasibility"]["ranking_key"][1]),
            -float(summaries[candidate_id]["feasibility"]["ranking_key"][2]),
            candidate_id,
        ),
    )
    expected_selected = deterministically_ranked[0] if deterministically_ranked else None
    selected = document.get("selected_candidate_id")
    if selected != expected_selected:
        raise IntegrityError("candidate-selection winner does not replay from the ranking rule")
    if selected is None:
        if (
            document.get("status") != "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL"
            or document.get("selected_bundle_sha256") is not None
        ):
            raise IntegrityError("no-candidate selection status is inconsistent")
    elif selected in CANDIDATE_IDS:
        if document.get("status") != "EXACTLY_ONE_CANDIDATE_SELECTED_BEFORE_GATE_CAL":
            raise IntegrityError("selected-candidate status is inconsistent")
        if not summaries[selected]["feasibility"]["feasible"]:
            raise IntegrityError("selected adapter did not pass feasibility")
        if document.get("selected_bundle_sha256") != summaries[selected]["bundle_sha256"]:
            raise IntegrityError("selected candidate bundle identity mismatch")
    else:
        raise IntegrityError("selection names an unknown candidate")
    if (
        document.get("target_pixels_read") != 0
        or document.get("target_labels_read") != 0
        or document.get("target_inputs") != []
    ):
        raise IntegrityError("candidate selection discloses target access")
    claimed = require_sha256(document.get("selection_sha256"), field="selection_sha256")
    unsigned = dict(document)
    unsigned.pop("selection_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("candidate-selection SHA-256 mismatch")


def calibrate_selected_candidate(
    selection: Mapping[str, Any],
    fit_bundle: Mapping[str, Any],
    calibration_bundle: Mapping[str, Any],
    *,
    study_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit/calibrate the gate only after an exactly-one selection proof."""

    validate_selection(selection, study_binding=study_binding)
    selected = selection.get("selected_candidate_id")
    if selected is None:
        raise NoFeasibleCandidateError("no adapter passed gate-fit feasibility; gate-calibration access is forbidden")
    validate_candidate_bundle(fit_bundle, study_binding=study_binding)
    validate_candidate_bundle(calibration_bundle, study_binding=study_binding)
    if fit_bundle.get("role") != GATE_FIT_ROLE or calibration_bundle.get("role") != GATE_CAL_ROLE:
        raise IntegrityError("gate calibration requires gate_fit then gate_cal bundles")
    if (
        fit_bundle["candidate_spec"]["candidate_id"] != selected
        or calibration_bundle["candidate_spec"]["candidate_id"] != selected
    ):
        raise IntegrityError("gate calibration bundle candidate differs from the sealed selection")
    if fit_bundle["bundle_sha256"] != selection["selected_bundle_sha256"]:
        raise IntegrityError("gate-fit bundle differs from the sealed selected bundle")
    fit_rows = [cell["gate_row"] for cell in fit_bundle["cells"]]
    calibration_rows = [cell["gate_row"] for cell in calibration_bundle["cells"]]
    return fit_calibrate_ridge_gate(
        fit_rows,
        calibration_rows,
        study_binding=study_binding,
    )


def build_gate_authorization(
    selection: Mapping[str, Any],
    fit_bundle: Mapping[str, Any],
    calibration_bundle: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    study_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the serialized gate to its selected adapter and development chain."""

    validate_selection(selection, study_binding=study_binding)
    validate_candidate_bundle(fit_bundle, study_binding=study_binding)
    validate_candidate_bundle(calibration_bundle, study_binding=study_binding)
    validate_gate_document(gate)
    selected = selection.get("selected_candidate_id")
    if selected is None:
        raise NoFeasibleCandidateError("a no-candidate selection cannot authorize a gate")
    if (
        fit_bundle.get("role") != GATE_FIT_ROLE
        or calibration_bundle.get("role") != GATE_CAL_ROLE
        or fit_bundle["candidate_spec"]["candidate_id"] != selected
        or calibration_bundle["candidate_spec"]["candidate_id"] != selected
        or fit_bundle["bundle_sha256"] != selection["selected_bundle_sha256"]
    ):
        raise IntegrityError("gate authorization candidate/bundle chain mismatch")
    if gate.get("study_binding") != dict(study_binding):
        raise IntegrityError("gate authorization study binding mismatch")
    gate_provenance = gate.get("development_provenance")
    if not isinstance(gate_provenance, Mapping):
        raise IntegrityError("gate authorization lacks gate development provenance")
    if (
        gate_provenance.get("fit_rows_sha256") != fit_bundle["gate_rows_sha256"]
        or gate_provenance.get("calibration_rows_sha256") != calibration_bundle["gate_rows_sha256"]
    ):
        raise IntegrityError("gate rows do not match the selected development bundles")
    for field in (
        "checkpoint_collection_canonical_sha256",
        "source_container_identity_sha256",
        "normalizer_sha256",
        "runner_code",
    ):
        if fit_bundle[field] != calibration_bundle[field]:
            raise IntegrityError(f"gate authorization fit/calibration {field} mismatch")
    specification = fit_bundle["candidate_spec"]
    authorization: dict[str, Any] = {
        "schema": GATE_AUTHORIZATION_SCHEMA,
        "status": "SEALED_DEVELOPMENT_CHAIN_BEFORE_TARGET_PIXEL_ACCESS",
        "study_binding_sha256": study_binding["binding_sha256"],
        "manifest_sha256": study_binding["manifest_sha256"],
        "population_identity_sha256": study_binding["population_identity_sha256"],
        "protocol_file_sha256": study_binding["protocol_file_sha256"],
        "protocol_document_sha256": study_binding["protocol_document_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "source_postrun_acceptance_artifact_sha256": selection[
            "source_postrun_acceptance"
        ]["source_postrun_acceptance_artifact_sha256"],
        "source_postrun_acceptance_canonical_document_sha256": selection[
            "source_postrun_acceptance"
        ]["source_postrun_acceptance_canonical_document_sha256"],
        "selected_candidate_id": selected,
        "selected_candidate_config_sha256": specification["candidate_config_sha256"],
        "selected_gate_fit_bundle_sha256": fit_bundle["bundle_sha256"],
        "selected_gate_cal_bundle_sha256": calibration_bundle["bundle_sha256"],
        "gate_fit_rows_sha256": fit_bundle["gate_rows_sha256"],
        "gate_cal_rows_sha256": calibration_bundle["gate_rows_sha256"],
        "gate_sha256": gate["gate_sha256"],
        "checkpoint_collection_canonical_sha256": fit_bundle["checkpoint_collection_canonical_sha256"],
        "source_container_identity_sha256": fit_bundle["source_container_identity_sha256"],
        "normalizer_sha256": fit_bundle["normalizer_sha256"],
        "runner_code_sha256": fit_bundle["runner_code"]["code_sha256"],
        "gate_fit_candidate_count": len(CANDIDATE_IDS),
        "gate_cal_candidate_count": 1,
        "target_pixels_read": 0,
        "target_labels_read": 0,
        "target_inputs": [],
    }
    authorization["authorization_sha256"] = stable_sha256(authorization)
    validate_gate_authorization(
        authorization,
        selection=selection,
        fit_bundle=fit_bundle,
        calibration_bundle=calibration_bundle,
        gate=gate,
        study_binding=study_binding,
    )
    return authorization


def validate_gate_authorization(
    authorization: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    gate: Mapping[str, Any],
    study_binding: Mapping[str, Any],
    fit_bundle: Mapping[str, Any] | None = None,
    calibration_bundle: Mapping[str, Any] | None = None,
) -> None:
    """Validate a gate-to-selection chain, optionally replaying both bundles."""

    expected_keys = {
        "schema",
        "status",
        "study_binding_sha256",
        "manifest_sha256",
        "population_identity_sha256",
        "protocol_file_sha256",
        "protocol_document_sha256",
        "selection_sha256",
        "source_postrun_acceptance_artifact_sha256",
        "source_postrun_acceptance_canonical_document_sha256",
        "selected_candidate_id",
        "selected_candidate_config_sha256",
        "selected_gate_fit_bundle_sha256",
        "selected_gate_cal_bundle_sha256",
        "gate_fit_rows_sha256",
        "gate_cal_rows_sha256",
        "gate_sha256",
        "checkpoint_collection_canonical_sha256",
        "source_container_identity_sha256",
        "normalizer_sha256",
        "runner_code_sha256",
        "gate_fit_candidate_count",
        "gate_cal_candidate_count",
        "target_pixels_read",
        "target_labels_read",
        "target_inputs",
        "authorization_sha256",
    }
    if not isinstance(authorization, Mapping) or set(authorization) != expected_keys:
        raise IntegrityError("gate authorization has unknown or missing fields")
    if (
        authorization.get("schema") != GATE_AUTHORIZATION_SCHEMA
        or authorization.get("status") != "SEALED_DEVELOPMENT_CHAIN_BEFORE_TARGET_PIXEL_ACCESS"
        or authorization.get("gate_fit_candidate_count") != 2
        or authorization.get("gate_cal_candidate_count") != 1
        or authorization.get("target_pixels_read") != 0
        or authorization.get("target_labels_read") != 0
        or authorization.get("target_inputs") != []
    ):
        raise IntegrityError("unknown, incomplete, or target-opened gate authorization")
    for field in expected_keys - {
        "schema",
        "status",
        "selected_candidate_id",
        "gate_fit_candidate_count",
        "gate_cal_candidate_count",
        "target_pixels_read",
        "target_labels_read",
        "target_inputs",
    }:
        require_sha256(authorization.get(field), field=f"gate_authorization.{field}")
    validate_selection(selection, study_binding=study_binding)
    validate_gate_document(gate)
    selected = selection.get("selected_candidate_id")
    if selected is None:
        raise NoFeasibleCandidateError("a no-candidate selection cannot validate a gate")
    expected_binding_fields = {
        "study_binding_sha256": study_binding["binding_sha256"],
        "manifest_sha256": study_binding["manifest_sha256"],
        "population_identity_sha256": study_binding["population_identity_sha256"],
        "protocol_file_sha256": study_binding["protocol_file_sha256"],
        "protocol_document_sha256": study_binding["protocol_document_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "source_postrun_acceptance_artifact_sha256": selection[
            "source_postrun_acceptance"
        ]["source_postrun_acceptance_artifact_sha256"],
        "source_postrun_acceptance_canonical_document_sha256": selection[
            "source_postrun_acceptance"
        ]["source_postrun_acceptance_canonical_document_sha256"],
        "selected_candidate_id": selected,
        "selected_candidate_config_sha256": selection["candidate_summaries"][selected]["candidate_config_sha256"],
        "selected_gate_fit_bundle_sha256": selection["selected_bundle_sha256"],
        "gate_sha256": gate["gate_sha256"],
        "gate_fit_rows_sha256": gate["development_provenance"]["fit_rows_sha256"],
        "gate_cal_rows_sha256": gate["development_provenance"]["calibration_rows_sha256"],
    }
    for field, expected in expected_binding_fields.items():
        if authorization.get(field) != expected:
            raise IntegrityError(f"gate authorization {field} mismatch")
    if gate.get("study_binding") != dict(study_binding):
        raise IntegrityError("authorized gate study binding mismatch")
    if (fit_bundle is None) != (calibration_bundle is None):
        raise IntegrityError("gate authorization bundle replay requires both bundles")
    if fit_bundle is not None and calibration_bundle is not None:
        validate_candidate_bundle(fit_bundle, study_binding=study_binding)
        validate_candidate_bundle(calibration_bundle, study_binding=study_binding)
        bundle_fields = {
            "selected_gate_fit_bundle_sha256": fit_bundle["bundle_sha256"],
            "selected_gate_cal_bundle_sha256": calibration_bundle["bundle_sha256"],
            "gate_fit_rows_sha256": fit_bundle["gate_rows_sha256"],
            "gate_cal_rows_sha256": calibration_bundle["gate_rows_sha256"],
            "checkpoint_collection_canonical_sha256": fit_bundle["checkpoint_collection_canonical_sha256"],
            "source_container_identity_sha256": fit_bundle["source_container_identity_sha256"],
            "normalizer_sha256": fit_bundle["normalizer_sha256"],
            "runner_code_sha256": fit_bundle["runner_code"]["code_sha256"],
        }
        if (
            fit_bundle["candidate_spec"]["candidate_id"] != selected
            or calibration_bundle["candidate_spec"]["candidate_id"] != selected
        ):
            raise IntegrityError("gate authorization bundle candidate mismatch")
        for field in (
            "checkpoint_collection_canonical_sha256",
            "source_container_identity_sha256",
            "normalizer_sha256",
            "runner_code",
        ):
            if fit_bundle[field] != calibration_bundle[field]:
                raise IntegrityError(f"gate authorization bundle {field} mismatch")
        for field, expected in bundle_fields.items():
            if authorization.get(field) != expected:
                raise IntegrityError(f"gate authorization replay {field} mismatch")
    claimed = require_sha256(authorization.get("authorization_sha256"), field="authorization_sha256")
    unsigned = dict(authorization)
    unsigned.pop("authorization_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("gate authorization SHA-256 mismatch")


def load_gate_authorization_with_receipt(
    authorization_path: str | os.PathLike[str],
    *,
    selection_path: str | os.PathLike[str],
    gate_path: str | os.PathLike[str],
    population_manifest_path: str | os.PathLike[str],
    fit_bundle_path: str | os.PathLike[str] | None = None,
    calibration_bundle_path: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Receipt-verify and load the authorization, selection, and gate chain."""

    if (fit_bundle_path is None) != (calibration_bundle_path is None):
        raise IntegrityError("authorization loader requires both bundle paths or neither")
    verify_artifact_receipt(authorization_path)
    verify_artifact_receipt(selection_path)
    authorization = strict_json_load(authorization_path)
    selection = strict_json_load(selection_path)
    gate = load_gate_with_receipt(gate_path)
    if not isinstance(authorization, Mapping) or not isinstance(selection, Mapping):
        raise IntegrityError("gate authorization/selection artifacts must be JSON mappings")
    study_binding = load_study_binding(population_manifest_path)
    fit_bundle: Mapping[str, Any] | None = None
    calibration_bundle: Mapping[str, Any] | None = None
    if fit_bundle_path is not None and calibration_bundle_path is not None:
        verify_artifact_receipt(fit_bundle_path)
        verify_artifact_receipt(calibration_bundle_path)
        fit_loaded = strict_json_load(fit_bundle_path)
        calibration_loaded = strict_json_load(calibration_bundle_path)
        if not isinstance(fit_loaded, Mapping) or not isinstance(calibration_loaded, Mapping):
            raise IntegrityError("gate authorization bundles must be JSON mappings")
        fit_bundle = fit_loaded
        calibration_bundle = calibration_loaded
    validate_gate_authorization(
        authorization,
        selection=selection,
        fit_bundle=fit_bundle,
        calibration_bundle=calibration_bundle,
        gate=gate,
        study_binding=study_binding,
    )
    return dict(authorization), dict(selection), gate


def _load_manifest_and_inventory(
    manifest_path: Path,
    training_geo: Path,
) -> tuple[dict[str, Any], dict[str, Any], DevelopmentInventory]:
    study_binding = load_study_binding(manifest_path)
    manifest = strict_json_load(manifest_path)
    if not isinstance(manifest, dict):
        raise IntegrityError("population manifest must be a JSON mapping")
    geo_index = VerifiedTrainingGeoIndex(manifest, training_geo)
    inventory = build_development_inventory(
        geo_index,
        manifest,
        study_binding=study_binding,
    )
    return study_binding, manifest, inventory


def _candidate_output_paths(destination: Path) -> dict[str, Path]:
    paths = {
        candidate_id: destination / f"so2sat_{candidate_id}.gate_fit.json"
        for candidate_id in CANDIDATE_IDS
    }
    paths["selection"] = destination / "so2sat_candidate_selection.json"
    return paths


def _inspect_candidate_output_state(destination: Path) -> dict[str, bool]:
    """Inspect only the exact create-only gate-fit namespace before expensive work."""

    paths = _candidate_output_paths(destination)
    state = dict.fromkeys(paths, False)
    if not destination.exists():
        return state
    if destination.is_symlink() or not destination.is_dir():
        raise IntegrityError("candidate output destination must be a real directory")
    allowed = {
        child.name
        for path in paths.values()
        for child in (path, path.with_name(path.name + ".receipt.json"))
    }
    allowed_sidecars = {"._" + name for name in allowed}
    entries = {entry.name: entry for entry in destination.iterdir()}
    observed = set(entries)
    unknown = sorted(observed - allowed - allowed_sidecars)
    if unknown:
        raise IntegrityError(
            "candidate output directory contains unknown state: " + ", ".join(unknown)
        )
    for sidecar_name in observed & allowed_sidecars:
        sidecar = entries[sidecar_name]
        if sidecar.is_symlink() or not sidecar.is_file():
            raise IntegrityError(
                f"candidate output has an invalid AppleDouble sidecar: {sidecar_name}"
            )
    for name, path in paths.items():
        receipt_path = path.with_name(path.name + ".receipt.json")
        members = (path, receipt_path)
        flags = tuple(member.exists() or member.is_symlink() for member in members)
        if any(
            present and (member.is_symlink() or not member.is_file())
            for present, member in zip(flags, members, strict=True)
        ):
            raise IntegrityError(
                f"candidate output pair contains a non-regular member: {path.name}"
            )
        if any(flags) and not all(flags):
            raise IntegrityError(
                f"candidate output has an incomplete artifact/receipt pair: {path.name}"
            )
        state[name] = all(flags)
    if state["selection"] and not all(
        state[candidate_id] for candidate_id in CANDIDATE_IDS
    ):
        raise IntegrityError("candidate selection exists without both candidate bundles")
    return state


def _load_reusable_candidate_bundle(
    path: Path,
    *,
    candidate_id: str,
    study_binding: Mapping[str, Any],
    checkpoint_collection: Mapping[str, Any],
    source_container_identity_sha256: str,
    normalizer_sha256: str,
    code_identity: Mapping[str, Any],
    development_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Load one complete candidate pair only when every live binding still matches."""

    verify_artifact_receipt(path)
    loaded = strict_json_load(path)
    if not isinstance(loaded, Mapping):
        raise IntegrityError("reusable candidate bundle must be a JSON mapping")
    validate_candidate_bundle(loaded, study_binding=study_binding)
    if (
        loaded.get("role") != GATE_FIT_ROLE
        or loaded.get("candidate_spec", {}).get("candidate_id") != candidate_id
        or loaded.get("checkpoint_collection_canonical_sha256")
        != stable_sha256(dict(checkpoint_collection))
        or loaded.get("source_container_identity_sha256")
        != source_container_identity_sha256
        or loaded.get("normalizer_sha256") != normalizer_sha256
        or loaded.get("runner_code") != dict(code_identity)
        or loaded.get("development_environment_identity")
        != dict(development_environment)
    ):
        raise IntegrityError("reusable candidate bundle differs from the current sealed run")
    return copy.deepcopy(dict(loaded))


def _load_reusable_selection(
    path: Path,
    *,
    bundles: Sequence[Mapping[str, Any]],
    study_binding: Mapping[str, Any],
    source_postrun_acceptance: Mapping[str, str],
) -> dict[str, Any]:
    """Return an idempotent selection only when it exactly replays from both bundles."""

    verify_artifact_receipt(path)
    loaded = strict_json_load(path)
    if not isinstance(loaded, Mapping):
        raise IntegrityError("reusable candidate selection must be a JSON mapping")
    validate_selection(loaded, study_binding=study_binding)
    expected = select_candidate(
        bundles,
        study_binding=study_binding,
        source_postrun_acceptance=source_postrun_acceptance,
    )
    if loaded != expected:
        raise IntegrityError("reusable candidate selection does not replay")
    return copy.deepcopy(dict(loaded))


def run_candidate_selection(
    *,
    population_manifest: str | os.PathLike[str],
    source_postrun_acceptance_path: str | os.PathLike[str],
    source_preflight_path: str | os.PathLike[str],
    training_geo: str | os.PathLike[str],
    training_data: str | os.PathLike[str],
    checkpoint_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate both candidates on gate_fit only and seal the decision."""

    requested_destination = Path(output_dir).expanduser()
    if requested_destination.is_symlink():
        raise IntegrityError("candidate output destination must not be a symlink")
    destination = requested_destination.resolve()
    output_state = _inspect_candidate_output_state(destination)
    manifest_path = Path(population_manifest).expanduser().resolve()
    source_acceptance, source_acceptance_receipt = (
        verify_source_postrun_acceptance_bindings(
            source_postrun_acceptance_path,
            population_manifest_path=manifest_path,
            source_preflight_path=source_preflight_path,
            training_data_path=training_data,
            checkpoint_dir=checkpoint_dir,
        )
    )
    source_acceptance_binding = source_postrun_acceptance_binding(
        source_acceptance, source_acceptance_receipt
    )
    study_binding = load_study_binding(manifest_path)
    checkpoint_collection, checkpoints = load_verified_checkpoints(checkpoint_dir)
    normalizer = load_sealed_band_normalizer(
        Path(checkpoint_dir).expanduser().resolve() / "so2sat_sen2_source_normalizer.json"
    )
    if normalizer.normalizer_sha256 != checkpoint_collection.get("normalizer_sha256"):
        raise IntegrityError("checkpoint collection and normalizer identity differ")
    code_start = _runner_code_identity()
    environment_start = development_environment_identity(device)
    postrun_source = source_acceptance.get("postrun_source_container")
    if not isinstance(postrun_source, Mapping):
        raise IntegrityError("source acceptance lacks its post-run source container")
    source_container_identity = require_sha256(
        postrun_source.get("source_container_identity_sha256"),
        field="postrun_source_container.source_container_identity_sha256",
    )
    output_paths = _candidate_output_paths(destination)
    bundles_by_id: dict[str, dict[str, Any]] = {}
    for candidate_id in CANDIDATE_IDS:
        if output_state[candidate_id]:
            bundles_by_id[candidate_id] = _load_reusable_candidate_bundle(
                output_paths[candidate_id],
                candidate_id=candidate_id,
                study_binding=study_binding,
                checkpoint_collection=checkpoint_collection,
                source_container_identity_sha256=source_container_identity,
                normalizer_sha256=normalizer.normalizer_sha256,
                code_identity=code_start,
                development_environment=environment_start,
            )

    missing_candidates = [
        candidate_id
        for candidate_id in CANDIDATE_IDS
        if candidate_id not in bundles_by_id
    ]
    if missing_candidates:
        # The source chain and phase authority are fixed before constructing
        # the first object capable of opening gate-fit image/label datasets.
        inventory_binding, _, inventory = _load_manifest_and_inventory(
            manifest_path,
            Path(training_geo).expanduser().resolve(),
        )
        if inventory_binding != study_binding:
            raise IntegrityError("development inventory study binding changed")
        data = DevelopmentData(
            training_data,
            inventory,
            normalizer,
            authorized_role=GATE_FIT_ROLE,
        )
        if data.container.identity_sha256 != source_container_identity:
            raise IntegrityError("gate-fit container differs from source acceptance")
        destination.mkdir(parents=True, exist_ok=True)
        for candidate_id in missing_candidates:
            cells = []
            for city in study_binding["gate_fit_cities"]:
                partition = inventory.partitions[GATE_FIT_ROLE][city]
                for checkpoint in checkpoints:
                    cells.append(
                        run_development_cell(
                            candidate_id=candidate_id,
                            role=GATE_FIT_ROLE,
                            partition=partition,
                            checkpoint=checkpoint,
                            data=data,
                            study_binding=study_binding,
                            device=device,
                            code_identity=code_start,
                        )
                    )
                    if device.type == "mps":
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
            bundle = build_candidate_bundle(
                candidate_id=candidate_id,
                role=GATE_FIT_ROLE,
                cells=cells,
                study_binding=study_binding,
                checkpoint_collection=checkpoint_collection,
                source_container_identity_sha256=data.container.identity_sha256,
                normalizer_sha256=normalizer.normalizer_sha256,
                code_identity=code_start,
                development_environment=environment_start,
            )
            write_immutable_json_with_receipt(
                output_paths[candidate_id],
                bundle,
            )
            bundles_by_id[candidate_id] = bundle

    bundles = [bundles_by_id[candidate_id] for candidate_id in CANDIDATE_IDS]
    if (
        _runner_code_identity() != code_start
        or development_environment_identity(device) != environment_start
    ):
        raise IntegrityError(
            "So2Sat development code/environment changed during candidate selection"
        )
    published_state = _inspect_candidate_output_state(destination)
    if not all(published_state[candidate_id] for candidate_id in CANDIDATE_IDS):
        raise IntegrityError("candidate publication did not produce both complete bundles")
    if output_state["selection"]:
        if not published_state["selection"]:
            raise IntegrityError("candidate selection disappeared during verified reuse")
        return _load_reusable_selection(
            output_paths["selection"],
            bundles=bundles,
            study_binding=study_binding,
            source_postrun_acceptance=source_acceptance_binding,
        )
    if published_state["selection"]:
        raise IntegrityError("candidate selection appeared during candidate publication")
    selection = select_candidate(
        bundles,
        study_binding=study_binding,
        source_postrun_acceptance=source_acceptance_binding,
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_immutable_json_with_receipt(output_paths["selection"], selection)
    return selection


def run_gate_calibration(
    *,
    selection_path: str | os.PathLike[str],
    source_postrun_acceptance_path: str | os.PathLike[str],
    source_preflight_path: str | os.PathLike[str],
    precalibration_seal_path: str | os.PathLike[str],
    target_boundary_amendment_path: str | os.PathLike[str],
    population_manifest: str | os.PathLike[str],
    training_geo: str | os.PathLike[str],
    training_data: str | os.PathLike[str],
    checkpoint_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    device: torch.device,
) -> dict[str, Any]:
    """After selection verification, evaluate only the selected adapter on gate_cal."""

    selection_file = Path(selection_path).expanduser().resolve()
    verify_artifact_receipt(selection_file)
    selection = strict_json_load(selection_file)
    manifest_path = Path(population_manifest).expanduser().resolve()
    study_binding = load_study_binding(manifest_path)
    if not isinstance(selection, Mapping):
        raise IntegrityError("candidate selection artifact must be a JSON mapping")
    validate_selection(selection, study_binding=study_binding)
    selected = selection.get("selected_candidate_id")
    if selected is None:
        raise NoFeasibleCandidateError(
            "selection sealed no feasible candidate; refusing to enter gate calibration"
        )
    source_acceptance, source_acceptance_receipt = (
        verify_source_postrun_acceptance_bindings(
            source_postrun_acceptance_path,
            population_manifest_path=manifest_path,
            source_preflight_path=source_preflight_path,
            training_data_path=training_data,
            checkpoint_dir=checkpoint_dir,
        )
    )
    source_acceptance_binding = source_postrun_acceptance_binding(
        source_acceptance, source_acceptance_receipt
    )
    if selection["source_postrun_acceptance"] != source_acceptance_binding:
        raise IntegrityError("candidate selection binds another source acceptance")

    destination = Path(output_dir).expanduser().resolve()
    fit_path = destination / f"so2sat_{selected}.gate_fit.json"
    verify_artifact_receipt(fit_path)
    fit_bundle = strict_json_load(fit_path)
    if not isinstance(fit_bundle, Mapping):
        raise IntegrityError("selected gate-fit bundle must be a JSON mapping")
    validate_candidate_bundle(fit_bundle, study_binding=study_binding)
    if fit_bundle["bundle_sha256"] != selection["selected_bundle_sha256"]:
        raise IntegrityError("selected gate-fit bundle does not match the selection artifact")

    # A distinct execution/configuration seal must predate the first gate-cal
    # data object.  It binds the exact selection, gate-fit ridge parameters,
    # source artifacts, target opaque hashes, code, runtime, and reveal registry.
    from .precalibration_seal import (
        development_calibration_environment_identity,
        load_precalibration_seal_with_receipt,
        precalibration_code_identity,
    )
    from .target_amendment import load_target_boundary_amendment
    from .target_contract import PRODUCTION_MODE, artifact_binding

    amendment, amendment_receipt = load_target_boundary_amendment(
        target_boundary_amendment_path
    )
    checkpoint_collection, checkpoints = load_verified_checkpoints(checkpoint_dir)
    collection_path = (
        Path(checkpoint_dir).expanduser().resolve()
        / "so2sat_source_checkpoint_collection.json"
    )
    collection_receipt = verify_artifact_receipt(collection_path)
    normalizer_path = (
        Path(checkpoint_dir).expanduser().resolve()
        / "so2sat_sen2_source_normalizer.json"
    )
    normalizer = load_sealed_band_normalizer(normalizer_path)
    normalizer_receipt = verify_artifact_receipt(normalizer_path)
    precalibration_seal, _ = load_precalibration_seal_with_receipt(
        precalibration_seal_path,
        study_binding=study_binding,
        selection=selection,
        fit_bundle=fit_bundle,
        target_boundary_amendment=amendment,
        checkpoint_collection=checkpoint_collection,
    )
    if precalibration_seal["execution_mode"] != PRODUCTION_MODE:
        raise IntegrityError("gate calibration requires a PRODUCTION precalibration seal")
    if (
        precalibration_seal["population_manifest_artifact"]
        != artifact_binding(verify_artifact_receipt(manifest_path))
        or precalibration_seal["selection_artifact"]
        != artifact_binding(verify_artifact_receipt(selection_file))
        or precalibration_seal["selected_gate_fit_bundle_artifact"]
        != artifact_binding(verify_artifact_receipt(fit_path))
        or precalibration_seal["target_boundary_amendment_artifact"]
        != artifact_binding(amendment_receipt)
        or precalibration_seal["checkpoint_collection_artifact"]
        != artifact_binding(collection_receipt)
        or precalibration_seal["normalizer_artifact"]
        != artifact_binding(normalizer_receipt)
        or precalibration_seal["normalizer_sha256"]
        != normalizer.normalizer_sha256
        or precalibration_seal["source_postrun_acceptance"]
        != source_acceptance_binding
        or precalibration_seal["source_postrun_training_container"]
        != source_acceptance["postrun_source_container"]
        or precalibration_seal["source_hdf5_runtime_disclosure"]
        != source_acceptance["source_hdf5_runtime_disclosure"]
        or precalibration_seal["source_checkpoint_selection_disclosure"]
        != source_acceptance["source_checkpoint_selection_disclosure"]
        or precalibration_seal["source_initialization_clarification"]
        != source_acceptance["source_initialization_clarification"]
        or precalibration_seal[
            "gate_fit_development_environment_identity"
        ]
        != fit_bundle["development_environment_identity"]
        or precalibration_seal["package_code_identity"]
        != precalibration_code_identity()
        or precalibration_seal["development_calibration_environment_identity"]
        != development_calibration_environment_identity(device)
    ):
        raise IntegrityError("gate calibration inputs/code/runtime differ from the prior seal")

    # The selection and its exact selected gate-fit bundle are verified before
    # construction of the first object capable of opening training.h5.
    _, _, inventory = _load_manifest_and_inventory(manifest_path, Path(training_geo).expanduser().resolve())
    if normalizer.normalizer_sha256 != checkpoint_collection.get("normalizer_sha256"):
        raise IntegrityError("checkpoint collection and normalizer identity differ")
    code_start = _runner_code_identity()
    environment_start = development_environment_identity(device)
    data = DevelopmentData(
        training_data,
        inventory,
        normalizer,
        authorized_role=GATE_CAL_ROLE,
    )
    cells = []
    for city in study_binding["gate_cal_cities"]:
        partition = inventory.partitions[GATE_CAL_ROLE][city]
        for checkpoint in checkpoints:
            cells.append(
                run_development_cell(
                    candidate_id=str(selected),
                    role=GATE_CAL_ROLE,
                    partition=partition,
                    checkpoint=checkpoint,
                    data=data,
                    study_binding=study_binding,
                    device=device,
                    code_identity=code_start,
                )
            )
            if device.type == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
    calibration_bundle = build_candidate_bundle(
        candidate_id=str(selected),
        role=GATE_CAL_ROLE,
        cells=cells,
        study_binding=study_binding,
        checkpoint_collection=checkpoint_collection,
        source_container_identity_sha256=data.container.identity_sha256,
        normalizer_sha256=normalizer.normalizer_sha256,
        code_identity=code_start,
        development_environment=environment_start,
    )
    if (
        _runner_code_identity() != code_start
        or development_environment_identity(device) != environment_start
    ):
        raise IntegrityError(
            "So2Sat development code/environment changed during gate calibration"
        )
    write_immutable_json_with_receipt(destination / f"so2sat_{selected}.gate_cal.json", calibration_bundle)
    gate = calibrate_selected_candidate(
        selection,
        fit_bundle,
        calibration_bundle,
        study_binding=study_binding,
    )
    if gate["ridge"] != precalibration_seal["frozen_gate_fit_model"]:
        raise IntegrityError("calibrated gate changed the ridge model frozen before calibration")
    write_gate_with_receipt(destination / "so2sat_ridge_gate.json", gate)
    authorization = build_gate_authorization(
        selection,
        fit_bundle,
        calibration_bundle,
        gate,
        study_binding=study_binding,
    )
    write_immutable_json_with_receipt(destination / "so2sat_gate_authorization.json", authorization)
    return gate


def _device(name: str) -> torch.device:
    normalized = name.lower()
    if normalized == "auto":
        normalized = "mps" if torch.backends.mps.is_available() else "cpu"
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise IntegrityError("MPS requested but unavailable")
    if normalized not in {"mps", "cpu"}:
        raise IntegrityError("development runner supports device=auto, mps, or cpu")
    return torch.device(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("select", "calibrate"):
        subparser = subparsers.add_parser(phase)
        subparser.add_argument("--population-manifest", type=Path, required=True)
        subparser.add_argument("--training-geo", type=Path, required=True)
        subparser.add_argument("--training-data", type=Path, required=True)
        subparser.add_argument(
            "--source-postrun-acceptance", type=Path, required=True
        )
        subparser.add_argument("--source-preflight", type=Path, required=True)
        subparser.add_argument("--checkpoint-dir", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
        if phase == "calibrate":
            subparser.add_argument("--selection", type=Path, required=True)
            subparser.add_argument("--precalibration-seal", type=Path, required=True)
            subparser.add_argument(
                "--target-boundary-amendment", type=Path, required=True
            )
    args = parser.parse_args()
    if args.phase == "select":
        result = run_candidate_selection(
            population_manifest=args.population_manifest,
            source_postrun_acceptance_path=args.source_postrun_acceptance,
            source_preflight_path=args.source_preflight,
            training_geo=args.training_geo,
            training_data=args.training_data,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device=_device(args.device),
        )
        print(
            f"So2Sat candidate selection: {result['status']} selected={result['selected_candidate_id']}",
            flush=True,
        )
        if result["selected_candidate_id"] is None:
            print(
                "STOP: no feasible candidate; do not create the pre-calibration seal, "
                "open gate-calibration rows, decompress target containers, or run target stages.",
                flush=True,
            )
            raise SystemExit(NO_FEASIBLE_CANDIDATE_EXIT_CODE)
    else:
        gate = run_gate_calibration(
            selection_path=args.selection,
            source_postrun_acceptance_path=args.source_postrun_acceptance,
            source_preflight_path=args.source_preflight,
            precalibration_seal_path=args.precalibration_seal,
            target_boundary_amendment_path=args.target_boundary_amendment,
            population_manifest=args.population_manifest,
            training_geo=args.training_geo,
            training_data=args.training_data,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device=_device(args.device),
        )
        print(f"So2Sat gate calibration: PASS gate={gate['gate_sha256']}", flush=True)


if __name__ == "__main__":
    main()
