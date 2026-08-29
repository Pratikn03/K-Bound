"""Synthetic tests for the two-process So2Sat target outcome boundary."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from experiments.kbound.so2sat import target_inference, target_runner, target_seal
from experiments.kbound.so2sat.adapters import (
    SAR_CANDIDATE_ID,
    TENT_CANDIDATE_ID,
    candidate_spec,
)
from experiments.kbound.so2sat.development import (
    build_candidate_bundle,
    build_gate_authorization,
    select_candidate,
)
from experiments.kbound.so2sat.features import extract_label_free_features
from experiments.kbound.so2sat.gate import (
    CHECKPOINT_IDS,
    fit_calibrate_ridge_gate,
    load_study_binding,
    trace_identity_sha256,
    write_gate_with_receipt,
)
from experiments.kbound.so2sat.integrity import (
    IntegrityError,
    LabelFirewallError,
    file_sha256,
    stable_sha256,
    strict_json_load,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from experiments.kbound.so2sat.label_firewall import (
    LabelFreeTargetLoader,
    VerifiedGeoIndex,
)
from experiments.kbound.so2sat.metadata_manifest import GeoRecord
from experiments.kbound.so2sat.model import ARCHITECTURE_ID, tensor_state_sha256
from experiments.kbound.so2sat.precalibration_seal import (
    build_precalibration_seal,
    ensure_reveal_registry_identity,
    precalibration_code_identity,
)
from experiments.kbound.so2sat.protocol import PROTOCOL_ID
from experiments.kbound.so2sat.source_acceptance import (
    source_postrun_acceptance_binding,
)
from experiments.kbound.so2sat.source_data import (
    KNOWN_TRAINING_ROLES,
    NORMALIZER_SCHEMA,
    NORMALIZER_STATUS,
    SENTINEL2_BAND_ORDER,
    SOURCE_TRAIN_ROLE,
    BandNormalizer,
    seal_band_normalizer,
)
from experiments.kbound.so2sat.target_amendment import (
    AMENDMENT_BASENAME,
    load_target_boundary_amendment,
)
from experiments.kbound.so2sat.target_inference import TorchTargetCellExecutor
from experiments.kbound.so2sat.target_runner import (
    TEST_ONLY_MODE,
    CellComputation,
    EvaluationComputation,
    ProbeComputation,
    _run_label_blind_target_for_test,
    build_execution_seal,
)
from experiments.kbound.so2sat.target_scorer import (
    _score_sealed_target_bundle_for_test,
    target_scorer_code_identity,
    target_scorer_environment_identity,
)

FIT_CITIES = [f"fitcity{index:02d}" for index in range(9)]
CAL_CITIES = [f"calcity{index:02d}" for index in range(19)]
TARGET_CITIES = [f"targetcity{index:02d}" for index in range(10)]
SYNTHETIC_TARGET_CODE_SHA256 = stable_sha256({"code": "synthetic"})
SYNTHETIC_TARGET_ENVIRONMENT_IDENTITY = {
    "schema": "synthetic_target_environment_v1",
    "environment": "synthetic",
}
SYNTHETIC_TARGET_ENVIRONMENT_IDENTITY["environment_identity_sha256"] = stable_sha256(
    SYNTHETIC_TARGET_ENVIRONMENT_IDENTITY
)
SYNTHETIC_TARGET_ENVIRONMENT_SHA256 = SYNTHETIC_TARGET_ENVIRONMENT_IDENTITY[
    "environment_identity_sha256"
]
AMENDMENT_PATH = (
    Path(target_inference.__file__).resolve().parent / AMENDMENT_BASENAME
)


class _PixelDataset:
    def __init__(self, split: str, count: int) -> None:
        self.split = split
        self.shape = (count, 32, 32, 10)

    def __getitem__(self, index: int) -> np.ndarray:
        value = np.zeros((32, 32, 10), dtype=np.float32)
        value[0, 0, 0] = float(index)
        return value


class _PixelHandle:
    def __init__(self, split: str, count: int, requests: list[tuple[str, str]]) -> None:
        self._split = split
        self._count = count
        self._requests = requests

    def __enter__(self) -> _PixelHandle:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def keys(self) -> list[str]:
        raise AssertionError("live target handle must never be enumerated")

    def __getitem__(self, key: str) -> _PixelDataset:
        self._requests.append((self._split, key))
        if key == "label":
            raise AssertionError("live target runner requested an outcome array")
        assert key == "sen2"
        return _PixelDataset(self._split, self._count)


class _PixelFactory:
    def __init__(self, count: int, before_open: Any | None = None) -> None:
        self.count = count
        self.requests: list[tuple[str, str]] = []
        self.before_open = before_open

    def __call__(self, path: Path) -> _PixelHandle:
        if self.before_open is not None:
            self.before_open(path)
        return _PixelHandle(path.stem, self.count, self.requests)


class _SyntheticGeoIndex:
    def __init__(self, population_identity_sha256: str) -> None:
        self.population_identity_sha256 = population_identity_sha256

    def record(self, split: str, row_index: int) -> GeoRecord:
        city = TARGET_CITIES[row_index]
        role = "target_probe" if split == "validation" else "target_evaluation"
        return GeoRecord(
            sample_id=f"{split}:{row_index}",
            official_split=split,
            row_index=row_index,
            city_id=city,
            epsg=32632,
            tfw=(10.0, 0.0, 0.0, -10.0, float(row_index * 6400), 0.0),
            spatial_block_id=f"32632:{row_index}:0",
            spatial_block_easting=row_index,
            spatial_block_northing=0,
            city_role="target",
            sample_role=role,
        )

    def iter_records(self, split: str) -> Any:
        for row_index in range(10):
            yield self.record(split, row_index)


def _synthetic_cell_computation(
    checkpoint: dict[str, Any],
    probe_samples: list[Any],
    evaluation_samples: list[Any],
) -> CellComputation:
    city_index = TARGET_CITIES.index(probe_samples[0].metadata.city_id)
    checkpoint_index = int(checkpoint["checkpoint_id"])
    signal = (
        0.10 if city_index < 4 else 1.90 if city_index < 8 else 1.0
    ) + checkpoint_index / 1000.0
    frozen_probe = np.zeros((len(probe_samples), 17), dtype=np.float64)
    adapted_probe = frozen_probe.copy()
    adapted_probe[:, 1] = signal
    frozen_eval = np.zeros((len(evaluation_samples), 17), dtype=np.float64)
    frozen_eval[:, 0] = 2.0
    adapted_eval = np.zeros((len(evaluation_samples), 17), dtype=np.float64)
    adapted_eval[:, 1] = 2.0
    return CellComputation(
        frozen_probe_logits=frozen_probe,
        adapted_probe_logits=adapted_probe,
        frozen_evaluation_logits=frozen_eval,
        adapted_evaluation_logits=adapted_eval,
        normalized_adapter_update_norm=signal,
        batchnorm_source_statistic_divergence=0.1 + signal / 20.0,
    )


def _synthetic_probe_computation(
    checkpoint: dict[str, Any], probe_samples: list[Any]
) -> ProbeComputation:
    city_index = TARGET_CITIES.index(probe_samples[0].metadata.city_id)
    checkpoint_index = int(checkpoint["checkpoint_id"])
    signal = (
        0.10 if city_index < 4 else 1.90 if city_index < 8 else 1.0
    ) + checkpoint_index / 1000.0
    frozen_probe = np.zeros((len(probe_samples), 17), dtype=np.float64)
    adapted_probe = frozen_probe.copy()
    adapted_probe[:, 1] = signal
    return ProbeComputation(
        frozen_probe_logits=frozen_probe,
        adapted_probe_logits=adapted_probe,
        normalized_adapter_update_norm=signal,
        batchnorm_source_statistic_divergence=0.1 + signal / 20.0,
        opaque_evaluation_state={
            "city_id": probe_samples[0].metadata.city_id,
            "checkpoint_id": checkpoint["checkpoint_id"],
        },
    )


def _synthetic_evaluation_computation(
    probe: ProbeComputation, evaluation_samples: list[Any]
) -> EvaluationComputation:
    assert probe.opaque_evaluation_state["city_id"] == evaluation_samples[0].metadata.city_id
    frozen_eval = np.zeros((len(evaluation_samples), 17), dtype=np.float64)
    frozen_eval[:, 0] = 2.0
    adapted_eval = np.zeros((len(evaluation_samples), 17), dtype=np.float64)
    adapted_eval[:, 1] = 2.0
    return EvaluationComputation(
        frozen_evaluation_logits=frozen_eval,
        adapted_evaluation_logits=adapted_eval,
    )


class _LabelDataset:
    def __init__(self, truth: np.ndarray, tracker: dict[str, int]) -> None:
        self._truth = truth
        self._tracker = tracker
        self.shape = (len(truth), 17)

    def __getitem__(self, index: Any) -> np.ndarray:
        self._tracker["full_reads"] += 1
        assert isinstance(index, slice) and index == slice(None)
        values = np.zeros((len(self._truth), 17), dtype=np.float32)
        values[np.arange(len(self._truth)), self._truth] = 1.0
        return values


class _LabelHandle:
    def __init__(self, truth: np.ndarray, tracker: dict[str, int]) -> None:
        self._truth = truth
        self._tracker = tracker

    def __enter__(self) -> _LabelHandle:
        self._tracker["opens"] += 1
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def keys(self) -> list[str]:
        raise AssertionError("offline target handle must never be enumerated")

    def __getitem__(self, key: str) -> _LabelDataset:
        self._tracker["requests"] += 1
        assert key == "label"
        return _LabelDataset(self._truth, self._tracker)


class _LabelFactory:
    def __init__(self, truth: np.ndarray) -> None:
        self.truth = truth
        self.tracker = {"opens": 0, "requests": 0, "full_reads": 0}
        self.paths: list[Path] = []

    def __call__(self, path: Path) -> _LabelHandle:
        self.paths.append(path)
        assert path.name == "testing.h5"
        return _LabelHandle(self.truth, self.tracker)


class _InvalidLabelDataset(_LabelDataset):
    def __getitem__(self, index: Any) -> np.ndarray:
        self._tracker["full_reads"] += 1
        assert isinstance(index, slice) and index == slice(None)
        return np.zeros((len(self._truth), 17), dtype=np.float32)


class _InvalidLabelHandle(_LabelHandle):
    def __getitem__(self, key: str) -> _LabelDataset:
        self._tracker["requests"] += 1
        assert key == "label"
        return _InvalidLabelDataset(self._truth, self._tracker)


class _InvalidLabelFactory(_LabelFactory):
    def __call__(self, path: Path) -> _LabelHandle:
        self.paths.append(path)
        assert path.name == "testing.h5"
        return _InvalidLabelHandle(self.truth, self.tracker)


class _NearOneHotLabelDataset(_LabelDataset):
    def __getitem__(self, index: Any) -> np.ndarray:
        values = super().__getitem__(index).astype(np.float64)
        values[0, self._truth[0]] = 1.0 - 5.0e-7
        values[0, (int(self._truth[0]) + 1) % 17] = 5.0e-7
        return values


class _NearOneHotLabelHandle(_LabelHandle):
    def __getitem__(self, key: str) -> _LabelDataset:
        self._tracker["requests"] += 1
        assert key == "label"
        return _NearOneHotLabelDataset(self._truth, self._tracker)


class _NearOneHotLabelFactory(_LabelFactory):
    def __call__(self, path: Path) -> _LabelHandle:
        self.paths.append(path)
        assert path.name == "testing.h5"
        return _NearOneHotLabelHandle(self.truth, self.tracker)


def _manifest() -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": "kbound_so2sat_label_free_population_manifest_v1",
        "status": "LABEL_FREE_METADATA_POPULATION_VERIFIED",
        "protocol_id": PROTOCOL_ID,
        "protocol_identity": {
            "file_sha256": stable_sha256({"protocol": "file"}),
            "canonical_document_sha256": stable_sha256({"protocol": "document"}),
        },
        "splits": {
            "validation": {
                "expected_samples": 10,
                "observed_samples": 10,
                "city_counts": dict.fromkeys(TARGET_CITIES, 1),
            },
            "testing": {
                "expected_samples": 10,
                "observed_samples": 10,
                "city_counts": dict.fromkeys(TARGET_CITIES, 1),
            },
        },
        "cities": {
            "training_roles": {
                "source_fit_ineligible": ["ineligible"],
                "source_fit_core": ["core"],
                "gate_fit": FIT_CITIES,
                "gate_cal": CAL_CITIES,
            },
            "target": TARGET_CITIES,
        },
        "population_identity_sha256": stable_sha256({"population": "synthetic-target"}),
    }
    document["manifest_sha256"] = stable_sha256(document)
    return document


def _feature(signal: float, identity: str) -> dict[str, Any]:
    frozen = np.zeros((3, 17), dtype=np.float64)
    adapted = frozen.copy()
    adapted[:, 1] = signal
    return extract_label_free_features(
        frozen,
        adapted,
        normalized_adapter_update_norm=signal,
        batchnorm_source_statistic_divergence=0.1 + signal / 20.0,
    )


def _development_row(
    *,
    binding: dict[str, Any],
    checkpoint_hashes: dict[str, dict[str, str]],
    role: str,
    city: str,
    checkpoint: str,
    signal: float,
) -> dict[str, Any]:
    feature = _feature(signal, f"{role}:{city}:{checkpoint}")
    partition_sha = stable_sha256({"role": role, "city": city})
    trace_id = f"{role}:{city}:{checkpoint}"
    hashes = checkpoint_hashes[checkpoint]
    trace_sha = trace_identity_sha256(
        role=role,
        city_id=city,
        checkpoint_id=checkpoint,
        checkpoint_tensor_sha256=hashes["tensor"],
        checkpoint_file_sha256=hashes["file"],
        trace_id=trace_id,
        partition_sha256=partition_sha,
        feature_sha256=feature["feature_sha256"],
        manifest_sha256=binding["manifest_sha256"],
        population_identity_sha256=binding["population_identity_sha256"],
        protocol_file_sha256=binding["protocol_file_sha256"],
        protocol_document_sha256=binding["protocol_document_sha256"],
    )
    return {
        "role": role,
        "city_id": city,
        "checkpoint_id": checkpoint,
        "checkpoint_tensor_sha256": hashes["tensor"],
        "checkpoint_file_sha256": hashes["file"],
        "trace_id": trace_id,
        "trace_sha256": trace_sha,
        "partition_sha256": partition_sha,
        "manifest_sha256": binding["manifest_sha256"],
        "population_identity_sha256": binding["population_identity_sha256"],
        "protocol_file_sha256": binding["protocol_file_sha256"],
        "protocol_document_sha256": binding["protocol_document_sha256"],
        "feature_document": feature,
        "observed_benefit": 0.45 * (signal - 1.0),
    }


def _checkpoint_collection(
    root: Path, checkpoint_hashes: dict[str, dict[str, str]]
) -> Path:
    rows = []
    config_sha256 = stable_sha256({"config": "synthetic"})
    data_identity_sha256 = stable_sha256({"data": "synthetic"})
    normalizer_sha256 = stable_sha256({"normalizer": "synthetic"})
    source_rows_sha256 = stable_sha256({"rows": "synthetic"})
    for checkpoint in CHECKPOINT_IDS:
        seed = int(checkpoint)
        checkpoint_path = root / f"source_seed{seed}.pt"
        model_state = {"synthetic_weight": torch.tensor([float(seed + 1)])}
        checkpoint_hashes[checkpoint]["tensor"] = tensor_state_sha256(model_state)
        initial_tensor_sha256 = stable_sha256({"initial": seed})
        torch.save(
            {
                "schema": "kbound_so2sat_source_checkpoint_v1",
                "architecture_id": ARCHITECTURE_ID,
                "model_seed": seed,
                "model_state": model_state,
                "checkpoint_tensor_sha256": checkpoint_hashes[checkpoint]["tensor"],
                "initial_tensor_sha256": initial_tensor_sha256,
                "normalizer_sha256": normalizer_sha256,
                "data_identity_sha256": data_identity_sha256,
                "source_rows_sha256": source_rows_sha256,
                "target_data_inputs": [],
            },
            checkpoint_path,
        )
        checkpoint_hashes[checkpoint]["file"] = file_sha256(checkpoint_path)
        training_path = root / f"source_seed{seed}.training.json"
        training = {
            "schema": "kbound_so2sat_source_training_receipt_v1",
            "status": "SOURCE_TRAINING_COMPLETE",
            "model_seed": seed,
            "checkpoint_basename": checkpoint_path.name,
            "checkpoint_file_sha256": checkpoint_hashes[checkpoint]["file"],
            "checkpoint_tensor_sha256": checkpoint_hashes[checkpoint]["tensor"],
            "initial_tensor_sha256": initial_tensor_sha256,
            "selection_data_role": "source_monitor",
            "optimization_data_role": "source_train",
            "config": {"target_data_inputs": []},
            "data": {
                "target_split_pixels_read": 0,
                "target_split_labels_read": 0,
                "other_role_label_rows_read": 0,
            },
            "data_identity_sha256": data_identity_sha256,
        }
        write_immutable_json_with_receipt(training_path, training)
        rows.append(
            {
                "model_seed": seed,
                "checkpoint_basename": checkpoint_path.name,
                "training_receipt_basename": training_path.name,
                "checkpoint_file_sha256": checkpoint_hashes[checkpoint]["file"],
                "checkpoint_tensor_sha256": checkpoint_hashes[checkpoint]["tensor"],
                "initial_tensor_sha256": initial_tensor_sha256,
                "best_epoch": seed,
                "best_source_monitor_macro_recall": 0.2 + seed / 100.0,
                "best_source_monitor_accuracy": 0.3 + seed / 100.0,
            }
        )
    collection = {
        "schema": "kbound_so2sat_source_checkpoint_collection_v1",
        "status": "FIVE_INDEPENDENT_SOURCE_CHECKPOINTS_VERIFIED",
        "model_seeds": [0, 1, 2, 3, 4],
        "all_checkpoint_tensor_hashes_distinct": True,
        "all_initial_tensor_hashes_distinct": True,
        "config_sha256": config_sha256,
        "data_identity_sha256": data_identity_sha256,
        "normalizer_sha256": normalizer_sha256,
        "source_rows_sha256": source_rows_sha256,
        "checkpoints": rows,
        "target_data_inputs": [],
    }
    path = root / "so2sat_source_checkpoint_collection.json"
    write_immutable_json_with_receipt(path, collection)
    return path


def _candidate_bundle(
    *,
    candidate_id: str,
    role: str,
    gate_rows: list[dict[str, Any]],
    benefit_scale: float,
    binding: dict[str, Any],
    checkpoint_collection: dict[str, Any],
) -> dict[str, Any]:
    specification = candidate_spec(candidate_id)
    normalizer_sha256 = checkpoint_collection["normalizer_sha256"]
    source_container_sha256 = stable_sha256({"source_container": "synthetic"})
    code_files = {"synthetic-development.py": stable_sha256({"code": "synthetic"})}
    code_identity = {
        "files_sha256": code_files,
        "code_sha256": stable_sha256(code_files),
    }
    cells: list[dict[str, Any]] = []
    for original in gate_rows:
        gate_row = copy.deepcopy(original)
        benefit = float(gate_row["observed_benefit"]) * benefit_scale
        gate_row["observed_benefit"] = benefit
        if candidate_id != TENT_CANDIDATE_ID:
            gate_row["trace_id"] = f"{candidate_id}:{gate_row['trace_id']}"
            gate_row["trace_sha256"] = trace_identity_sha256(
                role=role,
                city_id=gate_row["city_id"],
                checkpoint_id=gate_row["checkpoint_id"],
                checkpoint_tensor_sha256=gate_row["checkpoint_tensor_sha256"],
                checkpoint_file_sha256=gate_row["checkpoint_file_sha256"],
                trace_id=gate_row["trace_id"],
                partition_sha256=gate_row["partition_sha256"],
                feature_sha256=gate_row["feature_document"]["feature_sha256"],
                manifest_sha256=binding["manifest_sha256"],
                population_identity_sha256=binding["population_identity_sha256"],
                protocol_file_sha256=binding["protocol_file_sha256"],
                protocol_document_sha256=binding["protocol_document_sha256"],
            )
        probe_n = 3
        diagnostics = {
            "candidate_id": candidate_id,
            "selected_parameter_names": ["bn.weight", "bn.bias"],
            "probe_batches": 1,
            "optimizer_updates": 1,
            "reliable_examples": probe_n,
            "skipped_empty_reliable_batches": 0,
            "model_recovery_resets": 0,
            "normalized_adapter_update_norm": 0.02,
            "batchnorm_source_statistic_divergence": 0.10,
        }
        frozen_accuracy = 0.50
        cell = {
            "schema": "kbound_so2sat_development_adapter_cell_v1",
            "status": "DEVELOPMENT_ONLY_COMPLETE",
            "candidate_id": candidate_id,
            "candidate_config_sha256": specification["candidate_config_sha256"],
            "role": role,
            "city_id": gate_row["city_id"],
            "checkpoint_id": gate_row["checkpoint_id"],
            "probe_n": probe_n,
            "evaluation_n": 100,
            "frozen_evaluation_accuracy": frozen_accuracy,
            "adapted_evaluation_accuracy": frozen_accuracy + benefit,
            "observed_benefit": benefit,
            "adapter_diagnostics": diagnostics,
            "gate_row": gate_row,
            "source_training_receipt_sha256": stable_sha256(
                {"training_receipt": gate_row["checkpoint_id"]}
            ),
            "source_normalizer_sha256": normalizer_sha256,
            "source_container_identity_sha256": source_container_sha256,
            "runner_code_sha256": code_identity["code_sha256"],
            "probe_labels_read": 0,
            "evaluation_label_read_passes": 2,
            "target_pixels_read": 0,
            "target_labels_read": 0,
            "target_inputs": [],
        }
        cell["cell_sha256"] = stable_sha256(cell)
        cells.append(cell)
    return build_candidate_bundle(
        candidate_id=candidate_id,
        role=role,
        cells=cells,
        study_binding=binding,
        checkpoint_collection=checkpoint_collection,
        source_container_identity_sha256=source_container_sha256,
        normalizer_sha256=normalizer_sha256,
        code_identity=code_identity,
    )


def _fixture(tmp_path: Path) -> dict[str, Any]:
    manifest_path = tmp_path / "population.json"
    manifest_receipt = write_immutable_json_with_receipt(manifest_path, _manifest())
    binding = load_study_binding(manifest_path)
    checkpoint_hashes = {
        checkpoint: {"tensor": stable_sha256({"tensor": checkpoint}), "file": ""}
        for checkpoint in CHECKPOINT_IDS
    }
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_collection_path = _checkpoint_collection(
        checkpoint_dir, checkpoint_hashes
    )
    fit: list[dict[str, Any]] = []
    for city_index, city in enumerate(FIT_CITIES):
        for checkpoint_index, checkpoint in enumerate(CHECKPOINT_IDS):
            signal = 0.10 + 1.80 * city_index / 8.0 + checkpoint_index / 500.0
            fit.append(
                _development_row(
                    binding=binding,
                    checkpoint_hashes=checkpoint_hashes,
                    role="gate_fit",
                    city=city,
                    checkpoint=checkpoint,
                    signal=signal,
                )
            )
    calibration: list[dict[str, Any]] = []
    for city_index, city in enumerate(CAL_CITIES):
        for checkpoint_index, checkpoint in enumerate(CHECKPOINT_IDS):
            signal = 0.11 + 1.78 * city_index / 18.0 + checkpoint_index / 600.0
            calibration.append(
                _development_row(
                    binding=binding,
                    checkpoint_hashes=checkpoint_hashes,
                    role="gate_cal",
                    city=city,
                    checkpoint=checkpoint,
                    signal=signal,
                )
            )
    gate = fit_calibrate_ridge_gate(fit, calibration, study_binding=binding)
    gate_path = tmp_path / "gate.json"
    write_gate_with_receipt(gate_path, gate)
    gate_receipt = verify_artifact_receipt(gate_path)

    collection = strict_json_load(checkpoint_collection_path)
    source_acceptance_document = {
        "schema": "kbound_so2sat_source_postrun_acceptance_v1",
        "status": "SOURCE_POSTRUN_ACCEPTED",
        "population_manifest": {
            "manifest_sha256": binding["manifest_sha256"],
            "population_identity_sha256": binding[
                "population_identity_sha256"
            ],
        },
        "checkpoint_collection": {
            "normalizer_sha256": collection["normalizer_sha256"]
        },
        "postrun_source_container": {
            "basename": "training.h5",
            "bytes": 123,
            "sha256": stable_sha256({"training": "synthetic"}),
            "source_container_identity_sha256": stable_sha256(
                {"source_container": "synthetic"}
            ),
            "matches_source_preflight": True,
            "stable_during_hash": True,
            "hdf5_datasets_opened": False,
        },
        "source_hdf5_runtime_disclosure": {
            "source_preflight_schema": "kbound_so2sat_source_data_preflight_v1",
            "source_preflight_explicit_h5py_version_recorded": False,
            "source_training_scientific_identity_schema": (
                "kbound_so2sat_source_seed_identity_v1"
            ),
            "source_training_runtime_sha256_by_model_seed": {
                str(seed): stable_sha256({"synthetic_runtime": seed})
                for seed in range(5)
            },
            "source_training_explicit_h5py_version_by_model_seed": {
                str(seed): None for seed in range(5)
            },
            "all_source_training_receipts_explicitly_record_h5py_version": False,
            "postrun_acceptance_h5py_version": "synthetic-test-version",
            "postrun_acceptance_h5py_version_is_retroactive_source_runtime_proof": False,
            "required_reporting": "synthetic HDF5 runtime disclosure",
        },
        "source_checkpoint_selection_disclosure": {
            "checkpoint_selection_primary": "macro_recall_over_supported_classes",
            "source_monitor_supported_class_count": 15,
            "source_monitor_absent_class_ids": [0, 6],
            "is_17_class_macro_recall": False,
            "development_target_endpoint": "top1_accuracy",
            "required_reporting": "synthetic test disclosure",
        },
        "source_initialization_clarification": {
            "architecture_id": ARCHITECTURE_ID,
            "legacy_architecture_spec_initialization_label": (
                "independent_torchvision_kaiming_per_model_seed"
            ),
            "residual_body_initialization": (
                "torchvision_resnet18_constructor_initialization"
            ),
            "replacement_conv1_initialization": (
                "torch.nn.Conv2d_default_reset_parameters_kaiming_uniform_a_sqrt5"
            ),
            "replacement_fc_initialization": (
                "torch.nn.Linear_default_reset_parameters"
            ),
            "exact_initial_tensor_hashes_authoritative": True,
            "initial_tensor_sha256_by_model_seed": {
                str(seed): stable_sha256({"initial": seed}) for seed in range(5)
            },
            "numerical_artifacts_changed_by_clarification": False,
        },
    }
    source_acceptance_path = tmp_path / "so2sat_source_postrun_acceptance.json"
    source_acceptance_receipt = write_immutable_json_with_receipt(
        source_acceptance_path, source_acceptance_document
    )
    source_acceptance_binding = source_postrun_acceptance_binding(
        source_acceptance_document, source_acceptance_receipt
    )
    tent_fit_bundle = _candidate_bundle(
        candidate_id=TENT_CANDIDATE_ID,
        role="gate_fit",
        gate_rows=fit,
        benefit_scale=1.0,
        binding=binding,
        checkpoint_collection=collection,
    )
    sar_fit_bundle = _candidate_bundle(
        candidate_id=SAR_CANDIDATE_ID,
        role="gate_fit",
        gate_rows=fit,
        benefit_scale=0.65,
        binding=binding,
        checkpoint_collection=collection,
    )
    selected = select_candidate(
        [tent_fit_bundle, sar_fit_bundle],
        study_binding=binding,
        source_postrun_acceptance=source_acceptance_binding,
    )
    assert selected["selected_candidate_id"] == TENT_CANDIDATE_ID
    tent_cal_bundle = _candidate_bundle(
        candidate_id=TENT_CANDIDATE_ID,
        role="gate_cal",
        gate_rows=calibration,
        benefit_scale=1.0,
        binding=binding,
        checkpoint_collection=collection,
    )
    selected_path = tmp_path / "selected.json"
    selected_receipt = write_immutable_json_with_receipt(selected_path, selected)
    fit_bundle_path = tmp_path / "selected.gate_fit.json"
    cal_bundle_path = tmp_path / "selected.gate_cal.json"
    fit_bundle_receipt = write_immutable_json_with_receipt(
        fit_bundle_path, tent_fit_bundle
    )
    write_immutable_json_with_receipt(cal_bundle_path, tent_cal_bundle)
    gate_authorization = build_gate_authorization(
        selected,
        tent_fit_bundle,
        tent_cal_bundle,
        gate,
        study_binding=binding,
    )
    gate_authorization_path = tmp_path / "gate-authorization.json"
    gate_authorization_receipt = write_immutable_json_with_receipt(
        gate_authorization_path, gate_authorization
    )
    amendment, amendment_receipt = load_target_boundary_amendment(AMENDMENT_PATH)
    collection_receipt = strict_json_load(
        checkpoint_collection_path.with_name(checkpoint_collection_path.name + ".receipt.json")
    )

    data_paths = {
        "validation": tmp_path / "validation.h5",
        "testing": tmp_path / "testing.h5",
    }
    data_paths["validation"].write_bytes(b"synthetic-validation-container")
    data_paths["testing"].write_bytes(b"synthetic-testing-container")
    data_identities = {
        split: {
            "basename": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for split, path in data_paths.items()
    }
    normalizer_artifact = tmp_path / "normalizer.synthetic.json"
    normalizer_receipt = write_immutable_json_with_receipt(
        normalizer_artifact,
        {"normalizer_sha256": collection["normalizer_sha256"]},
    )
    registry_dir = tmp_path / "outcome-registry"
    registry_dir.mkdir()
    registry_identity, registry_identity_receipt = ensure_reveal_registry_identity(
        registry_dir,
        study_binding=binding,
        selection=selected,
    )
    development_environment = {
        "schema": "synthetic_development_environment_v1",
        "environment": "synthetic",
    }
    development_environment["environment_identity_sha256"] = stable_sha256(
        development_environment
    )
    precalibration = build_precalibration_seal(
        study_binding=binding,
        manifest_receipt=manifest_receipt,
        selection=selected,
        selection_receipt=selected_receipt,
        fit_bundle=tent_fit_bundle,
        fit_bundle_receipt=fit_bundle_receipt,
        target_boundary_amendment=amendment,
        target_boundary_amendment_receipt=amendment_receipt,
        checkpoint_collection=collection,
        checkpoint_collection_receipt=collection_receipt,
        normalizer_sha256=collection["normalizer_sha256"],
        normalizer_receipt=normalizer_receipt,
        source_postrun_acceptance=source_acceptance_document,
        source_postrun_acceptance_receipt=source_acceptance_receipt,
        target_data_identities=data_identities,
        reveal_registry_identity=registry_identity,
        reveal_registry_identity_receipt=registry_identity_receipt,
        package_code_identity=precalibration_code_identity(),
        development_environment_identity=development_environment,
        target_environment_identity=SYNTHETIC_TARGET_ENVIRONMENT_IDENTITY,
        scorer_environment_identity=target_scorer_environment_identity(),
        execution_mode=TEST_ONLY_MODE,
    )
    precalibration_path = tmp_path / "precalibration-seal.json"
    precalibration_receipt = write_immutable_json_with_receipt(
        precalibration_path, precalibration
    )
    seal = build_execution_seal(
        study_binding=binding,
        selected_candidate=selected,
        selected_candidate_receipt=selected_receipt,
        selected_gate_fit_bundle=tent_fit_bundle,
        gate=gate,
        gate_receipt=gate_receipt,
        gate_authorization=gate_authorization,
        gate_authorization_receipt=gate_authorization_receipt,
        target_boundary_amendment=amendment,
        target_boundary_amendment_receipt=amendment_receipt,
        checkpoint_collection=collection,
        checkpoint_collection_receipt=collection_receipt,
        precalibration_seal=precalibration,
        precalibration_seal_receipt=precalibration_receipt,
        target_data_identities=data_identities,
        code_identity_sha256=SYNTHETIC_TARGET_CODE_SHA256,
        environment_identity_sha256=SYNTHETIC_TARGET_ENVIRONMENT_SHA256,
        scorer_code_identity_sha256=target_scorer_code_identity()[
            "code_identity_sha256"
        ],
        scorer_environment_identity_sha256=target_scorer_environment_identity()[
            "environment_identity_sha256"
        ],
        execution_mode=TEST_ONLY_MODE,
    )
    seal_path = tmp_path / "execution-seal.json"
    write_immutable_json_with_receipt(seal_path, seal)

    geo = _SyntheticGeoIndex(binding["population_identity_sha256"])
    output_dir = tmp_path / "target-bundle"
    action_timing = {"testing_city_opens_after_five_actions": 0}

    def assert_actions_exist_before_testing_pixels(path: Path) -> None:
        if path.stem != "testing":
            return
        city = TARGET_CITIES[action_timing["testing_city_opens_after_five_actions"]]
        for checkpoint in CHECKPOINT_IDS:
            action_path = output_dir / f"target_{city}_checkpoint{checkpoint}.action.json"
            verify_artifact_receipt(action_path)
        action_timing["testing_city_opens_after_five_actions"] += 1

    pixel_factory = _PixelFactory(10, assert_actions_exist_before_testing_pixels)
    loader = LabelFreeTargetLoader(
        geo,  # type: ignore[arg-type]
        data_paths,
        data_identities,
        h5_factory=pixel_factory,
        expected_split_counts={"validation": 10, "testing": 10},
    )

    class SyntheticExecutor:
        candidate_id = TENT_CANDIDATE_ID
        normalizer_sha256 = collection["normalizer_sha256"]
        code_identity_sha256 = SYNTHETIC_TARGET_CODE_SHA256
        environment_identity_sha256 = SYNTHETIC_TARGET_ENVIRONMENT_SHA256

        def prepare_probe(
            self,
            checkpoint: dict[str, Any],
            _candidate: dict[str, Any],
            probe_samples: list[Any],
        ) -> ProbeComputation:
            return _synthetic_probe_computation(checkpoint, probe_samples)

        def evaluate_after_action(
            self,
            probe_computation: ProbeComputation,
            evaluation_samples: list[Any],
        ) -> EvaluationComputation:
            return _synthetic_evaluation_computation(
                probe_computation, evaluation_samples
            )

    executor = SyntheticExecutor()

    bundle_path = _run_label_blind_target_for_test(
        execution_seal_path=seal_path,
        population_manifest_path=manifest_path,
        source_postrun_acceptance_path=source_acceptance_path,
        selected_candidate_path=selected_path,
        selected_gate_fit_bundle_path=fit_bundle_path,
        selected_gate_cal_bundle_path=cal_bundle_path,
        precalibration_seal_path=precalibration_path,
        gate_path=gate_path,
        gate_authorization_path=gate_authorization_path,
        target_boundary_amendment_path=AMENDMENT_PATH,
        checkpoint_collection_path=checkpoint_collection_path,
        checkpoint_dir=checkpoint_collection_path.parent,
        geo_index=geo,  # type: ignore[arg-type]
        target_loader=loader,
        cell_executor=executor,  # type: ignore[arg-type]
        output_dir=output_dir,
        population_manifest_validator=lambda _document: None,
    )
    return {
        "manifest": manifest_path,
        "source_acceptance": source_acceptance_path,
        "selected": selected_path,
        "fit_bundle": fit_bundle_path,
        "cal_bundle": cal_bundle_path,
        "gate": gate_path,
        "gate_authorization": gate_authorization_path,
        "amendment": AMENDMENT_PATH,
        "precalibration_seal": precalibration_path,
        "registry_dir": registry_dir,
        "collection": checkpoint_collection_path,
        "seal": seal_path,
        "data_paths": data_paths,
        "bundle": bundle_path,
        "pixel_factory": pixel_factory,
        "action_timing": action_timing,
    }


def _score(fixture: dict[str, Any], label_factory: _LabelFactory, output: Path) -> Path:
    return _score_sealed_target_bundle_for_test(
        target_bundle_path=fixture["bundle"],
        execution_seal_path=fixture["seal"],
        population_manifest_path=fixture["manifest"],
        source_postrun_acceptance_path=fixture["source_acceptance"],
        selected_candidate_path=fixture["selected"],
        selected_gate_fit_bundle_path=fixture["fit_bundle"],
        selected_gate_cal_bundle_path=fixture["cal_bundle"],
        precalibration_seal_path=fixture["precalibration_seal"],
        gate_path=fixture["gate"],
        gate_authorization_path=fixture["gate_authorization"],
        target_boundary_amendment_path=fixture["amendment"],
        checkpoint_collection_path=fixture["collection"],
        checkpoint_dir=Path(fixture["collection"]).parent,
        reveal_registry_dir=fixture["registry_dir"],
        target_data_paths=fixture["data_paths"],
        output_path=output,
        h5_factory=label_factory,
        population_manifest_validator=lambda _document: None,
    )


def test_live_runner_is_label_blind_and_offline_scorer_opens_testing_labels_once(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    pixel_factory = fixture["pixel_factory"]
    assert pixel_factory.requests
    assert {key for _, key in pixel_factory.requests} == {"sen2"}
    assert len(pixel_factory.requests) == 20
    assert fixture["action_timing"]["testing_city_opens_after_five_actions"] == 10
    bundle = strict_json_load(fixture["bundle"])
    assert bundle["cell_count"] == 50
    assert bundle["probe_labels_opened"] is False
    assert bundle["probe_labels_scored"] is False
    assert bundle["evaluation_labels_opened"] is False
    assert bundle["source_postrun_acceptance_artifact_sha256"] == bundle[
        "source_postrun_acceptance"
    ]["source_postrun_acceptance_artifact_sha256"]

    truth = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 1], dtype=np.int64)
    label_factory = _LabelFactory(truth)
    result_path = _score(fixture, label_factory, tmp_path / "score.json")
    result = strict_json_load(result_path)
    assert result["execution_mode"] == TEST_ONLY_MODE
    assert result["status"].startswith("TEST_ONLY_")
    assert result["source_postrun_acceptance"] == bundle[
        "source_postrun_acceptance"
    ]
    assert result["source_postrun_training_container"]["basename"] == "training.h5"
    assert result["source_hdf5_runtime_disclosure"] == bundle[
        "source_hdf5_runtime_disclosure"
    ]
    assert result["strong_success_checks"]["strong_success"] is False
    assert label_factory.tracker == {"opens": 1, "requests": 1, "full_reads": 1}
    assert [path.name for path in label_factory.paths] == ["testing.h5"]
    assert result["outcome_access"]["validation_container_hdf5_open_count"] == 0
    assert result["outcome_access"]["validation_probe_labels_opened"] is False
    assert len(result["cell_metrics"]) == 50
    assert set(result["inference"]["comparisons"]) == {"always_adapt", "always_freeze"}
    for comparison in result["inference"]["comparisons"].values():
        assert comparison["positive_favors"] == "KGA"
        assert comparison["sign_flip"]["permutations"] == 1024
        assert comparison["confidence_interval"]["replicates"] == 20_000
        assert 0.0 <= comparison["multiplicity"]["holm_adjusted_p_value"] <= 1.0
    adapt_comparison = result["inference"]["comparisons"]["always_adapt"]
    freeze_comparison = result["inference"]["comparisons"]["always_freeze"]
    assert adapt_comparison["point_estimate"] == pytest.approx(0.4)
    assert adapt_comparison["confidence_interval"]["lower"] == pytest.approx(0.0)
    assert adapt_comparison["confidence_interval"]["upper"] == pytest.approx(0.8)
    assert adapt_comparison["sign_flip"]["two_sided_p_value"] == pytest.approx(
        0.21875
    )
    assert adapt_comparison["multiplicity"]["holm_adjusted_p_value"] == pytest.approx(
        0.25
    )
    assert freeze_comparison["point_estimate"] == pytest.approx(0.4)
    assert freeze_comparison["confidence_interval"]["lower"] == pytest.approx(0.1)
    assert freeze_comparison["confidence_interval"]["upper"] == pytest.approx(0.7)
    assert freeze_comparison["sign_flip"]["two_sided_p_value"] == pytest.approx(0.125)
    assert freeze_comparison["multiplicity"]["holm_adjusted_p_value"] == pytest.approx(
        0.25
    )
    for metric in result["cell_metrics"]:
        assert metric["fixed_freeze_regret_minus_kga_regret"] == pytest.approx(
            metric["kga_accuracy"] - metric["frozen_accuracy"]
        )
        assert metric["fixed_adapt_regret_minus_kga_regret"] == pytest.approx(
            metric["kga_accuracy"] - metric["adapted_accuracy"]
        )
    assert result["exposure"]["decision_counts"] == {
        "ADAPT": 20,
        "FREEZE": 20,
        "ABSTAIN": 10,
    }
    assert result["exposure"][
        "both_direct_adapt_and_freeze_meaningfully_exposed"
    ] is True
    assert result["exposure"]["abstain_realized_as_freeze"] is True

    second_factory = _LabelFactory(truth)
    with pytest.raises(IntegrityError, match="already authorized|forbids reopening"):
        _score(fixture, second_factory, tmp_path / "different-output-name.json")
    assert second_factory.tracker == {"opens": 0, "requests": 0, "full_reads": 0}

    copied_bundle_root = tmp_path / "copied-target-bundle"
    shutil.copytree(Path(fixture["bundle"]).parent, copied_bundle_root)
    copied_checkpoint_root = tmp_path / "copied-checkpoints"
    shutil.copytree(Path(fixture["collection"]).parent, copied_checkpoint_root)
    copied_fixture = dict(fixture)
    copied_fixture["bundle"] = copied_bundle_root / Path(fixture["bundle"]).name
    copied_fixture["collection"] = (
        copied_checkpoint_root / Path(fixture["collection"]).name
    )
    copied_factory = _LabelFactory(truth)
    with pytest.raises(IntegrityError, match="already authorized|forbids reopening"):
        _score(copied_fixture, copied_factory, tmp_path / "copied-score.json")
    assert copied_factory.tracker == {"opens": 0, "requests": 0, "full_reads": 0}


def test_incomplete_or_tampered_bundle_fails_before_any_outcome_open(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    master = strict_json_load(fixture["bundle"])
    first_cell = Path(fixture["bundle"]).parent / master["cells"][0]["cell_basename"]
    first_cell.unlink()
    label_factory = _LabelFactory(np.zeros(10, dtype=np.int64))
    with pytest.raises(IntegrityError, match="incomplete|artifact/receipt pair"):
        _score(fixture, label_factory, tmp_path / "score-incomplete.json")
    assert label_factory.tracker == {"opens": 0, "requests": 0, "full_reads": 0}


def test_source_acceptance_tamper_fails_before_any_outcome_open(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_acceptance = Path(fixture["source_acceptance"])
    os.chmod(source_acceptance, 0o644)
    document = json.loads(source_acceptance.read_text(encoding="utf-8"))
    document["postrun_source_container"]["bytes"] += 1
    source_acceptance.write_text(json.dumps(document), encoding="utf-8")
    label_factory = _LabelFactory(np.zeros(10, dtype=np.int64))
    with pytest.raises(IntegrityError, match="receipt|byte count|SHA-256"):
        _score(fixture, label_factory, tmp_path / "score-source-tampered.json")
    assert label_factory.tracker == {"opens": 0, "requests": 0, "full_reads": 0}


def test_prediction_tamper_and_container_tamper_fail_before_outcome_open(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    master = strict_json_load(fixture["bundle"])
    first_cell = Path(fixture["bundle"]).parent / master["cells"][0]["cell_basename"]
    os.chmod(first_cell, 0o644)
    cell = json.loads(first_cell.read_text(encoding="utf-8"))
    cell["evaluation"]["frozen_prediction_class_ids"][0] = 16
    first_cell.write_text(json.dumps(cell), encoding="utf-8")
    label_factory = _LabelFactory(np.zeros(10, dtype=np.int64))
    with pytest.raises(IntegrityError, match="receipt|SHA-256|byte count"):
        _score(fixture, label_factory, tmp_path / "score-tampered.json")
    assert label_factory.tracker["opens"] == 0

    # Restore the immutable cell bytes is intentionally unsupported.  A fresh
    # fixture is required, mirroring production fail-closed behavior.
    second_root = tmp_path / "container-case"
    second_root.mkdir()
    fixture2 = _fixture(second_root)
    testing = fixture2["data_paths"]["testing"]
    testing.write_bytes(testing.read_bytes() + b"tamper")
    label_factory2 = _LabelFactory(np.zeros(10, dtype=np.int64))
    with pytest.raises(IntegrityError, match="byte count|SHA-256"):
        _score(fixture2, label_factory2, second_root / "score.json")
    assert label_factory2.tracker["opens"] == 0

    third_root = tmp_path / "logit-tamper-case"
    third_root.mkdir()
    fixture3 = _fixture(third_root)
    master3 = strict_json_load(fixture3["bundle"])
    first_cell3 = strict_json_load(
        Path(fixture3["bundle"]).parent / master3["cells"][0]["cell_basename"]
    )
    archive3 = (
        Path(fixture3["bundle"]).parent
        / first_cell3["logit_archive"]["archive_basename"]
    )
    os.chmod(archive3, 0o644)
    archive3.write_bytes(archive3.read_bytes() + b"tamper")
    label_factory3 = _LabelFactory(np.zeros(10, dtype=np.int64))
    with pytest.raises(IntegrityError, match="logit archive"):
        _score(fixture3, label_factory3, third_root / "score.json")
    assert label_factory3.tracker["opens"] == 0

    fourth_root = tmp_path / "logit-missing-case"
    fourth_root.mkdir()
    fixture4 = _fixture(fourth_root)
    master4 = strict_json_load(fixture4["bundle"])
    first_cell4 = strict_json_load(
        Path(fixture4["bundle"]).parent / master4["cells"][0]["cell_basename"]
    )
    archive4 = (
        Path(fixture4["bundle"]).parent
        / first_cell4["logit_archive"]["archive_basename"]
    )
    archive4.unlink()
    label_factory4 = _LabelFactory(np.zeros(10, dtype=np.int64))
    with pytest.raises(IntegrityError, match="logit archive"):
        _score(fixture4, label_factory4, fourth_root / "score.json")
    assert label_factory4.tracker["opens"] == 0


def test_bad_score_output_parent_fails_before_reveal(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied", encoding="utf-8")
    label_factory = _LabelFactory(np.zeros(10, dtype=np.int64))
    with pytest.raises(IntegrityError, match="output parent"):
        _score(fixture, label_factory, parent_file / "score.json")
    assert label_factory.tracker == {"opens": 0, "requests": 0, "full_reads": 0}
    assert not list(tmp_path.glob("so2sat_target_outcome_reveal_*.json"))


def test_failed_post_reveal_validation_permanently_blocks_a_second_open(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    invalid_factory = _InvalidLabelFactory(np.zeros(10, dtype=np.int64))
    with pytest.raises(IntegrityError, match="one-hot"):
        _score(fixture, invalid_factory, tmp_path / "invalid-score.json")
    assert invalid_factory.tracker == {"opens": 1, "requests": 1, "full_reads": 1}
    retry_factory = _LabelFactory(np.zeros(10, dtype=np.int64))
    with pytest.raises(IntegrityError, match="already authorized|forbids reopening"):
        _score(fixture, retry_factory, tmp_path / "retry-score.json")
    assert retry_factory.tracker == {"opens": 0, "requests": 0, "full_reads": 0}


def test_near_one_hot_testing_labels_are_rejected_exactly(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    factory = _NearOneHotLabelFactory(np.zeros(10, dtype=np.int64))
    with pytest.raises(IntegrityError, match="exact zero/one"):
        _score(fixture, factory, tmp_path / "near-one-hot-score.json")
    assert factory.tracker == {"opens": 1, "requests": 1, "full_reads": 1}


def test_offline_scorer_import_is_separate_from_live_runner_and_inference() -> None:
    command = (
        "import sys; "
        "import experiments.kbound.so2sat.target_scorer; "
        "assert 'experiments.kbound.so2sat.target_runner' not in sys.modules; "
        "assert 'experiments.kbound.so2sat.target_inference' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_target_seal_import_does_not_import_offline_scorer_process() -> None:
    command = (
        "import sys; "
        "import experiments.kbound.so2sat.target_seal; "
        "assert 'experiments.kbound.so2sat.target_scorer' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_production_core_rejects_injected_live_dependencies(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(IntegrityError, match="production target core rejects injected"):
        target_runner._run_label_blind_target_core(
            execution_seal_path=fixture["seal"],
            population_manifest_path=fixture["manifest"],
            source_postrun_acceptance_path=fixture["source_acceptance"],
            selected_candidate_path=fixture["selected"],
            selected_gate_fit_bundle_path=fixture["fit_bundle"],
            selected_gate_cal_bundle_path=fixture["cal_bundle"],
            precalibration_seal_path=fixture["precalibration_seal"],
            gate_path=fixture["gate"],
            gate_authorization_path=fixture["gate_authorization"],
            target_boundary_amendment_path=fixture["amendment"],
            checkpoint_collection_path=fixture["collection"],
            checkpoint_dir=Path(fixture["collection"]).parent,
            geo_index=object(),  # type: ignore[arg-type]
            target_loader=object(),  # type: ignore[arg-type]
            cell_executor=object(),  # type: ignore[arg-type]
            output_dir=tmp_path / "must-not-exist",
            population_manifest_validator=lambda _document: None,
            expected_execution_mode=target_runner.PRODUCTION_MODE,
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_exact_live_types_with_injected_hdf5_factories_cannot_emit_production(
    tmp_path: Path,
) -> None:
    geo = object.__new__(VerifiedGeoIndex)
    geo._uses_canonical_h5_factory = False  # type: ignore[attr-defined]
    loader = object.__new__(LabelFreeTargetLoader)
    loader._uses_canonical_h5_factory = False  # type: ignore[attr-defined]
    loader._geo_index = geo  # type: ignore[attr-defined]
    executor = object.__new__(TorchTargetCellExecutor)
    with pytest.raises(IntegrityError, match="production target core rejects injected"):
        target_runner._run_label_blind_target_core(
            execution_seal_path=tmp_path / "unused-seal.json",
            population_manifest_path=tmp_path / "unused-manifest.json",
            source_postrun_acceptance_path=tmp_path / "unused-source-acceptance.json",
            selected_candidate_path=tmp_path / "unused-selection.json",
            selected_gate_fit_bundle_path=tmp_path / "unused-fit.json",
            selected_gate_cal_bundle_path=tmp_path / "unused-cal.json",
            precalibration_seal_path=tmp_path / "unused-precal.json",
            gate_path=tmp_path / "unused-gate.json",
            gate_authorization_path=tmp_path / "unused-authorization.json",
            target_boundary_amendment_path=tmp_path / "unused-amendment.json",
            checkpoint_collection_path=tmp_path / "unused-collection.json",
            checkpoint_dir=tmp_path,
            geo_index=geo,
            target_loader=loader,
            cell_executor=executor,
            output_dir=tmp_path / "must-not-exist",
            population_manifest_validator=target_runner.validate_population_manifest,
            expected_execution_mode=target_runner.PRODUCTION_MODE,
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_precalibration_seal_predates_and_is_extended_by_target_seal(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    precalibration = strict_json_load(fixture["precalibration_seal"])
    execution = strict_json_load(fixture["seal"])
    assert precalibration["seal_creation_audit"] == {
        "created_after_gate_fit_selection": True,
        "created_before_gate_calibration": True,
        "gate_calibration_rows_opened": 0,
        "gate_calibration_labels_opened": 0,
        "target_container_hash_method": "opaque_raw_file_bytes_sha256",
        "target_hdf5_datasets_deserialized": 0,
        "target_pixels_opened": 0,
        "target_labels_opened": 0,
    }
    assert execution["precalibration_seal_sha256"] == precalibration[
        "precalibration_seal_sha256"
    ]
    assert precalibration["source_postrun_acceptance_artifact_sha256"] == (
        precalibration["source_postrun_acceptance"][
            "source_postrun_acceptance_artifact_sha256"
        ]
    )
    assert precalibration["source_postrun_training_container"]["basename"] == (
        "training.h5"
    )
    assert execution["source_postrun_acceptance"] == precalibration[
        "source_postrun_acceptance"
    ]
    assert execution["source_hdf5_runtime_disclosure"] == precalibration[
        "source_hdf5_runtime_disclosure"
    ]
    assert execution["source_checkpoint_selection_disclosure"] == precalibration[
        "source_checkpoint_selection_disclosure"
    ]
    assert execution["outcome_reveal_registry"] == precalibration[
        "outcome_reveal_registry"
    ]
    assert execution["seal_creation_audit"]["extends_precalibration_seal"] is True


def test_production_cell_executor_uses_firewall_pixels_and_fresh_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm2d(10)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(10, 17)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.fc(self.pool(self.bn(images)).flatten(1))

    normalizer_unsigned = {
        "schema": NORMALIZER_SCHEMA,
        "status": NORMALIZER_STATUS,
        "fit_role": SOURCE_TRAIN_ROLE,
        "excluded_roles": sorted(KNOWN_TRAINING_ROLES - {SOURCE_TRAIN_ROLE}),
        "method": "float64_parallel_welford_population_moments",
        "band_order": list(SENTINEL2_BAND_ORDER),
        "mean": [0.0] * 10,
        "std": [1.0] * 10,
        "source_train_n": 1,
        "source_train_pixel_n": 1024,
        "source_rows_sha256": stable_sha256({"rows": "tiny"}),
        "source_container_identity_sha256": stable_sha256({"container": "tiny"}),
    }
    normalizer = BandNormalizer(
        mean=tuple(normalizer_unsigned["mean"]),
        std=tuple(normalizer_unsigned["std"]),
        source_train_n=1,
        source_train_pixel_n=1024,
        source_rows_sha256=normalizer_unsigned["source_rows_sha256"],
        source_container_identity_sha256=normalizer_unsigned[
            "source_container_identity_sha256"
        ],
        normalizer_sha256=stable_sha256(normalizer_unsigned),
    )
    normalizer_path = tmp_path / "normalizer.json"
    seal_band_normalizer(normalizer_path, normalizer)
    torch.manual_seed(5)
    source = TinyModel()
    state = source.state_dict()
    tensor_sha256 = tensor_state_sha256(state)
    checkpoint_path = tmp_path / "tiny.pt"
    torch.save(
        {
            "schema": "kbound_so2sat_source_checkpoint_v1",
            "model_seed": 0,
            "model_state": state,
            "checkpoint_tensor_sha256": tensor_sha256,
            "normalizer_sha256": normalizer.normalizer_sha256,
            "target_data_inputs": [],
        },
        checkpoint_path,
    )
    build_calls = 0

    def build_tiny_model() -> TinyModel:
        nonlocal build_calls
        build_calls += 1
        return TinyModel()

    monkeypatch.setattr(target_inference, "build_so2sat_resnet18", build_tiny_model)
    executor = TorchTargetCellExecutor(
        candidate_id=TENT_CANDIDATE_ID,
        normalizer_path=normalizer_path,
        device=torch.device("cpu"),
    )
    checkpoint = {
        "checkpoint_id": "0",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_basename": checkpoint_path.name,
        "checkpoint_file_sha256": file_sha256(checkpoint_path),
        "checkpoint_tensor_sha256": tensor_sha256,
    }
    geo = _SyntheticGeoIndex(stable_sha256({"population": "tiny"}))
    probe = [
        type("Sample", (), {"pixels": np.zeros((32, 32, 10), dtype=np.float32), "metadata": geo.record("validation", 0)})()
    ]
    evaluation = [
        type("Sample", (), {"pixels": np.ones((32, 32, 10), dtype=np.float32), "metadata": geo.record("testing", 0)})()
    ]
    result = executor(checkpoint, candidate_spec(TENT_CANDIDATE_ID), probe, evaluation)
    repeated = executor(
        checkpoint, candidate_spec(TENT_CANDIDATE_ID), probe, evaluation
    )
    assert np.asarray(result.frozen_probe_logits).shape == (1, 17)
    assert np.asarray(result.adapted_probe_logits).shape == (1, 17)
    assert np.asarray(result.frozen_evaluation_logits).shape == (1, 17)
    assert np.asarray(result.adapted_evaluation_logits).shape == (1, 17)
    # Each cell deliberately holds an untouched frozen model and a separately
    # adapted model so the action can be sealed before either sees testing pixels.
    assert build_calls == 4
    np.testing.assert_array_equal(
        np.asarray(result.adapted_evaluation_logits),
        np.asarray(repeated.adapted_evaluation_logits),
    )
    assert len(executor.environment_identity_sha256) == 64


def test_label_firewall_rejects_a_non_target_split_before_dataset_access(
    tmp_path: Path,
) -> None:
    paths = {
        "validation": tmp_path / "validation.h5",
        "testing": tmp_path / "testing.h5",
    }
    for path in paths.values():
        path.write_bytes(b"opaque-synthetic-target")
    identities = {
        split: {
            "basename": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for split, path in paths.items()
    }
    factory = _PixelFactory(10)
    geo = _SyntheticGeoIndex(stable_sha256({"population": "firewall"}))
    loader = LabelFreeTargetLoader(
        geo,  # type: ignore[arg-type]
        paths,
        identities,
        h5_factory=factory,
        expected_split_counts={"validation": 10, "testing": 10},
    )
    loader.verify_containers()
    with pytest.raises(LabelFirewallError, match="permits only"):
        loader.read("label", 0)
    assert factory.requests == []
    records = [geo.record("validation", index) for index in range(5)]
    samples = loader.read_verified_many("validation", records)
    assert len(samples) == 5
    assert factory.requests == [("validation", "sen2")]


def test_production_cli_builds_the_firewall_and_all_50_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    collection = strict_json_load(fixture["collection"])
    cli_pixel_factory = _PixelFactory(10)
    real_loader = LabelFreeTargetLoader
    real_run = target_runner._run_label_blind_target_for_test

    def synthetic_geo_index(
        manifest: dict[str, Any], _geo_paths: dict[str, str]
    ) -> _SyntheticGeoIndex:
        return _SyntheticGeoIndex(manifest["population_identity_sha256"])

    def synthetic_loader(
        geo_index: _SyntheticGeoIndex,
        data_paths: dict[str, str],
        identities: dict[str, dict[str, Any]],
        *,
        modality: str,
    ) -> LabelFreeTargetLoader:
        assert modality == "sen2_10_band"
        return real_loader(
            geo_index,  # type: ignore[arg-type]
            data_paths,
            identities,
            modality=modality,
            h5_factory=cli_pixel_factory,
            expected_split_counts={"validation": 10, "testing": 10},
        )

    class SyntheticProductionExecutor:
        def __init__(
            self,
            *,
            candidate_id: str,
            normalizer_path: str,
            device: torch.device,
        ) -> None:
            assert normalizer_path.endswith("normalizer.synthetic.json")
            assert device.type == "cpu"
            self.candidate_id = candidate_id
            self.normalizer_sha256 = collection["normalizer_sha256"]
            self.code_identity_sha256 = SYNTHETIC_TARGET_CODE_SHA256
            self.environment_identity_sha256 = SYNTHETIC_TARGET_ENVIRONMENT_SHA256

        def prepare_probe(
            self,
            checkpoint: dict[str, Any],
            _selected_spec: dict[str, Any],
            probe_samples: list[Any],
        ) -> ProbeComputation:
            return _synthetic_probe_computation(checkpoint, probe_samples)

        def evaluate_after_action(
            self,
            probe_computation: ProbeComputation,
            evaluation_samples: list[Any],
        ) -> EvaluationComputation:
            return _synthetic_evaluation_computation(
                probe_computation, evaluation_samples
            )

    def relaxed_run(**kwargs: Any) -> Path:
        kwargs.pop("_production_authority")
        kwargs["population_manifest_validator"] = lambda _document: None
        return real_run(**kwargs)

    monkeypatch.setattr(target_runner, "validate_population_manifest", lambda _document: None)
    monkeypatch.setattr(target_runner, "VerifiedGeoIndex", synthetic_geo_index)
    monkeypatch.setattr(target_runner, "LabelFreeTargetLoader", synthetic_loader)
    monkeypatch.setattr(target_runner, "run_label_blind_target", relaxed_run)
    monkeypatch.setattr(
        target_inference, "TorchTargetCellExecutor", SyntheticProductionExecutor
    )

    output_dir = tmp_path / "cli-target-bundle"
    normalizer_path = tmp_path / "normalizer.synthetic.json"
    assert (
        target_runner.main(
            [
                "--execution-seal",
                str(fixture["seal"]),
                "--population-manifest",
                str(fixture["manifest"]),
                "--source-postrun-acceptance",
                str(fixture["source_acceptance"]),
                "--selected-candidate",
                str(fixture["selected"]),
                "--selected-gate-fit-bundle",
                str(fixture["fit_bundle"]),
                "--selected-gate-cal-bundle",
                str(fixture["cal_bundle"]),
                "--precalibration-seal",
                str(fixture["precalibration_seal"]),
                "--gate",
                str(fixture["gate"]),
                "--gate-authorization",
                str(fixture["gate_authorization"]),
                "--target-boundary-amendment",
                str(fixture["amendment"]),
                "--checkpoint-collection",
                str(fixture["collection"]),
                "--checkpoint-dir",
                str(Path(fixture["collection"]).parent),
                "--normalizer",
                str(normalizer_path),
                "--training-geo",
                str(tmp_path / "training_geo.h5"),
                "--validation-geo",
                str(tmp_path / "validation_geo.h5"),
                "--testing-geo",
                str(tmp_path / "testing_geo.h5"),
                "--validation-data",
                str(fixture["data_paths"]["validation"]),
                "--testing-data",
                str(fixture["data_paths"]["testing"]),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    bundle_path = output_dir / "so2sat_target_bundle.json"
    assert strict_json_load(bundle_path)["cell_count"] == 50
    assert {key for _, key in cli_pixel_factory.requests} == {"sen2"}
    assert str(bundle_path) in capsys.readouterr().out


def test_seal_cli_writes_a_receipted_test_only_seal_without_hdf5_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)

    def synthetic_seal(**kwargs: Any) -> Path:
        assert kwargs.pop("device_name") == "cpu"
        assert str(kwargs.pop("normalizer_path")).endswith("normalizer.synthetic.json")
        collection = strict_json_load(kwargs["checkpoint_collection_path"])
        return target_seal._create_execution_seal_for_test(
            **kwargs,
            live_code_identity_sha256=SYNTHETIC_TARGET_CODE_SHA256,
            live_environment_identity_sha256=SYNTHETIC_TARGET_ENVIRONMENT_SHA256,
            live_normalizer_sha256=collection["normalizer_sha256"],
            population_manifest_validator=lambda _document: None,
        )

    monkeypatch.setattr(target_seal, "create_production_execution_seal", synthetic_seal)
    output = tmp_path / "cli-execution-seal.json"
    assert (
        target_seal.main(
            [
                "--population-manifest",
                str(fixture["manifest"]),
                "--source-postrun-acceptance",
                str(fixture["source_acceptance"]),
                "--selected-candidate",
                str(fixture["selected"]),
                "--selected-gate-fit-bundle",
                str(fixture["fit_bundle"]),
                "--selected-gate-cal-bundle",
                str(fixture["cal_bundle"]),
                "--precalibration-seal",
                str(fixture["precalibration_seal"]),
                "--gate",
                str(fixture["gate"]),
                "--gate-authorization",
                str(fixture["gate_authorization"]),
                "--target-boundary-amendment",
                str(fixture["amendment"]),
                "--checkpoint-collection",
                str(fixture["collection"]),
                "--checkpoint-dir",
                str(Path(fixture["collection"]).parent),
                "--reveal-registry-dir",
                str(fixture["registry_dir"]),
                "--normalizer",
                str(tmp_path / "normalizer.synthetic.json"),
                "--validation-data",
                str(fixture["data_paths"]["validation"]),
                "--testing-data",
                str(fixture["data_paths"]["testing"]),
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    created = strict_json_load(output)
    assert created["execution_mode"] == TEST_ONLY_MODE
    assert created["status"].startswith("TEST_ONLY_")
    assert verify_artifact_receipt(output)["schema"].endswith("_v2")
    assert str(output) in capsys.readouterr().out
