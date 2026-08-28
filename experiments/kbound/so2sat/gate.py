"""Sealed ridge gate and city-level split-conformal calibration for So2Sat.

This module consumes already-produced label-free probe feature documents plus
development-only benefit values.  It cannot load datasets or score target
outcomes.  The design is fixed at nine gate-fit cities, nineteen gate-
calibration cities, and five independently trained checkpoints per city.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_NAMES, N_CLASSES, feature_vector, validate_feature_document
from .integrity import (
    ARTIFACT_RECEIPT_SCHEMA_V1,
    ARTIFACT_RECEIPT_SCHEMA_V2,
    IntegrityError,
    require_sha256,
    stable_sha256,
    strict_json_load,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from .protocol import PROTOCOL_ID

RIDGE_PENALTY = 10.0
CALIBRATION_ALPHA = 0.10
CHECKPOINT_IDS = tuple(str(seed) for seed in range(5))
FIT_CITY_COUNT = 9
CALIBRATION_CITY_COUNT = 19
TARGET_CITY_COUNT = 10
FIT_TRACE_COUNT = FIT_CITY_COUNT * len(CHECKPOINT_IDS)
CALIBRATION_TRACE_COUNT = CALIBRATION_CITY_COUNT * len(CHECKPOINT_IDS)
CONFORMAL_RANK = 18
DECISIONS = ("ADAPT", "FREEZE", "ABSTAIN")

STUDY_BINDING_SCHEMA = "kbound_so2sat_gate_study_binding_v1"
GATE_SCHEMA = "kbound_so2sat_ridge_gate_v1"
ACTION_SCHEMA = "kbound_so2sat_label_free_action_v1"

_STUDY_BINDING_KEYS = {
    "schema",
    "status",
    "protocol_id",
    "manifest_artifact_sha256",
    "manifest_canonical_document_sha256",
    "manifest_sha256",
    "population_identity_sha256",
    "protocol_file_sha256",
    "protocol_document_sha256",
    "gate_fit_cities",
    "gate_cal_cities",
    "target_cities",
    "binding_sha256",
}
_DEVELOPMENT_ROW_KEYS = {
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
    "city_id",
    "checkpoint_id",
    "trace_id",
    "trace_sha256",
    "partition_sha256",
    "feature_sha256",
    "frozen_logits_tensor_sha256",
    "adapted_logits_tensor_sha256",
}
_ACTION_KEYS = {
    "schema",
    "status",
    "gate_sha256",
    "study_binding_sha256",
    "manifest_sha256",
    "population_identity_sha256",
    "protocol_file_sha256",
    "protocol_document_sha256",
    "city_id",
    "checkpoint_id",
    "checkpoint_tensor_sha256",
    "checkpoint_file_sha256",
    "trace_id",
    "trace_sha256",
    "partition_sha256",
    "feature_document",
    "support_status",
    "delta_hat",
    "epsilon",
    "lower",
    "upper",
    "decision",
    "realized_action",
    "action_sha256",
}


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise IntegrityError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise IntegrityError(f"{field} must be a finite number")
    return result


def _sorted_unique_strings(value: Any, *, field: str, count: int) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(not isinstance(item, str) or not item for item in value)
        or value != sorted(value)
        or len(set(value)) != len(value)
    ):
        raise IntegrityError(f"{field} must be {count} sorted unique non-empty strings")
    return list(value)


def build_study_binding(
    population_manifest: Mapping[str, Any],
    manifest_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract and seal only the manifest identities needed by the gate.

    File-backed callers should normally use :func:`load_study_binding`, which
    verifies the receipt against the artifact bytes before calling this pure
    constructor.
    """

    if not isinstance(population_manifest, Mapping):
        raise IntegrityError("population manifest must be a mapping")
    if population_manifest.get("schema") != "kbound_so2sat_label_free_population_manifest_v1":
        raise IntegrityError("unknown So2Sat population manifest schema")
    if population_manifest.get("status") != "LABEL_FREE_METADATA_POPULATION_VERIFIED":
        raise IntegrityError("population manifest is not label-free and verified")
    if population_manifest.get("protocol_id") != PROTOCOL_ID:
        raise IntegrityError("population manifest protocol id drift")
    manifest_sha256 = require_sha256(population_manifest.get("manifest_sha256"), field="manifest_sha256")
    unsigned_manifest = dict(population_manifest)
    unsigned_manifest.pop("manifest_sha256", None)
    if manifest_sha256 != stable_sha256(unsigned_manifest):
        raise IntegrityError("population manifest self-hash mismatch")

    if not isinstance(manifest_receipt, Mapping):
        raise IntegrityError("population manifest receipt must be a mapping")
    if manifest_receipt.get("schema") not in {
        ARTIFACT_RECEIPT_SCHEMA_V1,
        ARTIFACT_RECEIPT_SCHEMA_V2,
    }:
        raise IntegrityError("unknown So2Sat population manifest receipt schema")
    manifest_artifact_sha256 = require_sha256(
        manifest_receipt.get("artifact_sha256"), field="manifest_receipt.artifact_sha256"
    )
    manifest_canonical_sha256 = require_sha256(
        manifest_receipt.get("canonical_document_sha256"),
        field="manifest_receipt.canonical_document_sha256",
    )
    if manifest_canonical_sha256 != stable_sha256(dict(population_manifest)):
        raise IntegrityError("population manifest receipt canonical hash mismatch")
    artifact_bytes = manifest_receipt.get("artifact_bytes")
    if isinstance(artifact_bytes, bool) or not isinstance(artifact_bytes, int) or artifact_bytes < 1:
        raise IntegrityError("population manifest receipt has an invalid byte count")

    protocol_identity = population_manifest.get("protocol_identity")
    cities = population_manifest.get("cities")
    if not isinstance(protocol_identity, Mapping) or not isinstance(cities, Mapping):
        raise IntegrityError("population manifest lacks protocol or city identity")
    training_roles = cities.get("training_roles")
    if not isinstance(training_roles, Mapping):
        raise IntegrityError("population manifest lacks training city roles")
    gate_fit_cities = _sorted_unique_strings(
        training_roles.get("gate_fit"), field="cities.training_roles.gate_fit", count=FIT_CITY_COUNT
    )
    gate_cal_cities = _sorted_unique_strings(
        training_roles.get("gate_cal"),
        field="cities.training_roles.gate_cal",
        count=CALIBRATION_CITY_COUNT,
    )
    target_cities = _sorted_unique_strings(cities.get("target"), field="cities.target", count=TARGET_CITY_COUNT)
    if set(gate_fit_cities) & set(gate_cal_cities):
        raise IntegrityError("gate-fit and gate-calibration cities overlap")
    if (set(gate_fit_cities) | set(gate_cal_cities)) & set(target_cities):
        raise IntegrityError("development and target cities overlap")

    binding = {
        "schema": STUDY_BINDING_SCHEMA,
        "status": "SEALED_LABEL_FREE_POPULATION_BINDING",
        "protocol_id": PROTOCOL_ID,
        "manifest_artifact_sha256": manifest_artifact_sha256,
        "manifest_canonical_document_sha256": manifest_canonical_sha256,
        "manifest_sha256": manifest_sha256,
        "population_identity_sha256": require_sha256(
            population_manifest.get("population_identity_sha256"),
            field="population_identity_sha256",
        ),
        "protocol_file_sha256": require_sha256(
            protocol_identity.get("file_sha256"), field="protocol_identity.file_sha256"
        ),
        "protocol_document_sha256": require_sha256(
            protocol_identity.get("canonical_document_sha256"),
            field="protocol_identity.canonical_document_sha256",
        ),
        "gate_fit_cities": gate_fit_cities,
        "gate_cal_cities": gate_cal_cities,
        "target_cities": target_cities,
    }
    binding["binding_sha256"] = stable_sha256(binding)
    validate_study_binding(binding)
    return binding


def load_study_binding(
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a manifest artifact/receipt pair and return its gate binding."""

    receipt = verify_artifact_receipt(manifest_path, receipt_path)
    manifest = strict_json_load(manifest_path)
    if not isinstance(manifest, Mapping):
        raise IntegrityError("population manifest artifact must be a JSON mapping")
    return build_study_binding(manifest, receipt)


def validate_study_binding(binding: Mapping[str, Any]) -> None:
    if not isinstance(binding, Mapping) or set(binding) != _STUDY_BINDING_KEYS:
        raise IntegrityError("So2Sat gate study binding has unknown or missing fields")
    if (
        binding.get("schema") != STUDY_BINDING_SCHEMA
        or binding.get("status") != "SEALED_LABEL_FREE_POPULATION_BINDING"
        or binding.get("protocol_id") != PROTOCOL_ID
    ):
        raise IntegrityError("unknown or unsealed So2Sat gate study binding")
    for field in (
        "manifest_artifact_sha256",
        "manifest_canonical_document_sha256",
        "manifest_sha256",
        "population_identity_sha256",
        "protocol_file_sha256",
        "protocol_document_sha256",
    ):
        require_sha256(binding.get(field), field=f"study_binding.{field}")
    fit = _sorted_unique_strings(
        binding.get("gate_fit_cities"), field="study_binding.gate_fit_cities", count=FIT_CITY_COUNT
    )
    calibration = _sorted_unique_strings(
        binding.get("gate_cal_cities"),
        field="study_binding.gate_cal_cities",
        count=CALIBRATION_CITY_COUNT,
    )
    target = _sorted_unique_strings(
        binding.get("target_cities"), field="study_binding.target_cities", count=TARGET_CITY_COUNT
    )
    if set(fit) & set(calibration) or (set(fit) | set(calibration)) & set(target):
        raise IntegrityError("study binding city roles overlap")
    claimed = require_sha256(binding.get("binding_sha256"), field="binding_sha256")
    unsigned = dict(binding)
    unsigned.pop("binding_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("study binding SHA-256 mismatch")


def trace_identity_sha256(
    *,
    role: str,
    city_id: str,
    checkpoint_id: str,
    checkpoint_tensor_sha256: str,
    checkpoint_file_sha256: str,
    trace_id: str,
    partition_sha256: str,
    feature_sha256: str,
    manifest_sha256: str,
    population_identity_sha256: str,
    protocol_file_sha256: str,
    protocol_document_sha256: str,
) -> str:
    """Hash every identity that defines one probe trace."""

    if role not in {"gate_fit", "gate_cal", "target_probe"}:
        raise IntegrityError(f"unknown So2Sat trace role {role!r}")
    if not isinstance(city_id, str) or not city_id:
        raise IntegrityError("trace city_id must be a non-empty string")
    if checkpoint_id not in CHECKPOINT_IDS:
        raise IntegrityError(f"trace checkpoint_id must be one of {CHECKPOINT_IDS}")
    if not isinstance(trace_id, str) or not trace_id:
        raise IntegrityError("trace_id must be a non-empty string")
    payload = {
        "schema": "kbound_so2sat_probe_trace_identity_v1",
        "role": role,
        "city_id": city_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_tensor_sha256": require_sha256(checkpoint_tensor_sha256, field="checkpoint_tensor_sha256"),
        "checkpoint_file_sha256": require_sha256(checkpoint_file_sha256, field="checkpoint_file_sha256"),
        "trace_id": trace_id,
        "partition_sha256": require_sha256(partition_sha256, field="partition_sha256"),
        "feature_sha256": require_sha256(feature_sha256, field="feature_sha256"),
        "manifest_sha256": require_sha256(manifest_sha256, field="manifest_sha256"),
        "population_identity_sha256": require_sha256(population_identity_sha256, field="population_identity_sha256"),
        "protocol_file_sha256": require_sha256(protocol_file_sha256, field="protocol_file_sha256"),
        "protocol_document_sha256": require_sha256(protocol_document_sha256, field="protocol_document_sha256"),
    }
    return stable_sha256(payload)


def _normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    required_role: str,
    binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _DEVELOPMENT_ROW_KEYS:
            raise IntegrityError(f"{required_role} row {index} has unknown or missing fields")
        if row.get("role") != required_role:
            raise IntegrityError(f"row {index} role must be {required_role!r}, found {row.get('role')!r}")
        city_id = row.get("city_id")
        checkpoint_id = row.get("checkpoint_id")
        trace_id = row.get("trace_id")
        if not isinstance(city_id, str) or not city_id:
            raise IntegrityError(f"{required_role} row {index} lacks a city_id")
        if checkpoint_id not in CHECKPOINT_IDS:
            raise IntegrityError(f"{required_role} row {index} checkpoint_id must be {CHECKPOINT_IDS}")
        if not isinstance(trace_id, str) or not trace_id:
            raise IntegrityError(f"{required_role} row {index} lacks a trace_id")
        for field in (
            "manifest_sha256",
            "population_identity_sha256",
            "protocol_file_sha256",
            "protocol_document_sha256",
        ):
            observed = require_sha256(row.get(field), field=f"row.{field}")
            if observed != binding[field]:
                raise IntegrityError(f"{required_role} row {index} {field} mismatch")
        feature_document = row.get("feature_document")
        if not isinstance(feature_document, Mapping):
            raise IntegrityError(f"{required_role} row {index} lacks a feature document")
        validate_feature_document(feature_document)
        tensor_sha256 = require_sha256(row.get("checkpoint_tensor_sha256"), field="checkpoint_tensor_sha256")
        file_sha256 = require_sha256(row.get("checkpoint_file_sha256"), field="checkpoint_file_sha256")
        partition_sha256 = require_sha256(row.get("partition_sha256"), field="partition_sha256")
        trace_sha256 = require_sha256(row.get("trace_sha256"), field="trace_sha256")
        expected_trace_sha256 = trace_identity_sha256(
            role=required_role,
            city_id=city_id,
            checkpoint_id=checkpoint_id,
            checkpoint_tensor_sha256=tensor_sha256,
            checkpoint_file_sha256=file_sha256,
            trace_id=trace_id,
            partition_sha256=partition_sha256,
            feature_sha256=feature_document["feature_sha256"],
            manifest_sha256=binding["manifest_sha256"],
            population_identity_sha256=binding["population_identity_sha256"],
            protocol_file_sha256=binding["protocol_file_sha256"],
            protocol_document_sha256=binding["protocol_document_sha256"],
        )
        if trace_sha256 != expected_trace_sha256:
            raise IntegrityError(f"{required_role} row {index} trace identity hash mismatch")
        benefit = _finite_number(row.get("observed_benefit"), field="observed_benefit")
        if not -1.0 <= benefit <= 1.0:
            raise IntegrityError("development observed_benefit must lie in [-1, 1]")
        normalized.append(
            {
                "role": required_role,
                "city_id": city_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_tensor_sha256": tensor_sha256,
                "checkpoint_file_sha256": file_sha256,
                "trace_id": trace_id,
                "trace_sha256": trace_sha256,
                "partition_sha256": partition_sha256,
                "manifest_sha256": binding["manifest_sha256"],
                "population_identity_sha256": binding["population_identity_sha256"],
                "protocol_file_sha256": binding["protocol_file_sha256"],
                "protocol_document_sha256": binding["protocol_document_sha256"],
                "feature_document": copy.deepcopy(dict(feature_document)),
                "features": feature_vector(feature_document),
                "observed_benefit": benefit,
            }
        )
    normalized.sort(key=lambda value: (value["city_id"], value["checkpoint_id"]))
    return normalized


def _validate_complete_design(
    fit: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    *,
    binding: Mapping[str, Any],
) -> dict[str, tuple[str, str]]:
    expected_fit_cells = {(city, checkpoint) for city in binding["gate_fit_cities"] for checkpoint in CHECKPOINT_IDS}
    expected_calibration_cells = {
        (city, checkpoint) for city in binding["gate_cal_cities"] for checkpoint in CHECKPOINT_IDS
    }
    fit_cells = [(row["city_id"], row["checkpoint_id"]) for row in fit]
    calibration_cells = [(row["city_id"], row["checkpoint_id"]) for row in calibration]
    if len(fit_cells) != len(set(fit_cells)) or set(fit_cells) != expected_fit_cells:
        raise IntegrityError("gate-fit rows must cover exactly 9 cities x 5 checkpoints")
    if len(calibration_cells) != len(set(calibration_cells)) or set(calibration_cells) != expected_calibration_cells:
        raise IntegrityError("gate-calibration rows must cover exactly 19 cities x 5 checkpoints")
    if set(binding["gate_fit_cities"]) & set(binding["gate_cal_cities"]):
        raise IntegrityError("gate-fit and gate-calibration cities overlap")

    all_rows = fit + calibration
    trace_ids = [row["trace_id"] for row in all_rows]
    trace_hashes = [row["trace_sha256"] for row in all_rows]
    if len(set(trace_ids)) != len(trace_ids) or len(set(trace_hashes)) != len(trace_hashes):
        raise IntegrityError("development trace ids and hashes must be globally unique")

    checkpoint_identities: dict[str, tuple[str, str]] = {}
    for row in all_rows:
        identity = (row["checkpoint_tensor_sha256"], row["checkpoint_file_sha256"])
        prior = checkpoint_identities.setdefault(row["checkpoint_id"], identity)
        if prior != identity:
            raise IntegrityError("one checkpoint id maps to different tensor or file identities")
    if set(checkpoint_identities) != set(CHECKPOINT_IDS):
        raise IntegrityError("development rows do not use exactly five checkpoint ids")
    if len({value[0] for value in checkpoint_identities.values()}) != len(CHECKPOINT_IDS):
        raise IntegrityError("five checkpoint ids must bind five distinct tensor hashes")
    if len({value[1] for value in checkpoint_identities.values()}) != len(CHECKPOINT_IDS):
        raise IntegrityError("five checkpoint ids must bind five distinct file hashes")

    partition_by_city: dict[str, str] = {}
    for row in all_rows:
        prior = partition_by_city.setdefault(row["city_id"], row["partition_sha256"])
        if prior != row["partition_sha256"]:
            raise IntegrityError("one city maps to multiple probe/evaluation partitions")
    if len(set(partition_by_city.values())) != len(partition_by_city):
        raise IntegrityError("different development cities must have distinct partition hashes")
    return checkpoint_identities


def _row_payload_sha256(rows: list[dict[str, Any]]) -> str:
    payload = []
    for row in rows:
        payload.append({key: copy.deepcopy(row[key]) for key in _DEVELOPMENT_ROW_KEYS})
    return stable_sha256(payload)


def _cell_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    feature = row["feature_document"]
    return {
        "city_id": row["city_id"],
        "checkpoint_id": row["checkpoint_id"],
        "trace_id": row["trace_id"],
        "trace_sha256": row["trace_sha256"],
        "partition_sha256": row["partition_sha256"],
        "feature_sha256": feature["feature_sha256"],
        "frozen_logits_tensor_sha256": feature["frozen_logits_tensor_sha256"],
        "adapted_logits_tensor_sha256": feature["adapted_logits_tensor_sha256"],
    }


def _predict_arrays(
    x: np.ndarray,
    *,
    means: np.ndarray,
    scales: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        output = intercept + ((x - means) / scales) @ coefficients
    if not np.isfinite(output).all():
        raise IntegrityError("ridge prediction is numerically non-finite")
    return output


def fit_calibrate_ridge_gate(
    fit_rows: Iterable[Mapping[str, Any]],
    calibration_rows: Iterable[Mapping[str, Any]],
    *,
    study_binding: Mapping[str, Any],
    alpha: float = CALIBRATION_ALPHA,
    ridge_penalty: float = RIDGE_PENALTY,
) -> dict[str, Any]:
    """Fit the frozen ridge model and city-level split-conformal radius."""

    validate_study_binding(study_binding)
    if _finite_number(alpha, field="alpha") != CALIBRATION_ALPHA:
        raise IntegrityError(f"So2Sat gate alpha is frozen at {CALIBRATION_ALPHA}")
    if _finite_number(ridge_penalty, field="ridge_penalty") != RIDGE_PENALTY:
        raise IntegrityError(f"So2Sat ridge penalty is frozen at {RIDGE_PENALTY}")

    fit = _normalize_rows(fit_rows, required_role="gate_fit", binding=study_binding)
    calibration = _normalize_rows(calibration_rows, required_role="gate_cal", binding=study_binding)
    checkpoint_identities = _validate_complete_design(fit, calibration, binding=study_binding)

    x_fit = np.vstack([row["features"] for row in fit])
    y_fit = np.asarray([row["observed_benefit"] for row in fit], dtype=np.float64)
    means = x_fit.mean(axis=0)
    raw_scales = x_fit.std(axis=0, ddof=0)
    scales = np.where(raw_scales > 0.0, raw_scales, 1.0)
    standardized = (x_fit - means) / scales
    design = np.column_stack([np.ones(len(standardized)), standardized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_PENALTY
    penalty[0, 0] = 0.0
    try:
        solution = np.linalg.solve(design.T @ design + penalty, design.T @ y_fit)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - ridge regularizes
        raise IntegrityError(f"So2Sat ridge gate fit failed: {exc}") from exc
    if not np.isfinite(solution).all():
        raise IntegrityError("So2Sat ridge solution is non-finite")
    intercept = float(solution[0])
    coefficients = solution[1:]

    x_calibration = np.vstack([row["features"] for row in calibration])
    y_calibration = np.asarray([row["observed_benefit"] for row in calibration], dtype=np.float64)
    predicted = _predict_arrays(
        x_calibration,
        means=means,
        scales=scales,
        intercept=intercept,
        coefficients=coefficients,
    )
    residuals_by_city: dict[str, list[float]] = {}
    for row, residual in zip(calibration, np.abs(predicted - y_calibration), strict=True):
        residuals_by_city.setdefault(row["city_id"], []).append(float(residual))
    city_max_residuals = {city: max(residuals_by_city[city]) for city in study_binding["gate_cal_cities"]}
    residuals_sorted = sorted(city_max_residuals.values())
    rank = math.ceil((len(residuals_sorted) + 1) * (1.0 - CALIBRATION_ALPHA))
    if rank != CONFORMAL_RANK or rank > len(residuals_sorted):
        raise IntegrityError("So2Sat split-conformal rank contract is infeasible or drifted")
    epsilon = float(residuals_sorted[rank - 1])

    provenance = {
        "fit_trace_count": len(fit),
        "fit_city_count": len(study_binding["gate_fit_cities"]),
        "calibration_trace_count": len(calibration),
        "calibration_city_count": len(study_binding["gate_cal_cities"]),
        "checkpoint_ids": list(CHECKPOINT_IDS),
        "checkpoint_tensor_sha256_by_id": {
            checkpoint: checkpoint_identities[checkpoint][0] for checkpoint in CHECKPOINT_IDS
        },
        "checkpoint_file_sha256_by_id": {
            checkpoint: checkpoint_identities[checkpoint][1] for checkpoint in CHECKPOINT_IDS
        },
        "fit_cells": [_cell_payload(row) for row in fit],
        "calibration_cells": [_cell_payload(row) for row in calibration],
        "fit_rows_sha256": _row_payload_sha256(fit),
        "calibration_rows_sha256": _row_payload_sha256(calibration),
        "fit_calibration_cities_disjoint": True,
        "target_rows_used": 0,
    }
    gate = {
        "schema": GATE_SCHEMA,
        "status": "SEALED_DEVELOPMENT_ONLY",
        "n_classes": N_CLASSES,
        "feature_names": list(FEATURE_NAMES),
        "study_binding": copy.deepcopy(dict(study_binding)),
        "ridge": {
            "penalty": RIDGE_PENALTY,
            "intercept_unpenalized": True,
            "standardization": "gate_fit_population_sd; zero_sd_replaced_by_one",
            "intercept": intercept,
            "coefficients": [float(value) for value in coefficients],
            "fit_means": [float(value) for value in means],
            "fit_scales": [float(value) for value in scales],
        },
        "calibration": {
            "alpha": CALIBRATION_ALPHA,
            "method": "split_conformal_over_city_max_checkpoint_absolute_residual",
            "aggregation_within_city": "maximum_absolute_residual_over_five_checkpoints",
            "n_independent_cities": len(city_max_residuals),
            "order_statistic_rank_one_based": rank,
            "city_max_residuals": city_max_residuals,
            "residuals_sorted": [float(value) for value in residuals_sorted],
            "epsilon": epsilon,
        },
        "support": {
            "primary": "finite_values_and_exact_feature_schema",
            "failure_action": "ABSTAIN",
            "abstain_realized_action": "FREEZE",
        },
        "development_provenance": provenance,
        "decision_rule": {
            "adapt": "lower > 0",
            "freeze": "upper < 0",
            "otherwise": "ABSTAIN",
            "abstain_realized_action": "FREEZE",
        },
    }
    gate["gate_sha256"] = stable_sha256(gate)
    validate_gate_document(gate)
    return gate


def _validate_cells(
    cells: Any,
    *,
    cities: list[str],
    role_name: str,
) -> list[Mapping[str, Any]]:
    expected = {(city, checkpoint) for city in cities for checkpoint in CHECKPOINT_IDS}
    if not isinstance(cells, list) or len(cells) != len(expected):
        raise IntegrityError(f"gate {role_name} cell inventory is incomplete")
    observed: list[tuple[str, str]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping) or set(cell) != _CELL_KEYS:
            raise IntegrityError(f"gate {role_name} cell {index} schema drift")
        city = cell.get("city_id")
        checkpoint = cell.get("checkpoint_id")
        if not isinstance(city, str) or checkpoint not in CHECKPOINT_IDS:
            raise IntegrityError(f"gate {role_name} cell {index} identity drift")
        if not isinstance(cell.get("trace_id"), str) or not cell["trace_id"]:
            raise IntegrityError(f"gate {role_name} cell {index} lacks trace_id")
        for field in _CELL_KEYS - {"city_id", "checkpoint_id", "trace_id"}:
            require_sha256(cell.get(field), field=f"{role_name}_cells.{index}.{field}")
        observed.append((city, checkpoint))
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise IntegrityError(f"gate {role_name} cells do not form the complete city/checkpoint grid")
    return cells


def validate_gate_document(document: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "status",
        "n_classes",
        "feature_names",
        "study_binding",
        "ridge",
        "calibration",
        "support",
        "development_provenance",
        "decision_rule",
        "gate_sha256",
    }
    if not isinstance(document, Mapping) or set(document) != expected_keys:
        raise IntegrityError("So2Sat gate document has unknown or missing fields")
    if document.get("schema") != GATE_SCHEMA or document.get("status") != "SEALED_DEVELOPMENT_ONLY":
        raise IntegrityError("unknown or unsealed So2Sat gate document")
    if document.get("n_classes") != N_CLASSES:
        raise IntegrityError("So2Sat gate class-count drift")
    if tuple(document.get("feature_names", ())) != FEATURE_NAMES:
        raise IntegrityError("So2Sat gate feature schema drift")
    binding = document.get("study_binding")
    if not isinstance(binding, Mapping):
        raise IntegrityError("So2Sat gate lacks its study binding")
    validate_study_binding(binding)

    ridge = document.get("ridge")
    expected_ridge_keys = {
        "penalty",
        "intercept_unpenalized",
        "standardization",
        "intercept",
        "coefficients",
        "fit_means",
        "fit_scales",
    }
    if not isinstance(ridge, Mapping) or set(ridge) != expected_ridge_keys:
        raise IntegrityError("So2Sat ridge schema drift")
    if (
        ridge.get("penalty") != RIDGE_PENALTY
        or ridge.get("intercept_unpenalized") is not True
        or ridge.get("standardization") != "gate_fit_population_sd; zero_sd_replaced_by_one"
    ):
        raise IntegrityError("So2Sat ridge penalty or standardization drift")
    _finite_number(ridge.get("intercept"), field="ridge.intercept")
    for field in ("coefficients", "fit_means", "fit_scales"):
        values = ridge.get(field)
        if not isinstance(values, list) or len(values) != len(FEATURE_NAMES):
            raise IntegrityError(f"ridge.{field} shape drift")
        for value in values:
            _finite_number(value, field=f"ridge.{field}")
    if any(float(value) <= 0.0 for value in ridge["fit_scales"]):
        raise IntegrityError("ridge fit scales must be strictly positive")

    calibration = document.get("calibration")
    expected_calibration_keys = {
        "alpha",
        "method",
        "aggregation_within_city",
        "n_independent_cities",
        "order_statistic_rank_one_based",
        "city_max_residuals",
        "residuals_sorted",
        "epsilon",
    }
    if not isinstance(calibration, Mapping) or set(calibration) != expected_calibration_keys:
        raise IntegrityError("So2Sat calibration schema drift")
    if (
        calibration.get("alpha") != CALIBRATION_ALPHA
        or calibration.get("method") != "split_conformal_over_city_max_checkpoint_absolute_residual"
        or calibration.get("aggregation_within_city") != "maximum_absolute_residual_over_five_checkpoints"
        or calibration.get("n_independent_cities") != CALIBRATION_CITY_COUNT
        or calibration.get("order_statistic_rank_one_based") != CONFORMAL_RANK
    ):
        raise IntegrityError("So2Sat calibration contract drift")
    residuals_by_city = calibration.get("city_max_residuals")
    if not isinstance(residuals_by_city, Mapping) or set(residuals_by_city) != set(binding["gate_cal_cities"]):
        raise IntegrityError("So2Sat calibration city residuals are incomplete")
    normalized_city_residuals = {
        city: _finite_number(value, field=f"calibration.city_max_residuals.{city}")
        for city, value in residuals_by_city.items()
    }
    if any(value < 0.0 for value in normalized_city_residuals.values()):
        raise IntegrityError("So2Sat calibration residuals must be non-negative")
    residuals_sorted = calibration.get("residuals_sorted")
    if not isinstance(residuals_sorted, list) or len(residuals_sorted) != CALIBRATION_CITY_COUNT:
        raise IntegrityError("So2Sat gate must store nineteen sorted city residuals")
    normalized_sorted = [_finite_number(value, field="calibration.residuals_sorted") for value in residuals_sorted]
    if normalized_sorted != sorted(normalized_city_residuals.values()):
        raise IntegrityError("So2Sat calibration residual order/content drift")
    epsilon = _finite_number(calibration.get("epsilon"), field="calibration.epsilon")
    if epsilon < 0.0 or epsilon != normalized_sorted[CONFORMAL_RANK - 1]:
        raise IntegrityError("So2Sat alpha=.10 radius must use rank 18 of 19")

    provenance = document.get("development_provenance")
    expected_provenance_keys = {
        "fit_trace_count",
        "fit_city_count",
        "calibration_trace_count",
        "calibration_city_count",
        "checkpoint_ids",
        "checkpoint_tensor_sha256_by_id",
        "checkpoint_file_sha256_by_id",
        "fit_cells",
        "calibration_cells",
        "fit_rows_sha256",
        "calibration_rows_sha256",
        "fit_calibration_cities_disjoint",
        "target_rows_used",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != expected_provenance_keys:
        raise IntegrityError("So2Sat gate development provenance schema drift")
    if (
        provenance.get("fit_trace_count") != FIT_TRACE_COUNT
        or provenance.get("fit_city_count") != FIT_CITY_COUNT
        or provenance.get("calibration_trace_count") != CALIBRATION_TRACE_COUNT
        or provenance.get("calibration_city_count") != CALIBRATION_CITY_COUNT
        or provenance.get("checkpoint_ids") != list(CHECKPOINT_IDS)
        or provenance.get("fit_calibration_cities_disjoint") is not True
        or provenance.get("target_rows_used") != 0
    ):
        raise IntegrityError("So2Sat gate development design drift")
    tensor_by_id = provenance.get("checkpoint_tensor_sha256_by_id")
    file_by_id = provenance.get("checkpoint_file_sha256_by_id")
    if (
        not isinstance(tensor_by_id, Mapping)
        or set(tensor_by_id) != set(CHECKPOINT_IDS)
        or not isinstance(file_by_id, Mapping)
        or set(file_by_id) != set(CHECKPOINT_IDS)
    ):
        raise IntegrityError("So2Sat gate checkpoint identity inventory is incomplete")
    for checkpoint in CHECKPOINT_IDS:
        require_sha256(tensor_by_id[checkpoint], field=f"checkpoint_tensor.{checkpoint}")
        require_sha256(file_by_id[checkpoint], field=f"checkpoint_file.{checkpoint}")
    if len(set(tensor_by_id.values())) != 5 or len(set(file_by_id.values())) != 5:
        raise IntegrityError("So2Sat gate requires five distinct checkpoint identities")
    fit_cells = _validate_cells(
        provenance.get("fit_cells"),
        cities=binding["gate_fit_cities"],
        role_name="fit",
    )
    calibration_cells = _validate_cells(
        provenance.get("calibration_cells"),
        cities=binding["gate_cal_cities"],
        role_name="calibration",
    )
    trace_ids = [cell["trace_id"] for cell in fit_cells + calibration_cells]
    trace_hashes = [cell["trace_sha256"] for cell in fit_cells + calibration_cells]
    if len(set(trace_ids)) != len(trace_ids) or len(set(trace_hashes)) != len(trace_hashes):
        raise IntegrityError("gate provenance development trace identities overlap")
    partition_by_city: dict[str, str] = {}
    for cell in fit_cells + calibration_cells:
        prior = partition_by_city.setdefault(cell["city_id"], cell["partition_sha256"])
        if prior != cell["partition_sha256"]:
            raise IntegrityError("gate provenance maps one city to multiple partitions")
    if len(set(partition_by_city.values())) != len(partition_by_city):
        raise IntegrityError("gate provenance reuses one partition hash across cities")
    require_sha256(provenance.get("fit_rows_sha256"), field="fit_rows_sha256")
    require_sha256(provenance.get("calibration_rows_sha256"), field="calibration_rows_sha256")

    if document.get("support") != {
        "primary": "finite_values_and_exact_feature_schema",
        "failure_action": "ABSTAIN",
        "abstain_realized_action": "FREEZE",
    }:
        raise IntegrityError("So2Sat gate support contract drift")
    if document.get("decision_rule") != {
        "adapt": "lower > 0",
        "freeze": "upper < 0",
        "otherwise": "ABSTAIN",
        "abstain_realized_action": "FREEZE",
    }:
        raise IntegrityError("So2Sat gate decision contract drift")
    claimed = require_sha256(document.get("gate_sha256"), field="gate_sha256")
    unsigned = dict(document)
    unsigned.pop("gate_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("gate_sha256 does not match the So2Sat gate document")


def write_gate_with_receipt(path: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    validate_gate_document(document)
    return write_immutable_json_with_receipt(path, document)


def load_gate_with_receipt(
    path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    verify_artifact_receipt(path, receipt_path)
    document = strict_json_load(path)
    if not isinstance(document, Mapping):
        raise IntegrityError("So2Sat gate artifact must be a JSON mapping")
    validate_gate_document(document)
    return dict(document)


def _construct_action(
    gate: Mapping[str, Any],
    feature_document: Mapping[str, Any],
    *,
    city_id: str,
    checkpoint_id: str,
    checkpoint_tensor_sha256: str,
    checkpoint_file_sha256: str,
    trace_id: str,
    trace_sha256: str,
    partition_sha256: str,
) -> dict[str, Any]:
    validate_gate_document(gate)
    validate_feature_document(feature_document)
    binding = gate["study_binding"]
    if city_id not in binding["target_cities"]:
        raise IntegrityError("gate action city is not one of the ten sealed target cities")
    if checkpoint_id not in CHECKPOINT_IDS:
        raise IntegrityError(f"gate action checkpoint must be one of {CHECKPOINT_IDS}")
    provenance = gate["development_provenance"]
    tensor_sha = require_sha256(checkpoint_tensor_sha256, field="checkpoint_tensor_sha256")
    file_sha = require_sha256(checkpoint_file_sha256, field="checkpoint_file_sha256")
    if tensor_sha != provenance["checkpoint_tensor_sha256_by_id"][checkpoint_id]:
        raise IntegrityError("gate action checkpoint tensor identity mismatch")
    if file_sha != provenance["checkpoint_file_sha256_by_id"][checkpoint_id]:
        raise IntegrityError("gate action checkpoint file identity mismatch")
    partition_sha = require_sha256(partition_sha256, field="partition_sha256")
    claimed_trace_sha = require_sha256(trace_sha256, field="trace_sha256")
    expected_trace_sha = trace_identity_sha256(
        role="target_probe",
        city_id=city_id,
        checkpoint_id=checkpoint_id,
        checkpoint_tensor_sha256=tensor_sha,
        checkpoint_file_sha256=file_sha,
        trace_id=trace_id,
        partition_sha256=partition_sha,
        feature_sha256=feature_document["feature_sha256"],
        manifest_sha256=binding["manifest_sha256"],
        population_identity_sha256=binding["population_identity_sha256"],
        protocol_file_sha256=binding["protocol_file_sha256"],
        protocol_document_sha256=binding["protocol_document_sha256"],
    )
    if claimed_trace_sha != expected_trace_sha:
        raise IntegrityError("target probe trace identity hash mismatch")

    vector = feature_vector(feature_document).reshape(1, -1)
    ridge = gate["ridge"]
    delta_hat = float(
        _predict_arrays(
            vector,
            means=np.asarray(ridge["fit_means"], dtype=np.float64),
            scales=np.asarray(ridge["fit_scales"], dtype=np.float64),
            intercept=float(ridge["intercept"]),
            coefficients=np.asarray(ridge["coefficients"], dtype=np.float64),
        )[0]
    )
    epsilon = float(gate["calibration"]["epsilon"])
    lower = delta_hat - epsilon
    upper = delta_hat + epsilon
    if lower > 0.0:
        decision = "ADAPT"
    elif upper < 0.0:
        decision = "FREEZE"
    else:
        decision = "ABSTAIN"
    action = {
        "schema": ACTION_SCHEMA,
        "status": "SEALED_BEFORE_TARGET_EVALUATION",
        "gate_sha256": gate["gate_sha256"],
        "study_binding_sha256": binding["binding_sha256"],
        "manifest_sha256": binding["manifest_sha256"],
        "population_identity_sha256": binding["population_identity_sha256"],
        "protocol_file_sha256": binding["protocol_file_sha256"],
        "protocol_document_sha256": binding["protocol_document_sha256"],
        "city_id": city_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_tensor_sha256": tensor_sha,
        "checkpoint_file_sha256": file_sha,
        "trace_id": trace_id,
        "trace_sha256": claimed_trace_sha,
        "partition_sha256": partition_sha,
        "feature_document": copy.deepcopy(dict(feature_document)),
        "support_status": "IN_SUPPORT",
        "delta_hat": delta_hat,
        "epsilon": epsilon,
        "lower": lower,
        "upper": upper,
        "decision": decision,
        "realized_action": "ADAPT" if decision == "ADAPT" else "FREEZE",
    }
    action["action_sha256"] = stable_sha256(action)
    return action


def apply_gate(
    gate: Mapping[str, Any],
    feature_document: Mapping[str, Any],
    *,
    city_id: str,
    checkpoint_id: str,
    checkpoint_tensor_sha256: str,
    checkpoint_file_sha256: str,
    trace_id: str,
    trace_sha256: str,
    partition_sha256: str,
) -> dict[str, Any]:
    """Create one fully bound, label-free target action document.

    Any integrity exception is a fail-closed condition for a live runner and
    must be realized as frozen inference.  A valid interval ABSTAIN is recorded
    explicitly and is also realized as FREEZE.
    """

    action = _construct_action(
        gate,
        feature_document,
        city_id=city_id,
        checkpoint_id=checkpoint_id,
        checkpoint_tensor_sha256=checkpoint_tensor_sha256,
        checkpoint_file_sha256=checkpoint_file_sha256,
        trace_id=trace_id,
        trace_sha256=trace_sha256,
        partition_sha256=partition_sha256,
    )
    validate_action_document(action, gate=gate)
    return action


def validate_action_document(
    action: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
) -> None:
    validate_gate_document(gate)
    if not isinstance(action, Mapping) or set(action) != _ACTION_KEYS:
        raise IntegrityError("So2Sat action document has unknown or missing fields")
    if action.get("schema") != ACTION_SCHEMA or action.get("status") != "SEALED_BEFORE_TARGET_EVALUATION":
        raise IntegrityError("unknown or unsealed So2Sat action document")
    claimed = require_sha256(action.get("action_sha256"), field="action_sha256")
    unsigned = dict(action)
    unsigned.pop("action_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("action_sha256 does not match the action document")
    expected = _construct_action(
        gate,
        action.get("feature_document", {}),
        city_id=action.get("city_id"),
        checkpoint_id=action.get("checkpoint_id"),
        checkpoint_tensor_sha256=action.get("checkpoint_tensor_sha256"),
        checkpoint_file_sha256=action.get("checkpoint_file_sha256"),
        trace_id=action.get("trace_id"),
        trace_sha256=action.get("trace_sha256"),
        partition_sha256=action.get("partition_sha256"),
    )
    if dict(action) != expected:
        raise IntegrityError("So2Sat action does not replay from the sealed gate and trace")


def replay_action(action: Mapping[str, Any], *, gate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and deterministically reconstruct a sealed action."""

    validate_action_document(action, gate=gate)
    return _construct_action(
        gate,
        action["feature_document"],
        city_id=action["city_id"],
        checkpoint_id=action["checkpoint_id"],
        checkpoint_tensor_sha256=action["checkpoint_tensor_sha256"],
        checkpoint_file_sha256=action["checkpoint_file_sha256"],
        trace_id=action["trace_id"],
        trace_sha256=action["trace_sha256"],
        partition_sha256=action["partition_sha256"],
    )


def write_action_with_receipt(
    path: str | Path,
    action: Mapping[str, Any],
    *,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    validate_action_document(action, gate=gate)
    return write_immutable_json_with_receipt(path, action)


def load_action_with_receipt(
    path: str | Path,
    *,
    gate: Mapping[str, Any],
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    verify_artifact_receipt(path, receipt_path)
    action = strict_json_load(path)
    if not isinstance(action, Mapping):
        raise IntegrityError("So2Sat action artifact must be a JSON mapping")
    validate_action_document(action, gate=gate)
    return dict(action)
