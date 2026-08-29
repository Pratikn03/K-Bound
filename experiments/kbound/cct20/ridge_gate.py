"""Frozen low-capacity ridge gate with location-level exact-rank calibration."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from kga.certificate import split_conformal_rank_radius

from .integrity import IntegrityError, require_sha256, stable_sha256
from .label_free_traces import FEATURE_NAMES

RIDGE_PENALTY = 10.0
CALIBRATION_ALPHA = 0.10
DECISIONS = ("ADAPT", "FREEZE", "ABSTAIN")
FIT_UNITS = frozenset(("trans_val:125", "cis_test:33"))
CALIBRATION_UNITS = frozenset(
    f"cis_test:{location}" for location in (38, 43, 51, 61, 88, 90, 108, 115, 120)
)
CHECKPOINT_IDS = tuple(str(seed) for seed in range(5))


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise IntegrityError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise IntegrityError(f"{field} must be a finite number")
    return result


def _feature_vector(value: Mapping[str, Any]) -> np.ndarray:
    if not isinstance(value, Mapping) or set(value) != set(FEATURE_NAMES):
        raise IntegrityError(
            "gate features must contain exactly the frozen feature schema: "
            f"{FEATURE_NAMES}"
        )
    return np.asarray(
        [_finite_number(value[name], field=f"features.{name}") for name in FEATURE_NAMES],
        dtype=np.float64,
    )


def _normalize_rows(rows: Iterable[Mapping[str, Any]], *, required_role: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if row.get("role") != required_role:
            raise IntegrityError(
                f"row {index} role must be {required_role!r}, found {row.get('role')!r}"
            )
        trace_id = str(row.get("trace_id", ""))
        unit = str(row.get("calibration_unit", ""))
        checkpoint = str(row.get("checkpoint_id", ""))
        if not trace_id or not unit or not checkpoint:
            raise IntegrityError(
                f"row {index} requires trace_id, calibration_unit, and checkpoint_id"
            )
        benefit = _finite_number(row.get("observed_benefit"), field="observed_benefit")
        if not -1.0 <= benefit <= 1.0:
            raise IntegrityError("development observed_benefit must lie in [-1, 1]")
        output.append(
            {
                "trace_id": trace_id,
                "calibration_unit": unit,
                "checkpoint_id": checkpoint,
                "checkpoint_tensor_sha256": require_sha256(
                    row.get("checkpoint_tensor_sha256"),
                    field="checkpoint_tensor_sha256",
                ),
                "checkpoint_file_sha256": require_sha256(
                    row.get("checkpoint_file_sha256"),
                    field="checkpoint_file_sha256",
                ),
                "trace_sha256": require_sha256(
                    row.get("trace_sha256"), field="trace_sha256"
                ),
                "partition_sha256": require_sha256(
                    row.get("partition_sha256"), field="partition_sha256"
                ),
                "features": _feature_vector(row.get("features", {})),
                "observed_benefit": benefit,
            }
        )
    if not output:
        raise IntegrityError(f"{required_role} cannot be empty")
    ids = [row["trace_id"] for row in output]
    if len(set(ids)) != len(ids):
        raise IntegrityError(f"duplicate trace_id in {required_role}")
    # Canonicalize before every floating-point reduction so gate replay is
    # byte-stable even when an artifact loader returns the same rows in a
    # different order.
    output.sort(key=lambda row: row["trace_id"])
    return output


def _predict_arrays(
    x: np.ndarray,
    *,
    means: np.ndarray,
    scales: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    return intercept + ((x - means) / scales) @ coefficients


def _row_payload_sha256(rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "trace_id": row["trace_id"],
            "calibration_unit": row["calibration_unit"],
            "checkpoint_id": row["checkpoint_id"],
            "checkpoint_tensor_sha256": row["checkpoint_tensor_sha256"],
            "checkpoint_file_sha256": row["checkpoint_file_sha256"],
            "trace_sha256": row["trace_sha256"],
            "partition_sha256": row["partition_sha256"],
            "features": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, row["features"], strict=True)
            },
            "observed_benefit": float(row["observed_benefit"]),
        }
        for row in sorted(rows, key=lambda value: value["trace_id"])
    ]
    return stable_sha256(payload)


def fit_calibrate_ridge_gate(
    fit_rows: Iterable[Mapping[str, Any]],
    calibration_rows: Iterable[Mapping[str, Any]],
    *,
    alpha: float = CALIBRATION_ALPHA,
    ridge_penalty: float = RIDGE_PENALTY,
) -> dict[str, Any]:
    """Fit the fixed ridge map and calibrate by independent location units.

    Checkpoints within a calibration location are not treated as independent.
    Their absolute errors are collapsed by the maximum before the exact-rank
    radius is computed across locations.
    """

    if float(alpha) != CALIBRATION_ALPHA:
        raise IntegrityError(f"CCT-20 gate alpha is frozen at {CALIBRATION_ALPHA}")
    if float(ridge_penalty) != RIDGE_PENALTY:
        raise IntegrityError(f"CCT-20 ridge penalty is frozen at {RIDGE_PENALTY}")
    fit = _normalize_rows(fit_rows, required_role="development_fit")
    calibration = _normalize_rows(
        calibration_rows, required_role="development_calibration"
    )
    fit_units = {row["calibration_unit"] for row in fit}
    calibration_units = {row["calibration_unit"] for row in calibration}
    if fit_units != FIT_UNITS:
        raise IntegrityError(
            f"gate FIT units must be exactly {sorted(FIT_UNITS)}, found {sorted(fit_units)}"
        )
    if calibration_units != CALIBRATION_UNITS:
        raise IntegrityError(
            "gate CAL units must be exactly the nine sealed cis-test locations; "
            f"found {sorted(calibration_units)}"
        )
    overlap = fit_units & calibration_units
    if overlap:
        raise IntegrityError(f"gate FIT/CAL location units overlap: {sorted(overlap)}")
    if len(calibration_units) != 9:
        raise IntegrityError(
            "the frozen exact-rank calibration requires exactly nine "
            f"independent calibration locations, found {len(calibration_units)}"
        )
    if len(fit_units) != 2 or len(fit) != 10:
        raise IntegrityError(
            "the frozen gate FIT design requires two locations x five checkpoints "
            f"(10 traces), found {len(fit_units)} locations and {len(fit)} traces"
        )
    for role_name, rows_by_role in (("FIT", fit), ("CAL", calibration)):
        by_unit: dict[str, set[str]] = {}
        for row in rows_by_role:
            by_unit.setdefault(row["calibration_unit"], set()).add(row["checkpoint_id"])
        bad = {unit: len(checkpoints) for unit, checkpoints in by_unit.items() if len(checkpoints) != 5}
        if bad:
            raise IntegrityError(
                f"gate {role_name} requires five checkpoint traces per location: {bad}"
            )
        checkpoint_sets = {tuple(sorted(values)) for values in by_unit.values()}
        if len(checkpoint_sets) != 1:
            raise IntegrityError(f"gate {role_name} locations use different checkpoint identities")
        if role_name == "FIT":
            fit_checkpoint_set = next(iter(checkpoint_sets))
        elif next(iter(checkpoint_sets)) != fit_checkpoint_set:
            raise IntegrityError("gate FIT and CAL use different checkpoint identities")
        if next(iter(checkpoint_sets)) != CHECKPOINT_IDS:
            raise IntegrityError(
                f"gate {role_name} checkpoint identities must be {CHECKPOINT_IDS}"
            )

    checkpoint_identities: dict[str, tuple[str, str]] = {}
    for row in fit + calibration:
        identity = (
            row["checkpoint_tensor_sha256"],
            row["checkpoint_file_sha256"],
        )
        prior = checkpoint_identities.setdefault(row["checkpoint_id"], identity)
        if prior != identity:
            raise IntegrityError(
                "one checkpoint_id maps to different checkpoint tensor/file identities"
            )
    if len(checkpoint_identities) != 5 or len(
        {identity[0] for identity in checkpoint_identities.values()}
    ) != 5:
        raise IntegrityError("gate traces do not use five distinct checkpoint tensors")

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
    except np.linalg.LinAlgError as exc:  # pragma: no cover - ridge should regularize
        raise IntegrityError(f"ridge gate fit failed: {exc}") from exc
    intercept = float(solution[0])
    coefficients = solution[1:]

    x_cal = np.vstack([row["features"] for row in calibration])
    y_cal = np.asarray([row["observed_benefit"] for row in calibration], dtype=np.float64)
    predicted = _predict_arrays(
        x_cal,
        means=means,
        scales=scales,
        intercept=intercept,
        coefficients=coefficients,
    )
    residual_by_unit: dict[str, list[float]] = {}
    for row, residual in zip(calibration, np.abs(predicted - y_cal), strict=True):
        residual_by_unit.setdefault(row["calibration_unit"], []).append(float(residual))
    conservative_residuals = np.asarray(
        [max(residual_by_unit[unit]) for unit in sorted(residual_by_unit)],
        dtype=np.float64,
    )
    epsilon = split_conformal_rank_radius(
        conservative_residuals,
        alpha=CALIBRATION_ALPHA,
        on_infeasible="raise",
    )

    centered_fit = x_fit - means
    diagnostic_covariance = (centered_fit.T @ centered_fit) / max(1, len(x_fit))
    diagnostic_covariance += np.eye(len(FEATURE_NAMES), dtype=np.float64) * 1.0e-8
    diagnostic_precision = np.linalg.pinv(diagnostic_covariance)
    core = {
        "schema": "kbound_cct20_ridge_gate_v1",
        "status": "SEALED_DEVELOPMENT_ONLY",
        "feature_names": list(FEATURE_NAMES),
        "ridge": {
            "penalty": RIDGE_PENALTY,
            "intercept_unpenalized": True,
            "standardization": "development_fit_population_sd; zero_sd_replaced_by_one",
            "intercept": intercept,
            "coefficients": [float(value) for value in coefficients],
            "fit_means": [float(value) for value in means],
            "fit_scales": [float(value) for value in scales],
        },
        "calibration": {
            "alpha": CALIBRATION_ALPHA,
            "method": "exact_rank_over_location_max_checkpoint_residual",
            "aggregation_within_location": "maximum_absolute_residual_over_checkpoints",
            "n_independent_units": len(conservative_residuals),
            "residuals_sorted": [float(value) for value in np.sort(conservative_residuals)],
            "epsilon": float(epsilon),
        },
        "support": {
            "primary": "finite_values_and_exact_feature_schema",
            "failure_action": "ABSTAIN",
            "mahalanobis_role": "diagnostic_only_never_changes_action",
            "diagnostic_center": [float(value) for value in means],
            "diagnostic_precision": [
                [float(value) for value in row] for row in diagnostic_precision
            ],
        },
        "development_provenance": {
            "fit_trace_count": len(fit),
            "fit_unit_count": len(fit_units),
            "calibration_trace_count": len(calibration),
            "calibration_unit_count": len(calibration_units),
            "fit_trace_ids_sha256": stable_sha256(sorted(row["trace_id"] for row in fit)),
            "calibration_trace_ids_sha256": stable_sha256(
                sorted(row["trace_id"] for row in calibration)
            ),
            "fit_rows_sha256": _row_payload_sha256(fit),
            "calibration_rows_sha256": _row_payload_sha256(calibration),
            "fit_units": sorted(FIT_UNITS),
            "calibration_units": sorted(CALIBRATION_UNITS),
            "checkpoint_ids": list(CHECKPOINT_IDS),
            "checkpoint_tensor_sha256_by_id": {
                checkpoint_id: checkpoint_identities[checkpoint_id][0]
                for checkpoint_id in CHECKPOINT_IDS
            },
            "checkpoint_file_sha256_by_id": {
                checkpoint_id: checkpoint_identities[checkpoint_id][1]
                for checkpoint_id in CHECKPOINT_IDS
            },
            "fit_trace_sha256": sorted(row["trace_sha256"] for row in fit),
            "calibration_trace_sha256": sorted(
                row["trace_sha256"] for row in calibration
            ),
            "fit_calibration_units_disjoint": True,
            "target_rows_used": 0,
        },
        "decision_rule": {
            "adapt": "delta_hat - epsilon > 0",
            "freeze": "delta_hat + epsilon < 0",
            "otherwise": "ABSTAIN",
            "abstain_realized_action": "frozen",
        },
    }
    core["gate_sha256"] = stable_sha256(core)
    validate_gate_document(core)
    return core


def validate_gate_document(document: Mapping[str, Any]) -> None:
    if document.get("schema") != "kbound_cct20_ridge_gate_v1":
        raise IntegrityError("unknown CCT-20 gate schema")
    if tuple(document.get("feature_names", ())) != FEATURE_NAMES:
        raise IntegrityError("gate feature schema drift")
    if document.get("status") != "SEALED_DEVELOPMENT_ONLY":
        raise IntegrityError("gate is not sealed as development-only")
    calibration = document.get("calibration", {})
    if calibration.get("alpha") != CALIBRATION_ALPHA:
        raise IntegrityError("gate calibration alpha drift")
    if calibration.get("method") != "exact_rank_over_location_max_checkpoint_residual":
        raise IntegrityError("gate calibration method drift")
    if int(calibration.get("n_independent_units", 0)) != 9:
        raise IntegrityError("gate must have exactly nine independent calibration units")
    epsilon = _finite_number(calibration.get("epsilon"), field="calibration.epsilon")
    if epsilon < 0.0:
        raise IntegrityError("gate epsilon cannot be negative")
    residuals = calibration.get("residuals_sorted", ())
    if len(residuals) != 9 or any(
        _finite_number(value, field="calibration.residual") < 0.0 for value in residuals
    ):
        raise IntegrityError("gate must store nine finite non-negative calibration residuals")
    if list(residuals) != sorted(float(value) for value in residuals):
        raise IntegrityError("gate calibration residuals are not sorted")
    if epsilon != max(float(value) for value in residuals):
        raise IntegrityError("alpha=0.10 exact-rank radius must be the maximum of nine residuals")
    ridge = document.get("ridge", {})
    if (
        ridge.get("penalty") != RIDGE_PENALTY
        or ridge.get("intercept_unpenalized") is not True
        or ridge.get("standardization")
        != "development_fit_population_sd; zero_sd_replaced_by_one"
    ):
        raise IntegrityError("ridge penalty drift")
    _finite_number(ridge.get("intercept"), field="ridge.intercept")
    for field in ("coefficients", "fit_means", "fit_scales"):
        values = ridge.get(field, ())
        if len(values) != len(FEATURE_NAMES) or any(
            not math.isfinite(float(value)) for value in values
        ):
            raise IntegrityError(f"ridge.{field} shape/value drift")
    if any(float(value) <= 0.0 for value in ridge["fit_scales"]):
        raise IntegrityError("ridge fit scales must be strictly positive")
    provenance = document.get("development_provenance", {})
    if (
        provenance.get("fit_trace_count") != 10
        or provenance.get("fit_unit_count") != 2
        or provenance.get("calibration_trace_count") != 45
        or provenance.get("calibration_unit_count") != 9
        or provenance.get("fit_units") != sorted(FIT_UNITS)
        or provenance.get("calibration_units") != sorted(CALIBRATION_UNITS)
        or provenance.get("checkpoint_ids") != list(CHECKPOINT_IDS)
        or provenance.get("fit_calibration_units_disjoint") is not True
        or provenance.get("target_rows_used") != 0
    ):
        raise IntegrityError("gate development-only provenance contract drift")
    tensor_by_id = provenance.get("checkpoint_tensor_sha256_by_id")
    file_by_id = provenance.get("checkpoint_file_sha256_by_id")
    if (
        not isinstance(tensor_by_id, Mapping)
        or not isinstance(file_by_id, Mapping)
        or set(tensor_by_id) != set(CHECKPOINT_IDS)
        or set(file_by_id) != set(CHECKPOINT_IDS)
    ):
        raise IntegrityError("gate checkpoint identity provenance is incomplete")
    for checkpoint_id in CHECKPOINT_IDS:
        require_sha256(
            tensor_by_id.get(checkpoint_id),
            field=f"checkpoint_tensor_sha256_by_id.{checkpoint_id}",
        )
        require_sha256(
            file_by_id.get(checkpoint_id),
            field=f"checkpoint_file_sha256_by_id.{checkpoint_id}",
        )
    if len(set(tensor_by_id.values())) != 5 or len(set(file_by_id.values())) != 5:
        raise IntegrityError("gate does not preserve five distinct checkpoint identities")
    fit_trace_hashes = provenance.get("fit_trace_sha256")
    calibration_trace_hashes = provenance.get("calibration_trace_sha256")
    if not (
        isinstance(fit_trace_hashes, list)
        and len(fit_trace_hashes) == 10
        and isinstance(calibration_trace_hashes, list)
        and len(calibration_trace_hashes) == 45
        and len(set(fit_trace_hashes + calibration_trace_hashes)) == 55
    ):
        raise IntegrityError("gate does not preserve 55 distinct development trace hashes")
    for trace_hash in fit_trace_hashes + calibration_trace_hashes:
        require_sha256(trace_hash, field="development_trace_sha256")
    for field in (
        "fit_trace_ids_sha256",
        "calibration_trace_ids_sha256",
        "fit_rows_sha256",
        "calibration_rows_sha256",
    ):
        require_value = provenance.get(field)
        if (
            not isinstance(require_value, str)
            or len(require_value) != 64
            or any(character not in "0123456789abcdef" for character in require_value)
        ):
            raise IntegrityError(f"gate development_provenance.{field} is not a SHA-256")
    support = document.get("support", {})
    if (
        support.get("primary") != "finite_values_and_exact_feature_schema"
        or support.get("failure_action") != "ABSTAIN"
        or support.get("mahalanobis_role") != "diagnostic_only_never_changes_action"
    ):
        raise IntegrityError("gate support contract drift")
    center = np.asarray(support.get("diagnostic_center", ()), dtype=np.float64)
    precision = np.asarray(support.get("diagnostic_precision", ()), dtype=np.float64)
    if (
        center.shape != (len(FEATURE_NAMES),)
        or precision.shape != (len(FEATURE_NAMES), len(FEATURE_NAMES))
        or not np.isfinite(center).all()
        or not np.isfinite(precision).all()
    ):
        raise IntegrityError("gate diagnostic support arrays drifted")
    if document.get("decision_rule") != {
        "adapt": "delta_hat - epsilon > 0",
        "freeze": "delta_hat + epsilon < 0",
        "otherwise": "ABSTAIN",
        "abstain_realized_action": "frozen",
    }:
        raise IntegrityError("gate decision rule drift")
    claimed = document.get("gate_sha256")
    unsigned = dict(document)
    unsigned.pop("gate_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("gate_sha256 does not match the gate document")


def apply_gate(document: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the sealed gate; any schema/support failure returns ABSTAIN."""

    validate_gate_document(document)
    try:
        vector = _feature_vector(features)
    except IntegrityError as exc:
        return {
            "decision": "ABSTAIN",
            "support_status": "FAIL_CLOSED",
            "support_reasons": [str(exc)],
            "delta_hat": None,
            "epsilon": float(document["calibration"]["epsilon"]),
        }
    support = document["support"]
    if support.get("primary") != "finite_values_and_exact_feature_schema":
        raise IntegrityError("gate primary support rule drift")
    ridge = document["ridge"]
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = float(
            _predict_arrays(
                vector.reshape(1, -1),
                means=np.asarray(ridge["fit_means"], dtype=np.float64),
                scales=np.asarray(ridge["fit_scales"], dtype=np.float64),
                intercept=float(ridge["intercept"]),
                coefficients=np.asarray(ridge["coefficients"], dtype=np.float64),
            )[0]
        )
    epsilon = float(document["calibration"]["epsilon"])
    if not math.isfinite(prediction):
        return {
            "decision": "ABSTAIN",
            "support_status": "FAIL_CLOSED",
            "support_reasons": ["ridge prediction is numerically non-finite"],
            "delta_hat": None,
            "epsilon": epsilon,
        }
    center = np.asarray(support["diagnostic_center"], dtype=np.float64)
    precision = np.asarray(support["diagnostic_precision"], dtype=np.float64)
    difference = vector - center
    with np.errstate(over="ignore", invalid="ignore"):
        mahalanobis_squared = float(difference @ precision @ difference)
    mahalanobis = (
        math.sqrt(max(0.0, mahalanobis_squared))
        if math.isfinite(mahalanobis_squared)
        else None
    )
    if prediction - epsilon > 0.0:
        decision, status, reasons = "ADAPT", "IN_SUPPORT", []
    elif prediction + epsilon < 0.0:
        decision, status, reasons = "FREEZE", "IN_SUPPORT", []
    else:
        decision, status, reasons = "ABSTAIN", "IN_SUPPORT", ["interval crosses zero"]
    return {
        "decision": decision,
        "support_status": status,
        "support_reasons": reasons,
        "delta_hat": prediction,
        "epsilon": epsilon,
        "lower": prediction - epsilon,
        "upper": prediction + epsilon,
        "mahalanobis_diagnostic": mahalanobis,
    }
