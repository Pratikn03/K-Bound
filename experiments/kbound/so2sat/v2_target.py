"""Fail-closed v2 authorization for the So2Sat target boundary.

The legacy receipts prove only that bytes did not change after they were
written.  This module supplies the missing semantic gate: target authorization
is derived from an independently replayed, receipt-verified 95-cell
gate-calibration screen.  No caller-provided ``passed`` flag is trusted.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .development import validate_candidate_bundle, validate_selection
from .features import feature_vector
from .gate import CHECKPOINT_IDS, validate_gate_document
from .integrity import (
    IntegrityError,
    load_verified_json_mapping_with_receipt,
    read_secure_json_mapping_pair_once,
    stable_sha256,
)
from .protocol import (
    PROTOCOL_BASENAME,
    PROTOCOL_ID,
    PROTOCOL_RECEIPT_BASENAME,
    validate_protocol,
)
from .target_contract import (
    EXECUTION_SEAL_SCHEMA,
    PRODUCTION_MODE,
    TEST_ONLY_MODE,
    artifact_binding,
    validate_self_hash,
)

GATE_SCREEN_SCHEMA = "kbound_so2sat_canonical_gate_screen_v2"
TARGET_AUTHORIZATION_SCHEMA = "kbound_so2sat_target_authorization_v2"
GATE_SCREEN_CELL_COUNT = 95
MINIMUM_DIRECT_ACTION_CELL_FRACTION = 0.20
MINIMUM_DIRECT_ACTION_CITY_COUNT = 2


def _load_receipted_mapping(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return load_verified_json_mapping_with_receipt(path)


def _load_checked_in_protocol_pair(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path).expanduser()
    if source.name != PROTOCOL_BASENAME:
        raise IntegrityError(f"checked-in protocol must be named {PROTOCOL_BASENAME}")
    receipt_path = source.with_name(PROTOCOL_RECEIPT_BASENAME)
    (protocol, protocol_payload), (receipt, _) = read_secure_json_mapping_pair_once(
        source,
        receipt_path,
    )
    validate_protocol(protocol)
    expected_receipt = {
        "schema": "kbound_so2sat_protocol_receipt_v1",
        "artifact": PROTOCOL_BASENAME,
        "artifact_bytes": len(protocol_payload),
        "artifact_sha256": hashlib.sha256(protocol_payload).hexdigest(),
        "canonical_document_sha256": stable_sha256(protocol),
    }
    if receipt != expected_receipt:
        raise IntegrityError("checked-in So2Sat protocol receipt mismatch")
    return protocol, receipt


def _validate_screen_inputs(
    *,
    protocol: Mapping[str, Any],
    execution_seal: Mapping[str, Any],
    selection: Mapping[str, Any],
    calibration_bundle: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> Mapping[str, Any]:
    validate_protocol(protocol)
    validate_gate_document(gate)
    binding = gate.get("study_binding")
    if not isinstance(binding, Mapping):  # defensive narrowing after schema validation
        raise IntegrityError("gate study binding is missing or malformed")
    validate_selection(selection, study_binding=binding)
    validate_candidate_bundle(calibration_bundle, study_binding=binding)
    validate_self_hash(execution_seal, field="execution_seal_sha256")

    mode = execution_seal.get("execution_mode")
    expected_status = (
        "SEALED_BEFORE_ANY_TARGET_PIXEL_ACCESS"
        if mode == PRODUCTION_MODE
        else "TEST_ONLY_SEALED_WITH_SYNTHETIC_OR_INJECTED_DEPENDENCIES"
        if mode == TEST_ONLY_MODE
        else None
    )
    if (
        execution_seal.get("schema") != EXECUTION_SEAL_SCHEMA
        or execution_seal.get("status") != expected_status
        or execution_seal.get("protocol_id") != PROTOCOL_ID
    ):
        raise IntegrityError("unknown or unsealed execution seal for v2 authorization")
    if calibration_bundle.get("role") != "gate_cal":
        raise IntegrityError("v2 gate screen requires the selected gate-calibration bundle")
    selected = selection.get("selected_candidate_id")
    if (
        selected is None
        or calibration_bundle["candidate_spec"]["candidate_id"] != selected
        or execution_seal.get("candidate_id") != selected
        or execution_seal.get("selected_candidate_sha256") != selection.get("selection_sha256")
    ):
        raise IntegrityError("selection, calibration bundle, and execution seal disagree")

    protocol_document_sha256 = stable_sha256(dict(protocol))
    if (
        binding.get("protocol_document_sha256") != protocol_document_sha256
        or execution_seal.get("protocol_document_sha256") != protocol_document_sha256
        or execution_seal.get("study_binding_sha256") != binding.get("binding_sha256")
        or execution_seal.get("manifest_sha256") != binding.get("manifest_sha256")
        or execution_seal.get("population_identity_sha256") != binding.get("population_identity_sha256")
        or execution_seal.get("gate_sha256") != gate.get("gate_sha256")
    ):
        raise IntegrityError("protocol, seal, selection, bundle, and gate chain mismatch")
    provenance = gate.get("development_provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("calibration_trace_count") != GATE_SCREEN_CELL_COUNT
        or provenance.get("calibration_rows_sha256") != calibration_bundle.get("gate_rows_sha256")
    ):
        raise IntegrityError("gate does not bind the canonical 95-cell calibration bundle")
    audit = execution_seal.get("seal_creation_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("target_pixels_opened") != 0
        or audit.get("target_labels_opened") != 0
    ):
        raise IntegrityError("execution seal was not created before target access")
    return binding


def recompute_gate_screen(
    *,
    protocol: Mapping[str, Any],
    execution_seal: Mapping[str, Any],
    selection: Mapping[str, Any],
    calibration_bundle: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the canonical 19-city by 5-checkpoint development screen."""

    binding = _validate_screen_inputs(
        protocol=protocol,
        execution_seal=execution_seal,
        selection=selection,
        calibration_bundle=calibration_bundle,
        gate=gate,
    )
    ridge = gate["ridge"]
    means = np.asarray(ridge["fit_means"], dtype=np.float64)
    scales = np.asarray(ridge["fit_scales"], dtype=np.float64)
    coefficients = np.asarray(ridge["coefficients"], dtype=np.float64)
    intercept = float(ridge["intercept"])
    epsilon = float(gate["calibration"]["epsilon"])
    if (
        not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or not np.isfinite(coefficients).all()
        or not math.isfinite(intercept)
        or not math.isfinite(epsilon)
        or epsilon < 0.0
    ):
        raise IntegrityError("gate screen cannot use malformed ridge or radius values")

    rows: list[dict[str, Any]] = []
    for cell in calibration_bundle["cells"]:
        gate_row = cell["gate_row"]
        vector = feature_vector(gate_row["feature_document"])
        delta_hat = float(intercept + np.sum(((vector - means) / scales) * coefficients, dtype=np.float64))
        lower = delta_hat - epsilon
        upper = delta_hat + epsilon
        decision = "ADAPT" if lower > 0.0 else "FREEZE" if upper < 0.0 else "ABSTAIN"
        rows.append(
            {
                "city_id": cell["city_id"],
                "checkpoint_id": cell["checkpoint_id"],
                "trace_sha256": gate_row["trace_sha256"],
                "feature_sha256": gate_row["feature_document"]["feature_sha256"],
                "delta_hat": delta_hat,
                "epsilon": epsilon,
                "lower": lower,
                "upper": upper,
                "decision": decision,
                "realized_action": "ADAPT" if decision == "ADAPT" else "FREEZE",
            }
        )
    rows.sort(key=lambda row: (row["city_id"], row["checkpoint_id"]))
    identities = [(row["city_id"], row["checkpoint_id"]) for row in rows]
    expected_identities = {(city, checkpoint) for city in binding["gate_cal_cities"] for checkpoint in CHECKPOINT_IDS}
    if (
        len(rows) != GATE_SCREEN_CELL_COUNT
        or len(set(identities)) != GATE_SCREEN_CELL_COUNT
        or set(identities) != expected_identities
    ):
        raise IntegrityError("canonical gate screen is partial, duplicated, or stale")

    decisions = [row["decision"] for row in rows]
    decision_counts = {decision: decisions.count(decision) for decision in ("ADAPT", "FREEZE", "ABSTAIN")}
    direct_action_cities = {
        decision: sorted({str(row["city_id"]) for row in rows if row["decision"] == decision})
        for decision in ("ADAPT", "FREEZE")
    }
    minimum_cells = int(math.ceil(GATE_SCREEN_CELL_COUNT * MINIMUM_DIRECT_ACTION_CELL_FRACTION))
    checks = {
        "complete_95_cell_grid": True,
        "meaningful_adapt_cell_exposure": decision_counts["ADAPT"] >= minimum_cells,
        "meaningful_freeze_cell_exposure": decision_counts["FREEZE"] >= minimum_cells,
        "meaningful_adapt_city_exposure": len(direct_action_cities["ADAPT"]) >= MINIMUM_DIRECT_ACTION_CITY_COUNT,
        "meaningful_freeze_city_exposure": len(direct_action_cities["FREEZE"]) >= MINIMUM_DIRECT_ACTION_CITY_COUNT,
    }
    screen: dict[str, Any] = {
        "schema": GATE_SCREEN_SCHEMA,
        "status": (
            "CANONICAL_95_CELL_GATE_SCREEN_PASSED"
            if all(checks.values())
            else "CANONICAL_95_CELL_GATE_SCREEN_FAILED_NO_TARGET_ACCESS"
        ),
        "protocol_document_sha256": binding["protocol_document_sha256"],
        "execution_seal_sha256": execution_seal["execution_seal_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "calibration_bundle_sha256": calibration_bundle["bundle_sha256"],
        "gate_sha256": gate["gate_sha256"],
        "cell_count": len(rows),
        "cells": rows,
        "decision_counts": decision_counts,
        "direct_action_cities": direct_action_cities,
        "minimum_direct_action_cells": minimum_cells,
        "minimum_direct_action_cities": MINIMUM_DIRECT_ACTION_CITY_COUNT,
        "checks": checks,
        "passed": all(checks.values()),
        "target_pixels_read": 0,
        "target_labels_read": 0,
        "target_inputs": [],
    }
    screen["gate_screen_sha256"] = stable_sha256(screen)
    return screen


def authorize_target_execution(
    *,
    protocol_path: str | Path,
    execution_seal_path: str | Path,
    selection_path: str | Path,
    calibration_bundle_path: str | Path,
    gate_path: str | Path,
    gate_screen_path: str | Path,
) -> dict[str, Any]:
    """Authorize only an exact, receipt-verified replay of the canonical screen."""

    protocol, protocol_receipt = _load_checked_in_protocol_pair(protocol_path)
    execution_seal, execution_receipt = _load_receipted_mapping(execution_seal_path)
    selection, selection_receipt = _load_receipted_mapping(selection_path)
    calibration_bundle, calibration_receipt = _load_receipted_mapping(calibration_bundle_path)
    gate, gate_receipt = _load_receipted_mapping(gate_path)
    submitted_screen, screen_receipt = _load_receipted_mapping(gate_screen_path)

    if execution_seal.get("selected_candidate_artifact") != artifact_binding(selection_receipt):
        raise IntegrityError("execution seal does not bind the receipted selection")
    if execution_seal.get("gate_artifact") != artifact_binding(gate_receipt):
        raise IntegrityError("execution seal does not bind the receipted gate")
    if (
        gate["study_binding"]["protocol_file_sha256"] != protocol_receipt["artifact_sha256"]
        or gate["study_binding"]["protocol_document_sha256"] != protocol_receipt["canonical_document_sha256"]
    ):
        raise IntegrityError("gate does not bind the receipted protocol")

    first = recompute_gate_screen(
        protocol=protocol,
        execution_seal=execution_seal,
        selection=selection,
        calibration_bundle=calibration_bundle,
        gate=gate,
    )
    second = recompute_gate_screen(
        protocol=protocol,
        execution_seal=execution_seal,
        selection=selection,
        calibration_bundle=calibration_bundle,
        gate=gate,
    )
    if first != second:
        raise IntegrityError("canonical gate screen changed across deterministic replay")
    if submitted_screen != first:
        raise IntegrityError("receipted gate screen differs from canonical recomputation")
    if first["passed"] is not True:
        raise IntegrityError(
            "canonical gate screen failed meaningful ADAPT/FREEZE exposure; target access is forbidden"
        )

    authorization: dict[str, Any] = {
        "schema": TARGET_AUTHORIZATION_SCHEMA,
        "status": "AUTHORIZED_AFTER_CANONICAL_95_CELL_SCREEN",
        "protocol_artifact_sha256": protocol_receipt["artifact_sha256"],
        "protocol_document_sha256": protocol_receipt["canonical_document_sha256"],
        "execution_seal_artifact_sha256": execution_receipt["artifact_sha256"],
        "execution_seal_sha256": execution_seal["execution_seal_sha256"],
        "selection_artifact_sha256": selection_receipt["artifact_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "calibration_bundle_artifact_sha256": calibration_receipt["artifact_sha256"],
        "calibration_bundle_sha256": calibration_bundle["bundle_sha256"],
        "gate_artifact_sha256": gate_receipt["artifact_sha256"],
        "gate_sha256": gate["gate_sha256"],
        "gate_screen_artifact_sha256": screen_receipt["artifact_sha256"],
        "gate_screen_canonical_document_sha256": screen_receipt["canonical_document_sha256"],
        "gate_screen_sha256": first["gate_screen_sha256"],
        "gate_screen_cell_count": first["cell_count"],
        "target_pixels_read": 0,
        "target_labels_read": 0,
        "target_inputs": [],
    }
    authorization["authorization_sha256"] = stable_sha256(authorization)
    return copy.deepcopy(authorization)


__all__ = [
    "GATE_SCREEN_SCHEMA",
    "TARGET_AUTHORIZATION_SCHEMA",
    "GATE_SCREEN_CELL_COUNT",
    "recompute_gate_screen",
    "authorize_target_execution",
]
