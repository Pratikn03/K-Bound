"""Focused tests for the label-free So2Sat feature and ridge-gate layer."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.kbound.so2sat.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    FEATURE_STATUS,
    N_CLASSES,
    extract_label_free_features,
    validate_feature_document,
)
from experiments.kbound.so2sat.gate import (
    CALIBRATION_ALPHA,
    CALIBRATION_CITY_COUNT,
    CHECKPOINT_IDS,
    CONFORMAL_RANK,
    FIT_CITY_COUNT,
    RIDGE_PENALTY,
    apply_gate,
    fit_calibrate_ridge_gate,
    load_action_with_receipt,
    load_gate_with_receipt,
    load_study_binding,
    replay_action,
    trace_identity_sha256,
    validate_action_document,
    validate_gate_document,
    validate_study_binding,
    write_action_with_receipt,
    write_gate_with_receipt,
)
from experiments.kbound.so2sat.integrity import (
    IntegrityError,
    stable_sha256,
    write_immutable_json_with_receipt,
)
from experiments.kbound.so2sat.protocol import PROTOCOL_ID

FIT_CITIES = [f"fitcity{index:02d}" for index in range(FIT_CITY_COUNT)]
CALIBRATION_CITIES = [f"calcity{index:02d}" for index in range(CALIBRATION_CITY_COUNT)]
TARGET_CITIES = [f"targetcity{index:02d}" for index in range(10)]


def _minimal_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": "kbound_so2sat_label_free_population_manifest_v1",
        "status": "LABEL_FREE_METADATA_POPULATION_VERIFIED",
        "protocol_id": PROTOCOL_ID,
        "protocol_identity": {
            "file_sha256": stable_sha256({"protocol": "file"}),
            "canonical_document_sha256": stable_sha256({"protocol": "document"}),
        },
        "cities": {
            "training_roles": {
                "source_fit_ineligible": ["ineligible"],
                "source_fit_core": ["core"],
                "gate_fit": FIT_CITIES,
                "gate_cal": CALIBRATION_CITIES,
            },
            "target": TARGET_CITIES,
        },
        "population_identity_sha256": stable_sha256({"population": "synthetic"}),
    }
    manifest["manifest_sha256"] = stable_sha256(manifest)
    return manifest


def _study_binding(tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / "synthetic-population-manifest.json"
    write_immutable_json_with_receipt(path, _minimal_manifest())
    return load_study_binding(path)


def _manual_feature_document(signal: float, identity: str) -> dict[str, Any]:
    entropy_change = 0.05 * signal
    confidence_change = 0.05 * signal
    features = {
        "frozen_mean_entropy": 0.60,
        "adapted_mean_entropy": 0.60 - entropy_change,
        "entropy_change": entropy_change,
        "frozen_mean_confidence": 0.50,
        "adapted_mean_confidence": 0.50 + confidence_change,
        "confidence_change": confidence_change,
        "prediction_disagreement": 0.10 + 0.02 * abs(signal),
        "marginal_jensen_shannon_divergence": 0.02 + 0.01 * abs(signal),
        "normalized_predicted_class_effective_count": 0.60 + 0.02 * signal,
        "normalized_adapter_update_norm": 0.40 + 0.10 * signal,
        "batchnorm_source_statistic_divergence": 0.20 + 0.03 * abs(signal),
    }
    document: dict[str, Any] = {
        "schema": FEATURE_SCHEMA,
        "status": FEATURE_STATUS,
        "n_probe_images": 64,
        "n_classes": N_CLASSES,
        "feature_names": list(FEATURE_NAMES),
        "frozen_logits_tensor_sha256": stable_sha256({"identity": identity, "tensor": "frozen", "signal": signal}),
        "adapted_logits_tensor_sha256": stable_sha256({"identity": identity, "tensor": "adapted", "signal": signal}),
        "features": features,
    }
    document["feature_sha256"] = stable_sha256(document)
    validate_feature_document(document)
    return document


def _development_row(
    *,
    binding: dict[str, Any],
    role: str,
    city_id: str,
    checkpoint_id: str,
    signal: float,
    benefit: float,
) -> dict[str, Any]:
    tensor_sha = stable_sha256({"checkpoint": checkpoint_id, "kind": "tensor"})
    file_sha = stable_sha256({"checkpoint": checkpoint_id, "kind": "file"})
    trace_id = f"{role}:{city_id}:{checkpoint_id}"
    partition_sha = stable_sha256({"city": city_id, "partition": "west-east"})
    feature_document = _manual_feature_document(signal, trace_id)
    trace_sha = trace_identity_sha256(
        role=role,
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
    return {
        "role": role,
        "city_id": city_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_tensor_sha256": tensor_sha,
        "checkpoint_file_sha256": file_sha,
        "trace_id": trace_id,
        "trace_sha256": trace_sha,
        "partition_sha256": partition_sha,
        "manifest_sha256": binding["manifest_sha256"],
        "population_identity_sha256": binding["population_identity_sha256"],
        "protocol_file_sha256": binding["protocol_file_sha256"],
        "protocol_document_sha256": binding["protocol_document_sha256"],
        "feature_document": feature_document,
        "observed_benefit": benefit,
    }


def _mixed_effect_rows(
    binding: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit: list[dict[str, Any]] = []
    for city_index, city in enumerate(FIT_CITIES):
        city_signal = -1.4 + 2.8 * city_index / (len(FIT_CITIES) - 1)
        city_effect = 0.012 * math.sin(city_index)
        for checkpoint_index, checkpoint in enumerate(CHECKPOINT_IDS):
            signal = city_signal + 0.04 * (checkpoint_index - 2)
            checkpoint_effect = 0.004 * (checkpoint_index - 2)
            benefit = 0.45 * signal + city_effect + checkpoint_effect
            fit.append(
                _development_row(
                    binding=binding,
                    role="gate_fit",
                    city_id=city,
                    checkpoint_id=checkpoint,
                    signal=signal,
                    benefit=benefit,
                )
            )

    calibration: list[dict[str, Any]] = []
    for city_index, city in enumerate(CALIBRATION_CITIES):
        city_signal = -1.35 + 2.7 * city_index / (len(CALIBRATION_CITIES) - 1)
        city_effect = 0.012 * math.cos(city_index)
        calibration_shift = ((city_index % 5) - 2) * 0.004
        for checkpoint_index, checkpoint in enumerate(CHECKPOINT_IDS):
            signal = city_signal + 0.04 * (checkpoint_index - 2)
            checkpoint_effect = 0.004 * (checkpoint_index - 2)
            benefit = 0.45 * signal + city_effect + checkpoint_effect + calibration_shift
            calibration.append(
                _development_row(
                    binding=binding,
                    role="gate_cal",
                    city_id=city,
                    checkpoint_id=checkpoint,
                    signal=signal,
                    benefit=benefit,
                )
            )
    return fit, calibration


def _target_action(
    gate: dict[str, Any],
    *,
    city_id: str,
    signal: float,
    checkpoint_id: str = "0",
) -> dict[str, Any]:
    feature = _manual_feature_document(signal, f"target:{city_id}:{checkpoint_id}")
    provenance = gate["development_provenance"]
    tensor_sha = provenance["checkpoint_tensor_sha256_by_id"][checkpoint_id]
    file_sha = provenance["checkpoint_file_sha256_by_id"][checkpoint_id]
    trace_id = f"target_probe:{city_id}:{checkpoint_id}"
    partition_sha = stable_sha256({"city": city_id, "partition": "validation-probe"})
    binding = gate["study_binding"]
    trace_sha = trace_identity_sha256(
        role="target_probe",
        city_id=city_id,
        checkpoint_id=checkpoint_id,
        checkpoint_tensor_sha256=tensor_sha,
        checkpoint_file_sha256=file_sha,
        trace_id=trace_id,
        partition_sha256=partition_sha,
        feature_sha256=feature["feature_sha256"],
        manifest_sha256=binding["manifest_sha256"],
        population_identity_sha256=binding["population_identity_sha256"],
        protocol_file_sha256=binding["protocol_file_sha256"],
        protocol_document_sha256=binding["protocol_document_sha256"],
    )
    return apply_gate(
        gate,
        feature,
        city_id=city_id,
        checkpoint_id=checkpoint_id,
        checkpoint_tensor_sha256=tensor_sha,
        checkpoint_file_sha256=file_sha,
        trace_id=trace_id,
        trace_sha256=trace_sha,
        partition_sha256=partition_sha,
    )


def test_feature_extractor_is_exactly_17_class_label_free_and_content_bound() -> None:
    frozen = np.zeros((5, N_CLASSES), dtype=np.float64)
    adapted = frozen.copy()
    adapted[:, 3] = 2.0
    document = extract_label_free_features(
        frozen,
        adapted,
        normalized_adapter_update_norm=0.125,
        batchnorm_source_statistic_divergence=0.25,
    )
    validate_feature_document(document)
    assert document["n_classes"] == 17
    assert tuple(document["feature_names"]) == FEATURE_NAMES
    assert set(document["features"]) == set(FEATURE_NAMES)
    assert document["features"]["normalized_adapter_update_norm"] == 0.125
    assert document["features"]["batchnorm_source_statistic_divergence"] == 0.25
    assert document["frozen_logits_tensor_sha256"] != document["adapted_logits_tensor_sha256"]
    assert all("label" not in key.casefold() for key in document)
    assert all("label" not in key.casefold() for key in document["features"])


def test_feature_extractor_rejects_wrong_classes_nonfinite_and_negative_diagnostics() -> None:
    logits = np.zeros((3, N_CLASSES), dtype=np.float64)
    with pytest.raises(IntegrityError, match="shape"):
        extract_label_free_features(
            np.zeros((3, N_CLASSES - 1)),
            np.zeros((3, N_CLASSES - 1)),
            normalized_adapter_update_norm=0.1,
            batchnorm_source_statistic_divergence=0.2,
        )
    nonfinite = logits.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(IntegrityError, match="NaN or Infinity"):
        extract_label_free_features(
            nonfinite,
            logits,
            normalized_adapter_update_norm=0.1,
            batchnorm_source_statistic_divergence=0.2,
        )
    with pytest.raises(IntegrityError, match="non-negative"):
        extract_label_free_features(
            logits,
            logits,
            normalized_adapter_update_norm=-0.1,
            batchnorm_source_statistic_divergence=0.2,
        )
    with pytest.raises(IntegrityError, match="non-negative"):
        extract_label_free_features(
            logits,
            logits,
            normalized_adapter_update_norm=0.1,
            batchnorm_source_statistic_divergence=-0.2,
        )


def test_mixed_effect_gate_uses_fixed_ridge_city_max_and_rank_18_of_19(
    tmp_path: Path,
) -> None:
    binding = _study_binding(tmp_path)
    fit, calibration = _mixed_effect_rows(binding)
    gate = fit_calibrate_ridge_gate(fit, calibration, study_binding=binding)
    validate_gate_document(gate)
    assert gate["ridge"]["penalty"] == RIDGE_PENALTY
    assert gate["calibration"]["alpha"] == CALIBRATION_ALPHA
    assert gate["development_provenance"]["fit_trace_count"] == 45
    assert gate["development_provenance"]["calibration_trace_count"] == 95
    assert gate["calibration"]["n_independent_cities"] == 19
    assert gate["calibration"]["order_statistic_rank_one_based"] == CONFORMAL_RANK
    residuals = gate["calibration"]["residuals_sorted"]
    assert gate["calibration"]["epsilon"] == residuals[17]
    assert gate["calibration"]["epsilon"] != residuals[-1]
    assert fit_calibrate_ridge_gate(reversed(fit), reversed(calibration), study_binding=binding) == gate

    decisions = {
        _target_action(gate, city_id=TARGET_CITIES[0], signal=-3.0)["decision"],
        _target_action(gate, city_id=TARGET_CITIES[1], signal=0.0)["decision"],
        _target_action(gate, city_id=TARGET_CITIES[2], signal=3.0)["decision"],
    }
    assert decisions == {"ADAPT", "FREEZE", "ABSTAIN"}


def test_gate_fails_closed_on_overlap_missing_cells_bad_class_and_nonfinite(
    tmp_path: Path,
) -> None:
    binding = _study_binding(tmp_path)
    fit, calibration = _mixed_effect_rows(binding)

    overlap = copy.deepcopy(binding)
    overlap["gate_cal_cities"][0] = overlap["gate_fit_cities"][0]
    overlap["gate_cal_cities"].sort()
    overlap["binding_sha256"] = stable_sha256({key: value for key, value in overlap.items() if key != "binding_sha256"})
    with pytest.raises(IntegrityError, match="overlap"):
        validate_study_binding(overlap)

    with pytest.raises(IntegrityError, match="9 cities x 5 checkpoints"):
        fit_calibrate_ridge_gate(fit[:-1], calibration, study_binding=binding)

    nonfinite = copy.deepcopy(fit)
    nonfinite[0]["observed_benefit"] = float("nan")
    with pytest.raises(IntegrityError, match="finite"):
        fit_calibrate_ridge_gate(nonfinite, calibration, study_binding=binding)

    wrong_class = copy.deepcopy(fit)
    wrong_class[0]["feature_document"]["n_classes"] = 16
    wrong_class[0]["feature_document"]["feature_sha256"] = stable_sha256(
        {key: value for key, value in wrong_class[0]["feature_document"].items() if key != "feature_sha256"}
    )
    with pytest.raises(IntegrityError, match="exactly 17 classes"):
        fit_calibrate_ridge_gate(wrong_class, calibration, study_binding=binding)

    label_contaminated = copy.deepcopy(fit)
    label_contaminated[0]["feature_document"]["label"] = 3
    label_contaminated[0]["feature_document"]["feature_sha256"] = stable_sha256(
        {key: value for key, value in label_contaminated[0]["feature_document"].items() if key != "feature_sha256"}
    )
    with pytest.raises(IntegrityError, match="unknown or missing"):
        fit_calibrate_ridge_gate(label_contaminated, calibration, study_binding=binding)

    bad_trace = copy.deepcopy(fit)
    bad_trace[0]["trace_sha256"] = stable_sha256("not-the-bound-trace")
    with pytest.raises(IntegrityError, match="trace identity hash mismatch"):
        fit_calibrate_ridge_gate(bad_trace, calibration, study_binding=binding)


def test_receipts_hashes_and_action_replay_reject_tampering(tmp_path: Path) -> None:
    binding = _study_binding(tmp_path / "binding")
    fit, calibration = _mixed_effect_rows(binding)
    gate = fit_calibrate_ridge_gate(fit, calibration, study_binding=binding)

    gate_path = tmp_path / "ridge-gate.json"
    gate_receipt = write_gate_with_receipt(gate_path, gate)
    assert gate_receipt["canonical_document_sha256"] == stable_sha256(gate)
    loaded_gate = load_gate_with_receipt(gate_path)
    assert loaded_gate == gate

    action = _target_action(gate, city_id=TARGET_CITIES[0], signal=0.0)
    assert action["decision"] == "ABSTAIN"
    assert action["realized_action"] == "FREEZE"
    assert replay_action(action, gate=gate) == action

    action_path = tmp_path / "target-action.json"
    action_receipt = write_action_with_receipt(action_path, action, gate=gate)
    assert action_receipt["canonical_document_sha256"] == stable_sha256(action)
    assert load_action_with_receipt(action_path, gate=gate) == action

    tampered = copy.deepcopy(action)
    tampered["decision"] = "ADAPT"
    tampered["realized_action"] = "ADAPT"
    tampered["action_sha256"] = stable_sha256({key: value for key, value in tampered.items() if key != "action_sha256"})
    with pytest.raises(IntegrityError, match="does not replay"):
        validate_action_document(tampered, gate=gate)

    receipt_path = gate_path.with_name(gate_path.name + ".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifact_sha256"] = stable_sha256("tampered receipt")
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(IntegrityError, match="receipt file SHA-256 mismatch"):
        load_gate_with_receipt(gate_path)
