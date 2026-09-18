"""Fail-closed tests for the source-only So2Sat prospective-v2 lock."""

from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
from typing import Any

import pytest

from experiments.kbound.so2sat.integrity import (
    IntegrityError,
    stable_sha256,
    strict_json_load,
)

_DIGEST = "a" * 64
_V1_RESULT = Path(
    "experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/so2sat_candidate_selection.json"
)
_V1_TENT = Path(
    "experiments/kbound/results/so2sat_lcz42_prospective_v1/"
    "development_mps_bn_fix_v1/"
    "so2sat_tent_adam_bn_affine_probe_transfer_v1.gate_fit.json"
)

_REVIEWED_V2_RUNTIME_MODULES = (
    "__init__.py",
    "adapters.py",
    "development.py",
    "features.py",
    "gate.py",
    "integrity.py",
    "label_firewall.py",
    "metadata_manifest.py",
    "model.py",
    "precalibration_seal.py",
    "prospective_runner_v2.py",
    "prospective_v2.py",
    "protocol.py",
    "source_acceptance.py",
    "source_data.py",
    "source_preflight.py",
    "target_amendment.py",
    "target_contract.py",
    "target_inference.py",
    "target_runner.py",
    "train_source.py",
)


def _artifact(name: str, marker: str) -> dict[str, Any]:
    digest = marker * 64
    return {
        "schema": "kbound_so2sat_artifact_receipt_v2",
        "artifact_basename": name,
        "artifact_bytes": 123,
        "artifact_sha256": digest,
        "canonical_document_sha256": digest,
    }


def _source_bindings(controller: dict[str, Any]) -> dict[str, Any]:
    checkpoints = copy.deepcopy(controller["source_bindings"]["checkpoints"])
    return {
        "population_manifest_artifact": _artifact("population.json", "b"),
        "population_identity_sha256": "c" * 64,
        "source_postrun_acceptance_artifact": _artifact("acceptance.json", "d"),
        "source_postrun_acceptance_sha256": "e" * 64,
        "source_checkpoint_collection_artifact": _artifact("checkpoints.json", "f"),
        "source_checkpoint_collection_sha256": controller["source_bindings"]["checkpoint_collection_canonical_sha256"],
        "source_checkpoints": checkpoints,
        "source_normalizer_artifact": _artifact("normalizer.json", "1"),
        "source_normalizer_sha256": controller["source_bindings"]["normalizer_sha256"],
    }


def _runtime_bindings() -> dict[str, str]:
    from experiments.kbound.so2sat.prospective_v2 import (
        configuration_identity_v2,
        prospective_v2_code_identity,
    )

    return {
        "code_identity_sha256": prospective_v2_code_identity()["code_identity_sha256"],
        "configuration_identity_sha256": configuration_identity_v2(),
        "environment_identity_sha256": "4" * 64,
    }


def _opaque_target_identities() -> dict[str, dict[str, Any]]:
    return {
        "validation": {
            "container_role": "label_free_probe_pixels",
            "artifact_basename": "validation.h5",
            "artifact_bytes": 456,
            "raw_file_sha256": "5" * 64,
            "hashing_method": "sha256_raw_bytes_without_hdf5_deserialization",
            "hdf5_datasets_opened": 0,
        },
        "testing": {
            "container_role": "sealed_evaluation_pixels_and_outcomes",
            "artifact_basename": "testing.h5",
            "artifact_bytes": 789,
            "raw_file_sha256": "6" * 64,
            "hashing_method": "sha256_raw_bytes_without_hdf5_deserialization",
            "hdf5_datasets_opened": 0,
        },
    }


def _chronology() -> dict[str, Any]:
    return {
        "schema": "kbound_so2sat_zero_access_chronology_v2",
        "status": "DECLARED_ZERO_ACCESS_NOT_AN_EXTERNAL_TIMESTAMP",
        "ordered_events": [
            "v1_negative_result_closed",
            "v2_protocol_and_controller_fixed",
            "v2_precalibration_seal_created",
            "gate_calibration_may_begin_only_after_seal",
            "target_may_begin_only_after_gate_authorization",
        ],
        "gate_calibration_rows_read_before_v2_seal": 0,
        "gate_calibration_pixels_read_before_v2_seal": 0,
        "gate_calibration_labels_read_before_v2_seal": 0,
        "target_pixels_read_before_v2_seal": 0,
        "target_labels_read_before_v2_seal": 0,
        "external_timestamp_claimed": False,
    }


def _precalibration_seal() -> tuple[dict[str, Any], dict[str, Any]]:
    from experiments.kbound.so2sat.prospective_v2 import (
        build_precalibration_seal_v2,
        load_controller_v2,
        load_protocol_v2,
    )

    protocol = load_protocol_v2()
    controller = load_controller_v2()
    seal = build_precalibration_seal_v2(
        protocol=protocol,
        controller=controller,
        source_bindings=_source_bindings(controller),
        runtime_bindings=_runtime_bindings(),
        opaque_target_identities=_opaque_target_identities(),
        chronology=_chronology(),
    )
    return seal, controller


def _feature_values(controller: dict[str, Any]) -> dict[str, float]:
    return dict(zip(controller["feature_names"], controller["standardization"]["means"]))


def _feature_document(controller: dict[str, Any], *, marker: int) -> dict[str, Any]:
    from experiments.kbound.so2sat.features import FEATURE_NAMES

    selected = _feature_values(controller)
    frozen_entropy = 0.5
    features = {
        "frozen_mean_entropy": frozen_entropy,
        "adapted_mean_entropy": frozen_entropy - selected["entropy_change"],
        "entropy_change": selected["entropy_change"],
        "frozen_mean_confidence": 0.5,
        "adapted_mean_confidence": 0.5,
        "confidence_change": 0.0,
        "prediction_disagreement": selected["prediction_disagreement"],
        "marginal_jensen_shannon_divergence": selected["marginal_jensen_shannon_divergence"],
        "normalized_predicted_class_effective_count": 0.5,
        "normalized_adapter_update_norm": selected["normalized_adapter_update_norm"],
        "batchnorm_source_statistic_divergence": selected["batchnorm_source_statistic_divergence"],
    }
    document: dict[str, Any] = {
        "schema": "kbound_so2sat_label_free_probe_features_v1",
        "status": "LABEL_FREE_PROBE_FEATURES",
        "n_probe_images": 32,
        "n_classes": 17,
        "feature_names": list(FEATURE_NAMES),
        "frozen_logits_tensor_sha256": f"{marker + 1000:064x}",
        "adapted_logits_tensor_sha256": f"{marker + 2000:064x}",
        "features": features,
    }
    document["feature_sha256"] = stable_sha256(document)
    return document


def _calibration_bundle(
    seal: dict[str, Any],
    controller: dict[str, Any],
    *,
    large_residual: bool = False,
) -> dict[str, Any]:
    from experiments.kbound.so2sat.prospective_v2 import raw_prediction_v2

    cells: list[dict[str, Any]] = []
    feature_values = _feature_values(controller)
    city_probe_features = {str(seed): feature_values for seed in range(5)}
    for city_index in range(19):
        city_id = f"opaque-calibration-city-{city_index:02d}"
        for seed in range(5):
            checkpoint_id = str(seed)
            prediction = raw_prediction_v2(
                controller,
                checkpoint_id=checkpoint_id,
                city_probe_features=city_probe_features,
            )
            cells.append(
                {
                    "city_id": city_id,
                    "checkpoint_id": checkpoint_id,
                    "feature_document": _feature_document(
                        controller,
                        marker=city_index * 5 + seed + 1,
                    ),
                    "observed_benefit": prediction + (0.5 if large_residual else 0.0),
                    "trace_sha256": f"{city_index * 5 + seed + 1:064x}",
                }
            )
    document: dict[str, Any] = {
        "schema": "kbound_so2sat_gate_calibration_bundle_v2",
        "status": "COMPLETE_19_CITY_95_CELL_GATE_CALIBRATION",
        "precalibration_seal_sha256": seal["precalibration_seal_sha256"],
        "controller_sha256": controller["controller_sha256"],
        "action_unit": "city_checkpoint",
        "cells": cells,
        "gate_calibration_city_count": 19,
        "checkpoint_count": 5,
        "gate_calibration_rows_read": 95,
        "target_pixels_read": 0,
        "target_labels_read": 0,
        "target_inputs": [],
    }
    document["bundle_sha256"] = stable_sha256(document)
    return document


def test_v1_negative_result_remains_the_immutable_scientific_record() -> None:
    result = strict_json_load(_V1_RESULT)
    assert result["status"] == "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL"
    assert result["selected_candidate_id"] is None
    assert result["gate_cal_rows_read_before_selection"] == 0
    assert result["target_pixels_read"] == 0
    assert result["target_labels_read"] == 0


def test_reviewed_v2_runtime_module_closure_covers_every_relative_import() -> None:
    from experiments.kbound.so2sat import prospective_v2

    package = Path(prospective_v2.__file__).resolve().parent
    graph: dict[str, set[str]] = {}
    for module_path in package.glob("*.py"):
        dependencies: set[str] = set()
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue
            assert node.level == 1, f"unreviewed parent-relative import in {module_path.name}"
            imported_modules = (
                [node.module.split(".", 1)[0]] if node.module else [alias.name.split(".", 1)[0] for alias in node.names]
            )
            dependencies.update(f"{name}.py" for name in imported_modules if (package / f"{name}.py").is_file())
        graph[module_path.name] = dependencies

    observed: set[str] = set()
    pending = ["__init__.py", "prospective_v2.py", "prospective_runner_v2.py"]
    while pending:
        module = pending.pop()
        if module in observed:
            continue
        observed.add(module)
        pending.extend(sorted(graph[module] - observed))

    assert prospective_v2.PROSPECTIVE_V2_RUNTIME_MODULES == _REVIEWED_V2_RUNTIME_MODULES
    assert tuple(sorted(observed)) == tuple(sorted(prospective_v2.PROSPECTIVE_V2_RUNTIME_MODULES))


def test_v2_code_identity_is_deterministic_and_names_the_reviewed_closure() -> None:
    from experiments.kbound.so2sat import prospective_v2

    first = prospective_v2.prospective_v2_code_identity()
    second = prospective_v2.prospective_v2_code_identity()
    expected_names = (
        *prospective_v2.PROSPECTIVE_V2_RUNTIME_MODULES,
        prospective_v2.PROTOCOL_BASENAME,
        f"{prospective_v2.PROTOCOL_BASENAME}.receipt.json",
        prospective_v2.CONTROLLER_BASENAME,
        f"{prospective_v2.CONTROLLER_BASENAME}.receipt.json",
        prospective_v2.PRECALIBRATION_TEMPLATE_BASENAME,
    )

    assert first == second
    assert tuple(first["files_sha256"]) == expected_names
    assert first["code_identity_sha256"] == stable_sha256(first["files_sha256"])


@pytest.mark.parametrize("kind", ["protocol", "controller"])
def test_v2_configuration_loaders_consume_one_receipt_verified_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    from experiments.kbound.so2sat import integrity, prospective_v2

    real_secure_load = integrity.load_verified_json_mapping_with_receipt
    secure_calls: list[Path] = []

    def tracked_secure_load(
        artifact_path: str | Path,
        receipt_path: str | Path | None = None,
        *,
        receipt_schema: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        secure_calls.append(Path(artifact_path))
        return real_secure_load(
            artifact_path,
            receipt_path,
            receipt_schema=receipt_schema,
        )

    def forbidden_reopen(*args: object, **kwargs: object) -> object:
        raise AssertionError("configuration path was reopened after verification")

    monkeypatch.setattr(
        prospective_v2,
        "load_verified_json_mapping_with_receipt",
        tracked_secure_load,
        raising=False,
    )
    monkeypatch.setattr(prospective_v2, "strict_json_load", forbidden_reopen)
    monkeypatch.setattr(
        prospective_v2,
        "verify_artifact_receipt",
        forbidden_reopen,
        raising=False,
    )

    if kind == "protocol":
        document = prospective_v2.load_protocol_v2()
        receipt = prospective_v2.verify_protocol_v2_receipt()
        expected_path = prospective_v2.default_protocol_v2_path()
    else:
        document = prospective_v2.load_controller_v2()
        receipt = prospective_v2.verify_controller_v2_receipt()
        expected_path = prospective_v2.default_controller_v2_path()

    assert document["schema"].endswith("_v2")
    assert receipt["canonical_document_sha256"] == stable_sha256(document)
    assert secure_calls == [expected_path, expected_path]


def test_precalibration_seal_validation_uses_one_snapshot_per_configuration_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.kbound.so2sat import integrity, prospective_v2

    seal, _ = _precalibration_seal()
    real_secure_load = integrity.load_verified_json_mapping_with_receipt
    secure_calls: list[Path] = []

    def tracked_secure_load(
        artifact_path: str | Path,
        receipt_path: str | Path | None = None,
        *,
        receipt_schema: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        secure_calls.append(Path(artifact_path))
        return real_secure_load(
            artifact_path,
            receipt_path,
            receipt_schema=receipt_schema,
        )

    def forbidden_reopen(*args: object, **kwargs: object) -> object:
        raise AssertionError("seal validation reopened a configuration path")

    monkeypatch.setattr(
        prospective_v2,
        "load_verified_json_mapping_with_receipt",
        tracked_secure_load,
        raising=False,
    )
    monkeypatch.setattr(prospective_v2, "strict_json_load", forbidden_reopen)
    monkeypatch.setattr(
        prospective_v2,
        "verify_artifact_receipt",
        forbidden_reopen,
        raising=False,
    )

    prospective_v2.validate_precalibration_seal_v2(seal)

    assert secure_calls == [
        prospective_v2.default_controller_v2_path(),
        prospective_v2.default_protocol_v2_path(),
    ]


def test_precalibration_seal_builder_reuses_its_verified_configuration_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.kbound.so2sat import integrity, prospective_v2

    protocol = prospective_v2.load_protocol_v2()
    controller = prospective_v2.load_controller_v2()
    source_bindings = _source_bindings(controller)
    runtime_bindings = _runtime_bindings()
    identities = _opaque_target_identities()
    chronology = _chronology()
    real_secure_load = integrity.load_verified_json_mapping_with_receipt
    secure_calls: list[Path] = []

    def tracked_secure_load(
        artifact_path: str | Path,
        receipt_path: str | Path | None = None,
        *,
        receipt_schema: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        secure_calls.append(Path(artifact_path))
        return real_secure_load(
            artifact_path,
            receipt_path,
            receipt_schema=receipt_schema,
        )

    monkeypatch.setattr(
        prospective_v2,
        "load_verified_json_mapping_with_receipt",
        tracked_secure_load,
    )
    prospective_v2.build_precalibration_seal_v2(
        protocol=protocol,
        controller=controller,
        source_bindings=source_bindings,
        runtime_bindings=runtime_bindings,
        opaque_target_identities=identities,
        chronology=chronology,
    )

    assert secure_calls == [
        prospective_v2.default_protocol_v2_path(),
        prospective_v2.default_controller_v2_path(),
    ]


@pytest.mark.parametrize(
    "module_name",
    [
        "development.py",
        "gate.py",
        "metadata_manifest.py",
        "precalibration_seal.py",
        "protocol.py",
        "target_amendment.py",
        "target_contract.py",
    ],
)
def test_v2_code_identity_changes_when_transitive_runtime_module_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    from experiments.kbound.so2sat import prospective_v2

    baseline = prospective_v2.prospective_v2_code_identity()
    real_module_file = prospective_v2._module_file
    mutated_module = tmp_path / module_name
    original_module = real_module_file(module_name)
    mutated_module.write_bytes(original_module.read_bytes() + b"\n# identity mutation probe\n")

    def redirected_module_file(name: str) -> Path:
        return mutated_module if name == module_name else real_module_file(name)

    monkeypatch.setattr(prospective_v2, "_module_file", redirected_module_file)
    mutated = prospective_v2.prospective_v2_code_identity()

    assert module_name in baseline["files_sha256"]
    assert mutated["files_sha256"][module_name] != baseline["files_sha256"][module_name]
    assert mutated["code_identity_sha256"] != baseline["code_identity_sha256"]


def test_v2_protocol_and_controller_are_receipted_and_fully_fixed() -> None:
    from experiments.kbound.so2sat.prospective_v2 import (
        CONTROLLER_ID,
        PROTOCOL_ID,
        load_controller_v2,
        load_protocol_v2,
        verify_controller_v2_receipt,
        verify_protocol_v2_receipt,
    )

    protocol = load_protocol_v2()
    controller = load_controller_v2()
    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol["status"] == "SOURCE_ONLY_LOCK_NOT_PRECALIBRATION_SEALED"
    assert protocol["action_unit"] == "city_checkpoint"
    assert protocol["controller_id"] == CONTROLLER_ID
    assert protocol["target_inference"]["cluster_unit"] == "target_city"
    assert protocol["target_inference"]["confidence_intervals"] == ("nominal_95_percent_city_cluster_bootstrap")
    assert protocol["target_inference"]["checkpoint_only_baseline_role"] == ("required_secondary_diagnostic")
    assert protocol["target_inference"]["bootstrap"] == {
        "resamples": 10000,
        "seed": 20260902,
        "bit_generator": "PCG64",
        "interval": "percentile",
        "quantiles": [0.025, 0.975],
        "numpy_quantile_method": "linear",
    }
    assert protocol["gate_authorization"]["minimum_direct_adapt_cities"] == 7
    assert protocol["gate_authorization"]["minimum_direct_freeze_cities"] == 7
    assert protocol["live_execution"] == {
        "authority_order": [
            "recompute_precalibration_seal",
            "recompute_complete_95_cell_gate_screen",
            "recompute_target_authorization",
            "construct_target_paths_and_loader",
        ],
        "target_path_or_loader_before_authorization": "FORBIDDEN",
        "probe_pixel_access": "validation_sen2_only_after_authorization",
        "probe_label_access": "FORBIDDEN",
        "action_inventory": "exactly_10_target_cities_by_5_checkpoints",
        "action_receipts": "all_50_create_only_and_replayed_twice_before_evaluation",
        "evaluation_pixel_access": "testing_sen2_only_after_complete_action_plan",
        "target_outcome_access": "FORBIDDEN_TO_LIVE_RUNNER",
        "abstain_realized_action": "FREEZE",
    }
    assert controller["controller_id"] == CONTROLLER_ID
    assert controller["v1_negative_record"] == {
        "status": "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL",
        "selected_candidate_id": None,
        "selection_sha256": "8d50ab20ae6e88c96e36f4fb79b1432dde449e840118b9a2aade245ff9fd56d4",
        "artifact_receipt": {
            "schema": "kbound_so2sat_artifact_receipt_v2",
            "artifact_basename": "so2sat_candidate_selection.json",
            "artifact_bytes": 9801,
            "artifact_sha256": "8db11a797d98c5f104736a5ed982a422982f9f75fc8a7d1c6e13f07a826c0b79",
            "canonical_document_sha256": "eaa1a03b2a06f38ef23f0ff70abc1f196d3ac316b9d560dfa1180ab3946e2095",
        },
        "gate_calibration_rows_read": 0,
        "target_pixels_read": 0,
        "target_labels_read": 0,
    }
    assert controller["gate_fit_replay"]["loco"] == {
        "held_out_city_count": 1,
        "gain_over_best_fixed": pytest.approx(0.003718664350695591, abs=1e-15),
        "sign_accuracy": pytest.approx(0.6222222222222222, abs=1e-15),
        "raw_action_counts": {"ADAPT": 25, "FREEZE": 20},
    }
    assert controller["gate_fit_replay"]["leave_two_cities_out_gain_over_best_fixed"] == pytest.approx(
        0.0028688724217092406,
        abs=1e-15,
    )
    assert controller["gate_fit_replay"]["leave_three_cities_out_gain_over_best_fixed"] == pytest.approx(
        0.0014772760400262922,
        abs=1e-15,
    )
    assert "exploratory_confidence_intervals_cross_zero" not in repr(controller)
    assert "exploratory_p_values_nonsignificant" not in repr(controller)
    assert controller["ridge"]["penalty"] == 1.0
    assert controller["ridge"]["intercept_unpenalized"] is True
    assert controller["checkpoint_encoding"] == "I(checkpoint=s)-1/5_for_s_0_through_4"
    assert len(controller["checkpoint_coefficients"]) == 5
    assert len(controller["feature_coefficients"]) == 5
    assert verify_protocol_v2_receipt()["artifact_basename"] == ("prospective_protocol_v2.json")
    assert verify_controller_v2_receipt()["artifact_basename"] == ("tent_citymean5_checkpoint_ridge_v2.json")


def test_controller_replays_exactly_from_the_receipted_gate_fit_bundle() -> None:
    from experiments.kbound.so2sat.integrity import verify_artifact_receipt
    from experiments.kbound.so2sat.prospective_v2 import (
        derive_controller_v2,
        load_controller_v2,
    )

    bundle = strict_json_load(_V1_TENT)
    receipt = verify_artifact_receipt(_V1_TENT)
    derived = derive_controller_v2(bundle, receipt)

    assert derived == load_controller_v2()
    assert derived["intercept"] == pytest.approx(0.003170137031133156, abs=1e-15)
    assert derived["checkpoint_coefficients"] == pytest.approx(
        [
            0.012260325145721963,
            -0.007675710409203962,
            0.010508001440846964,
            -0.010018030859157426,
            -0.005074585318207513,
        ],
        abs=1e-15,
    )
    assert derived["feature_coefficients"] == pytest.approx(
        [
            -0.00999556450671667,
            0.010250128394567846,
            0.01502138472087773,
            -0.005656147948235403,
            -0.006536191702034257,
        ],
        abs=1e-15,
    )


def test_city_features_are_equal_weight_means_over_all_five_checkpoints() -> None:
    from experiments.kbound.so2sat.prospective_v2 import (
        load_controller_v2,
        raw_prediction_v2,
    )

    controller = load_controller_v2()
    base = _feature_values(controller)
    city_features = {str(seed): dict(base) for seed in range(5)}
    reference = raw_prediction_v2(
        controller,
        checkpoint_id="0",
        city_probe_features=city_features,
    )
    feature = controller["feature_names"][0]
    city_features["4"][feature] += controller["standardization"]["scales"][0]
    changed = raw_prediction_v2(
        controller,
        checkpoint_id="0",
        city_probe_features=city_features,
    )
    assert changed - reference == pytest.approx(
        controller["feature_coefficients"][0] / 5.0,
        abs=1e-15,
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "not-a-number"])
def test_missing_or_nonfinite_probe_features_abstain_and_retain_frozen_model(
    bad: object,
) -> None:
    from experiments.kbound.so2sat.prospective_v2 import (
        load_controller_v2,
        route_city_checkpoint_v2,
    )

    controller = load_controller_v2()
    features = {str(seed): _feature_values(controller) for seed in range(5)}
    features["2"] = dict(features["2"])
    features["2"][controller["feature_names"][0]] = bad

    action = route_city_checkpoint_v2(
        controller,
        checkpoint_id="2",
        city_probe_features=features,
        interval_radius=0.01,
    )
    assert action["support_status"] == "INSUFFICIENT_OR_NONFINITE_INPUT"
    assert action["decision"] == "ABSTAIN"
    assert action["realized_action"] == "FREEZE"
    assert action["delta_hat"] is None
    assert action["lower"] is None
    assert action["upper"] is None


def test_supported_interval_rule_uses_strict_bounds_and_abstain_is_frozen() -> None:
    from experiments.kbound.so2sat.prospective_v2 import (
        load_controller_v2,
        raw_prediction_v2,
        route_city_checkpoint_v2,
    )

    controller = load_controller_v2()
    features = {str(seed): _feature_values(controller) for seed in range(5)}
    prediction = raw_prediction_v2(
        controller,
        checkpoint_id="0",
        city_probe_features=features,
    )
    direct = route_city_checkpoint_v2(
        controller,
        checkpoint_id="0",
        city_probe_features=features,
        interval_radius=0.0,
    )
    boundary = route_city_checkpoint_v2(
        controller,
        checkpoint_id="0",
        city_probe_features=features,
        interval_radius=prediction,
    )
    assert direct["decision"] == "ADAPT"
    assert direct["realized_action"] == "ADAPT"
    assert boundary["lower"] == pytest.approx(0.0, abs=1e-15)
    assert boundary["decision"] == "ABSTAIN"
    assert boundary["realized_action"] == "FREEZE"


def test_invalid_checkpoint_identity_is_an_error_not_a_resigned_abstention() -> None:
    from experiments.kbound.so2sat.prospective_v2 import (
        load_controller_v2,
        route_city_checkpoint_v2,
    )

    controller = load_controller_v2()
    features = {str(seed): _feature_values(controller) for seed in range(5)}
    with pytest.raises(IntegrityError, match="checkpoint_id"):
        route_city_checkpoint_v2(
            controller,
            checkpoint_id="5",
            city_probe_features=features,
            interval_radius=0.0,
        )


def test_precalibration_template_is_explicitly_blocked_and_not_a_seal() -> None:
    from experiments.kbound.so2sat.prospective_v2 import (
        load_precalibration_template_v2,
    )

    template = load_precalibration_template_v2()
    assert template["status"] == "TEMPLATE_BLOCKED_NOT_AUTHORIZED"
    assert template["missing_required_bindings"] == [
        "opaque_validation_container_raw_byte_identity",
        "opaque_testing_container_raw_byte_identity",
        "source_artifact_receipts",
        "runtime_code_configuration_environment_hashes",
        "zero_access_chronology_declaration",
    ]
    assert template["gate_calibration_authorized"] is False
    assert template["target_access_authorized"] is False


def test_precalibration_seal_is_pure_portable_and_binds_every_required_identity() -> None:
    from experiments.kbound.so2sat.prospective_v2 import (
        validate_precalibration_seal_v2,
    )

    seal, controller = _precalibration_seal()
    validate_precalibration_seal_v2(seal)
    assert seal["status"] == "SEALED_BEFORE_GATE_CALIBRATION_AND_TARGET_ACCESS"
    assert seal["controller_sha256"] == controller["controller_sha256"]
    assert seal["action_unit"] == "city_checkpoint"
    assert seal["gate_calibration_authorized"] is True
    assert seal["target_access_authorized"] is False
    assert seal["access_audit"] == {
        "gate_calibration_rows_read": 0,
        "gate_calibration_pixels_read": 0,
        "gate_calibration_labels_read": 0,
        "target_pixels_read": 0,
        "target_labels_read": 0,
        "target_inputs": [],
    }
    rendered = repr(seal)
    assert "/Users/" not in rendered
    assert "/Volumes/" not in rendered
    assert "/private/" not in rendered


def test_precalibration_seal_rejects_duplicate_checkpoints_paths_and_opened_hdf5() -> None:
    from experiments.kbound.so2sat.prospective_v2 import (
        build_precalibration_seal_v2,
        load_controller_v2,
        load_protocol_v2,
    )

    protocol = load_protocol_v2()
    controller = load_controller_v2()
    bindings = _source_bindings(controller)
    bindings["source_checkpoints"]["4"] = copy.deepcopy(bindings["source_checkpoints"]["3"])
    with pytest.raises(IntegrityError, match="five independent checkpoints"):
        build_precalibration_seal_v2(
            protocol=protocol,
            controller=controller,
            source_bindings=bindings,
            runtime_bindings=_runtime_bindings(),
            opaque_target_identities=_opaque_target_identities(),
            chronology=_chronology(),
        )

    targets = _opaque_target_identities()
    targets["validation"]["artifact_basename"] = "/unsafe/validation.h5"
    with pytest.raises(IntegrityError, match="portable basename"):
        build_precalibration_seal_v2(
            protocol=protocol,
            controller=controller,
            source_bindings=_source_bindings(controller),
            runtime_bindings=_runtime_bindings(),
            opaque_target_identities=targets,
            chronology=_chronology(),
        )


def test_precalibration_seal_rejects_unique_but_gate_fit_unbound_checkpoint() -> None:
    from experiments.kbound.so2sat.prospective_v2 import (
        build_precalibration_seal_v2,
        load_controller_v2,
        load_protocol_v2,
    )

    protocol = load_protocol_v2()
    controller = load_controller_v2()
    bindings = _source_bindings(controller)
    bindings["source_checkpoints"]["0"]["checkpoint_file_sha256"] = "9" * 64
    with pytest.raises(IntegrityError, match="gate-fit checkpoint identities"):
        build_precalibration_seal_v2(
            protocol=protocol,
            controller=controller,
            source_bindings=bindings,
            runtime_bindings=_runtime_bindings(),
            opaque_target_identities=_opaque_target_identities(),
            chronology=_chronology(),
        )


def test_precalibration_seal_rejects_stale_code_or_configuration_identity() -> None:
    from experiments.kbound.so2sat.prospective_v2 import (
        build_precalibration_seal_v2,
        load_controller_v2,
        load_protocol_v2,
    )

    protocol = load_protocol_v2()
    controller = load_controller_v2()
    runtime = _runtime_bindings()
    runtime["code_identity_sha256"] = "9" * 64
    with pytest.raises(IntegrityError, match="code identity"):
        build_precalibration_seal_v2(
            protocol=protocol,
            controller=controller,
            source_bindings=_source_bindings(controller),
            runtime_bindings=runtime,
            opaque_target_identities=_opaque_target_identities(),
            chronology=_chronology(),
        )

    runtime = _runtime_bindings()
    runtime["configuration_identity_sha256"] = "9" * 64
    with pytest.raises(IntegrityError, match="configuration identity"):
        build_precalibration_seal_v2(
            protocol=protocol,
            controller=controller,
            source_bindings=_source_bindings(controller),
            runtime_bindings=runtime,
            opaque_target_identities=_opaque_target_identities(),
            chronology=_chronology(),
        )

    targets = _opaque_target_identities()
    targets["testing"]["hdf5_datasets_opened"] = 1
    with pytest.raises(IntegrityError, match="without HDF5 deserialization"):
        build_precalibration_seal_v2(
            protocol=protocol,
            controller=controller,
            source_bindings=_source_bindings(controller),
            runtime_bindings=_runtime_bindings(),
            opaque_target_identities=targets,
            chronology=_chronology(),
        )


def test_gate_screen_is_city_clustered_and_requires_seven_cities_per_direct_action() -> None:
    from experiments.kbound.so2sat.prospective_v2 import build_gate_screen_v2

    seal, controller = _precalibration_seal()
    bundle = _calibration_bundle(seal, controller)
    screen = build_gate_screen_v2(
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=bundle,
    )
    assert screen["cell_count"] == 95
    assert screen["calibration"]["cluster_unit"] == "gate_calibration_city"
    assert screen["calibration"]["city_residual_aggregation"] == ("maximum_absolute_residual_over_five_checkpoints")
    assert screen["calibration"]["order_statistic_rank_one_based"] == 18
    assert screen["minimum_direct_action_cities"] == 7
    assert len(screen["direct_action_cities"]["ADAPT"]) == 19
    assert len(screen["direct_action_cities"]["FREEZE"]) == 19
    assert screen["checks"]["at_least_7_direct_adapt_cities"] is True
    assert screen["checks"]["at_least_7_direct_freeze_cities"] is True
    assert screen["passed"] is True
    assert screen["checkpoint_only_baseline"]["role"] == ("required_secondary_diagnostic")


def test_gate_calibration_rejects_naked_unhashed_feature_values() -> None:
    from experiments.kbound.so2sat.prospective_v2 import build_gate_screen_v2

    seal, controller = _precalibration_seal()
    bundle = _calibration_bundle(seal, controller)
    for cell in bundle["cells"]:
        feature_document = cell.pop("feature_document")
        cell["feature_values"] = {name: feature_document["features"][name] for name in controller["feature_names"]}
    bundle["bundle_sha256"] = stable_sha256({key: value for key, value in bundle.items() if key != "bundle_sha256"})
    with pytest.raises(IntegrityError, match="feature document"):
        build_gate_screen_v2(
            precalibration_seal=seal,
            controller=controller,
            calibration_bundle=bundle,
        )


def test_all_abstain_gate_screen_fails_and_resigned_pass_flag_cannot_authorize() -> None:
    from experiments.kbound.so2sat.prospective_v2 import (
        authorize_target_execution_v2,
        build_gate_screen_v2,
    )

    seal, controller = _precalibration_seal()
    bundle = _calibration_bundle(seal, controller, large_residual=True)
    failed = build_gate_screen_v2(
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=bundle,
    )
    assert failed["decision_counts"] == {
        "ADAPT": 0,
        "FREEZE": 0,
        "ABSTAIN": 95,
    }
    assert failed["passed"] is False
    with pytest.raises(IntegrityError, match="failed.*seven-city"):
        authorize_target_execution_v2(
            precalibration_seal=seal,
            controller=controller,
            calibration_bundle=bundle,
            submitted_gate_screen=failed,
        )

    forged = copy.deepcopy(failed)
    forged["passed"] = True
    forged["status"] = "PASSED_GATE_AUTHORIZATION_SCREEN"
    forged["checks"] = dict.fromkeys(forged["checks"], True)
    forged["gate_screen_sha256"] = stable_sha256(
        {key: value for key, value in forged.items() if key != "gate_screen_sha256"}
    )
    with pytest.raises(IntegrityError, match="canonical recomputation"):
        authorize_target_execution_v2(
            precalibration_seal=seal,
            controller=controller,
            calibration_bundle=bundle,
            submitted_gate_screen=forged,
        )


def test_passing_gate_authorization_still_discloses_zero_target_access() -> None:
    from experiments.kbound.so2sat.prospective_v2 import (
        authorize_target_execution_v2,
        build_gate_screen_v2,
    )

    seal, controller = _precalibration_seal()
    bundle = _calibration_bundle(seal, controller)
    screen = build_gate_screen_v2(
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=bundle,
    )
    authorization = authorize_target_execution_v2(
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=bundle,
        submitted_gate_screen=screen,
    )
    assert authorization["status"] == "AUTHORIZED_AFTER_SEVEN_CITY_BIDIRECTIONAL_SCREEN"
    assert authorization["action_unit"] == "city_checkpoint"
    assert authorization["target_probe_pixel_access_authorized"] is True
    assert authorization["target_outcome_access_authorized"] is False
    assert authorization["target_pixels_read_before_authorization"] == 0
    assert authorization["target_labels_read_before_authorization"] == 0


def test_target_authorization_schema_rejects_resigned_semantic_drift() -> None:
    from experiments.kbound.so2sat.prospective_v2 import (
        authorize_target_execution_v2,
        build_gate_screen_v2,
        validate_target_authorization_v2,
    )

    seal, controller = _precalibration_seal()
    bundle = _calibration_bundle(seal, controller)
    screen = build_gate_screen_v2(
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=bundle,
    )
    authorization = authorize_target_execution_v2(
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=bundle,
        submitted_gate_screen=screen,
    )
    validate_target_authorization_v2(
        authorization,
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=bundle,
        gate_screen=screen,
    )

    forged = copy.deepcopy(authorization)
    forged["target_outcome_access_authorized"] = True
    forged["authorization_sha256"] = stable_sha256(
        {key: value for key, value in forged.items() if key != "authorization_sha256"}
    )
    with pytest.raises(IntegrityError, match="outcome access"):
        validate_target_authorization_v2(
            forged,
            precalibration_seal=seal,
            controller=controller,
            calibration_bundle=bundle,
            gate_screen=screen,
        )


def _scored_target_bundle(*, malformed_abstain: bool = False) -> dict[str, Any]:
    from experiments.kbound.so2sat.prospective_v2 import load_controller_v2

    cells: list[dict[str, Any]] = []
    for city_index in range(10):
        city_id = f"opaque-target-city-{city_index:02d}"
        benefit = 0.1 if city_index < 5 else -0.1
        decision = "ADAPT" if city_index < 5 else "FREEZE"
        for seed in range(5):
            cells.append(
                {
                    "city_id": city_id,
                    "checkpoint_id": str(seed),
                    "decision": decision,
                    "realized_action": "ADAPT" if decision == "ADAPT" else "FREEZE",
                    "observed_benefit": benefit,
                    "action_sha256": f"{city_index * 5 + seed + 3000:064x}",
                }
            )
    if malformed_abstain:
        cells[0]["decision"] = "ABSTAIN"
        cells[0]["realized_action"] = "ADAPT"
    document: dict[str, Any] = {
        "schema": "kbound_so2sat_scored_target_cells_v2",
        "status": "COMPLETE_10_CITY_50_CELL_SINGLE_REVEAL",
        "protocol_id": "KBOUND_SO2SAT_LCZ42_PROSPECTIVE_CONFIRMATION_v2",
        "controller_sha256": load_controller_v2()["controller_sha256"],
        "authorization_sha256": "8" * 64,
        "action_unit": "city_checkpoint",
        "cells": cells,
        "target_city_count": 10,
        "checkpoint_count": 5,
        "outcome_reveal_count": 1,
    }
    document["bundle_sha256"] = stable_sha256(document)
    return document


def _target_inference_inputs(*, malformed_abstain: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    from experiments.kbound.so2sat.prospective_v2 import (
        authorize_target_execution_v2,
        build_gate_screen_v2,
    )

    seal, controller = _precalibration_seal()
    calibration = _calibration_bundle(seal, controller)
    screen = build_gate_screen_v2(
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=calibration,
    )
    authorization = authorize_target_execution_v2(
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=calibration,
        submitted_gate_screen=screen,
    )
    bundle = _scored_target_bundle(malformed_abstain=malformed_abstain)
    bundle["authorization_sha256"] = authorization["authorization_sha256"]
    bundle["bundle_sha256"] = stable_sha256({key: value for key, value in bundle.items() if key != "bundle_sha256"})
    return bundle, authorization


def test_target_inference_uses_ten_city_clusters_and_fixed_randomness() -> None:
    from experiments.kbound.so2sat.prospective_v2 import summarize_target_inference_v2

    bundle, authorization = _target_inference_inputs()
    first = summarize_target_inference_v2(bundle, authorization=authorization)
    second = summarize_target_inference_v2(bundle, authorization=authorization)
    assert first == second
    assert first["cluster_unit"] == "target_city"
    assert first["cluster_count"] == 10
    assert first["checkpoints_per_cluster"] == 5
    assert first["exact_sign_flip_pattern_count"] == 1024
    assert first["comparisons"]["kbound_v2_vs_always_adapt"]["mean_city_cluster_difference"] == pytest.approx(0.05)
    assert first["comparisons"]["kbound_v2_vs_always_freeze"]["mean_city_cluster_difference"] == pytest.approx(0.05)
    assert first["checkpoint_only_baseline"]["role"] == "required_secondary_diagnostic"
    assert first["bootstrap"] == {
        "resamples": 10000,
        "seed": 20260902,
        "bit_generator": "PCG64",
        "interval": "percentile",
        "quantiles": [0.025, 0.975],
        "numpy_quantile_method": "linear",
    }


def test_target_inference_rejects_abstain_that_does_not_retain_frozen_model() -> None:
    from experiments.kbound.so2sat.prospective_v2 import summarize_target_inference_v2

    bundle, authorization = _target_inference_inputs(malformed_abstain=True)
    with pytest.raises(IntegrityError, match="ABSTAIN.*FREEZE"):
        summarize_target_inference_v2(bundle, authorization=authorization)


def test_target_inference_requires_the_bound_semantic_authorization() -> None:
    from experiments.kbound.so2sat.prospective_v2 import (
        authorize_target_execution_v2,
        build_gate_screen_v2,
        summarize_target_inference_v2,
    )

    seal, controller = _precalibration_seal()
    calibration = _calibration_bundle(seal, controller)
    screen = build_gate_screen_v2(
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=calibration,
    )
    authorization = authorize_target_execution_v2(
        precalibration_seal=seal,
        controller=controller,
        calibration_bundle=calibration,
        submitted_gate_screen=screen,
    )
    bundle = _scored_target_bundle()
    bundle["authorization_sha256"] = authorization["authorization_sha256"]
    bundle["bundle_sha256"] = stable_sha256({key: value for key, value in bundle.items() if key != "bundle_sha256"})
    forged = copy.deepcopy(authorization)
    forged["target_outcome_access_authorized"] = True
    forged["authorization_sha256"] = stable_sha256(
        {key: value for key, value in forged.items() if key != "authorization_sha256"}
    )
    with pytest.raises(IntegrityError, match="outcome access"):
        summarize_target_inference_v2(bundle, authorization=forged)
