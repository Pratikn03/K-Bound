"""Source-only prospective-v2 lock and authorization logic for So2Sat.

This module deliberately has no dataset loader and accepts no target path.  It
freezes the controller chosen from the already-open nine-city gate-fit panel,
builds a portable pre-calibration seal from externally supplied identities,
and recomputes the later 19-city authorization screen.  The existing v1
negative result remains a separate immutable scientific record.
"""

from __future__ import annotations

import copy
import itertools
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .development import GATE_FIT_ROLE, TENT_CANDIDATE_ID, validate_candidate_bundle
from .features import validate_feature_document
from .integrity import (
    ARTIFACT_RECEIPT_SCHEMA_V2,
    IntegrityError,
    file_sha256,
    load_verified_json_mapping_with_receipt,
    require_sha256,
    stable_sha256,
    strict_json_load,
)

PROTOCOL_SCHEMA = "kbound_so2sat_prospective_protocol_v2"
PROTOCOL_ID = "KBOUND_SO2SAT_LCZ42_PROSPECTIVE_CONFIRMATION_v2"
PROTOCOL_BASENAME = "prospective_protocol_v2.json"
CONTROLLER_SCHEMA = "kbound_so2sat_locked_controller_v2"
CONTROLLER_ID = "tent_citymean5_checkpoint_ridge_v2"
CONTROLLER_BASENAME = f"{CONTROLLER_ID}.json"
PRECALIBRATION_TEMPLATE_SCHEMA = "kbound_so2sat_precalibration_seal_template_v2"
PRECALIBRATION_TEMPLATE_BASENAME = "precalibration_seal_v2.template.json"
PRECALIBRATION_SEAL_SCHEMA = "kbound_so2sat_precalibration_execution_seal_v2"
CALIBRATION_BUNDLE_SCHEMA = "kbound_so2sat_gate_calibration_bundle_v2"
GATE_SCREEN_SCHEMA = "kbound_so2sat_gate_authorization_screen_v2"
TARGET_AUTHORIZATION_SCHEMA = "kbound_so2sat_target_authorization_v2"
ACTION_SCHEMA = "kbound_so2sat_label_free_action_v2"
TARGET_INFERENCE_SCHEMA = "kbound_so2sat_city_cluster_inference_v2"
SCORED_TARGET_BUNDLE_SCHEMA = "kbound_so2sat_scored_target_cells_v2"

CHECKPOINT_IDS = tuple(str(seed) for seed in range(5))
FEATURE_NAMES = (
    "entropy_change",
    "prediction_disagreement",
    "marginal_jensen_shannon_divergence",
    "normalized_adapter_update_norm",
    "batchnorm_source_statistic_divergence",
)
GATE_FIT_CITY_COUNT = 9
GATE_CALIBRATION_CITY_COUNT = 19
GATE_CALIBRATION_CELL_COUNT = 95
RIDGE_PENALTY = 1.0
CALIBRATION_ALPHA = 0.10
CONFORMAL_RANK = 18
MINIMUM_DIRECT_ACTION_CITIES = 7
TARGET_CITY_COUNT = 10
TARGET_CELL_COUNT = 50
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_902

# Reviewed conservative static-import closure of the prospective-v2 decision,
# live-routing, and inference roots.  The closure deliberately includes lazy
# relative imports so a module that can become live cannot change outside the
# identity sealed before gate calibration.
PROSPECTIVE_V2_RUNTIME_MODULES = (
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

_INTERCEPT = 0.003170137031133156
_CHECKPOINT_COEFFICIENTS = (
    0.012260325145721963,
    -0.007675710409203962,
    0.010508001440846964,
    -0.010018030859157426,
    -0.005074585318207513,
)
_FEATURE_COEFFICIENTS = (
    -0.00999556450671667,
    0.010250128394567846,
    0.01502138472087773,
    -0.005656147948235403,
    -0.006536191702034257,
)
_FEATURE_MEANS = (
    0.053228304777694885,
    0.09074234060282148,
    0.008307472555549696,
    0.031761693019734825,
    0.26852065128661207,
)
_FEATURE_SCALES = (
    0.030041202272761924,
    0.05606134283922513,
    0.008003693017937215,
    0.019087604482260104,
    0.20790065121454157,
)

_GATE_FIT_RECEIPT = {
    "schema": ARTIFACT_RECEIPT_SCHEMA_V2,
    "artifact_basename": "so2sat_tent_adam_bn_affine_probe_transfer_v1.gate_fit.json",
    "artifact_bytes": 271074,
    "artifact_sha256": "c91e6989f610408f0be223062620404275a1d040bcef5228743bef2366e6df2e",
    "canonical_document_sha256": "6572b32974924087dc8fb755b1e66835d4fbcbbfe992c2fa7e6895d97a7d1632",
}
_V1_NEGATIVE_SELECTION_RECEIPT = {
    "schema": ARTIFACT_RECEIPT_SCHEMA_V2,
    "artifact_basename": "so2sat_candidate_selection.json",
    "artifact_bytes": 9801,
    "artifact_sha256": "8db11a797d98c5f104736a5ed982a422982f9f75fc8a7d1c6e13f07a826c0b79",
    "canonical_document_sha256": "eaa1a03b2a06f38ef23f0ff70abc1f196d3ac316b9d560dfa1180ab3946e2095",
}
_V1_NEGATIVE_SELECTION_SHA256 = "8d50ab20ae6e88c96e36f4fb79b1432dde449e840118b9a2aade245ff9fd56d4"
_GATE_FIT_BUNDLE_SHA256 = "1c004d992936b8ab44cb5183b84157628546515979078ba389edf45e9361e069"
_GATE_ROWS_SHA256 = "ea765c34dc19335e473631e17d0d57d03969e022ebf0afa95a7a89ebab82c308"
_CANDIDATE_CONFIG_SHA256 = "6dd6513ef50848a4303ca142c0b8da4752d5c5c8d1fb0f76cda2b666765287a1"
_CHECKPOINT_COLLECTION_SHA256 = "351ad26a4cbc751365f42ecf43ced50773c301d68f66a3e0c39ca6b9ffc9c50c"
_NORMALIZER_SHA256 = "28f8d15ce5465c451cc2d741417af599851ce2630fae2ec5dba2d0680ff112d6"
_SOURCE_CONTAINER_IDENTITY_SHA256 = "339ee12eac8584b479f9765d112ab18faa3ba892781c78306c833db9b8028188"
_SOURCE_CHECKPOINTS = {
    "0": {
        "checkpoint_file_sha256": "ee305a9410da1c14f163f659db10c4c19f4851aadcae4e0d81676187bf73596f",
        "checkpoint_tensor_sha256": "4ed9fafe99d3e3e9fd7b0d609dedcc4945b5eaca30756adbc70fd02e3b38cd55",
    },
    "1": {
        "checkpoint_file_sha256": "ec9853874a0b90cb1afebede452a2a73791818f18eabd27e71032252c137f35b",
        "checkpoint_tensor_sha256": "7967404e2d983099337fcf928117955c9988f4ce4b094d91b1b094f5c049ecd6",
    },
    "2": {
        "checkpoint_file_sha256": "0b9033c7050b96c341eb2228ed214d490186546e5d49f0063856a4c2953dc814",
        "checkpoint_tensor_sha256": "6e6f8d4336db557c2754881adcaf55653f81d6722ae0a4a719dbde1756d13e4e",
    },
    "3": {
        "checkpoint_file_sha256": "dc5dbbab42fbeeadc52e5a6c5d6c68262b73393f87e786f884918ef804a69399",
        "checkpoint_tensor_sha256": "3e8a39ee2fed86e8b12214d3cc33ad411218eb188f681bf02e07e3c4c2e2b1e6",
    },
    "4": {
        "checkpoint_file_sha256": "a165d866c8698efbba9fa356344b2abf2ae4cffd7271a281262b89f6f27ebc17",
        "checkpoint_tensor_sha256": "956ffe9f2d86b6a1a45aa5dda908d5adcacc4d864206405ddd00e2b9ea5daee3",
    },
}
_LOCO_GAIN = 0.003718664350695591
_LOCO_SIGN_ACCURACY = 0.6222222222222222
_LEAVE_TWO_GAIN = 0.0028688724217092406
_LEAVE_THREE_GAIN = 0.0014772760400262922


def _module_file(name: str) -> Path:
    return Path(__file__).resolve().with_name(name)


def default_protocol_v2_path() -> Path:
    return _module_file(PROTOCOL_BASENAME)


def default_controller_v2_path() -> Path:
    return _module_file(CONTROLLER_BASENAME)


def default_precalibration_template_v2_path() -> Path:
    return _module_file(PRECALIBRATION_TEMPLATE_BASENAME)


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise IntegrityError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise IntegrityError(f"{field} must be a finite number")
    return result


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise IntegrityError(f"{field} must be a positive integer")
    return value


def _portable_basename(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or Path(value).name != value
    ):
        raise IntegrityError(f"{field} must be one portable basename")
    return value


def _reject_private_paths(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, member in value.items():
            _reject_private_paths(key)
            _reject_private_paths(member)
    elif isinstance(value, (list, tuple)):
        for member in value:
            _reject_private_paths(member)
    elif isinstance(value, str) and any(marker in value for marker in ("/Users/", "/Volumes/", "/private/", "file://")):
        raise IntegrityError("portable v2 artifacts cannot contain private absolute paths")


def _validate_artifact_binding(value: Any, *, field: str) -> dict[str, Any]:
    expected = {
        "schema",
        "artifact_basename",
        "artifact_bytes",
        "artifact_sha256",
        "canonical_document_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise IntegrityError(f"{field} must be one exact portable artifact receipt")
    if value.get("schema") != ARTIFACT_RECEIPT_SCHEMA_V2:
        raise IntegrityError(f"{field} must use the portable v2 receipt schema")
    _portable_basename(value.get("artifact_basename"), field=f"{field}.artifact_basename")
    _positive_integer(value.get("artifact_bytes"), field=f"{field}.artifact_bytes")
    require_sha256(value.get("artifact_sha256"), field=f"{field}.artifact_sha256")
    require_sha256(
        value.get("canonical_document_sha256"),
        field=f"{field}.canonical_document_sha256",
    )
    return dict(value)


def _expected_protocol_v2() -> dict[str, Any]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "SOURCE_ONLY_LOCK_NOT_PRECALIBRATION_SEALED",
        "version_relationship": {
            "v1_result": "PRESERVED_NEGATIVE_NO_FEASIBLE_CANDIDATE",
            "v2_is_not_an_amendment_to_v1": True,
            "v2_development_basis": "already_open_v1_nine_city_gate_fit_evidence_only",
            "gate_calibration_accessed_for_v2_design": False,
            "target_accessed_for_v2_design": False,
        },
        "dataset": {
            "name": "So2Sat LCZ42",
            "release": "v4.2",
            "split_scenario": "culture-10",
            "class_count": 17,
            "modality": "sen2_10_band",
        },
        "action_unit": "city_checkpoint",
        "checkpoint_ids": list(CHECKPOINT_IDS),
        "controller_id": CONTROLLER_ID,
        "controller_contract": {
            "candidate": TENT_CANDIDATE_ID,
            "city_feature_aggregation": "equal_mean_over_exactly_five_checkpoint_probe_documents",
            "features": list(FEATURE_NAMES),
            "checkpoint_encoding": "I(checkpoint=s)-1/5_for_s_0_through_4",
            "standardization": "nine_gate_fit_city_vectors_population_sd_ddof_0",
            "ridge_penalty": RIDGE_PENALTY,
            "ridge_penalized_terms": "all_nonintercept_coefficients",
            "intercept_unpenalized": True,
            "raw_routing_rule": "positive_ADAPT_negative_FREEZE_zero_ABSTAIN",
        },
        "development": {
            "gate_fit_city_count": GATE_FIT_CITY_COUNT,
            "gate_fit_checkpoint_cell_count": 45,
            "gate_fit_role": "controller_design_and_fit_only",
            "gate_calibration_city_count": GATE_CALIBRATION_CITY_COUNT,
            "gate_calibration_checkpoint_cell_count": GATE_CALIBRATION_CELL_COUNT,
            "gate_calibration_role": "split_conformal_radius_and_predeclared_exposure_screen_only",
            "no_postseal_selection": ("no_model_feature_penalty_threshold_adapter_or_baseline_selection_after_v2_seal"),
        },
        "calibration": {
            "method": "split_conformal_city_max_checkpoint_absolute_residual",
            "independent_cluster_unit": "gate_calibration_city",
            "within_city_residual": "maximum_absolute_residual_over_five_checkpoints",
            "alpha": CALIBRATION_ALPHA,
            "order_statistic_rank_one_based": CONFORMAL_RANK,
        },
        "decision_rule": {
            "adapt": "lower_bound_strictly_greater_than_zero",
            "freeze": "upper_bound_strictly_less_than_zero",
            "otherwise": "ABSTAIN",
            "abstain_realized_action": "FREEZE",
            "missing_or_nonfinite_feature_action": "ABSTAIN",
            "missing_or_nonfinite_feature_realized_action": "FREEZE",
        },
        "gate_authorization": {
            "minimum_direct_adapt_cities": MINIMUM_DIRECT_ACTION_CITIES,
            "minimum_direct_freeze_cities": MINIMUM_DIRECT_ACTION_CITIES,
            "direct_means": "interval_certified_ADAPT_or_FREEZE_not_ABSTAIN",
            "complete_grid_required": True,
            "checkpoint_only_baseline": "required_and_reported_without_selection",
            "failure": "NO_TARGET_ACCESS",
        },
        "live_execution": {
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
            "action_receipts": ("all_50_create_only_and_replayed_twice_before_evaluation"),
            "evaluation_pixel_access": ("testing_sen2_only_after_complete_action_plan"),
            "target_outcome_access": "FORBIDDEN_TO_LIVE_RUNNER",
            "abstain_realized_action": "FREEZE",
        },
        "target_inference": {
            "primary_estimand": (
                "mean_over_10_target_cities_of_within_city_mean_over_5_checkpoints_utility_difference"
            ),
            "cluster_unit": "target_city",
            "primary_comparison_family": [
                "kbound_v2_vs_always_adapt",
                "kbound_v2_vs_always_freeze",
            ],
            "sign_flip_assumption": (
                "independent_target_city_difference_vectors_with_joint_sign_symmetry_under_each_null"
            ),
            "sign_flip_tests": "exact_city_cluster_sign_flip",
            "multiplicity": "holm_familywise_0.05",
            "confidence_intervals": "nominal_95_percent_city_cluster_bootstrap",
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "bit_generator": "PCG64",
                "interval": "percentile",
                "quantiles": [0.025, 0.975],
                "numpy_quantile_method": "linear",
            },
            "confidence_interval_sign": "positive_favors_kbound_v2",
            "checkpoint_only_baseline_role": "required_secondary_diagnostic",
            "report_all_target_cities_and_all_five_checkpoints": True,
        },
        "preaccess_requirements": {
            "precalibration_seal_binds": [
                "v2_protocol_and_receipt",
                "v2_controller_and_receipt",
                "gate_fit_bundle_and_receipt",
                "source_acceptance_normalizer_and_five_independent_checkpoints",
                "code_configuration_and_environment_hashes",
                "opaque_validation_and_testing_container_raw_byte_identities",
                "zero_access_chronology",
            ],
            "gate_calibration_before_precalibration_seal": "FORBIDDEN",
            "target_before_passing_gate_authorization": "FORBIDDEN",
            "target_outcomes_available_to_live_runner": False,
        },
        "amendment_rule": ("any_change_requires_a_new_version_before_gate_calibration_or_target_access"),
    }


def validate_protocol_v2(document: Mapping[str, Any]) -> None:
    if not isinstance(document, Mapping) or dict(document) != _expected_protocol_v2():
        raise IntegrityError("So2Sat prospective-v2 protocol schema or content drift")


def _load_protocol_v2_pair(
    path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = default_protocol_v2_path() if path is None else Path(path)
    document, receipt = load_verified_json_mapping_with_receipt(
        source,
        receipt_schema=ARTIFACT_RECEIPT_SCHEMA_V2,
    )
    validate_protocol_v2(document)
    return document, receipt


def load_protocol_v2(path: str | Path | None = None) -> dict[str, Any]:
    document, _ = _load_protocol_v2_pair(path)
    return document


def verify_protocol_v2_receipt(path: str | Path | None = None) -> dict[str, Any]:
    _, receipt = _load_protocol_v2_pair(path)
    return receipt


def _controller_without_hash() -> dict[str, Any]:
    return {
        "schema": CONTROLLER_SCHEMA,
        "status": "FROZEN_FROM_RECEIPTED_GATE_FIT_ONLY",
        "controller_id": CONTROLLER_ID,
        "protocol_id": PROTOCOL_ID,
        "action_unit": "city_checkpoint",
        "candidate_id": TENT_CANDIDATE_ID,
        "candidate_config_sha256": _CANDIDATE_CONFIG_SHA256,
        "v1_negative_record": {
            "status": "NO_FEASIBLE_CANDIDATE_STOP_BEFORE_GATE_CAL",
            "selected_candidate_id": None,
            "selection_sha256": _V1_NEGATIVE_SELECTION_SHA256,
            "artifact_receipt": copy.deepcopy(_V1_NEGATIVE_SELECTION_RECEIPT),
            "gate_calibration_rows_read": 0,
            "target_pixels_read": 0,
            "target_labels_read": 0,
        },
        "gate_fit_evidence": {
            "role": GATE_FIT_ROLE,
            "city_count": GATE_FIT_CITY_COUNT,
            "checkpoint_count": len(CHECKPOINT_IDS),
            "cell_count": 45,
            "artifact_receipt": copy.deepcopy(_GATE_FIT_RECEIPT),
            "bundle_sha256": _GATE_FIT_BUNDLE_SHA256,
            "gate_rows_sha256": _GATE_ROWS_SHA256,
            "gate_calibration_rows_read": 0,
            "target_pixels_read": 0,
            "target_labels_read": 0,
        },
        "feature_names": list(FEATURE_NAMES),
        "city_feature_aggregation": "equal_mean_over_exactly_five_checkpoint_probe_documents",
        "checkpoint_ids": list(CHECKPOINT_IDS),
        "checkpoint_encoding": "I(checkpoint=s)-1/5_for_s_0_through_4",
        "standardization": {
            "population": "nine_gate_fit_city_vectors",
            "ddof": 0,
            "zero_scale_rule": "reject",
            "means": list(_FEATURE_MEANS),
            "scales": list(_FEATURE_SCALES),
        },
        "ridge": {
            "penalty": RIDGE_PENALTY,
            "intercept_unpenalized": True,
            "penalized_terms": "all_checkpoint_and_feature_coefficients",
        },
        "intercept": _INTERCEPT,
        "checkpoint_coefficients": list(_CHECKPOINT_COEFFICIENTS),
        "feature_coefficients": list(_FEATURE_COEFFICIENTS),
        "checkpoint_only_baseline": {
            "role": "required_secondary_diagnostic",
            "fit_data": "same_45_gate_fit_cells_only",
            "intercept": _INTERCEPT,
            "checkpoint_coefficients": list(_CHECKPOINT_COEFFICIENTS),
            "ridge_penalty": RIDGE_PENALTY,
            "intercept_unpenalized": True,
            "selection_use": "none",
        },
        "source_bindings": {
            "checkpoint_collection_canonical_sha256": _CHECKPOINT_COLLECTION_SHA256,
            "checkpoints": copy.deepcopy(_SOURCE_CHECKPOINTS),
            "normalizer_sha256": _NORMALIZER_SHA256,
            "source_container_identity_sha256": _SOURCE_CONTAINER_IDENTITY_SHA256,
        },
        "decision_contract": {
            "raw_positive": "ADAPT",
            "raw_negative": "FREEZE",
            "raw_zero": "ABSTAIN",
            "final_adapt": "lower_strictly_greater_than_zero",
            "final_freeze": "upper_strictly_less_than_zero",
            "otherwise": "ABSTAIN",
            "abstain_realized_action": "FREEZE",
        },
        "gate_fit_replay": {
            "status": "RECOMPUTED_FROM_RECEIPT_BOUND_GATE_FIT_ONLY",
            "estimand": "unweighted_mean_over_city_checkpoint_cells",
            "fold_standardization": "training_cities_only_population_sd_ddof_0",
            "ridge_penalty": RIDGE_PENALTY,
            "best_fixed_mean_utility": 0.003170137031133165,
            "loco": {
                "held_out_city_count": 1,
                "gain_over_best_fixed": _LOCO_GAIN,
                "sign_accuracy": _LOCO_SIGN_ACCURACY,
                "raw_action_counts": {"ADAPT": 25, "FREEZE": 20},
            },
            "leave_two_cities_out_gain_over_best_fixed": _LEAVE_TWO_GAIN,
            "leave_three_cities_out_gain_over_best_fixed": _LEAVE_THREE_GAIN,
            "confirmatory_use": "none",
            "selection_after_v2_seal_permitted": False,
        },
    }


def _expected_controller_v2() -> dict[str, Any]:
    document = _controller_without_hash()
    document["controller_sha256"] = stable_sha256(document)
    return document


def validate_controller_v2(document: Mapping[str, Any]) -> None:
    if not isinstance(document, Mapping) or dict(document) != _expected_controller_v2():
        raise IntegrityError("So2Sat prospective-v2 controller schema or content drift")


def _load_controller_v2_pair(
    path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = default_controller_v2_path() if path is None else Path(path)
    document, receipt = load_verified_json_mapping_with_receipt(
        source,
        receipt_schema=ARTIFACT_RECEIPT_SCHEMA_V2,
    )
    validate_controller_v2(document)
    return document, receipt


def load_controller_v2(path: str | Path | None = None) -> dict[str, Any]:
    document, _ = _load_controller_v2_pair(path)
    return document


def verify_controller_v2_receipt(path: str | Path | None = None) -> dict[str, Any]:
    _, receipt = _load_controller_v2_pair(path)
    return receipt


def prospective_v2_code_identity(
    *,
    protocol_pair: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
    controller_pair: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Hash the complete v2 decision, live-routing, and inference surface."""

    if protocol_pair is None:
        protocol_pair = _load_protocol_v2_pair()
    if controller_pair is None:
        controller_pair = _load_controller_v2_pair()
    protocol, protocol_receipt = protocol_pair
    controller, controller_receipt = controller_pair
    validate_protocol_v2(protocol)
    validate_controller_v2(controller)
    files = {name: file_sha256(_module_file(name)) for name in PROSPECTIVE_V2_RUNTIME_MODULES}
    files.update(
        {
            PROTOCOL_BASENAME: require_sha256(
                protocol_receipt.get("artifact_sha256"),
                field="protocol receipt artifact_sha256",
            ),
            f"{PROTOCOL_BASENAME}.receipt.json": stable_sha256(dict(protocol_receipt)),
            CONTROLLER_BASENAME: require_sha256(
                controller_receipt.get("artifact_sha256"),
                field="controller receipt artifact_sha256",
            ),
            f"{CONTROLLER_BASENAME}.receipt.json": stable_sha256(dict(controller_receipt)),
            PRECALIBRATION_TEMPLATE_BASENAME: file_sha256(_module_file(PRECALIBRATION_TEMPLATE_BASENAME)),
        }
    )
    return {
        "files_sha256": files,
        "code_identity_sha256": stable_sha256(files),
    }


def _configuration_identity_from_documents(
    protocol: Mapping[str, Any],
    controller: Mapping[str, Any],
) -> str:
    validate_protocol_v2(protocol)
    validate_controller_v2(controller)
    return stable_sha256(
        {
            "protocol_document_sha256": stable_sha256(protocol),
            "controller_sha256": controller["controller_sha256"],
            "action_unit": "city_checkpoint",
            "calibration": protocol["calibration"],
            "decision_rule": protocol["decision_rule"],
            "gate_authorization": protocol["gate_authorization"],
            "live_execution": protocol["live_execution"],
            "target_inference": protocol["target_inference"],
        }
    )


def configuration_identity_v2() -> str:
    """Return the exact receipt-verified non-runtime v2 configuration identity."""

    protocol, _ = _load_protocol_v2_pair()
    controller, _ = _load_controller_v2_pair()
    return _configuration_identity_from_documents(protocol, controller)


def derive_controller_v2(
    gate_fit_bundle: Mapping[str, Any],
    gate_fit_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the locked v2 design from the already-open v1 gate-fit cells."""

    if dict(gate_fit_receipt) != _GATE_FIT_RECEIPT:
        raise IntegrityError("v2 controller requires the exact receipt-bound Tent gate-fit bundle")
    if stable_sha256(dict(gate_fit_bundle)) != gate_fit_receipt["canonical_document_sha256"]:
        raise IntegrityError("gate-fit canonical document differs from its v2 receipt binding")
    study_binding = gate_fit_bundle.get("study_binding")
    if not isinstance(study_binding, Mapping):
        raise IntegrityError("gate-fit bundle lacks its study binding")
    validate_candidate_bundle(gate_fit_bundle, study_binding=study_binding)
    if (
        gate_fit_bundle.get("role") != GATE_FIT_ROLE
        or gate_fit_bundle.get("bundle_sha256") != _GATE_FIT_BUNDLE_SHA256
        or gate_fit_bundle.get("gate_rows_sha256") != _GATE_ROWS_SHA256
        or gate_fit_bundle.get("candidate_config_sha256") != _CANDIDATE_CONFIG_SHA256
        or gate_fit_bundle.get("checkpoint_collection_canonical_sha256") != _CHECKPOINT_COLLECTION_SHA256
        or gate_fit_bundle.get("normalizer_sha256") != _NORMALIZER_SHA256
        or gate_fit_bundle.get("source_container_identity_sha256") != _SOURCE_CONTAINER_IDENTITY_SHA256
    ):
        raise IntegrityError("gate-fit bundle differs from the frozen v2 evidence")

    cells = sorted(
        gate_fit_bundle["cells"],
        key=lambda cell: (str(cell["city_id"]), int(cell["checkpoint_id"])),
    )
    cities = sorted({str(cell["city_id"]) for cell in cells})
    if len(cities) != GATE_FIT_CITY_COUNT or len(cells) != 45:
        raise IntegrityError("v2 controller requires the exact 9-city by 5-checkpoint grid")
    city_vectors: dict[str, np.ndarray] = {}
    observed_checkpoint_identities: dict[str, dict[str, str]] = {}
    for city in cities:
        city_cells = [cell for cell in cells if cell["city_id"] == city]
        if [str(cell["checkpoint_id"]) for cell in city_cells] != list(CHECKPOINT_IDS):
            raise IntegrityError("v2 controller city is missing one of five checkpoints")
        matrix = np.asarray(
            [[cell["gate_row"]["feature_document"]["features"][name] for name in FEATURE_NAMES] for cell in city_cells],
            dtype=np.float64,
        )
        if matrix.shape != (5, len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
            raise IntegrityError("v2 controller gate-fit features are malformed")
        city_vectors[city] = matrix.mean(axis=0)
        for cell in city_cells:
            checkpoint_id = str(cell["checkpoint_id"])
            identity = {
                "checkpoint_file_sha256": str(cell["gate_row"]["checkpoint_file_sha256"]),
                "checkpoint_tensor_sha256": str(cell["gate_row"]["checkpoint_tensor_sha256"]),
            }
            previous = observed_checkpoint_identities.setdefault(checkpoint_id, identity)
            if previous != identity:
                raise IntegrityError("gate-fit checkpoint identity changes across cities")
    city_matrix = np.vstack([city_vectors[city] for city in cities])
    means = city_matrix.mean(axis=0)
    scales = city_matrix.std(axis=0, ddof=0)
    if np.any(scales <= 0.0) or not np.isfinite(scales).all():
        raise IntegrityError("v2 controller city feature scales must be finite and positive")

    design_rows: list[np.ndarray] = []
    outcomes: list[float] = []
    for cell in cells:
        checkpoint_id = str(cell["checkpoint_id"])
        checkpoint = (
            np.asarray(
                [1.0 if checkpoint_id == candidate else 0.0 for candidate in CHECKPOINT_IDS],
                dtype=np.float64,
            )
            - 0.2
        )
        standardized = (city_vectors[str(cell["city_id"])] - means) / scales
        design_rows.append(np.concatenate(([1.0], checkpoint, standardized)))
        outcomes.append(_finite(cell["observed_benefit"], field="observed_benefit"))
    design = np.vstack(design_rows)
    response = np.asarray(outcomes, dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_PENALTY
    penalty[0, 0] = 0.0
    fitted = np.linalg.solve(design.T @ design + penalty, design.T @ response)

    baseline_design = design[:, :6]
    baseline_penalty = np.eye(6, dtype=np.float64) * RIDGE_PENALTY
    baseline_penalty[0, 0] = 0.0
    baseline = np.linalg.solve(
        baseline_design.T @ baseline_design + baseline_penalty,
        baseline_design.T @ response,
    )
    declared = np.asarray(
        [_INTERCEPT, *_CHECKPOINT_COEFFICIENTS, *_FEATURE_COEFFICIENTS],
        dtype=np.float64,
    )
    declared_baseline = np.asarray([_INTERCEPT, *_CHECKPOINT_COEFFICIENTS])
    replay = _replay_gate_fit_cross_validation(cells, city_vectors)
    if (
        not np.allclose(means, _FEATURE_MEANS, rtol=0.0, atol=5.0e-15)
        or not np.allclose(scales, _FEATURE_SCALES, rtol=0.0, atol=5.0e-15)
        or not np.allclose(fitted, declared, rtol=0.0, atol=5.0e-15)
        or not np.allclose(baseline, declared_baseline, rtol=0.0, atol=5.0e-15)
        or not math.isclose(replay["loco_gain"], _LOCO_GAIN, rel_tol=0.0, abs_tol=5.0e-15)
        or not math.isclose(
            replay["loco_sign_accuracy"],
            _LOCO_SIGN_ACCURACY,
            rel_tol=0.0,
            abs_tol=5.0e-15,
        )
        or replay["loco_adapt"] != 25
        or replay["loco_freeze"] != 20
        or not math.isclose(
            replay["leave_two_gain"],
            _LEAVE_TWO_GAIN,
            rel_tol=0.0,
            abs_tol=5.0e-15,
        )
        or not math.isclose(
            replay["leave_three_gain"],
            _LEAVE_THREE_GAIN,
            rel_tol=0.0,
            abs_tol=5.0e-15,
        )
        or observed_checkpoint_identities != _SOURCE_CHECKPOINTS
    ):
        raise IntegrityError("receipt-bound gate-fit replay differs from the frozen v2 controller")
    return _expected_controller_v2()


def _replay_gate_fit_cross_validation(
    cells: list[Mapping[str, Any]],
    city_vectors: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Replay the exploratory leave-city-out checks without any later data."""

    cities = sorted(city_vectors)
    best_fixed = max(
        0.0,
        float(np.mean([float(cell["observed_benefit"]) for cell in cells])),
    )

    def held_out_predictions(held_out_count: int) -> list[tuple[float, float]]:
        predictions: list[tuple[float, float]] = []
        for held_out in itertools.combinations(cities, held_out_count):
            training_cities = [city for city in cities if city not in held_out]
            training_city_matrix = np.vstack([city_vectors[city] for city in training_cities])
            means = training_city_matrix.mean(axis=0)
            scales = training_city_matrix.std(axis=0, ddof=0)
            if np.any(scales <= 0.0) or not np.isfinite(scales).all():
                raise IntegrityError("leave-city-out feature scale is nonpositive")
            design_rows: list[np.ndarray] = []
            outcomes: list[float] = []
            for cell in cells:
                city = str(cell["city_id"])
                if city not in training_cities:
                    continue
                checkpoint_id = str(cell["checkpoint_id"])
                checkpoint = (
                    np.asarray(
                        [1.0 if checkpoint_id == candidate else 0.0 for candidate in CHECKPOINT_IDS],
                        dtype=np.float64,
                    )
                    - 0.2
                )
                standardized = (city_vectors[city] - means) / scales
                design_rows.append(np.concatenate(([1.0], checkpoint, standardized)))
                outcomes.append(float(cell["observed_benefit"]))
            design = np.vstack(design_rows)
            response = np.asarray(outcomes, dtype=np.float64)
            penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_PENALTY
            penalty[0, 0] = 0.0
            fitted = np.linalg.solve(
                design.T @ design + penalty,
                design.T @ response,
            )
            for cell in cells:
                city = str(cell["city_id"])
                if city not in held_out:
                    continue
                checkpoint_id = str(cell["checkpoint_id"])
                checkpoint = (
                    np.asarray(
                        [1.0 if checkpoint_id == candidate else 0.0 for candidate in CHECKPOINT_IDS],
                        dtype=np.float64,
                    )
                    - 0.2
                )
                standardized = (city_vectors[city] - means) / scales
                prediction = float(np.concatenate(([1.0], checkpoint, standardized)) @ fitted)
                predictions.append((float(cell["observed_benefit"]), prediction))
        return predictions

    def routed_gain(predictions: list[tuple[float, float]]) -> float:
        utility = float(np.mean([benefit if prediction > 0.0 else 0.0 for benefit, prediction in predictions]))
        return utility - best_fixed

    loco = held_out_predictions(1)
    leave_two = held_out_predictions(2)
    leave_three = held_out_predictions(3)
    return {
        "loco_gain": routed_gain(loco),
        "loco_sign_accuracy": float(np.mean([(prediction > 0.0) == (benefit > 0.0) for benefit, prediction in loco])),
        "loco_adapt": sum(prediction > 0.0 for _, prediction in loco),
        "loco_freeze": sum(prediction <= 0.0 for _, prediction in loco),
        "leave_two_gain": routed_gain(leave_two),
        "leave_three_gain": routed_gain(leave_three),
    }


def _city_feature_vector(city_probe_features: Any) -> np.ndarray:
    if not isinstance(city_probe_features, Mapping) or set(city_probe_features) != set(CHECKPOINT_IDS):
        raise IntegrityError("city features require exactly checkpoints 0 through 4")
    rows: list[list[float]] = []
    for checkpoint_id in CHECKPOINT_IDS:
        values = city_probe_features[checkpoint_id]
        if not isinstance(values, Mapping) or set(values) != set(FEATURE_NAMES):
            raise IntegrityError("each checkpoint requires the exact five v2 features")
        rows.append([_finite(values[name], field=f"features.{checkpoint_id}.{name}") for name in FEATURE_NAMES])
    matrix = np.asarray(rows, dtype=np.float64)
    result: np.ndarray = np.asarray(matrix.mean(axis=0), dtype=np.float64)
    return result


def raw_prediction_v2(
    controller: Mapping[str, Any],
    *,
    checkpoint_id: str,
    city_probe_features: Mapping[str, Mapping[str, Any]],
) -> float:
    validate_controller_v2(controller)
    if checkpoint_id not in CHECKPOINT_IDS:
        raise IntegrityError("checkpoint_id must be one of 0 through 4")
    city_vector = _city_feature_vector(city_probe_features)
    means = np.asarray(controller["standardization"]["means"], dtype=np.float64)
    scales = np.asarray(controller["standardization"]["scales"], dtype=np.float64)
    feature_coefficients = np.asarray(controller["feature_coefficients"], dtype=np.float64)
    checkpoint_coefficients = np.asarray(controller["checkpoint_coefficients"], dtype=np.float64)
    checkpoint = (
        np.asarray(
            [1.0 if checkpoint_id == candidate else 0.0 for candidate in CHECKPOINT_IDS],
            dtype=np.float64,
        )
        - 0.2
    )
    prediction = float(
        controller["intercept"]
        + checkpoint @ checkpoint_coefficients
        + ((city_vector - means) / scales) @ feature_coefficients
    )
    if not math.isfinite(prediction):  # pragma: no cover - validated inputs are finite
        raise IntegrityError("v2 controller produced a non-finite prediction")
    return prediction


def route_city_checkpoint_v2(
    controller: Mapping[str, Any],
    *,
    checkpoint_id: str,
    city_probe_features: Mapping[str, Mapping[str, Any]],
    interval_radius: Any,
) -> dict[str, Any]:
    validate_controller_v2(controller)
    radius = _finite(interval_radius, field="interval_radius")
    if radius < 0.0:
        raise IntegrityError("interval_radius must be non-negative")
    if checkpoint_id not in CHECKPOINT_IDS:
        raise IntegrityError("checkpoint_id must be one of 0 through 4")
    try:
        prediction = raw_prediction_v2(
            controller,
            checkpoint_id=checkpoint_id,
            city_probe_features=city_probe_features,
        )
    except IntegrityError:
        return {
            "schema": ACTION_SCHEMA,
            "support_status": "INSUFFICIENT_OR_NONFINITE_INPUT",
            "checkpoint_id": checkpoint_id,
            "delta_hat": None,
            "interval_radius": radius,
            "lower": None,
            "upper": None,
            "decision": "ABSTAIN",
            "realized_action": "FREEZE",
        }
    lower = prediction - radius
    upper = prediction + radius
    decision = "ADAPT" if lower > 0.0 else "FREEZE" if upper < 0.0 else "ABSTAIN"
    return {
        "schema": ACTION_SCHEMA,
        "support_status": "SUPPORTED",
        "checkpoint_id": checkpoint_id,
        "delta_hat": prediction,
        "interval_radius": radius,
        "lower": lower,
        "upper": upper,
        "decision": decision,
        "realized_action": "ADAPT" if decision == "ADAPT" else "FREEZE",
    }


def _expected_template_v2() -> dict[str, Any]:
    return {
        "schema": PRECALIBRATION_TEMPLATE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "controller_id": CONTROLLER_ID,
        "status": "TEMPLATE_BLOCKED_NOT_AUTHORIZED",
        "purpose": "populate_only_after_external_raw_byte_identities_are_available",
        "missing_required_bindings": [
            "opaque_validation_container_raw_byte_identity",
            "opaque_testing_container_raw_byte_identity",
            "source_artifact_receipts",
            "runtime_code_configuration_environment_hashes",
            "zero_access_chronology_declaration",
        ],
        "forbidden_inputs": [
            "gate_calibration_pixels_labels_metrics_or_outputs",
            "target_pixels_labels_metrics_or_outputs",
            "private_absolute_paths",
        ],
        "gate_calibration_authorized": False,
        "target_access_authorized": False,
        "note": "receipt_integrity_is_not_an_external_timestamp_or_proof_of_historical_nonaccess",
    }


def load_precalibration_template_v2(path: str | Path | None = None) -> dict[str, Any]:
    source = default_precalibration_template_v2_path() if path is None else Path(path)
    document = strict_json_load(source)
    if not isinstance(document, Mapping) or dict(document) != _expected_template_v2():
        raise IntegrityError("So2Sat v2 pre-calibration template drift")
    return dict(document)


def _validate_source_bindings(value: Any, controller: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "population_manifest_artifact",
        "population_identity_sha256",
        "source_postrun_acceptance_artifact",
        "source_postrun_acceptance_sha256",
        "source_checkpoint_collection_artifact",
        "source_checkpoint_collection_sha256",
        "source_checkpoints",
        "source_normalizer_artifact",
        "source_normalizer_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise IntegrityError("source_bindings has unknown or missing fields")
    result = copy.deepcopy(dict(value))
    for name in (
        "population_manifest_artifact",
        "source_postrun_acceptance_artifact",
        "source_checkpoint_collection_artifact",
        "source_normalizer_artifact",
    ):
        _validate_artifact_binding(result[name], field=f"source_bindings.{name}")
    for name in (
        "population_identity_sha256",
        "source_postrun_acceptance_sha256",
        "source_checkpoint_collection_sha256",
        "source_normalizer_sha256",
    ):
        require_sha256(result[name], field=f"source_bindings.{name}")
    if (
        result["source_checkpoint_collection_sha256"]
        != controller["source_bindings"]["checkpoint_collection_canonical_sha256"]
        or result["source_normalizer_sha256"] != controller["source_bindings"]["normalizer_sha256"]
    ):
        raise IntegrityError("source artifacts differ from the controller gate-fit bindings")
    checkpoints = result["source_checkpoints"]
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(CHECKPOINT_IDS):
        raise IntegrityError("source bindings require exactly five independent checkpoints")
    identities: list[tuple[str, str]] = []
    for checkpoint_id in CHECKPOINT_IDS:
        identity = checkpoints[checkpoint_id]
        if not isinstance(identity, Mapping) or set(identity) != {
            "checkpoint_file_sha256",
            "checkpoint_tensor_sha256",
        }:
            raise IntegrityError("source checkpoint identity schema drift")
        file_digest = require_sha256(
            identity["checkpoint_file_sha256"],
            field=f"source_checkpoints.{checkpoint_id}.checkpoint_file_sha256",
        )
        tensor_digest = require_sha256(
            identity["checkpoint_tensor_sha256"],
            field=f"source_checkpoints.{checkpoint_id}.checkpoint_tensor_sha256",
        )
        identities.append((file_digest, tensor_digest))
    if len({item[0] for item in identities}) != 5 or len({item[1] for item in identities}) != 5:
        raise IntegrityError("source bindings do not prove five independent checkpoints")
    if dict(checkpoints) != controller["source_bindings"]["checkpoints"]:
        raise IntegrityError("source bindings differ from the gate-fit checkpoint identities")
    return result


def _validate_runtime_bindings(
    value: Any,
    *,
    protocol: Mapping[str, Any],
    protocol_receipt: Mapping[str, Any],
    controller: Mapping[str, Any],
    controller_receipt: Mapping[str, Any],
) -> dict[str, str]:
    expected = {
        "code_identity_sha256",
        "configuration_identity_sha256",
        "environment_identity_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise IntegrityError("runtime_bindings has unknown or missing fields")
    result = {name: require_sha256(value[name], field=f"runtime_bindings.{name}") for name in sorted(expected)}
    expected_code_identity = prospective_v2_code_identity(
        protocol_pair=(protocol, protocol_receipt),
        controller_pair=(controller, controller_receipt),
    )["code_identity_sha256"]
    if result["code_identity_sha256"] != expected_code_identity:
        raise IntegrityError("v2 runtime binding has a stale code identity")
    if result["configuration_identity_sha256"] != _configuration_identity_from_documents(
        protocol,
        controller,
    ):
        raise IntegrityError("v2 runtime binding has a stale configuration identity")
    return result


def _validate_opaque_target_identities(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {"validation", "testing"}:
        raise IntegrityError("opaque target identities require validation and testing containers")
    roles = {
        "validation": ("label_free_probe_pixels", "validation.h5"),
        "testing": ("sealed_evaluation_pixels_and_outcomes", "testing.h5"),
    }
    expected = {
        "container_role",
        "artifact_basename",
        "artifact_bytes",
        "raw_file_sha256",
        "hashing_method",
        "hdf5_datasets_opened",
    }
    result: dict[str, dict[str, Any]] = {}
    for split, (role, basename) in roles.items():
        identity = value[split]
        if not isinstance(identity, Mapping) or set(identity) != expected:
            raise IntegrityError(f"opaque target identity {split} schema drift")
        if (
            identity.get("container_role") != role
            or _portable_basename(
                identity.get("artifact_basename"),
                field=f"opaque_target_identities.{split}.artifact_basename",
            )
            != basename
            or identity.get("hashing_method") != "sha256_raw_bytes_without_hdf5_deserialization"
            or identity.get("hdf5_datasets_opened") != 0
        ):
            raise IntegrityError("opaque target identities must be raw-byte hashes without HDF5 deserialization")
        _positive_integer(identity.get("artifact_bytes"), field=f"{split}.artifact_bytes")
        require_sha256(identity.get("raw_file_sha256"), field=f"{split}.raw_file_sha256")
        result[split] = dict(identity)
    if result["validation"]["raw_file_sha256"] == result["testing"]["raw_file_sha256"]:
        raise IntegrityError("validation and testing opaque identities must be distinct")
    return result


def _validate_chronology(value: Any) -> dict[str, Any]:
    expected = {
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
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise IntegrityError("v2 zero-access chronology is incomplete or makes an unsupported claim")
    return copy.deepcopy(expected)


def build_precalibration_seal_v2(
    *,
    protocol: Mapping[str, Any],
    controller: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    runtime_bindings: Mapping[str, Any],
    opaque_target_identities: Mapping[str, Any],
    chronology: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a pre-calibration seal without accepting or opening any data path."""

    validate_protocol_v2(protocol)
    validate_controller_v2(controller)
    sealed_protocol, protocol_receipt = _load_protocol_v2_pair()
    sealed_controller, controller_receipt = _load_controller_v2_pair()
    if dict(protocol) != sealed_protocol or dict(controller) != sealed_controller:
        raise IntegrityError("pre-calibration inputs differ from the receipt-verified configuration snapshots")
    document: dict[str, Any] = {
        "schema": PRECALIBRATION_SEAL_SCHEMA,
        "status": "SEALED_BEFORE_GATE_CALIBRATION_AND_TARGET_ACCESS",
        "protocol_id": PROTOCOL_ID,
        "protocol_artifact": protocol_receipt,
        "protocol_document_sha256": stable_sha256(dict(protocol)),
        "controller_id": CONTROLLER_ID,
        "controller_artifact": controller_receipt,
        "controller_sha256": controller["controller_sha256"],
        "gate_fit_evidence": copy.deepcopy(controller["gate_fit_evidence"]),
        "candidate_id": controller["candidate_id"],
        "candidate_config_sha256": controller["candidate_config_sha256"],
        "action_unit": "city_checkpoint",
        "source_bindings": _validate_source_bindings(source_bindings, controller),
        "runtime_bindings": _validate_runtime_bindings(
            runtime_bindings,
            protocol=sealed_protocol,
            protocol_receipt=protocol_receipt,
            controller=sealed_controller,
            controller_receipt=controller_receipt,
        ),
        "opaque_target_identities": _validate_opaque_target_identities(opaque_target_identities),
        "chronology": _validate_chronology(chronology),
        "access_audit": {
            "gate_calibration_rows_read": 0,
            "gate_calibration_pixels_read": 0,
            "gate_calibration_labels_read": 0,
            "target_pixels_read": 0,
            "target_labels_read": 0,
            "target_inputs": [],
        },
        "gate_calibration_authorized": True,
        "target_access_authorized": False,
        "receipt_scope_disclosure": ("content_integrity_only_not_external_timestamp_or_historical_nonaccess_proof"),
    }
    _reject_private_paths(document)
    document["precalibration_seal_sha256"] = stable_sha256(document)
    _validate_precalibration_seal_v2_with_pairs(
        document,
        protocol_pair=(sealed_protocol, protocol_receipt),
        controller_pair=(sealed_controller, controller_receipt),
    )
    return document


def _validate_precalibration_seal_v2_with_pairs(
    document: Mapping[str, Any],
    *,
    protocol_pair: tuple[Mapping[str, Any], Mapping[str, Any]],
    controller_pair: tuple[Mapping[str, Any], Mapping[str, Any]],
) -> None:
    expected_keys = {
        "schema",
        "status",
        "protocol_id",
        "protocol_artifact",
        "protocol_document_sha256",
        "controller_id",
        "controller_artifact",
        "controller_sha256",
        "gate_fit_evidence",
        "candidate_id",
        "candidate_config_sha256",
        "action_unit",
        "source_bindings",
        "runtime_bindings",
        "opaque_target_identities",
        "chronology",
        "access_audit",
        "gate_calibration_authorized",
        "target_access_authorized",
        "receipt_scope_disclosure",
        "precalibration_seal_sha256",
    }
    if not isinstance(document, Mapping) or set(document) != expected_keys:
        raise IntegrityError("v2 pre-calibration seal has unknown or missing fields")
    protocol, protocol_receipt = protocol_pair
    controller, controller_receipt = controller_pair
    validate_protocol_v2(protocol)
    validate_controller_v2(controller)
    if (
        document.get("schema") != PRECALIBRATION_SEAL_SCHEMA
        or document.get("status") != "SEALED_BEFORE_GATE_CALIBRATION_AND_TARGET_ACCESS"
        or document.get("protocol_id") != PROTOCOL_ID
        or document.get("protocol_artifact") != protocol_receipt
        or document.get("protocol_document_sha256") != stable_sha256(protocol)
        or document.get("controller_id") != CONTROLLER_ID
        or document.get("controller_artifact") != controller_receipt
        or document.get("controller_sha256") != controller["controller_sha256"]
        or document.get("gate_fit_evidence") != controller["gate_fit_evidence"]
        or document.get("candidate_id") != TENT_CANDIDATE_ID
        or document.get("candidate_config_sha256") != _CANDIDATE_CONFIG_SHA256
        or document.get("action_unit") != "city_checkpoint"
        or document.get("gate_calibration_authorized") is not True
        or document.get("target_access_authorized") is not False
        or document.get("receipt_scope_disclosure")
        != "content_integrity_only_not_external_timestamp_or_historical_nonaccess_proof"
    ):
        raise IntegrityError("unknown or unsealed v2 pre-calibration artifact")
    _validate_source_bindings(document["source_bindings"], controller)
    _validate_runtime_bindings(
        document["runtime_bindings"],
        protocol=protocol,
        protocol_receipt=protocol_receipt,
        controller=controller,
        controller_receipt=controller_receipt,
    )
    _validate_opaque_target_identities(document["opaque_target_identities"])
    _validate_chronology(document["chronology"])
    if document.get("access_audit") != {
        "gate_calibration_rows_read": 0,
        "gate_calibration_pixels_read": 0,
        "gate_calibration_labels_read": 0,
        "target_pixels_read": 0,
        "target_labels_read": 0,
        "target_inputs": [],
    }:
        raise IntegrityError("v2 pre-calibration seal does not prove its declared zero-access state")
    _reject_private_paths(document)
    claimed = require_sha256(
        document.get("precalibration_seal_sha256"),
        field="precalibration_seal_sha256",
    )
    unsigned = dict(document)
    unsigned.pop("precalibration_seal_sha256")
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("v2 pre-calibration seal self-hash mismatch")


def validate_precalibration_seal_v2(document: Mapping[str, Any]) -> None:
    """Validate one seal against single receipt-verified configuration reads."""

    controller_pair = _load_controller_v2_pair()
    protocol_pair = _load_protocol_v2_pair()
    _validate_precalibration_seal_v2_with_pairs(
        document,
        protocol_pair=protocol_pair,
        controller_pair=controller_pair,
    )


def _validate_calibration_bundle(
    bundle: Mapping[str, Any],
    *,
    precalibration_seal: Mapping[str, Any],
    controller: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    expected = {
        "schema",
        "status",
        "precalibration_seal_sha256",
        "controller_sha256",
        "action_unit",
        "cells",
        "gate_calibration_city_count",
        "checkpoint_count",
        "gate_calibration_rows_read",
        "target_pixels_read",
        "target_labels_read",
        "target_inputs",
        "bundle_sha256",
    }
    if not isinstance(bundle, Mapping) or set(bundle) != expected:
        raise IntegrityError("v2 gate-calibration bundle has unknown or missing fields")
    claimed = require_sha256(bundle.get("bundle_sha256"), field="bundle_sha256")
    unsigned = dict(bundle)
    unsigned.pop("bundle_sha256")
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("v2 gate-calibration bundle self-hash mismatch")
    if (
        bundle.get("schema") != CALIBRATION_BUNDLE_SCHEMA
        or bundle.get("status") != "COMPLETE_19_CITY_95_CELL_GATE_CALIBRATION"
        or bundle.get("precalibration_seal_sha256") != precalibration_seal["precalibration_seal_sha256"]
        or bundle.get("controller_sha256") != controller["controller_sha256"]
        or bundle.get("action_unit") != "city_checkpoint"
        or bundle.get("gate_calibration_city_count") != GATE_CALIBRATION_CITY_COUNT
        or bundle.get("checkpoint_count") != len(CHECKPOINT_IDS)
        or bundle.get("gate_calibration_rows_read") != GATE_CALIBRATION_CELL_COUNT
        or bundle.get("target_pixels_read") != 0
        or bundle.get("target_labels_read") != 0
        or bundle.get("target_inputs") != []
    ):
        raise IntegrityError("v2 gate-calibration bundle identity or access contract drift")
    cells = bundle.get("cells")
    if not isinstance(cells, list) or len(cells) != GATE_CALIBRATION_CELL_COUNT:
        raise IntegrityError("v2 gate-calibration bundle must contain exactly 95 cells")
    cell_keys = {
        "city_id",
        "checkpoint_id",
        "feature_document",
        "observed_benefit",
        "trace_sha256",
    }
    identities: list[tuple[str, str]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping) or set(cell) != cell_keys:
            if isinstance(cell, Mapping) and "feature_values" in cell:
                raise IntegrityError(f"v2 gate-calibration cell {index} requires a hashed feature document")
            raise IntegrityError(f"v2 gate-calibration cell {index} schema drift")
        city = cell.get("city_id")
        checkpoint = cell.get("checkpoint_id")
        if not isinstance(city, str) or not city or checkpoint not in CHECKPOINT_IDS:
            raise IntegrityError(f"v2 gate-calibration cell {index} identity drift")
        _reject_private_paths(city)
        feature_document = cell.get("feature_document")
        if not isinstance(feature_document, Mapping):
            raise IntegrityError(f"v2 gate-calibration cell {index} requires a hashed feature document")
        validate_feature_document(feature_document)
        benefit = _finite(cell.get("observed_benefit"), field=f"cells.{index}.observed_benefit")
        if not -1.0 <= benefit <= 1.0:
            raise IntegrityError("observed accuracy benefit must lie in [-1, 1]")
        require_sha256(cell.get("trace_sha256"), field=f"cells.{index}.trace_sha256")
        identities.append((city, str(checkpoint)))
    cities = sorted({city for city, _ in identities})
    expected_grid = {(city, checkpoint) for city in cities for checkpoint in CHECKPOINT_IDS}
    if (
        len(cities) != GATE_CALIBRATION_CITY_COUNT
        or identities != sorted(identities, key=lambda item: (item[0], int(item[1])))
        or len(set(identities)) != GATE_CALIBRATION_CELL_COUNT
        or set(identities) != expected_grid
    ):
        raise IntegrityError("v2 gate-calibration bundle is not the exact sorted 19-by-5 grid")
    return cells


def _checkpoint_only_prediction(controller: Mapping[str, Any], checkpoint_id: str) -> float:
    coefficients = np.asarray(
        controller["checkpoint_only_baseline"]["checkpoint_coefficients"],
        dtype=np.float64,
    )
    encoded = (
        np.asarray(
            [1.0 if checkpoint_id == candidate else 0.0 for candidate in CHECKPOINT_IDS],
            dtype=np.float64,
        )
        - 0.2
    )
    return float(controller["checkpoint_only_baseline"]["intercept"] + encoded @ coefficients)


def build_gate_screen_v2(
    *,
    precalibration_seal: Mapping[str, Any],
    controller: Mapping[str, Any],
    calibration_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the fixed city-cluster conformal screen from 95 development cells."""

    validate_precalibration_seal_v2(precalibration_seal)
    validate_controller_v2(controller)
    cells = _validate_calibration_bundle(
        calibration_bundle,
        precalibration_seal=precalibration_seal,
        controller=controller,
    )
    by_city: dict[str, dict[str, Mapping[str, Any]]] = {}
    for cell in cells:
        by_city.setdefault(str(cell["city_id"]), {})[str(cell["checkpoint_id"])] = cell
    raw_rows: list[dict[str, Any]] = []
    city_max_residuals: dict[str, float] = {}
    for city in sorted(by_city):
        city_cells = by_city[city]
        city_features = {
            checkpoint: {name: city_cells[checkpoint]["feature_document"]["features"][name] for name in FEATURE_NAMES}
            for checkpoint in CHECKPOINT_IDS
        }
        city_residuals: list[float] = []
        for checkpoint in CHECKPOINT_IDS:
            cell = city_cells[checkpoint]
            prediction = raw_prediction_v2(
                controller,
                checkpoint_id=checkpoint,
                city_probe_features=city_features,
            )
            benefit = float(cell["observed_benefit"])
            residual = abs(benefit - prediction)
            city_residuals.append(residual)
            raw_rows.append(
                {
                    "city_id": city,
                    "checkpoint_id": checkpoint,
                    "trace_sha256": cell["trace_sha256"],
                    "observed_benefit": benefit,
                    "delta_hat": prediction,
                    "absolute_residual": residual,
                }
            )
        city_max_residuals[city] = max(city_residuals)
    ordered_residuals = sorted(city_max_residuals.values())
    radius = ordered_residuals[CONFORMAL_RANK - 1]

    output_rows: list[dict[str, Any]] = []
    controller_utilities: list[float] = []
    checkpoint_utilities: list[float] = []
    benefits: list[float] = []
    for row in raw_rows:
        lower = row["delta_hat"] - radius
        upper = row["delta_hat"] + radius
        decision = "ADAPT" if lower > 0.0 else "FREEZE" if upper < 0.0 else "ABSTAIN"
        realized = "ADAPT" if decision == "ADAPT" else "FREEZE"
        checkpoint_prediction = _checkpoint_only_prediction(controller, str(row["checkpoint_id"]))
        checkpoint_decision = "ADAPT" if checkpoint_prediction > 0.0 else "FREEZE"
        benefit = float(row["observed_benefit"])
        controller_utility = benefit if realized == "ADAPT" else 0.0
        checkpoint_utility = benefit if checkpoint_decision == "ADAPT" else 0.0
        benefits.append(benefit)
        controller_utilities.append(controller_utility)
        checkpoint_utilities.append(checkpoint_utility)
        output_rows.append(
            {
                **row,
                "interval_radius": radius,
                "lower": lower,
                "upper": upper,
                "decision": decision,
                "realized_action": realized,
                "realized_utility": controller_utility,
                "checkpoint_only_prediction": checkpoint_prediction,
                "checkpoint_only_decision": checkpoint_decision,
                "checkpoint_only_realized_utility": checkpoint_utility,
            }
        )
    decisions = [row["decision"] for row in output_rows]
    decision_counts = {decision: decisions.count(decision) for decision in ("ADAPT", "FREEZE", "ABSTAIN")}
    direct_action_cities = {
        decision: sorted({str(row["city_id"]) for row in output_rows if row["decision"] == decision})
        for decision in ("ADAPT", "FREEZE")
    }
    checks = {
        "complete_19_city_95_cell_grid": True,
        "at_least_7_direct_adapt_cities": (len(direct_action_cities["ADAPT"]) >= MINIMUM_DIRECT_ACTION_CITIES),
        "at_least_7_direct_freeze_cities": (len(direct_action_cities["FREEZE"]) >= MINIMUM_DIRECT_ACTION_CITIES),
        "zero_target_access": True,
    }
    passed = all(checks.values())
    screen: dict[str, Any] = {
        "schema": GATE_SCREEN_SCHEMA,
        "status": (
            "PASSED_GATE_AUTHORIZATION_SCREEN" if passed else "FAILED_GATE_AUTHORIZATION_SCREEN_NO_TARGET_ACCESS"
        ),
        "protocol_id": PROTOCOL_ID,
        "precalibration_seal_sha256": precalibration_seal["precalibration_seal_sha256"],
        "controller_sha256": controller["controller_sha256"],
        "calibration_bundle_sha256": calibration_bundle["bundle_sha256"],
        "action_unit": "city_checkpoint",
        "calibration": {
            "method": "split_conformal_city_max_checkpoint_absolute_residual",
            "cluster_unit": "gate_calibration_city",
            "city_residual_aggregation": "maximum_absolute_residual_over_five_checkpoints",
            "alpha": CALIBRATION_ALPHA,
            "order_statistic_rank_one_based": CONFORMAL_RANK,
            "city_max_absolute_residuals": city_max_residuals,
            "ordered_city_max_absolute_residuals": ordered_residuals,
            "interval_radius": radius,
        },
        "cell_count": len(output_rows),
        "cells": output_rows,
        "decision_counts": decision_counts,
        "direct_action_cities": direct_action_cities,
        "minimum_direct_action_cities": MINIMUM_DIRECT_ACTION_CITIES,
        "checks": checks,
        "passed": passed,
        "utility_diagnostics": {
            "kbound_v2_mean_realized_utility": float(np.mean(controller_utilities)),
            "always_adapt_mean_realized_utility": float(np.mean(benefits)),
            "always_freeze_mean_realized_utility": 0.0,
        },
        "checkpoint_only_baseline": {
            "role": "required_secondary_diagnostic",
            "mean_realized_utility": float(np.mean(checkpoint_utilities)),
            "selection_use": "none",
        },
        "target_pixels_read": 0,
        "target_labels_read": 0,
        "target_inputs": [],
    }
    screen["gate_screen_sha256"] = stable_sha256(screen)
    return screen


def authorize_target_execution_v2(
    *,
    precalibration_seal: Mapping[str, Any],
    controller: Mapping[str, Any],
    calibration_bundle: Mapping[str, Any],
    submitted_gate_screen: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize label-free target probes only after exact screen replay."""

    expected = build_gate_screen_v2(
        precalibration_seal=precalibration_seal,
        controller=controller,
        calibration_bundle=calibration_bundle,
    )
    if not isinstance(submitted_gate_screen, Mapping) or dict(submitted_gate_screen) != expected:
        raise IntegrityError("submitted v2 gate screen differs from canonical recomputation")
    if not expected["passed"]:
        raise IntegrityError("v2 gate screen failed the conservative seven-city direct-action requirements")
    authorization: dict[str, Any] = {
        "schema": TARGET_AUTHORIZATION_SCHEMA,
        "status": "AUTHORIZED_AFTER_SEVEN_CITY_BIDIRECTIONAL_SCREEN",
        "protocol_id": PROTOCOL_ID,
        "precalibration_seal_sha256": precalibration_seal["precalibration_seal_sha256"],
        "controller_sha256": controller["controller_sha256"],
        "calibration_bundle_sha256": calibration_bundle["bundle_sha256"],
        "gate_screen_sha256": expected["gate_screen_sha256"],
        "action_unit": "city_checkpoint",
        "target_probe_pixel_access_authorized": True,
        "target_evaluation_pixel_access_authorized_only_after_actions_sealed": True,
        "target_outcome_access_authorized": False,
        "target_pixels_read_before_authorization": 0,
        "target_labels_read_before_authorization": 0,
        "abstain_realized_action": "FREEZE",
    }
    authorization["authorization_sha256"] = stable_sha256(authorization)
    validate_target_authorization_v2(
        authorization,
        precalibration_seal=precalibration_seal,
        controller=controller,
        calibration_bundle=calibration_bundle,
        gate_screen=expected,
    )
    return authorization


def validate_target_authorization_v2(
    document: Mapping[str, Any],
    *,
    precalibration_seal: Mapping[str, Any],
    controller: Mapping[str, Any],
    calibration_bundle: Mapping[str, Any],
    gate_screen: Mapping[str, Any],
) -> None:
    """Validate every semantic and hash binding in a v2 target authorization."""

    _validate_target_authorization_document_v2(document)
    validate_precalibration_seal_v2(precalibration_seal)
    validate_controller_v2(controller)
    if gate_screen.get("passed") is not True:
        raise IntegrityError("v2 target authorization requires a passing gate screen")
    if (
        document.get("precalibration_seal_sha256") != precalibration_seal["precalibration_seal_sha256"]
        or document.get("controller_sha256") != controller["controller_sha256"]
        or document.get("calibration_bundle_sha256") != calibration_bundle.get("bundle_sha256")
        or document.get("gate_screen_sha256") != gate_screen.get("gate_screen_sha256")
    ):
        raise IntegrityError("v2 target authorization provenance drift")


def _validate_target_authorization_document_v2(document: Mapping[str, Any]) -> None:
    """Validate a portable authorization without dereferencing upstream artifacts."""

    expected_keys = {
        "schema",
        "status",
        "protocol_id",
        "precalibration_seal_sha256",
        "controller_sha256",
        "calibration_bundle_sha256",
        "gate_screen_sha256",
        "action_unit",
        "target_probe_pixel_access_authorized",
        "target_evaluation_pixel_access_authorized_only_after_actions_sealed",
        "target_outcome_access_authorized",
        "target_pixels_read_before_authorization",
        "target_labels_read_before_authorization",
        "abstain_realized_action",
        "authorization_sha256",
    }
    if not isinstance(document, Mapping) or set(document) != expected_keys:
        raise IntegrityError("v2 target authorization has unknown or missing fields")
    if document.get("target_outcome_access_authorized") is not False:
        raise IntegrityError("v2 target authorization can never grant target outcome access")
    controller = load_controller_v2()
    if (
        document.get("schema") != TARGET_AUTHORIZATION_SCHEMA
        or document.get("status") != "AUTHORIZED_AFTER_SEVEN_CITY_BIDIRECTIONAL_SCREEN"
        or document.get("protocol_id") != PROTOCOL_ID
        or document.get("controller_sha256") != controller["controller_sha256"]
        or document.get("action_unit") != "city_checkpoint"
        or document.get("target_probe_pixel_access_authorized") is not True
        or document.get("target_evaluation_pixel_access_authorized_only_after_actions_sealed") is not True
        or document.get("target_pixels_read_before_authorization") != 0
        or document.get("target_labels_read_before_authorization") != 0
        or document.get("abstain_realized_action") != "FREEZE"
    ):
        raise IntegrityError("v2 target authorization semantic or provenance drift")
    for field in (
        "precalibration_seal_sha256",
        "controller_sha256",
        "calibration_bundle_sha256",
        "gate_screen_sha256",
    ):
        require_sha256(document.get(field), field=field)
    claimed = require_sha256(
        document.get("authorization_sha256"),
        field="authorization_sha256",
    )
    unsigned = dict(document)
    unsigned.pop("authorization_sha256")
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("v2 target authorization self-hash mismatch")


def _validate_scored_target_bundle_v2(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    expected = {
        "schema",
        "status",
        "protocol_id",
        "controller_sha256",
        "authorization_sha256",
        "action_unit",
        "cells",
        "target_city_count",
        "checkpoint_count",
        "outcome_reveal_count",
        "bundle_sha256",
    }
    if not isinstance(bundle, Mapping) or set(bundle) != expected:
        raise IntegrityError("v2 scored-target bundle has unknown or missing fields")
    claimed = require_sha256(bundle.get("bundle_sha256"), field="bundle_sha256")
    unsigned = dict(bundle)
    unsigned.pop("bundle_sha256")
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("v2 scored-target bundle self-hash mismatch")
    controller = load_controller_v2()
    if (
        bundle.get("schema") != SCORED_TARGET_BUNDLE_SCHEMA
        or bundle.get("status") != "COMPLETE_10_CITY_50_CELL_SINGLE_REVEAL"
        or bundle.get("protocol_id") != PROTOCOL_ID
        or bundle.get("controller_sha256") != controller["controller_sha256"]
        or bundle.get("action_unit") != "city_checkpoint"
        or bundle.get("target_city_count") != TARGET_CITY_COUNT
        or bundle.get("checkpoint_count") != len(CHECKPOINT_IDS)
        or bundle.get("outcome_reveal_count") != 1
    ):
        raise IntegrityError("v2 scored-target bundle identity or reveal contract drift")
    require_sha256(bundle.get("authorization_sha256"), field="authorization_sha256")
    cells = bundle.get("cells")
    if not isinstance(cells, list) or len(cells) != TARGET_CELL_COUNT:
        raise IntegrityError("v2 scored-target bundle must contain exactly 50 cells")
    cell_keys = {
        "city_id",
        "checkpoint_id",
        "decision",
        "realized_action",
        "observed_benefit",
        "action_sha256",
    }
    identities: list[tuple[str, str]] = []
    action_hashes: list[str] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping) or set(cell) != cell_keys:
            raise IntegrityError(f"v2 scored-target cell {index} schema drift")
        city = cell.get("city_id")
        checkpoint = cell.get("checkpoint_id")
        decision = cell.get("decision")
        realized = cell.get("realized_action")
        if not isinstance(city, str) or not city or checkpoint not in CHECKPOINT_IDS:
            raise IntegrityError(f"v2 scored-target cell {index} identity drift")
        _reject_private_paths(city)
        if decision not in {"ADAPT", "FREEZE", "ABSTAIN"}:
            raise IntegrityError(f"v2 scored-target cell {index} decision drift")
        expected_realized = "ADAPT" if decision == "ADAPT" else "FREEZE"
        if realized != expected_realized:
            if decision == "ABSTAIN":
                raise IntegrityError("v2 target ABSTAIN must realize FREEZE")
            raise IntegrityError(f"v2 scored-target cell {index} realized-action drift")
        benefit = _finite(cell.get("observed_benefit"), field=f"cells.{index}.observed_benefit")
        if not -1.0 <= benefit <= 1.0:
            raise IntegrityError("target observed accuracy benefit must lie in [-1, 1]")
        action_hashes.append(require_sha256(cell.get("action_sha256"), field=f"cells.{index}.action_sha256"))
        identities.append((city, str(checkpoint)))
    cities = sorted({city for city, _ in identities})
    expected_grid = {(city, checkpoint) for city in cities for checkpoint in CHECKPOINT_IDS}
    if (
        len(cities) != TARGET_CITY_COUNT
        or identities != sorted(identities, key=lambda item: (item[0], int(item[1])))
        or len(set(identities)) != TARGET_CELL_COUNT
        or set(identities) != expected_grid
        or len(set(action_hashes)) != TARGET_CELL_COUNT
    ):
        raise IntegrityError("v2 scored-target bundle is not the exact sorted 10-by-5 action grid")
    return cells


def _exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    exceedances = 0
    tolerance = 1.0e-15
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(float(np.mean(differences * np.asarray(signs, dtype=np.float64))))
        if statistic + tolerance >= observed:
            exceedances += 1
    return exceedances / float(2 ** len(differences))


def _nominal_cluster_bootstrap_interval(differences: np.ndarray) -> list[float]:
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = generator.integers(
        0,
        len(differences),
        size=(BOOTSTRAP_RESAMPLES, len(differences)),
    )
    replicates = differences[indices].mean(axis=1)
    interval = np.quantile(replicates, [0.025, 0.975], method="linear")
    return [float(interval[0]), float(interval[1])]


def _holm_adjust(raw_pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw_pvalues, key=lambda name: (raw_pvalues[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * raw_pvalues[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def summarize_target_inference_v2(
    scored_target_bundle: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute fixed city-cluster inference from an already scored 50-cell bundle.

    This pure function opens no dataset and accepts no path.  The nominal
    bootstrap resamples ten city-level averages; the exact sign-flip test
    enumerates the 2^10 joint city signs under the protocol's symmetry null.
    """

    _validate_target_authorization_document_v2(authorization)
    if scored_target_bundle.get("authorization_sha256") != authorization.get("authorization_sha256"):
        raise IntegrityError("scored target bundle is not bound to the supplied authorization")
    cells = _validate_scored_target_bundle_v2(scored_target_bundle)
    controller = load_controller_v2()
    by_city: dict[str, list[Mapping[str, Any]]] = {}
    for cell in cells:
        by_city.setdefault(str(cell["city_id"]), []).append(cell)

    city_rows: list[dict[str, Any]] = []
    for city in sorted(by_city):
        city_cells = by_city[city]
        benefits = np.asarray(
            [float(cell["observed_benefit"]) for cell in city_cells],
            dtype=np.float64,
        )
        kbound_utilities = np.asarray(
            [float(cell["observed_benefit"]) if cell["realized_action"] == "ADAPT" else 0.0 for cell in city_cells],
            dtype=np.float64,
        )
        checkpoint_utilities = np.asarray(
            [
                float(cell["observed_benefit"])
                if _checkpoint_only_prediction(controller, str(cell["checkpoint_id"])) > 0.0
                else 0.0
                for cell in city_cells
            ],
            dtype=np.float64,
        )
        city_rows.append(
            {
                "city_id": city,
                "kbound_v2_mean_utility": float(np.mean(kbound_utilities)),
                "always_adapt_mean_utility": float(np.mean(benefits)),
                "always_freeze_mean_utility": 0.0,
                "checkpoint_only_mean_utility": float(np.mean(checkpoint_utilities)),
                "kbound_v2_vs_always_adapt": float(np.mean(kbound_utilities - benefits)),
                "kbound_v2_vs_always_freeze": float(np.mean(kbound_utilities)),
            }
        )

    names = (
        "kbound_v2_vs_always_adapt",
        "kbound_v2_vs_always_freeze",
    )
    arrays = {name: np.asarray([row[name] for row in city_rows], dtype=np.float64) for name in names}
    raw_pvalues = {name: _exact_sign_flip_pvalue(arrays[name]) for name in names}
    adjusted = _holm_adjust(raw_pvalues)
    comparisons = {
        name: {
            "mean_city_cluster_difference": float(np.mean(arrays[name])),
            "nominal_95_percent_city_cluster_bootstrap_interval": (_nominal_cluster_bootstrap_interval(arrays[name])),
            "exact_city_cluster_sign_flip_pvalue": raw_pvalues[name],
            "holm_adjusted_pvalue": adjusted[name],
            "holm_reject_at_0_05": adjusted[name] <= 0.05,
        }
        for name in names
    }
    decisions = [str(cell["decision"]) for cell in cells]
    report: dict[str, Any] = {
        "schema": TARGET_INFERENCE_SCHEMA,
        "status": "COMPLETE_CITY_CLUSTER_INFERENCE",
        "protocol_id": PROTOCOL_ID,
        "controller_sha256": scored_target_bundle["controller_sha256"],
        "authorization_sha256": scored_target_bundle["authorization_sha256"],
        "scored_target_bundle_sha256": scored_target_bundle["bundle_sha256"],
        "action_unit": "city_checkpoint",
        "cluster_unit": "target_city",
        "cluster_count": TARGET_CITY_COUNT,
        "checkpoints_per_cluster": len(CHECKPOINT_IDS),
        "city_cluster_rows": city_rows,
        "exact_sign_flip_assumption": (
            "independent_target_city_difference_vectors_with_joint_sign_symmetry_under_each_null"
        ),
        "exact_sign_flip_pattern_count": 2**TARGET_CITY_COUNT,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "bit_generator": "PCG64",
            "interval": "percentile",
            "quantiles": [0.025, 0.975],
            "numpy_quantile_method": "linear",
        },
        "comparisons": comparisons,
        "multiplicity": "holm_familywise_0.05",
        "decision_counts": {decision: decisions.count(decision) for decision in ("ADAPT", "FREEZE", "ABSTAIN")},
        "checkpoint_only_baseline": {
            "role": "required_secondary_diagnostic",
            "mean_city_cluster_utility": float(np.mean([row["checkpoint_only_mean_utility"] for row in city_rows])),
            "selection_use": "none",
        },
        "confidence_interval_scope": "nominal_not_simultaneous_or_population_certificate",
    }
    report["inference_sha256"] = stable_sha256(report)
    return report
