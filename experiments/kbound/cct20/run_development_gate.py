#!/usr/bin/env python3
"""Generate sealed CCT-20 development traces and fit the frozen ridge gate.

Only the protocol's two FIT cameras and nine CAL cameras are accepted.  Each
checkpoint-camera stream starts from a fresh source checkpoint, processes the
hash-selected probe stream before its evaluation stream, and uses evaluation
set-membership accuracy only for the development adaptation-benefit response.
No ``trans_test`` input is accepted by this command.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .integrity import IntegrityError, require_sha256, stable_sha256
from .label_free_traces import (
    TARGET_BATCH_SIZE,
    extract_label_free_features,
    sequence_atomic_batches,
    sequence_atomic_partition,
)
from .ridge_gate import fit_calibrate_ridge_gate, validate_gate_document
from .runner_runtime import (
    VerifiedImageStore,
    build_shared_runtime_identity,
    clear_device_cache,
    configure_deterministic_inference,
    expected_backend_strategy,
    load_checkpoint_model_pair,
    load_json_object,
    load_sealed_json_object,
    paired_forward,
    select_inference_device,
    shared_runtime_dependency_paths,
    validate_shared_runtime_identity,
    validate_shared_runtime_identity_artifact,
    validate_source_training_seal_identity,
    verify_checkpoint_audit_document,
    write_or_verify_immutable_json,
)
from .tent_official import (
    BN_GAUSSIAN_KL_NUMERIC_CLIPPING,
    BN_GAUSSIAN_KL_NUMERIC_IMPLEMENTATION,
    BN_GAUSSIAN_KL_SCHEMA,
    BN_GAUSSIAN_KL_TAYLOR_TERMS,
    BN_GAUSSIAN_KL_TAYLOR_THRESHOLD,
    OFFICIAL_TENT_COMMIT,
    OFFICIAL_TENT_FILE_SHA256,
    OFFICIAL_TENT_TREE,
    FrozenBatchNormMomentAccumulator,
    install_locked_root_bn_cpu_fallback,
    new_checkpoint_location_session,
    verify_official_tent,
)
from .train_source import LabeledSplit, load_labeled_split

PROBE_FRACTION = 0.30
PARTITION_SALT = "KBOUND_CCT20_PROBE_EVAL_v1"
BATCH_SIZE = TARGET_BATCH_SIZE
FIT_SPECS = (
    ("trans_val", "125", "development_fit"),
    ("cis_test", "33", "development_fit"),
)
CAL_SPECS = tuple(
    ("cis_test", str(location), "development_calibration") for location in (38, 43, 51, 61, 88, 90, 108, 115, 120)
)
ALL_SPECS = FIT_SPECS + CAL_SPECS


def _id_key(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise IntegrityError(f"development image id is invalid: {value!r}")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _development_rows(
    split: LabeledSplit,
    *,
    annotation_document: Mapping[str, Any],
    location_id: str,
) -> tuple[list[dict[str, Any]], dict[str, frozenset[int]], list[dict[str, Any]]]:
    raw_images = annotation_document.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise IntegrityError(f"{split.role} development document lacks images")
    raw_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(raw_images):
        if not isinstance(row, Mapping) or "id" not in row:
            raise IntegrityError(f"{split.role} images[{index}] lacks id")
        key = _id_key(row["id"])
        if key in raw_by_id:
            raise IntegrityError(f"duplicate {split.role} image id")
        raw_by_id[key] = row
    if set(raw_by_id) != {_id_key(sample.image_id) for sample in split.samples}:
        raise IntegrityError(f"{split.role} metadata and verified sample populations differ")

    rows: list[dict[str, Any]] = []
    truth: dict[str, frozenset[int]] = {}
    expected_samples: list[dict[str, Any]] = []
    for sample in split.samples:
        if str(sample.location) != location_id:
            continue
        raw = raw_by_id[_id_key(sample.image_id)]
        date_captured = raw.get("date_captured", raw.get("datetime", ""))
        frame_num = raw.get("frame_num", 0)
        if not isinstance(date_captured, str) or not date_captured:
            raise IntegrityError(f"native development stream requires date_captured for {sample.image_id!r}")
        if isinstance(frame_num, bool) or not isinstance(frame_num, int) or frame_num < 0:
            raise IntegrityError(f"invalid frame_num for development image {sample.image_id!r}")
        image_id = str(sample.image_id)
        if image_id in truth:
            raise IntegrityError("string-normalized development image ids are not unique")
        rows.append(
            {
                "image_id": image_id,
                "sequence_id": sample.seq_id,
                "location_id": location_id,
                "file_name": sample.file_name,
                "frame_num": frame_num,
                "date_captured": date_captured,
            }
        )
        truth[image_id] = frozenset(int(value) for value in sample.labels)
        expected_samples.append(
            {
                "image_id": image_id,
                "file_name": sample.file_name,
                "image_bytes": sample.image_bytes,
                "image_sha256": sample.image_sha256,
            }
        )
    if not rows:
        raise IntegrityError(f"sealed development location {split.role}:{location_id} is empty")
    return rows, truth, expected_samples


def _trace_document(
    *,
    split_name: str,
    role: str,
    location_id: str,
    checkpoint_row: Mapping[str, Any],
    partition: Mapping[str, Any],
    binding_receipt: Mapping[str, Any],
    feature_record: Mapping[str, Any],
    probe_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    shared_runtime_sha256: str,
) -> dict[str, Any]:
    frozen_correct = sum(bool(row["frozen_correct"]) for row in evaluation_rows)
    adapted_correct = sum(bool(row["adapted_correct"]) for row in evaluation_rows)
    n_evaluation = len(evaluation_rows)
    if n_evaluation < 1:
        raise IntegrityError("development trace has no evaluation images")
    frozen_accuracy = frozen_correct / n_evaluation
    adapted_accuracy = adapted_correct / n_evaluation
    calibration_unit = f"{split_name}:{location_id}"
    seed = int(checkpoint_row["model_seed"])
    core = {
        "schema": "kbound_cct20_development_trace_v1",
        "status": "SEALED_DEVELOPMENT_ONLY",
        "role": role,
        "trace_id": f"{calibration_unit}:checkpoint-{seed}",
        "calibration_unit": calibration_unit,
        "checkpoint_id": str(seed),
        "checkpoint_tensor_sha256": checkpoint_row["tensor_sha256"],
        "checkpoint_file_sha256": checkpoint_row["file_sha256"],
        "shared_runtime_sha256": require_sha256(shared_runtime_sha256, field="shared_runtime_sha256"),
        "partition_sha256": stable_sha256(dict(partition)),
        "partition": dict(partition),
        "official_tent_binding": dict(binding_receipt),
        "features": dict(feature_record["features"]),
        "probe_feature_record": dict(feature_record),
        "observed_benefit": adapted_accuracy - frozen_accuracy,
        "benefit_sign": "adapted_set_membership_top1_minus_frozen_set_membership_top1",
        "evaluation_metrics": {
            "n": n_evaluation,
            "frozen_correct": frozen_correct,
            "adapted_correct": adapted_correct,
            "frozen_set_membership_top1": frozen_accuracy,
            "adapted_set_membership_top1": adapted_accuracy,
        },
        "probe_rows": [dict(row) for row in probe_rows],
        "evaluation_rows": [dict(row) for row in evaluation_rows],
    }
    core["trace_sha256"] = stable_sha256(core)
    validate_development_trace(core)
    return core


def validate_development_trace(document: Mapping[str, Any]) -> None:
    if document.get("schema") != "kbound_cct20_development_trace_v1":
        raise IntegrityError("unknown CCT-20 development trace schema")
    unsigned = dict(document)
    claimed = unsigned.pop("trace_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("development trace SHA-256 mismatch")
    unit = str(document.get("calibration_unit", ""))
    role = document.get("role")
    expected_role_by_unit = {
        **{f"{split}:{location}": role_name for split, location, role_name in FIT_SPECS},
        **{f"{split}:{location}": role_name for split, location, role_name in CAL_SPECS},
    }
    if expected_role_by_unit.get(unit) != role:
        raise IntegrityError("development trace has an unsealed unit or FIT/CAL role")
    checkpoint_id = str(document.get("checkpoint_id", ""))
    if checkpoint_id not in {str(value) for value in range(5)}:
        raise IntegrityError("development trace checkpoint identity must be seed 0..4")
    if document.get("trace_id") != f"{unit}:checkpoint-{checkpoint_id}":
        raise IntegrityError("development trace_id does not match its unit/checkpoint")
    checkpoint_hash = require_sha256(document.get("checkpoint_tensor_sha256"), field="checkpoint_tensor_sha256")
    require_sha256(document.get("checkpoint_file_sha256"), field="checkpoint_file_sha256")
    require_sha256(document.get("shared_runtime_sha256"), field="shared_runtime_sha256")
    binding = document.get("official_tent_binding", {})
    location_id = unit.split(":", maxsplit=1)[1]
    parameter_names = binding.get("parameter_names", ()) if isinstance(binding, Mapping) else ()
    provenance = binding.get("provenance", {}) if isinstance(binding, Mapping) else {}
    optimizer = provenance.get("optimizer", {}) if isinstance(provenance, Mapping) else {}
    if (
        not isinstance(binding, Mapping)
        or binding.get("schema") != "kbound_cct20_official_tent_binding_v1"
        or binding.get("checkpoint_tensor_sha256") != checkpoint_hash
        or str(binding.get("location_id")) != location_id
        or binding.get("reset_scope") != f"{checkpoint_hash}:{location_id}"
        or not isinstance(parameter_names, list)
        or not parameter_names
        or len(set(parameter_names)) != len(parameter_names)
        or binding.get("n_parameters") != len(parameter_names)
        or binding.get("update_norm_formula") != "l2(after_probe-before_probe)/max(l2(before_probe),1e-12)"
    ):
        raise IntegrityError("development trace official-Tent binding identity mismatch")
    backend = binding.get("backend_installation")
    expected_backend_fields = {
        "schema",
        "strategy",
        "fallback_layer",
        "source_module_class",
        "fallback_module_class",
        "fallback_input_device",
        "fallback_compute_device",
        "fallback_parameter_device",
        "fallback_output_device",
        "num_features",
        "eps",
        "momentum",
        "affine",
        "preconfigure_track_running_stats",
        "source_bn_state_sha256",
        "installed_bn_state_sha256",
        "state_hash_equal",
        "official_tent_parameter_devices",
        "configured_track_running_stats",
        "configured_running_moments_absent",
    }
    backend_strategy = expected_backend_strategy()
    if (
        not isinstance(backend, Mapping)
        or set(backend) != expected_backend_fields
        or backend.get("schema") != "kbound_cct20_backend_installation_v1"
        or backend.get("strategy") != backend_strategy["strategy"]
        or backend.get("fallback_layer") != "bn1"
        or backend.get("source_module_class") != "torch.nn.BatchNorm2d"
        or backend.get("fallback_module_class") != backend_strategy["fallback_module_class"]
        or backend.get("fallback_input_device") != backend_strategy["fallback_input_device"]
        or backend.get("fallback_compute_device") != backend_strategy["fallback_compute_device"]
        or backend.get("fallback_parameter_device") != "cpu"
        or backend.get("fallback_output_device") != backend_strategy["fallback_output_device"]
        or backend.get("num_features") != 64
        or backend.get("eps") != 1.0e-5
        or backend.get("momentum") != 0.1
        or backend.get("affine") is not True
        or backend.get("preconfigure_track_running_stats") is not True
        or backend.get("state_hash_equal") is not True
        or backend.get("official_tent_parameter_devices") != ["cpu", "mps"]
        or backend.get("configured_track_running_stats") is not False
        or backend.get("configured_running_moments_absent") is not True
    ):
        raise IntegrityError("development trace hybrid-backend receipt identity mismatch")
    source_bn_hash = require_sha256(backend.get("source_bn_state_sha256"), field="source_bn_state_sha256")
    installed_bn_hash = require_sha256(backend.get("installed_bn_state_sha256"), field="installed_bn_state_sha256")
    if source_bn_hash != installed_bn_hash:
        raise IntegrityError("development trace hybrid-backend BN state changed")
    try:
        initial_norm = float(binding.get("initial_bn_affine_l2"))
    except (TypeError, ValueError) as exc:
        raise IntegrityError("development trace Tent initial norm is invalid") from exc
    if not math.isfinite(initial_norm) or initial_norm <= 0.0:
        raise IntegrityError("development trace Tent initial norm is invalid")
    if not (
        isinstance(provenance, Mapping)
        and provenance.get("git_commit") == OFFICIAL_TENT_COMMIT
        and provenance.get("git_tree") == OFFICIAL_TENT_TREE
        and provenance.get("tent_py_sha256") == OFFICIAL_TENT_FILE_SHA256
        and provenance.get("tracked_worktree_clean") is True
        and provenance.get("configure_function") == "tent.configure_model"
        and provenance.get("parameter_function") == "tent.collect_params"
        and provenance.get("adapter_class") == "tent.Tent"
        and provenance.get("reset_scope") == "source_checkpoint_x_camera_location"
        and isinstance(optimizer, Mapping)
        and optimizer.get("class") == "torch.optim.Adam"
        and optimizer.get("lr") == 0.001
        and optimizer.get("betas") == [0.9, 0.999]
        and optimizer.get("weight_decay") == 0.0
        and provenance.get("steps") == 1
        and provenance.get("episodic") is False
    ):
        raise IntegrityError("development trace does not bind the pinned official Tent settings")
    if document.get("benefit_sign") != ("adapted_set_membership_top1_minus_frozen_set_membership_top1"):
        raise IntegrityError("development trace benefit sign convention drift")
    probe_rows = list(document.get("probe_rows", ()))
    evaluation_rows = list(document.get("evaluation_rows", ()))
    if not probe_rows or not evaluation_rows:
        raise IntegrityError("development trace requires non-empty probe and evaluation rows")
    partition = document.get("partition")
    if not isinstance(partition, Mapping) or document.get("partition_sha256") != stable_sha256(dict(partition)):
        raise IntegrityError("development trace partition identity mismatch")
    partition_roles = partition.get("roles", {})
    if not isinstance(partition_roles, Mapping) or set(partition_roles) != {
        "probe",
        "evaluation",
    }:
        raise IntegrityError("development trace partition lacks exact probe/evaluation roles")
    role_ids = {
        role_name: [str(row.get("image_id", "")) for row in partition_roles[role_name]]
        for role_name in ("probe", "evaluation")
    }
    if (
        not role_ids["probe"]
        or not role_ids["evaluation"]
        or len(set(role_ids["probe"])) != len(role_ids["probe"])
        or len(set(role_ids["evaluation"])) != len(role_ids["evaluation"])
        or set(role_ids["probe"]) & set(role_ids["evaluation"])
        or role_ids["probe"] != [str(row.get("image_id", "")) for row in probe_rows]
        or role_ids["evaluation"] != [str(row.get("image_id", "")) for row in evaluation_rows]
    ):
        raise IntegrityError("development trace partition/row coverage is not exact and disjoint")
    sequence_roles: dict[str, str] = {}
    for role_name in ("probe", "evaluation"):
        for row in partition_roles[role_name]:
            sequence = str(row.get("sequence_id", ""))
            if not sequence:
                raise IntegrityError("development partition has an empty sequence id")
            prior = sequence_roles.setdefault(sequence, role_name)
            if prior != role_name:
                raise IntegrityError("development sequence crosses probe/evaluation roles")
    feature_record = document.get("probe_feature_record", {})
    features = feature_record.get("features", {})
    diagnostics = feature_record.get("diagnostic_receipts", {})
    update_receipt = diagnostics.get("tent_update", {})
    moment_receipt = diagnostics.get("frozen_bn_probe_moments", {})
    if not (
        isinstance(diagnostics, Mapping)
        and isinstance(update_receipt, Mapping)
        and update_receipt.get("schema") == "kbound_cct20_tent_probe_update_v1"
        and update_receipt.get("checkpoint_tensor_sha256") == checkpoint_hash
        and str(update_receipt.get("location_id")) == location_id
        and update_receipt.get("reset_scope") == f"{checkpoint_hash}:{location_id}"
        and update_receipt.get("parameter_names") == parameter_names
        and update_receipt.get("formula") == "l2(after_probe-before_probe)/max(l2(before_probe),1e-12)"
    ):
        raise IntegrityError("development trace Tent update receipt is invalid")
    try:
        update_norm = float(update_receipt.get("normalized_tent_update_norm"))
    except (TypeError, ValueError) as exc:
        raise IntegrityError("development trace Tent update norm is invalid") from exc
    if (
        not math.isfinite(update_norm)
        or update_norm < 0.0
        or not np.isclose(
            update_norm,
            float(features.get("normalized_tent_update_norm", np.nan)),
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise IntegrityError("development trace Tent update norm does not reconcile")
    if not (
        isinstance(moment_receipt, Mapping)
        and moment_receipt.get("schema") == BN_GAUSSIAN_KL_SCHEMA
        and moment_receipt.get("formula") == "channel_weighted_mean_gaussian_kl_probe_to_source"
        and moment_receipt.get("numeric_implementation") == BN_GAUSSIAN_KL_NUMERIC_IMPLEMENTATION
        and moment_receipt.get("taylor_threshold") == BN_GAUSSIAN_KL_TAYLOR_THRESHOLD
        and moment_receipt.get("taylor_terms") == BN_GAUSSIAN_KL_TAYLOR_TERMS
        and moment_receipt.get("numeric_clipping") == BN_GAUSSIAN_KL_NUMERIC_CLIPPING
    ):
        raise IntegrityError("development trace frozen-BN moment receipt is invalid")
    layers = moment_receipt.get("layers")
    if not isinstance(layers, list) or not layers:
        raise IntegrityError("development trace frozen-BN receipt has no layers")
    layer_names: set[str] = set()
    channel_count = 0
    weighted_kl = 0.0
    taylor_branch_channels = 0
    minimum_channel_kl = math.inf
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise IntegrityError("development trace frozen-BN layer is invalid")
        name = layer.get("layer")
        channels = layer.get("channels")
        values_per_channel = layer.get("values_per_channel")
        layer_taylor_channels = layer.get("taylor_branch_channels")
        try:
            mean_kl = float(layer.get("mean_kl"))
            min_kl = float(layer.get("min_kl"))
            eps = float(layer.get("bn_eps"))
        except (TypeError, ValueError) as exc:
            raise IntegrityError("development trace frozen-BN layer is invalid") from exc
        if (
            not isinstance(name, str)
            or not name
            or name in layer_names
            or isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels < 1
            or isinstance(values_per_channel, bool)
            or not isinstance(values_per_channel, int)
            or values_per_channel < 1
            or isinstance(layer_taylor_channels, bool)
            or not isinstance(layer_taylor_channels, int)
            or layer_taylor_channels < 0
            or layer_taylor_channels > channels
            or not math.isfinite(mean_kl)
            or mean_kl < 0.0
            or not math.isfinite(min_kl)
            or min_kl < 0.0
            or min_kl > mean_kl
            or not math.isfinite(eps)
            or eps <= 0.0
        ):
            raise IntegrityError("development trace frozen-BN layer is invalid")
        layer_names.add(name)
        channel_count += channels
        weighted_kl += channels * mean_kl
        taylor_branch_channels += layer_taylor_channels
        minimum_channel_kl = min(minimum_channel_kl, min_kl)
    expected_layer_names: dict[str, set[str]] = {}
    for parameter_name in parameter_names:
        layer_name, separator, suffix = parameter_name.rpartition(".")
        if not separator or suffix not in {"weight", "bias"}:
            raise IntegrityError("development trace has an invalid BN-affine parameter name")
        expected_layer_names.setdefault(layer_name, set()).add(suffix)
    try:
        declared_channels = int(moment_receipt.get("channel_count"))
        declared_divergence = float(moment_receipt.get("batchnorm_batch_source_statistic_divergence"))
        declared_taylor_channels = int(moment_receipt.get("taylor_branch_channels"))
        declared_minimum_kl = float(moment_receipt.get("minimum_channel_kl"))
    except (TypeError, ValueError) as exc:
        raise IntegrityError("development trace frozen-BN aggregate is invalid") from exc
    if (
        any(suffixes != {"weight", "bias"} for suffixes in expected_layer_names.values())
        or layer_names != set(expected_layer_names)
        or channel_count != declared_channels
        or taylor_branch_channels != declared_taylor_channels
        or minimum_channel_kl != declared_minimum_kl
        or not math.isfinite(declared_divergence)
        or declared_divergence < 0.0
        or not np.isclose(
            weighted_kl / channel_count,
            declared_divergence,
            rtol=1e-12,
            atol=1e-12,
        )
        or not np.isclose(
            declared_divergence,
            float(features.get("batchnorm_batch_source_statistic_divergence", np.nan)),
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise IntegrityError("development trace frozen-BN aggregate does not reconcile")
    replay = extract_label_free_features(
        [row.get("frozen_logits") for row in probe_rows],
        [row.get("adapted_logits") for row in probe_rows],
        normalized_tent_update_norm=features.get("normalized_tent_update_norm"),
        batchnorm_batch_source_statistic_divergence=features.get("batchnorm_batch_source_statistic_divergence"),
    )
    if feature_record.get("n_probe_images") != len(probe_rows):
        raise IntegrityError("development trace probe count mismatch")
    for name, expected in replay["features"].items():
        observed = features.get(name)
        if not isinstance(observed, (int, float)) or not np.isclose(float(observed), expected, rtol=1e-12, atol=1e-12):
            raise IntegrityError(f"development feature {name!r} does not replay")
    frozen_correct = 0
    adapted_correct = 0
    for row in evaluation_rows:
        try:
            frozen_logits = np.asarray(row.get("frozen_logits"), dtype=np.float64)
            adapted_logits = np.asarray(row.get("adapted_logits"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise IntegrityError("development evaluation logits are non-numeric") from exc
        if (
            frozen_logits.shape != (16,)
            or adapted_logits.shape != (16,)
            or not np.isfinite(frozen_logits).all()
            or not np.isfinite(adapted_logits).all()
        ):
            raise IntegrityError("development evaluation logits must be finite 16-vectors")
        frozen_prediction = int(np.argmax(frozen_logits))
        adapted_prediction = int(np.argmax(adapted_logits))
        if row.get("frozen_prediction") != frozen_prediction or row.get("adapted_prediction") != adapted_prediction:
            raise IntegrityError("development evaluation prediction does not replay from logits")
        raw_truth = row.get("ground_truth_output_indices")
        if (
            not isinstance(raw_truth, list)
            or not raw_truth
            or len(set(raw_truth)) != len(raw_truth)
            or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 16 for value in raw_truth)
        ):
            raise IntegrityError("development evaluation truth set is invalid")
        frozen_is_correct = frozen_prediction in raw_truth
        adapted_is_correct = adapted_prediction in raw_truth
        if row.get("frozen_correct") is not frozen_is_correct or row.get("adapted_correct") is not adapted_is_correct:
            raise IntegrityError("development correctness does not replay from logits/truth")
        frozen_correct += int(frozen_is_correct)
        adapted_correct += int(adapted_is_correct)
    metrics = document.get("evaluation_metrics", {})
    n = len(evaluation_rows)
    expected_benefit = (adapted_correct - frozen_correct) / n
    if (
        metrics.get("n") != n
        or metrics.get("frozen_correct") != frozen_correct
        or metrics.get("adapted_correct") != adapted_correct
        or not np.isclose(
            float(metrics.get("frozen_set_membership_top1", np.nan)),
            frozen_correct / n,
            rtol=0.0,
            atol=1e-15,
        )
        or not np.isclose(
            float(metrics.get("adapted_set_membership_top1", np.nan)),
            adapted_correct / n,
            rtol=0.0,
            atol=1e-15,
        )
        or not np.isclose(
            float(document.get("observed_benefit", np.nan)),
            expected_benefit,
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise IntegrityError("development adaptation benefit does not replay")


def run_development_cell(
    *,
    split_name: str,
    role: str,
    location_id: str,
    metadata_rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, frozenset[int]],
    image_store: VerifiedImageStore,
    checkpoint_row: Mapping[str, Any],
    tent_repo: Path,
    device: Any,
    shared_runtime_sha256: str,
) -> dict[str, Any]:
    partition = sequence_atomic_partition(
        metadata_rows,
        probe_fraction=PROBE_FRACTION,
        salt=PARTITION_SALT,
    )
    plans = {
        role_name: sequence_atomic_batches(
            partition["roles"][role_name],
            max_images=BATCH_SIZE,
            order="native",
            merge_singleton_final=True,
        )
        for role_name in ("probe", "evaluation")
    }
    if any(len(batch) == 1 for batches in plans.values() for batch in batches):
        raise IntegrityError("development stream produced an unmergeable singleton batch")
    frozen, adapted_source = load_checkpoint_model_pair(checkpoint_row, device=device)
    backend_installation = install_locked_root_bn_cpu_fallback(adapted_source)
    binding = new_checkpoint_location_session(
        adapted_source,
        repo_root=tent_repo,
        checkpoint_tensor_sha256=checkpoint_row["tensor_sha256"],
        location_id=location_id,
        backend_installation_receipt=backend_installation,
    )
    accumulator = FrozenBatchNormMomentAccumulator(frozen)
    probe_rows = []
    for batch in plans["probe"]:
        images = image_store.tensor_batch(batch)
        frozen_logits, adapted_logits = paired_forward(frozen, binding.adapter, images, device=device)
        for metadata, frozen_row, adapted_row in zip(batch, frozen_logits, adapted_logits, strict=True):
            probe_rows.append(
                {
                    "image_id": metadata["image_id"],
                    "sequence_id": metadata["sequence_id"],
                    "frozen_logits": [float(value) for value in frozen_row],
                    "adapted_logits": [float(value) for value in adapted_row],
                }
            )
    moment_receipt = accumulator.finalize()
    update_receipt = binding.probe_update_receipt()
    feature_record = extract_label_free_features(
        [row["frozen_logits"] for row in probe_rows],
        [row["adapted_logits"] for row in probe_rows],
        normalized_tent_update_norm=update_receipt["normalized_tent_update_norm"],
        batchnorm_batch_source_statistic_divergence=moment_receipt["batchnorm_batch_source_statistic_divergence"],
    )
    feature_record["diagnostic_receipts"] = {
        "tent_update": update_receipt,
        "frozen_bn_probe_moments": moment_receipt,
    }

    evaluation_rows = []
    for batch in plans["evaluation"]:
        images = image_store.tensor_batch(batch)
        frozen_logits, adapted_logits = paired_forward(frozen, binding.adapter, images, device=device)
        for metadata, frozen_row, adapted_row in zip(batch, frozen_logits, adapted_logits, strict=True):
            image_id = str(metadata["image_id"])
            values = truth.get(image_id)
            if not values:
                raise IntegrityError(f"development image {image_id!r} lacks complete truth")
            frozen_prediction = int(np.argmax(frozen_row))
            adapted_prediction = int(np.argmax(adapted_row))
            evaluation_rows.append(
                {
                    "image_id": image_id,
                    "sequence_id": metadata["sequence_id"],
                    "ground_truth_output_indices": sorted(values),
                    "frozen_prediction": frozen_prediction,
                    "adapted_prediction": adapted_prediction,
                    "frozen_logits": [float(value) for value in frozen_row],
                    "adapted_logits": [float(value) for value in adapted_row],
                    "frozen_correct": frozen_prediction in values,
                    "adapted_correct": adapted_prediction in values,
                }
            )
    result = _trace_document(
        split_name=split_name,
        role=role,
        location_id=location_id,
        checkpoint_row=checkpoint_row,
        partition=partition,
        binding_receipt=binding.receipt(),
        feature_record=feature_record,
        probe_rows=probe_rows,
        evaluation_rows=evaluation_rows,
        shared_runtime_sha256=shared_runtime_sha256,
    )
    del binding, adapted_source, frozen
    clear_device_cache(device)
    return result


def _gate_row(trace: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": trace["role"],
        "trace_id": trace["trace_id"],
        "calibration_unit": trace["calibration_unit"],
        "checkpoint_id": trace["checkpoint_id"],
        "checkpoint_tensor_sha256": trace["checkpoint_tensor_sha256"],
        "checkpoint_file_sha256": trace["checkpoint_file_sha256"],
        "shared_runtime_sha256": trace["shared_runtime_sha256"],
        "trace_sha256": trace["trace_sha256"],
        "partition_sha256": trace["partition_sha256"],
        "features": dict(trace["features"]),
        "observed_benefit": trace["observed_benefit"],
    }


def validate_resumed_development_trace(
    trace: Mapping[str, Any],
    *,
    split_name: str,
    location_id: str,
    checkpoint_row: Mapping[str, Any],
    shared_runtime_sha256: str,
) -> None:
    """Reject any resume whose scientific or runtime identity has changed."""

    validate_development_trace(trace)
    if (
        trace.get("calibration_unit") != f"{split_name}:{location_id}"
        or trace.get("checkpoint_id") != str(checkpoint_row["model_seed"])
        or trace.get("checkpoint_tensor_sha256") != checkpoint_row["tensor_sha256"]
        or trace.get("checkpoint_file_sha256") != checkpoint_row["file_sha256"]
        or trace.get("shared_runtime_sha256") != shared_runtime_sha256
    ):
        raise IntegrityError("existing development trace identity differs from this run")


def validate_development_trace_collection(
    document: Mapping[str, Any],
    *,
    gate_document: Mapping[str, Any],
    checkpoint_audit: Mapping[str, Any],
    verify_trace_files: bool = True,
    verify_runtime_file: bool = True,
) -> None:
    """Replay the gate and its 55 immutable trace/receipt dependencies."""

    if (
        document.get("schema") != "kbound_cct20_development_trace_collection_v1"
        or document.get("status") != "SEALED_BEFORE_TARGET_INFERENCE"
    ):
        raise IntegrityError("unknown or unsealed CCT-20 development trace collection")
    unsigned = dict(document)
    claimed = unsigned.pop("collection_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("development trace collection SHA-256 mismatch")
    validate_gate_document(gate_document)
    if document.get("gate_sha256") != gate_document.get("gate_sha256"):
        raise IntegrityError("development trace collection gate identity mismatch")
    if document.get("checkpoint_audit_sha256") != stable_sha256(checkpoint_audit):
        raise IntegrityError("development trace collection checkpoint-audit mismatch")
    runtime_binding = document.get("shared_runtime_identity")
    if not isinstance(runtime_binding, Mapping) or set(runtime_binding) != {
        "shared_runtime_sha256",
        "artifact_path",
        "artifact_receipt",
    }:
        raise IntegrityError("development trace collection lacks the shared runtime binding")
    runtime_sha256 = require_sha256(runtime_binding.get("shared_runtime_sha256"), field="shared_runtime_sha256")
    runtime_artifact_path = str(runtime_binding.get("artifact_path", ""))
    runtime_receipt = runtime_binding.get("artifact_receipt")
    if (
        not runtime_artifact_path
        or Path(runtime_artifact_path).expanduser().resolve() != Path(runtime_artifact_path)
        or not isinstance(runtime_receipt, Mapping)
        or runtime_receipt.get("schema") != "kbound_cct20_artifact_receipt_v1"
        or runtime_receipt.get("artifact_path") != runtime_artifact_path
    ):
        raise IntegrityError("development trace shared-runtime artifact identity mismatch")
    require_sha256(
        runtime_receipt.get("artifact_sha256"),
        field="shared_runtime_identity.artifact_sha256",
    )
    require_sha256(
        runtime_receipt.get("canonical_document_sha256"),
        field="shared_runtime_identity.canonical_document_sha256",
    )
    if verify_runtime_file:
        runtime_document, observed_runtime_receipt = load_sealed_json_object(runtime_artifact_path)
        if dict(runtime_receipt) != observed_runtime_receipt:
            raise IntegrityError("development trace shared-runtime receipt mismatch")
        validate_shared_runtime_identity_artifact(runtime_document)
        if runtime_document.get("runtime_sha256") != runtime_sha256:
            raise IntegrityError("development trace shared-runtime SHA-256 mismatch")
    rows = document.get("gate_rows")
    artifacts = document.get("trace_artifacts")
    if not isinstance(rows, list) or not isinstance(artifacts, list):
        raise IntegrityError("development trace collection lacks rows/artifacts")
    expected_row_fields = {
        "role",
        "trace_id",
        "calibration_unit",
        "checkpoint_id",
        "checkpoint_tensor_sha256",
        "checkpoint_file_sha256",
        "shared_runtime_sha256",
        "trace_sha256",
        "partition_sha256",
        "features",
        "observed_benefit",
    }
    if len(rows) != 55 or any(not isinstance(row, Mapping) or set(row) != expected_row_fields for row in rows):
        raise IntegrityError("development trace collection must contain 55 exact gate rows")
    by_trace_id = {str(row["trace_id"]): dict(row) for row in rows}
    if len(by_trace_id) != 55:
        raise IntegrityError("development trace collection has duplicate trace IDs")
    if {str(row.get("shared_runtime_sha256", "")) for row in rows} != {runtime_sha256}:
        raise IntegrityError("development gate rows use a different shared runtime")
    fit_rows = [row for row in rows if row.get("role") == "development_fit"]
    calibration_rows = [row for row in rows if row.get("role") == "development_calibration"]
    if (
        len(fit_rows) != 10
        or len(calibration_rows) != 45
        or document.get("fit_trace_count") != 10
        or document.get("calibration_trace_count") != 45
    ):
        raise IntegrityError("development trace collection FIT/CAL counts drift")
    replayed_gate = fit_calibrate_ridge_gate(fit_rows, calibration_rows)
    if replayed_gate != dict(gate_document):
        raise IntegrityError("development trace collection does not replay the sealed gate")
    audit_rows = checkpoint_audit.get("checkpoints", ())
    expected_tensor_by_id = {str(row["model_seed"]): row["tensor_sha256"] for row in audit_rows}
    expected_file_by_id = {str(row["model_seed"]): row["file_sha256"] for row in audit_rows}
    provenance = gate_document.get("development_provenance", {})
    if (
        provenance.get("checkpoint_tensor_sha256_by_id") != expected_tensor_by_id
        or provenance.get("checkpoint_file_sha256_by_id") != expected_file_by_id
    ):
        raise IntegrityError("development gate checkpoint identities differ from the audit")
    if len(artifacts) != 55:
        raise IntegrityError("development trace collection must bind 55 trace artifacts")
    if not all(isinstance(record, Mapping) for record in artifacts):
        raise IntegrityError("development trace collection has a non-mapping artifact")
    trace_hashes = [str(record.get("trace_sha256", "")) for record in artifacts]
    if (
        len(set(trace_hashes)) != 55
        or document.get("trace_sha256") != sorted(trace_hashes)
        or set(trace_hashes) != {str(row["trace_sha256"]) for row in rows}
    ):
        raise IntegrityError("development trace artifact hashes do not match gate rows")
    seen_paths: set[str] = set()
    for record in artifacts:
        if not isinstance(record, Mapping) or set(record) != {
            "trace_id",
            "trace_sha256",
            "artifact_path",
            "artifact_receipt",
        }:
            raise IntegrityError("development trace artifact record schema drift")
        trace_id = str(record["trace_id"])
        artifact_path = str(record["artifact_path"])
        embedded_receipt = record["artifact_receipt"]
        if (
            trace_id not in by_trace_id
            or record["trace_sha256"] != by_trace_id[trace_id]["trace_sha256"]
            or not artifact_path
            or artifact_path in seen_paths
            or not isinstance(embedded_receipt, Mapping)
            or embedded_receipt.get("schema") != "kbound_cct20_artifact_receipt_v1"
            or embedded_receipt.get("artifact_path") != artifact_path
        ):
            raise IntegrityError("development trace artifact identity mismatch")
        require_sha256(
            embedded_receipt.get("artifact_sha256"),
            field="development_trace.artifact_sha256",
        )
        require_sha256(
            embedded_receipt.get("canonical_document_sha256"),
            field="development_trace.canonical_document_sha256",
        )
        seen_paths.add(artifact_path)
        if verify_trace_files:
            trace, receipt = load_sealed_json_object(artifact_path)
            if embedded_receipt != receipt:
                raise IntegrityError("development trace embedded receipt mismatch")
            validate_development_trace(trace)
            if (
                trace.get("trace_id") != trace_id
                or trace.get("trace_sha256") != record["trace_sha256"]
                or trace.get("shared_runtime_sha256") != runtime_sha256
                or _gate_row(trace) != by_trace_id[trace_id]
            ):
                raise IntegrityError("development trace file does not replay its gate row")


def _load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, LabeledSplit], dict[str, dict[str, Any]]]:
    audit = load_json_object(args.checkpoint_audit)
    verify_checkpoint_audit_document(audit)
    trans_document = load_json_object(args.trans_val_annotations)
    cis_document = load_json_object(args.cis_test_annotations)
    trans_split = load_labeled_split(
        args.trans_val_annotations,
        args.image_root,
        role="trans_val",
        expected_basename="trans_val_annotations.json",
    )
    cis_split = load_labeled_split(
        args.cis_test_annotations,
        args.image_root,
        role="cis_test",
        expected_basename="cis_test_annotations.json",
        reference_categories=trans_split.categories,
    )
    return (
        audit,
        {"trans_val": trans_split, "cis_test": cis_split},
        {"trans_val": trans_document, "cis_test": cis_document},
    )


def main() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--trans-val-annotations", type=Path, required=True)
    parser.add_argument("--cis-test-annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--tent-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-training-seal",
        type=Path,
        default=repository_root / "research_lock" / "KBOUND_CCT20_SOURCE_TRAINING_SEAL_v1.json",
    )
    parser.add_argument(
        "--runtime-addendum",
        type=Path,
        default=repository_root / "research_lock" / "KBOUND_CCT20_EXECUTION_RUNTIME_ADDENDUM_v2.yaml",
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    args.tent_repo = args.tent_repo.expanduser().resolve()
    args.source_training_seal = args.source_training_seal.expanduser().resolve()
    args.runtime_addendum = args.runtime_addendum.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    configure_deterministic_inference()
    device = select_inference_device(args.device)
    verify_official_tent(args.tent_repo)
    source_training_seal, source_training_receipt = load_sealed_json_object(args.source_training_seal)
    validate_source_training_seal_identity(source_training_seal, source_training_receipt)
    runtime_dependencies = shared_runtime_dependency_paths(
        repository_root,
        tent_repo=args.tent_repo,
        runtime_addendum=args.runtime_addendum,
    )
    shared_runtime = build_shared_runtime_identity(
        device,
        source_training_seal_artifact_sha256=source_training_receipt["artifact_sha256"],
        source_training_seal_document_sha256=source_training_receipt["canonical_document_sha256"],
        dependency_paths=runtime_dependencies,
    )
    output.mkdir(parents=True, exist_ok=True)
    shared_runtime_path = output / "shared_runtime_identity.json"
    shared_runtime_receipt = write_or_verify_immutable_json(shared_runtime_path, shared_runtime)
    validate_shared_runtime_identity(
        shared_runtime,
        device=device,
        dependency_paths=runtime_dependencies,
    )
    audit, splits, documents = _load_inputs(args)
    traces_dir = output / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_rows = [dict(row) for row in audit["checkpoints"]]
    traces = []
    trace_artifacts = []
    prepared: dict[tuple[str, str], tuple[list[dict[str, Any]], dict[str, frozenset[int]], VerifiedImageStore]] = {}
    for split_name, location_id, role in ALL_SPECS:
        key = (split_name, location_id)
        if key not in prepared:
            rows, truth, expected_samples = _development_rows(
                splits[split_name],
                annotation_document=documents[split_name],
                location_id=location_id,
            )
            prepared[key] = (
                rows,
                truth,
                VerifiedImageStore(args.image_root, expected_samples),
            )
        metadata_rows, truth, image_store = prepared[key]
        for checkpoint_row in checkpoint_rows:
            validate_shared_runtime_identity(
                shared_runtime,
                device=device,
                dependency_paths=runtime_dependencies,
            )
            seed = int(checkpoint_row["model_seed"])
            trace_path = traces_dir / f"{role}_{split_name}_{location_id}_seed{seed}.json"
            if trace_path.exists() or trace_path.with_name(trace_path.name + ".receipt.json").exists():
                trace, trace_receipt = load_sealed_json_object(trace_path)
                validate_resumed_development_trace(
                    trace,
                    split_name=split_name,
                    location_id=location_id,
                    checkpoint_row=checkpoint_row,
                    shared_runtime_sha256=shared_runtime["runtime_sha256"],
                )
                print(f"development trace verified/resumed: {trace_path}", flush=True)
            else:
                trace = run_development_cell(
                    split_name=split_name,
                    role=role,
                    location_id=location_id,
                    metadata_rows=metadata_rows,
                    truth=truth,
                    image_store=image_store,
                    checkpoint_row=checkpoint_row,
                    tent_repo=args.tent_repo,
                    device=device,
                    shared_runtime_sha256=shared_runtime["runtime_sha256"],
                )
                trace_receipt = write_or_verify_immutable_json(trace_path, trace)
                print(
                    f"development trace sealed: {trace['trace_id']} benefit={trace['observed_benefit']:.6f}",
                    flush=True,
                )
            traces.append(trace)
            trace_artifacts.append(
                {
                    "trace_id": trace["trace_id"],
                    "trace_sha256": trace["trace_sha256"],
                    "artifact_path": str(trace_path.resolve()),
                    "artifact_receipt": trace_receipt,
                }
            )

    validate_shared_runtime_identity(
        shared_runtime,
        device=device,
        dependency_paths=runtime_dependencies,
    )
    fit_rows = [_gate_row(trace) for trace in traces if trace["role"] == "development_fit"]
    calibration_rows = [_gate_row(trace) for trace in traces if trace["role"] == "development_calibration"]
    gate = fit_calibrate_ridge_gate(fit_rows, calibration_rows)
    validate_gate_document(gate)
    collection = {
        "schema": "kbound_cct20_development_trace_collection_v1",
        "status": "SEALED_BEFORE_TARGET_INFERENCE",
        "fit_trace_count": len(fit_rows),
        "calibration_trace_count": len(calibration_rows),
        "checkpoint_audit_sha256": stable_sha256(audit),
        "shared_runtime_identity": {
            "shared_runtime_sha256": shared_runtime["runtime_sha256"],
            "artifact_path": str(shared_runtime_path),
            "artifact_receipt": shared_runtime_receipt,
        },
        "trace_sha256": sorted(trace["trace_sha256"] for trace in traces),
        "trace_artifacts": sorted(
            trace_artifacts,
            key=lambda row: row["trace_id"],
        ),
        "gate_rows": sorted(
            fit_rows + calibration_rows,
            key=lambda row: row["trace_id"],
        ),
        "gate_sha256": gate["gate_sha256"],
    }
    collection["collection_sha256"] = stable_sha256(collection)
    validate_development_trace_collection(
        collection,
        gate_document=gate,
        checkpoint_audit=audit,
        verify_trace_files=True,
        verify_runtime_file=True,
    )
    collection_path = output / "development_trace_collection.json"
    gate_path = output / "ridge_gate.json"
    write_or_verify_immutable_json(collection_path, collection)
    write_or_verify_immutable_json(gate_path, gate)
    print(
        f"development gate sealed: epsilon={gate['calibration']['epsilon']:.8f} "
        f"gate_sha256={gate['gate_sha256']} -> {gate_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
