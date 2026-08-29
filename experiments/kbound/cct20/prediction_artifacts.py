"""Strictly label-free prediction/action artifacts for target inference.

This module intentionally has no import of the post-seal scorer.  Target code
may create and validate prediction artifacts without making target outcomes
reachable in its module graph.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .integrity import IntegrityError, require_sha256, stable_sha256, strict_json_load
from .label_free_traces import (
    PARTITION_SALT,
    PROBE_FRACTION,
    TARGET_BATCH_SIZE,
    assert_label_free,
    extract_label_free_features,
    normalize_metadata_row,
    sequence_atomic_batches,
    sequence_atomic_partition,
)
from .protocol_seal import (
    EXPECTED_CLASS_COUNT,
    EXPECTED_MODEL_SEEDS,
    EXPECTED_TARGET_IMAGES,
    EXPECTED_TARGET_LOCATIONS,
    verify_artifact_receipt,
)
from .ridge_gate import DECISIONS


def _class_index(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IntegrityError(f"{field} must be an integer output index")
    if not 0 <= value < EXPECTED_CLASS_COUNT:
        raise IntegrityError(f"{field} must lie in [0, {EXPECTED_CLASS_COUNT - 1}]")
    return value


def _finite_optional(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise IntegrityError(f"{field} must be finite or null")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"{field} must be finite or null") from exc
    if not math.isfinite(result):
        raise IntegrityError(f"{field} must be finite or null")
    return result


def _logit_vector(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != EXPECTED_CLASS_COUNT:
        raise IntegrityError(f"{field} must contain exactly 16 logits")
    result = []
    for item in value:
        if isinstance(item, bool):
            raise IntegrityError(f"{field} contains a non-numeric value")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise IntegrityError(f"{field} contains a non-numeric value") from exc
        if not math.isfinite(number):
            raise IntegrityError(f"{field} contains NaN or Infinity")
        result.append(number)
    return result


def _gate_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        "decision": value.get("decision"),
        "support_status": str(value.get("support_status", "")),
        "support_reasons": [str(item) for item in value.get("support_reasons", ())],
        "delta_hat": _finite_optional(value.get("delta_hat"), field="delta_hat"),
        "epsilon": _finite_optional(value.get("epsilon"), field="epsilon"),
        "lower": _finite_optional(value.get("lower"), field="lower"),
        "upper": _finite_optional(value.get("upper"), field="upper"),
        "mahalanobis_diagnostic": _finite_optional(
            value.get("mahalanobis_diagnostic"), field="mahalanobis_diagnostic"
        ),
        "partition_sha256": value.get("partition_sha256"),
        "probe_trace_sha256": value.get("probe_trace_sha256"),
    }
    if "probe_feature_record" in value:
        record["probe_feature_record"] = value["probe_feature_record"]
    return record


def _validate_action_evidence(
    gate_record: Mapping[str, Any],
    *,
    protocol_seal_sha256: str,
    gate_sha256: str,
    target_manifest_sha256: str,
    checkpoint_seed: int,
    checkpoint_tensor_sha256: str,
    location_id: str,
) -> None:
    action_hash = require_sha256(gate_record.get("action_sha256"), field="action_sha256")
    artifact_hash = require_sha256(
        gate_record.get("action_artifact_sha256"), field="action_artifact_sha256"
    )
    partition_hash = require_sha256(
        gate_record.get("partition_sha256"), field="partition_sha256"
    )
    probe_trace_hash = require_sha256(
        gate_record.get("probe_trace_sha256"), field="probe_trace_sha256"
    )
    receipt = gate_record.get("action_receipt")
    if not isinstance(receipt, Mapping):
        raise IntegrityError("prediction cell lacks the immutable action receipt")
    artifact_path = receipt.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise IntegrityError("action receipt lacks artifact_path")
    verified_receipt = verify_artifact_receipt(artifact_path)
    if dict(receipt) != verified_receipt:
        raise IntegrityError("embedded action receipt differs from the immutable receipt")
    if verified_receipt.get("artifact_sha256") != artifact_hash:
        raise IntegrityError("action artifact SHA-256 differs from the prediction-cell claim")
    action = strict_json_load(artifact_path)
    if not isinstance(action, Mapping):
        raise IntegrityError("sealed action artifact is not a JSON object")
    unsigned = dict(action)
    claimed = unsigned.pop("action_sha256", None)
    if claimed != action_hash or claimed != stable_sha256(unsigned):
        raise IntegrityError("sealed action action_sha256 mismatch")
    expected_identity = {
        "schema": "kbound_cct20_label_free_action_v1",
        "status": "SEALED_BEFORE_EVALUATION_STREAM",
        "protocol_seal_sha256": protocol_seal_sha256,
        "gate_sha256": gate_sha256,
        "target_manifest_sha256": target_manifest_sha256,
        "partition_sha256": partition_hash,
        "probe_trace_sha256": probe_trace_hash,
        "checkpoint_seed": checkpoint_seed,
        "checkpoint_tensor_sha256": checkpoint_tensor_sha256,
        "location_id": location_id,
    }
    for field, expected in expected_identity.items():
        if action.get(field) != expected:
            raise IntegrityError(f"sealed action {field} identity mismatch")
    action_gate = action.get("gate_result")
    if not isinstance(action_gate, Mapping):
        raise IntegrityError("sealed action lacks its gate result")
    if stable_sha256(_gate_semantics(action_gate)) != stable_sha256(
        _gate_semantics(gate_record)
    ):
        raise IntegrityError("prediction-cell gate evidence differs from the pre-evaluation action")


def build_prediction_cell(
    *,
    protocol_seal_sha256: str,
    gate_sha256: str,
    target_manifest_sha256: str,
    checkpoint_seed: int,
    checkpoint_tensor_sha256: str,
    location_id: str | int,
    gate_result: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one sealed-label-free checkpoint x location result shard."""

    protocol_hash = require_sha256(protocol_seal_sha256, field="protocol_seal_sha256")
    gate_hash = require_sha256(gate_sha256, field="gate_sha256")
    manifest_hash = require_sha256(target_manifest_sha256, field="target_manifest_sha256")
    checkpoint_hash = require_sha256(
        checkpoint_tensor_sha256, field="checkpoint_tensor_sha256"
    )
    if checkpoint_seed not in EXPECTED_MODEL_SEEDS:
        raise IntegrityError(f"checkpoint_seed must be one of {EXPECTED_MODEL_SEEDS}")
    location = str(location_id)
    if not location:
        raise IntegrityError("location_id cannot be empty")
    assert_label_free(gate_result, path="gate_result")
    decision = gate_result.get("decision")
    if decision not in DECISIONS:
        raise IntegrityError(f"invalid gate decision: {decision!r}")
    gate_record = {
        **_gate_semantics(gate_result),
        "decision": decision,
        "realized_action": "adapted" if decision == "ADAPT" else "frozen",
        "action_sha256": gate_result.get("action_sha256"),
        "action_artifact_sha256": gate_result.get("action_artifact_sha256"),
        "action_receipt": gate_result.get("action_receipt"),
    }
    if gate_record["action_sha256"] is not None:
        _validate_action_evidence(
            gate_record,
            protocol_seal_sha256=protocol_hash,
            gate_sha256=gate_hash,
            target_manifest_sha256=manifest_hash,
            checkpoint_seed=checkpoint_seed,
            checkpoint_tensor_sha256=checkpoint_hash,
            location_id=location,
        )
    elif any(
        gate_record.get(field) is not None
        for field in (
            "action_artifact_sha256",
            "action_receipt",
            "partition_sha256",
            "probe_trace_sha256",
        )
    ):
        raise IntegrityError("partial pre-evaluation action evidence is forbidden")
    if "probe_feature_record" in gate_record:
        probe_record = gate_record["probe_feature_record"]
        assert_label_free(probe_record, path="gate_result.probe_feature_record")
    support_status = gate_record["support_status"]
    if support_status not in {"IN_SUPPORT", "FAIL_CLOSED"}:
        raise IntegrityError(f"invalid gate support status: {support_status!r}")
    if support_status == "FAIL_CLOSED":
        if decision != "ABSTAIN":
            raise IntegrityError("a fail-closed support result must ABSTAIN")
    else:
        delta_hat = gate_record["delta_hat"]
        epsilon = gate_record["epsilon"]
        if delta_hat is None or epsilon is None or epsilon < 0.0:
            raise IntegrityError("an in-support gate result requires delta_hat and non-negative epsilon")
        expected_decision = (
            "ADAPT"
            if delta_hat - epsilon > 0.0
            else "FREEZE"
            if delta_hat + epsilon < 0.0
            else "ABSTAIN"
        )
        if decision != expected_decision:
            raise IntegrityError(
                f"gate decision {decision!r} violates the strict interval rule {expected_decision!r}"
            )
        for key, expected in (("lower", delta_hat - epsilon), ("upper", delta_hat + epsilon)):
            observed = gate_record[key]
            if observed is None or not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise IntegrityError(f"gate {key} does not equal delta_hat {'-' if key == 'lower' else '+'} epsilon")
    normalized = []
    seen: set[str] = set()
    for expected_stream_index, row in enumerate(rows):
        assert_label_free(row, path=f"rows[{expected_stream_index}]")
        image_id = str(row.get("image_id", ""))
        sequence_id = str(row.get("sequence_id", ""))
        role = str(row.get("role", ""))
        row_location = str(row.get("location_id", ""))
        stream_index = row.get("stream_index")
        if not image_id or not sequence_id:
            raise IntegrityError("prediction row requires image_id and sequence_id")
        if image_id in seen:
            raise IntegrityError(f"duplicate prediction image_id {image_id!r}")
        seen.add(image_id)
        if row_location != location:
            raise IntegrityError(
                f"prediction row location {row_location!r} does not match cell {location!r}"
            )
        if role not in {"probe", "evaluation"}:
            raise IntegrityError(f"prediction role must be probe/evaluation, found {role!r}")
        if stream_index != expected_stream_index:
            raise IntegrityError(
                "prediction stream_index must be contiguous and preserve execution order: "
                f"expected {expected_stream_index}, found {stream_index!r}"
            )
        frozen = _class_index(row.get("frozen_prediction"), field="frozen_prediction")
        adapted = _class_index(row.get("adapted_prediction"), field="adapted_prediction")
        selected = adapted if decision == "ADAPT" else frozen
        frozen_vector = _logit_vector(row.get("frozen_logits"), field="frozen_logits")
        adapted_vector = _logit_vector(row.get("adapted_logits"), field="adapted_logits")
        if max(range(16), key=frozen_vector.__getitem__) != frozen:
            raise IntegrityError("frozen prediction is not the argmax of frozen_logits")
        if max(range(16), key=adapted_vector.__getitem__) != adapted:
            raise IntegrityError("adapted prediction is not the argmax of adapted_logits")
        normalized_row = {
            "stream_index": expected_stream_index,
            "image_id": image_id,
            "sequence_id": sequence_id,
            "location_id": location,
            "role": role,
            "frozen_prediction": frozen,
            "adapted_prediction": adapted,
            "kga_prediction": selected,
            "frozen_logits": frozen_vector,
            "adapted_logits": adapted_vector,
        }
        normalized.append(normalized_row)
    if not normalized:
        raise IntegrityError("prediction cell cannot be empty")
    document = {
        "schema": "kbound_cct20_label_free_prediction_cell_v1",
        "protocol_seal_sha256": protocol_hash,
        "gate_sha256": gate_hash,
        "target_manifest_sha256": manifest_hash,
        "checkpoint_seed": checkpoint_seed,
        "checkpoint_tensor_sha256": checkpoint_hash,
        "location_id": location,
        "n_images": len(normalized),
        "gate": gate_record,
        "rows": normalized,
    }
    assert_label_free(document)
    document["cell_sha256"] = stable_sha256(document)
    validate_prediction_cell(document)
    return document


def validate_prediction_cell(document: Mapping[str, Any]) -> None:
    if document.get("schema") != "kbound_cct20_label_free_prediction_cell_v1":
        raise IntegrityError("unknown target prediction-cell schema")
    unsigned = dict(document)
    claimed = unsigned.pop("cell_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("prediction cell_sha256 mismatch")
    assert_label_free(unsigned)
    for field in (
        "protocol_seal_sha256",
        "gate_sha256",
        "target_manifest_sha256",
        "checkpoint_tensor_sha256",
    ):
        require_sha256(document.get(field), field=field)
    checkpoint_seed = document.get("checkpoint_seed")
    if checkpoint_seed not in EXPECTED_MODEL_SEEDS:
        raise IntegrityError(f"prediction cell checkpoint seed must be {EXPECTED_MODEL_SEEDS}")
    decision = document.get("gate", {}).get("decision")
    if decision not in DECISIONS:
        raise IntegrityError("prediction cell has invalid decision")
    selected_action = "adapted" if decision == "ADAPT" else "frozen"
    if document.get("gate", {}).get("realized_action") != selected_action:
        raise IntegrityError("prediction cell realized action does not match gate decision")
    gate = document["gate"]
    support_status = gate.get("support_status")
    if support_status == "FAIL_CLOSED":
        if decision != "ABSTAIN":
            raise IntegrityError("fail-closed prediction cell must ABSTAIN")
    elif support_status == "IN_SUPPORT":
        delta_hat = gate.get("delta_hat")
        epsilon = gate.get("epsilon")
        if not isinstance(delta_hat, (int, float)) or not isinstance(epsilon, (int, float)):
            raise IntegrityError("in-support prediction cell lacks interval values")
        if not math.isfinite(float(delta_hat)) or not math.isfinite(float(epsilon)) or epsilon < 0.0:
            raise IntegrityError("in-support prediction cell has invalid interval values")
        expected_decision = (
            "ADAPT"
            if delta_hat - epsilon > 0.0
            else "FREEZE"
            if delta_hat + epsilon < 0.0
            else "ABSTAIN"
        )
        if decision != expected_decision:
            raise IntegrityError("prediction-cell decision violates the strict interval rule")
        for field, expected in (
            ("lower", float(delta_hat) - float(epsilon)),
            ("upper", float(delta_hat) + float(epsilon)),
        ):
            observed = gate.get(field)
            if not isinstance(observed, (int, float)) or not math.isclose(
                float(observed), expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise IntegrityError(f"prediction-cell gate {field} does not replay")
    else:
        raise IntegrityError("prediction cell has invalid support status")
    location = str(document.get("location_id", ""))
    if location not in EXPECTED_TARGET_LOCATIONS:
        raise IntegrityError("prediction cell location is not in the sealed target set")
    rows = list(document.get("rows", ()))
    if document.get("n_images") != len(rows) or not rows:
        raise IntegrityError("prediction cell row count mismatch")
    seen = set()
    roles = set()
    sequence_roles: dict[str, str] = {}
    for index, row in enumerate(rows):
        if row.get("stream_index") != index:
            raise IntegrityError("prediction cell stream index is not contiguous")
        image_id = str(row.get("image_id", ""))
        if not image_id or image_id in seen:
            raise IntegrityError("prediction cell has empty/duplicate image_id")
        seen.add(image_id)
        if str(row.get("location_id", "")) != location:
            raise IntegrityError("prediction row/cell location mismatch")
        role = row.get("role")
        if role not in {"probe", "evaluation"}:
            raise IntegrityError("prediction row has invalid probe/evaluation role")
        roles.add(role)
        sequence_id = str(row.get("sequence_id", ""))
        if not sequence_id:
            raise IntegrityError("prediction row has empty sequence_id")
        prior_role = sequence_roles.setdefault(sequence_id, role)
        if prior_role != role:
            raise IntegrityError("one sequence occurs in both probe and evaluation roles")
        frozen = _class_index(row.get("frozen_prediction"), field="frozen_prediction")
        adapted = _class_index(row.get("adapted_prediction"), field="adapted_prediction")
        frozen_vector = _logit_vector(row.get("frozen_logits"), field="frozen_logits")
        adapted_vector = _logit_vector(row.get("adapted_logits"), field="adapted_logits")
        if max(range(16), key=frozen_vector.__getitem__) != frozen:
            raise IntegrityError("stored frozen argmax is inconsistent")
        if max(range(16), key=adapted_vector.__getitem__) != adapted:
            raise IntegrityError("stored adapted argmax is inconsistent")
        expected = adapted if decision == "ADAPT" else frozen
        if row.get("kga_prediction") != expected:
            raise IntegrityError("kga_prediction does not follow the frozen gate action")
    if roles != {"probe", "evaluation"}:
        raise IntegrityError("prediction cell must contain both probe and evaluation images")
    probe_rows = [row for row in rows if row["role"] == "probe"]
    probe_feature_record = gate.get("probe_feature_record")
    if probe_feature_record is not None:
        features = probe_feature_record.get("features", {})
        recomputed = extract_label_free_features(
            [row["frozen_logits"] for row in probe_rows],
            [row["adapted_logits"] for row in probe_rows],
            normalized_tent_update_norm=features.get("normalized_tent_update_norm"),
            batchnorm_batch_source_statistic_divergence=features.get(
                "batchnorm_batch_source_statistic_divergence"
            ),
        )
        if probe_feature_record.get("n_probe_images") != len(probe_rows):
            raise IntegrityError("probe feature record image count mismatch")
        if tuple(probe_feature_record.get("feature_names", ())) != tuple(
            recomputed["feature_names"]
        ):
            raise IntegrityError("probe feature record schema mismatch")
        for name, expected in recomputed["features"].items():
            observed = features.get(name)
            if not isinstance(observed, (int, float)) or not math.isclose(
                float(observed), expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise IntegrityError(f"probe feature {name!r} does not replay from stored logits")
    if gate.get("action_sha256") is not None:
        probe_trace = [
            {
                "image_id": row["image_id"],
                "sequence_id": row["sequence_id"],
                "location_id": row["location_id"],
                "frozen_logits": row["frozen_logits"],
                "adapted_logits": row["adapted_logits"],
            }
            for row in probe_rows
        ]
        if gate.get("probe_trace_sha256") != stable_sha256(probe_trace):
            raise IntegrityError("stored probe logits do not match the pre-evaluation trace hash")
        _validate_action_evidence(
            gate,
            protocol_seal_sha256=str(document.get("protocol_seal_sha256", "")),
            gate_sha256=str(document.get("gate_sha256", "")),
            target_manifest_sha256=str(document.get("target_manifest_sha256", "")),
            checkpoint_seed=int(checkpoint_seed),
            checkpoint_tensor_sha256=str(document.get("checkpoint_tensor_sha256", "")),
            location_id=location,
        )


def build_prediction_collection(
    cells: Iterable[Mapping[str, Any]],
    *,
    target_index: Iterable[Mapping[str, Any]],
    target_location_ids: Sequence[str | int],
    expected_target_images: int = EXPECTED_TARGET_IMAGES,
    require_replayable_probe_features: bool = True,
) -> dict[str, Any]:
    """Reconcile all 5 x 9 shards against the label-free target population."""

    locations = tuple(str(value) for value in target_location_ids)
    if locations != EXPECTED_TARGET_LOCATIONS:
        raise IntegrityError(
            "prediction collection target locations/order must be exactly "
            f"{EXPECTED_TARGET_LOCATIONS}"
        )
    if isinstance(expected_target_images, bool) or expected_target_images < 1:
        raise IntegrityError("expected_target_images must be positive")
    index: dict[str, str] = {}
    normalized_target_rows: list[dict[str, Any]] = []
    for position, row in enumerate(target_index):
        assert_label_free(row, path=f"target_index[{position}]")
        normalized_row = normalize_metadata_row(row).as_dict()
        image_id = normalized_row["image_id"]
        location = normalized_row["location_id"]
        if not image_id or location not in locations:
            raise IntegrityError("target index has empty image id or unexpected location")
        if image_id in index:
            raise IntegrityError(f"duplicate image_id in target index: {image_id!r}")
        index[image_id] = location
        normalized_target_rows.append(normalized_row)
    if len(index) != expected_target_images:
        raise IntegrityError(
            f"target index count mismatch: expected {expected_target_images}, found {len(index)}"
        )
    expected_plan: dict[str, list[tuple[str, str, str]]] = {}
    expected_partition_sha256: dict[str, str] = {}
    for location in locations:
        location_rows = [
            row for row in normalized_target_rows if row["location_id"] == location
        ]
        partition = sequence_atomic_partition(
            location_rows,
            probe_fraction=PROBE_FRACTION,
            salt=PARTITION_SALT,
        )
        expected_partition_sha256[location] = stable_sha256(partition)
        plan: list[tuple[str, str, str]] = []
        for role in ("probe", "evaluation"):
            batches = sequence_atomic_batches(
                partition["roles"][role],
                max_images=TARGET_BATCH_SIZE,
                order="native",
                merge_singleton_final=True,
            )
            plan.extend(
                (row["image_id"], row["sequence_id"], role)
                for batch in batches
                for row in batch
            )
        expected_plan[location] = plan

    documents = [dict(cell) for cell in cells]
    for document in documents:
        validate_prediction_cell(document)
        if require_replayable_probe_features and "probe_feature_record" not in document["gate"]:
            raise IntegrityError("prediction cell lacks replayable probe feature evidence")
        if require_replayable_probe_features and not document["gate"].get("action_sha256"):
            raise IntegrityError("prediction cell lacks a pre-evaluation immutable action seal")
        if require_replayable_probe_features and not document["gate"].get("action_receipt"):
            raise IntegrityError("prediction cell lacks a verified pre-evaluation action receipt")
    keyed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for document in documents:
        key = (document.get("checkpoint_seed"), str(document.get("location_id")))
        if key in keyed:
            raise IntegrityError(f"duplicate checkpoint-location prediction cell: {key}")
        keyed[key] = document
    expected_keys = {(seed, location) for seed in EXPECTED_MODEL_SEEDS for location in locations}
    if set(keyed) != expected_keys:
        missing = sorted(expected_keys - set(keyed))
        extra = sorted(set(keyed) - expected_keys)
        raise IntegrityError(f"prediction cell grid mismatch; missing={missing}, extra={extra}")

    checkpoint_hashes: dict[int, str] = {}
    action_counts = dict.fromkeys(DECISIONS, 0)
    reference_identity_order: list[tuple[str, str, str]] | None = None
    for seed in EXPECTED_MODEL_SEEDS:
        seen_for_seed: set[str] = set()
        identity_order: list[tuple[str, str, str]] = []
        for location in locations:
            document = keyed[(seed, location)]
            if document["gate"].get("partition_sha256") != expected_partition_sha256[location]:
                raise IntegrityError(
                    f"prediction cell partition hash differs from the sealed hash rule: {seed}, {location}"
                )
            observed_plan = [
                (str(row["image_id"]), str(row["sequence_id"]), str(row["role"]))
                for row in document["rows"]
            ]
            if observed_plan != expected_plan[location]:
                raise IntegrityError(
                    f"prediction cell stream/role plan differs from the sealed target index: {seed}, {location}"
                )
            checkpoint_hash = str(document["checkpoint_tensor_sha256"])
            prior = checkpoint_hashes.setdefault(seed, checkpoint_hash)
            if prior != checkpoint_hash:
                raise IntegrityError(f"checkpoint tensor hash changed across locations for seed {seed}")
            action_counts[document["gate"]["decision"]] += 1
            for row in document["rows"]:
                image_id = str(row["image_id"])
                if index.get(image_id) != location:
                    raise IntegrityError(
                        f"prediction image {image_id!r} is missing or assigned to wrong location"
                    )
                if image_id in seen_for_seed:
                    raise IntegrityError(f"seed {seed} predicts image {image_id!r} more than once")
                seen_for_seed.add(image_id)
                identity_order.append((image_id, str(row["sequence_id"]), str(row["role"])))
        if seen_for_seed != set(index):
            raise IntegrityError(f"seed {seed} does not cover the complete target population")
        if reference_identity_order is None:
            reference_identity_order = identity_order
        elif identity_order != reference_identity_order:
            raise IntegrityError(
                "checkpoint seeds disagree on target stream order, sequence identity, or probe/evaluation role"
            )
    if len(set(checkpoint_hashes.values())) != len(EXPECTED_MODEL_SEEDS):
        raise IntegrityError("prediction collection does not use five distinct checkpoint tensors")
    common_fields = ("protocol_seal_sha256", "gate_sha256", "target_manifest_sha256")
    identities = {field: {document[field] for document in documents} for field in common_fields}
    if any(len(values) != 1 for values in identities.values()):
        raise IntegrityError(f"prediction cells disagree on sealed identities: {identities}")

    summaries = [
        {
            "checkpoint_seed": seed,
            "location_id": location,
            "n_images": keyed[(seed, location)]["n_images"],
            "decision": keyed[(seed, location)]["gate"]["decision"],
            "cell_sha256": keyed[(seed, location)]["cell_sha256"],
        }
        for seed in EXPECTED_MODEL_SEEDS
        for location in locations
    ]
    document = {
        "schema": "kbound_cct20_label_free_prediction_collection_v1",
        "status": "SEALED_BEFORE_LABEL_JOIN",
        "protocol_seal_sha256": next(iter(identities["protocol_seal_sha256"])),
        "gate_sha256": next(iter(identities["gate_sha256"])),
        "target_manifest_sha256": next(iter(identities["target_manifest_sha256"])),
        "target_image_count": len(index),
        "prediction_row_count": len(index) * len(EXPECTED_MODEL_SEEDS),
        "checkpoint_count": len(EXPECTED_MODEL_SEEDS),
        "location_count": len(locations),
        "cell_count": len(summaries),
        "checkpoint_tensor_sha256": {
            str(seed): checkpoint_hashes[seed] for seed in EXPECTED_MODEL_SEEDS
        },
        "action_counts_at_cell_unit": action_counts,
        "replayable_probe_features_required": require_replayable_probe_features,
        "pre_evaluation_action_seals_required": require_replayable_probe_features,
        "cells": summaries,
    }
    assert_label_free(document)
    document["collection_sha256"] = stable_sha256(document)
    return document
