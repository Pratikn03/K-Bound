#!/usr/bin/env python3
"""Build the immutable, data-driven CCT-20 paper release artifacts.

This is a downstream release step.  It accepts only completed sealed artifacts;
it has no target-annotation argument, does not import the one-shot scorer, and
does not compute target predictions.  Before emitting any paper-facing value it
replays byte receipts, upstream validators, all 45 checkpoint-by-location
identities, and the frozen two-way inference analysis.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.kbound.cct20.integrity import (  # noqa: E402
    IntegrityError,
    file_sha256,
    require_sha256,
    stable_sha256,
)
from experiments.kbound.cct20.prediction_artifacts import (  # noqa: E402
    build_prediction_collection,
    validate_prediction_cell,
)
from experiments.kbound.cct20.prospective_data import (  # noqa: E402
    validate_locked_target_population,
)
from experiments.kbound.cct20.protocol_seal import (  # noqa: E402
    EXPECTED_DEVELOPMENT_TRACE_COUNT,
    EXPECTED_MODEL_SEEDS,
    EXPECTED_TARGET_IMAGES,
    EXPECTED_TARGET_LOCATIONS,
    REQUIRED_CODE_DEPENDENCY_NAMES,
    REQUIRED_DATA_DEPENDENCY_NAMES,
    validate_execution_seal,
    verify_artifact_receipt,
    verify_execution_environment,
)
from experiments.kbound.cct20.ridge_gate import DECISIONS, validate_gate_document  # noqa: E402
from experiments.kbound.cct20.run_development_gate import (  # noqa: E402
    validate_development_trace_collection,
)
from experiments.kbound.cct20.run_locked_target import normalize_target_manifest  # noqa: E402
from experiments.kbound.cct20.runner_runtime import verify_checkpoint_audit_document  # noqa: E402
from experiments.kbound.cct20.two_way_inference import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COMPARISONS,
    analyze_score_document,
)

RELEASE_SCHEMA = "kbound_cct20_release_manifest_v1"
RELEASE_STATUS = "RELEASE_COMPLETE"
BENEFIT_SIGN = "adapted_accuracy_minus_frozen_accuracy"
PRIMARY_CONTRAST_SIGN = "baseline_regret_minus_kga_regret; positive_favors_kga"
SIMULTANEOUS_INTERVAL_LEVEL = 0.975
LOCATION_CLUSTER_COUNT = 9
CELL_COUNT = 45
CHECKPOINT_COUNT = 5
SCORING_LABEL_CONTRACT = "set_membership_top1_and_16_indicator_multilabel_macro_f1"


def _load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot load JSON object {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"JSON artifact is not an object: {source}")
    return value


def _load_received_json(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = verify_artifact_receipt(path)
    return _load_json_object(path), receipt


def _received_identity(path: str | Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    artifact = Path(path).expanduser().resolve()
    receipt_path = artifact.with_name(artifact.name + ".receipt.json")
    if receipt.get("artifact_path") != str(artifact):
        raise IntegrityError(f"receipt path does not identify {artifact}")
    byte_count = receipt.get("artifact_bytes")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
        raise IntegrityError(f"receipt has an invalid byte count for {artifact}")
    return {
        "path": str(artifact),
        "bytes": byte_count,
        "sha256": require_sha256(receipt.get("artifact_sha256"), field="artifact_sha256"),
        "canonical_document_sha256": require_sha256(
            receipt.get("canonical_document_sha256"),
            field="canonical_document_sha256",
        ),
        "receipt_path": str(receipt_path),
        "receipt_bytes": receipt_path.stat().st_size,
        "receipt_sha256": file_sha256(receipt_path),
    }


def _plain_json_identity(path: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    artifact = Path(path).expanduser().resolve()
    try:
        payload = artifact.read_bytes()
        observed = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot snapshot plain JSON dependency {artifact}: {exc}") from exc
    if observed != dict(document):
        raise IntegrityError(f"plain JSON dependency changed after validation: {artifact}")
    return {
        "path": str(artifact),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_document_sha256": stable_sha256(observed),
    }


def _plain_file_identity(path: str | Path) -> dict[str, Any]:
    artifact = Path(path).expanduser().resolve()
    if not artifact.is_file():
        raise IntegrityError(f"release dependency is missing: {artifact}")
    try:
        payload = artifact.read_bytes()
    except OSError as exc:
        raise IntegrityError(f"cannot snapshot release dependency {artifact}: {exc}") from exc
    return {
        "path": str(artifact),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _dependency_by_name(execution: Mapping[str, Any], field: str) -> dict[str, dict[str, Any]]:
    rows = execution.get(field)
    if not isinstance(rows, list) or not rows or not all(isinstance(row, Mapping) for row in rows):
        raise IntegrityError(f"execution seal {field} is not a non-empty dependency list")
    keyed = {str(row.get("name", "")): dict(row) for row in rows}
    if "" in keyed or len(keyed) != len(rows):
        raise IntegrityError(f"execution seal {field} has an empty or duplicate name")
    return keyed


def _require_dependency_path(
    dependencies: Mapping[str, Mapping[str, Any]],
    name: str,
    path: str | Path,
) -> None:
    record = dependencies.get(name)
    artifact = Path(path).expanduser().resolve()
    if not isinstance(record, Mapping):
        raise IntegrityError(f"execution seal lacks dependency {name!r}")
    if not artifact.is_file():
        raise IntegrityError(f"execution dependency {name!r} is missing: {artifact}")
    expected = {
        "path": str(artifact),
        "bytes": artifact.stat().st_size,
        "sha256": file_sha256(artifact),
    }
    if any(record.get(field) != value for field, value in expected.items()):
        raise IntegrityError(f"execution dependency {name!r} does not identify {artifact}")


def _execution_dependency_bundle(
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose every execution dependency, rather than only its transitive seal hash."""

    dataset = _dependency_by_name(execution, "dataset_dependencies")
    code = _dependency_by_name(execution, "code_dependencies")
    if set(dataset) != set(REQUIRED_DATA_DEPENDENCY_NAMES):
        raise IntegrityError("release dataset-dependency names differ from the execution contract")
    if set(code) != set(REQUIRED_CODE_DEPENDENCY_NAMES):
        raise IntegrityError("release code-dependency names differ from the execution contract")
    dataset_items = [dataset[name] for name in sorted(dataset)]
    code_items = [code[name] for name in sorted(code)]
    return {
        "dataset_count": len(dataset_items),
        "code_count": len(code_items),
        "total_count": len(dataset_items) + len(code_items),
        "aggregate_sha256": stable_sha256({"dataset_dependencies": dataset_items, "code_dependencies": code_items}),
        "dataset_items": dataset_items,
        "code_items": code_items,
    }


def _development_trace_bundle(
    development: Mapping[str, Any],
    *,
    code_dependencies: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-bind all 55 trace/receipt pairs to their execution-seal slots."""

    artifacts = development.get("trace_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != EXPECTED_DEVELOPMENT_TRACE_COUNT:
        raise IntegrityError("release requires exactly 55 development trace artifacts")
    if not all(isinstance(record, Mapping) for record in artifacts):
        raise IntegrityError("development trace artifact ledger contains a non-object")
    ordered = sorted(artifacts, key=lambda record: str(record.get("trace_id", "")))
    identities = []
    for index, record in enumerate(ordered):
        trace_path = Path(str(record.get("artifact_path", ""))).expanduser().resolve()
        receipt_path = trace_path.with_name(trace_path.name + ".receipt.json")
        _require_dependency_path(
            code_dependencies,
            f"development_trace_{index:02d}",
            trace_path,
        )
        _require_dependency_path(
            code_dependencies,
            f"development_trace_receipt_{index:02d}",
            receipt_path,
        )
        receipt = verify_artifact_receipt(trace_path)
        if dict(record.get("artifact_receipt", {})) != receipt:
            raise IntegrityError("development trace ledger receipt differs from its artifact")
        identity = _received_identity(trace_path, receipt)
        identity.update(
            {
                "trace_id": str(record.get("trace_id", "")),
                "trace_sha256": require_sha256(record.get("trace_sha256"), field="development_trace.trace_sha256"),
            }
        )
        identities.append(identity)
    if len({row["path"] for row in identities}) != EXPECTED_DEVELOPMENT_TRACE_COUNT:
        raise IntegrityError("development trace release ledger has duplicate artifact paths")
    return {
        "count": len(identities),
        "aggregate_sha256": stable_sha256(identities),
        "items": identities,
    }


def _prediction_action_bundle(
    prediction_cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Retain the 45 immutable pre-evaluation action receipts in the release chain."""

    identities = []
    for cell in prediction_cells:
        receipt = cell.get("gate", {}).get("action_receipt")
        if not isinstance(receipt, Mapping):
            raise IntegrityError("prediction cell lacks its pre-evaluation action receipt")
        action_path = receipt.get("artifact_path")
        if not isinstance(action_path, str) or not action_path:
            raise IntegrityError("prediction action receipt lacks an artifact path")
        identity = _received_identity(action_path, receipt)
        identity.update(
            {
                "checkpoint_seed": int(cell["checkpoint_seed"]),
                "location_id": str(cell["location_id"]),
                "action_sha256": require_sha256(cell.get("gate", {}).get("action_sha256"), field="action_sha256"),
            }
        )
        identities.append(identity)
    identities.sort(
        key=lambda row: (
            int(row["checkpoint_seed"]),
            str(row["location_id"]),
        )
    )
    if (
        len(identities) != CELL_COUNT
        or len({row["path"] for row in identities}) != CELL_COUNT
        or len({row["sha256"] for row in identities}) != CELL_COUNT
    ):
        raise IntegrityError("release requires 45 distinct pre-evaluation action artifacts")
    return {
        "count": len(identities),
        "aggregate_sha256": stable_sha256(identities),
        "items": identities,
    }


def _validate_prediction_collection(document: Mapping[str, Any]) -> None:
    if (
        document.get("schema") != "kbound_cct20_label_free_prediction_collection_v1"
        or document.get("status") != "SEALED_BEFORE_LABEL_JOIN"
    ):
        raise IntegrityError("prediction collection is not the sealed CCT-20 collection")
    unsigned = dict(document)
    claimed = unsigned.pop("collection_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("prediction collection SHA-256 mismatch")
    if (
        document.get("checkpoint_count") != CHECKPOINT_COUNT
        or document.get("location_count") != LOCATION_CLUSTER_COUNT
        or document.get("cell_count") != CELL_COUNT
        or len(document.get("cells", ())) != CELL_COUNT
    ):
        raise IntegrityError("prediction collection is not the complete 5 x 9 grid")
    if (
        document.get("target_image_count") != EXPECTED_TARGET_IMAGES
        or document.get("prediction_row_count") != EXPECTED_TARGET_IMAGES * CHECKPOINT_COUNT
    ):
        raise IntegrityError("prediction collection target population/count drift")
    if (
        document.get("replayable_probe_features_required") is not True
        or document.get("pre_evaluation_action_seals_required") is not True
    ):
        raise IntegrityError("prediction collection does not require replayable sealed actions")


def _prediction_grid(
    collection: Mapping[str, Any],
    prediction_cells: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    if len(prediction_cells) != CELL_COUNT:
        raise IntegrityError(f"release requires exactly 45 prediction cells, found {len(prediction_cells)}")
    keyed: dict[tuple[int, str], dict[str, Any]] = {}
    by_hash: dict[str, dict[str, Any]] = {}
    for raw in prediction_cells:
        cell = dict(raw)
        validate_prediction_cell(cell)
        key = (int(cell.get("checkpoint_seed", -1)), str(cell.get("location_id", "")))
        if key in keyed:
            raise IntegrityError(f"duplicate prediction cell {key}")
        keyed[key] = cell
        cell_hash = require_sha256(cell.get("cell_sha256"), field="cell_sha256")
        if cell_hash in by_hash:
            raise IntegrityError("duplicate prediction-cell content hash")
        by_hash[cell_hash] = cell
    expected = {(seed, location) for seed in EXPECTED_MODEL_SEEDS for location in EXPECTED_TARGET_LOCATIONS}
    if set(keyed) != expected:
        raise IntegrityError("prediction cells do not form the exact sealed 5 x 9 product")
    summaries = collection.get("cells")
    if not isinstance(summaries, list) or not all(isinstance(row, Mapping) for row in summaries):
        raise IntegrityError("prediction collection cell summaries are invalid")
    expected_summaries = {str(row.get("cell_sha256")): dict(row) for row in summaries}
    if len(expected_summaries) != CELL_COUNT or set(expected_summaries) != set(by_hash):
        raise IntegrityError("prediction collection hashes differ from the 45 cell artifacts")
    action_counts = dict.fromkeys(DECISIONS, 0)
    checkpoint_hashes: dict[str, str] = {}
    for cell_hash, cell in by_hash.items():
        summary = expected_summaries[cell_hash]
        expected_summary = {
            "checkpoint_seed": cell["checkpoint_seed"],
            "location_id": cell["location_id"],
            "n_images": cell["n_images"],
            "decision": cell["gate"]["decision"],
            "cell_sha256": cell_hash,
        }
        if summary != expected_summary:
            raise IntegrityError("prediction collection summary differs from its cell")
        action_counts[cell["gate"]["decision"]] += 1
        seed = str(cell["checkpoint_seed"])
        tensor = str(cell["checkpoint_tensor_sha256"])
        if checkpoint_hashes.setdefault(seed, tensor) != tensor:
            raise IntegrityError("checkpoint tensor changes across target locations")
    if collection.get("action_counts_at_cell_unit") != action_counts:
        raise IntegrityError("prediction collection action counts do not reconcile")
    if collection.get("checkpoint_tensor_sha256") != checkpoint_hashes:
        raise IntegrityError("prediction collection checkpoint identities do not reconcile")
    return keyed


def _replay_prediction_collection(
    collection: Mapping[str, Any],
    *,
    prediction_cells: Sequence[Mapping[str, Any]],
    target_manifest: Mapping[str, Any],
) -> None:
    """Replay full target ID, stream, role, and 5 x 9 coverage without labels."""

    manifest = dict(target_manifest)
    validate_locked_target_population(manifest)
    metadata, _ = normalize_target_manifest(manifest)
    replayed = build_prediction_collection(
        prediction_cells,
        target_index=metadata,
        target_location_ids=EXPECTED_TARGET_LOCATIONS,
        expected_target_images=EXPECTED_TARGET_IMAGES,
        require_replayable_probe_features=True,
    )
    if dict(collection) != replayed:
        changed = sorted(key for key in set(collection) | set(replayed) if collection.get(key) != replayed.get(key))
        raise IntegrityError(f"prediction collection differs from the full label-free target replay: {changed}")


def _reconcile_score_with_predictions(
    score: Mapping[str, Any],
    prediction_grid: Mapping[tuple[int, str], Mapping[str, Any]],
) -> None:
    score_cells = score.get("cells")
    if not isinstance(score_cells, list) or len(score_cells) != CELL_COUNT:
        raise IntegrityError("score does not contain 45 cells")
    keyed = {
        (int(row.get("checkpoint_seed", -1)), str(row.get("location_id", ""))): row
        for row in score_cells
        if isinstance(row, Mapping)
    }
    if len(keyed) != CELL_COUNT or set(keyed) != set(prediction_grid):
        raise IntegrityError("scored and prediction grids differ")
    totals_by_seed = {seed: {"target": 0, "probe": 0, "evaluation": 0} for seed in EXPECTED_MODEL_SEEDS}
    for key, scored in keyed.items():
        predicted = prediction_grid[key]
        rows = list(predicted.get("rows", ()))
        n_probe = sum(row.get("role") == "probe" for row in rows)
        n_evaluation = sum(row.get("role") == "evaluation" for row in rows)
        expected = {
            "checkpoint_tensor_sha256": predicted.get("checkpoint_tensor_sha256"),
            "decision": predicted.get("gate", {}).get("decision"),
            "n_target_images": predicted.get("n_images"),
            "n_probe_images": n_probe,
            "n_evaluation_images": n_evaluation,
        }
        for field, value in expected.items():
            if scored.get(field) != value:
                raise IntegrityError(f"scored cell {key} differs from prediction field {field}")
        seed_totals = totals_by_seed[key[0]]
        seed_totals["target"] += len(rows)
        seed_totals["probe"] += n_probe
        seed_totals["evaluation"] += n_evaluation
    expected_totals = {
        "target": score.get("target_image_count"),
        "probe": score.get("probe_image_count"),
        "evaluation": score.get("evaluation_image_count"),
    }
    for seed, totals in totals_by_seed.items():
        if totals != expected_totals:
            raise IntegrityError(f"score global target/probe/evaluation counts differ from seed {seed} cells")


def _validate_multilabel_report(
    report: Any,
    *,
    n_evaluation: int,
    label: str,
) -> tuple[list[int], int]:
    if not isinstance(report, Mapping):
        raise IntegrityError(f"{label} multilabel report is missing")
    zero_convention = report.get("zero_denominator_convention")
    if report.get("n_output_indicators") != 16 or isinstance(zero_convention, bool) or zero_convention != 0.0:
        raise IntegrityError(f"{label} multilabel report contract drift")
    rows = report.get("per_class")
    if not isinstance(rows, list) or len(rows) != 16:
        raise IntegrityError(f"{label} multilabel report must contain 16 class rows")
    truth_positives = []
    predicted_positives = 0
    true_positives = 0
    f1_values = []
    for output_index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"output_index", "tp", "fp", "fn", "f1"}:
            raise IntegrityError(f"{label} multilabel class-row schema drift")
        if row.get("output_index") != output_index:
            raise IntegrityError(f"{label} multilabel output indices are not 0..15")
        counts = []
        for field in ("tp", "fp", "fn"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= n_evaluation:
                raise IntegrityError(f"{label} multilabel {field} is invalid")
            counts.append(value)
        tp, fp, fn = counts
        if tp + fp > n_evaluation or tp + fn > n_evaluation:
            raise IntegrityError(f"{label} multilabel counts exceed the evaluation population")
        denominator = 2 * tp + fp + fn
        expected_f1 = 0.0 if denominator == 0 else (2.0 * tp) / denominator
        observed_f1 = row.get("f1")
        if (
            isinstance(observed_f1, bool)
            or not isinstance(observed_f1, (int, float))
            or not math.isfinite(float(observed_f1))
            or not math.isclose(float(observed_f1), expected_f1, abs_tol=1e-15)
        ):
            raise IntegrityError(f"{label} multilabel F1 does not reconcile to TP/FP/FN")
        truth_positives.append(tp + fn)
        predicted_positives += tp + fp
        true_positives += tp
        f1_values.append(expected_f1)
    if predicted_positives != n_evaluation:
        raise IntegrityError(f"{label} is not a one-hot prediction report")
    expected_macro = math.fsum(f1_values) / 16.0
    macro = report.get("macro_f1")
    if (
        isinstance(macro, bool)
        or not isinstance(macro, (int, float))
        or not math.isfinite(float(macro))
        or not math.isclose(float(macro), expected_macro, abs_tol=1e-15)
    ):
        raise IntegrityError(f"{label} macro-F1 does not reconcile to its class rows")
    return truth_positives, true_positives


def _validate_score_cell_secondary(row: Mapping[str, Any], *, index: int) -> None:
    n_evaluation = row.get("n_evaluation_images")
    if isinstance(n_evaluation, bool) or not isinstance(n_evaluation, int) or n_evaluation < 1:
        raise IntegrityError(f"one-shot score cell {index} has an invalid evaluation count")
    reports = row.get("multilabel_macro_f1")
    expected_names = {"always_freeze", "always_adapt", "kga"}
    if not isinstance(reports, Mapping) or set(reports) != expected_names:
        raise IntegrityError(f"one-shot score cell {index} lacks the complete secondary outcome")
    accuracies = row.get("set_membership_top1_accuracy")
    if not isinstance(accuracies, Mapping) or set(accuracies) != expected_names:
        raise IntegrityError(f"one-shot score cell {index} accuracy schema drift")
    truth_vectors = []
    for name in ("always_freeze", "always_adapt", "kga"):
        truth_vector, true_positives = _validate_multilabel_report(
            reports[name],
            n_evaluation=n_evaluation,
            label=f"score cell {index} {name}",
        )
        truth_vectors.append(truth_vector)
        if not math.isclose(float(accuracies[name]), true_positives / n_evaluation, abs_tol=1e-15):
            raise IntegrityError(f"one-shot score cell {index} {name} primary/secondary outcomes disagree")
    if truth_vectors[1:] != truth_vectors[:-1]:
        raise IntegrityError(f"one-shot score cell {index} policies use different truth indicators")
    selected = "always_adapt" if row.get("decision") == "ADAPT" else "always_freeze"
    if reports["kga"] != reports[selected]:
        raise IntegrityError(f"one-shot score cell {index} KGA secondary outcome ignores its action")


def _validate_inference_replay(
    score: Mapping[str, Any],
    inference: Mapping[str, Any],
) -> dict[str, Any]:
    replayed = analyze_score_document(score)
    if dict(inference) != replayed:
        changed = sorted(key for key in set(inference) | set(replayed) if inference.get(key) != replayed.get(key))
        raise IntegrityError(f"two-way inference differs from the frozen replay: {changed}")
    return replayed


def _validate_score_release_contract(
    score: Mapping[str, Any],
    *,
    target_annotations_sha256: str,
) -> None:
    expected_label_contract = {
        "primary": "top1_correct_iff_prediction_in_complete_distinct_category_set",
        "secondary": "macro_f1_over_all_16_indicators_with_top1_as_one_hot_prediction",
        "repeated_same_category": "collapsed",
        "zero_annotation_image": "experiment_failure",
    }
    if score.get("label_contract") != expected_label_contract:
        raise IntegrityError("one-shot score label contract drift")
    if score.get("target_annotations_file_sha256") != target_annotations_sha256:
        raise IntegrityError("one-shot score target-annotation identity differs from the execution seal")
    require_sha256(score.get("truth_join_sha256"), field="truth_join_sha256")
    target = score.get("target_image_count")
    probe = score.get("probe_image_count")
    evaluation = score.get("evaluation_image_count")
    if (
        target != EXPECTED_TARGET_IMAGES
        or isinstance(probe, bool)
        or not isinstance(probe, int)
        or isinstance(evaluation, bool)
        or not isinstance(evaluation, int)
        or probe < 1
        or evaluation < 1
        or probe + evaluation != target
        or score.get("evaluation_prediction_row_count") != evaluation * CHECKPOINT_COUNT
    ):
        raise IntegrityError("one-shot score target/probe/evaluation counts drift")
    cells = score.get("cells")
    if not isinstance(cells, list) or len(cells) != CELL_COUNT:
        raise IntegrityError("one-shot score does not contain exactly 45 cells")
    for index, row in enumerate(cells):
        if (
            not isinstance(row, Mapping)
            or row.get("probe_predictions_scored") is not False
            or row.get("evaluation_predictions_scored") is not True
        ):
            raise IntegrityError(f"one-shot score cell {index} probe/evaluation scoring scope drift")
        _validate_score_cell_secondary(row, index=index)


def _validate_scoring_marker(
    marker: Mapping[str, Any],
    *,
    marker_path: str | Path,
    score_path: str | Path,
    execution_artifact_sha256: str,
    prediction_collection_sha256: str,
    prediction_cell_sha256: Sequence[str],
) -> None:
    """Require the exact marker spent before the one-shot truth loader ran."""

    path = Path(marker_path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise IntegrityError("one-shot scoring marker is not a regular file")
    if set(marker) != {"schema", "status", "request", "request_sha256"}:
        raise IntegrityError("one-shot scoring marker schema fields drift")
    if (
        marker.get("schema") != "kbound_cct20_one_shot_score_marker_v1"
        or marker.get("status") != "SPENT_BEFORE_GROUND_TRUTH_LOAD"
    ):
        raise IntegrityError("one-shot scoring marker has the wrong schema/status")
    request = marker.get("request")
    if not isinstance(request, Mapping):
        raise IntegrityError("one-shot scoring marker lacks its request")
    expected_request = {
        "execution_seal_artifact_sha256": require_sha256(
            execution_artifact_sha256, field="execution_seal_artifact_sha256"
        ),
        "prediction_collection_sha256": require_sha256(
            prediction_collection_sha256, field="prediction_collection_sha256"
        ),
        "prediction_cell_sha256": sorted(
            require_sha256(value, field="prediction_cell_sha256") for value in prediction_cell_sha256
        ),
        "output_path": str(Path(score_path).expanduser().resolve()),
        "expected_target_images": EXPECTED_TARGET_IMAGES,
        "label_contract": SCORING_LABEL_CONTRACT,
    }
    if dict(request) != expected_request:
        raise IntegrityError("one-shot scoring marker request differs from the release chain")
    if marker.get("request_sha256") != stable_sha256(expected_request):
        raise IntegrityError("one-shot scoring marker request SHA-256 mismatch")


def _classify_effect(value: float) -> str:
    if not math.isfinite(value):
        raise IntegrityError("adaptation effect is not finite")
    return "helpful" if value > 0.0 else "harmful" if value < 0.0 else "zero"


def _location_effects(score: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    raw_cells = score.get("cells", ())
    keyed = {(int(row["checkpoint_seed"]), str(row["location_id"])): row for row in raw_cells}
    cells_by_sign: dict[str, list[dict[str, Any]]] = {
        "harmful": [],
        "zero": [],
        "helpful": [],
    }
    output = []
    for location in EXPECTED_TARGET_LOCATIONS:
        cells = [keyed[(seed, location)] for seed in EXPECTED_MODEL_SEEDS]
        evaluation_counts = {int(row["n_evaluation_images"]) for row in cells}
        if len(evaluation_counts) != 1:
            raise IntegrityError(f"location {location} evaluation count changes across checkpoints")
        actions = {decision: sum(row["decision"] == decision for row in cells) for decision in DECISIONS}
        signs = dict.fromkeys(("harmful", "zero", "helpful"), 0)
        cell_rows = []
        for row in cells:
            benefit = float(row["adaptation_benefit"])
            sign = _classify_effect(benefit)
            signs[sign] += 1
            item = {
                "checkpoint_seed": int(row["checkpoint_seed"]),
                "decision": str(row["decision"]),
                "adaptation_benefit": benefit,
                "effect_sign": sign,
                "versus_always_adapt": float(row["baseline_regret_minus_kga_regret"]["versus_always_adapt"]),
                "versus_always_freeze": float(row["baseline_regret_minus_kga_regret"]["versus_always_freeze"]),
            }
            cell_rows.append(item)
            cells_by_sign[sign].append({"location_id": location, **item})
        output.append(
            {
                "location_id": location,
                "n_checkpoint_cells": CHECKPOINT_COUNT,
                "n_evaluation_images_per_checkpoint": next(iter(evaluation_counts)),
                "action_counts": actions,
                "effect_counts": signs,
                "mean_adaptation_benefit": math.fsum(row["adaptation_benefit"] for row in cell_rows) / CHECKPOINT_COUNT,
                "mean_versus_always_adapt": math.fsum(row["versus_always_adapt"] for row in cell_rows)
                / CHECKPOINT_COUNT,
                "mean_versus_always_freeze": math.fsum(row["versus_always_freeze"] for row in cell_rows)
                / CHECKPOINT_COUNT,
                "cells": cell_rows,
            }
        )
    for values in cells_by_sign.values():
        values.sort(key=lambda row: (int(row["checkpoint_seed"]), str(row["location_id"])))
    return output, cells_by_sign


def _verdict(inference: Mapping[str, Any]) -> dict[str, Any]:
    checks = inference.get("strong_success_checks", {})
    safe = inference.get("safe_utility", {})
    strong = checks.get("protocol_strong_success") is True
    expanded = checks.get("expanded_empirical_bundle_including_mixed_effects") is True
    safe_pass = safe.get("passes") is True
    if expanded:
        code = "CONFIRMATORY_STRONG_SUCCESS"
        claim = (
            "The prospective CCT-20 result satisfies both simultaneous and exact-test criteria, "
            "passes action-exposure thresholds, and contains both helpful and harmful adaptation cases."
        )
    elif strong:
        code = "CONFIRMATORY_PRIMARY_SUCCESS_MIXED_EFFECTS_MISSING"
        claim = (
            "The prospective CCT-20 primary contrasts pass the locked criteria, but the expanded "
            "mixed helpful/harmful evidence requirement is not met."
        )
    elif safe_pass:
        code = "SAFE_UTILITY_ONLY"
        claim = (
            "The prospective CCT-20 result passes only the locked safe-utility check; it does not "
            "establish the preregistered strong-success claim."
        )
    else:
        code = "NO_CONFIRMATORY_SUCCESS"
        claim = (
            "The prospective CCT-20 result does not satisfy the preregistered strong-success or "
            "safe-utility criteria; the complete result is reported without promotion."
        )
    return {
        "code": code,
        "confirmatory_strong_claim_supported": expanded,
        "primary_confirmatory_claim_supported": strong,
        "protocol_strong_success": strong,
        "expanded_mixed_effects_success": expanded,
        "safe_utility_passes": safe_pass,
        "manuscript_claim": claim,
    }


def _primary_comparisons(inference: Mapping[str, Any]) -> dict[str, Any]:
    bootstrap = inference["paired_two_way_product_bootstrap"]
    if (
        bootstrap.get("replicates") != BOOTSTRAP_REPLICATES
        or bootstrap.get("random_seed") != BOOTSTRAP_SEED
        or bootstrap.get("checkpoint_rows_and_location_columns_resampled_independently") is not True
        or bootstrap.get("comparisons_share_every_resample") is not True
    ):
        raise IntegrityError("paired two-way bootstrap configuration drift")
    results = bootstrap.get("results", {})
    exact = inference.get("exact_nine_location_sign_flip_and_holm", {})
    output = {}
    for name in COMPARISONS:
        interval = results.get(name, {}).get("simultaneous_bonferroni_97_5_ci")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in interval)
            or not all(math.isfinite(float(value)) for value in interval)
            or float(interval[0]) > float(interval[1])
        ):
            raise IntegrityError(f"{name} lacks a finite ordered 97.5% simultaneous interval")
        test = exact.get(name, {})
        if (
            test.get("n_locations") != LOCATION_CLUSTER_COUNT
            or test.get("enumerated_sign_patterns") != 512
            or test.get("alternative") != "mean_contrast_greater_than_zero"
        ):
            raise IntegrityError(f"{name} exact location sign-flip configuration drift")
        raw_p = float(test.get("p_value_one_sided"))
        holm_p = float(test.get("holm_adjusted_p"))
        if not 0.0 <= raw_p <= holm_p <= 1.0:
            raise IntegrityError(f"{name} raw/Holm p-values are invalid")
        output[name] = {
            "comparator": "always_adapt" if name.endswith("adapt") else "always_freeze",
            "point_estimate": float(results[name]["point_estimate"]),
            "pointwise_95_ci": [float(value) for value in results[name]["pointwise_95_ci"]],
            "simultaneous_bonferroni_confidence_level": SIMULTANEOUS_INTERVAL_LEVEL,
            "simultaneous_bonferroni_97_5_ci": [float(value) for value in interval],
            "exact_location_sign_flip_p_one_sided": raw_p,
            "holm_adjusted_p": holm_p,
            "holm_reject_at_familywise_0_05": bool(test["holm_reject_at_familywise_0_05"]),
        }
    return output


def _publication_safety(
    score: Mapping[str, Any],
    exposure: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive false-adapt accounting at the locked checkpoint-location unit."""

    cells = score.get("cells", ())
    if not isinstance(cells, list) or len(cells) != CELL_COUNT:
        raise IntegrityError("false-adapt accounting requires the complete 45-cell score")
    false_adapt_count = sum(
        row.get("decision") == "ADAPT" and float(row.get("adaptation_benefit")) <= 0.0
        for row in cells
    )
    adapt_count = int(exposure["counts"]["ADAPT"])
    if false_adapt_count > adapt_count:
        raise IntegrityError("false-adapt count exceeds adapt exposure")
    return {
        "event": "decision == ADAPT and adaptation_benefit <= 0",
        "unit": "checkpoint_x_location",
        "denominator_all_cells": CELL_COUNT,
        "adapt_count": adapt_count,
        "false_adapt_count": false_adapt_count,
        "false_adapt_rate_unconditional": false_adapt_count / CELL_COUNT,
        "false_adapt_rate_conditional": (
            None if adapt_count == 0 else false_adapt_count / adapt_count
        ),
    }


def _publication_safe_utility(
    inference: Mapping[str, Any],
    comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Expose the pointwise evidence behind the locked safe-utility fallback.

    The upstream inference schema historically calls these point estimates
    ``kga_minus_*`` even though their frozen sign is baseline regret minus KGA
    regret.  Validate those values, then publish unambiguous comparison names
    without changing the sealed inference document.
    """

    raw = inference.get("safe_utility", {})
    margin = raw.get("frozen_noninferiority_margin")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(float(margin)):
        raise IntegrityError("safe-utility noninferiority margin is invalid")
    for comparison, legacy_key in (
        ("versus_always_freeze", "kga_minus_freeze_point_estimate"),
        ("versus_always_adapt", "kga_minus_adapt_point_estimate"),
    ):
        observed = raw.get(legacy_key)
        expected = comparisons[comparison]["point_estimate"]
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isclose(float(observed), float(expected), abs_tol=1e-15)
        ):
            raise IntegrityError(f"safe-utility {comparison} point estimate has sign/value drift")
    if not isinstance(raw.get("passes"), bool):
        raise IntegrityError("safe-utility pass flag is invalid")
    return {
        "contrast_sign": PRIMARY_CONTRAST_SIGN,
        "frozen_noninferiority_margin": float(margin),
        "versus_always_freeze": {
            "point_estimate": comparisons["versus_always_freeze"]["point_estimate"],
            "pointwise_95_ci": comparisons["versus_always_freeze"]["pointwise_95_ci"],
        },
        "versus_always_adapt": {
            "point_estimate": comparisons["versus_always_adapt"]["point_estimate"],
            "pointwise_95_ci": comparisons["versus_always_adapt"]["pointwise_95_ci"],
        },
        "passes": raw["passes"],
    }


def build_release_core(
    *,
    score: Mapping[str, Any],
    inference: Mapping[str, Any],
    upstream_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the paper-facing release payload from already verified inputs."""

    if score.get("benefit_sign") != BENEFIT_SIGN:
        raise IntegrityError("release refuses adaptation-benefit sign drift")
    if score.get("primary_contrast_sign") != PRIMARY_CONTRAST_SIGN:
        raise IntegrityError("release refuses primary-contrast sign drift")
    if inference.get("primary_contrast_sign") != PRIMARY_CONTRAST_SIGN:
        raise IntegrityError("inference/release primary-contrast sign drift")
    design = inference.get("design", {})
    if (
        design.get("checkpoint_seeds") != list(EXPECTED_MODEL_SEEDS)
        or design.get("location_ids") != list(EXPECTED_TARGET_LOCATIONS)
        or design.get("matrix_shape") != [CHECKPOINT_COUNT, LOCATION_CLUSTER_COUNT]
        or design.get("cell_count") != CELL_COUNT
    ):
        raise IntegrityError("inference design is not the locked 5 x 9 panel")
    location_effects, cells_by_sign = _location_effects(score)
    mixed = inference.get("adaptation_effect_mix", {})
    observed_mix = {
        "helpful_cells_strictly_positive": len(cells_by_sign["helpful"]),
        "neutral_cells_exactly_zero": len(cells_by_sign["zero"]),
        "harmful_cells_strictly_negative": len(cells_by_sign["harmful"]),
        "mixed_helpful_and_harmful_present": bool(cells_by_sign["helpful"] and cells_by_sign["harmful"]),
    }
    if mixed != observed_mix:
        raise IntegrityError("inference adaptation-effect mix does not reconcile to scored cells")
    exposure = inference.get("action_exposure_at_checkpoint_location_unit", {})
    counts = exposure.get("counts", {})
    if set(counts) != set(DECISIONS) or sum(int(counts[name]) for name in DECISIONS) != CELL_COUNT:
        raise IntegrityError("inference action exposure does not cover all 45 cells")
    primary_comparisons = _primary_comparisons(inference)
    return {
        "schema": RELEASE_SCHEMA,
        "status": RELEASE_STATUS,
        "artifacts_complete": True,
        "prospective_disclosure": (
            "outcome-unopened before model execution; aggregate target metadata had already been "
            "inspected during candidate ranking, so this is not described as literally label-unopened"
        ),
        "sign_conventions": {
            "adaptation_benefit": BENEFIT_SIGN,
            "primary_contrast": PRIMARY_CONTRAST_SIGN,
            "helpful": "adaptation_benefit > 0",
            "zero": "adaptation_benefit = 0",
            "harmful": "adaptation_benefit < 0",
            "target_selection_lock_nonpositive_boundary": (
                "adaptation_benefit <= 0; the release separates exact zero from strictly harmful"
            ),
        },
        "design": {
            **dict(design),
            "checkpoint_count": CHECKPOINT_COUNT,
            "location_cluster_count": LOCATION_CLUSTER_COUNT,
            "cross_classified_design": "5 independent checkpoints x 9 camera-location clusters",
            "cluster_unit_for_exact_test": "camera_location",
            "target_image_count": int(score["target_image_count"]),
            "probe_image_count": int(score["probe_image_count"]),
            "evaluation_image_count": int(score["evaluation_image_count"]),
            "independent_checkpoint_tensor_identities_verified": True,
        },
        "primary_comparisons": primary_comparisons,
        "action_exposure": dict(exposure),
        "false_adapt_accounting": _publication_safety(score, exposure),
        "adaptation_effect_mix": dict(mixed),
        "adaptation_effect_cells_by_sign": cells_by_sign,
        "location_effects": location_effects,
        "safe_utility": _publication_safe_utility(inference, primary_comparisons),
        "secondary_outcome_reporting": {
            "metric": "16-indicator multilabel macro-F1 with top-1 as a one-hot prediction set",
            "scope": "complete cell-level score artifact",
            "aggregate_claim": None,
            "disclosure": (
                "Cell-level 16-indicator macro-F1 is archived as descriptive secondary evidence; "
                "no post-hoc aggregate or inference claim is made."
            ),
        },
        "strong_success_checks": dict(inference["strong_success_checks"]),
        "verdict": _verdict(inference),
        "upstream_artifacts": dict(upstream_artifacts),
    }


def _tex_number(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise IntegrityError("cannot emit a non-finite TeX number")
    if numeric == 0.0:
        return "0"
    rendered = f"{numeric:.8f}".rstrip("0").rstrip(".")
    if rendered in {"0", "-0"}:
        return f"{numeric:.3e}"
    return rendered


def _tex_text(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def render_numbers_tex(release: Mapping[str, Any]) -> str:
    comparison = release["primary_comparisons"]
    adapt = comparison["versus_always_adapt"]
    freeze = comparison["versus_always_freeze"]
    actions = release["action_exposure"]
    safety = release["false_adapt_accounting"]
    mix = release["adaptation_effect_mix"]
    design = release["design"]
    safe = release["safe_utility"]
    safe_adapt = safe["versus_always_adapt"]
    safe_freeze = safe["versus_always_freeze"]
    verdict = release["verdict"]
    secondary = release["secondary_outcome_reporting"]
    lines = [
        "% AUTO-GENERATED by scripts/build_cct20_release.py. Do not edit by hand.",
        rf"\newcommand{{\CCTCheckpointCount}}{{{design['checkpoint_count']}}}",
        rf"\newcommand{{\CCTLocationCount}}{{{design['location_cluster_count']}}}",
        rf"\newcommand{{\CCTCellCount}}{{{design['cell_count']}}}",
        rf"\newcommand{{\CCTTargetImageCount}}{{{design['target_image_count']}}}",
        rf"\newcommand{{\CCTProbeImageCount}}{{{design['probe_image_count']}}}",
        rf"\newcommand{{\CCTEvaluationImageCount}}{{{design['evaluation_image_count']}}}",
        rf"\newcommand{{\CCTAdaptCount}}{{{actions['counts']['ADAPT']}}}",
        rf"\newcommand{{\CCTFreezeCount}}{{{actions['counts']['FREEZE']}}}",
        rf"\newcommand{{\CCTAbstainCount}}{{{actions['counts']['ABSTAIN']}}}",
        rf"\newcommand{{\CCTAdaptRate}}{{{_tex_number(actions['rates']['ADAPT'])}}}",
        rf"\newcommand{{\CCTFreezeRate}}{{{_tex_number(actions['rates']['FREEZE'])}}}",
        rf"\newcommand{{\CCTAbstainRate}}{{{_tex_number(actions['rates']['ABSTAIN'])}}}",
        rf"\newcommand{{\CCTStrictDecisionCoverage}}{{{_tex_number(actions['strict_decision_coverage'])}}}",
        rf"\newcommand{{\CCTFalseAdaptCount}}{{{safety['false_adapt_count']}}}",
        rf"\newcommand{{\CCTFalseAdaptRate}}{{{_tex_number(safety['false_adapt_rate_unconditional'])}}}",
        rf"\newcommand{{\CCTHelpfulCount}}{{{mix['helpful_cells_strictly_positive']}}}",
        rf"\newcommand{{\CCTZeroCount}}{{{mix['neutral_cells_exactly_zero']}}}",
        rf"\newcommand{{\CCTHarmfulCount}}{{{mix['harmful_cells_strictly_negative']}}}",
        rf"\newcommand{{\CCTVsAdaptPoint}}{{{_tex_number(adapt['point_estimate'])}}}",
        rf"\newcommand{{\CCTVsAdaptCILower}}{{{_tex_number(adapt['simultaneous_bonferroni_97_5_ci'][0])}}}",
        rf"\newcommand{{\CCTVsAdaptCIUpper}}{{{_tex_number(adapt['simultaneous_bonferroni_97_5_ci'][1])}}}",
        rf"\newcommand{{\CCTVsAdaptExactP}}{{{_tex_number(adapt['exact_location_sign_flip_p_one_sided'])}}}",
        rf"\newcommand{{\CCTVsAdaptHolmP}}{{{_tex_number(adapt['holm_adjusted_p'])}}}",
        rf"\newcommand{{\CCTVsFreezePoint}}{{{_tex_number(freeze['point_estimate'])}}}",
        rf"\newcommand{{\CCTVsFreezeCILower}}{{{_tex_number(freeze['simultaneous_bonferroni_97_5_ci'][0])}}}",
        rf"\newcommand{{\CCTVsFreezeCIUpper}}{{{_tex_number(freeze['simultaneous_bonferroni_97_5_ci'][1])}}}",
        rf"\newcommand{{\CCTVsFreezeExactP}}{{{_tex_number(freeze['exact_location_sign_flip_p_one_sided'])}}}",
        rf"\newcommand{{\CCTVsFreezeHolmP}}{{{_tex_number(freeze['holm_adjusted_p'])}}}",
        rf"\newcommand{{\CCTSafeUtilityMargin}}{{{_tex_number(safe['frozen_noninferiority_margin'])}}}",
        rf"\newcommand{{\CCTSafeVsAdaptPoint}}{{{_tex_number(safe_adapt['point_estimate'])}}}",
        rf"\newcommand{{\CCTSafeVsAdaptCILower}}{{{_tex_number(safe_adapt['pointwise_95_ci'][0])}}}",
        rf"\newcommand{{\CCTSafeVsAdaptCIUpper}}{{{_tex_number(safe_adapt['pointwise_95_ci'][1])}}}",
        rf"\newcommand{{\CCTSafeVsFreezePoint}}{{{_tex_number(safe_freeze['point_estimate'])}}}",
        rf"\newcommand{{\CCTSafeVsFreezeCILower}}{{{_tex_number(safe_freeze['pointwise_95_ci'][0])}}}",
        rf"\newcommand{{\CCTSafeVsFreezeCIUpper}}{{{_tex_number(safe_freeze['pointwise_95_ci'][1])}}}",
        rf"\newcommand{{\CCTSafeUtilityPass}}{{\textnormal{{{'yes' if safe['passes'] else 'no'}}}}}",
        rf"\newcommand{{\CCTVerdict}}{{\textnormal{{{_tex_text(verdict['code'].replace('_', ' '))}}}}}",
        rf"\newcommand{{\CCTManuscriptClaim}}{{\textnormal{{{_tex_text(verdict['manuscript_claim'])}}}}}",
        rf"\newcommand{{\CCTSecondaryMetricDisclosure}}{{\textnormal{{{_tex_text(secondary['disclosure'])}}}}}",
        rf"\newcommand{{\CCTInferenceSHA}}{{{release['upstream_artifacts']['two_way_inference']['canonical_document_sha256']}}}",
    ]
    return "\n".join(lines) + "\n"


def render_primary_table_tex(release: Mapping[str, Any]) -> str:
    rows = []
    labels = {
        "versus_always_adapt": "Always adapt",
        "versus_always_freeze": "Always freeze",
    }
    for name in COMPARISONS:
        row = release["primary_comparisons"][name]
        ci = row["simultaneous_bonferroni_97_5_ci"]
        reject = "yes" if row["holm_reject_at_familywise_0_05"] else "no"
        rows.append(
            f"{labels[name]} & {_tex_number(row['point_estimate'])} & "
            f"[{_tex_number(ci[0])}, {_tex_number(ci[1])}] & "
            f"{_tex_number(row['exact_location_sign_flip_p_one_sided'])} & "
            f"{_tex_number(row['holm_adjusted_p'])} & {reject} \\\\"
        )
    return "\n".join(
        [
            "% AUTO-GENERATED by scripts/build_cct20_release.py. Do not edit by hand.",
            "% Contrast is baseline regret minus KGA regret; positive values favor KGA.",
            "% Two 97.5% per-comparison intervals give a 95% simultaneous Bonferroni family.",
            r"\begin{tabular}{@{}lrrrrc@{}}",
            r"\toprule",
            r"Comparator & Baseline regret $-$ KGA regret & Bonferroni 97.5\% CI & Exact $p$ & Holm $p$ & Reject at .05 \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def render_location_effects_tex(release: Mapping[str, Any]) -> str:
    rows = []
    for row in release["location_effects"]:
        action = row["action_counts"]
        effects = row["effect_counts"]
        rows.append(
            f"{_tex_text(str(row['location_id']))} & {row['n_evaluation_images_per_checkpoint']} & "
            f"{action['ADAPT']}/{action['FREEZE']}/{action['ABSTAIN']} & "
            f"{effects['helpful']}/{effects['zero']}/{effects['harmful']} & "
            f"{_tex_number(row['mean_adaptation_benefit'])} & "
            f"{_tex_number(row['mean_versus_always_adapt'])} & "
            f"{_tex_number(row['mean_versus_always_freeze'])} \\\\"
        )
    return "\n".join(
        [
            "% AUTO-GENERATED by scripts/build_cct20_release.py. Do not edit by hand.",
            "% Effect counts are helpful (>0), zero (=0), and harmful (<0).",
            "% Eval. n is per checkpoint; A/F/U and effect counts each use five checkpoint cells.",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"Camera & Eval. $n$/checkpoint & A/F/U (5 cells) & Helpful/zero/harmful (5 cells) & Adapt acc. $-$ freeze acc. & KGA acc. $-$ adapt acc. & KGA acc. $-$ freeze acc. \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise IntegrityError(f"refusing to overwrite immutable release artifact: {path}") from exc
    finally:
        if descriptor is not None:  # pragma: no cover
            os.close(descriptor)


def _rendered_identity(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _immutable_json_payload(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(document),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _check_existing_immutable_payload(path: Path, payload: bytes) -> bool:
    """Return whether an exact immutable file exists; reject every other state."""

    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"release output is not a regular file: {path}")
    try:
        observed = path.read_bytes()
    except OSError as exc:
        raise IntegrityError(f"cannot verify existing release output {path}: {exc}") from exc
    if observed != payload:
        raise IntegrityError(f"existing release output differs from the requested bytes: {path}")
    return True


def _publish_exact_immutable(path: Path, payload: bytes) -> None:
    if not _check_existing_immutable_payload(path, payload):
        _exclusive_write(path, payload)
        return

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise IntegrityError(f"cannot securely open existing release output {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise IntegrityError(f"release output is not a regular file: {path}")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            observed = handle.read()
        if observed != payload:
            raise IntegrityError(
                f"existing release output differs from the requested bytes: {path}"
            )
        os.fchmod(descriptor, 0o444)
        after = os.fstat(descriptor)
        if stat.S_IMODE(after.st_mode) != 0o444:
            raise IntegrityError(f"existing release output was not made immutable: {path}")
        current = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != before.st_dev
            or current.st_ino != before.st_ino
        ):
            raise IntegrityError(f"release output changed while being hardened: {path}")
    except OSError as exc:
        raise IntegrityError(f"cannot harden existing release output {path}: {exc}") from exc
    finally:
        os.close(descriptor)


def _release_refresh_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    projection = copy.deepcopy(dict(document))
    projection.pop("release_sha256", None)
    projection.pop("generated_artifacts", None)
    upstream = projection.get("upstream_artifacts")
    if not isinstance(upstream, dict):
        raise IntegrityError("release manifest lacks upstream_artifacts")
    upstream.pop("release_generator", None)
    return projection


def _verified_existing_release_payloads(
    *,
    manifest_path: Path,
    receipt_path: Path,
    outputs: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise IntegrityError(
            f"existing release manifest is not a regular file: {manifest_path}"
        )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise IntegrityError(
            f"existing release receipt is not a regular file: {receipt_path}"
        )
    verify_artifact_receipt(manifest_path)
    manifest_payload = manifest_path.read_bytes()
    old_document = _load_json_object(manifest_path)
    unsigned = dict(old_document)
    claimed_release_sha = unsigned.pop("release_sha256", None)
    if claimed_release_sha != stable_sha256(unsigned):
        raise IntegrityError("existing release manifest has an invalid release_sha256")
    generated = old_document.get("generated_artifacts")
    if not isinstance(generated, Mapping) or set(generated) != set(outputs):
        raise IntegrityError("existing release manifest has an unexpected generated artifact set")

    payloads = {
        "release_manifest": manifest_payload,
        "release_receipt": receipt_path.read_bytes(),
    }
    for name, expected_path in outputs.items():
        record = generated.get(name)
        if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
            raise IntegrityError(f"existing generated artifact identity is malformed: {name}")
        raw_recorded_path = Path(str(record["path"])).expanduser().absolute()
        if raw_recorded_path.is_symlink():
            raise IntegrityError(f"existing generated artifact is not a regular file: {name}")
        recorded_path = raw_recorded_path.resolve()
        if recorded_path != expected_path.resolve():
            raise IntegrityError(f"existing generated artifact path drift: {name}")
        if recorded_path.is_symlink() or not recorded_path.is_file():
            raise IntegrityError(f"existing generated artifact is not a regular file: {name}")
        payload = recorded_path.read_bytes()
        if (
            isinstance(record["bytes"], bool)
            or not isinstance(record["bytes"], int)
            or record["bytes"] != len(payload)
            or not isinstance(record["sha256"], str)
            or record["sha256"] != hashlib.sha256(payload).hexdigest()
        ):
            raise IntegrityError(f"existing generated artifact commitment mismatch: {name}")
        payloads[name] = payload
    return old_document, payloads


def _archive_release_generation(
    *,
    history_dir: Path,
    payloads: Mapping[str, bytes],
) -> Path:
    raw_history = history_dir.expanduser().absolute()
    parts = raw_history.parts
    if any(Path(*parts[:index]).is_symlink() for index in range(1, len(parts) + 1)):
        raise IntegrityError(f"release history directory traverses a symlink: {history_dir}")
    history = raw_history.resolve()
    manifest_payload = payloads["release_manifest"]
    generation_sha = hashlib.sha256(manifest_payload).hexdigest()
    generation = history / generation_sha
    archive_records: dict[str, dict[str, Any]] = {}
    for role in sorted(payloads):
        payload = payloads[role]
        archived_name = f"{role}.bin"
        _publish_exact_immutable(generation / archived_name, payload)
        archive_records[role] = {
            "bytes": len(payload),
            "path": archived_name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    archive_manifest = _immutable_json_payload(
        {
            "artifacts": archive_records,
            "release_manifest_sha256": generation_sha,
            "schema": "kbound_cct20_superseded_release_generation_v1",
        }
    )
    _publish_exact_immutable(generation / "archive_manifest.json", archive_manifest)
    return generation


def _stage_replacement(path: Path, payload: bytes, *, tag: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.{tag}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
        return temporary
    except BaseException:
        if descriptor >= 0:  # pragma: no cover - defensive descriptor cleanup
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _replace_release_transaction(
    *,
    targets: Mapping[str, Path],
    new_payloads: Mapping[str, bytes],
    old_payloads: Mapping[str, bytes],
) -> None:
    order = [
        "cct20_location_effects_tex",
        "cct20_numbers_tex",
        "cct20_primary_table_tex",
        "release_manifest",
        "release_receipt",
    ]
    staged_new: dict[str, Path] = {}
    staged_old: dict[str, Path] = {}
    replaced: list[str] = []
    try:
        for role in order:
            staged_new[role] = _stage_replacement(
                targets[role],
                new_payloads[role],
                tag="new",
            )
            staged_old[role] = _stage_replacement(
                targets[role],
                old_payloads[role],
                tag="rollback",
            )
        for role in order:
            os.replace(staged_new[role], targets[role])
            replaced.append(role)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for role in reversed(replaced):
            try:
                os.replace(staged_old[role], targets[role])
            except OSError as rollback_exc:  # pragma: no cover - catastrophic I/O
                rollback_errors.append(f"{role}: {rollback_exc}")
        if rollback_errors:
            raise IntegrityError(
                "release refresh failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise IntegrityError(f"release refresh transaction failed: {exc}") from exc
    finally:
        for staged in [*staged_new.values(), *staged_old.values()]:
            staged.unlink(missing_ok=True)


def emit_release(
    release_core: Mapping[str, Any],
    *,
    release_manifest_path: str | Path,
    generated_dir: str | Path,
    refresh_existing: bool = False,
    history_dir: str | Path | None = None,
) -> dict[str, Any]:
    raw_manifest_path = Path(release_manifest_path).expanduser().absolute()
    if raw_manifest_path.is_symlink():
        raise IntegrityError(f"release output is not a regular file: {raw_manifest_path}")
    manifest_path = raw_manifest_path.resolve()
    if manifest_path.name != "cct20_release_manifest.json":
        raise IntegrityError("release manifest must be named cct20_release_manifest.json")
    generated = Path(generated_dir).expanduser().resolve()
    outputs = {
        "cct20_numbers_tex": generated / "cct20_numbers.tex",
        "cct20_primary_table_tex": generated / "cct20_primary_table.tex",
        "cct20_location_effects_tex": generated / "cct20_location_effects.tex",
    }
    payloads = {
        "cct20_numbers_tex": render_numbers_tex(release_core).encode("ascii"),
        "cct20_primary_table_tex": render_primary_table_tex(release_core).encode("ascii"),
        "cct20_location_effects_tex": render_location_effects_tex(release_core).encode("ascii"),
    }
    receipt_path = manifest_path.with_name(manifest_path.name + ".receipt.json")
    manifest_exists = manifest_path.exists() or manifest_path.is_symlink()
    receipt_exists = receipt_path.exists() or receipt_path.is_symlink()
    if manifest_exists != receipt_exists:
        raise IntegrityError(
            "incomplete release artifact/receipt pair; refusing recovery without "
            f"both exact files: {manifest_path}"
        )
    complete_pair_exists = manifest_exists and receipt_exists
    document = {
        **dict(release_core),
        "generated_artifacts": {name: _rendered_identity(outputs[name], payloads[name]) for name in sorted(outputs)},
    }
    document["release_sha256"] = stable_sha256(document)
    manifest_payload = _immutable_json_payload(document)
    receipt = {
        "schema": "kbound_cct20_artifact_receipt_v1",
        "artifact_path": str(manifest_path),
        "artifact_bytes": len(manifest_payload),
        "artifact_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "canonical_document_sha256": stable_sha256(document),
    }
    receipt_payload = _immutable_json_payload(receipt)
    targets = {
        **outputs,
        "release_manifest": manifest_path,
        "release_receipt": receipt_path,
    }
    requested_payloads = {
        **payloads,
        "release_manifest": manifest_payload,
        "release_receipt": receipt_payload,
    }

    # Exact generated files left before the manifest/receipt transaction are
    # safe to resume.  A complete exact pair is idempotently re-hardened, while
    # a partial pair or any byte mismatch fails closed.
    if complete_pair_exists:
        exact_error: IntegrityError | None = None
        try:
            for role, target in targets.items():
                if not _check_existing_immutable_payload(
                    target,
                    requested_payloads[role],
                ):
                    raise IntegrityError(
                        f"complete release pair is missing a generated artifact: {target}"
                    )
        except IntegrityError as exc:
            exact_error = exc
        if exact_error is None:
            for role, target in targets.items():
                _publish_exact_immutable(target, requested_payloads[role])
            verify_artifact_receipt(manifest_path)
            return document
        if not refresh_existing:
            raise exact_error
        if history_dir is None:
            raise IntegrityError("release refresh requires an explicit history_dir")
        old_document, old_payloads = _verified_existing_release_payloads(
            manifest_path=manifest_path,
            receipt_path=receipt_path,
            outputs=outputs,
        )
        if _release_refresh_projection(old_document) != _release_refresh_projection(
            document
        ):
            raise IntegrityError(
                "release refresh would alter receipt-bound scientific content; "
                "only the release-generator binding and rendered outputs may change"
            )
        _archive_release_generation(
            history_dir=Path(history_dir),
            payloads=old_payloads,
        )
        _replace_release_transaction(
            targets=targets,
            new_payloads=requested_payloads,
            old_payloads=old_payloads,
        )
        for role, target in targets.items():
            _publish_exact_immutable(target, requested_payloads[role])
        verify_artifact_receipt(manifest_path)
        return document

    if refresh_existing:
        raise IntegrityError("release refresh requires an existing complete artifact/receipt pair")
    for name in sorted(outputs):
        _check_existing_immutable_payload(outputs[name], payloads[name])
    for name in sorted(outputs):
        _publish_exact_immutable(outputs[name], payloads[name])
    _publish_exact_immutable(manifest_path, manifest_payload)
    _publish_exact_immutable(receipt_path, receipt_payload)
    verify_artifact_receipt(manifest_path)
    return document


def build_release(
    *,
    checkpoint_audit_path: str | Path,
    development_gate_path: str | Path,
    development_collection_path: str | Path,
    execution_seal_path: str | Path,
    prediction_collection_path: str | Path,
    prediction_cell_paths: Sequence[str | Path],
    scoring_marker_path: str | Path,
    score_path: str | Path,
    inference_path: str | Path,
    release_manifest_path: str | Path,
    generated_dir: str | Path,
    refresh_existing: bool = False,
    history_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the complete immutable chain and emit release-only artifacts."""

    checkpoint_audit = _load_json_object(checkpoint_audit_path)
    checkpoint_rows = verify_checkpoint_audit_document(checkpoint_audit)
    gate, gate_receipt = _load_received_json(development_gate_path)
    validate_gate_document(gate)
    development, development_receipt = _load_received_json(development_collection_path)
    validate_development_trace_collection(
        development,
        gate_document=gate,
        checkpoint_audit=checkpoint_audit,
        verify_trace_files=True,
    )
    execution, execution_receipt = _load_received_json(execution_seal_path)
    validate_execution_seal(execution)
    verify_execution_environment(execution)
    if execution.get("checkpoint_audit") != checkpoint_audit:
        raise IntegrityError("execution seal embeds a different checkpoint audit")
    if execution.get("gate_sha256") != gate.get("gate_sha256"):
        raise IntegrityError("execution seal embeds a different development gate")
    dependencies = _dependency_by_name(execution, "code_dependencies")
    _require_dependency_path(dependencies, "checkpoint_audit", checkpoint_audit_path)
    _require_dependency_path(dependencies, "development_gate", development_gate_path)
    _require_dependency_path(
        dependencies,
        "development_gate_receipt",
        Path(development_gate_path)
        .expanduser()
        .resolve()
        .with_name(Path(development_gate_path).name + ".receipt.json"),
    )
    _require_dependency_path(dependencies, "development_trace_collection", development_collection_path)
    _require_dependency_path(
        dependencies,
        "development_trace_collection_receipt",
        Path(development_collection_path)
        .expanduser()
        .resolve()
        .with_name(Path(development_collection_path).name + ".receipt.json"),
    )
    development_trace_bundle = _development_trace_bundle(
        development,
        code_dependencies=dependencies,
    )
    data_dependencies = _dependency_by_name(execution, "dataset_dependencies")
    target_split = str(execution.get("population", {}).get("target_split", ""))
    expected_annotation_basename = f"{target_split}_annotations.json"
    target_manifest_dependency = data_dependencies.get("label_free_target_manifest")
    if not isinstance(target_manifest_dependency, Mapping):
        raise IntegrityError("execution seal lacks the label-free target manifest")
    target_manifest_path = Path(str(target_manifest_dependency.get("path", ""))).expanduser().resolve()
    _require_dependency_path(
        data_dependencies,
        "label_free_target_manifest",
        target_manifest_path,
    )
    target_manifest = _load_json_object(target_manifest_path)
    validate_locked_target_population(target_manifest)
    if (
        target_manifest.get("status") != "LABEL_FREE_POPULATION_VERIFIED"
        or target_manifest.get("target_role") != "trans_test"
        or target_manifest.get("target_annotation_envelope_basename") != expected_annotation_basename
    ):
        raise IntegrityError("label-free target manifest role/status/envelope drift")
    if target_manifest.get("manifest_sha256") != execution.get("population", {}).get("target_manifest_sha256"):
        raise IntegrityError("label-free target manifest differs from the execution population")
    if len({str(row["tensor_sha256"]) for row in checkpoint_rows}) != CHECKPOINT_COUNT:
        raise IntegrityError("checkpoint audit does not contain five independent tensors")

    prediction_collection, prediction_collection_receipt = _load_received_json(prediction_collection_path)
    _validate_prediction_collection(prediction_collection)
    loaded_cells = []
    cell_identities = []
    for raw_path in prediction_cell_paths:
        cell, receipt = _load_received_json(raw_path)
        loaded_cells.append(cell)
        identity = _received_identity(raw_path, receipt)
        identity.update(
            {
                "checkpoint_seed": int(cell.get("checkpoint_seed", -1)),
                "location_id": str(cell.get("location_id", "")),
                "cell_sha256": require_sha256(cell.get("cell_sha256"), field="cell_sha256"),
            }
        )
        cell_identities.append(identity)
    prediction_grid = _prediction_grid(prediction_collection, loaded_cells)
    _replay_prediction_collection(
        prediction_collection,
        prediction_cells=loaded_cells,
        target_manifest=target_manifest,
    )
    prediction_action_bundle = _prediction_action_bundle(loaded_cells)
    execution_artifact_sha256 = execution_receipt["artifact_sha256"]
    checkpoint_tensor_sha256 = {str(row["model_seed"]): row["tensor_sha256"] for row in checkpoint_rows}
    expected_collection_identities = {
        "protocol_seal_sha256": execution_artifact_sha256,
        "gate_sha256": gate["gate_sha256"],
        "target_manifest_sha256": execution["population"]["target_manifest_sha256"],
        "checkpoint_tensor_sha256": checkpoint_tensor_sha256,
    }
    for field, value in expected_collection_identities.items():
        if prediction_collection.get(field) != value:
            raise IntegrityError(f"prediction collection {field} differs from the execution chain")

    scoring_marker = _load_json_object(scoring_marker_path)
    _validate_scoring_marker(
        scoring_marker,
        marker_path=scoring_marker_path,
        score_path=score_path,
        execution_artifact_sha256=execution_artifact_sha256,
        prediction_collection_sha256=prediction_collection["collection_sha256"],
        prediction_cell_sha256=[cell["cell_sha256"] for cell in loaded_cells],
    )
    score, score_receipt = _load_received_json(score_path)
    inference, inference_receipt = _load_received_json(inference_path)
    replayed_inference = _validate_inference_replay(score, inference)
    target_annotation_dependency = data_dependencies.get("target_annotations_json", {})
    if not isinstance(target_annotation_dependency, Mapping):
        raise IntegrityError("execution seal lacks its opaque target-annotation dependency")
    target_annotation_path = Path(str(target_annotation_dependency.get("path", ""))).expanduser().resolve()
    if target_annotation_path.name != expected_annotation_basename:
        raise IntegrityError("opaque target-annotation dependency has the wrong basename")
    _require_dependency_path(
        data_dependencies,
        "target_annotations_json",
        target_annotation_path,
    )
    target_annotation_sha256 = require_sha256(
        target_annotation_dependency.get("sha256"),
        field="target_annotations_json.sha256",
    )
    _validate_score_release_contract(
        score,
        target_annotations_sha256=target_annotation_sha256,
    )
    expected_score_identities = {
        "prediction_collection_sha256": prediction_collection["collection_sha256"],
        "execution_seal_artifact_sha256": execution_artifact_sha256,
        "protocol_seal_sha256": prediction_collection["protocol_seal_sha256"],
        "gate_sha256": gate["gate_sha256"],
        "target_manifest_sha256": prediction_collection["target_manifest_sha256"],
        "checkpoint_count": CHECKPOINT_COUNT,
        "location_count": LOCATION_CLUSTER_COUNT,
        "cell_count": CELL_COUNT,
    }
    for field, value in expected_score_identities.items():
        if score.get(field) != value:
            raise IntegrityError(f"one-shot score {field} differs from the execution chain")
    _reconcile_score_with_predictions(score, prediction_grid)
    if replayed_inference.get("score_sha256") != score.get("score_sha256"):
        raise IntegrityError("inference score identity mismatch")
    if replayed_inference.get("execution_seal_artifact_sha256") != execution_artifact_sha256:
        raise IntegrityError("inference execution-seal identity mismatch")
    if (
        replayed_inference["action_exposure_at_checkpoint_location_unit"]["counts"]
        != prediction_collection["action_counts_at_cell_unit"]
    ):
        raise IntegrityError("inference and prediction collection action counts differ")

    cell_identities.sort(key=lambda row: row["path"])
    execution_dependency_bundle = _execution_dependency_bundle(execution)
    upstream = {
        "checkpoint_audit": _plain_json_identity(checkpoint_audit_path, checkpoint_audit),
        "development_gate": _received_identity(development_gate_path, gate_receipt),
        "development_trace_collection": _received_identity(development_collection_path, development_receipt),
        "execution_seal": _received_identity(execution_seal_path, execution_receipt),
        "prediction_collection": _received_identity(prediction_collection_path, prediction_collection_receipt),
        "one_shot_scoring_marker": _plain_json_identity(scoring_marker_path, scoring_marker),
        "one_shot_score": _received_identity(score_path, score_receipt),
        "two_way_inference": _received_identity(inference_path, inference_receipt),
        "release_generator": _plain_file_identity(__file__),
        "execution_dependencies": execution_dependency_bundle,
        "development_traces": development_trace_bundle,
        "prediction_cells": {
            "count": len(cell_identities),
            "aggregate_sha256": stable_sha256(cell_identities),
            "items": cell_identities,
        },
        "prediction_actions": prediction_action_bundle,
    }
    core = build_release_core(score=score, inference=replayed_inference, upstream_artifacts=upstream)
    return emit_release(
        core,
        release_manifest_path=release_manifest_path,
        generated_dir=generated_dir,
        refresh_existing=refresh_existing,
        history_dir=history_dir,
    )


_REFRESH_UPSTREAM_BINDINGS = {
    "checkpoint_audit_path": "checkpoint_audit",
    "development_gate_path": "development_gate",
    "development_collection_path": "development_trace_collection",
    "execution_seal_path": "execution_seal",
    "prediction_collection_path": "prediction_collection",
    "scoring_marker_path": "one_shot_scoring_marker",
    "score_path": "one_shot_score",
    "inference_path": "two_way_inference",
}


def _committed_release_path(
    record: Any,
    *,
    field: str,
) -> Path:
    if not isinstance(record, Mapping):
        raise IntegrityError(f"existing release manifest lacks {field} identity")
    value = record.get("path")
    if not isinstance(value, str) or not value or "\x00" in value:
        raise IntegrityError(f"existing release manifest has an invalid {field} path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise IntegrityError(
            f"existing release manifest {field} path is not canonical absolute"
        )
    canonical = path.resolve(strict=False)
    if str(canonical) != value:
        raise IntegrityError(
            f"existing release manifest {field} path is not canonical absolute"
        )
    return canonical


def refresh_release_from_existing(
    *,
    release_manifest_path: str | Path,
    generated_dir: str | Path,
    history_dir: str | Path,
) -> dict[str, Any]:
    """Replay only the upstream paths committed by a verified prior release.

    The old manifest/receipt and three rendered outputs must be complete and
    byte-valid before any upstream is opened.  ``build_release`` then replays
    those exact committed paths and permits ``emit_release`` to replace only
    the canonical five-file release set.
    """

    raw_manifest = Path(release_manifest_path).expanduser().absolute()
    if raw_manifest.is_symlink():
        raise IntegrityError(f"release output is not a regular file: {raw_manifest}")
    manifest_path = raw_manifest.resolve()
    if manifest_path.name != "cct20_release_manifest.json":
        raise IntegrityError("release manifest must be named cct20_release_manifest.json")
    generated = Path(generated_dir).expanduser().resolve()
    outputs = {
        "cct20_numbers_tex": generated / "cct20_numbers.tex",
        "cct20_primary_table_tex": generated / "cct20_primary_table.tex",
        "cct20_location_effects_tex": generated / "cct20_location_effects.tex",
    }
    receipt_path = manifest_path.with_name(manifest_path.name + ".receipt.json")
    old_document, _ = _verified_existing_release_payloads(
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        outputs=outputs,
    )
    if (
        old_document.get("schema") != RELEASE_SCHEMA
        or old_document.get("status") != RELEASE_STATUS
    ):
        raise IntegrityError("existing release manifest has the wrong schema or status")
    upstream = old_document.get("upstream_artifacts")
    if not isinstance(upstream, Mapping):
        raise IntegrityError("existing release manifest lacks upstream_artifacts")

    replay_inputs = {
        argument: _committed_release_path(
            upstream.get(identity),
            field=f"upstream_artifacts.{identity}",
        )
        for argument, identity in _REFRESH_UPSTREAM_BINDINGS.items()
    }
    cell_ledger = upstream.get("prediction_cells")
    if not isinstance(cell_ledger, Mapping):
        raise IntegrityError("existing release manifest lacks its prediction-cell ledger")
    count = cell_ledger.get("count")
    items = cell_ledger.get("items")
    if (
        isinstance(count, bool)
        or count != CELL_COUNT
        or not isinstance(items, list)
        or len(items) != CELL_COUNT
    ):
        raise IntegrityError(
            f"existing release prediction-cell ledger is not the exact {CELL_COUNT}-cell set"
        )
    require_sha256(
        cell_ledger.get("aggregate_sha256"),
        field="upstream_artifacts.prediction_cells.aggregate_sha256",
    )
    prediction_cell_paths = [
        _committed_release_path(
            item,
            field=f"prediction-cell[{index}]",
        )
        for index, item in enumerate(items)
    ]
    if len(set(prediction_cell_paths)) != CELL_COUNT:
        raise IntegrityError("existing release prediction-cell ledger has duplicate paths")
    if prediction_cell_paths != sorted(prediction_cell_paths, key=str):
        raise IntegrityError("existing release prediction-cell ledger is not canonically ordered")

    return build_release(
        **replay_inputs,
        prediction_cell_paths=prediction_cell_paths,
        release_manifest_path=manifest_path,
        generated_dir=generated,
        refresh_existing=True,
        history_dir=Path(history_dir).expanduser().resolve(),
    )


def _prediction_paths(args: argparse.Namespace) -> list[Path]:
    if args.prediction_cell:
        paths = [Path(value).expanduser().resolve() for value in args.prediction_cell]
    else:
        directory = args.prediction_cells_dir.expanduser().resolve()
        paths = sorted(directory.glob("seed*_location*_predictions.json"))
    if len(paths) != CELL_COUNT:
        raise IntegrityError(f"exactly 45 prediction-cell paths are required, found {len(paths)}")
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    refresh_requested = "--refresh-from-existing" in raw_arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-audit", type=Path, required=not refresh_requested)
    parser.add_argument("--development-gate", type=Path, required=not refresh_requested)
    parser.add_argument("--development-collection", type=Path, required=not refresh_requested)
    parser.add_argument("--execution-seal", type=Path, required=not refresh_requested)
    parser.add_argument("--prediction-collection", type=Path, required=not refresh_requested)
    cells = parser.add_mutually_exclusive_group(required=not refresh_requested)
    cells.add_argument("--prediction-cells-dir", type=Path)
    cells.add_argument("--prediction-cell", type=Path, action="append")
    parser.add_argument("--scoring-marker", type=Path, required=not refresh_requested)
    parser.add_argument("--score", type=Path, required=not refresh_requested)
    parser.add_argument("--inference", type=Path, required=not refresh_requested)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "paper" / "generated",
    )
    parser.add_argument(
        "--refresh-from-existing",
        action="store_true",
        help=(
            "derive every upstream path from a fully verified prior release and replace "
            "only its canonical five generated outputs"
        ),
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        help="required with --refresh-from-existing; preserves the exact prior release",
    )
    args = parser.parse_args(raw_arguments)
    if args.refresh_from_existing:
        overrides = [
            name
            for name in (
                "checkpoint_audit",
                "development_gate",
                "development_collection",
                "execution_seal",
                "prediction_collection",
                "prediction_cells_dir",
                "prediction_cell",
                "scoring_marker",
                "score",
                "inference",
            )
            if getattr(args, name) is not None
        ]
        if overrides:
            parser.error(
                "--refresh-from-existing does not accept upstream overrides: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in overrides)
            )
        if args.history_dir is None:
            parser.error("--refresh-from-existing requires --history-dir")
        release = refresh_release_from_existing(
            release_manifest_path=args.release_manifest,
            generated_dir=args.generated_dir,
            history_dir=args.history_dir,
        )
    else:
        if args.history_dir is not None:
            parser.error("--history-dir requires --refresh-from-existing")
        release = build_release(
            checkpoint_audit_path=args.checkpoint_audit,
            development_gate_path=args.development_gate,
            development_collection_path=args.development_collection,
            execution_seal_path=args.execution_seal,
            prediction_collection_path=args.prediction_collection,
            prediction_cell_paths=_prediction_paths(args),
            scoring_marker_path=args.scoring_marker,
            score_path=args.score,
            inference_path=args.inference,
            release_manifest_path=args.release_manifest,
            generated_dir=args.generated_dir,
        )
    print(
        f"CCT-20 release complete: verdict={release['verdict']['code']} "
        f"release_sha256={release['release_sha256']} -> {Path(args.release_manifest).resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
