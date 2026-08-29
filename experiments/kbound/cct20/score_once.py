"""One-shot, post-seal scorer for set-valued CCT-20 ground truth.

Nothing in the label-free target path imports this module.  This scorer first
verifies byte receipts for the sealed prediction collection and all 45 shards,
then irreversibly spends a local scoring marker, and only then calls the supplied
ground-truth loader.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from .integrity import IntegrityError, file_sha256, stable_sha256
from .prediction_artifacts import validate_prediction_cell
from .protocol_seal import (
    EXPECTED_CLASS_COUNT,
    EXPECTED_MODEL_SEEDS,
    EXPECTED_TARGET_IMAGES,
    EXPECTED_TARGET_LOCATIONS,
    verify_artifact_receipt,
    verify_execution_environment,
    write_immutable_json_with_receipt,
)

TruthLoader = Callable[[], Mapping[str, Iterable[int]]]

FROZEN_CATEGORY_ID_TO_OUTPUT_INDEX = {
    1: 0,
    3: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    10: 7,
    11: 8,
    16: 9,
    21: 10,
    30: 11,
    33: 12,
    34: 13,
    51: 14,
    99: 15,
}


def cct20_truth_loader(annotation_path: str | Path) -> TruthLoader:
    """Return a lazy loader; construction performs no target-label I/O."""

    path = Path(annotation_path).expanduser().resolve()
    if path.name != "trans_test_annotations.json":
        raise IntegrityError("one-shot CCT-20 truth must come from trans_test_annotations.json")

    def load() -> Mapping[str, Iterable[int]]:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"cannot load one-shot target annotations: {exc}") from exc
        if not isinstance(document, dict):
            raise IntegrityError("target annotation envelope must be a JSON object")
        images = document.get("images")
        annotations = document.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise IntegrityError("target annotation envelope lacks images/annotations arrays")
        result: dict[str, set[int]] = {}
        for index, image in enumerate(images):
            if not isinstance(image, dict) or "id" not in image:
                raise IntegrityError(f"target images[{index}] lacks id")
            image_id = str(image["id"])
            if not image_id or image_id in result:
                raise IntegrityError(f"duplicate/empty target image id {image_id!r}")
            result[image_id] = set()
        for index, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                raise IntegrityError(f"target annotations[{index}] is not an object")
            image_id = str(annotation.get("image_id", ""))
            category_id = annotation.get("category_id")
            if image_id not in result:
                raise IntegrityError(
                    f"target annotation {index} references unknown image {image_id!r}"
                )
            if (
                isinstance(category_id, bool)
                or not isinstance(category_id, int)
                or category_id not in FROZEN_CATEGORY_ID_TO_OUTPUT_INDEX
            ):
                raise IntegrityError(
                    f"target annotation {index} has unknown category_id {category_id!r}"
                )
            result[image_id].add(FROZEN_CATEGORY_ID_TO_OUTPUT_INDEX[int(category_id)])
        return result

    load.kbound_truth_source_path = str(path)
    return load


def _load_sealed_json(path: str | Path) -> dict[str, Any]:
    verify_artifact_receipt(path)
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read sealed JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"sealed artifact must contain a JSON object: {path}")
    return value


def _validate_collection_manifest(collection: Mapping[str, Any]) -> None:
    if collection.get("schema") != "kbound_cct20_label_free_prediction_collection_v1":
        raise IntegrityError("unknown prediction collection schema")
    if collection.get("status") != "SEALED_BEFORE_LABEL_JOIN":
        raise IntegrityError("prediction collection was not sealed before label join")
    unsigned = dict(collection)
    claimed = unsigned.pop("collection_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("prediction collection SHA-256 mismatch")
    if collection.get("checkpoint_count") != 5 or collection.get("location_count") != 9:
        raise IntegrityError("prediction collection is not the locked 5 x 9 design")
    if collection.get("cell_count") != 45 or len(collection.get("cells", ())) != 45:
        raise IntegrityError("prediction collection does not contain 45 cells")
    if collection.get("replayable_probe_features_required") is not True:
        raise IntegrityError("prediction collection lacks mandatory replayable probe features")
    if collection.get("pre_evaluation_action_seals_required") is not True:
        raise IntegrityError("prediction collection lacks pre-evaluation action seals")


def _normalise_truth(
    raw: Mapping[str, Iterable[int]],
    *,
    expected_image_ids: set[str],
) -> dict[str, frozenset[int]]:
    if not isinstance(raw, Mapping):
        raise IntegrityError("ground-truth loader must return image_id -> category-index iterable")
    truth: dict[str, frozenset[int]] = {}
    for raw_image_id, raw_values in raw.items():
        image_id = str(raw_image_id)
        try:
            values = tuple(raw_values)
        except TypeError as exc:
            raise IntegrityError(f"truth for image {image_id!r} is not iterable") from exc
        categories = set()
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise IntegrityError(f"truth category for image {image_id!r} is not an integer")
            if not 0 <= value < EXPECTED_CLASS_COUNT:
                raise IntegrityError(f"unknown truth category index {value} for image {image_id!r}")
            categories.add(value)
        if not categories:
            raise IntegrityError(f"zero-annotation image {image_id!r}; experiment fails closed")
        truth[image_id] = frozenset(categories)
    if set(truth) != expected_image_ids:
        missing = len(expected_image_ids - set(truth))
        extra = len(set(truth) - expected_image_ids)
        raise IntegrityError(f"truth/prediction population mismatch: missing={missing}, extra={extra}")
    return truth


def _membership_accuracy(predictions: Sequence[int], truths: Sequence[frozenset[int]]) -> float:
    if len(predictions) != len(truths) or not predictions:
        raise IntegrityError("accuracy inputs are empty or have different lengths")
    return sum(prediction in truth for prediction, truth in zip(predictions, truths, strict=True)) / len(
        predictions
    )


def _multilabel_macro_f1(
    predictions: Sequence[int], truths: Sequence[frozenset[int]]
) -> dict[str, Any]:
    per_class = []
    for category in range(EXPECTED_CLASS_COUNT):
        tp = fp = fn = 0
        for prediction, truth in zip(predictions, truths, strict=True):
            predicted_positive = prediction == category
            true_positive = category in truth
            tp += int(predicted_positive and true_positive)
            fp += int(predicted_positive and not true_positive)
            fn += int(not predicted_positive and true_positive)
        denominator = 2 * tp + fp + fn
        f1 = 0.0 if denominator == 0 else (2.0 * tp) / denominator
        per_class.append({"output_index": category, "tp": tp, "fp": fp, "fn": fn, "f1": f1})
    return {
        "macro_f1": sum(row["f1"] for row in per_class) / EXPECTED_CLASS_COUNT,
        "n_output_indicators": EXPECTED_CLASS_COUNT,
        "zero_denominator_convention": 0.0,
        "per_class": per_class,
    }


def _score_cell(document: Mapping[str, Any], truth: Mapping[str, frozenset[int]]) -> dict[str, Any]:
    validate_prediction_cell(document)
    all_rows = list(document["rows"])
    rows = [row for row in all_rows if row["role"] == "evaluation"]
    if not rows:
        raise IntegrityError("checkpoint-location cell has no evaluation images")
    truths = [truth[str(row["image_id"])] for row in rows]
    frozen = [int(row["frozen_prediction"]) for row in rows]
    adapted = [int(row["adapted_prediction"]) for row in rows]
    kga = [int(row["kga_prediction"]) for row in rows]
    accuracies = {
        "always_freeze": _membership_accuracy(frozen, truths),
        "always_adapt": _membership_accuracy(adapted, truths),
        "kga": _membership_accuracy(kga, truths),
    }
    oracle = max(accuracies["always_freeze"], accuracies["always_adapt"])
    regrets = {name: oracle - value for name, value in accuracies.items()}
    contrasts = {
        "versus_always_adapt": regrets["always_adapt"] - regrets["kga"],
        "versus_always_freeze": regrets["always_freeze"] - regrets["kga"],
    }
    return {
        "checkpoint_seed": document["checkpoint_seed"],
        "checkpoint_tensor_sha256": document["checkpoint_tensor_sha256"],
        "location_id": document["location_id"],
        "n_target_images": len(all_rows),
        "n_probe_images": sum(row["role"] == "probe" for row in all_rows),
        "n_evaluation_images": len(rows),
        "probe_predictions_scored": False,
        "evaluation_predictions_scored": True,
        "decision": document["gate"]["decision"],
        "set_membership_top1_accuracy": accuracies,
        "adaptation_benefit": accuracies["always_adapt"] - accuracies["always_freeze"],
        "oracle_fixed_action_accuracy": oracle,
        "regret_to_better_fixed_action": regrets,
        "baseline_regret_minus_kga_regret": contrasts,
        "multilabel_macro_f1": {
            "always_freeze": _multilabel_macro_f1(frozen, truths),
            "always_adapt": _multilabel_macro_f1(adapted, truths),
            "kga": _multilabel_macro_f1(kga, truths),
        },
    }


def _spend_marker(path: Path, request: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "schema": "kbound_cct20_one_shot_score_marker_v1",
            "status": "SPENT_BEFORE_GROUND_TRUTH_LOAD",
            "request": dict(request),
            "request_sha256": stable_sha256(dict(request)),
        },
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    descriptor = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise IntegrityError(f"one-shot scoring marker is already spent: {path}") from exc
    finally:
        if descriptor is not None:  # pragma: no cover
            os.close(descriptor)


def score_once(
    *,
    execution_seal_path: str | Path,
    prediction_collection_path: str | Path,
    prediction_cell_paths: Sequence[str | Path],
    truth_loader: TruthLoader,
    output_path: str | Path,
    spent_marker_path: str | Path,
    expected_target_images: int = EXPECTED_TARGET_IMAGES,
) -> dict[str, Any]:
    """Verify seals, spend the one-shot marker, then join set-valued truth."""

    output = Path(output_path).expanduser().resolve()
    marker = Path(spent_marker_path).expanduser().resolve()
    if output.exists() or output.with_name(output.name + ".receipt.json").exists():
        raise IntegrityError(f"refusing to overwrite scoring output: {output}")
    execution_receipt = verify_artifact_receipt(execution_seal_path)
    execution_seal = _load_sealed_json(execution_seal_path)
    verify_execution_environment(execution_seal)
    collection = _load_sealed_json(prediction_collection_path)
    _validate_collection_manifest(collection)
    execution_artifact_sha256 = execution_receipt["artifact_sha256"]
    if collection.get("protocol_seal_sha256") != execution_artifact_sha256:
        raise IntegrityError("prediction collection is not bound to this execution-seal artifact")
    if collection.get("gate_sha256") != execution_seal.get("gate_sha256"):
        raise IntegrityError("prediction collection gate differs from the execution seal")
    if collection.get("target_manifest_sha256") != execution_seal.get("population", {}).get(
        "target_manifest_sha256"
    ):
        raise IntegrityError("prediction collection target manifest differs from the execution seal")
    sealed_checkpoint_hashes = {
        str(row["model_seed"]): row["tensor_sha256"]
        for row in execution_seal.get("checkpoints", ())
    }
    if collection.get("checkpoint_tensor_sha256") != sealed_checkpoint_hashes:
        raise IntegrityError("prediction collection checkpoint tensors differ from the execution seal")
    annotation_dependencies = [
        row
        for row in execution_seal.get("dataset_dependencies", ())
        if row.get("name") == "target_annotations_json"
    ]
    if len(annotation_dependencies) != 1:
        raise IntegrityError(
            "execution seal must contain exactly one target_annotations_json dependency"
        )
    truth_source = getattr(truth_loader, "kbound_truth_source_path", None)
    if not isinstance(truth_source, str) or not truth_source:
        raise IntegrityError("truth loader lacks its sealed source-path identity")
    if str(Path(truth_source).expanduser().resolve()) != annotation_dependencies[0]["path"]:
        raise IntegrityError("truth loader source differs from sealed target_annotations_json")
    if collection.get("target_image_count") != expected_target_images:
        raise IntegrityError(
            f"locked target count is {expected_target_images}, "
            f"collection has {collection.get('target_image_count')}"
        )
    cells = [_load_sealed_json(path) for path in prediction_cell_paths]
    if len(cells) != 45:
        raise IntegrityError(f"scorer requires exactly 45 prediction cells, found {len(cells)}")
    by_hash = {cell.get("cell_sha256"): cell for cell in cells}
    expected_hashes = {row.get("cell_sha256") for row in collection["cells"]}
    if len(by_hash) != 45 or set(by_hash) != expected_hashes:
        raise IntegrityError("prediction shard hashes do not match the sealed collection")
    summaries_by_hash = {row.get("cell_sha256"): row for row in collection["cells"]}
    if len(summaries_by_hash) != 45:
        raise IntegrityError("prediction collection has duplicate cell summaries")
    observed_action_counts = {"ADAPT": 0, "FREEZE": 0, "ABSTAIN": 0}
    observed_cell_keys: set[tuple[int, str]] = set()
    for cell_hash, cell in by_hash.items():
        validate_prediction_cell(cell)
        for field in (
            "protocol_seal_sha256",
            "gate_sha256",
            "target_manifest_sha256",
        ):
            if cell.get(field) != collection.get(field):
                raise IntegrityError(f"prediction cell {field} differs from its collection")
        seed = int(cell["checkpoint_seed"])
        location = str(cell["location_id"])
        key = (seed, location)
        if key in observed_cell_keys:
            raise IntegrityError(f"duplicate checkpoint-location prediction cell: {key}")
        observed_cell_keys.add(key)
        if cell.get("checkpoint_tensor_sha256") != sealed_checkpoint_hashes[str(seed)]:
            raise IntegrityError(
                "prediction cell checkpoint tensor differs from the execution seal"
            )
        summary = summaries_by_hash[cell_hash]
        expected_summary = {
            "checkpoint_seed": cell.get("checkpoint_seed"),
            "location_id": cell.get("location_id"),
            "n_images": cell.get("n_images"),
            "decision": cell.get("gate", {}).get("decision"),
            "cell_sha256": cell_hash,
        }
        if summary != expected_summary:
            raise IntegrityError("prediction collection summary differs from its cell shard")
        observed_action_counts[expected_summary["decision"]] += 1
    if collection.get("action_counts_at_cell_unit") != observed_action_counts:
        raise IntegrityError("prediction collection action counts do not reconcile to cells")
    expected_cell_keys = {
        (seed, location)
        for seed in EXPECTED_MODEL_SEEDS
        for location in EXPECTED_TARGET_LOCATIONS
    }
    if observed_cell_keys != expected_cell_keys:
        raise IntegrityError("prediction shards do not form the exact sealed 5 x 9 grid")
    expected_ids_by_seed: dict[int, set[str]] = {seed: set() for seed in EXPECTED_MODEL_SEEDS}
    for cell in cells:
        seed = int(cell["checkpoint_seed"])
        for row in cell["rows"]:
            image_id = str(row["image_id"])
            if image_id in expected_ids_by_seed[seed]:
                raise IntegrityError(f"duplicate prediction for seed {seed}, image {image_id!r}")
            expected_ids_by_seed[seed].add(image_id)
    population = expected_ids_by_seed[EXPECTED_MODEL_SEEDS[0]]
    if len(population) != expected_target_images:
        raise IntegrityError("prediction shard population count mismatch")
    if any(ids != population for ids in expected_ids_by_seed.values()):
        raise IntegrityError("five checkpoints do not predict the identical target population")
    evaluation_population = {
        str(row["image_id"])
        for cell in cells
        if int(cell["checkpoint_seed"]) == EXPECTED_MODEL_SEEDS[0]
        for row in cell["rows"]
        if row["role"] == "evaluation"
    }
    if not evaluation_population or not evaluation_population < population:
        raise IntegrityError("locked target partition must contain non-empty probe and evaluation sets")

    request = {
        "execution_seal_artifact_sha256": execution_artifact_sha256,
        "prediction_collection_sha256": collection["collection_sha256"],
        "prediction_cell_sha256": sorted(expected_hashes),
        "output_path": str(output),
        "expected_target_images": expected_target_images,
        "label_contract": "set_membership_top1_and_16_indicator_multilabel_macro_f1",
    }
    _spend_marker(marker, request)
    raw_truth = truth_loader()  # The first and only ground-truth access point.
    if file_sha256(truth_source) != annotation_dependencies[0]["sha256"]:
        raise IntegrityError("sealed target annotation file changed during one-shot scoring")
    truth = _normalise_truth(raw_truth, expected_image_ids=population)
    scored_cells = [_score_cell(cell, truth) for cell in cells]
    scored_cells.sort(key=lambda row: (int(row["checkpoint_seed"]), str(row["location_id"])))
    document = {
        "schema": "kbound_cct20_set_valued_score_v1",
        "status": "ALL_LOCKED_CELLS_SCORED",
        "prediction_collection_sha256": collection["collection_sha256"],
        "execution_seal_artifact_sha256": execution_artifact_sha256,
        "target_annotations_file_sha256": annotation_dependencies[0]["sha256"],
        "protocol_seal_sha256": collection["protocol_seal_sha256"],
        "gate_sha256": collection["gate_sha256"],
        "target_manifest_sha256": collection["target_manifest_sha256"],
        "target_image_count": len(population),
        "evaluation_image_count": len(evaluation_population),
        "evaluation_prediction_row_count": len(evaluation_population) * len(EXPECTED_MODEL_SEEDS),
        "probe_image_count": len(population - evaluation_population),
        "checkpoint_count": 5,
        "location_count": 9,
        "cell_count": 45,
        "label_contract": {
            "primary": "top1_correct_iff_prediction_in_complete_distinct_category_set",
            "secondary": "macro_f1_over_all_16_indicators_with_top1_as_one_hot_prediction",
            "repeated_same_category": "collapsed",
            "zero_annotation_image": "experiment_failure",
        },
        "benefit_sign": "adapted_accuracy_minus_frozen_accuracy",
        "primary_contrast_sign": "baseline_regret_minus_kga_regret; positive_favors_kga",
        "truth_join_sha256": stable_sha256(
            [(image_id, sorted(truth[image_id])) for image_id in sorted(truth)]
        ),
        "cells": scored_cells,
    }
    document["score_sha256"] = stable_sha256(document)
    write_immutable_json_with_receipt(output, document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-seal", type=Path, required=True)
    parser.add_argument("--prediction-collection", type=Path, required=True)
    parser.add_argument(
        "--prediction-cell",
        type=Path,
        action="append",
        required=True,
        help="repeat exactly 45 times",
    )
    parser.add_argument("--target-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spent-marker", type=Path, required=True)
    args = parser.parse_args()
    result = score_once(
        execution_seal_path=args.execution_seal,
        prediction_collection_path=args.prediction_collection,
        prediction_cell_paths=args.prediction_cell,
        truth_loader=cct20_truth_loader(args.target_annotations),
        output_path=args.output,
        spent_marker_path=args.spent_marker,
    )
    print(
        f"one-shot scoring complete: {result['score_sha256']} -> {args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
