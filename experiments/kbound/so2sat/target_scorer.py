"""Offline outcome scorer for a complete sealed So2Sat target bundle.

This module is intentionally separate from ``target_runner``.  It validates
the master bundle, all 50 cell receipts, all upstream seals, and both opaque
target-container byte identities *before* it opens ``testing.h5/label`` once.
It never opens ``validation.h5`` as HDF5, and therefore never reads or scores
probe labels.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import shutil
from collections import Counter
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .development import load_gate_authorization_with_receipt
from .gate import CHECKPOINT_IDS, load_gate_with_receipt
from .integrity import (
    IntegrityError,
    file_sha256,
    stable_sha256,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)
from .metadata_manifest import validate_population_manifest
from .precalibration_seal import (
    load_precalibration_seal_with_receipt,
    validate_reveal_registry_directory,
)
from .target_amendment import load_target_boundary_amendment
from .target_contract import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    INFERENCE_ALPHA,
    PRODUCTION_MODE,
    TEST_ONLY_MODE,
    load_complete_target_bundle,
    load_source_postrun_acceptance_pair,
    target_scorer_code_identity,
    target_scorer_environment_identity,
    validate_checkpoint_collection,
    validate_execution_seal,
    validate_selected_candidate,
)
from .target_contract import (
    artifact_binding as _artifact_binding,
)
from .target_contract import (
    load_receipted_document as _load_verified,
)
from .target_contract import (
    selected_candidate_view as _selected_candidate_view,
)

SCORE_SCHEMA = "kbound_so2sat_offline_target_score_v1"
REVEAL_SCHEMA = "kbound_so2sat_single_outcome_reveal_authorization_v1"
OUTPUT_RESERVATION_SCHEMA = "kbound_so2sat_offline_score_output_reservation_v1"
MINIMUM_OUTPUT_FREE_BYTES = 64 * 1024 * 1024
LABEL_DATASET_NAME = "label"
ONE_HOT_ATOL = 0.0

H5Factory = Callable[[Path], AbstractContextManager[Any]]


def _default_h5_factory(path: Path) -> AbstractContextManager[Any]:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("So2Sat offline scoring requires h5py") from exc
    return h5py.File(path, "r")


_CANONICAL_H5_FACTORY = _default_h5_factory
_CANONICAL_MANIFEST_VALIDATOR = validate_population_manifest


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _verify_target_container_paths(
    paths: Mapping[str, str | Path], identities: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Path], dict[str, tuple[int, int, int, int]]]:
    if set(paths) != {"validation", "testing"}:
        raise IntegrityError("offline scorer requires validation and testing container paths")
    resolved: dict[str, Path] = {}
    signatures: dict[str, tuple[int, int, int, int]] = {}
    for split in ("validation", "testing"):
        path = Path(paths[split]).expanduser().resolve()
        identity = identities[split]
        if path.name != identity["basename"] or not path.is_file():
            raise IntegrityError(f"missing or misnamed sealed {split} target container")
        if path.stat().st_size != identity["bytes"]:
            raise IntegrityError(f"{split} target container byte count changed")
        if file_sha256(path) != identity["sha256"]:
            raise IntegrityError(f"{split} target container SHA-256 changed")
        resolved[split] = path
        signatures[split] = _stat_signature(path)
    return resolved, signatures


def _read_testing_outcomes_once(
    testing_path: Path,
    *,
    expected_rows: int,
    h5_factory: H5Factory | None,
) -> np.ndarray:
    factory = _CANONICAL_H5_FACTORY if h5_factory is None else h5_factory
    with factory(testing_path) as handle:
        # Do not enumerate the co-located target container.  One literal
        # outcome name is requested once, after every precondition is sealed.
        dataset = handle[LABEL_DATASET_NAME]
        if getattr(dataset, "shape", None) != (expected_rows, 17):
            raise IntegrityError(
                f"testing.h5/label shape drift: expected {(expected_rows, 17)}, "
                f"found {getattr(dataset, 'shape', None)}"
            )
        values = dataset[:]
    try:
        labels = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError("testing outcomes must be a numeric 17-class matrix") from exc
    if labels.shape != (expected_rows, 17) or not np.isfinite(labels).all():
        raise IntegrityError("testing outcomes have invalid shape or non-finite values")
    if not np.logical_or(labels == 0.0, labels == 1.0).all():
        raise IntegrityError("testing outcomes are not exact zero/one values")
    if not np.all(labels.sum(axis=1) == 1.0):
        raise IntegrityError("testing outcomes are not one-hot over 17 classes")
    return labels.argmax(axis=1).astype(np.int64)


def _accuracy(predictions: Any, truth: np.ndarray) -> float:
    predicted = np.asarray(predictions, dtype=np.int64)
    if predicted.shape != truth.shape or predicted.size < 1:
        raise IntegrityError("prediction/truth vectors are empty or misaligned")
    value = float(np.mean(predicted == truth))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise IntegrityError("computed target accuracy is invalid")
    return value


def _crossed_bootstrap_interval(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.shape != (10, 5) or not np.isfinite(matrix).all():
        raise IntegrityError("crossed inference matrix must be finite 10 cities x 5 checkpoints")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    city_draws = generator.integers(0, 10, size=(BOOTSTRAP_REPLICATES, 10))
    checkpoint_draws = generator.integers(0, 5, size=(BOOTSTRAP_REPLICATES, 5))
    replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    # Looping over 20k tiny 10x5 matrices avoids a large advanced-index tensor.
    for index in range(BOOTSTRAP_REPLICATES):
        replicates[index] = float(
            matrix[np.ix_(city_draws[index], checkpoint_draws[index])].mean()
        )
    lower, upper = np.quantile(
        replicates,
        [INFERENCE_ALPHA / 2.0, 1.0 - INFERENCE_ALPHA / 2.0],
        method="linear",
    )
    return {
        "method": "paired_two_way_city_by_checkpoint_resampling_with_replacement",
        "city_cluster_count": 10,
        "crossed_checkpoint_count": 5,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 1.0 - INFERENCE_ALPHA,
        "lower": float(lower),
        "upper": float(upper),
    }


def _exact_sign_flip(city_means: np.ndarray) -> dict[str, Any]:
    if city_means.shape != (10,) or not np.isfinite(city_means).all():
        raise IntegrityError("sign-flip input must contain ten finite city means")
    observed = float(city_means.mean())
    permuted = np.empty(2**10, dtype=np.float64)
    for index, signs in enumerate(itertools.product((-1.0, 1.0), repeat=10)):
        permuted[index] = float(np.mean(city_means * np.asarray(signs)))
    tolerance = 1.0e-15
    two_sided = float(np.mean(np.abs(permuted) >= abs(observed) - tolerance))
    positive_one_sided = float(np.mean(permuted >= observed - tolerance))
    negative_one_sided = float(np.mean(permuted <= observed + tolerance))
    return {
        "method": "exact_sign_flip_of_10_city_means_under_joint_sign_symmetry",
        "assumption": (
            "Joint target-city effect-sign symmetry under the null; this is not a "
            "design-based randomization test because the sealed gate is deterministic."
        ),
        "permutations": int(permuted.size),
        "observed_city_mean_effect": observed,
        "two_sided_p_value": two_sided,
        "positive_one_sided_p_value_descriptive": positive_one_sided,
        "negative_one_sided_p_value_descriptive": negative_one_sided,
        "multiplicity_input": "two_sided_p_value",
    }


def _holm_adjust(p_values: Mapping[str, float], *, alpha: float) -> dict[str, dict[str, Any]]:
    if set(p_values) != {"always_adapt", "always_freeze"}:
        raise IntegrityError("Holm family must contain exactly the two fixed policies")
    ordered = sorted(p_values.items(), key=lambda row: (row[1], row[0]))
    running = 0.0
    adjusted: dict[str, float] = {}
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered, start=1):
        if not 0.0 <= value <= 1.0:
            raise IntegrityError("Holm input p-value is outside [0, 1]")
        running = max(running, min(1.0, (count - rank + 1) * value))
        adjusted[name] = running
    return {
        name: {
            "raw_two_sided_p_value": float(p_values[name]),
            "holm_adjusted_p_value": float(adjusted[name]),
            "holm_reject_at_0_05": bool(adjusted[name] <= alpha),
        }
        for name in sorted(p_values)
    }


def _score_cells(
    cells: list[dict[str, Any]], truth: np.ndarray, cities: list[str]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    metrics: list[dict[str, Any]] = []
    effects = {
        "always_adapt": np.empty((10, 5), dtype=np.float64),
        "always_freeze": np.empty((10, 5), dtype=np.float64),
    }
    city_index = {city: index for index, city in enumerate(cities)}
    checkpoint_index = {checkpoint: index for index, checkpoint in enumerate(CHECKPOINT_IDS)}
    for cell in cells:
        indices = np.asarray(cell["evaluation"]["row_indices"], dtype=np.int64)
        if indices.size < 1 or indices.min() < 0 or indices.max() >= truth.size:
            raise IntegrityError("target cell outcome indices are out of range")
        cell_truth = truth[indices]
        frozen = cell["evaluation"]["frozen_prediction_class_ids"]
        adapted = cell["evaluation"]["adapted_prediction_class_ids"]
        frozen_accuracy = _accuracy(frozen, cell_truth)
        adapted_accuracy = _accuracy(adapted, cell_truth)
        realized = cell["action"]["realized_action"]
        kga_accuracy = adapted_accuracy if realized == "ADAPT" else frozen_accuracy
        oracle_accuracy = max(frozen_accuracy, adapted_accuracy)
        kga_regret = oracle_accuracy - kga_accuracy
        freeze_regret = oracle_accuracy - frozen_accuracy
        adapt_regret = oracle_accuracy - adapted_accuracy
        effect_vs_freeze = freeze_regret - kga_regret
        effect_vs_adapt = adapt_regret - kga_regret
        adaptation_benefit = adapted_accuracy - frozen_accuracy
        if math.isclose(adaptation_benefit, 0.0, abs_tol=1.0e-15):
            adaptation_direction = "TIE"
        elif adaptation_benefit > 0.0:
            adaptation_direction = "HELPFUL"
        else:
            adaptation_direction = "HARMFUL"
        # Numerically identical identities are recorded and checked explicitly.
        if not math.isclose(effect_vs_freeze, kga_accuracy - frozen_accuracy, abs_tol=1e-15):
            raise IntegrityError("KGA-vs-freeze regret polarity identity failed")
        if not math.isclose(effect_vs_adapt, kga_accuracy - adapted_accuracy, abs_tol=1e-15):
            raise IntegrityError("KGA-vs-adapt regret polarity identity failed")
        i = city_index[cell["city_id"]]
        j = checkpoint_index[cell["checkpoint_id"]]
        effects["always_freeze"][i, j] = effect_vs_freeze
        effects["always_adapt"][i, j] = effect_vs_adapt
        metrics.append(
            {
                "city_id": cell["city_id"],
                "checkpoint_id": cell["checkpoint_id"],
                "evaluation_samples": int(indices.size),
                "decision": cell["action"]["decision"],
                "realized_action": realized,
                "frozen_accuracy": frozen_accuracy,
                "adapted_accuracy": adapted_accuracy,
                "kga_accuracy": kga_accuracy,
                "oracle_accuracy": oracle_accuracy,
                "frozen_regret": freeze_regret,
                "adapted_regret": adapt_regret,
                "kga_regret": kga_regret,
                "adaptation_benefit": adaptation_benefit,
                "adaptation_direction": adaptation_direction,
                "fixed_freeze_regret_minus_kga_regret": effect_vs_freeze,
                "fixed_adapt_regret_minus_kga_regret": effect_vs_adapt,
                "positive_effect_favors": "KGA",
            }
        )
    metrics.sort(key=lambda row: (row["city_id"], row["checkpoint_id"]))
    return metrics, effects


def _exposure(cell_metrics: list[dict[str, Any]], inference_contract: Mapping[str, Any]) -> dict[str, Any]:
    decisions = Counter(row["decision"] for row in cell_metrics)
    realized = Counter(row["realized_action"] for row in cell_metrics)
    cities_by_action = {
        action: sorted(
            {row["city_id"] for row in cell_metrics if row["realized_action"] == action}
        )
        for action in ("ADAPT", "FREEZE")
    }
    minimum_fraction = float(inference_contract["minimum_realized_action_cell_fraction"])
    minimum_cities = int(inference_contract["minimum_realized_action_city_count"])
    realized_checks = {
        action: {
            "cell_count": realized[action],
            "cell_fraction": realized[action] / 50.0,
            "city_count": len(cities_by_action[action]),
            "cities": cities_by_action[action],
            "meets_sealed_minimum": bool(
                realized[action] / 50.0 >= minimum_fraction
                and len(cities_by_action[action]) >= minimum_cities
            ),
        }
        for action in ("ADAPT", "FREEZE")
    }
    direct_minimum_fraction = float(
        inference_contract["minimum_direct_decision_cell_fraction"]
    )
    direct_minimum_cities = int(inference_contract["minimum_direct_decision_city_count"])
    direct_checks = {
        decision: {
            "cell_count": decisions[decision],
            "cell_fraction": decisions[decision] / 50.0,
            "city_count": len(
                {
                    row["city_id"]
                    for row in cell_metrics
                    if row["decision"] == decision
                }
            ),
            "cities": sorted(
                {
                    row["city_id"]
                    for row in cell_metrics
                    if row["decision"] == decision
                }
            ),
            "meets_sealed_minimum": bool(
                decisions[decision] / 50.0 >= direct_minimum_fraction
                and len(
                    {
                        row["city_id"]
                        for row in cell_metrics
                        if row["decision"] == decision
                    }
                )
                >= direct_minimum_cities
            ),
        }
        for decision in ("ADAPT", "FREEZE")
    }
    adaptation_directions = Counter(row["adaptation_direction"] for row in cell_metrics)
    adaptation_direction_cities = {
        direction: sorted(
            {
                row["city_id"]
                for row in cell_metrics
                if row["adaptation_direction"] == direction
            }
        )
        for direction in ("HELPFUL", "HARMFUL", "TIE")
    }
    return {
        "decision_counts": {name: decisions[name] for name in ("ADAPT", "FREEZE", "ABSTAIN")},
        "realized_action_counts": {name: realized[name] for name in ("ADAPT", "FREEZE")},
        "abstain_realized_as_freeze": all(
            row["realized_action"] == "FREEZE"
            for row in cell_metrics
            if row["decision"] == "ABSTAIN"
        ),
        "sealed_minimum_cell_fraction": minimum_fraction,
        "sealed_minimum_city_count": minimum_cities,
        "by_realized_action": realized_checks,
        "both_actions_meaningfully_exposed": all(
            row["meets_sealed_minimum"] for row in realized_checks.values()
        ),
        "sealed_direct_decision_minimum_cell_fraction": direct_minimum_fraction,
        "sealed_direct_decision_minimum_city_count": direct_minimum_cities,
        "by_direct_decision": direct_checks,
        "both_direct_adapt_and_freeze_meaningfully_exposed": all(
            row["meets_sealed_minimum"] for row in direct_checks.values()
        ),
        "adaptation_outcome_direction_counts": {
            direction: adaptation_directions[direction]
            for direction in ("HELPFUL", "HARMFUL", "TIE")
        },
        "adaptation_outcome_direction_cities": adaptation_direction_cities,
        "both_helpful_and_harmful_adaptation_observed": bool(
            adaptation_directions["HELPFUL"] > 0
            and adaptation_directions["HARMFUL"] > 0
        ),
    }


def _score_sealed_target_bundle_core(
    *,
    target_bundle_path: str | Path,
    execution_seal_path: str | Path,
    population_manifest_path: str | Path,
    source_postrun_acceptance_path: str | Path,
    selected_candidate_path: str | Path,
    selected_gate_fit_bundle_path: str | Path,
    selected_gate_cal_bundle_path: str | Path,
    precalibration_seal_path: str | Path,
    gate_path: str | Path,
    gate_authorization_path: str | Path,
    target_boundary_amendment_path: str | Path,
    checkpoint_collection_path: str | Path,
    checkpoint_dir: str | Path,
    reveal_registry_dir: str | Path,
    target_data_paths: Mapping[str, str | Path],
    output_path: str | Path,
    expected_execution_mode: str,
    h5_factory: H5Factory | None = None,
    population_manifest_validator: Callable[[Mapping[str, Any]], None] = validate_population_manifest,
) -> Path:
    """Verify the frozen evidence, reveal testing outcomes once, and score all cells."""

    if expected_execution_mode == PRODUCTION_MODE and (
        h5_factory is not None
        or population_manifest_validator is not _CANONICAL_MANIFEST_VALIDATOR
    ):
        raise IntegrityError("production scorer rejects injected factories or validators")
    if expected_execution_mode not in {PRODUCTION_MODE, TEST_ONLY_MODE}:
        raise IntegrityError("offline scorer execution mode is invalid")

    # Phase A: every condition below is checked before the outcome HDF5 handle
    # is constructed.  Keep this ordering explicit and covered by tests.
    manifest, manifest_receipt = _load_verified(population_manifest_path)
    population_manifest_validator(manifest)
    source_acceptance, _, source_acceptance_binding = (
        load_source_postrun_acceptance_pair(
            source_postrun_acceptance_path,
            strict_document=expected_execution_mode == PRODUCTION_MODE,
        )
    )
    gate = load_gate_with_receipt(gate_path)
    gate_receipt = verify_artifact_receipt(gate_path)
    binding = gate["study_binding"]
    if (
        manifest.get("manifest_sha256") != binding["manifest_sha256"]
        or manifest.get("population_identity_sha256") != binding["population_identity_sha256"]
        or _artifact_binding(manifest_receipt)["artifact_sha256"]
        != binding["manifest_artifact_sha256"]
        or _artifact_binding(manifest_receipt)["canonical_document_sha256"]
        != binding["manifest_canonical_document_sha256"]
    ):
        raise IntegrityError("scoring manifest differs from the calibrated gate")
    selection, selected_receipt = _load_verified(selected_candidate_path)
    validate_selected_candidate(selection, study_binding=binding)
    if selection["source_postrun_acceptance"] != source_acceptance_binding:
        raise IntegrityError("offline scorer selection binds another source acceptance")
    selected = _selected_candidate_view(selection)
    fit_bundle, fit_bundle_receipt = _load_verified(selected_gate_fit_bundle_path)
    gate_authorization, authorized_selection, authorized_gate = (
        load_gate_authorization_with_receipt(
            gate_authorization_path,
            selection_path=selected_candidate_path,
            gate_path=gate_path,
            population_manifest_path=population_manifest_path,
            fit_bundle_path=selected_gate_fit_bundle_path,
            calibration_bundle_path=selected_gate_cal_bundle_path,
        )
    )
    gate_authorization_receipt = verify_artifact_receipt(gate_authorization_path)
    if authorized_selection != selection or authorized_gate != gate:
        raise IntegrityError("gate authorization loader returned a different selection or gate")
    amendment, amendment_receipt = load_target_boundary_amendment(
        target_boundary_amendment_path
    )
    collection, collection_receipt = _load_verified(checkpoint_collection_path)
    validate_checkpoint_collection(
        collection,
        collection_receipt=collection_receipt,
        collection_path=checkpoint_collection_path,
        checkpoint_dir=checkpoint_dir,
    )
    if (
        gate_authorization["checkpoint_collection_canonical_sha256"]
        != stable_sha256(collection)
        or gate_authorization["normalizer_sha256"] != collection["normalizer_sha256"]
    ):
        raise IntegrityError("gate authorization differs from the verified checkpoint collection")
    precalibration_seal, precalibration_seal_receipt = (
        load_precalibration_seal_with_receipt(
            precalibration_seal_path,
            study_binding=binding,
            selection=selection,
            fit_bundle=fit_bundle,
            target_boundary_amendment=amendment,
            checkpoint_collection=collection,
        )
    )
    seal, seal_receipt = _load_verified(execution_seal_path)
    validate_execution_seal(
        seal,
        study_binding=binding,
        selected_candidate=selection,
        gate=gate,
        gate_authorization=gate_authorization,
        target_boundary_amendment=amendment,
        checkpoint_collection=collection,
        precalibration_seal=precalibration_seal,
    )
    if seal["execution_mode"] != expected_execution_mode:
        raise IntegrityError(
            f"offline scorer expected {expected_execution_mode}, found {seal['execution_mode']}"
        )
    if (
        seal["selected_candidate_artifact"] != _artifact_binding(selected_receipt)
        or seal["checkpoint_collection_artifact"] != _artifact_binding(collection_receipt)
        or seal["gate_artifact"] != _artifact_binding(gate_receipt)
        or seal["gate_authorization_artifact"]
        != _artifact_binding(gate_authorization_receipt)
        or seal["target_boundary_amendment_artifact"]
        != _artifact_binding(amendment_receipt)
        or seal["precalibration_seal_artifact"]
        != _artifact_binding(precalibration_seal_receipt)
        or precalibration_seal["selected_gate_fit_bundle_artifact"]
        != _artifact_binding(fit_bundle_receipt)
    ):
        raise IntegrityError("execution seal artifact bindings differ at scoring")
    if (
        seal["source_postrun_acceptance"] != source_acceptance_binding
        or seal["source_postrun_training_container"]
        != source_acceptance["postrun_source_container"]
        or seal["source_hdf5_runtime_disclosure"]
        != source_acceptance["source_hdf5_runtime_disclosure"]
        or seal["source_checkpoint_selection_disclosure"]
        != source_acceptance["source_checkpoint_selection_disclosure"]
        or seal["source_initialization_clarification"]
        != source_acceptance["source_initialization_clarification"]
    ):
        raise IntegrityError("offline scorer source acceptance provenance drift")
    scorer_code_identity = target_scorer_code_identity()
    scorer_environment_identity = target_scorer_environment_identity()
    if (
        seal["scorer_code_identity_sha256"]
        != scorer_code_identity["code_identity_sha256"]
        or seal["scorer_environment_identity_sha256"]
        != scorer_environment_identity["environment_identity_sha256"]
    ):
        raise IntegrityError("offline scorer code or environment differs from the execution seal")
    master, cells = load_complete_target_bundle(
        target_bundle_path,
        seal=seal,
        gate_authorization=gate_authorization,
        gate=gate,
        selected_candidate=selected,
    )
    bundle_receipt = verify_artifact_receipt(target_bundle_path)
    if (
        master["execution_seal_artifact"] != _artifact_binding(seal_receipt)
        or master["gate_authorization_artifact"]
        != _artifact_binding(gate_authorization_receipt)
        or master["target_boundary_amendment_artifact"]
        != _artifact_binding(amendment_receipt)
        or master["precalibration_seal_artifact"]
        != _artifact_binding(precalibration_seal_receipt)
        or master["population_manifest_artifact"] != _artifact_binding(manifest_receipt)
        or master["selected_candidate_artifact"] != _artifact_binding(selected_receipt)
        or master["checkpoint_collection_artifact"] != _artifact_binding(collection_receipt)
    ):
        raise IntegrityError("complete target bundle upstream artifact binding mismatch")
    output = Path(output_path).expanduser().resolve()
    output_receipt = output.with_name(output.name + ".receipt.json")
    output_reservation_path = output.with_name(output.name + ".reservation.json")
    output_reservation_receipt_path = output_reservation_path.with_name(
        output_reservation_path.name + ".receipt.json"
    )
    if (
        not output.parent.is_dir()
        or not os.access(output.parent, os.W_OK)
        or shutil.disk_usage(output.parent).free < MINIMUM_OUTPUT_FREE_BYTES
    ):
        raise IntegrityError(
            "offline score output parent must already exist, be writable, and have "
            "at least 64 MiB free before outcome reveal"
        )
    registry_root = validate_reveal_registry_directory(
        reveal_registry_dir, seal["outcome_reveal_registry"]
    )
    reveal_path = registry_root / (
        f"so2sat_target_outcome_reveal_{seal['execution_seal_sha256']}.json"
    )
    reveal_receipt_path = reveal_path.with_name(reveal_path.name + ".receipt.json")
    if output.exists() or output_receipt.exists():
        raise IntegrityError("refusing to overwrite an offline target score artifact")
    if output_reservation_path.exists() or output_reservation_receipt_path.exists():
        raise IntegrityError("offline target score output was already reserved")
    if reveal_path.exists() or reveal_receipt_path.exists():
        raise IntegrityError(
            "a single testing-label reveal was already authorized; fail-closed recovery "
            "forbids reopening outcomes"
        )
    paths, signatures = _verify_target_container_paths(
        target_data_paths, seal["target_data_identities"]
    )
    split_counts = {
        split: manifest.get("splits", {}).get(split, {}).get("observed_samples")
        for split in ("validation", "testing")
    }
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in split_counts.values()
    ):
        raise IntegrityError("scoring manifest has an invalid target population count")
    validation_count = int(split_counts["validation"])
    testing_count = int(split_counts["testing"])
    bundle_validation_rows = sorted(
        {
            row_index
            for cell in cells
            if cell["checkpoint_id"] == CHECKPOINT_IDS[0]
            for row_index in cell["probe"]["row_indices"]
        }
    )
    bundle_testing_rows = sorted(
        {
            row_index
            for cell in cells
            if cell["checkpoint_id"] == CHECKPOINT_IDS[0]
            for row_index in cell["evaluation"]["row_indices"]
        }
    )
    if bundle_validation_rows != list(range(validation_count)):
        raise IntegrityError("complete target bundle does not cover the sealed probe population")
    if bundle_testing_rows != list(range(testing_count)):
        raise IntegrityError("complete target bundle does not cover the sealed testing population")
    if (
        master["access_audit"]["validation_pixel_rows"] != validation_count
        or master["access_audit"]["testing_pixel_rows"] != testing_count
    ):
        raise IntegrityError("target bundle live access counts differ from the manifest")

    output_reservation = {
        "schema": OUTPUT_RESERVATION_SCHEMA,
        "status": (
            "OUTPUT_DESTINATION_RESERVED_BEFORE_TARGET_OUTCOME_REVEAL"
            if seal["execution_mode"] == PRODUCTION_MODE
            else "TEST_ONLY_OUTPUT_DESTINATION_RESERVATION"
        ),
        "execution_mode": seal["execution_mode"],
        "output_basename": output.name,
        "output_parent_sha256": stable_sha256(str(output.parent)),
        "bundle_sha256": master["bundle_sha256"],
        "execution_seal_sha256": seal["execution_seal_sha256"],
        "minimum_free_bytes": MINIMUM_OUTPUT_FREE_BYTES,
        "observed_free_bytes": shutil.disk_usage(output.parent).free,
        "target_outcomes_opened": False,
    }
    output_reservation["reservation_sha256"] = stable_sha256(output_reservation)
    output_reservation_receipt = write_immutable_json_with_receipt(
        output_reservation_path, output_reservation
    )

    # Exclusively reserve the one permitted outcome reveal.  If this process
    # crashes after this create-only ledger is written, recovery is deliberately
    # fail-closed: the label array must not be reopened in another invocation.
    reveal = {
        "schema": REVEAL_SCHEMA,
        "status": (
            "AUTHORIZED_ONE_TESTING_LABEL_ARRAY_REVEAL_WITHIN_SEALED_REGISTRY"
            if seal["execution_mode"] == PRODUCTION_MODE
            else "TEST_ONLY_AUTHORIZED_SYNTHETIC_LABEL_ARRAY_REVEAL"
        ),
        "execution_mode": seal["execution_mode"],
        "output_basename": output.name,
        "target_bundle_artifact": _artifact_binding(bundle_receipt),
        "bundle_sha256": master["bundle_sha256"],
        "execution_seal_sha256": seal["execution_seal_sha256"],
        "gate_authorization_sha256": gate_authorization["authorization_sha256"],
        "target_boundary_amendment_artifact": _artifact_binding(amendment_receipt),
        "target_boundary_amendment_sha256": seal[
            "target_boundary_amendment_sha256"
        ],
        "precalibration_seal_artifact": _artifact_binding(
            precalibration_seal_receipt
        ),
        "precalibration_seal_sha256": precalibration_seal[
            "precalibration_seal_sha256"
        ],
        "source_postrun_acceptance": seal["source_postrun_acceptance"],
        "source_postrun_acceptance_artifact_sha256": seal[
            "source_postrun_acceptance_artifact_sha256"
        ],
        "source_postrun_training_container": seal[
            "source_postrun_training_container"
        ],
        "source_hdf5_runtime_disclosure": seal[
            "source_hdf5_runtime_disclosure"
        ],
        "source_checkpoint_selection_disclosure": seal[
            "source_checkpoint_selection_disclosure"
        ],
        "source_initialization_clarification": seal[
            "source_initialization_clarification"
        ],
        "outcome_reveal_registry": seal["outcome_reveal_registry"],
        "gate_sha256": gate["gate_sha256"],
        "selected_candidate_sha256": selected["selected_candidate_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "population_identity_sha256": manifest["population_identity_sha256"],
        "target_data_identities": seal["target_data_identities"],
        "scorer_code_identity": scorer_code_identity,
        "scorer_environment_identity": scorer_environment_identity,
        "testing_label_dataset_name": LABEL_DATASET_NAME,
        "testing_label_expected_shape": [testing_count, 17],
        "maximum_testing_label_open_count": 1,
        "validation_container_hdf5_open_count": 0,
        "validation_probe_labels_opened": False,
        "validation_probe_labels_scored": False,
        "crash_recovery_rule": "NEVER_REOPEN; REPORT_SCORING_ATTEMPT_INCOMPLETE",
    }
    reveal["reveal_authorization_sha256"] = stable_sha256(reveal)
    reveal_receipt = write_immutable_json_with_receipt(reveal_path, reveal)

    # Phase B: the single permitted outcome read.  Validation is never passed
    # to the HDF5 factory and its label array is structurally unreachable here.
    truth = _read_testing_outcomes_once(
        paths["testing"], expected_rows=testing_count, h5_factory=h5_factory
    )
    for split in ("validation", "testing"):
        identity = seal["target_data_identities"][split]
        if (
            _stat_signature(paths[split]) != signatures[split]
            or paths[split].stat().st_size != identity["bytes"]
            or file_sha256(paths[split]) != identity["sha256"]
        ):
            raise IntegrityError("target container changed during the one-time outcome reveal")

    cell_metrics, effect_matrices = _score_cells(cells, truth, master["target_cities"])
    comparisons: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    for policy in ("always_adapt", "always_freeze"):
        matrix = effect_matrices[policy]
        city_means = matrix.mean(axis=1)
        sign_flip = _exact_sign_flip(city_means)
        raw_p_values[policy] = sign_flip["two_sided_p_value"]
        comparisons[policy] = {
            "comparison": f"KGA_vs_{policy}",
            "effect_definition": "fixed_policy_regret_minus_kga_regret",
            "algebraic_identity": "kga_accuracy_minus_fixed_policy_accuracy",
            "positive_favors": "KGA",
            "point_estimate": float(matrix.mean()),
            "point_estimate_percentage_points": float(100.0 * matrix.mean()),
            "city_mean_effects": {
                city: float(city_means[index])
                for index, city in enumerate(master["target_cities"])
            },
            "effect_matrix": {
                "row_axis": "target_city",
                "row_ids": list(master["target_cities"]),
                "column_axis": "source_checkpoint",
                "column_ids": list(CHECKPOINT_IDS),
                "city_by_checkpoint_values": matrix.tolist(),
            },
            "confidence_interval": _crossed_bootstrap_interval(matrix),
            "sign_flip": sign_flip,
        }
    holm = _holm_adjust(raw_p_values, alpha=INFERENCE_ALPHA)
    for policy in comparisons:
        comparisons[policy]["multiplicity"] = holm[policy]

    exposure = _exposure(cell_metrics, seal["inference_contract"])
    both_ci_positive = all(
        comparisons[policy]["confidence_interval"]["lower"] > 0.0
        for policy in comparisons
    )
    both_holm_positive = all(
        comparisons[policy]["point_estimate"] > 0.0
        and comparisons[policy]["multiplicity"]["holm_reject_at_0_05"]
        for policy in comparisons
    )
    numerical_strong_success = bool(
        both_ci_positive
        and both_holm_positive
        and exposure["both_actions_meaningfully_exposed"]
        and exposure["both_direct_adapt_and_freeze_meaningfully_exposed"]
        and exposure["both_helpful_and_harmful_adaptation_observed"]
        and len(cell_metrics) == 50
    )
    strong_success = bool(
        numerical_strong_success and seal["execution_mode"] == PRODUCTION_MODE
    )
    class_counts = np.bincount(truth, minlength=17).astype(int)
    document = {
        "schema": SCORE_SCHEMA,
        "status": (
            "OFFLINE_TARGET_OUTCOMES_SCORED_AND_REPORTED_ALL_DIRECTIONS"
            if seal["execution_mode"] == PRODUCTION_MODE
            else "TEST_ONLY_SYNTHETIC_OUTCOMES_SCORED_NOT_CONFIRMATORY_EVIDENCE"
        ),
        "execution_mode": seal["execution_mode"],
        "target_bundle_artifact": _artifact_binding(bundle_receipt),
        "bundle_sha256": master["bundle_sha256"],
        "reveal_authorization_artifact": _artifact_binding(reveal_receipt),
        "reveal_authorization_sha256": reveal["reveal_authorization_sha256"],
        "output_reservation_artifact": _artifact_binding(output_reservation_receipt),
        "output_reservation_sha256": output_reservation["reservation_sha256"],
        "scorer_code_identity": scorer_code_identity,
        "scorer_environment_identity": scorer_environment_identity,
        "execution_seal_artifact": _artifact_binding(seal_receipt),
        "execution_seal_sha256": seal["execution_seal_sha256"],
        "gate_authorization_artifact": _artifact_binding(gate_authorization_receipt),
        "gate_authorization_sha256": gate_authorization["authorization_sha256"],
        "target_boundary_amendment_artifact": _artifact_binding(amendment_receipt),
        "target_boundary_amendment_sha256": seal[
            "target_boundary_amendment_sha256"
        ],
        "precalibration_seal_artifact": _artifact_binding(
            precalibration_seal_receipt
        ),
        "precalibration_seal_sha256": precalibration_seal[
            "precalibration_seal_sha256"
        ],
        "source_postrun_acceptance": seal["source_postrun_acceptance"],
        "source_postrun_acceptance_artifact_sha256": seal[
            "source_postrun_acceptance_artifact_sha256"
        ],
        "source_postrun_training_container": seal[
            "source_postrun_training_container"
        ],
        "source_hdf5_runtime_disclosure": seal[
            "source_hdf5_runtime_disclosure"
        ],
        "source_checkpoint_selection_disclosure": seal[
            "source_checkpoint_selection_disclosure"
        ],
        "source_initialization_clarification": seal[
            "source_initialization_clarification"
        ],
        "outcome_reveal_registry": seal["outcome_reveal_registry"],
        "gate_sha256": gate["gate_sha256"],
        "selected_candidate_sha256": selected["selected_candidate_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "population_identity_sha256": manifest["population_identity_sha256"],
        "target_data_identities": seal["target_data_identities"],
        "outcome_access": {
            "testing_container_open_count": 1,
            "testing_label_dataset_request_count": 1,
            "testing_label_full_array_read_count": 1,
            "validation_container_hdf5_open_count": 0,
            "validation_label_dataset_request_count": 0,
            "validation_probe_labels_opened": False,
            "validation_probe_labels_scored": False,
            "testing_outcomes_opened_only_after_complete_bundle_verification": True,
        },
        "testing_label_quality": {
            "rows": int(truth.size),
            "classes": 17,
            "finite": True,
            "one_hot": True,
            "one_hot_absolute_tolerance": ONE_HOT_ATOL,
            "class_counts": class_counts.tolist(),
            "all_classes_observed": bool(np.all(class_counts > 0)),
        },
        "cell_metrics": cell_metrics,
        "inference": {
            "estimand": seal["inference_contract"]["estimand"],
            "within_cell_weighting": seal["inference_contract"][
                "within_cell_weighting"
            ],
            "target_city_weighting": seal["inference_contract"][
                "target_city_weighting"
            ],
            "source_checkpoint_weighting": seal["inference_contract"][
                "source_checkpoint_weighting"
            ],
            "primary_cluster_unit": "target_city",
            "crossed_unit": "source_checkpoint",
            "comparisons": comparisons,
            "holm_family_size": 2,
            "confidence_interval_sign": "positive_favors_kga",
            "all_outcomes_reported_regardless_of_direction": True,
        },
        "exposure": exposure,
        "strong_success_checks": {
            "both_crossed_bootstrap_intervals_above_zero": both_ci_positive,
            "both_two_sided_city_sign_flip_tests_holm_significant_with_positive_effect": both_holm_positive,
            "meaningful_adapt_and_freeze_exposure": exposure[
                "both_actions_meaningfully_exposed"
            ],
            "meaningful_direct_adapt_and_freeze_decisions": exposure[
                "both_direct_adapt_and_freeze_meaningfully_exposed"
            ],
            "helpful_and_harmful_adaptation_cases_observed": exposure[
                "both_helpful_and_harmful_adaptation_observed"
            ],
            "all_50_checkpoint_by_city_cells_complete": len(cell_metrics) == 50,
            "numerical_criteria_met_before_evidence_mode_guard": numerical_strong_success,
            "production_evidence_mode": seal["execution_mode"] == PRODUCTION_MODE,
            "strong_success": strong_success,
        },
        "interpretation_guard": (
            "This scorer reports the sealed result as observed. TEST_ONLY artifacts are never "
            "confirmatory evidence, and a false strong_success field must not be rewritten as "
            "confirmatory evidence. The one-reveal ledger is enforced within the receipt-bound "
            "selected registry; global enforcement across cloned registries requires external "
            "append-only storage or access controls."
        ),
    }
    document["score_sha256"] = stable_sha256(document)
    write_immutable_json_with_receipt(output, document)
    return output


def score_sealed_target_bundle(
    *,
    target_bundle_path: str | Path,
    execution_seal_path: str | Path,
    population_manifest_path: str | Path,
    source_postrun_acceptance_path: str | Path,
    selected_candidate_path: str | Path,
    selected_gate_fit_bundle_path: str | Path,
    selected_gate_cal_bundle_path: str | Path,
    precalibration_seal_path: str | Path,
    gate_path: str | Path,
    gate_authorization_path: str | Path,
    target_boundary_amendment_path: str | Path,
    checkpoint_collection_path: str | Path,
    checkpoint_dir: str | Path,
    reveal_registry_dir: str | Path,
    target_data_paths: Mapping[str, str | Path],
    output_path: str | Path,
) -> Path:
    """Production-only scorer using the canonical h5py and manifest validators."""

    return _score_sealed_target_bundle_core(
        target_bundle_path=target_bundle_path,
        execution_seal_path=execution_seal_path,
        population_manifest_path=population_manifest_path,
        source_postrun_acceptance_path=source_postrun_acceptance_path,
        selected_candidate_path=selected_candidate_path,
        selected_gate_fit_bundle_path=selected_gate_fit_bundle_path,
        selected_gate_cal_bundle_path=selected_gate_cal_bundle_path,
        precalibration_seal_path=precalibration_seal_path,
        gate_path=gate_path,
        gate_authorization_path=gate_authorization_path,
        target_boundary_amendment_path=target_boundary_amendment_path,
        checkpoint_collection_path=checkpoint_collection_path,
        checkpoint_dir=checkpoint_dir,
        reveal_registry_dir=reveal_registry_dir,
        target_data_paths=target_data_paths,
        output_path=output_path,
        expected_execution_mode=PRODUCTION_MODE,
        h5_factory=None,
        population_manifest_validator=_CANONICAL_MANIFEST_VALIDATOR,
    )


def _score_sealed_target_bundle_for_test(
    *,
    h5_factory: H5Factory,
    population_manifest_validator: Callable[[Mapping[str, Any]], None],
    **kwargs: Any,
) -> Path:
    """Injected scorer restricted to TEST_ONLY seals and nonconfirmatory output."""

    return _score_sealed_target_bundle_core(
        **kwargs,
        expected_execution_mode=TEST_ONLY_MODE,
        h5_factory=h5_factory,
        population_manifest_validator=population_manifest_validator,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-bundle", required=True)
    parser.add_argument("--execution-seal", required=True)
    parser.add_argument("--population-manifest", required=True)
    parser.add_argument("--source-postrun-acceptance", required=True)
    parser.add_argument("--selected-candidate", required=True)
    parser.add_argument("--selected-gate-fit-bundle", required=True)
    parser.add_argument("--selected-gate-cal-bundle", required=True)
    parser.add_argument("--precalibration-seal", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--gate-authorization", required=True)
    parser.add_argument("--target-boundary-amendment", required=True)
    parser.add_argument("--checkpoint-collection", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--reveal-registry-dir", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--testing-data", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = score_sealed_target_bundle(
        target_bundle_path=arguments.target_bundle,
        execution_seal_path=arguments.execution_seal,
        population_manifest_path=arguments.population_manifest,
        source_postrun_acceptance_path=arguments.source_postrun_acceptance,
        selected_candidate_path=arguments.selected_candidate,
        selected_gate_fit_bundle_path=arguments.selected_gate_fit_bundle,
        selected_gate_cal_bundle_path=arguments.selected_gate_cal_bundle,
        precalibration_seal_path=arguments.precalibration_seal,
        gate_path=arguments.gate,
        gate_authorization_path=arguments.gate_authorization,
        target_boundary_amendment_path=arguments.target_boundary_amendment,
        checkpoint_collection_path=arguments.checkpoint_collection,
        checkpoint_dir=arguments.checkpoint_dir,
        reveal_registry_dir=arguments.reveal_registry_dir,
        target_data_paths={
            "validation": arguments.validation_data,
            "testing": arguments.testing_data,
        },
        output_path=arguments.output,
    )
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI integration
    raise SystemExit(main())
