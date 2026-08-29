"""Locked paired two-way inference for the 5-checkpoint x 9-location panel."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .integrity import IntegrityError, require_sha256, stable_sha256
from .protocol_seal import (
    EXPECTED_MODEL_SEEDS,
    EXPECTED_TARGET_LOCATIONS,
    verify_artifact_receipt,
    verify_execution_environment,
    write_immutable_json_with_receipt,
)
from .ridge_gate import DECISIONS

COMPARISONS = ("versus_always_adapt", "versus_always_freeze")
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_828
FAMILYWISE_ALPHA = 0.05


def _validate_score_document(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    if document.get("schema") != "kbound_cct20_set_valued_score_v1":
        raise IntegrityError("unknown CCT-20 score schema")
    if document.get("status") != "ALL_LOCKED_CELLS_SCORED":
        raise IntegrityError("CCT-20 score document is incomplete")
    unsigned = dict(document)
    claimed = unsigned.pop("score_sha256", None)
    if claimed != stable_sha256(unsigned):
        raise IntegrityError("CCT-20 score SHA-256 mismatch")
    if document.get("benefit_sign") != "adapted_accuracy_minus_frozen_accuracy":
        raise IntegrityError("CCT-20 score adaptation-benefit sign drift")
    if (
        document.get("primary_contrast_sign")
        != "baseline_regret_minus_kga_regret; positive_favors_kga"
    ):
        raise IntegrityError("CCT-20 score primary-contrast sign drift")
    require_sha256(
        document.get("execution_seal_artifact_sha256"),
        field="execution_seal_artifact_sha256",
    )
    cells = [dict(row) for row in document.get("cells", ())]
    if len(cells) != 45:
        raise IntegrityError(f"inference requires 45 scored cells, found {len(cells)}")
    keys = {(row.get("checkpoint_seed"), str(row.get("location_id"))) for row in cells}
    locations = list(EXPECTED_TARGET_LOCATIONS)
    expected = {(seed, location) for seed in EXPECTED_MODEL_SEEDS for location in locations}
    if keys != expected:
        raise IntegrityError("scored cells do not form the complete 5 x 9 product")
    for row in cells:
        if row.get("decision") not in DECISIONS:
            raise IntegrityError("scored cell has invalid action")
        if int(row.get("n_evaluation_images", 0)) < 1:
            raise IntegrityError("scored cell has no evaluation images")
        if row.get("n_target_images") != row.get("n_probe_images", 0) + row.get(
            "n_evaluation_images", 0
        ):
            raise IntegrityError("scored cell target/probe/evaluation counts do not reconcile")
        require_sha256(
            row.get("checkpoint_tensor_sha256"), field="checkpoint_tensor_sha256"
        )
        accuracies = row.get("set_membership_top1_accuracy", {})
        if set(accuracies) != {"always_freeze", "always_adapt", "kga"}:
            raise IntegrityError("scored cell accuracy schema drift")
        for name, value in accuracies.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise IntegrityError(f"scored cell has invalid {name} accuracy")
        expected_kga = (
            float(accuracies["always_adapt"])
            if row["decision"] == "ADAPT"
            else float(accuracies["always_freeze"])
        )
        if not math.isclose(float(accuracies["kga"]), expected_kga, abs_tol=1e-15):
            raise IntegrityError("scored-cell KGA accuracy does not follow its sealed action")
        expected_benefit = float(accuracies["always_adapt"]) - float(
            accuracies["always_freeze"]
        )
        if not math.isclose(
            float(row.get("adaptation_benefit")), expected_benefit, abs_tol=1e-15
        ):
            raise IntegrityError("scored-cell adaptation-benefit sign/value mismatch")
        oracle = max(
            float(accuracies["always_freeze"]), float(accuracies["always_adapt"])
        )
        if not math.isclose(
            float(row.get("oracle_fixed_action_accuracy")), oracle, abs_tol=1e-15
        ):
            raise IntegrityError("scored-cell fixed-action oracle mismatch")
        regrets = row.get("regret_to_better_fixed_action", {})
        if set(regrets) != set(accuracies):
            raise IntegrityError("scored cell regret schema drift")
        for name in accuracies:
            expected_regret = oracle - float(accuracies[name])
            if not math.isclose(float(regrets.get(name)), expected_regret, abs_tol=1e-15):
                raise IntegrityError(f"scored-cell {name} regret does not reconcile")
        contrasts = row.get("baseline_regret_minus_kga_regret", {})
        for comparison in COMPARISONS:
            value = contrasts.get(comparison)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise IntegrityError(f"scored cell lacks numeric {comparison}")
            if not math.isfinite(float(value)):
                raise IntegrityError(f"scored cell has non-finite {comparison}")
            baseline = "always_adapt" if comparison.endswith("adapt") else "always_freeze"
            expected_contrast = float(regrets[baseline]) - float(regrets["kga"])
            if not math.isclose(float(value), expected_contrast, abs_tol=1e-15):
                raise IntegrityError(f"scored-cell {comparison} sign/value mismatch")
            if not math.isclose(
                expected_contrast,
                float(accuracies["kga"]) - float(accuracies[baseline]),
                abs_tol=1e-15,
            ):
                raise IntegrityError(f"scored-cell {comparison} algebra mismatch")
    checkpoint_hashes: dict[int, str] = {}
    for row in cells:
        seed = int(row["checkpoint_seed"])
        observed = str(row["checkpoint_tensor_sha256"])
        prior = checkpoint_hashes.setdefault(seed, observed)
        if prior != observed:
            raise IntegrityError("checkpoint tensor identity changes across locations")
    if len(set(checkpoint_hashes.values())) != 5:
        raise IntegrityError("inference score does not use five distinct checkpoint tensors")
    return cells


def _matrices(cells: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, np.ndarray]]:
    locations = list(EXPECTED_TARGET_LOCATIONS)
    keyed = {(int(row["checkpoint_seed"]), str(row["location_id"])): row for row in cells}
    matrices = {}
    for comparison in COMPARISONS:
        matrices[comparison] = np.asarray(
            [
                [
                    float(keyed[(seed, location)]["baseline_regret_minus_kga_regret"][comparison])
                    for location in locations
                ]
                for seed in EXPECTED_MODEL_SEEDS
            ],
            dtype=np.float64,
        )
    return locations, matrices


def _empirical_interval(samples: np.ndarray, level: float) -> list[float]:
    if not 0.0 < level < 1.0:
        raise IntegrityError("bootstrap interval level must lie in (0, 1)")
    tail = (1.0 - level) / 2.0
    low = float(np.quantile(samples, tail, method="lower"))
    high = float(np.quantile(samples, 1.0 - tail, method="higher"))
    return [low, high]


def paired_two_way_product_bootstrap(
    matrices: Mapping[str, np.ndarray],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    random_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Resample checkpoint rows and location columns independently, paired."""

    if replicates != BOOTSTRAP_REPLICATES or random_seed != BOOTSTRAP_SEED:
        raise IntegrityError(
            f"bootstrap is frozen at {BOOTSTRAP_REPLICATES} replicates and seed {BOOTSTRAP_SEED}"
        )
    arrays = {name: np.asarray(matrices[name], dtype=np.float64) for name in COMPARISONS}
    if any(array.shape != (5, 9) or not np.isfinite(array).all() for array in arrays.values()):
        raise IntegrityError("each paired bootstrap contrast must be a finite 5 x 9 matrix")
    generator = np.random.default_rng(random_seed)
    samples = {name: np.empty(replicates, dtype=np.float64) for name in COMPARISONS}
    for iteration in range(replicates):
        checkpoint_index = generator.integers(0, 5, size=5)
        location_index = generator.integers(0, 9, size=9)
        selection = np.ix_(checkpoint_index, location_index)
        for name in COMPARISONS:
            samples[name][iteration] = arrays[name][selection].mean()
    return {
        name: {
            "point_estimate": float(arrays[name].mean()),
            "pointwise_95_ci": _empirical_interval(samples[name], 0.95),
            "simultaneous_bonferroni_97_5_ci": _empirical_interval(samples[name], 0.975),
        }
        for name in COMPARISONS
    }


def exact_location_sign_flip(location_differences: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(location_differences, dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise IntegrityError("exact sign-flip test requires nine finite location differences")
    observed = float(values.mean())
    null_means = np.asarray(
        [
            float(np.mean(values * np.asarray(signs, dtype=np.float64)))
            for signs in itertools.product((-1.0, 1.0), repeat=9)
        ],
        dtype=np.float64,
    )
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(observed)) * 16.0
    extreme = int(np.sum(null_means >= observed - tolerance))
    return {
        "n_locations": 9,
        "enumerated_sign_patterns": 512,
        "alternative": "mean_contrast_greater_than_zero",
        "observed_location_mean": observed,
        "p_value_one_sided": extreme / 512.0,
        "extreme_patterns_including_observed": extreme,
    }


def holm_adjust(raw: Mapping[str, float]) -> dict[str, float]:
    if set(raw) != set(COMPARISONS):
        raise IntegrityError(f"Holm family must be exactly {COMPARISONS}")
    ordered = sorted(((name, float(value)) for name, value in raw.items()), key=lambda pair: pair[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        if not 0.0 <= value <= 1.0:
            raise IntegrityError("Holm p-values must lie in [0, 1]")
        running = max(running, (family_size - rank) * value)
        adjusted[name] = min(1.0, running)
    return adjusted


def analyze_score_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Produce the frozen primary inferential report, regardless of direction."""

    cells = _validate_score_document(document)
    locations, matrices = _matrices(cells)
    bootstrap = paired_two_way_product_bootstrap(matrices)
    sign_flip = {
        name: exact_location_sign_flip(matrices[name].mean(axis=0)) for name in COMPARISONS
    }
    adjusted = holm_adjust({name: sign_flip[name]["p_value_one_sided"] for name in COMPARISONS})
    for name in COMPARISONS:
        sign_flip[name]["holm_adjusted_p"] = adjusted[name]
        sign_flip[name]["holm_reject_at_familywise_0_05"] = adjusted[name] <= FAMILYWISE_ALPHA

    action_counts = {decision: sum(row["decision"] == decision for row in cells) for decision in DECISIONS}
    rates = {decision: action_counts[decision] / 45.0 for decision in DECISIONS}
    exposure = {
        "counts": action_counts,
        "rates": rates,
        "strict_decision_coverage": rates["ADAPT"] + rates["FREEZE"],
        "thresholds": {
            "minimum_adapt_rate": 0.10,
            "minimum_freeze_rate": 0.10,
            "minimum_strict_decision_coverage": 0.30,
        },
    }
    exposure["passes"] = {
        "adapt": rates["ADAPT"] >= 0.10,
        "freeze": rates["FREEZE"] >= 0.10,
        "strict_decision_coverage": exposure["strict_decision_coverage"] >= 0.30,
    }
    benefits = np.asarray([float(row["adaptation_benefit"]) for row in cells], dtype=np.float64)
    mixed_effects = {
        "helpful_cells_strictly_positive": int(np.sum(benefits > 0.0)),
        "neutral_cells_exactly_zero": int(np.sum(benefits == 0.0)),
        "harmful_cells_strictly_negative": int(np.sum(benefits < 0.0)),
        "mixed_helpful_and_harmful_present": bool(
            np.any(benefits > 0.0) and np.any(benefits < 0.0)
        ),
    }
    ci_pass = {
        name: bootstrap[name]["simultaneous_bonferroni_97_5_ci"][0] > 0.0
        for name in COMPARISONS
    }
    holm_pass = {
        name: sign_flip[name]["holm_reject_at_familywise_0_05"] for name in COMPARISONS
    }
    protocol_strong_success = (
        all(ci_pass.values())
        and all(holm_pass.values())
        and all(exposure["passes"].values())
    )
    safe_utility = {
        "frozen_noninferiority_margin": -0.005,
        "kga_minus_freeze_point_estimate": bootstrap["versus_always_freeze"]["point_estimate"],
        "kga_minus_adapt_point_estimate": bootstrap["versus_always_adapt"]["point_estimate"],
        "passes": (
            bootstrap["versus_always_freeze"]["pointwise_95_ci"][0] > -0.005
            and bootstrap["versus_always_adapt"]["pointwise_95_ci"][0] > 0.0
        ),
    }
    report = {
        "schema": "kbound_cct20_two_way_inference_v1",
        "status": "COMPLETE_REPORT_REGARDLESS_OF_RESULT",
        "score_sha256": document["score_sha256"],
        "execution_seal_artifact_sha256": document["execution_seal_artifact_sha256"],
        "primary_contrast_sign": "baseline_regret_minus_kga_regret; positive_favors_kga",
        "design": {
            "checkpoint_seeds": list(EXPECTED_MODEL_SEEDS),
            "location_ids": locations,
            "matrix_shape": [5, 9],
            "cell_count": 45,
            "estimand": "equal_weight_mean_over_checkpoints_and_locations",
        },
        "paired_two_way_product_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "random_seed": BOOTSTRAP_SEED,
            "checkpoint_rows_and_location_columns_resampled_independently": True,
            "comparisons_share_every_resample": True,
            "percentile_endpoint_rule": "lower_tail_lower_order_statistic; upper_tail_higher_order_statistic",
            "results": bootstrap,
        },
        "exact_nine_location_sign_flip_and_holm": sign_flip,
        "action_exposure_at_checkpoint_location_unit": exposure,
        "adaptation_effect_mix": mixed_effects,
        "strong_success_checks": {
            "both_simultaneous_intervals_strictly_above_zero": all(ci_pass.values()),
            "both_exact_tests_survive_holm_0_05": all(holm_pass.values()),
            "all_action_exposure_thresholds_pass": all(exposure["passes"].values()),
            "all_45_cells_complete": True,
            "protocol_strong_success": protocol_strong_success,
            "expanded_empirical_bundle_including_mixed_effects": (
                protocol_strong_success and mixed_effects["mixed_helpful_and_harmful_present"]
            ),
        },
        "safe_utility": safe_utility,
    }
    report["inference_sha256"] = stable_sha256(report)
    return report


def analyze_sealed_score_once(
    score_path: str | Path,
    *,
    execution_seal_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Verify the score receipt, run the frozen analysis, and seal its report."""

    verify_artifact_receipt(score_path)
    execution_receipt = verify_artifact_receipt(execution_seal_path)
    try:
        score = json.loads(Path(score_path).read_text(encoding="utf-8"))
        execution_seal = json.loads(
            Path(execution_seal_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read sealed CCT-20 score/execution document: {exc}") from exc
    if not isinstance(execution_seal, Mapping):
        raise IntegrityError("sealed CCT-20 execution document is not an object")
    verify_execution_environment(execution_seal)
    if score.get("execution_seal_artifact_sha256") != execution_receipt.get(
        "artifact_sha256"
    ):
        raise IntegrityError("score document is not bound to this execution seal")
    report = analyze_score_document(score)
    write_immutable_json_with_receipt(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--execution-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_sealed_score_once(
        args.score,
        execution_seal_path=args.execution_seal,
        output_path=args.output,
    )
    print(
        f"two-way inference complete: {report['inference_sha256']} -> {args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
