#!/usr/bin/env python3
"""Import and reconcile the release's natural/corruption result panels.

The original experiment archives contain large prediction arrays and several
superseded aggregate summaries.  This script has two deliberately separate
jobs:

1. ``--import-from`` creates compact, source-hashed panel artifacts containing
   every field needed to replay the decision metrics.
2. The default reconciliation pass applies the canonical exact-rank KGA rule
   and writes one JSON/Markdown/LaTeX result panel.

No training is performed and no metric is accepted from prose.  PACS is the one
partial exception: its archived seed summaries are validated against each
other, but the saved per-cell files omit ``b_hat`` and calibration residuals,
so PACS decisions cannot be replayed.  The output marks that limitation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kga.certificate import split_conformal_rank_radius  # noqa: E402
from kga.policy import decide_batch, decide_kga  # noqa: E402

ALPHA = 0.10
EXPECTED_NUMPY_VERSION = "2.4.4"
EXPECTED_SKLEARN_VERSION = "1.8.0"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260809
KAPPA_GRID = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)
RESULT_ROOT = ROOT / "experiments/kbound/results/reconciled_panels_v1"
SOURCE_ROOT = RESULT_ROOT / "source"

RECORD_KEYS = (
    "seed",
    "dataset",
    "benchmark",
    "domain",
    "location",
    "location_n",
    "location_classes",
    "split",
    "comp",
    "regime",
    "aggr",
    "mode",
    "method",
    "candidate",
    "condition",
    "metric",
    "Z",
    "Z_names",
    "B",
    "a0",
    "aa",
    "a_adapted",
    "b_hat",
    "eps_conformal",
    "kga_decision",
    "oracle_action",
)

META_KEYS = (
    "schema",
    "dataset",
    "benchmark",
    "role",
    "metric",
    "config",
    "config_sha8",
    "num_classes",
    "evidence_names",
    "candidates",
    "seed",
    "method",
    "alpha",
    "n_conditions",
    "kga_backend",
)


def _json_bytes(data: Any) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(data))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_specs(archive: Path) -> list[tuple[Path, Path]]:
    results = archive / "experiments/kbound/results"
    specs = [
        (
            results / "officehome_full_targetval/result_target_val_361a1e8c.json",
            SOURCE_ROOT / "officehome/calibration_seed01.json",
        ),
        (
            results / "officehome_full_targettest/result_target_test_6605675d.json",
            SOURCE_ROOT / "officehome/test_seed01.json",
        ),
        (
            results / "officehome_protocol_m_repl_targetval/result_target_val_eb504dd6.json",
            SOURCE_ROOT / "officehome/calibration_seed234.json",
        ),
        (
            results / "officehome_protocol_m_repl_targettest/result_target_test_f761540b.json",
            SOURCE_ROOT / "officehome/test_seed234.json",
        ),
        (
            results / "iwildcam_full_test/result_e40faf29.json",
            SOURCE_ROOT / "iwildcam/test_seed01.json",
        ),
        (
            results / "iwildcam_protocol_H_v2/protocol_result.json",
            SOURCE_ROOT / "iwildcam/superseded_protocol_result.json",
        ),
        (
            results / "pacs_multiseed_v1/PACS_MULTISEED_RESULTS.json",
            SOURCE_ROOT / "pacs/PACS_MULTISEED_RESULTS.json",
        ),
        (
            results / "win_hunt_v5/pacs_aggr/pacs_result.json",
            SOURCE_ROOT / "pacs/pacs_seed0.json",
        ),
        (results / "pacs_seed1.json", SOURCE_ROOT / "pacs/pacs_seed1.json"),
        (results / "pacs_seed2.json", SOURCE_ROOT / "pacs/pacs_seed2.json"),
    ]
    imagenetc = results / "win_hunt_v5_imagenetc_ms/pooled_5seed"
    specs.extend(
        (path, SOURCE_ROOT / "imagenetc" / path.name)
        for path in sorted(imagenetc.glob("per_condition_imagenetc_*_seed*.json"))
    )
    specs.extend(
        (path, SOURCE_ROOT / "pacs/per_cell" / path.name)
        for path in sorted((results / "per_cell").glob("pacs_*_seed*_percell.json"))
    )
    imagenetr = results / "imagenetr_protocol_d_multiseed_v1"
    specs.extend(
        (path, SOURCE_ROOT / "imagenetr" / path.name)
        for path in sorted(imagenetr.glob("per_condition_imagenet-r_*_seed*.json"))
    )
    cifar10c = results / "stress_grid_multiseed_v1"
    for candidate in ("tent", "eata"):
        specs.extend(
            (path, SOURCE_ROOT / "cifar10c" / path.name)
            for path in sorted(cifar10c.glob(f"seed*/per_condition_cifar10c_{candidate}_seed*.json"))
        )
    specs.extend(
        (path, SOURCE_ROOT / "cifar10c" / path.name)
        for path in sorted(
            (results / "cifar10c_sar_rebuild_v2").glob("seed[0-4]/per_condition_cifar10c_sar_seed*.json")
        )
    )
    specs.append(
        (
            results / "camelyon17_richZ_F_v1/result_884129ba.json",
            SOURCE_ROOT / "camelyon17_ood/result_camelyon17_richz.json",
        )
    )
    specs.extend(
        (path, SOURCE_ROOT / "camelyon17_b_v2" / path.name)
        for path in sorted((results / "camelyon17_fullscale_B_v2").glob("per_condition_camelyon17_*_seed*.json"))
    )
    for model_seed, filename in (
        (0, "result_3f579e72.json"),
        (1, "result_eef46aea.json"),
        (2, "result_6585f5b7.json"),
    ):
        specs.append(
            (
                results / f"rxrx1_protocol_c_9plus_modelseed{model_seed}/{filename}",
                SOURCE_ROOT / f"rxrx1/modelseed{model_seed}.json",
            )
        )
    specs.extend(
        (path, SOURCE_ROOT / "cifar101" / path.name)
        for path in sorted((results / "cifar101_multiseed_v1").glob("seed*/per_condition_cifar101_tent_seed*.json"))
    )
    return specs


def _portable_metadata(value: Any) -> Any:
    """Redact machine-local path prefixes while retaining useful basenames."""
    if isinstance(value, dict):
        return {key: _portable_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_metadata(item) for item in value]
    if isinstance(value, str) and (value.startswith("/Users/") or value.startswith("/Volumes/")):
        return f"<external>/{Path(value).name}"
    return value


def _compact_payload(data: dict[str, Any], *, provenance: dict[str, Any]) -> dict[str, Any]:
    if "records" not in data:
        copied = cast(dict[str, Any], _portable_metadata(dict(data)))
        copied["_provenance"] = provenance
        return copied
    records = []
    for record in data["records"]:
        compact = {key: record[key] for key in RECORD_KEYS if key in record}
        if "aa" in compact and "a_adapted" not in compact:
            compact["a_adapted"] = compact.pop("aa")
        records.append(compact)
    return {
        "schema": "kbound-compact-panel-source-v1",
        "metadata": _portable_metadata({key: data[key] for key in META_KEYS if key in data}),
        "records": records,
        "_provenance": provenance,
    }


def import_sources(archive: Path) -> dict[str, Any]:
    imported: list[dict[str, Any]] = []
    specs = _source_specs(archive)
    missing = [str(src) for src, _ in specs if not src.is_file()]
    if missing:
        raise FileNotFoundError("missing archived panel sources:\n" + "\n".join(missing))
    for source, destination in specs:
        data = json.loads(source.read_text())
        provenance = {
            "archive_relative_path": source.relative_to(archive).as_posix(),
            "original_sha256": _sha256(source),
            "original_bytes": source.stat().st_size,
        }
        compact = _compact_payload(data, provenance=provenance)
        _write_json(destination, compact)
        imported.append(
            {
                "source": provenance["archive_relative_path"],
                "destination": destination.relative_to(ROOT).as_posix(),
                "original_sha256": provenance["original_sha256"],
                "original_bytes": provenance["original_bytes"],
                "compact_sha256": _sha256(destination),
                "compact_bytes": destination.stat().st_size,
                "records": len(compact.get("records", [])),
            }
        )
    manifest = {
        "schema": "kbound-panel-source-manifest-v1",
        "generator": "scripts/reconcile_result_panels.py --import-from",
        "generator_sha256": _sha256(Path(__file__)),
        "source_archive": archive.name,
        "source_archive_note": "local import path intentionally omitted from the public artifact",
        "files": imported,
        "file_count": len(imported),
        "original_bytes": sum(row["original_bytes"] for row in imported),
        "compact_bytes": sum(row["compact_bytes"] for row in imported),
    }
    _write_json(RESULT_ROOT / "source_manifest.json", manifest)
    return manifest


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text()))


def _records(path: Path) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], _load(path)["records"])


def _path_list(paths: Path | Sequence[Path]) -> list[Path]:
    return [paths] if isinstance(paths, Path) else list(paths)


def _records_many(paths: Path | Sequence[Path]) -> list[dict[str, Any]]:
    return [record for path in _path_list(paths) for record in _records(path)]


def _evidence_contract(paths: Path | Sequence[Path], records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Recover the exact feature width/names represented by compact sources."""

    dimensions = sorted({len(row["Z"]) for row in records if "Z" in row})
    if len(dimensions) != 1:
        raise ValueError(f"mixed or missing evidence dimensions: {dimensions}")
    names_seen: set[tuple[str, ...]] = set()
    for row in records:
        if row.get("Z_names"):
            names_seen.add(tuple(str(name) for name in row["Z_names"]))
    for path in _path_list(paths):
        metadata = _load(path).get("metadata", {})
        if metadata.get("evidence_names"):
            names_seen.add(tuple(str(name) for name in metadata["evidence_names"]))
    if len(names_seen) > 1:
        raise ValueError(f"mixed evidence-name schemas: {sorted(names_seen)!r}")
    names = next(iter(names_seen), None)
    dimension = dimensions[0]
    if names is not None and len(names) != dimension:
        raise ValueError(f"evidence name count {len(names)} does not match width {dimension}")
    payload = {
        "dimension": dimension,
        "feature_names": list(names) if names is not None else None,
        "names_recovered": names is not None,
    }
    payload["schema_sha256"] = hashlib.sha256(_json_bytes(payload)).hexdigest()
    return payload


def _source_provenance(paths: Path | Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in _path_list(paths):
        data = _load(path)
        source = cast(dict[str, Any], data.get("_provenance", {}))
        rows.append(
            {
                "compact_path": path.relative_to(ROOT).as_posix(),
                "compact_sha256": _sha256(path),
                "archive_relative_path": source.get("archive_relative_path"),
                "original_sha256": source.get("original_sha256"),
                "original_bytes": source.get("original_bytes"),
            }
        )
    return rows


def _metric_arrays(records: Sequence[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    benefit = np.asarray([row["B"] for row in records], dtype=float)
    frozen = np.asarray([row["a0"] for row in records], dtype=float)
    adapted = np.asarray([row["a_adapted"] for row in records], dtype=float)
    return benefit, frozen, adapted


def score_decisions(records: Sequence[dict[str, Any]], decisions: Any) -> dict[str, Any]:
    benefit, frozen, adapted = _metric_arrays(records)
    decision = np.asarray(decisions, dtype=object)
    is_adapt = decision == "ADAPT"
    is_freeze = decision == "FREEZE"
    is_abstain = decision == "ABSTAIN"
    harmful = benefit <= 0.0
    oracle = np.maximum(frozen, adapted)
    kga_accuracy = np.where(is_adapt, adapted, frozen)
    fa_count = int(np.sum(is_adapt & harmful))
    n_adapt = int(np.sum(is_adapt))
    n = len(records)
    regret = {
        "kga": float(np.mean(oracle - kga_accuracy)),
        "always_adapt": float(np.mean(oracle - adapted)),
        "always_freeze": float(np.mean(oracle - frozen)),
    }
    return {
        "n": n,
        "regret": regret,
        "fa_u": fa_count / n,
        "fa_c": fa_count / n_adapt if n_adapt else None,
        "false_adapt_count": fa_count,
        "adapt_count": n_adapt,
        "freeze_count": int(np.sum(is_freeze)),
        "abstain_count": int(np.sum(is_abstain)),
        "adapt_rate": n_adapt / n,
        "decision_coverage": float(np.mean(~is_abstain)),
        "point_beats_both": bool(regret["kga"] < regret["always_adapt"] and regret["kga"] < regret["always_freeze"]),
    }


def _seed_bootstrap(records: Sequence[dict[str, Any]], decisions: Any) -> dict[str, Any]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        grouped[int(row["seed"])].append(index)
    seed_rows: list[dict[str, Any]] = []
    decisions_array = np.asarray(decisions, dtype=object)
    for seed in sorted(grouped):
        indexes = grouped[seed]
        subset = [records[index] for index in indexes]
        score = score_decisions(subset, decisions_array[indexes])
        seed_rows.append(
            {
                "seed": seed,
                "n": score["n"],
                "regret": score["regret"],
                "fa_u": score["fa_u"],
                "adapt_rate": score["adapt_rate"],
                "decision_coverage": score["decision_coverage"],
                "point_beats_both": score["point_beats_both"],
            }
        )
    if len(seed_rows) < 2:
        return {
            "per_seed": seed_rows,
            "paired_seed_bootstrap": None,
            "ci_robust_beats_both": False,
            "reason": "fewer than two independent seed units",
        }
    gaps = {
        baseline: np.asarray(
            [row["regret"][baseline] - row["regret"]["kga"] for row in seed_rows],
            dtype=float,
        )
        for baseline in ("always_adapt", "always_freeze")
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample = rng.integers(0, len(seed_rows), size=(BOOTSTRAP_REPLICATES, len(seed_rows)))
    intervals: dict[str, dict[str, Any]] = {}
    for baseline, values in gaps.items():
        draws = values[sample].mean(axis=1)
        intervals[baseline] = {
            "mean_gap_baseline_minus_kga": float(values.mean()),
            "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        }
    descriptive_positive = all(row["ci95"][0] > 0.0 for row in intervals.values())
    return {
        "per_seed": seed_rows,
        "descriptive_seed_bootstrap": {
            "unit": "run-seed mean conditional on the archived checkpoint/protocol",
            "replicates": BOOTSTRAP_REPLICATES,
            "random_seed": BOOTSTRAP_SEED,
            "gaps": intervals,
            "both_lower_bounds_positive": descriptive_positive,
        },
        "ci_robust_beats_both": False,
        "reason": (
            "independent checkpoint identities are not recorded for these run seeds; "
            "the percentile interval is descriptive and cannot promote a CI-robust claim"
        ),
    }


def _annotate_score(records: Sequence[dict[str, Any]], decisions: Any) -> dict[str, Any]:
    out = score_decisions(records, decisions)
    out["seed_inference"] = _seed_bootstrap(records, decisions)
    return out


def _kappa_sweep(records: Sequence[dict[str, Any]], prediction: Any, epsilon: Any) -> list[dict[str, Any]]:
    """Replay the interval gate while scaling only its already-fitted radius."""
    prediction_array = np.asarray(prediction, dtype=float)
    epsilon_array = np.asarray(epsilon, dtype=float)
    rows: list[dict[str, Any]] = []
    for kappa in KAPPA_GRID:
        score = score_decisions(
            records,
            decide_batch(prediction_array, kappa * epsilon_array, alpha=ALPHA),
        )
        rows.append(
            {
                "kappa": kappa,
                "regret": score["regret"]["kga"],
                "yield": score["decision_coverage"],
                "adapt_rate": score["adapt_rate"],
                "fa_u": score["fa_u"],
                "fa_c": score["fa_c"],
                "adapt_count": score["adapt_count"],
                "freeze_count": score["freeze_count"],
                "abstain_count": score["abstain_count"],
                "false_adapt_count": score["false_adapt_count"],
            }
        )
    return rows


def _radius_diagnostics(
    records: Sequence[dict[str, Any]], prediction: Any, epsilon: Any, decisions: Any
) -> dict[str, Any]:
    benefit = np.asarray([row["B"] for row in records], dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    epsilon_array = np.asarray(epsilon, dtype=float)
    decision_array = np.asarray(decisions, dtype=object)
    abstain = decision_array == "ABSTAIN"
    ratio = np.divide(
        np.abs(prediction_array),
        epsilon_array,
        out=np.full_like(prediction_array, np.inf),
        where=epsilon_array > 0,
    )
    ceiling = float(np.mean(np.abs(benefit) > epsilon_array))
    decision_yield = float(np.mean(~abstain))
    if np.any(abstain):
        median_ratio = float(np.median(ratio[abstain]))
        signal_limited = float(np.mean(ratio[abstain] < 0.25))
        radius_limited = float(np.mean(ratio[abstain] > 0.50))
    else:
        median_ratio = signal_limited = radius_limited = None
    return {
        "n": len(records),
        "abstain_rate": float(np.mean(abstain)),
        "yield": decision_yield,
        "eps_mean": float(np.mean(epsilon_array)),
        "mean_abs_b_hat": float(np.mean(np.abs(prediction_array))),
        "median_abs_b_hat_over_eps_among_abstains": median_ratio,
        "frac_abstains_signal_limited_ratio_lt_0.25": signal_limited,
        "frac_abstains_radius_limited_ratio_gt_0.50": radius_limited,
        "mean_abs_Delta": float(np.mean(np.abs(benefit))),
        "sd_Delta": float(np.std(benefit)),
        "eps_over_mean_abs_Delta": (
            float(np.mean(epsilon_array) / np.mean(np.abs(benefit))) if np.mean(np.abs(benefit)) > 0 else None
        ),
        "yield_ceiling_P_absDelta_gt_eps": ceiling,
        "estimator_share_of_ceiling": decision_yield / ceiling if ceiling > 0 else None,
        "verdict": (
            "ESTIMATOR LIMITED: effects exceed the radius more often than the gate commits"
            if ceiling > 0 and decision_yield < 0.5 * ceiling
            else "NEAR THE EMPIRICAL RADIUS CEILING"
        ),
    }


def _fit_gbr(features: np.ndarray, benefit: np.ndarray):
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(
        n_estimators=250,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        random_state=0,
    ).fit(features, benefit)


def _transfer_panel(
    cal_path: Path | Sequence[Path],
    test_path: Path | Sequence[Path],
    *,
    candidate: str,
    cal_seeds: set[int],
    test_seeds: set[int],
    record_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    filters = record_filters or {}

    def selected(row: dict[str, Any], seeds: set[int]) -> bool:
        row_candidate = row.get("candidate") or row.get("method")
        return (
            row_candidate == candidate
            and int(row["seed"]) in seeds
            and all(row.get(key) == value for key, value in filters.items())
        )

    cal = [row for row in _records_many(cal_path) if selected(row, cal_seeds)]
    test = [row for row in _records_many(test_path) if selected(row, test_seeds)]
    if len(cal) < 2 or not test:
        raise ValueError(f"insufficient transfer records for {candidate}: {len(cal)=}, {len(test)=}")
    z_cal = np.asarray([row["Z"] for row in cal], dtype=float)
    b_cal = np.asarray([row["B"] for row in cal], dtype=float)
    z_test = np.asarray([row["Z"] for row in test], dtype=float)
    full_model = _fit_gbr(z_cal, b_cal)
    full_cal_prediction = full_model.predict(z_cal)
    test_prediction = full_model.predict(z_test)
    loo_prediction = np.empty(len(cal), dtype=float)
    for index in range(len(cal)):
        keep = np.arange(len(cal)) != index
        loo_prediction[index] = _fit_gbr(z_cal[keep], b_cal[keep]).predict(z_cal[index : index + 1])[0]
    residuals = np.abs(loo_prediction - b_cal)
    epsilon = split_conformal_rank_radius(residuals, ALPHA)
    decisions = decide_batch(test_prediction, epsilon, alpha=ALPHA)
    stability_gap = np.abs(full_cal_prediction - loo_prediction)
    observed_stability_radius = float(np.max(stability_gap))
    inflated_epsilon = epsilon + observed_stability_radius
    inflated_decisions = decide_batch(test_prediction, inflated_epsilon, alpha=ALPHA)
    rows = []
    for source, prediction, decision, inflated_decision in zip(
        test, test_prediction, decisions, inflated_decisions, strict=True
    ):
        row = {key: source[key] for key in RECORD_KEYS if key in source and key != "Z"}
        row["delta_hat"] = float(prediction)
        row["epsilon"] = float(epsilon)
        row["decision"] = str(decision)
        row["observed_stability_inflated_epsilon"] = float(inflated_epsilon)
        row["observed_stability_inflated_decision"] = str(inflated_decision)
        rows.append(row)
    return {
        "candidate": candidate,
        "record_filters": filters,
        "calibration_seeds": sorted(cal_seeds),
        "test_seeds": sorted(test_seeds),
        "n_calibration": len(cal),
        "n_test": len(test),
        "evidence_contract": _evidence_contract(
            list(dict.fromkeys(_path_list(cal_path) + _path_list(test_path))), cal + test
        ),
        "estimator": {
            "class": "sklearn.ensemble.GradientBoostingRegressor",
            "hyperparameters": {
                "n_estimators": 250,
                "max_depth": 2,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "random_state": 0,
            },
        },
        "calibration": {
            "method": (
                "leave-one-calibration-record-out empirical residuals with an exact-rank "
                "order statistic; not exact split conformal or jackknife+"
            ),
            "alpha": ALPHA,
            "epsilon": float(epsilon),
            "a7_status": "not_established",
            "a7_reason": (
                "the deployment estimator is refit on all calibration records; no uniform, "
                "predeclared full-fit-versus-LOO stability bound was archived"
            ),
            "observed_full_fit_vs_loo_max_gap": observed_stability_radius,
            "observed_full_fit_vs_loo_mean_gap": float(np.mean(stability_gap)),
            "observed_stability_inflation_is_diagnostic_only": True,
        },
        "exact_rank_transfer_score": _annotate_score(test, decisions),
        "observed_stability_inflated_sensitivity": _annotate_score(test, inflated_decisions),
        "records": rows,
    }


def reconcile_transfer_panels() -> dict[str, Any]:
    office_primary = _transfer_panel(
        SOURCE_ROOT / "officehome/calibration_seed01.json",
        SOURCE_ROOT / "officehome/test_seed01.json",
        candidate="sar_online_aggressive",
        cal_seeds={0, 1},
        test_seeds={0, 1},
    )
    office_replication = _transfer_panel(
        SOURCE_ROOT / "officehome/calibration_seed234.json",
        SOURCE_ROOT / "officehome/test_seed234.json",
        candidate="sar_online_aggressive",
        cal_seeds={2, 3, 4},
        test_seeds={2, 3, 4},
    )
    iwild_source = SOURCE_ROOT / "iwildcam/test_seed01.json"
    iwild = _transfer_panel(
        iwild_source,
        iwild_source,
        candidate="tent_episodic",
        cal_seeds={0},
        test_seeds={1},
    )
    historical_iwild = _load(SOURCE_ROOT / "iwildcam/superseded_protocol_result.json")
    historical_score = historical_iwild["test_locked"]
    return {
        "officehome": {
            "protocol": "M-v2 locked SAR, target-val calibration to target-test evaluation",
            "primary": office_primary,
            "test_stream_seed_replication": office_replication,
            "claim_scope": "descriptive transfer result; A7 stability premise was not predeclared",
        },
        "iwildcam": {
            "protocol": "H-v2 locked Tent episodic, seed-0 calibration to seed-1 evaluation",
            "primary": iwild,
            "claim_scope": (
                "withheld from release-level numerical claims because the archived scorer used "
                "sklearn macro-F1 rather than the official WILDS label-present macro-F1 contract"
            ),
            "release_promotion": {
                "eligible": False,
                "status": "withheld_metric_contract_rerun_required",
                "reason": (
                    "the archived scorer includes prediction-only classes; a pinned rerun with "
                    "the official WILDS metric and a sealed population manifest is required"
                ),
            },
            "historical_reconciliation": {
                "status": "superseded_not_promotable",
                "historical_artifact": (
                    "experiments/kbound/results/reconciled_panels_v1/source/iwildcam/superseded_protocol_result.json"
                ),
                "historical_claim": {
                    "epsilon": historical_score["eps_global"],
                    "regret_kga": historical_score["regret_kga"],
                    "regret_always_adapt": historical_score["regret_adapt"],
                    "regret_always_freeze": historical_score["regret_freeze"],
                    "beats_both": historical_score["beats_both"],
                },
                "corrected_claim": {
                    "epsilon": iwild["calibration"]["epsilon"],
                    "regret_kga": iwild["exact_rank_transfer_score"]["regret"]["kga"],
                    "regret_always_adapt": iwild["exact_rank_transfer_score"]["regret"]["always_adapt"],
                    "regret_always_freeze": iwild["exact_rank_transfer_score"]["regret"]["always_freeze"],
                    "point_beats_both": iwild["exact_rank_transfer_score"]["point_beats_both"],
                },
                "reason": (
                    "the historical radius is consistent with optimistic in-sample residual "
                    "calibration; replay with leave-one-calibration-record-out residuals makes "
                    "the radius wider, removes the only ADAPT action, and ties always-freeze"
                ),
            },
        },
    }


def _grid_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("per_condition_*.json"))


def _grid_panel(directory: Path, *, expected_seeds: set[int]) -> dict[str, Any]:
    files = _grid_files(directory)
    if not files:
        raise FileNotFoundError(f"no per-condition files under {directory}")
    by_candidate: dict[str, list[tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    seen_seeds: dict[str, set[int]] = defaultdict(set)
    for path in files:
        records = _records(path)
        if not records:
            raise ValueError(f"empty panel file: {path}")
        seed = int(records[0]["seed"])
        candidate = str(records[0].get("method") or records[0].get("candidate"))
        if seed in seen_seeds[candidate]:
            raise ValueError(f"duplicate candidate/seed panel: {candidate}/{seed}")
        seen_seeds[candidate].add(seed)
        benefit = np.asarray([row["B"] for row in records], dtype=float)
        prediction = np.asarray([row["b_hat"] for row in records], dtype=float)
        epsilon, decisions = decide_kga(prediction, benefit, alpha=ALPHA, calibration="loo")
        by_candidate[candidate].append((records, prediction, epsilon, decisions))
    if any(seeds != expected_seeds for seeds in seen_seeds.values()):
        raise ValueError(f"incomplete seed panel: {dict(seen_seeds)}; expected {sorted(expected_seeds)}")
    candidates: dict[str, dict[str, Any]] = {}
    all_records: list[dict[str, Any]] = []
    all_predictions: list[float] = []
    all_epsilons: list[float] = []
    all_decisions: list[str] = []
    for candidate in sorted(by_candidate):
        candidate_records: list[dict[str, Any]] = []
        candidate_predictions: list[float] = []
        candidate_epsilons: list[float] = []
        candidate_decisions: list[str] = []
        per_file: list[dict[str, Any]] = []
        for records, prediction, epsilon, decisions in sorted(
            by_candidate[candidate], key=lambda item: int(item[0][0]["seed"])
        ):
            candidate_records.extend(records)
            candidate_predictions.extend(float(value) for value in prediction)
            candidate_epsilons.extend(float(value) for value in epsilon)
            candidate_decisions.extend(str(value) for value in decisions)
            per_file.append(
                {
                    "seed": int(records[0]["seed"]),
                    "n": len(records),
                    "epsilon_min": float(np.min(epsilon)),
                    "epsilon_mean": float(np.mean(epsilon)),
                    "epsilon_max": float(np.max(epsilon)),
                    "score": score_decisions(records, decisions),
                }
            )
        score = _annotate_score(candidate_records, candidate_decisions)
        score["per_file"] = per_file
        score["kappa_sweep"] = _kappa_sweep(candidate_records, candidate_predictions, candidate_epsilons)
        score["radius_diagnostics"] = _radius_diagnostics(
            candidate_records,
            candidate_predictions,
            candidate_epsilons,
            candidate_decisions,
        )
        candidates[candidate] = score
        all_records.extend(candidate_records)
        all_predictions.extend(candidate_predictions)
        all_epsilons.extend(candidate_epsilons)
        all_decisions.extend(candidate_decisions)
    aggregate = _annotate_score(all_records, all_decisions)
    aggregate["kappa_sweep"] = _kappa_sweep(all_records, all_predictions, all_epsilons)
    aggregate["radius_diagnostics"] = _radius_diagnostics(all_records, all_predictions, all_epsilons, all_decisions)
    return {
        "rule": "per-candidate, per-seed exact-rank leave-one-condition-out KGA",
        "calibration_scope": (
            "cross-fitted empirical residual calibration; direct self-inclusion removed, "
            "exchangeability and independence not established"
        ),
        "alpha": ALPHA,
        "seeds": sorted(expected_seeds),
        "seed_scope": (
            "run/stream seeds conditional on archived checkpoint identities unless a track "
            "separately supplies independently trained checkpoint hashes"
        ),
        "evidence_contract": _evidence_contract(files, [row for path in files for row in _records(path)]),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "architecture_panel_aggregate": aggregate,
        "aggregate_scope": (
            "diagnostic average across separately calibrated candidates/backbones; not one deployment policy"
        ),
    }


def reconcile_grid_panels() -> dict[str, Any]:
    return {
        "imagenetc": {
            "protocol": "authoritative 27-cell x 5-seed panel",
            "panel": _grid_panel(SOURCE_ROOT / "imagenetc", expected_seeds={0, 1, 2, 3, 4}),
        },
        "imagenet_r": {
            "protocol": "IMAGENETR_PROTOCOL_D_v1, 12 cells x 4 seeds x 10 backbones",
            "panel": _grid_panel(SOURCE_ROOT / "imagenetr", expected_seeds={0, 1, 2, 3}),
            "claim_scope": "negative architecture-panel diagnostic; not a natural beats-both claim",
        },
    }


def _model_seed_robustness(
    source_paths: Sequence[Path],
    *,
    candidate: str,
    cal_seeds: set[int],
    test_seeds: set[int],
) -> dict[str, Any]:
    per_model_seed = []
    for model_seed, path in enumerate(source_paths):
        replay = _transfer_panel(
            path,
            path,
            candidate=candidate,
            cal_seeds=cal_seeds,
            test_seeds=test_seeds,
        )
        per_model_seed.append(
            {
                "model_seed": model_seed,
                "calibration": replay["calibration"],
                "exact_rank_transfer_score": replay["exact_rank_transfer_score"],
            }
        )
    aggregate: dict[str, Any] = {
        "n_model_seeds": len(per_model_seed),
        "regret_mean_std": {},
        "fa_u_max": float(max(row["exact_rank_transfer_score"]["fa_u"] for row in per_model_seed)),
        "all_tie_always_freeze": all(
            math.isclose(
                row["exact_rank_transfer_score"]["regret"]["kga"],
                row["exact_rank_transfer_score"]["regret"]["always_freeze"],
                abs_tol=1e-12,
            )
            for row in per_model_seed
        ),
        "any_point_beats_both": any(row["exact_rank_transfer_score"]["point_beats_both"] for row in per_model_seed),
    }
    for policy in ("kga", "always_adapt", "always_freeze"):
        values = np.asarray(
            [row["exact_rank_transfer_score"]["regret"][policy] for row in per_model_seed],
            dtype=float,
        )
        aggregate["regret_mean_std"][policy] = {
            "mean": float(np.mean(values)),
            "std_population": float(np.std(values)),
        }
    return {
        "unit": "independently trained base-model checkpoint",
        "per_model_seed": per_model_seed,
        "aggregate": aggregate,
        "source_provenance": _source_provenance(source_paths),
        "claim_scope": (
            "diagnostic model-seed robustness of the locked no-harm result; "
            "all model seeds select freeze, so this is not a beats-both result"
        ),
    }


def reconcile_missing_locked_panels() -> dict[str, Any]:
    cifar10c_files = _grid_files(SOURCE_ROOT / "cifar10c")
    cifar10c = _grid_panel(SOURCE_ROOT / "cifar10c", expected_seeds={0, 1, 2, 3, 4})

    camelyon_ood_source = SOURCE_ROOT / "camelyon17_ood/result_camelyon17_richz.json"
    camelyon_ood = _transfer_panel(
        camelyon_ood_source,
        camelyon_ood_source,
        candidate="eata_online",
        cal_seeds={0, 1},
        test_seeds={2, 3, 4},
        record_filters={"domain": "test"},
    )

    camelyon_b_files = _grid_files(SOURCE_ROOT / "camelyon17_b_v2")
    camelyon_b = _grid_panel(SOURCE_ROOT / "camelyon17_b_v2", expected_seeds={0, 1, 2})

    rxrx1_sources = [SOURCE_ROOT / f"rxrx1/modelseed{seed}.json" for seed in range(3)]
    rxrx1_primary = _transfer_panel(
        rxrx1_sources[0],
        rxrx1_sources[0],
        candidate="sar_online",
        cal_seeds={0, 1, 2, 3, 4},
        test_seeds={5, 6, 7, 8, 9},
    )

    cifar101_files = _grid_files(SOURCE_ROOT / "cifar101")
    cifar101 = _transfer_panel(
        cifar101_files,
        cifar101_files,
        candidate="tent",
        cal_seeds={0, 1, 2},
        test_seeds={3, 4},
    )

    return {
        "cifar10c": {
            "protocol": (
                "five-seed, 432-condition controlled stress grid; Tent/EATA from "
                "stress_grid_multiseed_v1 and SAR from the completed rebuild"
            ),
            "panel": cifar10c,
            "candidate_claim_scope": {
                "tent": "controlled mixed-regime result; not a natural-shift claim",
                "eata": "controlled result; no cluster-robust beats-both promotion",
                "sar": "completed negative rebuild; supersedes the incomplete historical SAR panel",
            },
            "headline_promotion": {
                "tent": "eligible only as a controlled-grid candidate result",
                "eata": "withheld from cluster-robust beats-both claims",
                "sar": "withheld; exact-rank rebuild loses to always-adapt",
            },
            "claim_scope": (
                "candidate-level controlled-grid evidence only; the aggregate averages "
                "separately calibrated adapters and is not one deployment policy"
            ),
            "source_provenance": _source_provenance(cifar10c_files),
        },
        "camelyon17": {
            "ood": {
                "protocol": (
                    "OOD test-domain-only replay; dev seeds 0-1 calibrate and test seeds "
                    "2-4 are scored once with EATA-online"
                ),
                "replay": camelyon_ood,
                "claim_scope": (
                    "archived opened OOD diagnostic that ties always-adapt on an all-helpful panel; "
                    "not prospective, not a beats-both result, and FA_u=0 is vacuous here"
                ),
                "headline_promotion": {
                    "eligible": False,
                    "reason": (
                        "the OOD evaluation was already opened and one fixed policy is "
                        "oracle-equivalent on every condition"
                    ),
                },
                "source_provenance": _source_provenance(camelyon_ood_source),
            },
            "b_v2_diagnostic": {
                "protocol": "three-seed, 36-condition-per-adapter within-seed LOO stress grid",
                "panel": camelyon_b,
                "claim_scope": (
                    "diagnostic cross-fitted stress result only; it is not an untouched "
                    "held-out natural-domain evaluation and no historical win is promoted"
                ),
                "headline_promotion": {
                    "eligible": False,
                    "reason": "within-seed diagnostic rather than untouched target-domain transfer",
                },
                "source_provenance": _source_provenance(camelyon_b_files),
            },
        },
        "rxrx1": {
            "protocol": (
                "Protocol J, SAR-online; dev stream seeds 0-4 calibrate and test stream seeds 5-9 are scored once"
            ),
            "primary_model_seed0": rxrx1_primary,
            "model_seed_robustness": _model_seed_robustness(
                rxrx1_sources,
                candidate="sar_online",
                cal_seeds={0, 1, 2, 3, 4},
                test_seeds={5, 6, 7, 8, 9},
            ),
            "claim_scope": (
                "held-out harmful-dominated no-harm result; KGA always freezes and ties "
                "always-freeze, so it is not a beats-both result"
            ),
            "headline_promotion": {
                "eligible": False,
                "reason": "KGA ties the freeze oracle on an all-harmful panel",
            },
            "source_provenance": _source_provenance(rxrx1_sources),
        },
        "cifar101": {
            "protocol": ("Protocol K, Tent; dev stream seeds 0-2 calibrate and test stream seeds 3-4 are scored once"),
            "replay": cifar101,
            "claim_scope": (
                "locked negative cross-seed diagnostic; the corrected exact-rank replay "
                "has FA_u=0 but ties always-freeze and does not beat both"
            ),
            "superseded_summary_note": (
                "the archived analyze_F_results.json used an older scoring state and reports "
                "adapt decisions; this source-record replay uses the released exact-rank rule"
            ),
            "headline_promotion": {
                "eligible": False,
                "reason": "exact-rank KGA ties always-freeze rather than beating both",
            },
            "source_provenance": _source_provenance(cifar101_files),
        },
    }


def _pacs_seed_payload(path: Path) -> dict[str, Any]:
    data = _load(path)
    data.pop("_provenance", None)
    return data


def reconcile_pacs() -> dict[str, Any]:
    aggregate = _pacs_seed_payload(SOURCE_ROOT / "pacs/PACS_MULTISEED_RESULTS.json")
    seed_runs = [_pacs_seed_payload(SOURCE_ROOT / f"pacs/pacs_seed{seed}.json") for seed in (0, 1, 2)]
    domains = sorted(aggregate["per_domain"])
    mismatches: list[dict[str, Any]] = []
    for domain in domains:
        aggregate_row = aggregate["per_domain"][domain]
        for seed, run in enumerate(seed_runs):
            seed_row = run["per_domain"][domain]
            checks = {
                "kga": aggregate_row["regret_K_Bound"]["per_seed"][seed],
                "always_adapt": aggregate_row["regret_always_adapt"]["per_seed"][seed],
                "always_freeze": aggregate_row["regret_always_freeze"]["per_seed"][seed],
                "fa_u": aggregate_row["FA_u"]["per_seed"][seed],
                "adapt_rate": aggregate_row["adapt_rate"]["per_seed"][seed],
                "decision_coverage": aggregate_row["coverage"]["per_seed"][seed],
            }
            expected = {
                "kga": seed_row["regret"]["K_Bound"],
                "always_adapt": seed_row["regret"]["always_adapt"],
                "always_freeze": seed_row["regret"]["always_freeze"],
                "fa_u": seed_row["FA_u"],
                "adapt_rate": seed_row["adapt_rate"],
                "decision_coverage": seed_row["coverage"],
            }
            for metric in checks:
                if not math.isclose(float(checks[metric]), float(expected[metric]), abs_tol=1e-12):
                    mismatches.append(
                        {
                            "domain": domain,
                            "seed": seed,
                            "metric": metric,
                            "aggregate": checks[metric],
                            "seed_file": expected[metric],
                        }
                    )
    seed_metrics: list[dict[str, Any]] = []
    for seed, run in enumerate(seed_runs):
        rows = [run["per_domain"][domain] for domain in domains]
        seed_metrics.append(
            {
                "seed": seed,
                "regret": {
                    policy: float(np.mean([row["regret"]["K_Bound" if policy == "kga" else policy] for row in rows]))
                    for policy in ("kga", "always_adapt", "always_freeze")
                },
                "fa_u": float(np.mean([row["FA_u"] for row in rows])),
                "adapt_rate": float(np.mean([row["adapt_rate"] for row in rows])),
                "decision_coverage": float(np.mean([row["coverage"] for row in rows])),
            }
        )
    pooled: dict[str, Any] = {
        "n_domain_seed_units": len(seed_metrics) * len(domains),
        "regret": {
            policy: float(np.mean([row["regret"][policy] for row in seed_metrics]))
            for policy in ("kga", "always_adapt", "always_freeze")
        },
        "fa_u": float(np.mean([row["fa_u"] for row in seed_metrics])),
        "adapt_rate": float(np.mean([row["adapt_rate"] for row in seed_metrics])),
        "decision_coverage": float(np.mean([row["decision_coverage"] for row in seed_metrics])),
    }
    pooled["point_beats_both"] = bool(
        pooled["regret"]["kga"] < pooled["regret"]["always_adapt"]
        and pooled["regret"]["kga"] < pooled["regret"]["always_freeze"]
    )
    return {
        "protocol": "PACS locked three-seed aggregate, four held-out domains",
        "seeds": [0, 1, 2],
        "domains": domains,
        "aggregate_matches_seed_files": not mismatches,
        "mismatches": mismatches,
        "per_seed": seed_metrics,
        "pooled_domain_seed_mean": pooled,
        "decision_replay_available": False,
        "decision_replay_blocker": (
            "archived per-cell files omit b_hat and the calibration-domain residual records; "
            "the saved seed summaries can be cross-validated but the gate cannot be rerun"
        ),
        "claim_scope": "null diagnostic only",
    }


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"


def _panel_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    office = result["panels"]["officehome"]
    for name, key in (("Office-Home M-v2", "primary"), ("Office-Home replication", "test_stream_seed_replication")):
        score = office[key]["exact_rank_transfer_score"]
        rows.append({"name": name, "score": score, "replay": "yes", "scope": "descriptive/A7 open"})
    iwild_panel = result["panels"]["iwildcam"]
    iwild = iwild_panel["primary"]["exact_rank_transfer_score"]
    rows.append(
        {
            "name": "iWildCam H-v2",
            "score": iwild,
            "replay": "archived only",
            "scope": "withheld: official-metric rerun required",
            "withheld": not iwild_panel.get("release_promotion", {}).get("eligible", False),
        }
    )
    imagenetc = result["panels"]["imagenetc"]["panel"]["candidates"]
    for candidate in ("sar", "tent", "eata"):
        rows.append(
            {
                "name": f"ImageNet-C {candidate.upper()}",
                "score": imagenetc[candidate],
                "replay": "yes",
                "scope": "corrected LOO grid",
            }
        )
    pacs = result["panels"]["pacs"]["pooled_domain_seed_mean"]
    rows.append({"name": "PACS", "score": pacs, "replay": "aggregate only", "scope": "null diagnostic"})
    imagenetr = result["panels"]["imagenet_r"]["panel"]["architecture_panel_aggregate"]
    rows.append({"name": "ImageNet-R", "score": imagenetr, "replay": "yes", "scope": "architecture diagnostic"})
    cifar10c = result["panels"]["cifar10c"]["panel"]["candidates"]
    for candidate in ("tent", "eata", "sar"):
        rows.append(
            {
                "name": f"CIFAR-10-C {candidate.upper()}",
                "score": cifar10c[candidate],
                "replay": "yes",
                "scope": "controlled exact-rank grid",
            }
        )
    camelyon_ood = result["panels"]["camelyon17"]["ood"]["replay"]["exact_rank_transfer_score"]
    rows.append(
        {
            "name": "Camelyon17 OOD",
            "score": camelyon_ood,
            "replay": "yes",
            "scope": "archived opened OOD diagnostic",
        }
    )
    camelyon_b = result["panels"]["camelyon17"]["b_v2_diagnostic"]["panel"]["candidates"]
    for candidate in ("tent", "eata", "sar"):
        rows.append(
            {
                "name": f"Camelyon17 B-v2 {candidate.upper()}",
                "score": camelyon_b[candidate],
                "replay": "yes",
                "scope": "within-seed diagnostic",
            }
        )
    rxrx1 = result["panels"]["rxrx1"]["primary_model_seed0"]["exact_rank_transfer_score"]
    rows.append(
        {
            "name": "RxRx1 J",
            "score": rxrx1,
            "replay": "yes",
            "scope": "held-out no-harm",
        }
    )
    cifar101 = result["panels"]["cifar101"]["replay"]["exact_rank_transfer_score"]
    rows.append(
        {
            "name": "CIFAR-10.1 K",
            "score": cifar101,
            "replay": "yes",
            "scope": "negative cross-seed diagnostic",
        }
    )
    return rows


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Canonical Reconciled Result Panels",
        "",
        "Generated by `scripts/reconcile_result_panels.py` from compact, source-hashed artifacts.",
        "Regret order is KGA / always-adapt / always-freeze. Abstention keeps the frozen model.",
        "",
        "| Panel | N | KGA | Adapt | Freeze | FA_u | Adapt rate | Coverage | Replay | Scope |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in _panel_rows(result):
        score = row["score"]
        regret = score["regret"]
        if row.get("withheld"):
            lines.append(
                f"| {row['name']} | withheld | withheld | withheld | withheld | withheld | withheld | withheld "
                f"| {row['replay']} | {row['scope']} |"
            )
            continue
        lines.append(
            f"| {row['name']} | {score.get('n', score.get('n_domain_seed_units', '--'))} "
            f"| {_fmt(regret['kga'])} | {_fmt(regret['always_adapt'])} "
            f"| {_fmt(regret['always_freeze'])} | {_fmt(score.get('fa_u'))} "
            f"| {_fmt(score.get('adapt_rate'))} | {_fmt(score.get('decision_coverage'))} "
            f"| {row['replay']} | {row['scope']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Office-Home is numerically reconciled from its saved per-condition records, but remains descriptive because no predeclared uniform A7 full-fit-versus-LOO stability bound was archived.",
            "- iWildCam is withheld from the release-level numerical panel because its archived records used sklearn macro-F1 rather than the official WILDS metric. Historical and cross-fitted values under that invalid contract remain hash-locked for audit only; a pinned official-metric rerun is required.",
            "- ImageNet-C is recomputed with exact-rank, leave-one-condition-out radii. It is a corrected controlled-grid panel, not a natural-shift claim.",
            "- PACS seed summaries agree with the three-seed aggregate, but the absent `b_hat` and calibration residuals prevent decision replay.",
            "- ImageNet-R is replayed per backbone and seed. Its aggregate is an architecture-panel diagnostic, not one deployable policy and not a beats-both result.",
            "- CIFAR-10-C uses the completed SAR rebuild alongside the original Tent/EATA stress-grid records. Candidate scores are separate policies; their aggregate is diagnostic only.",
            "- Camelyon17 OOD is an archived, already-opened, all-helpful diagnostic; RxRx1 J is a no-harm result that ties always-freeze. Camelyon17 B-v2 is retained separately as a within-seed diagnostic.",
            "- CIFAR-10.1 K is the locked cross-seed replay. It is retained as a negative diagnostic and is not promoted from historical within-seed summaries.",
            "",
        ]
    )
    return "\n".join(lines)


def render_latex(result: dict[str, Any]) -> str:
    lines = [
        "% Generated by scripts/reconcile_result_panels.py; do not edit manually.",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Panel & $n$ & KGA & Adapt & Freeze & $\\mathrm{FA}_u$ & Coverage \\\\",
        "\\midrule",
    ]
    for row in _panel_rows(result):
        score = row["score"]
        regret = score["regret"]
        name = row["name"].replace("-", "--")
        n = score.get("n", score.get("n_domain_seed_units", "--"))
        if row.get("withheld"):
            lines.append(
                f"{name} & \\multicolumn{{6}}{{c}}{{withheld: official-metric rerun required}} \\\\"
            )
            continue
        lines.append(
            f"{name} & {n} & {_fmt(regret['kga'])} & {_fmt(regret['always_adapt'])} & "
            f"{_fmt(regret['always_freeze'])} & {_fmt(score.get('fa_u'))} & "
            f"{_fmt(score.get('decision_coverage'))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def reconcile(*, reuse_transfer: bool = False) -> dict[str, Any]:
    manifest = _load(RESULT_ROOT / "source_manifest.json")
    panels = {}
    if reuse_transfer:
        existing = _load(RESULT_ROOT / "canonical_panel_results.json")["panels"]
        panels["officehome"] = existing["officehome"]
        panels["iwildcam"] = existing["iwildcam"]
    else:
        panels.update(reconcile_transfer_panels())
    panels.update(reconcile_grid_panels())
    panels["pacs"] = reconcile_pacs()
    panels.update(reconcile_missing_locked_panels())
    return {
        "schema": "kbound-canonical-panel-results-v2",
        "generator": "scripts/reconcile_result_panels.py",
        "generator_sha256": _sha256(Path(__file__)),
        "alpha": ALPHA,
        "decision_rule": "ADAPT iff delta_hat-epsilon>0; FREEZE iff delta_hat+epsilon<0; otherwise ABSTAIN",
        "abstention_semantics": "retain frozen model",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "source_manifest_sha256": _sha256(RESULT_ROOT / "source_manifest.json"),
        "source_file_count": manifest["file_count"],
        "panels": panels,
    }


def write_outputs(result: dict[str, Any]) -> None:
    _write_json(RESULT_ROOT / "canonical_panel_results.json", result)
    (RESULT_ROOT / "CANONICAL_PANEL_RESULTS.md").write_text(render_markdown(result))
    (RESULT_ROOT / "canonical_panel_table.tex").write_text(render_latex(result))


def remove_appledouble_files() -> None:
    """Remove macOS sidecar metadata from the generated release tree."""
    for path in RESULT_ROOT.rglob("._*"):
        if path.is_file():
            path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--import-from",
        type=Path,
        help="archive checkout containing experiments/kbound/results",
    )
    parser.add_argument(
        "--clean-import",
        action="store_true",
        help="remove only the generated compact source directory before import",
    )
    parser.add_argument(
        "--reuse-transfer",
        action="store_true",
        help="reuse the existing Office-Home/iWildCam transfer replay and refresh grid panels only",
    )
    args = parser.parse_args()
    runtime_mismatch = []
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        runtime_mismatch.append(f"numpy={np.__version__} (expected {EXPECTED_NUMPY_VERSION})")
    if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
        runtime_mismatch.append(f"scikit-learn={sklearn.__version__} (expected {EXPECTED_SKLEARN_VERSION})")
    if runtime_mismatch:
        parser.error(
            "canonical reconciliation requires the analysis runtime pinned in "
            "requirements.lock.txt; " + ", ".join(runtime_mismatch)
        )
    if args.import_from:
        if args.clean_import and SOURCE_ROOT.exists():
            # ExFAT/APFS metadata races can make a directory entry disappear
            # between scandir and unlink. This tree is generated output only.
            shutil.rmtree(SOURCE_ROOT, ignore_errors=True)
        manifest = import_sources(args.import_from.resolve())
        print(f"Imported {manifest['file_count']} source artifacts into {SOURCE_ROOT}")
    if not (RESULT_ROOT / "source_manifest.json").is_file():
        parser.error("source manifest missing; run once with --import-from ARCHIVE_ROOT")
    manifest_path = RESULT_ROOT / "source_manifest.json"
    manifest = _load(manifest_path)
    generator_hash = _sha256(Path(__file__))
    if manifest.get("generator_sha256") != generator_hash:
        manifest["generator"] = "scripts/reconcile_result_panels.py"
        manifest["generator_sha256"] = generator_hash
        _write_json(manifest_path, manifest)
    if args.reuse_transfer and not (RESULT_ROOT / "canonical_panel_results.json").is_file():
        parser.error("--reuse-transfer requires an existing canonical_panel_results.json")
    result = reconcile(reuse_transfer=args.reuse_transfer)
    write_outputs(result)
    remove_appledouble_files()
    print(f"Wrote {RESULT_ROOT / 'canonical_panel_results.json'}")
    print(f"Wrote {RESULT_ROOT / 'CANONICAL_PANEL_RESULTS.md'}")
    print(f"Wrote {RESULT_ROOT / 'canonical_panel_table.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
