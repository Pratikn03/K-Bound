#!/usr/bin/env python3
"""Forensic, read-only audit of the KBOUND empirical result ecosystem.

This script does not edit source results, tune thresholds, or promote claims.  It
recomputes integrity, provenance, feasibility, and metric checks from archived
files and writes bounded audit tables for the companion notebook/report.

Run from the repository root with the environment used for reconciliation:

    .venv/bin/python \
      docs/research/kbound/scripts/audit_empirical_data_quality_2026_08_27.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import re
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "experiments/kbound/results"
CANONICAL_DIR = RESULTS / "reconciled_panels_v1"
CANONICAL_PATH = CANONICAL_DIR / "canonical_panel_results.json"
SOURCE_MANIFEST_PATH = CANONICAL_DIR / "source_manifest.json"
OUT_DEFAULT = ROOT / "docs/research/kbound/audits/empirical_data_quality_2026_08_27"
QUARANTINE_MANIFEST_PATH = OUT_DEFAULT / "quarantine_manifest.json"
GENERATED_AT = "2026-08-27T00:00:00-05:00"
CONTROLLED_HOLM_STATUS = (
    "RETROSPECTIVE_HOLM_OVER_SIX_PROSPECTIVELY_NAMED_CONTRASTS_FAILED"
)
CONTROLLED_HOLM_BASIS = (
    "Tent has a beats-both point estimate and positive ordinary six-family intervals, "
    "but p-values from retrospective Holm adjustment over the six prospectively named "
    "contrasts are both 0.09375. The analysis is retrospective and non-confirmatory; "
    "EATA and SAR also fail that gate, and independent-checkpoint inference is unavailable."
)
OUTER_SEAL_STATUS = "CONTROL_IMPLEMENTED_VERIFIED_BY_FINAL_RELEASE_GATE"


def outer_seal_remediation() -> dict[str, str]:
    """Describe the non-self-referential outer checksum control."""

    return {
        "remediation_status": OUTER_SEAL_STATUS,
        "remediation_action": (
            "The release runner writes and independently verifies the outer checksum seal "
            "after all audit, manuscript, PDF, DOCX, and source-seal outputs are final."
        ),
        "verification_evidence": (
            "docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt; "
            "docs/research/kbound/scripts/verify_release_checksums.py; "
            "docs/research/kbound/runbooks/release_candidate.sh"
        ),
        "release_disposition": "PASS_ONLY_IF_FINAL_RELEASE_CHECKSUM_GATE_PASSES",
        "remaining_requirement": (
            "The final release_candidate.sh checksum stage must pass; this audit artifact "
            "does not self-certify the checksum file that binds it."
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_json_strict(path: Path) -> Any:
    def reject(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token}")

    document = json.loads(path.read_text(), parse_constant=reject)
    if any(isinstance(value, float) and not math.isfinite(value) for value in walk(document)):
        raise ValueError(f"decoded JSON contains a non-finite numeric value: {path}")
    return document


def json_token(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if len(fields) != len(set(fields)):
        duplicates = sorted(name for name, count in Counter(fields).items() if count > 1)
        raise ValueError(f"duplicate CSV field names are forbidden: {duplicates}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def is_finite_number(value: Any) -> bool:
    return not isinstance(value, float) or math.isfinite(value)


def walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def dataset_from_destination(destination: str) -> str:
    parts = Path(destination).parts
    try:
        return parts[parts.index("source") + 1]
    except (ValueError, IndexError):
        return "unknown"


KEY_FIELDS = (
    "model_seed",
    "seed",
    "stream_seed",
    "domain",
    "location",
    "split",
    "comp",
    "regime",
    "aggr",
    "mode",
    "method",
    "candidate",
    "condition",
    "architecture",
    "backbone",
    "corruption",
    "severity",
)


def source_bundle_audit(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    original_hash_failures = 0
    compact_hash_failures = 0
    original_size_failures = 0
    compact_size_failures = 0
    row_count_failures = 0
    strict_json_failures = 0
    exact_duplicate_records = 0
    duplicate_dimension_keys = 0
    b_identity_failures = 0
    core_nonfinite_values = 0
    out_of_range_scores = 0
    missing_metric_records = 0
    record_docs = 0
    records_total = 0
    docs_missing_metadata_dataset = 0
    records_missing_metadata_dataset = 0
    dataset_counts: Counter[str] = Counter()
    dataset_profile: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "dataset": "",
            "source_files": 0,
            "record_files": 0,
            "records": 0,
            "missing_metric_records": 0,
            "missing_metadata_dataset_records": 0,
            "strict_json_failures": 0,
        }
    )

    for spec in manifest["files"]:
        source = ROOT / spec["source"]
        destination = ROOT / spec["destination"]
        dataset = dataset_from_destination(spec["destination"])
        profile = dataset_profile[dataset]
        profile["dataset"] = dataset
        profile["source_files"] += 1

        if sha256(source) != spec["original_sha256"]:
            original_hash_failures += 1
        if source.stat().st_size != spec["original_bytes"]:
            original_size_failures += 1
        if sha256(destination) != spec["compact_sha256"]:
            compact_hash_failures += 1
        if destination.stat().st_size != spec["compact_bytes"]:
            compact_size_failures += 1

        try:
            document = load_json_strict(destination)
        except (ValueError, json.JSONDecodeError):
            strict_json_failures += 1
            profile["strict_json_failures"] += 1
            continue

        records = document.get("records") if isinstance(document, dict) else None
        if not isinstance(records, list):
            if int(spec.get("records", 0)) != 0:
                row_count_failures += 1
            continue

        record_docs += 1
        profile["record_files"] += 1
        n_records = len(records)
        records_total += n_records
        dataset_counts[dataset] += n_records
        profile["records"] += n_records
        if n_records != int(spec.get("records", -1)):
            row_count_failures += 1

        metadata = document.get("metadata") or {}
        metadata_dataset_missing = not metadata.get("dataset")
        if metadata_dataset_missing:
            docs_missing_metadata_dataset += 1
            records_missing_metadata_dataset += n_records
            profile["missing_metadata_dataset_records"] += n_records

        tokens: Counter[str] = Counter()
        keys: Counter[tuple[tuple[str, str], ...]] = Counter()
        for record in records:
            tokens[json_token(record)] += 1
            context = {
                "metadata_dataset": metadata.get("dataset"),
                "metadata_seed": metadata.get("seed"),
            }
            key = tuple(
                (name, json_token(record.get(name, context.get(name))))
                for name in KEY_FIELDS
                if name in record
            )
            if key:
                keys[key] += 1

            if "metric" not in record:
                missing_metric_records += 1
                profile["missing_metric_records"] += 1
            if "B" in record and "a0" in record and ("a_adapted" in record or "aa" in record):
                adapted = record.get("a_adapted", record.get("aa"))
                if all(isinstance(x, (int, float)) for x in (record["B"], record["a0"], adapted)):
                    if abs(float(record["B"]) - (float(adapted) - float(record["a0"]))) > 1e-10:
                        b_identity_failures += 1
            for name in ("B", "a0", "a_adapted", "aa", "b_hat", "eps_conformal"):
                value = record.get(name)
                if isinstance(value, (int, float)) and not is_finite_number(value):
                    core_nonfinite_values += 1
            for name in ("a0", "a_adapted", "aa"):
                value = record.get(name)
                if isinstance(value, (int, float)) and not (0.0 <= float(value) <= 1.0):
                    out_of_range_scores += 1

        exact_duplicate_records += sum(count - 1 for count in tokens.values() if count > 1)
        duplicate_dimension_keys += sum(count - 1 for count in keys.values() if count > 1)

    profile_rows = sorted(dataset_profile.values(), key=lambda row: row["dataset"])
    summary = {
        "manifest_file_count": len(manifest["files"]),
        "record_file_count": record_docs,
        "summary_artifact_count": len(manifest["files"]) - record_docs,
        "records_total": records_total,
        "dataset_record_counts": dict(sorted(dataset_counts.items())),
        "original_hash_failures": original_hash_failures,
        "compact_hash_failures": compact_hash_failures,
        "original_size_failures": original_size_failures,
        "compact_size_failures": compact_size_failures,
        "row_count_failures": row_count_failures,
        "strict_json_failures": strict_json_failures,
        "exact_duplicate_records_within_file": exact_duplicate_records,
        "duplicate_dimension_keys_within_file": duplicate_dimension_keys,
        "benefit_identity_failures": b_identity_failures,
        "core_nonfinite_values": core_nonfinite_values,
        "out_of_range_score_values": out_of_range_scores,
        "missing_metric_records": missing_metric_records,
        "missing_metric_rate": missing_metric_records / records_total,
        "docs_missing_metadata_dataset": docs_missing_metadata_dataset,
        "records_missing_metadata_dataset": records_missing_metadata_dataset,
        "records_missing_metadata_dataset_rate": records_missing_metadata_dataset / records_total,
    }
    return summary, profile_rows


def recursive_score_audit(document: Any) -> dict[str, Any]:
    score_nodes = 0
    problems: list[str] = []

    def visit(value: Any, path: str = "root") -> None:
        nonlocal score_nodes
        if isinstance(value, dict):
            required = {"n", "adapt_count", "freeze_count", "abstain_count"}
            if required.issubset(value):
                score_nodes += 1
                n = int(value["n"])
                adapt = int(value["adapt_count"])
                freeze = int(value["freeze_count"])
                abstain = int(value["abstain_count"])
                false = int(value.get("false_adapt_count", 0))
                if adapt + freeze + abstain != n:
                    problems.append(f"{path}: action counts do not sum to n")
                if "adapt_rate" in value and abs(float(value["adapt_rate"]) - adapt / n) > 1e-10:
                    problems.append(f"{path}: adapt_rate mismatch")
                if "decision_coverage" in value and abs(float(value["decision_coverage"]) - (adapt + freeze) / n) > 1e-10:
                    problems.append(f"{path}: decision_coverage mismatch")
                if "yield" in value and abs(float(value["yield"]) - (adapt + freeze) / n) > 1e-10:
                    problems.append(f"{path}: yield mismatch")
                if "fa_u" in value and value["fa_u"] is not None and abs(float(value["fa_u"]) - false / n) > 1e-10:
                    problems.append(f"{path}: fa_u mismatch")
                if "fa_c" in value and value["fa_c"] is not None and adapt > 0:
                    if abs(float(value["fa_c"]) - false / adapt) > 1e-10:
                        problems.append(f"{path}: fa_c mismatch")
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(document)
    return {"score_nodes_checked": score_nodes, "problem_count": len(problems), "problems": problems}


def sign(value: float) -> int:
    return int(value > 0) - int(value < 0)


def route_b_archive_audit() -> dict[str, Any]:
    counters: Counter[str] = Counter()
    files_with_route = 0
    for path in sorted(RESULTS.glob("**/result_*.json")):
        try:
            document = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        local_cells = 0
        for condition in document.get("conditions", []):
            route = condition.get("route") or {}
            b_hat = route.get("b_hat")
            b_tilde = route.get("b_tilde")
            if not isinstance(b_hat, list) or not isinstance(b_tilde, list) or not b_hat or len(b_hat) != len(b_tilde):
                continue
            local_cells += 1
            counters["route_cells"] += 1
            if any(sign(float(left)) != sign(float(right)) for left, right in zip(b_hat, b_tilde)):
                counters["b_hat_b_tilde_sign_disagreement_cells"] += 1
            if any(abs(float(value)) > 1.0 + 1e-12 for value in b_hat):
                counters["b_hat_outside_unit_interval_cells"] += 1
            if route.get("gate_pass"):
                counters["gate_pass_cells"] += 1
                if float(b_hat[0]) < 0:
                    counters["gate_pass_negative_anchor_cells"] += 1
            if route.get("decision") == "ADAPT":
                counters["adapt_cells"] += 1
                if float(b_hat[0]) < 0:
                    counters["adapt_negative_anchor_cells"] += 1
                choice = route.get("choice")
                values = condition.get("aa_all")
                benefit = None
                if isinstance(values, list) and isinstance(choice, int) and 0 <= choice < len(values):
                    benefit = float(values[choice]) - float(values[0])
                elif isinstance(condition.get("true_B_selected"), (int, float)):
                    benefit = float(condition["true_B_selected"])
                elif isinstance(condition.get("B"), (int, float)):
                    benefit = float(condition["B"])
                if benefit is not None and benefit <= 0:
                    counters["adapt_nonpositive_realized_benefit_cells"] += 1
                if benefit is not None and benefit < 0:
                    counters["adapt_negative_realized_benefit_cells"] += 1
        files_with_route += bool(local_cells)
    counters["files_with_route_cells"] = files_with_route
    return dict(counters)


def condition_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("model_seed"),
        record.get("seed"),
        record.get("domain"),
        record.get("location"),
        record.get("split"),
        record.get("comp"),
        record.get("regime"),
        record.get("aggr"),
    )


def focused_officehome_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = RESULTS / "natural_replication_strengthening_v1/officehome_focused_multicandidate"
    paths = sorted(base.glob("**/result_*.json"))
    route_conditions: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    checkpoints: set[str] = set()
    literal_infinity = 0
    strict_failures = 0
    checkpoint_rows: list[dict[str, Any]] = []
    exact_eata_sar = 0

    for path in paths:
        text = path.read_text()
        literal_infinity += len(re.findall(r"(?<![A-Za-z])Infinity(?![A-Za-z])", text))
        try:
            load_json_strict(path)
        except (ValueError, json.JSONDecodeError):
            strict_failures += 1
        document = load_json(path)
        checkpoint = document["f0_checkpoint_sha256"]
        checkpoints.add(checkpoint)
        file_conditions = document.get("conditions", [])
        file_records = document.get("records", [])
        route_conditions.extend(file_conditions)
        records.extend(file_records)

        predictions: dict[tuple[Any, ...], dict[str, list[int]]] = defaultdict(dict)
        for record in file_records:
            predictions[condition_key(record)][record["candidate"]] = record.get("preds", [])
        file_exact = 0
        for candidates in predictions.values():
            if (
                "eata_online_aggressive" in candidates
                and "sar_online_aggressive" in candidates
                and candidates["eata_online_aggressive"] == candidates["sar_online_aggressive"]
            ):
                file_exact += 1
        exact_eata_sar += file_exact

        freeze = np.asarray([float(row["a0"]) for row in file_conditions])
        adapted = np.asarray([list(map(float, row["aa_all"][1:])) for row in file_conditions])
        sar_index = file_conditions[0]["cand_names"].index("sar_online_aggressive") - 1
        sar = adapted[:, sar_index]
        oracle = np.maximum(freeze, adapted.max(axis=1))
        checkpoint_rows.append(
            {
                "model_seed": int(document["model_seed"]),
                "checkpoint_sha256": checkpoint,
                "conditions": len(file_conditions),
                "mean_freeze_accuracy": float(freeze.mean()),
                "mean_sar_accuracy": float(sar.mean()),
                "mean_oracle_accuracy": float(oracle.mean()),
                "sar_minus_freeze": float((sar - freeze).mean()),
                "oracle_minus_sar": float((oracle - sar).mean()),
                "eata_sar_exact_prediction_cells": file_exact,
            }
        )

    actions = Counter((row.get("route") or {}).get("decision", "MISSING") for row in route_conditions)
    tau = [float(row["route"]["tau"]) for row in route_conditions]
    negative_anchor = sum(float(row["route"]["anchor_b0"]) < 0 for row in route_conditions)
    outside_cells = sum(any(abs(float(x)) > 1.0 + 1e-12 for x in row["route"]["b_hat"]) for row in route_conditions)
    outside_values = sum(
        abs(float(value)) > 1.0 + 1e-12
        for row in route_conditions
        for value in row["route"]["b_hat"]
    )
    freeze = np.asarray([float(row["a0"]) for row in route_conditions])
    adapted = np.asarray([list(map(float, row["aa_all"][1:])) for row in route_conditions])
    names = route_conditions[0]["cand_names"][1:]
    mean_candidate = dict(zip(names, adapted.mean(axis=0), strict=True))
    sar = adapted[:, names.index("sar_online_aggressive")]
    oracle = np.maximum(freeze, adapted.max(axis=1))
    summary = {
        "result_files": len(paths),
        "distinct_checkpoint_hashes": len(checkpoints),
        "candidate_records": len(records),
        "route_conditions": len(route_conditions),
        "route_action_counts": dict(actions),
        "tau_min": min(tau),
        "tau_max": max(tau),
        "locked_tau_star": 0.52,
        "negative_anchor_cells": negative_anchor,
        "b_hat_outside_unit_interval_cells": outside_cells,
        "b_hat_outside_unit_interval_values": outside_values,
        "b_hat_values_total": sum(len(row["route"]["b_hat"]) for row in route_conditions),
        "eata_sar_exact_prediction_cells": exact_eata_sar,
        "single_candidate_calibration_cells_per_checkpoint": 3,
        "minimum_total_cells_for_alpha_0_10_exact_rank_loo": 10,
        "single_candidate_route_feasible": False,
        "literal_infinity_values": literal_infinity,
        "strict_json_files_failed": strict_failures,
        "mean_accuracy": {
            "freeze_or_kga": float(freeze.mean()),
            **{name: float(value) for name, value in mean_candidate.items()},
            "oracle": float(oracle.mean()),
        },
        "regret": {
            "kga": float((oracle - freeze).mean()),
            "sar": float((oracle - sar).mean()),
        },
    }
    return summary, sorted(checkpoint_rows, key=lambda row: row["model_seed"])


def resume_contamination_audit() -> dict[str, Any]:
    test_path = RESULTS / "officehome_kbound_run/result_target_test_d2f4bf2c.json"
    val_path = RESULTS / "officehome_kbound_run/result_target_val_ce1e4380.json"
    test = load_json(test_path)
    val = load_json(val_path)

    def role_counts(document: dict[str, Any]) -> Counter[str]:
        out: Counter[str] = Counter()
        for row in document["conditions"]:
            domain = row.get("domain")
            split = row.get("split")
            if domain == "Real_World" and split == "val":
                out["source_validation"] += 1
            elif split == "val":
                out["target_validation"] += 1
            elif split == "test":
                out["target_test"] += 1
            else:
                out["other"] += 1
        return out

    test_counts = role_counts(test)
    val_counts = role_counts(val)
    non_test = len(test["conditions"]) - test_counts["target_test"]
    return {
        "declared_target_test_file": str(test_path.relative_to(ROOT)),
        "declared_target_test_conditions": len(test["conditions"]),
        "declared_target_test_role_counts": dict(test_counts),
        "non_test_conditions_in_target_test_artifact": non_test,
        "non_test_fraction": non_test / len(test["conditions"]),
        "declared_target_validation_file": str(val_path.relative_to(ROOT)),
        "declared_target_validation_conditions": len(val["conditions"]),
        "declared_target_validation_role_counts": dict(val_counts),
    }


def multiseed_lineage_audit() -> dict[str, Any]:
    office_dir = RESULTS / "multiseed/officehome/extracted"
    office_files = sorted(office_dir.glob("per_condition_officehome_sar_online_aggressive_seed*.json"))
    office_aggregate_path = office_dir / "multiseed_officehome_sar_online_aggressive.json"
    iwild_dir = RESULTS / "multiseed/iwildcam/extracted"
    iwild_manifest_path = iwild_dir / "extract_manifest_iwildcam.json"
    iwild_aggregate_path = iwild_dir / "multiseed_iwildcam_tent_episodic.json"
    expected_derived = [
        *office_files,
        office_aggregate_path,
        iwild_manifest_path,
        iwild_aggregate_path,
    ]
    if not office_files and not any(path.exists() for path in expected_derived):
        quarantine = load_json_strict(QUARANTINE_MANIFEST_PATH)
        historical = quarantine.get("historical_evidence")
        if not isinstance(historical, dict):
            raise ValueError(
                "quarantined multiseed artifacts are absent but their historical evidence is missing"
            )
        quarantined_paths = {row.get("path") for row in quarantine.get("artifacts", [])}
        required_paths = {
            "experiments/kbound/results/multiseed/officehome/extracted/"
            "multiseed_officehome_sar_online_aggressive.json",
            "experiments/kbound/results/multiseed/iwildcam/extracted/"
            "extract_manifest_iwildcam.json",
            "experiments/kbound/results/multiseed/iwildcam/extracted/"
            "multiseed_iwildcam_tent_episodic.json",
        }
        if not required_paths.issubset(quarantined_paths):
            raise ValueError("quarantine manifest does not cover all invalid multiseed derivatives")
        return {
            "status": "quarantined_invalid_derived_artifacts",
            "current_release_artifacts_present": False,
            "raw_sources_deleted": bool(quarantine.get("raw_sources_deleted", True)),
            "officehome": {
                **historical["officehome"],
                "status": "historical_invalid_evidence_quarantined",
            },
            "iwildcam": {
                **historical["iwildcam"],
                "status": "historical_invalid_evidence_quarantined",
            },
        }

    office_rows = []
    total_rows = 0
    unique_seed_condition: set[tuple[Any, ...]] = set()
    oracle_matches = 0
    for path in office_files:
        document = load_json(path)
        records = document["records"]
        conditions = [record.get("condition") for record in records]
        total_rows += len(records)
        unique_seed_condition.update((record.get("seed"), record.get("condition")) for record in records)
        oracle_matches += sum(record.get("kga_decision") == record.get("oracle_action") for record in records)
        office_rows.append(
            {
                "file": path.name,
                "seed": document.get("seed"),
                "rows": len(records),
                "unique_conditions": len(set(conditions)),
            }
        )

    office_aggregate = load_json(office_aggregate_path)
    iwild_manifest = load_json(iwild_manifest_path)
    iwild_aggregate = load_json(iwild_aggregate_path)
    source_path = next((RESULTS / "multiseed/iwildcam").glob("**/result_0e82b624.json"))
    source_candidates = sorted({row.get("candidate") for row in load_json(source_path)["records"]})
    return {
        "status": "invalid_derived_artifacts_present",
        "current_release_artifacts_present": True,
        "officehome": {
            "files": office_rows,
            "rows": total_rows,
            "unique_seed_condition_keys": len(unique_seed_condition),
            "duplicate_rows": total_rows - len(unique_seed_condition),
            "oracle_action_matches": oracle_matches,
            "reported_regret_kga": office_aggregate.get("regret_kga"),
            "reported_verdict": office_aggregate.get("verdict"),
            "reported_inference_unit": office_aggregate.get("inference_unit"),
            "source_checkpoint_identity": "same seed-0 checkpoint path; no model_seed or checkpoint SHA-256 recorded",
        },
        "iwildcam": {
            "current_source": str(source_path.relative_to(ROOT)),
            "current_source_candidates": source_candidates,
            "requested_candidate": "tent_episodic",
            "current_serialization_written": iwild_manifest["serialize"]["written"],
            "stale_aggregate_emitted": bool(iwild_manifest["aggregates"]),
            "aggregate_seed_count": iwild_aggregate.get("n_seeds"),
            "aggregate_seeds": iwild_aggregate.get("seeds"),
            "aggregate_verdict": iwild_aggregate.get("verdict"),
        },
    }


def macro_f1_present_true_labels(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for label in np.unique(y_true):
        true_positive = int(np.sum((y_true == label) & (y_pred == label)))
        false_positive = int(np.sum((y_true != label) & (y_pred == label)))
        false_negative = int(np.sum((y_true == label) & (y_pred != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(scores))


def corrected_iwild_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = load_json(path)
    keys = ("seed", "location", "split", "comp", "regime", "aggr")
    conditions = {tuple(row.get(key) for key in keys): row for row in document["conditions"]}
    corrected = []
    shifts = []
    sign_flips = 0
    for source in document["records"]:
        row = dict(source)
        condition = conditions[tuple(row.get(key) for key in keys)]
        y_true = np.asarray(condition["eval_y"], dtype=int)
        frozen = np.asarray(condition["preds_frozen"], dtype=int)
        adapted = np.asarray(row["preds"], dtype=int)
        a0 = macro_f1_present_true_labels(y_true, frozen)
        aa = macro_f1_present_true_labels(y_true, adapted)
        corrected_b = aa - a0
        stored_b = float(row["B"])
        shifts.append(corrected_b - stored_b)
        sign_flips += sign(corrected_b) != sign(stored_b)
        row["a0"] = a0
        row["aa"] = aa
        row["a_adapted"] = aa
        row["B"] = corrected_b
        corrected.append(row)
    return {
        "records": len(corrected),
        "mean_corrected_minus_stored_benefit": float(np.mean(shifts)),
        "benefit_sign_flips": sign_flips,
    }, corrected


def iwild_metric_audit() -> dict[str, Any]:
    primary_path = RESULTS / "win_hunt_v5_iwildcam/result_0ba633eb.json"
    full_path = RESULTS / "iwildcam_full_test/result_e40faf29.json"
    primary_summary, _ = corrected_iwild_records(primary_path)
    full_summary, corrected = corrected_iwild_records(full_path)
    replay: dict[str, Any]
    try:
        spec = importlib.util.spec_from_file_location("kbound_reconcile_audit", ROOT / "scripts/reconcile_result_panels.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load reconciliation module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        keep = []
        for row in corrected:
            compact = {key: row[key] for key in module.RECORD_KEYS if key in row}
            keep.append(compact)
        original = load_json(full_path)
        payload = {
            "dataset": original.get("dataset"),
            "metric": "macro_f1_present_true_labels",
            "evidence_names": original.get("evidence_names"),
            "records": keep,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
            json.dump(payload, handle)
            handle.flush()
            panel = module._transfer_panel(
                Path(handle.name),
                Path(handle.name),
                candidate="tent_episodic",
                cal_seeds={0},
                test_seeds={1},
            )
        score = panel["exact_rank_transfer_score"]
        expected_numpy = getattr(module, "EXPECTED_NUMPY_VERSION", None)
        expected_sklearn = getattr(module, "EXPECTED_SKLEARN_VERSION", None)
        runtime_sklearn = __import__("sklearn").__version__
        runtime_matches_reconciliation = bool(
            np.__version__ == expected_numpy and runtime_sklearn == expected_sklearn
        )
        replay = {
            "status": (
                "diagnostic_only_pinned_runtime_but_no_sealed_official_metric_rerun"
                if runtime_matches_reconciliation
                else "diagnostic_only_runtime_differs_from_pinned_reconciliation"
            ),
            "runtime_numpy": np.__version__,
            "runtime_sklearn": runtime_sklearn,
            "expected_numpy": expected_numpy,
            "expected_sklearn": expected_sklearn,
            "runtime_matches_reconciliation": runtime_matches_reconciliation,
            "n_calibration": panel["n_calibration"],
            "n_test": panel["n_test"],
            "epsilon": panel["calibration"]["epsilon"],
            "action_counts": {
                "adapt": score["adapt_count"],
                "freeze": score["freeze_count"],
                "abstain": score["abstain_count"],
            },
            "regret": score["regret"],
            "point_beats_both": score["point_beats_both"],
        }
    except Exception as exc:  # pragma: no cover - retained as a reportable access issue
        replay = {"status": "not_run", "reason": repr(exc)}
    return {
        "metric_contract": "official WILDS macro-F1 averages labels present in y_true only",
        "primary_48_record_diagnostic": primary_summary,
        "full_test_864_record_diagnostic": full_summary,
        "corrected_transfer_replay": replay,
    }


def release_checksum_audit() -> dict[str, Any]:
    path = ROOT / "docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt"
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$", line)
        if not match:
            rows.append({"path": line, "status": "unparsed"})
            continue
        expected, raw_path = match.groups()
        target = ROOT / raw_path
        actual = sha256(target) if target.exists() else None
        rows.append(
            {
                "path": raw_path,
                "expected_sha256": expected.lower(),
                "actual_sha256": actual,
                "status": "match" if actual == expected.lower() else ("missing" if actual is None else "mismatch"),
            }
        )
    counts = Counter(row["status"] for row in rows)
    return {"entries": len(rows), "status_counts": dict(counts), "rows": rows}


def natural_policy_rows(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    panels = [
        (
            "CIFAR-10-C Tent (controlled)",
            "accuracy",
            canonical["panels"]["cifar10c"]["panel"]["candidates"]["tent"]["regret"],
            "controlled",
            "controlled_point_estimate_with_retrospective_sensitivity",
            True,
        ),
        (
            "Office-Home primary",
            "accuracy",
            canonical["panels"]["officehome"]["primary"]["exact_rank_transfer_score"]["regret"],
            "natural",
            "opened_target_null_diagnostic",
            True,
        ),
        (
            "Office-Home replication",
            "accuracy",
            canonical["panels"]["officehome"]["test_stream_seed_replication"]["exact_rank_transfer_score"]["regret"],
            "natural",
            "opened_target_stream_seed_diagnostic",
            True,
        ),
        (
            "iWildCam H-v2",
            "macro_f1",
            canonical["panels"]["iwildcam"]["primary"]["exact_rank_transfer_score"]["regret"],
            "natural",
            "withheld_invalid_archived_metric_contract_historical_values",
            False,
        ),
        (
            "ImageNet-R",
            "accuracy",
            canonical["panels"]["imagenet_r"]["panel"]["architecture_panel_aggregate"]["regret"],
            "natural",
            "opened_target_negative_diagnostic",
            True,
        ),
        (
            "PACS",
            "accuracy",
            canonical["panels"]["pacs"]["pooled_domain_seed_mean"]["regret"],
            "natural",
            "aggregate_only_null_diagnostic",
            True,
        ),
        (
            "Camelyon17 OOD",
            "accuracy",
            canonical["panels"]["camelyon17"]["ood"]["replay"]["exact_rank_transfer_score"]["regret"],
            "natural",
            "opened_target_one_sided_diagnostic",
            True,
        ),
        (
            "RxRx1",
            "accuracy",
            canonical["panels"]["rxrx1"]["primary_model_seed0"]["exact_rank_transfer_score"]["regret"],
            "natural",
            "opened_target_one_sided_diagnostic",
            True,
        ),
        (
            "CIFAR-10.1",
            "accuracy",
            canonical["panels"]["cifar101"]["replay"]["exact_rank_transfer_score"]["regret"],
            "natural",
            "opened_target_null_diagnostic",
            True,
        ),
    ]
    rows = []
    for panel, metric, regrets, regime, claim_scope, numeric_release_eligible in panels:
        for key, label in (
            ("kga", "KGA"),
            ("always_adapt", "Always adapt"),
            ("always_freeze", "Always freeze"),
        ):
            rows.append(
                {
                    "panel": panel,
                    "metric": metric,
                    "regime": regime,
                    "policy": label,
                    "regret": float(regrets[key]),
                    "claim_scope": claim_scope,
                    "numeric_release_eligible": numeric_release_eligible,
                }
            )
    return rows


def benefit_sign_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    benefit = [float(row["B"]) for row in records]
    return {
        "helpful": sum(value > 0 for value in benefit),
        "harmful": sum(value < 0 for value in benefit),
        "tied": sum(value == 0 for value in benefit),
    }


def opportunity_rows(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    office_primary = canonical["panels"]["officehome"]["primary"]
    office_rep = canonical["panels"]["officehome"]["test_stream_seed_replication"]
    iwild = canonical["panels"]["iwildcam"]["primary"]
    rows = []
    for name, panel, claim_scope, numeric_release_eligible in (
        ("Office-Home primary", office_primary, "opened_target_null_diagnostic", True),
        (
            "Office-Home replication",
            office_rep,
            "opened_target_stream_seed_diagnostic",
            True,
        ),
        (
            "iWildCam H-v2",
            iwild,
            "withheld_invalid_archived_metric_contract_historical_values",
            False,
        ),
    ):
        signs = benefit_sign_counts(panel["records"])
        score = panel["exact_rank_transfer_score"]
        rows.append(
            {
                "panel": name,
                "n": score["n"],
                **signs,
                "adapt_count": score["adapt_count"],
                "freeze_count": score["freeze_count"],
                "abstain_count": score["abstain_count"],
                "regret_kga": score["regret"]["kga"],
                "regret_best_fixed": min(score["regret"]["always_adapt"], score["regret"]["always_freeze"]),
                "point_beats_both": score["point_beats_both"],
                "claim_scope": claim_scope,
                "numeric_release_eligible": numeric_release_eligible,
            }
        )
    rows.extend(
        [
            {
                "panel": "ImageNet-R",
                "n": 480,
                "helpful": 383,
                "harmful": 92,
                "tied": 5,
                "adapt_count": 165,
                "freeze_count": 29,
                "abstain_count": 286,
                "regret_kga": 0.01496875,
                "regret_best_fixed": 0.006359375,
                "point_beats_both": False,
                "claim_scope": "opened_target_negative_diagnostic",
                "numeric_release_eligible": True,
            },
            {
                "panel": "Camelyon17 OOD",
                "n": 18,
                "helpful": 18,
                "harmful": 0,
                "tied": 0,
                "adapt_count": 18,
                "freeze_count": 0,
                "abstain_count": 0,
                "regret_kga": 0.0,
                "regret_best_fixed": 0.0,
                "point_beats_both": False,
                "claim_scope": "opened_target_one_sided_diagnostic",
                "numeric_release_eligible": True,
            },
            {
                "panel": "RxRx1",
                "n": 60,
                "helpful": 0,
                "harmful": 60,
                "tied": 0,
                "adapt_count": 0,
                "freeze_count": 60,
                "abstain_count": 0,
                "regret_kga": 0.0,
                "regret_best_fixed": 0.0,
                "point_beats_both": False,
                "claim_scope": "opened_target_one_sided_diagnostic",
                "numeric_release_eligible": True,
            },
        ]
    )
    return rows


def initial_findings() -> list[dict[str, Any]]:
    """Return the immutable 2026-08-27 initial-audit findings.

    These statements describe the state observed before remediation.  They are
    intentionally retained even when the current code contains a fix, because a
    code fix does not retroactively validate an affected historical artifact.
    """
    items = [
        (1, "Critical", "Route-B estimator", "Arbitrary-sign b_hat drives routing although b_tilde is anchor-oriented; 105/138 archived ADAPT cells strictly harmed.", "analysis.py:270-296", "Use a bounded, anchor-oriented estimator; test invariance to eigenvector sign; fail closed when orientation is unidentified."),
        (2, "Critical", "Theory/task mismatch", "The binary correctness-agreement identity is applied to 65-, 182-, and 200-class tasks and to macro-F1.", "val_multicandidate_residual.py:13-23", "Restrict Route B to binary accuracy or derive and validate a multiclass confusion-tensor method."),
        (3, "Critical", "Duplicate leakage", "Office-Home five-seed zero-regret aggregate has 180 rows but only 90 unique seed-condition keys; each LOO holdout retains its twin.", "extract_multiseed_natural.py:84-95; per_condition_serialize.py:194-214", "Reject duplicate scientific keys and aggregate one canonical run only."),
        (4, "Critical", "Stale lineage", "An iWildCam aggregate was emitted from stale seed files even though the current source wrote no requested candidate rows.", "extract_multiseed_natural.py:256-279", "Use fresh atomic staging and aggregate only files returned by the current invocation."),
        (5, "Critical", "Resume contamination", "A declared Office-Home target-test result contains 24/42 source/target-validation conditions.", "run_officehome_kbound.py:101-110", "Key partials by canonical scientific-config hash and refuse role/split/checkpoint mismatches."),
        (6, "High", "Metric parity", "iWildCam sklearn macro-F1 includes prediction-only classes, unlike WILDS; benefit changes by +0.0294 on average with 60/864 sign flips.", "run_iwildcam_kbound.py:48-50", "Call the WILDS evaluator or pass labels=np.unique(y_true), zero_division=0; add parity tests."),
        (7, "High", "Target-label leakage", "Broad multiseed extraction can combine validation and test labels in the same LOO fitting pool and labels stream seeds as model seeds.", "run_multiseed.sh:171-181,221-231", "Fit on source/development data, freeze once, and require model_seed plus checkpoint SHA-256 for model-level inference."),
        (8, "High", "Structural infeasibility", "Single-adapter natural runners have M=2 while Route B requires M>=4; focused Office-Home has only three LOO cells at alpha=.10.", "analysis.py:260-263", "Preflight calibration feasibility and require four unique, sufficiently ranked predictors."),
        (9, "High", "Error handling", "Route ERROR and incomplete cells can be scored as frozen-equivalent or published without a completeness failure.", "analysis.py:258-259; run_camelyon17_kbound.py:61-65", "Never score ERROR; enforce expected/completed/failed ledgers before publication."),
        (10, "High", "PACS denominator", "The draft calls 108 twin-pairs condition cells; reported rates use 216 decision evaluations.", "kbound_submission_body.tex:523-525", "Report 216 evaluations and 108 paired settings separately; keep n=12 inference units."),
        (11, "High", "Raw-population drift", "Current iWildCam present-file population is 12,530 versus 14,453 archived selected-location images (-13.31%).", "run_iwildcam_kbound.py present-file filtering", "Archive sample IDs and a population-manifest hash for every draw."),
        (12, "High", "Release integrity", "Six of fourteen outer release checksum entries were stale at the initial pre-remediation audit, including both PDFs and canonical manifests.", "KBOUND_RELEASE_SHA256SUMS.txt", "Regenerate the checksum seal only after all result and PDF artifacts are frozen."),
        (13, "High", "Prospective eligibility", "All eight inventoried natural tracks are opened; zero verified unopened targets remain.", "NATURAL_TARGET_PROVENANCE_AUDIT.json", "Use a genuinely new cohort or hidden-label evaluation for a confirmatory natural claim."),
        (14, "Medium", "JSON interoperability", "Five focused Office-Home files contain 57 literal Infinity values rejected by strict JSON.", "officehome_focused_multicandidate/result_*.json", "Serialize infeasible radii as null plus an explicit feasibility status."),
        (15, "Medium", "Schema completeness", "84.42% of canonical source rows omit explicit metric and all rows overload legacy seed semantics.", "reconciled_panels_v1/source", "Require explicit metric, model_seed, stream_seed, split, sample population, and checkpoint hash."),
    ]
    return [
        {
            "rank": rank,
            "severity": severity,
            "category": category,
            "finding": finding,
            "evidence": evidence,
            "minimum_safe_fix": fix,
            "finding_stage": "initial_audit_pre_remediation",
        }
        for rank, severity, category, finding, evidence, fix in items
    ]


def remediation_rows(
    release: dict[str, Any],
    lineage: dict[str, Any],
) -> list[dict[str, Any]]:
    """Map every initial finding to its current control and release disposition."""

    quarantine_active = lineage.get("status") == "quarantined_invalid_derived_artifacts"

    rows = [
        {
            "rank": 1,
            "remediation_status": "CONTROL_IMPLEMENTED_HISTORICAL_ROUTE_WITHHELD",
            "remediation_action": "Route B now uses a bounded anchor-oriented median-of-minors estimator and fails closed when orientation, range, rank, or candidate-identity checks fail.",
            "verification_evidence": "experiments/kbound/wilds/analysis.py; tests/test_wilds_multicandidate_route_hardening.py",
            "release_disposition": "Historical Route-B decisions are non-promotable; only a fresh hardened run may be evaluated.",
            "remaining_requirement": "Run the sealed task-compatible protocol and verify its complete ledger before any numerical claim.",
        },
        {
            "rank": 2,
            "remediation_status": "CONTROL_IMPLEMENTED_HISTORICAL_ROUTE_WITHHELD",
            "remediation_action": "Route B now requires binary classification, exactly two classes, an accuracy objective, and an explicit trusted-anchor premise; multiclass and macro-F1 calls return unscorable UNSUPPORTED states.",
            "verification_evidence": "experiments/kbound/wilds/analysis.py; tests/test_wilds_multicandidate_route_hardening.py; tests/test_remaining_wilds_runner_integrity.py",
            "release_disposition": "No multiclass Route-B action or score from the historical archive may be released.",
            "remaining_requirement": "Use Route A or a separately derived and validated metric-specific multiclass method.",
        },
        {
            "rank": 3,
            "remediation_status": (
                "QUARANTINED_AND_CONTROL_IMPLEMENTED"
                if quarantine_active
                else "CONTROL_IMPLEMENTED_QUARANTINE_NOT_VERIFIED"
            ),
            "remediation_action": "Duplicate scientific keys now hard-fail before estimation and aggregation; the invalid Office-Home derivatives were removed from the release tree with hashes retained in the quarantine manifest.",
            "verification_evidence": "docs/research/kbound/audits/empirical_data_quality_2026_08_27/quarantine_manifest.json; tests/test_multiseed_natural_extraction_lineage.py",
            "release_disposition": "QUARANTINED_NO_RELEASE_USE",
            "remaining_requirement": "Regenerate only from one canonical run per scientific key under the hardened extractor.",
        },
        {
            "rank": 4,
            "remediation_status": (
                "QUARANTINED_AND_CONTROL_IMPLEMENTED"
                if quarantine_active
                else "CONTROL_IMPLEMENTED_QUARANTINE_NOT_VERIFIED"
            ),
            "remediation_action": "Extraction now writes to a fresh atomic stage and aggregates only files produced by the current invocation; the stale iWildCam derivatives were quarantined.",
            "verification_evidence": "docs/research/kbound/scripts/extract_multiseed_natural.py; docs/research/kbound/audits/empirical_data_quality_2026_08_27/quarantine_manifest.json; tests/test_multiseed_natural_extraction_lineage.py",
            "release_disposition": "QUARANTINED_NO_RELEASE_USE",
            "remaining_requirement": "A fresh official-metric source run is required before regeneration.",
        },
        {
            "rank": 5,
            "remediation_status": "CONTROL_IMPLEMENTED_HISTORICAL_ARTIFACT_WITHHELD",
            "remediation_action": "Natural runners now require runner-specific semantic validation while sealing and loading partial state. The validators recompute scientific cell identities, deterministic sample provenance and labels from live indexes, evidence/TTA protocol contracts, score arithmetic, and current checkpoint/config context. RxRx1 completion receipts are additionally bound to the exact run directory, result, population, checkpoint tensors, implementation hashes, and scientific configuration.",
            "verification_evidence": "experiments/kbound/wilds/run_integrity.py; experiments/kbound/wilds/run_camelyon17_kbound.py; experiments/kbound/wilds/run_imagenetr_kbound.py; experiments/kbound/wilds/run_geoshift_kbound.py; experiments/kbound/wilds/run_iwildcam_kbound.py; experiments/kbound/officehome/run_officehome_kbound.py; experiments/kbound/wilds/run_rxrx1_kbound.py; tests/test_wilds_run_integrity.py; tests/test_camelyon_imagenetr_runner_integrity.py; tests/test_remaining_wilds_runner_integrity.py; tests/test_natural_runner_integrity.py",
            "release_disposition": "The contaminated historical target-test artifact remains audit-only and cannot be promoted.",
            "remaining_requirement": "Rerun target-test in a fresh directory with the hardened role contract.",
        },
        {
            "rank": 6,
            "remediation_status": "CONTROL_IMPLEMENTED_RERUN_REQUIRED",
            "remediation_action": "The iWildCam scorer now matches WILDS label-present macro-F1 semantics, and the current canonical/manuscript release explicitly withholds the numerical and action row.",
            "verification_evidence": "experiments/kbound/wilds/run_iwildcam_kbound.py; tests/test_natural_runner_integrity.py; experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json",
            "release_disposition": "WITHHELD_INVALID_METRIC_CONTRACT_DIAGNOSTIC_ONLY",
            "remaining_requirement": "Run the official metric prospectively in the pinned runtime with a sealed population manifest.",
        },
        {
            "rank": 7,
            "remediation_status": "CONTROL_IMPLEMENTED_DEVELOPMENT_DIAGNOSTIC_ONLY",
            "remediation_action": "The extractor prohibits target-test labels, separates model_seed from stream_seed, requires distinct checkpoint hashes for model-seed inference, and marks within-development LOO aggregates non-confirmatory.",
            "verification_evidence": "docs/research/kbound/scripts/extract_multiseed_natural.py; tests/test_multiseed_natural_extraction_lineage.py; tests/test_independent_checkpoint_audit.py",
            "release_disposition": "DEVELOPMENT_DIAGNOSTIC_ONLY_NO_HELDOUT_PROMOTION",
            "remaining_requirement": "Obtain independent checkpoints and a sealed held-out evaluation for confirmatory inference.",
        },
        {
            "rank": 8,
            "remediation_status": "CONTROL_IMPLEMENTED_HISTORICAL_RESULT_WITHHELD",
            "remediation_action": "Exact-rank infeasibility now serializes as null plus an explicit INFEASIBLE status and ABSTAIN; candidate count, uniqueness, and effective-rank checks fail closed.",
            "verification_evidence": "experiments/kbound/wilds/analysis.py; docs/research/kbound/scripts/per_condition_serialize.py; tests/test_multiseed_natural_extraction_lineage.py; tests/test_camelyon_imagenetr_runner_integrity.py",
            "release_disposition": "Historical structurally infeasible routing results are non-promotable.",
            "remaining_requirement": "Supply a predeclared calibration pool and candidate geometry that satisfy the exact-rank requirements.",
        },
        {
            "rank": 9,
            "remediation_status": "CONTROL_IMPLEMENTED_RERUN_REQUIRED",
            "remediation_action": "ERROR/UNSUPPORTED routes are unscorable, strict ledgers track expected/completed/failed/missing cells, and incomplete runs cannot receive publication_eligible=true.",
            "verification_evidence": "experiments/kbound/wilds/run_integrity.py; tests/test_wilds_run_integrity.py; tests/test_camelyon_imagenetr_runner_integrity.py",
            "release_disposition": "Only complete, failure-free manifests may feed extraction or release tables.",
            "remaining_requirement": "Rerun any historical panel whose original manifest lacks the hardened ledger.",
        },
        {
            "rank": 10,
            "remediation_status": "MANUSCRIPT_CORRECTED",
            "remediation_action": "The manuscript now distinguishes 216 decision evaluations, 108 paired settings, and 12 domain-seed inference units.",
            "verification_evidence": "docs/research/kbound/kbound_submission_body.tex; scripts/reconcile_result_panels.py",
            "release_disposition": "Corrected denominator language is eligible for release.",
            "remaining_requirement": "Preserve the three denominators in every generated table and caption.",
        },
        {
            "rank": 11,
            "remediation_status": "CONTROL_IMPLEMENTED_LEGACY_POPULATION_WITHHELD",
            "remediation_action": "Current runners hash ordered sample identities, labels/locations or paths, split identity, and checkpoint/config state into population and resume manifests. OfficeHome and iWildCam resume validation now independently reconstructs the exact stream/evaluation samples and labels from the current live index instead of trusting stored hashes.",
            "verification_evidence": "experiments/kbound/wilds/run_iwildcam_kbound.py; experiments/kbound/officehome/run_officehome_kbound.py; tests/test_natural_runner_integrity.py",
            "release_disposition": "The legacy iWildCam population mismatch blocks promotion of the archived numerical row.",
            "remaining_requirement": "Use the new population manifest in a fresh official-metric rerun.",
        },
        {
            "rank": 12,
            **outer_seal_remediation(),
        },
        {
            "rank": 13,
            "remediation_status": "UNRESOLVED_REQUIRES_NEW_UNOPENED_TARGET",
            "remediation_action": "No code change can make an already opened target prospective; the release now treats all inventoried natural targets as diagnostic or null/boundary evidence.",
            "verification_evidence": "experiments/kbound/results/natural_target_provenance_v1/NATURAL_TARGET_PROVENANCE_AUDIT.json",
            "release_disposition": "NO_CONFIRMATORY_NATURAL_WIN_CLAIM_ALLOWED",
            "remaining_requirement": "Evaluate one genuinely new cohort or hidden-label target after sealing code, thresholds, checkpoints, samples, and inference rules.",
        },
        {
            "rank": 14,
            "remediation_status": "CONTROL_IMPLEMENTED_HISTORICAL_JSON_WITHHELD",
            "remediation_action": "Current runners use strict atomic JSON, reject decoded non-finite values, and serialize infeasible radii as null plus an explicit status.",
            "verification_evidence": "experiments/kbound/wilds/run_integrity.py; tests/test_wilds_run_integrity.py; tests/test_camelyon_imagenetr_runner_integrity.py",
            "release_disposition": "Historical files containing literal Infinity remain non-interoperable audit evidence only.",
            "remaining_requirement": "Regenerate those results under the strict writer before any machine-consumed release use.",
        },
        {
            "rank": 15,
            "remediation_status": "PARTIALLY_REMEDIATED_LEGACY_SCHEMA_DISCLOSED",
            "remediation_action": "Current natural-run schemas require explicit metric, model/stream seed roles, split and population identity, checkpoint hashes, and claim/completion eligibility; extractors reject unknown schemas for promotion.",
            "verification_evidence": "docs/research/kbound/scripts/extract_multiseed_natural.py; docs/research/kbound/scripts/make_multiseed_natural_forest.py; tests/test_multiseed_natural_extraction_lineage.py; tests/test_multiseed_natural_forest_lineage.py",
            "release_disposition": "Legacy canonical rows retain disclosed schema gaps and may support only their explicitly bounded historical claims.",
            "remaining_requirement": "A new-schema rerun is required for claims needing explicit population, checkpoint, or seed-role provenance.",
        },
    ]
    if [row["rank"] for row in rows] != list(range(1, 16)):
        raise AssertionError("remediation table must cover every finding rank exactly once")
    return rows


def findings_with_remediation(
    release: dict[str, Any],
    lineage: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    initial = initial_findings()
    remediation = remediation_rows(release, lineage)
    remediation_by_rank = {row["rank"]: row for row in remediation}
    if set(remediation_by_rank) != {row["rank"] for row in initial}:
        raise AssertionError("every initial finding requires exactly one remediation disposition")
    combined = [{**row, **remediation_by_rank[row["rank"]]} for row in initial]
    return combined, remediation


def reviewer_scorecard() -> list[dict[str, Any]]:
    return [
        {"stage": "current_evidence", "dimension": "Canonical bundle integrity", "score_out_of_10": 9.5, "score_status": "SCORED", "basis": "106/106 source and compact hashes pass; 12,619 rows and 117 aggregates reconcile. This is internal integrity, not blanket claim validity."},
        {
            "stage": "current_evidence",
            "dimension": "Controlled CIFAR-10-C evidence",
            "score_out_of_10": None,
            "score_status": "NOT_RESCORED_AFTER_INFERENCE_CORRECTION",
            "basis": CONTROLLED_HOLM_BASIS,
        },
        {"stage": "current_evidence", "dimension": "Natural-shift routing evidence", "score_out_of_10": 4.0, "score_status": "UNCHANGED_AFTER_CODE_REMEDIATION", "basis": "No defensible natural beats-both win exists. Hardening code does not convert opened, invalid, duplicated, or one-sided archives into new evidence."},
        {"stage": "initial_audit", "dimension": "Estimator/task validity (pre-remediation)", "score_out_of_10": 4.0, "score_status": "HISTORICAL_BASELINE", "basis": "The initial audit found arbitrary Route-B orientation and multiclass misuse; current controls address these defects only for fresh runs."},
        {"stage": "initial_audit", "dimension": "Provenance and reproducibility (pre-remediation)", "score_out_of_10": 5.5, "score_status": "HISTORICAL_BASELINE", "basis": "The initial audit found resume contamination, stale release hashes, sample drift, and overloaded seed semantics."},
        {"stage": "initial_audit", "dimension": "Initial empirical readiness (pre-remediation)", "score_out_of_10": 5.8, "score_status": "HISTORICAL_BASELINE", "basis": "Senior-reviewer judgment at initial diagnosis; not a computed scientific metric and not the current state of the code controls."},
        {"stage": "post_remediation", "dimension": "Post-remediation overall empirical readiness", "score_out_of_10": None, "score_status": "NOT_RESCORED_PENDING_PROSPECTIVE_EVIDENCE", "basis": "A new overall score is withheld until hardened pipelines produce sealed, complete reruns and an unopened or hidden-label natural evaluation. The natural-evidence component remains 4.0/10."},
    ]


def pacs_denominator_audit(canonical: dict[str, Any]) -> dict[str, Any]:
    pooled = canonical["panels"]["pacs"]["pooled_domain_seed_mean"]
    evaluations = 3 * 4 * 18
    adaptations = round(float(pooled["adapt_rate"]) * evaluations)
    covered = round(float(pooled["decision_coverage"]) * evaluations)
    false_adapt = round(float(pooled["fa_u"]) * evaluations)
    return {
        "inference_units": int(pooled["n_domain_seed_units"]),
        "decision_evaluations": evaluations,
        "paired_settings": evaluations // 2,
        "adaptations": adaptations,
        "freezes": covered - adaptations,
        "abstentions": evaluations - covered,
        "false_adaptations": false_adapt,
        "reported_adapt_rate": pooled["adapt_rate"],
        "reported_coverage": pooled["decision_coverage"],
        "reported_fa_u": pooled["fa_u"],
    }


def refresh_release_wording_only(out_dir: Path) -> None:
    """Correct bounded release wording without restoring quarantined diagnostics.

    No number is recomputed or changed. The invalid focused Office-Home derivatives used by
    the historical audit were deliberately quarantined, so a wording correction must not
    restore or silently re-run them.
    """

    summary_path = out_dir / "audit_summary.json"
    scorecard_path = out_dir / "reviewer_scorecard.csv"
    if not summary_path.is_file() or not scorecard_path.is_file():
        raise FileNotFoundError("wording-only refresh requires the existing audit JSON and CSV")
    summary = load_json_strict(summary_path)
    release_decision = summary.get("release_decision")
    if not isinstance(release_decision, dict):
        raise ValueError("audit summary lacks release_decision")
    prior = release_decision.get("controlled_cifar10c_tent_claim")
    allowed_prior = {
        (
            "SUPPORTED_POINT_ESTIMATE_WITH_RETROSPECTIVE_UNADJUSTED_INTERVAL_SENSITIVITY; "
            "PREREGISTERED_SIX_COMPARISON_HOLM_FAILED"
        ),
        (
            "SUPPORTED_POINT_ESTIMATE_WITH_RETROSPECTIVE_UNADJUSTED_INTERVAL_SENSITIVITY; "
            + CONTROLLED_HOLM_STATUS
        ),
    }
    if prior not in allowed_prior:
        raise ValueError("unexpected controlled-claim status in audit summary")
    release_decision["controlled_cifar10c_tent_claim"] = (
        "SUPPORTED_POINT_ESTIMATE_WITH_RETROSPECTIVE_UNADJUSTED_INTERVAL_SENSITIVITY; "
        + CONTROLLED_HOLM_STATUS
    )
    current_rel = (
        "experiments/kbound/results/reconciled_panels_v1/"
        "current_policy_cluster_inference.json"
    )
    checksum_rows = (summary.get("release_checksums") or {}).get("rows")
    if not isinstance(checksum_rows, list):
        raise ValueError("audit summary lacks release checksum rows")
    current_rows = [row for row in checksum_rows if row.get("path") == current_rel]
    if len(current_rows) != 1:
        raise ValueError("audit summary must contain one current-policy checksum row")
    current_hash = sha256(ROOT / current_rel)
    current_rows[0]["actual_sha256"] = current_hash
    current_rows[0]["status"] = (
        "match" if current_rows[0].get("expected_sha256") == current_hash else "mismatch"
    )
    status_counts = Counter(row.get("status") for row in checksum_rows)
    summary["release_checksums"]["status_counts"] = dict(status_counts)
    summary["release_checksums"]["verification_scope"] = (
        "Historical/pre-release observation only. The authoritative outer checksum is "
        "written and verified after this audit artifact is finalized."
    )
    release_decision["outer_checksum_seal"] = OUTER_SEAL_STATUS
    bottom_line = summary.get("bottom_line") or {}
    prior_key = "controlled_cifar10c_preregistered_cluster_win"
    if prior_key in bottom_line:
        prior_value = bottom_line.pop(prior_key)
        if prior_value is not False:
            raise ValueError("unexpected prior controlled-cluster status")
    bottom_line["controlled_cifar10c_retrospective_six_contrast_holm_win"] = False
    summary["bottom_line"] = bottom_line

    with scorecard_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError("reviewer scorecard has no CSV header")
    matches = [row for row in rows if row.get("dimension") == "Controlled CIFAR-10-C evidence"]
    if len(matches) != 1:
        raise ValueError("reviewer scorecard must contain one controlled CIFAR-10-C row")
    matches[0]["basis"] = CONTROLLED_HOLM_BASIS
    write_csv(scorecard_path, rows, fieldnames)

    outer = outer_seal_remediation()
    for name in ("remediation_status.csv", "findings.csv"):
        path = out_dir / name
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            output_fields = reader.fieldnames
            output_rows = list(reader)
        if not output_fields:
            raise ValueError(f"{name} has no CSV header")
        rank_rows = [row for row in output_rows if row.get("rank") == "12"]
        if len(rank_rows) != 1:
            raise ValueError(f"{name} must contain one rank-12 release-seal row")
        rank_rows[0].update(outer)
        write_csv(path, output_rows, output_fields)

    write_json(summary_path, summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument(
        "--wording-only",
        action="store_true",
        help=(
            "update bounded release wording and metadata in existing audit outputs; do "
            "not recompute quarantined historical diagnostics"
        ),
    )
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.wording_only:
        refresh_release_wording_only(out_dir)
        print(json.dumps({"out_dir": out_dir.relative_to(ROOT).as_posix(), "wording_only": True}))
        return 0

    canonical = load_json_strict(CANONICAL_PATH)
    source_manifest = load_json_strict(SOURCE_MANIFEST_PATH)
    source_summary, dataset_profile = source_bundle_audit(source_manifest)
    score_summary = recursive_score_audit(canonical)
    focused_summary, checkpoint_rows = focused_officehome_audit()
    natural_rows = natural_policy_rows(canonical)
    opportunity = opportunity_rows(canonical)
    scorecard_rows = reviewer_scorecard()

    provenance_path = RESULTS / "natural_target_provenance_v1/NATURAL_TARGET_PROVENANCE_AUDIT.json"
    provenance = load_json_strict(provenance_path)
    release = release_checksum_audit()
    lineage = multiseed_lineage_audit()
    finding_rows, remediation = findings_with_remediation(release, lineage)
    remediation_status_counts = dict(
        sorted(Counter(row["remediation_status"] for row in remediation).items())
    )
    quarantine = load_json_strict(QUARANTINE_MANIFEST_PATH)
    summary = {
        "schema": "kbound_empirical_data_quality_audit_v2",
        "generated_at": GENERATED_AT,
        "repository_path_binding": {
            "schema": "git-repository-relative-posix-v1",
            "root": ".",
            "root_role": "git_repository_root",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "python_executable_basename": Path(sys.executable).name,
        },
        "intended_grain": {
            "candidate_record": "dataset x checkpoint/model_seed x stream_seed x domain/location/backbone x split x composition x regime x aggressiveness x candidate",
            "route_record": "same condition key without candidate",
            "inference_unit_warning": "inference units, condition evaluations, repeated twin settings, and stream seeds are not interchangeable",
        },
        "canonical": {
            "canonical_sha256": sha256(CANONICAL_PATH),
            "source_manifest_sha256": sha256(SOURCE_MANIFEST_PATH),
            "canonical_declared_source_manifest_sha256": canonical["source_manifest_sha256"],
            "source_manifest_hash_matches_canonical": sha256(SOURCE_MANIFEST_PATH) == canonical["source_manifest_sha256"],
            "bundle": source_summary,
            "aggregate_score_checks": score_summary,
        },
        "route_b_archive": route_b_archive_audit(),
        "focused_officehome": focused_summary,
        "resume_contamination": resume_contamination_audit(),
        "multiseed_lineage": lineage,
        "iwild_metric": iwild_metric_audit(),
        "pacs_denominator": pacs_denominator_audit(canonical),
        "natural_target_provenance": {
            "verdict": provenance["verdict"],
            "track_count": len(provenance["tracks"]),
            "verified_unopened_tracks": provenance["verified_unopened_tracks"],
            "prospective_natural_track_available": provenance["prospective_natural_track_available"],
        },
        "release_checksums": release,
        "finding_count": len(finding_rows),
        "critical_findings": sum(row["severity"] == "Critical" for row in finding_rows),
        "high_findings": sum(row["severity"] == "High" for row in finding_rows),
        "audit_stages": {
            "initial_diagnosis": {
                "finding_count": len(finding_rows),
                "critical_findings": sum(row["severity"] == "Critical" for row in finding_rows),
                "high_findings": sum(row["severity"] == "High" for row in finding_rows),
                "empirical_readiness_score_out_of_10": 5.8,
                "score_kind": "senior-reviewer judgment, not a scientific estimand",
            },
            "post_remediation": {
                "remediation_rows": len(remediation),
                "status_counts": remediation_status_counts,
                "invalid_derived_artifacts_quarantined": len(quarantine.get("artifacts", [])),
                "raw_sources_deleted": bool(quarantine.get("raw_sources_deleted", True)),
                "natural_shift_routing_evidence_score_out_of_10": 4.0,
                "overall_empirical_readiness_score_out_of_10": None,
                "overall_score_status": "WITHHELD_PENDING_SEALED_PROSPECTIVE_EVIDENCE",
                "historical_results_revalidated_by_code_fixes": False,
            },
        },
        "release_decision": {
            "controlled_cifar10c_tent_claim": (
                "SUPPORTED_POINT_ESTIMATE_WITH_RETROSPECTIVE_UNADJUSTED_INTERVAL_SENSITIVITY; "
                + CONTROLLED_HOLM_STATUS
            ),
            "iwildcam_numerical_and_action_claim": "WITHHELD_PENDING_PINNED_OFFICIAL_METRIC_RERUN",
            "historical_invalid_natural_derivatives": "QUARANTINED_OR_NONPROMOTABLE",
            "natural_beats_both_claim": "NOT_SUPPORTED",
            "outer_checksum_seal": next(
                row["remediation_status"] for row in remediation if row["rank"] == 12
            ),
        },
        "bottom_line": {
            "defensible_natural_beats_both_win": False,
            "controlled_cifar10c_tent_remains_strong": True,
            "controlled_cifar10c_retrospective_six_contrast_holm_win": False,
            "can_guarantee_9_5_empirical_score": False,
            "code_hardening_retroactively_repairs_historical_results": False,
            "recommended_claim": "Strong controlled routing evidence; hardened natural-shift code is ready for sealed reruns, while existing natural evidence remains a transparent boundary/null result rather than a natural win.",
        },
    }

    write_json(out_dir / "audit_summary.json", summary)
    write_csv(
        out_dir / "findings.csv",
        finding_rows,
        [
            "rank",
            "severity",
            "category",
            "finding",
            "evidence",
            "minimum_safe_fix",
            "finding_stage",
            "remediation_status",
            "remediation_action",
            "verification_evidence",
            "release_disposition",
            "remaining_requirement",
        ],
    )
    write_csv(
        out_dir / "remediation_status.csv",
        remediation,
        [
            "rank",
            "remediation_status",
            "remediation_action",
            "verification_evidence",
            "release_disposition",
            "remaining_requirement",
        ],
    )
    write_csv(
        out_dir / "canonical_dataset_profile.csv",
        dataset_profile,
        [
            "dataset",
            "source_files",
            "record_files",
            "records",
            "missing_metric_records",
            "missing_metadata_dataset_records",
            "strict_json_failures",
        ],
    )
    write_csv(
        out_dir / "natural_policy_regret.csv",
        natural_rows,
        [
            "panel",
            "metric",
            "regime",
            "policy",
            "regret",
            "claim_scope",
            "numeric_release_eligible",
        ],
    )
    write_csv(
        out_dir / "natural_opportunity.csv",
        opportunity,
        [
            "panel",
            "n",
            "helpful",
            "harmful",
            "tied",
            "adapt_count",
            "freeze_count",
            "abstain_count",
            "regret_kga",
            "regret_best_fixed",
            "point_beats_both",
            "claim_scope",
            "numeric_release_eligible",
        ],
    )
    write_csv(
        out_dir / "focused_officehome_checkpoints.csv",
        checkpoint_rows,
        [
            "model_seed",
            "checkpoint_sha256",
            "conditions",
            "mean_freeze_accuracy",
            "mean_sar_accuracy",
            "mean_oracle_accuracy",
            "sar_minus_freeze",
            "oracle_minus_sar",
            "eata_sar_exact_prediction_cells",
        ],
    )
    write_csv(
        out_dir / "reviewer_scorecard.csv",
        scorecard_rows,
        ["stage", "dimension", "score_out_of_10", "score_status", "basis"],
    )
    try:
        portable_out_dir = out_dir.relative_to(ROOT).as_posix()
    except ValueError:
        portable_out_dir = out_dir.name
    print(json.dumps({"out_dir": portable_out_dir, "summary": summary["bottom_line"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
