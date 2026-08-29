#!/usr/bin/env python3
"""Validate and replay PACS/VLCS per-cell decision artifacts.

Historical ``kbound_pacs_seed_v1`` summaries do not contain the information
required here. They remain aggregate-only evidence and must not be upgraded by
inference. A future rerun from ``pacs_vlcs_runner.py`` emits v2 records that this
validator can reproduce without model inference or target labels at decision
time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REQUIRED_RECORD_FIELDS = {
    "dataset",
    "domain",
    "calibration_domain",
    "seed",
    "split",
    "condition",
    "candidate",
    "metric",
    "Z",
    "Z_names",
    "evidence_schema_version",
    "a0",
    "aa",
    "loss_frozen",
    "loss_adapted",
    "B",
    "b_hat",
    "eps_conformal",
    "kga_decision",
    "oracle_action",
    "source_checkpoint_sha256",
    "run_config_sha256",
    "residual_pool_sha256",
    "record_id",
}


class PACSReplayError(ValueError):
    """Raised when a PACS replay artifact is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_action(b_hat: float, epsilon: float | None) -> str:
    if not math.isfinite(b_hat):
        raise PACSReplayError("b_hat must be finite")
    if epsilon is None:
        return "ABSTAIN"
    if not math.isfinite(epsilon) or epsilon < 0:
        raise PACSReplayError("epsilon must be null or finite and nonnegative")
    if b_hat - epsilon > 0:
        return "ADAPT"
    if b_hat + epsilon < 0:
        return "FREEZE"
    return "ABSTAIN"


def replay_records(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema") != "kbound_pacs_percell_v2":
        raise PACSReplayError("expected kbound_pacs_percell_v2")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise PACSReplayError("records must be a nonempty list")

    actions: list[str] = []
    benefits: list[float] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        missing = REQUIRED_RECORD_FIELDS - set(record)
        if missing:
            raise PACSReplayError(f"record {index} missing {sorted(missing)}")
        if record["dataset"] != document.get("dataset"):
            raise PACSReplayError(f"record {index} dataset mismatch")
        if record["domain"] != document.get("domain"):
            raise PACSReplayError(f"record {index} domain mismatch")
        if int(record["seed"]) != int(document.get("seed")):
            raise PACSReplayError(f"record {index} seed mismatch")
        if record["split"] != "test":
            raise PACSReplayError(f"record {index} is not a test record")
        if not isinstance(record["Z"], list) or not record["Z"]:
            raise PACSReplayError(f"record {index} has an empty evidence vector")
        if len(record["Z"]) != len(record["Z_names"]):
            raise PACSReplayError(f"record {index} evidence-name mismatch")
        if not all(math.isfinite(float(value)) for value in record["Z"]):
            raise PACSReplayError(f"record {index} has nonfinite evidence")

        key = (str(record["candidate"]), str(record["condition"]))
        if key in seen:
            raise PACSReplayError(f"duplicate candidate/condition {key}")
        seen.add(key)

        a0 = float(record["a0"])
        aa = float(record["aa"])
        benefit = float(record["B"])
        if not math.isclose(benefit, aa - a0, rel_tol=0.0, abs_tol=1e-12):
            raise PACSReplayError(f"record {index} has B != aa-a0")
        if not math.isclose(float(record["loss_frozen"]), 1.0 - a0, abs_tol=1e-12):
            raise PACSReplayError(f"record {index} frozen loss mismatch")
        if not math.isclose(float(record["loss_adapted"]), 1.0 - aa, abs_tol=1e-12):
            raise PACSReplayError(f"record {index} adapted loss mismatch")
        for hash_field in (
            "source_checkpoint_sha256",
            "run_config_sha256",
            "residual_pool_sha256",
            "record_id",
        ):
            value = str(record[hash_field])
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise PACSReplayError(f"record {index} invalid {hash_field}")
        epsilon_value = record["eps_conformal"]
        expected_action = canonical_action(
            float(record["b_hat"]),
            None if epsilon_value is None else float(epsilon_value),
        )
        if record["kga_decision"] != expected_action:
            raise PACSReplayError(
                f"record {index} action mismatch: {record['kga_decision']} != {expected_action}"
            )
        expected_oracle = "ADAPT" if benefit > 0 else "FREEZE"
        if record["oracle_action"] != expected_oracle:
            raise PACSReplayError(f"record {index} oracle action mismatch")
        actions.append(expected_action)
        benefits.append(benefit)

    n = len(records)
    regret_kga = sum(
        max(benefit, 0.0) - (benefit if action == "ADAPT" else 0.0)
        for benefit, action in zip(benefits, actions, strict=True)
    ) / n
    false_adapt = sum(
        action == "ADAPT" and benefit <= 0
        for benefit, action in zip(benefits, actions, strict=True)
    )
    adapt_count = actions.count("ADAPT")
    return {
        "n": n,
        "regret": {
            "K_Bound": regret_kga,
            "always_adapt": sum(max(-benefit, 0.0) for benefit in benefits) / n,
            "always_freeze": sum(max(benefit, 0.0) for benefit in benefits) / n,
        },
        "FA_u": false_adapt / n,
        "FA_c": false_adapt / adapt_count if adapt_count else None,
        "adapt_rate": adapt_count / n,
        "coverage": sum(action != "ABSTAIN" for action in actions) / n,
        "actions": {name: actions.count(name) for name in ("ADAPT", "FREEZE", "ABSTAIN")},
    }


def validate_seed_summary(path: Path) -> dict[str, dict[str, Any]]:
    summary = json.loads(path.read_text())
    if summary.get("schema") != "kbound_pacs_seed_v1":
        raise PACSReplayError(f"{path}: expected kbound_pacs_seed_v1")
    domains = summary.get("per_domain")
    if not isinstance(domains, dict) or not domains:
        raise PACSReplayError(f"{path}: missing per_domain summaries")

    replayed: dict[str, dict[str, Any]] = {}
    for domain, row in domains.items():
        relative = row.get("per_cell_artifact")
        expected_hash = row.get("per_cell_sha256")
        if not relative or not expected_hash:
            raise PACSReplayError(
                f"{path}: {domain} lacks a v2 per-cell path/hash; historical aggregate is not replayable"
            )
        artifact = (path.parent / relative).resolve()
        if not artifact.is_file():
            raise PACSReplayError(f"{path}: missing {artifact}")
        if _sha256(artifact) != expected_hash:
            raise PACSReplayError(f"{path}: hash mismatch for {artifact}")
        result = replay_records(json.loads(artifact.read_text()))
        if result["n"] != int(row.get("n_test_cells", -1)):
            raise PACSReplayError(f"{path}: {domain} cell count mismatch")
        for policy, value in result["regret"].items():
            if not math.isclose(value, float(row["regret"][policy]), abs_tol=1e-12):
                raise PACSReplayError(f"{path}: {domain}/{policy} regret mismatch")
        for metric in ("FA_u", "adapt_rate", "coverage"):
            if not math.isclose(result[metric], float(row[metric]), abs_tol=1e-12):
                raise PACSReplayError(f"{path}: {domain}/{metric} mismatch")
        replayed[domain] = result
    return replayed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--audit-out", type=Path)
    args = parser.parse_args()
    replayed = validate_seed_summary(args.summary)
    audit = {
        "schema": "kbound_pacs_replay_audit_v1",
        "summary": str(args.summary),
        "status": "SMOKE_PASS",
        "scope": "per-seed partial replay; not a full multi-domain, multi-seed panel",
        "domains": sorted(replayed),
        "complete_cells": sum(row["n"] for row in replayed.values()),
        "duplicate_ids": 0,
        "missing_fields": 0,
        "aggregate_match": True,
        "excluded_cells": [],
    }
    if args.audit_out:
        args.audit_out.parent.mkdir(parents=True, exist_ok=True)
        args.audit_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        f"PACS replay: SMOKE_PASS ({len(replayed)} domains, "
        f"{sum(row['n'] for row in replayed.values())} decision records)"
    )


if __name__ == "__main__":
    main()
