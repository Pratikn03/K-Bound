"""kbound_repro.schema -- versioned result schemas, validators, and migration.

Provides strict, versioned validators for every artifact in the K-Bound
evidence chain, plus cross-field checks that JSON-Schema cannot express and a
migration adapter for historical artifacts.

Artifacts covered (each carries its own ``schema_version``):

    per_condition            one evaluated cell (raw decisions / integer counts)
    per_seed_summary         one seed's rolled-up decision metrics
    multiseed_aggregate      cross-seed aggregate (identical condition order)
    claim_ledger             the wording/status authority
    result_manifest          the single numerical source for promoted results
    empirical_decision_metrics  FA_u/FA_c/regret block (empirical, not theoretical)

Hard rules encoded here (Phase 6):

* Every per-condition record must retain **integer** action counts OR the raw
  per-condition decision list.
* Publication counts are NEVER reconstructed from rounded rates.  When only
  historical *rates* survive, the record must carry ``count: null`` and
  ``status: "not_retained"`` -- never ``round(rate * n)``.
* Migrations preserve the original artifact and write a new normalized one; they
  never silently rewrite historical raw results.
* Before a paired multiseed analysis, seeds must be unique and the condition
  order identical across seeds.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import deps

__all__ = [
    "SchemaError",
    "SCHEMAS",
    "SCHEMA_VERSIONS",
    "validate",
    "check_counts_not_from_rates",
    "check_unique_claim_ids",
    "check_seed_uniqueness",
    "check_identical_condition_order",
    "migrate_historical_per_condition",
    "dump_schemas",
]


class SchemaError(ValueError):
    """Raised when an artifact fails schema or cross-field validation."""


_ACTION_COUNTS = {
    "type": "object",
    "properties": {
        "adapt": {"type": ["integer", "null"], "minimum": 0},
        "freeze": {"type": ["integer", "null"], "minimum": 0},
        "abstain": {"type": ["integer", "null"], "minimum": 0},
        "status": {"enum": ["retained", "not_retained"]},
    },
    "required": ["status"],
    "additionalProperties": True,
}

# Provenance fields required on any produced (non-historical) artifact.
_PROVENANCE_REQUIRED = [
    "schema_version",
    "dataset",
    "protocol",
    "condition_id",
    "model_id",
    "config_hash",
    "quantile_rule",
    "source_artifact",
    "resolved_device",
    "created_at",
]

# The wording ledger records every claim state, including evidence that is
# deliberately withheld.  The numerical manifest is narrower: it may contain
# supported results and clearly labelled descriptive/diagnostic results, but
# never pending, withdrawn, or withheld claims.
_CLAIM_LEDGER_STATUSES = [
    "supported",
    "no-harm",
    "descriptive",
    "diagnostic",
    "withdrawn",
    "withheld",
    "pending",
]
_RESULT_MANIFEST_STATUSES = [
    "supported",
    "no-harm",
    "descriptive",
    "diagnostic",
]

# ---------------------------------------------------------------------------
# Schema documents (draft-07). Authored in-module = one versioned source of truth.
# ---------------------------------------------------------------------------
SCHEMAS: dict[str, dict[str, Any]] = {
    "per_condition": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "kbound per-condition result",
        "type": "object",
        "properties": {
            "schema_version": {"const": "kbound-per-condition-v1"},
            "dataset": {"type": "string"},
            "protocol": {"type": "string"},
            "seed": {"type": "integer"},
            "condition_id": {"type": "string"},
            "model_id": {"type": "string"},
            "config_hash": {"type": "string"},
            "quantile_rule": {"type": "string"},
            "source_artifact": {"type": "string"},
            "resolved_device": {"type": "string"},
            "created_at": {"type": "string"},
            "code_commit": {"type": ["string", "null"]},
            "false_adapt_boundary": {"type": "string"},
            "decisions": {
                "type": "array",
                "items": {"enum": ["adapt", "freeze", "abstain"]},
            },
            "counts": _ACTION_COUNTS,
            "delta": {"type": ["array", "number", "null"]},
        },
        "required": _PROVENANCE_REQUIRED + ["seed"],
        # 'decisions' OR a counts block must be present (raw evidence retained).
        "anyOf": [{"required": ["decisions"]}, {"required": ["counts"]}],
        "additionalProperties": True,
    },
    "per_seed_summary": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "kbound per-seed summary",
        "type": "object",
        "properties": {
            "schema_version": {"const": "kbound-per-seed-v1"},
            "dataset": {"type": "string"},
            "protocol": {"type": "string"},
            "seed": {"type": "integer"},
            "condition_id": {"type": "string"},
            "model_id": {"type": "string"},
            "config_hash": {"type": "string"},
            "quantile_rule": {"type": "string"},
            "source_artifact": {"type": "string"},
            "resolved_device": {"type": "string"},
            "created_at": {"type": "string"},
            "code_commit": {"type": ["string", "null"]},
            "n_conditions": {"type": "integer", "minimum": 1},
            "counts": _ACTION_COUNTS,
            "metrics": {"type": "object"},
        },
        "required": _PROVENANCE_REQUIRED + ["seed", "n_conditions", "counts"],
        "additionalProperties": True,
    },
    "multiseed_aggregate": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "kbound multiseed aggregate",
        "type": "object",
        "properties": {
            "schema_version": {"const": "kbound-multiseed-v1"},
            "dataset": {"type": "string"},
            "protocol": {"type": "string"},
            "seeds": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
            "condition_order": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "model_id": {"type": "string"},
            "config_hash": {"type": "string"},
            "quantile_rule": {"type": "string"},
            "source_artifact": {"type": "string"},
            "resolved_device": {"type": "string"},
            "created_at": {"type": "string"},
            "code_commit": {"type": ["string", "null"]},
            "condition_id": {"type": "string"},
            "aggregate_metrics": {"type": "object"},
        },
        "required": _PROVENANCE_REQUIRED + ["seeds", "condition_order"],
        "additionalProperties": True,
    },
    "claim_ledger": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "kbound claim ledger",
        "type": "object",
        "properties": {
            "schema_version": {"type": "string"},
            "generated_at": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string", "pattern": "^KB-CLAIM-[0-9]+$"},
                        "claim_text": {"type": "string"},
                        "claim_type": {"enum": ["theorem", "empirical", "protocol", "limitation"]},
                        "status": {"enum": _CLAIM_LEDGER_STATUSES},
                        "allowed_wording": {"type": ["string", "array"]},
                        "forbidden_wording": {"type": ["string", "array"]},
                    },
                    "required": ["claim_id", "claim_text", "claim_type", "status"],
                    "additionalProperties": True,
                },
            },
        },
        "required": ["schema_version", "claims"],
        "additionalProperties": True,
    },
    "result_manifest": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "kbound canonical result manifest",
        "type": "object",
        "properties": {
            "schema_version": {"const": "kbound-result-manifest-v1"},
            "created_at": {"type": "string"},
            "code_commit": {"type": ["string", "null"]},
            "runtime": {"type": "object"},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string", "pattern": "^KB-CLAIM-[0-9]+$"},
                        "dataset": {"type": "string"},
                        "protocol": {"type": "string"},
                        "status": {"enum": _RESULT_MANIFEST_STATUSES},
                        "source_artifact": {"type": "string"},
                        "config_hash": {"type": ["string", "null"]},
                        "quantile_rule": {"type": ["string", "null"]},
                        "metrics": {"type": "object"},
                    },
                    "required": ["claim_id", "dataset", "protocol", "status", "source_artifact"],
                    "additionalProperties": True,
                },
            },
        },
        "required": ["schema_version", "results"],
        "additionalProperties": True,
    },
    "empirical_decision_metrics": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "kbound empirical decision metrics",
        "type": "object",
        "properties": {
            "schema_version": {"const": "kbound-empirical-metrics-v1"},
            "false_adapt_boundary": {"const": "delta_le_0"},
            "fa_u": {"type": "number", "minimum": 0, "maximum": 1},
            "fa_c": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "regret_kga": {"type": "number"},
            "regret_always_adapt": {"type": "number"},
            "regret_always_freeze": {"type": "number"},
            "counts": _ACTION_COUNTS,
            "coverage_kind": {"enum": ["empirical", "theoretical"]},
        },
        "required": ["schema_version", "false_adapt_boundary", "fa_u", "counts"],
        "additionalProperties": True,
    },
}

SCHEMA_VERSIONS = {
    "per_condition": "kbound-per-condition-v1",
    "per_seed_summary": "kbound-per-seed-v1",
    "multiseed_aggregate": "kbound-multiseed-v1",
    "result_manifest": "kbound-result-manifest-v1",
    "empirical_decision_metrics": "kbound-empirical-metrics-v1",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _matches_type(value: Any, expected: str) -> bool:
    """Return JSON-type compatibility without treating bool as an integer."""
    return {
        "null": value is None,
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
    }.get(expected, False)


def _validate_builtin(value: Any, spec: dict[str, Any], path: str = "$") -> None:
    """Validate the deliberately small JSON-Schema subset used above.

    ``jsonschema`` remains the preferred engine when installed.  This strict
    standard-library fallback makes provenance validation portable in clean
    checkouts; it is intentionally limited to keywords present in ``SCHEMAS``.
    """
    if "const" in spec and value != spec["const"]:
        raise SchemaError(f"{path}: expected constant {spec['const']!r}")
    if "enum" in spec and value not in spec["enum"]:
        raise SchemaError(f"{path}: {value!r} is not one of {spec['enum']!r}")

    expected = spec.get("type")
    if expected is not None:
        choices = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(value, choice) for choice in choices):
            raise SchemaError(f"{path}: expected JSON type {expected!r}")

    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if "minimum" in spec and is_number and value < spec["minimum"]:
        raise SchemaError(f"{path}: {value!r} is below minimum {spec['minimum']!r}")
    if "maximum" in spec and is_number and value > spec["maximum"]:
        raise SchemaError(f"{path}: {value!r} exceeds maximum {spec['maximum']!r}")
    if "pattern" in spec and not re.search(spec["pattern"], value):
        raise SchemaError(f"{path}: {value!r} does not match {spec['pattern']!r}")

    if isinstance(value, dict):
        missing = [key for key in spec.get("required", []) if key not in value]
        if missing:
            raise SchemaError(f"{path}: missing required properties {missing!r}")
        for key, child in spec.get("properties", {}).items():
            if key in value:
                _validate_builtin(value[key], child, f"{path}.{key}")

    if isinstance(value, list):
        if len(value) < spec.get("minItems", 0):
            raise SchemaError(f"{path}: array has fewer than {spec['minItems']} items")
        child = spec.get("items")
        if child:
            for index, item in enumerate(value):
                _validate_builtin(item, child, f"{path}[{index}]")

    alternatives = spec.get("anyOf")
    if alternatives:
        failures = []
        for alternative in alternatives:
            try:
                _validate_builtin(value, alternative, path)
                break
            except SchemaError as exc:
                failures.append(str(exc))
        else:
            raise SchemaError(f"{path}: no anyOf alternative matched ({'; '.join(failures)})")


def validate(record: dict, schema_name: str) -> None:
    """Validate ``record`` against a named schema; raise ``SchemaError`` on failure."""
    if schema_name not in SCHEMAS:
        raise SchemaError(f"unknown schema {schema_name!r}; known: {sorted(SCHEMAS)}")
    jsonschema = deps.optional("jsonschema")
    if jsonschema is None:
        _validate_builtin(record, SCHEMAS[schema_name])
    else:
        try:
            jsonschema.validate(record, SCHEMAS[schema_name])
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
            raise SchemaError(
                f"{schema_name} validation failed at {location}: {exc.message}"
            ) from exc
    # cross-field rule shared by per-condition / per-seed / empirical metrics
    if "counts" in record:
        check_counts_not_from_rates(record)
    if schema_name == "claim_ledger":
        check_unique_claim_ids(record.get("claims", []), artifact="claim ledger")
    elif schema_name == "result_manifest":
        check_unique_claim_ids(record.get("results", []), artifact="result manifest")


def check_unique_claim_ids(rows: Iterable[dict], *, artifact: str) -> None:
    """Reject repeated claim IDs before any last-value-wins indexing can occur."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        claim_id = row.get("claim_id")
        if not isinstance(claim_id, str):
            continue  # the structural schema reports missing or mistyped IDs
        if claim_id in seen:
            duplicates.add(claim_id)
        seen.add(claim_id)
    if duplicates:
        raise SchemaError(f"{artifact} contains duplicate claim IDs: {sorted(duplicates)}")


def check_counts_not_from_rates(record: dict) -> None:
    """Enforce: counts are raw integers, or explicitly ``not_retained`` + null.

    Rejects the exact defect this project cares about -- a count silently
    reconstructed as ``round(rate * n)``.
    """
    counts = record.get("counts")
    if not isinstance(counts, dict):
        return
    status = counts.get("status")
    numeric = {k: counts.get(k) for k in ("adapt", "freeze", "abstain")}
    if status == "not_retained":
        bad = {k: v for k, v in numeric.items() if v is not None}
        if bad:
            raise SchemaError(
                f"counts.status == 'not_retained' but non-null counts present {bad}; "
                "historical rate-only records must use count: null (never round(rate*n))."
            )
    elif status == "retained":
        missing = [k for k, v in numeric.items() if not isinstance(v, int)]
        if missing:
            raise SchemaError(
                f"counts.status == 'retained' but non-integer/missing counts for {missing}."
            )


def check_seed_uniqueness(seeds: Iterable[int]) -> None:
    """Raise if any seed repeats (guards paired multiseed analysis)."""
    seeds = list(seeds)
    dupes = {s for s in seeds if seeds.count(s) > 1}
    if dupes:
        raise SchemaError(f"duplicate seeds in multiseed analysis: {sorted(dupes)}")


def check_identical_condition_order(per_seed_condition_lists: list[list[str]]) -> None:
    """Raise unless every seed presents conditions in the identical order."""
    if not per_seed_condition_lists:
        raise SchemaError("no per-seed condition lists provided")
    ref = per_seed_condition_lists[0]
    for i, lst in enumerate(per_seed_condition_lists[1:], start=1):
        if lst != ref:
            raise SchemaError(
                f"condition order for seed index {i} differs from seed index 0; "
                "paired multiseed analysis requires identical condition order."
            )


# ---------------------------------------------------------------------------
# Historical migration (preserve original, write normalized copy)
# ---------------------------------------------------------------------------
def migrate_historical_per_condition(
    src: str | Path,
    dst: str | Path,
    *,
    dataset: str,
    protocol: str,
    seed: int,
    condition_id: str,
    model_id: str = "unknown",
    config_hash: str = "unknown",
    quantile_rule: str = "unknown",
    resolved_device: str = "unknown",
    created_at: str,
    code_commit: str | None = None,
) -> dict:
    """Normalize a historical per-condition artifact to the v1 schema.

    * The **original file is never modified**; a new normalized artifact is
      written to ``dst``.
    * If the historical artifact retained raw decisions or integer counts, they
      are carried over verbatim.  If it only had *rates*, the migrated record
      gets ``counts: {..: null, status: not_retained}`` -- counts are NOT
      reconstructed from the rounded rates.
    """
    src, dst = Path(src), Path(dst)
    original = json.loads(src.read_text())

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSIONS["per_condition"],
        "dataset": dataset,
        "protocol": protocol,
        "seed": seed,
        "condition_id": condition_id,
        "model_id": model_id,
        "config_hash": config_hash,
        "quantile_rule": quantile_rule,
        "source_artifact": str(src),
        "resolved_device": resolved_device,
        "created_at": created_at,
        "code_commit": code_commit,
        "migrated_from_schema": "historical-unversioned",
    }

    decisions = original.get("decisions")
    counts = original.get("counts")
    if isinstance(decisions, list) and decisions:
        record["decisions"] = decisions
        c = {"adapt": 0, "freeze": 0, "abstain": 0}
        for d in decisions:
            if d in c:
                c[d] += 1
        record["counts"] = {**c, "status": "retained"}
    elif isinstance(counts, dict) and all(
        isinstance(counts.get(k), int) for k in ("adapt", "freeze", "abstain")
    ):
        record["counts"] = {
            "adapt": counts["adapt"],
            "freeze": counts["freeze"],
            "abstain": counts["abstain"],
            "status": "retained",
        }
    else:
        # Only rates survived -> DO NOT reconstruct counts from rounded rates.
        record["counts"] = {"adapt": None, "freeze": None, "abstain": None, "status": "not_retained"}
        for rk in ("adapt_rate", "freeze_rate", "abstain_rate"):
            if rk in original:
                record[rk] = original[rk]

    validate(record, "per_condition")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def dump_schemas(out_dir: str | Path) -> list[Path]:
    """Write each schema to ``out_dir/<name>.schema.json`` (for external tools)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, schema in SCHEMAS.items():
        p = out_dir / f"{name}.schema.json"
        p.write_text(json.dumps(schema, indent=2, sort_keys=True))
        written.append(p)
    return written
