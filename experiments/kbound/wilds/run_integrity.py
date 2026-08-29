"""Fail-closed state and completion helpers for KBOUND natural-shift runners.

The helpers are deliberately independent of torch.  A partial run is reusable only
when its full scientific configuration hash matches, every stored row belongs to a
completed cell, and the expected cell grid is unchanged.  Final JSON writes are
atomic and reject NaN/Infinity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


PARTIAL_SCHEMA = "kbound_partial_run_state_v3"

SemanticValidator = Callable[
    [list[dict[str, Any]], list[dict[str, Any]]],
    None,
]


class RunIntegrityError(ValueError):
    """Raised when run state cannot be trusted or safely resumed."""


def _reject_constant(token: str) -> None:
    raise RunIntegrityError(f"non-standard JSON constant {token}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return strict, deterministic JSON bytes suitable for scientific hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_cell_id(**fields: Any) -> str:
    """Hash a complete, named scientific cell identity."""

    if not fields or any(value is None for value in fields.values()):
        raise RunIntegrityError("cell identity fields must be present and non-null")
    return stable_sha256(fields)


def validate_scientific_cell_identity(
    cell_id: Any,
    identity: Any,
    *,
    context: str = "cell",
) -> dict[str, Any]:
    """Require a complete identity object whose canonical hash is ``cell_id``.

    A stored digest is not evidence of identity unless the identity payload is
    archived and the digest is recomputed on every resume.  Returning a plain
    ``dict`` also normalizes arbitrary Mapping implementations before hashing.
    """

    if not isinstance(cell_id, str) or len(cell_id) != 64:
        raise RunIntegrityError(f"{context} cell_id must be a SHA-256 hex digest")
    try:
        int(cell_id, 16)
    except ValueError as exc:
        raise RunIntegrityError(f"{context} cell_id must be a SHA-256 hex digest") from exc
    if not isinstance(identity, Mapping) or not identity:
        raise RunIntegrityError(f"{context} is missing scientific_cell_identity")
    normalized = dict(identity)
    if any(value is None for value in normalized.values()):
        raise RunIntegrityError(f"{context} scientific_cell_identity contains null fields")
    recomputed = make_cell_id(**normalized)
    if recomputed != cell_id:
        raise RunIntegrityError(
            f"{context} cell_id does not match its scientific_cell_identity"
        )
    return normalized


def validate_evidence_record(
    record: Mapping[str, Any],
    evidence_names: Iterable[str],
    *,
    expected_tta_protocol: Mapping[str, Any] | None = None,
    context: str = "candidate record",
) -> None:
    """Validate the archived label-free evidence and update magnitude.

    This does not pretend that a JSON file can recreate model activations.  It
    enforces every deterministic invariant available at resume time: exact
    feature dimension, finite numeric values, non-negative update norm, equality
    between the named ``update_norm`` evidence coordinate and ``upd_norm``, and
    the exact data-use protocol derived from the candidate mode.
    """

    names = list(evidence_names)
    if not names or len(names) != len(set(names)):
        raise RunIntegrityError("evidence_names must be non-empty and unique")
    evidence = record.get("Z")
    if not isinstance(evidence, list) or len(evidence) != len(names):
        raise RunIntegrityError(
            f"{context} Z must contain exactly {len(names)} named evidence values"
        )
    parsed = [
        _as_finite_float(value, field=f"Z[{index}]", cell_id=context)
        for index, value in enumerate(evidence)
    ]
    update_norm = _as_finite_float(
        record.get("upd_norm"), field="upd_norm", cell_id=context
    )
    if update_norm < 0.0:
        raise RunIntegrityError(f"{context} upd_norm must be non-negative")
    if "update_norm" in names:
        index = names.index("update_norm")
        if not math.isclose(parsed[index], update_norm, rel_tol=0.0, abs_tol=1e-12):
            raise RunIntegrityError(
                f"{context} Z[update_norm] does not match upd_norm"
            )
    if expected_tta_protocol is not None:
        if record.get("tta_protocol") != dict(expected_tta_protocol):
            raise RunIntegrityError(
                f"{context} tta_protocol does not match the recomputed candidate protocol"
            )


def deterministic_seed(cell_id: str) -> int:
    """Map a cell identity to a stable NumPy/PyTorch-compatible 32-bit seed."""

    if not isinstance(cell_id, str) or len(cell_id) != 64:
        raise RunIntegrityError("cell_id must be a SHA-256 hex digest")
    return int(cell_id[:8], 16)


def atomic_json_dump(payload: Any, path: str | os.PathLike[str], *, indent: int = 2) -> None:
    """Atomically write strict JSON in the destination directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=indent, sort_keys=True, allow_nan=False)
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def strict_json_load(path: str | os.PathLike[str]) -> Any:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunIntegrityError(f"cannot read strict JSON state {path}: {exc}") from exc
    # ``parse_constant`` rejects the non-standard NaN/Infinity spellings, but a
    # standards-compliant token such as ``1e999`` still overflows to ``inf`` in
    # Python's decoder.  Reject the decoded tree as well.
    if not finite_tree(document):
        raise RunIntegrityError(f"strict JSON state {path} contains a non-finite numeric value")
    return document


def upsert_failure(failures: list[dict[str, Any]], failure: Mapping[str, Any]) -> None:
    """Keep only the latest explicit failure for one cell."""

    cell_id = failure.get("cell_id")
    if not isinstance(cell_id, str):
        raise RunIntegrityError("failure record is missing cell_id")
    failures[:] = [row for row in failures if row.get("cell_id") != cell_id]
    failures.append(dict(failure))


def clear_failure(failures: list[dict[str, Any]], cell_id: str) -> None:
    failures[:] = [row for row in failures if row.get("cell_id") != cell_id]


def build_ledger(
    expected_cell_ids: Iterable[str],
    conditions: Iterable[Mapping[str, Any]],
    failures: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate cell accounting and return an execution-completeness ledger.

    Completing every planned cell is necessary for scientific use, but it is not
    sufficient for publication.  In particular, completion says nothing about
    preregistration, target opening, metric validity, calibration/test separation,
    or independent checkpoints.  ``publication_eligible`` is retained only as an
    explicit fail-closed compatibility field and is therefore always false.
    """

    expected = list(expected_cell_ids)
    if len(expected) != len(set(expected)):
        raise RunIntegrityError("expected cell grid contains duplicate identities")
    completed = [row.get("cell_id") for row in conditions]
    failed = [row.get("cell_id") for row in failures]
    if any(not isinstance(value, str) for value in completed):
        raise RunIntegrityError("every completed condition must carry cell_id")
    if any(not isinstance(value, str) for value in failed):
        raise RunIntegrityError("every failure must carry cell_id")
    if len(completed) != len(set(completed)):
        raise RunIntegrityError("completed conditions contain duplicate cell identities")
    if len(failed) != len(set(failed)):
        raise RunIntegrityError("failure ledger contains duplicate cell identities")

    expected_set = set(expected)
    completed_set = set(completed)
    failed_set = set(failed)
    unexpected = sorted((completed_set | failed_set) - expected_set)
    if unexpected:
        raise RunIntegrityError(f"run state contains {len(unexpected)} unexpected cells")
    overlap = sorted(completed_set & failed_set)
    if overlap:
        raise RunIntegrityError(f"cells cannot be both completed and failed: {overlap[:3]}")
    missing = [cell_id for cell_id in expected if cell_id not in completed_set | failed_set]
    complete = completed_set == expected_set and not failed_set
    return {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "execution_complete": bool(complete),
        "publication_eligible": False,
        "publication_eligibility_note": (
            "execution completeness alone cannot establish publication eligibility"
        ),
        "expected_cells": len(expected),
        "completed_cells": len(completed),
        "failed_cells": len(failed),
        "missing_cells": len(missing),
        "failed_cell_ids": sorted(failed_set),
        "missing_cell_ids": missing,
    }


def _as_finite_float(value: Any, *, field: str, cell_id: str) -> float:
    """Parse one scientific score without accepting bool, NaN, or infinity."""

    if isinstance(value, bool):
        raise RunIntegrityError(f"cell {cell_id} field {field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RunIntegrityError(
            f"cell {cell_id} field {field} must be a finite number"
        ) from exc
    if not math.isfinite(parsed):
        raise RunIntegrityError(f"cell {cell_id} field {field} must be a finite number")
    return parsed


def _validate_completed_cell_records(
    records: Iterable[Mapping[str, Any]],
    conditions: Iterable[Mapping[str, Any]],
    *,
    require_scientific_cell_identity: bool = False,
) -> dict[str, Any]:
    """Require an exact, internally consistent record inventory for every cell.

    All current natural-shift runners store one condition row with ``cand_names``
    (the frozen anchor followed by every evaluated candidate) and one record per
    non-anchor candidate.  Earlier resume files treated the condition row alone
    as proof of completion, so an empty, truncated, duplicated, or wrong-candidate
    record set could be skipped on resume.  This validator makes the row inventory
    part of the completion contract and also checks the score arithmetic copied
    into the condition summary.
    """

    records_by_cell: dict[str, list[Mapping[str, Any]]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise RunIntegrityError(f"record {index} must be an object")
        cell_id = record.get("cell_id")
        if not isinstance(cell_id, str):
            raise RunIntegrityError(f"record {index} is missing cell_id")
        records_by_cell.setdefault(cell_id, []).append(record)

    inventory: dict[str, Any] = {}
    identity_fields = (
        "seed",
        "model_seed",
        "stream_seed",
        "domain",
        "location",
        "split",
        "comp",
        "regime",
        "aggr",
        "checkpoint_sha256",
    )
    for index, condition in enumerate(conditions):
        if not isinstance(condition, Mapping):
            raise RunIntegrityError(f"condition {index} must be an object")
        cell_id = condition.get("cell_id")
        if not isinstance(cell_id, str):
            raise RunIntegrityError(f"condition {index} is missing cell_id")
        rows = records_by_cell.get(cell_id, [])
        if not rows:
            raise RunIntegrityError(
                f"completed cell {cell_id} has zero candidate records"
            )

        condition_identity = condition.get("scientific_cell_identity")
        if require_scientific_cell_identity or condition_identity is not None:
            normalized_identity = validate_scientific_cell_identity(
                cell_id, condition_identity, context=f"completed cell {cell_id}"
            )
            for row in rows:
                if row.get("scientific_cell_identity") != normalized_identity:
                    raise RunIntegrityError(
                        f"cell {cell_id} candidate record scientific identity mismatch"
                    )

        candidate_names = condition.get("cand_names")
        scores = condition.get("aa_all")
        if not isinstance(candidate_names, list) or len(candidate_names) < 2:
            raise RunIntegrityError(
                f"completed cell {cell_id} lacks anchor-plus-candidate cand_names"
            )
        if any(not isinstance(name, str) or not name for name in candidate_names):
            raise RunIntegrityError(f"cell {cell_id} has invalid candidate names")
        if len(candidate_names) != len(set(candidate_names)):
            raise RunIntegrityError(f"cell {cell_id} has duplicate candidate names")
        if not isinstance(scores, list) or len(scores) != len(candidate_names):
            raise RunIntegrityError(
                f"cell {cell_id} aa_all must align exactly with cand_names"
            )
        parsed_scores = [
            _as_finite_float(value, field=f"aa_all[{score_index}]", cell_id=cell_id)
            for score_index, value in enumerate(scores)
        ]
        a0 = _as_finite_float(condition.get("a0"), field="a0", cell_id=cell_id)
        if not math.isclose(parsed_scores[0], a0, rel_tol=0.0, abs_tol=1e-12):
            raise RunIntegrityError(
                f"cell {cell_id} anchor score does not match condition a0"
            )

        expected_candidates = candidate_names[1:]
        observed_candidates = [row.get("candidate") for row in rows]
        if any(not isinstance(name, str) or not name for name in observed_candidates):
            raise RunIntegrityError(f"cell {cell_id} has a record without candidate")
        if len(observed_candidates) != len(set(observed_candidates)):
            raise RunIntegrityError(f"cell {cell_id} has duplicate candidate records")
        if set(observed_candidates) != set(expected_candidates):
            missing = sorted(set(expected_candidates) - set(observed_candidates))
            unexpected = sorted(set(observed_candidates) - set(expected_candidates))
            raise RunIntegrityError(
                f"cell {cell_id} candidate record inventory mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )

        score_by_candidate = dict(zip(candidate_names, parsed_scores))
        for row in rows:
            candidate = str(row["candidate"])
            row_a0 = _as_finite_float(row.get("a0"), field=f"{candidate}.a0", cell_id=cell_id)
            adapted = _as_finite_float(row.get("aa"), field=f"{candidate}.aa", cell_id=cell_id)
            benefit = _as_finite_float(row.get("B"), field=f"{candidate}.B", cell_id=cell_id)
            if not math.isclose(row_a0, a0, rel_tol=0.0, abs_tol=1e-12):
                raise RunIntegrityError(f"cell {cell_id} candidate {candidate} has wrong a0")
            if not math.isclose(
                adapted, score_by_candidate[candidate], rel_tol=0.0, abs_tol=1e-12
            ):
                raise RunIntegrityError(
                    f"cell {cell_id} candidate {candidate} score differs from aa_all"
                )
            if not math.isclose(benefit, adapted - a0, rel_tol=0.0, abs_tol=1e-12):
                raise RunIntegrityError(
                    f"cell {cell_id} candidate {candidate} violates B = aa - a0"
                )
            for field in identity_fields:
                if field in condition and field in row and row[field] != condition[field]:
                    raise RunIntegrityError(
                        f"cell {cell_id} candidate {candidate} has mismatched {field}"
                    )

        expected_best = max(parsed_scores[1:])
        if "best_adapt" in condition and not math.isclose(
            _as_finite_float(condition["best_adapt"], field="best_adapt", cell_id=cell_id),
            expected_best,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RunIntegrityError(f"cell {cell_id} best_adapt is inconsistent")
        if "oracle" in condition and not math.isclose(
            _as_finite_float(condition["oracle"], field="oracle", cell_id=cell_id),
            max(parsed_scores),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RunIntegrityError(f"cell {cell_id} oracle is inconsistent")

        inventory[cell_id] = {
            "candidate_count": len(expected_candidates),
            "candidates": list(expected_candidates),
            "condition_sha256": stable_sha256(dict(condition)),
            "records_sha256": stable_sha256(
                sorted((dict(row) for row in rows), key=lambda row: str(row["candidate"]))
            ),
        }
    return inventory


def _run_semantic_validator(
    semantic_validator: SemanticValidator,
    records: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> None:
    """Run a caller's external semantic checks on isolated JSON copies.

    The inventory commitment proves only that a document is internally
    self-consistent.  It cannot prove that stored predictions, evidence, sample
    identities, or protocols match the current dataset and scientific
    configuration.  Natural-shift runners therefore supply a validator that
    recomputes those facts from trusted runtime context.  Isolated copies keep a
    buggy validator from mutating the document that is about to be sealed.
    """

    copied_records = json.loads(canonical_json_bytes(records).decode("ascii"))
    copied_conditions = json.loads(canonical_json_bytes(conditions).decode("ascii"))
    semantic_validator(copied_records, copied_conditions)


def _require_semantic_validator(
    semantic_validator: SemanticValidator | None,
) -> SemanticValidator:
    if semantic_validator is None or not callable(semantic_validator):
        raise RunIntegrityError(
            "partial state requires a runner-specific semantic_validator; "
            "internal inventory hashes are not an authenticity check"
        )
    return semantic_validator


def partial_document(
    *,
    run_config_sha256: str,
    expected_cell_ids: Iterable[str],
    records: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    progress: str,
    require_scientific_cell_identity: bool = False,
    semantic_validator: SemanticValidator | None = None,
) -> dict[str, Any]:
    semantic_validator = _require_semantic_validator(semantic_validator)
    if not finite_tree({"records": records, "conditions": conditions, "failures": failures}):
        raise RunIntegrityError("partial state contains a non-finite numeric value")
    ledger = build_ledger(expected_cell_ids, conditions, failures)
    completed = {row["cell_id"] for row in conditions}
    orphan_records = [row.get("cell_id") for row in records if row.get("cell_id") not in completed]
    if orphan_records:
        raise RunIntegrityError(
            f"partial state contains {len(orphan_records)} records outside completed cells"
        )
    record_inventory = _validate_completed_cell_records(
        records,
        conditions,
        require_scientific_cell_identity=require_scientific_cell_identity,
    )
    _run_semantic_validator(semantic_validator, records, conditions)
    return {
        "schema": PARTIAL_SCHEMA,
        "run_config_sha256": run_config_sha256,
        "progress": progress,
        "ledger": ledger,
        "record_inventory": record_inventory,
        "records": records,
        "conditions": conditions,
        "failures": failures,
    }


def load_partial_state(
    path: str | os.PathLike[str],
    *,
    run_config_sha256: str,
    expected_cell_ids: Iterable[str],
    require_scientific_cell_identity: bool = False,
    semantic_validator: SemanticValidator | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    semantic_validator = _require_semantic_validator(semantic_validator)
    source = Path(path)
    if not source.exists():
        return [], [], []
    document = strict_json_load(source)
    if not isinstance(document, dict) or document.get("schema") != PARTIAL_SCHEMA:
        raise RunIntegrityError(
            f"refusing legacy or unrecognized partial state {source}; start in a fresh run directory"
        )
    if document.get("run_config_sha256") != run_config_sha256:
        raise RunIntegrityError(
            f"partial state config mismatch for {source}; start in a fresh run directory"
        )
    records = document.get("records")
    conditions = document.get("conditions")
    failures = document.get("failures")
    if not all(isinstance(value, list) for value in (records, conditions, failures)):
        raise RunIntegrityError("partial state records, conditions, and failures must be lists")
    rebuilt = partial_document(
        run_config_sha256=run_config_sha256,
        expected_cell_ids=expected_cell_ids,
        records=records,
        conditions=conditions,
        failures=failures,
        progress=str(document.get("progress", "resume")),
        require_scientific_cell_identity=require_scientific_cell_identity,
        semantic_validator=semantic_validator,
    )
    if document.get("record_inventory") != rebuilt["record_inventory"]:
        raise RunIntegrityError(
            f"partial state record_inventory commitment mismatch for {source}"
        )
    if document.get("ledger") != rebuilt["ledger"]:
        raise RunIntegrityError(f"partial state completion ledger mismatch for {source}")
    return records, conditions, failures


def finite_tree(value: Any) -> bool:
    """Return whether every numeric leaf is finite (bool is not treated as numeric)."""

    if isinstance(value, Mapping):
        return all(finite_tree(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(child) for child in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True
