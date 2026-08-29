#!/usr/bin/env python3
"""Leakage-safe natural-shift multi-seed extraction.

This utility converts monolithic WILDS/Office-Home result documents into the
per-condition schema used by the K-Bound stability analysis. Its contract is
deliberately strict:

* each invocation consumes one exact candidate set and one exact seed set;
* duplicate scientific cells are fatal (duplicate files are not replications);
* target-test labels are never admitted to the within-partition cross-fitted estimator;
* model-seed inference requires an explicit model seed and a distinct, valid
  checkpoint SHA-256 for every seed;
* stream-seed runs are labelled descriptive and can never receive a
  ``beats-both`` verdict; and
* aggregation consumes only files returned by the current serialization, from
  a fresh staging directory. It never globs pre-existing outputs.

The present cross-fitted analysis is therefore a development/stability diagnostic. A
held-out test aggregate must instead be produced by a scorer whose decisions
and calibration radius were locked on a disjoint validation partition.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
WILDS = REPO / "experiments" / "kbound" / "wilds"
sys.path.insert(0, str(WILDS))

import per_condition_serialize as pcs  # noqa: E402


LOCKED = {
    "officehome": ["sar_online_aggressive"],
    "iwildcam": ["tent_episodic"],
    "rxrx1": ["sar_online"],
}

DATASET_SLUG = {
    "officehome": "officehome",
    "iwildcam": "iwildcam",
    "rxrx1": "rxrx1",
}

METHOD_FIELD = {
    "officehome": "candidate",
    "iwildcam": "candidate",
    "rxrx1": "candidate",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HELDOUT_PARTITIONS = {"test", "target_test", "id_test", "heldout", "held_out"}
_CURRENT_SOURCE_SCHEMAS = {
    "kbound_officehome_v4",
    "kbound_wilds_iwildcam_finder_v0.5",
    "kbound_rxrx1_v0.6",
    "kbound_wilds_camelyon17_v0.8",
    "kbound_imagenetr_v0.7",
}
_SOURCE_CONTRACTS = {
    "officehome": {
        "schema": "kbound_officehome_v4",
        "dataset": "office-home",
        "metric": "accuracy",
    },
    "iwildcam": {
        "schema": "kbound_wilds_iwildcam_finder_v0.5",
        "dataset": "wilds-iwildcam",
        "metric": "diagnostic_per_cell_sklearn_macro_f1",
        "official_wilds_metric": False,
    },
    "rxrx1": {
        "schema": "kbound_rxrx1_v0.6",
        "dataset": "wilds-rxrx1",
        "metric": "balanced_accuracy",
    },
}


class LineageError(ValueError):
    """The requested aggregate does not have defensible scientific lineage."""


def _json_load(path):
    with open(path, encoding="utf-8") as handle:
        document = json.load(
            handle,
            parse_constant=lambda value: (_ for _ in ()).throw(
                LineageError(f"non-standard JSON constant {value!r}: {path}")
            ),
        )
    _require_finite_json_tree(document, path=str(path))
    return document


def _require_finite_json_tree(value, *, path="$"):
    if isinstance(value, float) and not math.isfinite(value):
        raise LineageError(f"non-finite JSON number at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_json_tree(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_json_tree(child, path=f"{path}[{index}]")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o644)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _expand(patterns):
    """Expand globs without basename- or size-based scientific de-duplication."""
    files = []
    for pattern in patterns:
        hits = sorted(glob.glob(pattern, recursive=True))
        if not hits and os.path.isfile(pattern):
            hits = [pattern]
        files.extend(hits)
    unique = []
    seen = set()
    for raw in files:
        path = str(Path(raw).resolve())
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _normalise_partition(value):
    if value is None:
        return None
    value = str(value).strip().lower().replace("-", "_")
    aliases = {
        "validation": "val",
        "target_validation": "target_val",
        "targettest": "target_test",
        "targetval": "target_val",
    }
    return aliases.get(value, value)


def _partition_family(value):
    value = _normalise_partition(value)
    if value in {"target_val", "val", "id_val", "source", "source_val", "dev"}:
        return "development"
    if value in _HELDOUT_PARTITIONS:
        return "heldout_test"
    return value or "unspecified"


def _source_partition(document, records, path):
    config = document.get("config") or {}
    data = document.get("data") or {}
    role = _normalise_partition(document.get("role") or config.get("role"))
    split = _normalise_partition(data.get("split") or config.get("split"))
    record_splits = {
        _normalise_partition(record.get("split"))
        for record in records
        if record.get("split") is not None
    }
    if len(record_splits) > 1:
        raise LineageError(f"source mixes record-level partitions {sorted(record_splits)}: {path}")
    record_split = next(iter(record_splits), None)
    declared = [value for value in (role, split, record_split) if value]
    families = {_partition_family(value) for value in declared}
    if len(families) > 1:
        raise LineageError(f"inconsistent partition declarations {declared}: {path}")
    # Role is more specific than a physical split (target_val + val).
    return role or split or record_split or "unspecified"


def _model_seed(document, record=None):
    config = document.get("config") or {}
    config_checkpoint = config.get("checkpoint")
    if not isinstance(config_checkpoint, dict):
        config_checkpoint = {}
    document_identity = document.get("model_identity")
    if not isinstance(document_identity, dict):
        document_identity = {}
    config_identity = config.get("model_identity")
    if not isinstance(config_identity, dict):
        config_identity = {}
    candidates = [
        document.get("model_seed"),
        config.get("model_seed"),
        config.get("train_seed"),
        document_identity.get("model_seed"),
        config_identity.get("model_seed"),
        config_checkpoint.get("model_seed"),
    ]
    if record is not None:
        candidates.insert(0, record.get("model_seed"))
    values = {int(value) for value in candidates if value is not None}
    if len(values) > 1:
        raise LineageError(f"conflicting model-seed declarations: {sorted(values)}")
    return next(iter(values), None)


def _checkpoint_sha256(document):
    config = document.get("config") or {}
    training = document.get("f0_training") or {}
    document_checkpoint = document.get("checkpoint")
    if not isinstance(document_checkpoint, dict):
        document_checkpoint = {}
    config_checkpoint = config.get("checkpoint")
    if not isinstance(config_checkpoint, dict):
        config_checkpoint = {}
    document_identity = document.get("model_identity")
    if not isinstance(document_identity, dict):
        document_identity = {}
    config_identity = config.get("model_identity")
    if not isinstance(config_identity, dict):
        config_identity = {}
    candidates = [
        document.get("f0_checkpoint_sha256"),
        document.get("checkpoint_sha256"),
        document.get("source_checkpoint_sha256"),
        training.get("best_checkpoint_sha256"),
        config.get("f0_checkpoint_sha256"),
        config.get("checkpoint_sha256"),
        document_checkpoint.get("sha256"),
        document_checkpoint.get("file_sha256"),
        config_checkpoint.get("sha256"),
        config_checkpoint.get("file_sha256"),
        document_identity.get("checkpoint_sha256"),
        config_identity.get("checkpoint_sha256"),
    ]
    values = {str(value).lower() for value in candidates if value not in (None, "")}
    invalid = sorted(value for value in values if not _SHA256.fullmatch(value))
    if invalid:
        raise LineageError(f"invalid checkpoint SHA-256 value(s): {invalid}")
    if len(values) > 1:
        raise LineageError("conflicting checkpoint SHA-256 declarations in one source")
    return next(iter(values), None)


def _checkpoint_path(document, source_path):
    config = document.get("config") or {}
    resume_payload = (document.get("resume_contract") or {}).get("payload") or {}
    resume_checkpoint = resume_payload.get("checkpoint") or {}
    document_checkpoint = document.get("checkpoint")
    if not isinstance(document_checkpoint, dict):
        document_checkpoint = {}
    config_checkpoint = config.get("checkpoint")
    if not isinstance(config_checkpoint, dict):
        config_checkpoint = {}
    candidates = [
        document.get("f0_checkpoint"),
        document.get("checkpoint") if not isinstance(document.get("checkpoint"), dict) else None,
        config.get("f0_checkpoint"),
        config.get("checkpoint") if not isinstance(config.get("checkpoint"), dict) else None,
        config.get("ckpt_resolved"),
        document_checkpoint.get("path"),
        config_checkpoint.get("path"),
        resume_checkpoint.get("path"),
    ]
    resolved = set()
    for value in candidates:
        if value in (None, "") or not isinstance(value, (str, os.PathLike)):
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(source_path).resolve().parent / path
        resolved.add(str(path.resolve()))
    if len(resolved) > 1:
        raise LineageError(f"conflicting checkpoint paths in source: {sorted(resolved)}")
    return next(iter(resolved), None)


def _checkpoint_tensor_sha256(document):
    config = document.get("config") or {}
    resume_payload = (document.get("resume_contract") or {}).get("payload") or {}
    resume_checkpoint = resume_payload.get("checkpoint") or {}
    document_checkpoint = document.get("checkpoint")
    if not isinstance(document_checkpoint, dict):
        document_checkpoint = {}
    config_checkpoint = config.get("checkpoint")
    if not isinstance(config_checkpoint, dict):
        config_checkpoint = {}
    document_identity = document.get("model_identity")
    if not isinstance(document_identity, dict):
        document_identity = {}
    config_identity = config.get("model_identity")
    if not isinstance(config_identity, dict):
        config_identity = {}
    values = {
        str(value).lower()
        for value in (
            document.get("f0_checkpoint_tensor_sha256"),
            document.get("checkpoint_tensor_sha256"),
            config.get("f0_checkpoint_tensor_sha256"),
            config.get("checkpoint_tensor_sha256"),
            document_checkpoint.get("tensor_sha256"),
            config_checkpoint.get("tensor_sha256"),
            document_identity.get("checkpoint_tensor_sha256"),
            config_identity.get("checkpoint_tensor_sha256"),
            resume_checkpoint.get("tensor_sha256"),
        )
        if value not in (None, "")
    }
    if len(values) > 1 or any(not _SHA256.fullmatch(value) for value in values):
        raise LineageError("conflicting or invalid checkpoint tensor SHA-256 declarations")
    return next(iter(values), None)


def _recompute_checkpoint_tensor_sha256(path):
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise LineageError(
            "model-seed extraction requires PyTorch to validate checkpoint tensor state"
        ) from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise LineageError(f"checkpoint is not a loadable PyTorch artifact: {path}") from exc
    if isinstance(payload, dict) and "algorithm" in payload:
        state = payload["algorithm"]
    elif isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        state = payload["model"]
    elif isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        state = payload["state_dict"]
    elif isinstance(payload, dict) and payload and all(isinstance(key, str) for key in payload):
        state = payload
    else:
        raise LineageError(f"checkpoint does not contain a typed state dictionary: {path}")
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if not isinstance(state, dict) or not state:
        raise LineageError(f"checkpoint does not contain a typed state dictionary: {path}")
    digest = hashlib.sha256()
    tensor_count = 0
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {"name": str(name), "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        tensor_count += 1
    if tensor_count == 0:
        raise LineageError(f"checkpoint contains no tensors: {path}")
    return digest.hexdigest()


def _validate_source_contract(document, source_path, track):
    expected = _SOURCE_CONTRACTS[track]
    mismatches = {
        field: (document.get(field), value)
        for field, value in expected.items()
        if document.get(field) != value
    }
    if mismatches:
        raise LineageError(f"source dataset/schema/metric contract mismatch: {mismatches}: {source_path}")
    resume = document.get("resume_contract")
    if track in {"officehome", "iwildcam"}:
        if not isinstance(resume, dict) or not isinstance(resume.get("payload"), dict):
            raise LineageError(f"current {track} source lacks a bound resume/scientific contract: {source_path}")
        payload = json.loads(json.dumps(resume["payload"]))
        payload.pop("checkpoint", None)
        seed_semantics = payload.get("seed_semantics")
        if not isinstance(seed_semantics, dict):
            raise LineageError(f"source scientific contract lacks seed semantics: {source_path}")
        seed_semantics.pop("model_seed", None)
    else:
        scientific_config = json.loads(json.dumps(document.get("config") or {}))
        # RxRx1's runner correctly embeds the independently trained model identity
        # in its scientific config.  Those fields must be verified separately, but
        # cannot participate in the cross-seed protocol fingerprint: otherwise two
        # legitimate model seeds are rejected merely because their checkpoints differ.
        for field in (
            "model_seed",
            "checkpoint",
            "model_identity",
            "f0_checkpoint",
            "f0_checkpoint_sha256",
            "f0_checkpoint_tensor_sha256",
            "checkpoint_sha256",
            "checkpoint_tensor_sha256",
        ):
            scientific_config.pop(field, None)
        payload = {
            "scientific_config_without_model_identity": scientific_config,
            "population_sha256": (document.get("data") or {}).get("population_sha256"),
        }
    population = (
        document.get("population_manifest")
        if track == "officehome"
        else (document.get("data") or {}).get("population_manifest")
        if track == "iwildcam"
        else (document.get("data") or {}).get("population_sha256")
    )
    if not population:
        raise LineageError(f"source lacks an evaluation-population fingerprint: {source_path}")
    protocol = {
        "contract": expected,
        "partition": _source_partition(document, document.get("records") or [], source_path),
        "resume_payload_without_model": payload,
        "population": population,
    }
    return _sha256_json(protocol), protocol


def _ledger_publication_state(document, path):
    """Return fail-closed source-ledger eligibility for downstream promotion."""
    schema = document.get("schema")
    declared = document.get("publication_eligible") is True
    ledger = (
        document.get("completion_ledger")
        or document.get("run_ledger")
        or document.get("ledger")
    )
    if not isinstance(ledger, dict):
        if declared:
            raise LineageError(
                f"source declares publication_eligible=true without a completion ledger: {path}"
            )
        return {
            "schema": schema,
            "publication_eligible_declared": False,
            "ledger_present": False,
            "ledger_complete_failure_free": False,
            "publication_ready": False,
            "diagnostic_only_reason": (
                "current schema lacks a completion ledger"
                if schema in _CURRENT_SOURCE_SCHEMAS
                else "legacy/unknown schema has no completion ledger"
            ),
        }

    status = str(ledger.get("status", "")).strip().upper()
    expected = ledger.get("expected_cells", ledger.get("expected"))
    completed = ledger.get("completed_cells", ledger.get("completed"))
    failed = ledger.get("failed_cells", ledger.get("failed", 0))
    pending = ledger.get("missing_cells", ledger.get("pending", 0))
    if isinstance(failed, list):
        failed = len(failed)
    if isinstance(pending, list):
        pending = len(pending)
    try:
        expected_i = int(expected)
        completed_i = int(completed)
        failed_i = int(failed)
        pending_i = int(pending)
    except (TypeError, ValueError) as exc:
        raise LineageError(f"source completion ledger has non-numeric counts: {path}") from exc

    failure_lists = [
        ledger.get("failed_cell_ids", []),
        ledger.get("pending_keys", []),
        ledger.get("missing_cell_ids", []),
        ledger.get("failure_history", []),
        document.get("failures", []),
    ]
    failure_lists_empty = all(isinstance(rows, list) and not rows for rows in failure_lists)
    ledger_complete = (
        status == "COMPLETE"
        and expected_i > 0
        and expected_i == completed_i
        and failed_i == 0
        and pending_i == 0
        and failure_lists_empty
        and ledger.get("execution_complete", True) is True
    )
    conditions = document.get("conditions")
    if schema in _CURRENT_SOURCE_SCHEMAS and (
        not isinstance(conditions, list) or len(conditions) != expected_i
    ):
        ledger_complete = False
    if not ledger_complete:
        raise LineageError(
            "source completion ledger is incomplete or records failures: "
            f"expected={expected_i}, completed={completed_i}, failed={failed_i}, "
            f"pending={pending_i}, status={status!r}: {path}"
        )
    # A legacy/unknown document cannot become promotable merely by carrying
    # fields that resemble the current ledger contract.  Its payload was not
    # produced under a schema whose completeness semantics we know how to
    # verify, so retain it only as diagnostic evidence.
    current_schema = schema in _CURRENT_SOURCE_SCHEMAS
    publication_ready = bool(current_schema and declared and ledger_complete)
    return {
        "schema": schema,
        "publication_eligible_declared": declared,
        "ledger_present": True,
        "ledger_complete_failure_free": True,
        "expected": expected_i,
        "completed": completed_i,
        "failed": failed_i,
        "pending": pending_i,
        "publication_ready": publication_ready,
        "diagnostic_only_reason": (
            None
            if publication_ready
            else (
                "legacy/unknown schema is diagnostic-only"
                if not current_schema
                else "source lacks top-level publication_eligible=true"
            )
        ),
    }


def _load_records(paths, *, track=None, verify_checkpoints=False):
    records = []
    evidence = None
    sources = []
    for path in paths:
        document = _json_load(path)
        source_records = document.get("records") or []
        if not source_records:
            raise LineageError(f"no records[] in {path}")
        names = document.get("evidence_names")
        if names is not None:
            names = list(names)
            if evidence is not None and names != evidence:
                raise LineageError(f"evidence-name mismatch across sources: {path}")
            evidence = names
        partition = _source_partition(document, source_records, path)
        checkpoint = _checkpoint_sha256(document)
        checkpoint_path = _checkpoint_path(document, path)
        checkpoint_tensor = _checkpoint_tensor_sha256(document)
        checkpoint_verified = False
        if verify_checkpoints:
            if checkpoint is None or checkpoint_path is None or checkpoint_tensor is None:
                raise LineageError(
                    f"model-seed source lacks checkpoint path/file/tensor hashes: {path}"
                )
            artifact_path = Path(checkpoint_path)
            if not artifact_path.is_file() or _sha256_file(artifact_path) != checkpoint:
                raise LineageError(f"declared checkpoint file hash does not match an existing artifact: {path}")
            if _recompute_checkpoint_tensor_sha256(artifact_path) != checkpoint_tensor:
                raise LineageError(f"declared checkpoint tensor hash mismatch: {artifact_path}")
            checkpoint_verified = True
        protocol_fingerprint = None
        protocol = None
        if track is not None:
            protocol_fingerprint, protocol = _validate_source_contract(document, path, track)
        document_model_seed = _model_seed(document)
        publication = _ledger_publication_state(document, path)
        source_info = {
            "path": str(Path(path).resolve()),
            "sha256": _sha256_file(path),
            "n_records": len(source_records),
            "partition": partition,
            "model_seed": document_model_seed,
            "checkpoint_sha256": checkpoint,
            "checkpoint_path": checkpoint_path,
            "checkpoint_tensor_sha256": checkpoint_tensor,
            "checkpoint_verified": checkpoint_verified,
            "source_contract_validated": track is not None,
            "protocol_fingerprint": protocol_fingerprint,
            "protocol": protocol,
            "publication": publication,
        }
        sources.append(source_info)
        for source_index, original in enumerate(source_records):
            record = dict(original)
            record_model_seed = _model_seed(document, record)
            record["_source_path"] = source_info["path"]
            record["_source_sha256"] = source_info["sha256"]
            record["_source_partition"] = partition
            record["_source_model_seed"] = record_model_seed
            record["_source_checkpoint_sha256"] = checkpoint
            record["_source_checkpoint_path"] = checkpoint_path
            record["_source_checkpoint_tensor_sha256"] = checkpoint_tensor
            record["_source_checkpoint_verified"] = checkpoint_verified
            record["_source_protocol_fingerprint"] = protocol_fingerprint
            sample_provenance = original.get("sample_provenance")
            record["_evaluation_population_sha256"] = (
                _sha256_json(sample_provenance)
                if isinstance(sample_provenance, dict) and sample_provenance
                else None
            )
            record["_source_record_index"] = source_index
            record["_source_record_sha256"] = _sha256_json(original)
            records.append(record)
    return records, evidence, sources


def _validate_candidate_set(records, method_field, requested):
    missing_field = [index for index, record in enumerate(records) if record.get(method_field) is None]
    if missing_field:
        raise LineageError(
            f"{len(missing_field)} records lack required candidate field {method_field!r}"
        )
    observed = {str(record[method_field]) for record in records}
    requested = set(requested)
    if observed != requested:
        missing = sorted(requested - observed)
        unexpected = sorted(observed - requested)
        raise LineageError(
            "source candidate set is not the exact requested set; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _validate_metric_rows(records):
    for index, record in enumerate(records):
        try:
            a0 = float(record["a0"])
            adapted = float(record["aa"])
            benefit = float(record["B"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LineageError(f"record {index} lacks numeric a0/aa/B") from exc
        if not np.isfinite([a0, adapted, benefit]).all():
            raise LineageError(f"record {index} has non-finite a0/aa/B")
        if not np.isclose(benefit, adapted - a0, rtol=0.0, atol=1e-10):
            raise LineageError(
                f"record {index} violates B = aa - a0: {benefit} != {adapted - a0}"
            )


def _expected_tta_protocol(record):
    candidate = str(record.get("candidate", record.get("method", "")))
    mode = record.get("mode")
    if mode is None:
        tokens = candidate.split("_")
        mode = next((token for token in tokens if token in {"online", "episodic"}), None)
    if mode in {"online", "episodic"}:
        gradient_reads_eval = mode == "episodic"
        semantics = (
            "episodic_transductive_eval_batch_update_and_evaluation"
            if gradient_reads_eval
            else "online_disjoint_stream_update_then_transductive_bn_evaluation"
        )
        return {
            "schema": "kbound_tta_candidate_protocol_v1",
            "mode": mode,
            "semantics": semantics,
            "requires_auxiliary_stream_eval_disjoint": True,
            "gradient_update_reads_eval_x": gradient_reads_eval,
            "prediction_uses_eval_batch_statistics": True,
            "candidate_evaluation_is_transductive": True,
            "candidate_adaptation_eval_disjoint": False,
            "target_labels_used_for_adaptation_or_prediction": False,
        }
    if candidate in {"labelshift", "conservative"}:
        return {
            "schema": "kbound_tta_candidate_protocol_v1",
            "mode": "inference_only_stream_prior",
            "semantics": (
                f"{candidate}_fit_on_disjoint_auxiliary_stream_then_inductive_evaluation"
            ),
            "requires_auxiliary_stream_eval_disjoint": True,
            "gradient_update_reads_eval_x": False,
            "prediction_uses_eval_batch_statistics": False,
            "candidate_evaluation_is_transductive": False,
            "candidate_adaptation_eval_disjoint": True,
            "target_labels_used_for_adaptation_or_prediction": False,
        }
    raise LineageError(
        f"cannot derive a reviewed TTA data-use protocol for candidate {candidate!r}"
    )


def _validate_tta_protocol_records(records, method_field):
    by_candidate = {}
    for index, record in enumerate(records):
        candidate = str(record.get(method_field, ""))
        expected = _expected_tta_protocol(record)
        if record.get("tta_protocol") != expected:
            raise LineageError(
                f"record {index} candidate {candidate!r} lacks its exact TTA data-use protocol"
            )
        prior = by_candidate.setdefault(candidate, expected)
        if prior != expected:
            raise LineageError(f"candidate {candidate!r} has conflicting TTA protocols")
    return by_candidate


def _prepare_records(records, sources, track, candidates, expected_seeds, seed_kind):
    slug = DATASET_SLUG[track]
    method_field = METHOD_FIELD[track]
    expected = {int(seed) for seed in expected_seeds}
    if not expected:
        raise LineageError("expected seed set must not be empty")
    if len(expected) != len(expected_seeds):
        raise LineageError("expected seed list contains duplicates")
    _validate_candidate_set(records, method_field, candidates)
    _validate_metric_rows(records)
    tta_protocol_by_candidate = _validate_tta_protocol_records(records, method_field)

    partitions = {source["partition"] for source in sources}
    families = {_partition_family(partition) for partition in partitions}
    if len(families) != 1 or len(partitions) != 1:
        raise LineageError(f"one extraction may consume exactly one partition; found {sorted(partitions)}")
    partition = next(iter(partitions))
    family = _partition_family(partition)
    if family == "heldout_test":
        raise LineageError(
            f"partition {partition!r} is held-out test data. Its labels cannot enter the cross-fit "
            "calibration pool; score it only with decisions/radii locked on a disjoint "
            "validation partition. No aggregate was emitted."
        )
    if family != "development":
        raise LineageError(
            f"source partition {partition!r} is not in the closed development allowlist; "
            "unknown/test-like labels cannot enter cross-fit calibration"
        )
    unvalidated_sources = [source["path"] for source in sources if source.get("source_contract_validated") is not True]
    if unvalidated_sources:
        raise LineageError(
            f"source dataset/schema/metric/scientific contracts were not validated: {unvalidated_sources}"
        )
    protocol_fingerprints = {source.get("protocol_fingerprint") for source in sources}
    if None in protocol_fingerprints or len(protocol_fingerprints) != 1:
        raise LineageError(
            "model-seed sources differ in scientific protocol or evaluation-population fingerprint"
        )
    sources_publication_ready = bool(sources) and all(
        source["publication"]["publication_ready"] for source in sources
    )

    prepared = []
    seed_metadata = {}
    if seed_kind == "model":
        missing_model = [source["path"] for source in sources if source["model_seed"] is None]
        missing_checkpoint = [source["path"] for source in sources if source["checkpoint_sha256"] is None]
        if missing_model or missing_checkpoint:
            raise LineageError(
                "model-seed inference requires model_seed and checkpoint SHA-256 on every source; "
                f"missing_model_seed={missing_model}, missing_checkpoint_sha256={missing_checkpoint}"
            )
        unverified_checkpoints = [
            source["path"]
            for source in sources
            if source.get("checkpoint_verified") is not True
            or not source.get("checkpoint_path")
            or not source.get("checkpoint_tensor_sha256")
        ]
        if unverified_checkpoints:
            raise LineageError(
                "model-seed inference requires existing, loadable, file- and tensor-hash-verified "
                f"checkpoint artifacts: {unverified_checkpoints}"
            )
        checkpoint_by_seed = defaultdict(set)
        checkpoint_tensor_by_seed = defaultdict(set)
        stream_sets = defaultdict(set)
        observed = set()
        for record in records:
            model_seed = record["_source_model_seed"]
            checkpoint = record["_source_checkpoint_sha256"]
            checkpoint_tensor = record["_source_checkpoint_tensor_sha256"]
            if model_seed is None or checkpoint is None:
                raise LineageError("record is missing inherited model/checkpoint provenance")
            if record.get("_source_checkpoint_verified") is not True:
                raise LineageError("record is not bound to a verified checkpoint artifact")
            if not record.get("_source_protocol_fingerprint"):
                raise LineageError("record is missing a scientific-protocol fingerprint")
            if not record.get("_evaluation_population_sha256"):
                raise LineageError("record is missing ordered evaluation-sample provenance")
            if "seed" not in record:
                raise LineageError("record is missing stream seed")
            stream_seed = int(record["seed"])
            model_seed = int(model_seed)
            observed.add(model_seed)
            checkpoint_by_seed[model_seed].add(checkpoint)
            checkpoint_tensor_by_seed[model_seed].add(checkpoint_tensor)
            stream_sets[model_seed].add(stream_seed)
            normalised = dict(record)
            normalised["stream_seed"] = stream_seed
            normalised["model_seed"] = model_seed
            normalised["checkpoint_sha256"] = checkpoint
            normalised["seed"] = model_seed
            prepared.append(normalised)
        if observed != expected:
            raise LineageError(
                f"model-seed set mismatch: expected={sorted(expected)}, observed={sorted(observed)}"
            )
        for seed in sorted(observed):
            if len(checkpoint_by_seed[seed]) != 1:
                raise LineageError(f"model seed {seed} maps to multiple checkpoints")
            if len(checkpoint_tensor_by_seed[seed]) != 1:
                raise LineageError(f"model seed {seed} maps to multiple checkpoint tensor states")
            if len(stream_sets[seed]) != 1:
                raise LineageError(
                    f"model seed {seed} has multiple stream seeds {sorted(stream_sets[seed])}; "
                    "model-level aggregation requires a fixed stream seed"
                )
        hashes = [next(iter(checkpoint_by_seed[seed])) for seed in sorted(observed)]
        if len(set(hashes)) != len(hashes):
            raise LineageError(
                "checkpoint hashes are not unique across model seeds; repeated checkpoints "
                "are not independent model replications"
            )
        tensor_hashes = [
            next(iter(checkpoint_tensor_by_seed[seed])) for seed in sorted(observed)
        ]
        if len(set(tensor_hashes)) != len(tensor_hashes):
            raise LineageError(
                "checkpoint tensor-state hashes are not unique across model seeds; "
                "byte-different containers of the same model are not independent replications"
            )
        stream_seed_values = {next(iter(values)) for values in stream_sets.values()}
        if len(stream_seed_values) != 1:
            raise LineageError(
                "the stream seed must be fixed across model seeds to isolate model variation; "
                f"found {sorted(stream_seed_values)}"
            )
        for seed in sorted(observed):
            seed_metadata[seed] = {
                "seed_kind": "model_seed",
                "model_seed": seed,
                "stream_seed": next(iter(stream_sets[seed])),
                "checkpoint_sha256": next(iter(checkpoint_by_seed[seed])),
                "checkpoint_tensor_sha256": next(iter(checkpoint_tensor_by_seed[seed])),
                "inference_unit": "independent model checkpoint",
                "evaluation_partition": partition,
            }
        inference = {
            "seed_kind": "model_seed",
            "inference_unit": "independent model checkpoint",
            "model_seed_ci_eligible": True,
            "confirmatory_ci_eligible": False,
            "sources_publication_ready": sources_publication_ready,
            "claim_scope": "development-partition stability; not held-out test confirmation",
            "evaluation_partition": partition,
            "tta_protocol_by_candidate": tta_protocol_by_candidate,
        }
    else:
        observed = set()
        for record in records:
            if "seed" not in record:
                raise LineageError("record is missing stream seed")
            stream_seed = int(record["seed"])
            observed.add(stream_seed)
            normalised = dict(record)
            normalised["stream_seed"] = stream_seed
            normalised["seed"] = stream_seed
            prepared.append(normalised)
        if observed != expected:
            raise LineageError(
                f"stream-seed set mismatch: expected={sorted(expected)}, observed={sorted(observed)}"
            )
        for seed in sorted(observed):
            seed_metadata[seed] = {
                "seed_kind": "stream_seed",
                "stream_seed": seed,
                "inference_unit": "stream order (shared or unknown model checkpoint)",
                "evaluation_partition": partition,
                "confirmatory_ci_eligible": False,
            }
        inference = {
            "seed_kind": "stream_seed",
            "inference_unit": "stream order (shared or unknown model checkpoint)",
            "model_seed_ci_eligible": False,
            "confirmatory_ci_eligible": False,
            "sources_publication_ready": sources_publication_ready,
            "claim_scope": "descriptive stream-seed sensitivity only; no beats-both promotion",
            "evaluation_partition": partition,
            "tta_protocol_by_candidate": tta_protocol_by_candidate,
        }

    key_fn = pcs.CONDITION_KEYS[slug]
    scientific_keys = [
        (str(record[method_field]), int(record["seed"]), key_fn(record))
        for record in prepared
    ]
    duplicates = [key for key, count in Counter(scientific_keys).items() if count > 1]
    if duplicates:
        preview = "; ".join(repr(key) for key in sorted(duplicates)[:5])
        raise LineageError(
            f"duplicate scientific condition keys detected ({len(duplicates)}): {preview}. "
            "Do not concatenate historical copies or repeated evaluations as independent rows."
        )
    if seed_kind == "model":
        population_by_condition = defaultdict(dict)
        for record in prepared:
            condition = (str(record[method_field]), key_fn(record))
            population_by_condition[condition][int(record["model_seed"])] = record[
                "_evaluation_population_sha256"
            ]
        for condition, by_seed in population_by_condition.items():
            if set(by_seed) != expected:
                raise LineageError(
                    f"scientific condition {condition!r} is not present for every model seed"
                )
            if len(set(by_seed.values())) != 1:
                raise LineageError(
                    f"evaluation sample population differs across model seeds for {condition!r}"
                )
    return prepared, seed_metadata, inference


def _bind_serialized_tta_protocols(serialize, protocols):
    """Enrich and reseal the fresh serializer generation with validated protocols."""
    methods = list(serialize["methods"])
    seeds = [int(seed) for seed in serialize["seeds"]]
    if set(protocols) != set(methods):
        raise LineageError("serialized candidate inventory differs from validated TTA protocols")
    payloads = {}
    for raw_path in serialize["written"]:
        path = Path(raw_path)
        document = _json_load(path)
        method = document.get("method")
        protocol = protocols.get(method)
        if protocol is None:
            raise LineageError(f"serialized file has no validated TTA protocol: {path}")
        document.pop("serialization_generation_id", None)
        document["tta_protocol"] = protocol
        document["tta_protocol_sha256"] = _sha256_json(protocol)
        for record in document.get("records") or []:
            record["tta_protocol"] = protocol
            record["tta_protocol_sha256"] = _sha256_json(protocol)
        payloads[path.name] = document
    generation_material = {
        "schema": "kbound_per_condition_generation_v1",
        "dataset": serialize["dataset"],
        "methods": methods,
        "seeds": seeds,
        "payloads": payloads,
    }
    generation_id = _sha256_json(generation_material)
    descriptors = {}
    for raw_path in serialize["written"]:
        path = Path(raw_path)
        payload = payloads[path.name]
        payload["serialization_generation_id"] = generation_id
        _atomic_json_dump(payload, path)
        descriptors[path.name] = {
            "sha256": _sha256_file(path),
            "n_conditions": int(payload["n_conditions"]),
        }
    manifest_path = Path(serialize["manifest"])
    _atomic_json_dump(
        {
            "schema": "kbound_per_condition_generation_v1",
            "generation_id": generation_id,
            "generation_committed": True,
            "dataset": serialize["dataset"],
            "methods": methods,
            "seeds": seeds,
            "expected_cells": len(methods) * len(seeds),
            "validated_tta_protocol_by_candidate": protocols,
            "files": descriptors,
        },
        manifest_path,
    )
    serialize["generation_id"] = generation_id
    return serialize


def _per_seed_from_file(path):
    document = _json_load(path)
    records = document.get("records") or []
    if not records:
        raise LineageError(f"no per-condition records in {path}")
    backend = document.get("kga_backend")
    if document.get("estimator_publication_eligible") is not False:
        raise LineageError(f"inconsistent estimator publication eligibility in {path}")
    conditions = [str(record.get("condition")) for record in records]
    duplicates = [key for key, count in Counter(conditions).items() if count > 1]
    if duplicates:
        raise LineageError(f"duplicate per-condition keys in {path}: {duplicates[:5]}")
    a0 = np.array([record["a0"] for record in records], float)
    adapted = np.array([record["a_adapted"] for record in records], float)
    decisions = [str(record.get("kga_decision", "")).upper() for record in records]
    feasibility = []
    for index, record in enumerate(records):
        if not isinstance(record.get("calibration_feasible"), bool):
            raise LineageError(
                f"record {index} lacks explicit calibration_feasible boolean in {path}"
            )
        feasible = record["calibration_feasible"]
        expected_status = "FINITE" if feasible else "INFEASIBLE"
        if record.get("radius_status") != expected_status:
            raise LineageError(
                f"record {index} has inconsistent radius_status/calibration_feasible in {path}"
            )
        if feasible and record.get("eps_conformal") is None:
            raise LineageError(f"record {index} has a feasible but null radius in {path}")
        if not feasible and (
            record.get("eps_conformal") is not None
            or record.get("benefit_ci") is not None
            or record.get("gamma_ci") is not None
            or decisions[index] != "ABSTAIN"
            or not np.isclose(
                float(record.get("a_kbound", record["a0"])),
                float(record["a0"]),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise LineageError(f"record {index} violates INFEASIBLE/null/ABSTAIN contract in {path}")
        feasibility.append(feasible)
    adapt = np.array([decision == "ADAPT" for decision in decisions])
    oracle = np.array(
        [
            record["a_oracle"]
            if record.get("a_oracle") is not None
            else max(record["a0"], record["a_adapted"])
            for record in records
        ],
        float,
    )
    routed = np.array(
        [
            record["a_kbound"]
            if record.get("a_kbound") is not None
            else (adapted[index] if adapt[index] else a0[index])
            for index, record in enumerate(records)
        ],
        float,
    )
    benefit = np.array([record["B"] for record in records], float)
    return {
        "seed": int(document["seed"]),
        "seed_kind": document.get("seed_kind"),
        "checkpoint_sha256": document.get("checkpoint_sha256"),
        "checkpoint_tensor_sha256": document.get("checkpoint_tensor_sha256"),
        "stream_seed": document.get("stream_seed"),
        "partition": document.get("evaluation_partition"),
        "tta_protocol": document.get("tta_protocol"),
        "n": len(records),
        "conditions": conditions,
        "rk": float((oracle - routed).mean()),
        "ra": float((oracle - adapted).mean()),
        "rf": float((oracle - a0).mean()),
        "fau": float(np.mean(adapt & (benefit <= 0))),
        "backend": backend,
        "n_calibration_infeasible": int(sum(not value for value in feasibility)),
        "calibration_feasible_all": bool(all(feasibility)),
        "n_abstain": int(sum(decision == "ABSTAIN" for decision in decisions)),
        "all_abstain_due_infeasible": bool(
            all(decision == "ABSTAIN" for decision in decisions)
            and any(not value for value in feasibility)
        ),
    }


def _boot(values, nb=5000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    n = len(values)
    draws = np.empty(nb)
    for index in range(nb):
        draws[index] = values[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def aggregate_candidate(
    dataset,
    candidate,
    per_condition_files,
    expected_seeds,
    inference,
    alpha=0.10,
):
    """Aggregate exactly the current serializer outputs; never inspect a directory."""
    files = [str(Path(path).resolve()) for path in per_condition_files]
    if not files:
        raise LineageError(f"current source produced no files for requested candidate {candidate!r}")
    summaries = []
    for path in files:
        document = _json_load(path)
        if document.get("benchmark") != dataset or document.get("method") != candidate:
            raise LineageError(f"unexpected dataset/candidate payload in {path}")
        summaries.append(_per_seed_from_file(path))
    seeds = [summary["seed"] for summary in summaries]
    if len(seeds) != len(set(seeds)):
        raise LineageError(f"duplicate seed files in current serialization: {seeds}")
    expected = {int(seed) for seed in expected_seeds}
    if set(seeds) != expected:
        raise LineageError(
            f"serialized seed set mismatch: expected={sorted(expected)}, observed={sorted(seeds)}"
        )
    summaries.sort(key=lambda summary: summary["seed"])
    reference_conditions = summaries[0]["conditions"]
    for summary in summaries[1:]:
        if summary["conditions"] != reference_conditions:
            raise LineageError(
                "condition order/set differs across seeds; paired model-seed summaries are invalid"
            )
    expected_tta_protocol = inference["tta_protocol_by_candidate"][candidate]
    if any(summary["tta_protocol"] != expected_tta_protocol for summary in summaries):
        raise LineageError(
            f"serialized TTA protocol differs from validated source contract for {candidate!r}"
        )

    rk = np.array([summary["rk"] for summary in summaries])
    ra = np.array([summary["ra"] for summary in summaries])
    rf = np.array([summary["rf"] for summary in summaries])
    fau = np.array([summary["fau"] for summary in summaries])
    better = "freeze" if rf.mean() <= ra.mean() else "adapt"
    gap_vs_adapt = ra - rk
    gap_vs_freeze = rf - rk
    gap_better = gap_vs_freeze if better == "freeze" else gap_vs_adapt
    gap_worse = gap_vs_adapt if better == "freeze" else gap_vs_freeze

    model_inference = inference["seed_kind"] == "model_seed"
    enough_model_seeds = model_inference and len(summaries) >= 2
    calibration_feasible_all = all(
        summary["calibration_feasible_all"] for summary in summaries
    )
    backend_set = sorted({summary["backend"] for summary in summaries if summary["backend"]})
    estimator_publication_eligible = False
    statistical_verdict_withheld = not calibration_feasible_all
    if enough_model_seeds and calibration_feasible_all:
        ci_b = _boot(gap_better)
        ci_w = _boot(gap_worse)
        ci_a = _boot(gap_vs_adapt)
        ci_f = _boot(gap_vs_freeze)
        ties_better = ci_b[0] <= 0 <= ci_b[1]
        beats_worse = ci_w[0] > 0
        beats_both = ci_b[0] > 0 and beats_worse
        fa_ok = bool(np.all(fau <= alpha))
        statistical_verdict = (
            "DEVELOPMENT_BEATS_BOTH_DIAGNOSTIC"
            if beats_both and fa_ok
            else "DEVELOPMENT_STABLE_NO_HARM_DIAGNOSTIC"
            if ties_better and beats_worse and fa_ok
            else "DEVELOPMENT_OTHER_DIAGNOSTIC"
        )
    elif statistical_verdict_withheld:
        ci_b = ci_w = ci_a = ci_f = None
        beats_both = False
        fa_ok = bool(np.all(fau <= alpha))
        statistical_verdict = "WITHHELD_INFEASIBLE_EXACT_RANK_CALIBRATION"
    else:
        ci_b = ci_w = ci_a = ci_f = None
        beats_both = False
        fa_ok = bool(np.all(fau <= alpha))
        statistical_verdict = "DESCRIPTIVE_STREAM_SEED_ONLY"

    # Cross-fitting consumes labels from disjoint development folds. It is useful for
    # debugging/stability, but it is never promoted as a held-out natural win.
    verdict = statistical_verdict
    output = {
        "schema": "kbound_natural_multiseed_aggregate_v2",
        "lineage_contract": "leakage_safe_multiseed_v2",
        "dataset": dataset,
        "candidate": candidate,
        "analysis": "crossfit_split_within_development_partition_single_candidate",
        "seed_kind": inference["seed_kind"],
        "inference_unit": inference["inference_unit"],
        "evaluation_partition": inference["evaluation_partition"],
        "claim_scope": inference["claim_scope"],
        "tta_protocol": inference["tta_protocol_by_candidate"][candidate],
        "tta_protocol_sha256": _sha256_json(
            inference["tta_protocol_by_candidate"][candidate]
        ),
        "external_authenticity_verified": False,
        "publication_eligible": False,
        "sources_publication_ready": bool(inference.get("sources_publication_ready")),
        "estimator_publication_eligible": estimator_publication_eligible,
        "model_seed_ci_eligible": bool(enough_model_seeds and calibration_feasible_all),
        "confirmatory_ci_eligible": False,
        "heldout_promotion_eligible": False,
        "beats_both_promoted": False,
        "analysis_note": (
            "The scored fold is excluded from both estimator fitting and radius calibration, "
            "and target-test labels are prohibited. Model-seed intervals resample independent checkpoint "
            "means. This development diagnostic cannot substantiate a held-out natural-shift claim."
        ),
        "seeds": [summary["seed"] for summary in summaries],
        "n_seeds": len(summaries),
        "conditions_per_seed": summaries[0]["n"],
        "alpha": alpha,
        "bootstrap_replicates": 5000,
        "bootstrap_seed": 0,
        "kga_backend": backend_set,
        "calibration_feasible_all": calibration_feasible_all,
        "n_calibration_infeasible_total": int(
            sum(summary["n_calibration_infeasible"] for summary in summaries)
        ),
        "n_calibration_infeasible_per_seed": {
            str(summary["seed"]): summary["n_calibration_infeasible"]
            for summary in summaries
        },
        "all_abstain_due_infeasible_per_seed": {
            str(summary["seed"]): summary["all_abstain_due_infeasible"]
            for summary in summaries
        },
        "statistical_verdict_withheld": statistical_verdict_withheld,
        "statistical_verdict_withheld_reason": (
            "one or more scored records have INFEASIBLE exact-rank calibration"
            if statistical_verdict_withheld
            else None
        ),
        "checkpoint_sha256_by_seed": {
            str(summary["seed"]): summary["checkpoint_sha256"]
            for summary in summaries
            if summary["checkpoint_sha256"]
        },
        "checkpoint_tensor_sha256_by_seed": {
            str(summary["seed"]): summary["checkpoint_tensor_sha256"]
            for summary in summaries
            if summary["checkpoint_tensor_sha256"]
        },
        "stream_seed_by_seed": {
            str(summary["seed"]): summary["stream_seed"] for summary in summaries
        },
        "regret_kga": [round(float(rk.mean()), 4), round(float(rk.std()), 4)],
        "regret_adapt": [round(float(ra.mean()), 4), round(float(ra.std()), 4)],
        "regret_freeze": [round(float(rf.mean()), 4), round(float(rf.std()), 4)],
        "FA_u_per_seed": [round(float(value), 4) for value in fau],
        "FA_u_max": round(float(fau.max()), 4),
        "better_policy": better,
        "gap_vs_better_ci95": ci_b,
        "gap_vs_worse_ci95": ci_w,
        "gap_vs_adapt": {
            "mean": round(float(gap_vs_adapt.mean()), 4),
            "ci95": ci_a,
        },
        "gap_vs_freeze": {
            "mean": round(float(gap_vs_freeze.mean()), 4),
            "ci95": ci_f,
        },
        "development_beats_both": bool(
            beats_both and fa_ok and not statistical_verdict_withheld
        ),
        "verdict_code": verdict,
        "verdict": verdict,
        "files": [os.path.basename(path) for path in files],
        "file_sha256": {os.path.basename(path): _sha256_file(path) for path in files},
    }
    output["latex_row"] = (
        f"{dataset} ({candidate}) & {len(summaries)} & "
        f"{rk.mean():.4f}$\\pm${rk.std():.4f} & "
        f"{ra.mean():.4f} & {rf.mean():.4f} & {fau.max():.3f} & {verdict} \\\\"
    )
    return output


def _publish_staged_directory(stage, destination):
    """Atomically replace one derived-output directory with the completed stage."""
    stage = Path(stage)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.previous-{uuid.uuid4().hex}")
    had_destination = destination.exists()
    if had_destination:
        os.replace(destination, backup)
    try:
        os.replace(stage, destination)
    except Exception:
        if had_destination and backup.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True, choices=sorted(LOCKED))
    parser.add_argument("--result", nargs="+", required=True, help="exact result paths/globs")
    parser.add_argument("--candidates", nargs="+", default=None)
    parser.add_argument("--expected-seeds", nargs="+", required=True, type=int)
    parser.add_argument(
        "--seed-kind",
        required=True,
        choices=["model", "stream"],
        help="model requires distinct checkpoint hashes; stream is descriptive only",
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--prefer", default="auto", choices=["auto", "sklearn", "numpy"])
    parser.add_argument(
        "--allow-diagnostic-fallback",
        action="store_true",
        help=(
            "allow auto to use the numpy estimator only when sklearn cannot be imported; "
            "such output is diagnostic and never publication-eligible"
        ),
    )
    parser.add_argument(
        "--skip-serialize",
        action="store_true",
        help="removed unsafe mode; retained only to return an actionable error",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.skip_serialize:
        raise LineageError(
            "--skip-serialize is unsafe and no longer supported: it cannot prove that "
            "pre-existing per-condition files belong to the current sources"
        )
    track = args.track
    slug = DATASET_SLUG[track]
    candidates = list(args.candidates or LOCKED[track])
    if len(candidates) != len(set(candidates)):
        raise LineageError("requested candidate list contains duplicates")
    destination = Path(
        args.out_dir
        or REPO / "experiments" / "kbound" / "results" / "multiseed" / track / "extracted"
    ).resolve()
    paths = _expand(args.result)
    if not paths:
        raise LineageError(f"no result files matched: {args.result}")

    records, evidence, sources = _load_records(
        paths,
        track=track,
        verify_checkpoints=args.seed_kind == "model",
    )
    records, seed_metadata, inference = _prepare_records(
        records,
        sources,
        track,
        candidates,
        args.expected_seeds,
        args.seed_kind,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    os.chmod(stage, 0o755)
    try:
        serialize = pcs.serialize_run(
            records,
            dataset=slug,
            out_dir=str(stage),
            seeds=sorted(args.expected_seeds),
            methods=candidates,
            alpha=args.alpha,
            z_names=evidence,
            prefer=args.prefer,
            allow_diagnostic_fallback=args.allow_diagnostic_fallback,
            method_field=METHOD_FIELD[track],
            seed_metadata=seed_metadata,
            extra_top={
                "source_result_files": sources,
                "source_result_sha256": {
                    source["path"]: source["sha256"] for source in sources
                },
                "extract_contract": "leakage_safe_multiseed_v2",
                "claim_scope": inference["claim_scope"],
                "validated_tta_protocol_by_candidate": inference[
                    "tta_protocol_by_candidate"
                ],
            },
        )
        serialize = _bind_serialized_tta_protocols(
            serialize, inference["tta_protocol_by_candidate"]
        )

        aggregates = []
        aggregate_names = []
        aggregate_sha256 = {}
        for candidate in candidates:
            current_files = [
                path
                for path in serialize["written"]
                if _json_load(path).get("method") == candidate
            ]
            aggregate = aggregate_candidate(
                slug,
                candidate,
                current_files,
                expected_seeds=args.expected_seeds,
                inference=inference,
                alpha=args.alpha,
            )
            name = f"multiseed_{slug}_{candidate}.json"
            _atomic_json_dump(aggregate, stage / name)
            aggregates.append(aggregate)
            aggregate_names.append(name)
            aggregate_sha256[name] = _sha256_file(stage / name)

        manifest = {
            "schema": "kbound_natural_multiseed_extract_v2",
            "track": track,
            "dataset": slug,
            "requested_candidates": candidates,
            "expected_seeds": sorted(args.expected_seeds),
            "seed_kind": inference["seed_kind"],
            "inference": inference,
            "validated_tta_protocol_by_candidate": inference[
                "tta_protocol_by_candidate"
            ],
            "sources": sources,
            "serialize": {
                **serialize,
                "written": [os.path.basename(path) for path in serialize["written"]],
                "manifest": os.path.basename(serialize["manifest"]),
                "out_dir": ".",
            },
            "aggregates": aggregate_names,
            "aggregate_sha256": aggregate_sha256,
            "aggregate_verdicts": {
                aggregate["candidate"]: aggregate["verdict"] for aggregate in aggregates
            },
        }
        _atomic_json_dump(manifest, stage / f"extract_manifest_{slug}.json")
        _publish_staged_directory(stage, destination)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    print(json.dumps(manifest, indent=2))
    print(f"published fresh extraction -> {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LineageError as error:
        print(f"LINEAGE ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
