#!/usr/bin/env python3
"""Build a lineage-verified natural-shift forest plot and LaTeX table.

The default scope validates a held-out-confirmation *candidate*. It accepts only
v2 aggregates that declare decisions were locked on a disjoint validation
partition before an unopened test partition was evaluated. The code verifies
bound files, hashes, source-row coverage, calculations, and declarations; local
files alone cannot prove wall-clock chronology or independent preregistration.
A separate ``development-diagnostic`` scope exists for
auditing independent-model-seed validation results; those outputs are visibly
labelled and can never be interpreted as held-out ``beats-both`` evidence.

Legacy aggregates, stream-seed pseudo-replications, opened-test analyses, stale
per-condition files, hash mismatches, and within-test LOO analyses are fatal.
Nothing is silently skipped or reconstructed from point estimates.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FIG_DIR = REPO / "docs" / "research" / "kbound" / "figures"
GREEN, GRAY = "#1b7837", "#888888"

AGGREGATE_SCHEMA = "kbound_natural_multiseed_aggregate_v2"
EXTRACT_SCHEMA = "kbound_natural_multiseed_extract_v2"
LINEAGE_CONTRACT = "leakage_safe_multiseed_v2"
DECISION_LOCK_SCHEMA = "kbound_natural_decision_lock_v3"
LOCKED_BACKEND = "sklearn_gradient_boost_locked_validation_v1"
HELDOUT_SOURCE_SCHEMA = "kbound_natural_locked_holdout_source_v1"
CALIBRATION_SOURCE_SCHEMA = "kbound_natural_calibration_source_v1"
LOCKED_ESTIMATOR_SCHEMA = "kbound_natural_locked_estimator_v1"
LOCKED_PROTOCOL_SCHEMA = "kbound_natural_locked_protocol_v1"
PREOPENING_RECEIPT_SCHEMA = "kbound_natural_preopening_receipt_v1"
HELDOUT_PARTITIONS = {"test", "target_test", "id_test", "heldout", "held_out"}
DEVELOPMENT_PARTITIONS = {"val", "target_val", "id_val", "source", "source_val", "dev"}
CURRENT_SOURCE_SCHEMAS = {
    "kbound_officehome_v4",
    "kbound_wilds_iwildcam_finder_v0.5",
    "kbound_rxrx1_v0.6",
    "kbound_wilds_camelyon17_v0.8",
    "kbound_imagenetr_v0.7",
}

# A held-out scorer must have a reviewed dataset/metric pairing. Other natural
# tracks remain development-only until an official split-level scorer contract
# is added here and covered by adversarial tests.
HELDOUT_METRIC_CONTRACTS = {
    "officehome": {
        "name": "accuracy",
        "official": True,
        "direction": "higher_is_better",
        "unit": "per_condition",
        "range": [0.0, 1.0],
    },
}

DISPLAY = {
    "officehome": "Office-Home",
    "iwildcam": "iWildCam",
    "rxrx1": "RxRx1",
    "camelyon17": "Camelyon17",
}


class LineageError(ValueError):
    """An aggregate cannot support the requested scientific output scope."""


def _reject_json_constant(value):
    raise LineageError(f"non-standard JSON constant {value!r} is forbidden")


def _strict_json_load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise LineageError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise LineageError(f"JSON root must be an object: {path}")
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


def _checkpoint_tensor_sha256(path):
    """Hash the effective model tensor state, not its serialization container."""
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise LineageError(
            "model-seed forest validation requires PyTorch to inspect checkpoint tensors"
        ) from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise LineageError(f"checkpoint is not a safely loadable PyTorch artifact: {path}") from exc
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


def _sha256_json(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json_dump(payload, path):
    # separators/defaults remain human-readable; allow_nan=False enforces RFC JSON.
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    _atomic_text(text, path)


def _normalise_partition(value):
    if value is None:
        return None
    return str(value).strip().lower().replace("-", "_")


def _safe_child(directory, name):
    if not name or Path(name).name != name:
        raise LineageError(f"lineage file must be a safe basename, got {name!r}")
    return directory / name


def _expected_metric_contract(dataset):
    contract = HELDOUT_METRIC_CONTRACTS.get(dataset)
    if contract is None:
        raise LineageError(
            f"no reviewed held-out scorer/metric contract exists for dataset {dataset!r}"
        )
    # Return a JSON round-trip copy so callers cannot mutate the global lock.
    return json.loads(json.dumps(contract))


def _parse_utc_timestamp(value, field):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise LineageError(f"{field} must be an RFC3339 UTC timestamp ending in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise LineageError(f"{field} is not a valid RFC3339 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise LineageError(f"{field} must be UTC")
    return parsed


def _load_bound_artifact(parent, reference, *, field, schema):
    display = field.replace("_", "-")
    if not isinstance(reference, dict):
        raise LineageError(f"decision lock lacks a bound {display} artifact")
    path = _safe_child(parent, reference.get("file"))
    expected_sha = _require_hex_sha256(reference.get("sha256"), f"{field}.sha256")
    if not path.is_file() or _sha256_file(path) != expected_sha:
        raise LineageError(f"decision-lock {display} artifact/hash mismatch: {path}")
    document = _strict_json_load(path)
    if document.get("schema") != schema:
        raise LineageError(f"decision-lock {display} schema is unknown: {path}")
    return path, expected_sha, document


def _finite_triplet(metric, value):
    if not isinstance(value, dict):
        raise LineageError(f"{metric} must be an object with mean/ci95")
    ci = value.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise LineageError(f"{metric}.ci95 must contain two finite endpoints")
    try:
        mean, lo, hi = float(value["mean"]), float(ci[0]), float(ci[1])
    except (KeyError, TypeError, ValueError) as exc:
        raise LineageError(f"{metric} contains non-numeric values") from exc
    if not all(math.isfinite(number) for number in (mean, lo, hi)):
        raise LineageError(f"{metric} contains non-finite values")
    if lo > hi or mean < lo - 1e-12 or mean > hi + 1e-12:
        raise LineageError(f"{metric} interval/order is invalid: mean={mean}, ci95={ci}")
    return mean, lo, hi


def _finite_pair(metric, value):
    if not isinstance(value, list) or len(value) != 2:
        raise LineageError(f"{metric} must contain [mean, standard_deviation]")
    try:
        pair = tuple(float(number) for number in value)
    except (TypeError, ValueError) as exc:
        raise LineageError(f"{metric} contains non-numeric values") from exc
    if not all(math.isfinite(number) and number >= 0 for number in pair):
        raise LineageError(f"{metric} must contain finite non-negative values")
    return pair


def _close(actual, expected, *, atol=5.1e-5):
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol)


def _bootstrap_ci(values, *, replicates, seed):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise LineageError("bootstrap input must contain at least two finite seed values")
    if not isinstance(replicates, int) or replicates < 1000:
        raise LineageError("bootstrap_replicates must be an integer >= 1000")
    if not isinstance(seed, int):
        raise LineageError("bootstrap_seed must be an integer")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        draws[index] = values[rng.integers(0, len(values), len(values))].mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def _require_hex_sha256(value, field):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise LineageError(f"{field} must be a lowercase SHA-256 digest")
    return value.lower()


def _validate_tta_protocol(protocol, *, candidate, context):
    if not isinstance(protocol, dict):
        raise LineageError(f"{context} lacks a validated candidate tta_protocol")
    mode = protocol.get("mode")
    if mode in {"online", "episodic"}:
        gradient_reads_eval = mode == "episodic"
        expected = {
            "schema": "kbound_tta_candidate_protocol_v1",
            "mode": mode,
            "semantics": (
                "episodic_transductive_eval_batch_update_and_evaluation"
                if gradient_reads_eval
                else "online_disjoint_stream_update_then_transductive_bn_evaluation"
            ),
            "requires_auxiliary_stream_eval_disjoint": True,
            "gradient_update_reads_eval_x": gradient_reads_eval,
            "prediction_uses_eval_batch_statistics": True,
            "candidate_evaluation_is_transductive": True,
            "candidate_adaptation_eval_disjoint": False,
            "target_labels_used_for_adaptation_or_prediction": False,
        }
    elif mode == "inference_only_stream_prior" and candidate in {
        "labelshift",
        "conservative",
    }:
        expected = {
            "schema": "kbound_tta_candidate_protocol_v1",
            "mode": mode,
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
    else:
        raise LineageError(f"{context} has an unrecognized candidate tta_protocol mode")
    if protocol != expected:
        raise LineageError(f"{context} candidate tta_protocol is missing, altered, or contradictory")
    return expected


def _validate_serialization_generation(manifest, manifest_path, expected_protocols):
    reference = manifest.get("serialize")
    if not isinstance(reference, dict):
        raise LineageError(f"held-out extraction lacks a committed serializer generation: {manifest_path}")
    commit_path = _safe_child(manifest_path.parent, reference.get("manifest"))
    commit = _strict_json_load(commit_path)
    if (
        commit.get("schema") != "kbound_per_condition_generation_v1"
        or commit.get("generation_committed") is not True
        or commit.get("generation_id") != reference.get("generation_id")
    ):
        raise LineageError(f"per-condition serializer generation is not committed: {commit_path}")
    generation_id = _require_hex_sha256(
        commit.get("generation_id"), "per-condition generation_id"
    )
    if commit.get("validated_tta_protocol_by_candidate") != expected_protocols:
        raise LineageError(
            f"serializer commit does not preserve validated candidate tta_protocols: {commit_path}"
        )
    files = commit.get("files")
    if not isinstance(files, dict) or not files:
        raise LineageError(f"serializer commit has no file inventory: {commit_path}")
    referenced_files = {Path(value).name for value in reference.get("written", [])}
    if referenced_files != set(files):
        raise LineageError(f"serializer/extraction file inventories differ: {commit_path}")
    for name, descriptor in files.items():
        path = _safe_child(manifest_path.parent, name)
        if (
            not isinstance(descriptor, dict)
            or not path.is_file()
            or _sha256_file(path) != descriptor.get("sha256")
        ):
            raise LineageError(f"per-condition hash mismatch in serializer commit: {path}")
        document = _strict_json_load(path)
        if document.get("serialization_generation_id") != generation_id:
            raise LineageError(f"serialized row file is not bound to its generation: {path}")
        candidate = document.get("method")
        protocol = expected_protocols.get(candidate)
        if (
            protocol is None
            or document.get("tta_protocol") != protocol
            or document.get("tta_protocol_sha256") != _sha256_json(protocol)
        ):
            raise LineageError(f"serialized file does not preserve candidate tta_protocol: {path}")


def _route_from_record(record, path, index):
    required = (
        "condition", "B", "a0", "a_adapted", "a_kbound", "a_oracle",
        "b_hat", "eps_conformal", "kga_decision", "calibration_feasible",
        "radius_status", "source_file_sha256", "source_record_index",
        "source_record_sha256",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise LineageError(f"record {index} lacks required scoring fields {missing}: {path}")
    condition = str(record["condition"])
    if not condition:
        raise LineageError(f"record {index} has an empty condition: {path}")
    try:
        benefit = float(record["B"])
        frozen = float(record["a0"])
        adapted = float(record["a_adapted"])
        routed = float(record["a_kbound"])
        oracle = float(record["a_oracle"])
        bhat = float(record["b_hat"])
    except (TypeError, ValueError) as exc:
        raise LineageError(f"record {index} has non-numeric scoring fields: {path}") from exc
    if not all(math.isfinite(value) for value in (benefit, frozen, adapted, routed, oracle, bhat)):
        raise LineageError(f"record {index} has non-finite scoring fields: {path}")
    if any(value < 0.0 or value > 1.0 for value in (frozen, adapted, routed, oracle)):
        raise LineageError(f"record {index} has a score outside the reviewed metric range [0,1]: {path}")
    if benefit < -1.0 or benefit > 1.0:
        raise LineageError(f"record {index} has benefit outside [-1,1]: {path}")
    if not _close(benefit, adapted - frozen, atol=1e-10):
        raise LineageError(f"record {index} violates B=a_adapted-a0: {path}")
    if not _close(oracle, max(frozen, adapted), atol=1e-10):
        raise LineageError(f"record {index} has inconsistent oracle accuracy: {path}")

    decision = str(record["kga_decision"]).upper()
    if decision not in {"ADAPT", "FREEZE", "ABSTAIN"}:
        raise LineageError(f"record {index} has an invalid KGA decision: {path}")
    feasible = record["calibration_feasible"]
    if not isinstance(feasible, bool):
        raise LineageError(f"record {index} lacks explicit calibration feasibility: {path}")
    if feasible:
        try:
            epsilon = float(record["eps_conformal"])
        except (TypeError, ValueError) as exc:
            raise LineageError(f"record {index} has a non-numeric radius: {path}") from exc
        if not math.isfinite(epsilon) or epsilon < 0 or record["radius_status"] != "FINITE":
            raise LineageError(f"record {index} has an invalid finite-radius contract: {path}")
        expected_decision = (
            "ADAPT" if bhat - epsilon > 0 else
            "FREEZE" if bhat + epsilon < 0 else
            "ABSTAIN"
        )
        if decision != expected_decision:
            raise LineageError(f"record {index} decision disagrees with b_hat/radius: {path}")
    else:
        epsilon = None
        if record["eps_conformal"] is not None or record["radius_status"] != "INFEASIBLE":
            raise LineageError(f"record {index} violates infeasible-radius contract: {path}")
        if decision != "ABSTAIN":
            raise LineageError(f"record {index} must abstain when calibration is infeasible: {path}")
    expected_routed = adapted if decision == "ADAPT" else frozen
    if not _close(routed, expected_routed, atol=1e-10):
        raise LineageError(f"record {index} has inconsistent routed accuracy: {path}")
    return {
        "condition": condition,
        "B": benefit,
        "a0": frozen,
        "a_adapted": adapted,
        "a_kbound": routed,
        "a_oracle": oracle,
        "b_hat": bhat,
        "eps_conformal": epsilon,
        "kga_decision": decision,
        "calibration_feasible": feasible,
        "source_file_sha256": _require_hex_sha256(
            record["source_file_sha256"], "source_file_sha256"
        ),
        "source_record_index": int(record["source_record_index"]),
        "source_record_sha256": _require_hex_sha256(
            record["source_record_sha256"], "source_record_sha256"
        ),
    }


def _source_condition_key(dataset, record):
    mode = str(record.get("mode", ""))
    if dataset == "officehome":
        base = (
            f"{record['domain']}|{record.get('split') or 'test'}|"
            f"{record['comp']}|{record['regime']}"
        )
        return base
    elif dataset == "iwildcam":
        location = record.get("location", record.get("domain", "loc"))
        base = f"{location}|{record['comp']}|{record['regime']}|{record.get('aggr', '')}"
    elif dataset == "rxrx1":
        base = (
            f"{record.get('domain', 'rxrx1')}|{record['comp']}|"
            f"{record['regime']}|{record.get('aggr', '')}"
        )
    elif dataset == "camelyon17":
        base = f"{record['domain']}|{record['comp']}|{record['regime']}|{record['aggr']}"
    else:
        raise LineageError(f"no source condition-key contract for dataset {dataset!r}")
    return f"{base}|{mode}" if mode else base


def _source_document_candidate_ready(document, dataset):
    if document.get("schema") != HELDOUT_SOURCE_SCHEMA:
        return False
    if document.get("dataset") != dataset:
        return False
    try:
        expected_metric = _expected_metric_contract(dataset)
    except LineageError:
        return False
    if document.get("metric_contract") != expected_metric:
        return False
    if document.get("publication_eligible") is not False:
        return False
    if document.get("computational_candidate_ready") is not True:
        return False
    ledger = (
        document.get("completion_ledger")
        or document.get("run_ledger")
        or document.get("ledger")
    )
    if not isinstance(ledger, dict):
        return False
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
        counts_complete = (
            int(expected) > 0
            and int(expected) == int(completed)
            and int(failed) == 0
            and int(pending) == 0
        )
    except (TypeError, ValueError):
        return False
    failure_lists = (
        ledger.get("failed_cell_ids", []),
        ledger.get("pending_keys", []),
        ledger.get("missing_cell_ids", []),
        ledger.get("failure_history", []),
        document.get("failures", []),
    )
    conditions = document.get("conditions")
    records = document.get("records")
    conditions_complete = (
        isinstance(conditions, list)
        and isinstance(records, list)
        and len(conditions) == len(records) == int(expected)
    )
    return bool(
        status == "COMPLETE"
        and counts_complete
        and conditions_complete
        and ledger.get("execution_complete", True) is True
        and all(isinstance(rows, list) and not rows for rows in failure_lists)
    )


def _validate_heldout_source_structure(document, path, dataset, expected_metric):
    """Validate the dedicated scorer schema and its condition/record ledger."""
    if document.get("schema") != HELDOUT_SOURCE_SCHEMA:
        raise LineageError(f"held-out source lacks dedicated locked-scorer schema: {path}")
    if document.get("dataset") != dataset:
        raise LineageError(f"held-out source dataset mismatch: {path}")
    if document.get("metric_contract") != expected_metric:
        raise LineageError(f"held-out source metric contract mismatch: {path}")
    if (
        document.get("publication_eligible") is not False
        or document.get("computational_candidate_ready") is not True
    ):
        raise LineageError(
            f"held-out source must be computationally complete but publication-ineligible: {path}"
        )
    partition = _normalise_partition(document.get("evaluation_partition"))
    if partition not in HELDOUT_PARTITIONS:
        raise LineageError(f"held-out source has a non-held-out evaluation partition: {path}")
    records = document.get("records")
    conditions = document.get("conditions")
    if not isinstance(records, list) or not records:
        raise LineageError(f"held-out source records[] are missing: {path}")
    if not isinstance(conditions, list) or len(conditions) != len(records):
        raise LineageError(f"held-out conditions[] must exactly cover records[]: {path}")
    ledger = document.get("completion_ledger")
    if not isinstance(ledger, dict):
        raise LineageError(f"held-out source completion ledger is missing: {path}")
    expected = ledger.get("expected_cells", ledger.get("expected"))
    completed = ledger.get("completed_cells", ledger.get("completed"))
    failed = ledger.get("failed_cells", ledger.get("failed", 0))
    pending = ledger.get("missing_cells", ledger.get("pending", 0))
    if isinstance(failed, list):
        failed = len(failed)
    if isinstance(pending, list):
        pending = len(pending)
    try:
        complete = (
            str(ledger.get("status", "")).upper() == "COMPLETE"
            and ledger.get("execution_complete") is True
            and int(expected) == int(completed) == len(records)
            and int(failed) == 0
            and int(pending) == 0
        )
    except (TypeError, ValueError):
        complete = False
    if not complete:
        raise LineageError(f"held-out source completion ledger is inconsistent: {path}")
    seen = set()
    for entry in conditions:
        if not isinstance(entry, dict):
            raise LineageError(f"held-out condition ledger entry is not an object: {path}")
        try:
            index = int(entry["record_index"])
            raw = records[index]
            raw_seed = int(raw["model_seed"])
            entry_seed = int(entry["model_seed"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LineageError(f"held-out condition ledger has an invalid record index: {path}") from exc
        if index < 0 or index >= len(records) or index in seen:
            raise LineageError(f"held-out condition ledger record coverage is not one-to-one: {path}")
        seen.add(index)
        raw_candidate = raw.get("candidate", raw.get("method"))
        try:
            raw_partition = _normalise_partition(raw["split"])
            raw_condition = _source_condition_key(dataset, raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise LineageError(f"held-out source record cannot form a condition key: {path}#{index}") from exc
        if raw_partition != partition:
            raise LineageError(
                f"held-out source record partition disagrees with its document envelope: {path}#{index}"
            )
        if (
            str(entry.get("status", "")).upper() != "COMPLETE"
            or entry.get("candidate") != raw_candidate
            or entry_seed != raw_seed
            or entry.get("condition") != raw_condition
        ):
            raise LineageError(f"held-out conditions[] disagrees with records[]: {path}#{index}")
    if seen != set(range(len(records))):
        raise LineageError(f"held-out conditions[] does not cover every source record: {path}")


def _validate_source_hashes(
    manifest,
    manifest_path,
    *,
    dataset,
    expected_metric=None,
    expected_seeds=None,
    require_heldout_candidate=False,
):
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise LineageError(f"manifest has no source lineage: {manifest_path}")
    partitions = set()
    source_records = {}
    verified_checkpoints = {}
    for source in sources:
        if not isinstance(source, dict):
            raise LineageError(f"invalid source lineage entry: {manifest_path}")
        source_path = source.get("path")
        expected = source.get("sha256")
        if not source_path or not expected:
            raise LineageError(f"source path/hash missing: {manifest_path}")
        path = Path(source_path)
        if not path.is_file():
            raise LineageError(f"source file is unavailable for hash verification: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise LineageError(f"source hash mismatch: {path}")
        source_document = _strict_json_load(path)
        candidate_ready = _source_document_candidate_ready(source_document, dataset)
        publication = source.get("publication") or {}
        if publication.get("publication_ready") is not False:
            raise LineageError(f"source must remain publication-ineligible without external audit: {path}")
        declared_candidate = publication.get("computational_candidate_ready") is True
        if declared_candidate != candidate_ready:
            raise LineageError(f"source computational-candidate ledger metadata mismatch: {path}")
        if require_heldout_candidate and not candidate_ready:
            raise LineageError(
                f"held-out candidate requires publication_eligible=false, the dedicated locked "
                f"scorer/reviewed metric, and a complete failure-free ledger: {path}"
            )
        if require_heldout_candidate:
            _validate_heldout_source_structure(
                source_document,
                path,
                dataset,
                expected_metric,
            )
        partition = _normalise_partition(source.get("partition"))
        if not partition:
            raise LineageError(f"source partition is missing: {path}")
        if require_heldout_candidate and partition != _normalise_partition(
            source_document.get("evaluation_partition")
        ):
            raise LineageError(f"source manifest/document partition mismatch: {path}")
        partitions.add(partition)
        records = source_document.get("records")
        if not isinstance(records, list) or not records:
            raise LineageError(f"hashed source contains no records[] for row joining: {path}")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise LineageError(f"source record {index} is not an object: {path}")
            key = (actual, index)
            if key in source_records:
                raise LineageError(f"duplicate source record identity: {key}")
            source_records[key] = {
                "record": record,
                "sha256": _sha256_json(record),
                "path": str(path),
                "document": source_document,
            }
        if require_heldout_candidate:
            checkpoint_refs = source_document.get("checkpoint_artifacts_by_seed")
            if not isinstance(checkpoint_refs, dict) or not checkpoint_refs:
                raise LineageError(f"held-out source checkpoint artifact inventory is missing: {path}")
            for seed_text, reference in checkpoint_refs.items():
                try:
                    seed = int(seed_text)
                except (TypeError, ValueError) as exc:
                    raise LineageError(f"checkpoint inventory contains a non-integer seed: {path}") from exc
                reference = reference or {}
                try:
                    reference_seed = int(reference.get("model_seed"))
                except (TypeError, ValueError) as exc:
                    raise LineageError(
                        f"checkpoint inventory lacks a model-seed binding for seed {seed}: {path}"
                    ) from exc
                if reference_seed != seed:
                    raise LineageError(
                        f"checkpoint inventory model-seed binding mismatch for seed {seed}: {path}"
                    )
                checkpoint_path = _safe_child(path.parent, reference.get("file"))
                checkpoint_sha = _require_hex_sha256(
                    reference.get("sha256"),
                    f"checkpoint_artifacts_by_seed[{seed}]",
                )
                tensor_sha = _require_hex_sha256(
                    reference.get("tensor_sha256"),
                    f"checkpoint_artifacts_by_seed[{seed}].tensor_sha256",
                )
                if not checkpoint_path.is_file() or _sha256_file(checkpoint_path) != checkpoint_sha:
                    raise LineageError(f"checkpoint artifact/hash mismatch for seed {seed}: {checkpoint_path}")
                actual_tensor_sha = _checkpoint_tensor_sha256(checkpoint_path)
                if actual_tensor_sha != tensor_sha:
                    raise LineageError(
                        f"checkpoint tensor-state hash mismatch for seed {seed}: {checkpoint_path}"
                    )
                identity = {
                    "model_seed": seed,
                    "sha256": checkpoint_sha,
                    "tensor_sha256": tensor_sha,
                }
                prior = verified_checkpoints.setdefault(seed, identity)
                if prior != identity:
                    raise LineageError(f"conflicting checkpoint artifacts for seed {seed}")
    if require_heldout_candidate:
        expected_seed_set = {int(seed) for seed in (expected_seeds or [])}
        if set(verified_checkpoints) != expected_seed_set:
            raise LineageError("checkpoint artifact inventory does not match expected model seeds")
        tensor_hashes = [
            verified_checkpoints[seed]["tensor_sha256"] for seed in sorted(verified_checkpoints)
        ]
        if len(set(tensor_hashes)) != len(tensor_hashes):
            raise LineageError(
                "checkpoint tensor-state hashes must be unique by model seed; byte-different "
                "containers of the same model are not independent replications"
            )
        for source in source_records.values():
            raw = source["record"]
            try:
                seed = int(raw["model_seed"])
            except (KeyError, TypeError, ValueError) as exc:
                raise LineageError(
                    f"held-out source row lacks model_seed: {source['path']}"
                ) from exc
            identity = verified_checkpoints.get(seed)
            if identity is None:
                raise LineageError(
                    f"held-out source row references an undeclared model seed {seed}: {source['path']}"
                )
            if (
                raw.get("checkpoint_sha256") != identity["sha256"]
                or raw.get("checkpoint_tensor_sha256") != identity["tensor_sha256"]
            ):
                raise LineageError(
                    "held-out source row is not bound to its model-seed checkpoint/tensor identity: "
                    f"{source['path']}"
                )
            raw_candidate = raw.get("candidate", raw.get("method"))
            _validate_tta_protocol(
                raw.get("tta_protocol"),
                candidate=raw_candidate,
                context=f"held-out source row {source['path']}",
            )
    return partitions, source_records, verified_checkpoints


def _expected_candidate_source_keys(source_records, *, dataset, candidate, seeds):
    """Return the exact source rows this aggregate must score, with no cherry-picking."""
    expected_seeds = {int(seed) for seed in seeds}
    keys = set()
    conditions_by_seed = {seed: set() for seed in expected_seeds}
    for key, source in source_records.items():
        raw = source["record"]
        raw_candidate = raw.get("candidate", raw.get("method"))
        if raw_candidate != candidate:
            raise LineageError(
                f"held-out source contains an unregistered competing candidate {raw_candidate!r}: "
                f"{source['path']}#{key[1]}"
            )
        try:
            seed = int(raw["model_seed"])
            condition = _source_condition_key(dataset, raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise LineageError(
                f"candidate source row lacks model-seed/condition identity: {source['path']}#{key[1]}"
            ) from exc
        if seed not in expected_seeds:
            raise LineageError(
                f"candidate source contains an undeclared model seed {seed}: {source['path']}#{key[1]}"
            )
        if condition in conditions_by_seed[seed]:
            raise LineageError(f"duplicate candidate condition for model seed {seed}: {condition}")
        conditions_by_seed[seed].add(condition)
        keys.add(key)
    if not keys or any(not values for values in conditions_by_seed.values()):
        raise LineageError("source has no complete candidate rows for every expected model seed")
    inventories = list(conditions_by_seed.values())
    if any(values != inventories[0] for values in inventories[1:]):
        raise LineageError("source condition inventory differs across model seeds")
    return keys


def _validate_per_condition_files(
    aggregate,
    aggregate_path,
    *,
    source_records,
    expected_source_keys=None,
    expected_metric=None,
    verified_checkpoints=None,
    locked_routes=None,
):
    directory = aggregate_path.parent
    files = aggregate.get("files")
    hashes = aggregate.get("file_sha256")
    if not isinstance(files, list) or not files or not isinstance(hashes, dict):
        raise LineageError(f"per-condition file/hash lineage missing: {aggregate_path}")
    if set(files) != set(hashes):
        raise LineageError(f"per-condition file/hash inventory mismatch: {aggregate_path}")
    seeds = set()
    checkpoint_by_seed = {}
    checkpoint_tensor_by_seed = {}
    infeasible_total = 0
    backends = set()
    per_seed = []
    reference_conditions = None
    observed_locked_keys = set()
    observed_source_keys = set()
    for name in files:
        path = _safe_child(directory, name)
        if not path.is_file():
            raise LineageError(f"per-condition lineage file is missing: {path}")
        if _sha256_file(path) != hashes[name]:
            raise LineageError(f"per-condition hash mismatch: {path}")
        document = _strict_json_load(path)
        if document.get("extract_contract") != LINEAGE_CONTRACT:
            raise LineageError(f"per-condition file lacks hardened contract: {path}")
        if document.get("benchmark") != aggregate.get("dataset"):
            raise LineageError(f"per-condition dataset mismatch: {path}")
        if document.get("method") != aggregate.get("candidate"):
            raise LineageError(f"per-condition candidate mismatch: {path}")
        aggregate_protocol = aggregate.get("tta_protocol")
        if document.get("tta_protocol") != aggregate_protocol:
            raise LineageError(f"per-condition candidate tta_protocol mismatch: {path}")
        if document.get("tta_protocol_sha256") != _sha256_json(aggregate_protocol):
            raise LineageError(f"per-condition candidate tta_protocol hash mismatch: {path}")
        if document.get("seed_kind") != aggregate.get("seed_kind"):
            raise LineageError(f"per-condition seed-kind mismatch: {path}")
        if _normalise_partition(document.get("evaluation_partition")) != _normalise_partition(
            aggregate.get("evaluation_partition")
        ):
            raise LineageError(f"per-condition partition mismatch: {path}")
        if expected_metric is not None and document.get("metric_contract") != expected_metric:
            raise LineageError(f"per-condition metric contract mismatch: {path}")
        if not isinstance(document.get("records"), list) or not document["records"]:
            raise LineageError(f"per-condition records are missing: {path}")
        parsed = [
            _route_from_record(record, path, index)
            for index, record in enumerate(document["records"])
        ]
        for record_index, record in enumerate(document["records"]):
            if (
                record.get("tta_protocol") != aggregate_protocol
                or record.get("tta_protocol_sha256") != _sha256_json(aggregate_protocol)
            ):
                raise LineageError(
                    f"per-condition row {record_index} does not preserve candidate tta_protocol: {path}"
                )
        conditions = [record["condition"] for record in parsed]
        if len(conditions) != len(set(conditions)):
            raise LineageError(f"duplicate scientific conditions in {path}")
        if reference_conditions is None:
            reference_conditions = conditions
        elif conditions != reference_conditions:
            raise LineageError(f"condition order/set differs across model seeds: {path}")
        feasibility = [record["calibration_feasible"] for record in parsed]
        if any(not isinstance(value, bool) for value in feasibility):
            raise LineageError(f"per-condition calibration feasibility is missing: {path}")
        file_infeasible = sum(not value for value in feasibility)
        if document.get("n_calibration_infeasible") != file_infeasible:
            raise LineageError(f"per-condition infeasible-radius count mismatch: {path}")
        infeasible_total += file_infeasible
        if document.get("kga_backend"):
            backends.add(document["kga_backend"])
        locked_backend = document.get("kga_backend") == LOCKED_BACKEND
        if document.get("estimator_publication_eligible") is not False:
            raise LineageError(f"local estimator artifact cannot be publication-eligible: {path}")
        if locked_routes is not None and document.get("estimator_computationally_locked") is not True:
            raise LineageError(f"held-out candidate lacks a computationally locked estimator: {path}")
        if locked_routes is None and document.get("estimator_computationally_locked", False) is not False:
            raise LineageError(f"development estimator is mislabeled as externally locked: {path}")
        if locked_routes is not None and not locked_backend:
            raise LineageError(f"held-out candidate uses an unlocked estimator backend: {path}")
        if _normalise_partition(aggregate.get("evaluation_partition")) in HELDOUT_PARTITIONS:
            expected_lock_hash = (aggregate.get("decision_lock") or {}).get("sha256")
            if document.get("decision_lock_sha256") != expected_lock_hash:
                raise LineageError(f"per-condition decision-lock lineage mismatch: {path}")
        seed = int(document.get("seed"))
        if seed in seeds:
            raise LineageError(f"duplicate per-condition seed {seed}: {aggregate_path}")
        seeds.add(seed)
        if aggregate.get("seed_kind") == "model_seed":
            try:
                document_model_seed = int(document["model_seed"])
            except (KeyError, TypeError, ValueError) as exc:
                raise LineageError(f"per-condition model_seed is missing: {path}") from exc
            if document_model_seed != seed:
                raise LineageError(f"per-condition seed/model_seed binding mismatch: {path}")
        for record_index, record in enumerate(parsed):
            source_key = (record["source_file_sha256"], record["source_record_index"])
            source = source_records.get(source_key)
            if source is None:
                raise LineageError(f"scored row has no exact hashed source record: {source_key}")
            if source_key in observed_source_keys:
                raise LineageError(f"one source record is scored more than once: {source_key}")
            observed_source_keys.add(source_key)
            if record["source_record_sha256"] != source["sha256"]:
                raise LineageError(f"source-record hash mismatch: {source['path']}#{source_key[1]}")
            raw = source["record"]
            raw_candidate = raw.get("candidate", raw.get("method"))
            if raw_candidate != aggregate.get("candidate"):
                raise LineageError(f"source/scored candidate mismatch: {source['path']}#{source_key[1]}")
            if raw.get("tta_protocol") != aggregate_protocol:
                raise LineageError(
                    f"source/scored candidate tta_protocol mismatch: {source['path']}#{source_key[1]}"
                )
            try:
                raw_condition = _source_condition_key(aggregate.get("dataset"), raw)
                raw_values = (float(raw["a0"]), float(raw["aa"]), float(raw["B"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise LineageError(
                    f"source record lacks exact condition/a0/aa/B fields: {source['path']}#{source_key[1]}"
                ) from exc
            if raw_condition != record["condition"]:
                raise LineageError(f"source/scored condition mismatch: {source['path']}#{source_key[1]}")
            for field, actual, expected in zip(
                ("a0", "a_adapted", "B"),
                (record["a0"], record["a_adapted"], record["B"]),
                raw_values,
            ):
                if not _close(actual, expected, atol=1e-12):
                    raise LineageError(
                        f"scored {field} differs from hashed source record: {source['path']}#{source_key[1]}"
                    )
            if aggregate.get("seed_kind") == "model_seed":
                raw_model_seed = raw.get("model_seed", source["document"].get("model_seed"))
                if raw_model_seed is None or int(raw_model_seed) != seed:
                    raise LineageError(
                        f"source/scored model-seed mismatch: {source['path']}#{source_key[1]}"
                    )
        checkpoint = document.get("checkpoint_sha256")
        checkpoint_tensor = document.get("checkpoint_tensor_sha256")
        if aggregate.get("seed_kind") == "model_seed":
            checkpoint_by_seed[str(seed)] = _require_hex_sha256(
                checkpoint, f"per-condition checkpoint_sha256 seed {seed}"
            )
            checkpoint_tensor_by_seed[str(seed)] = _require_hex_sha256(
                checkpoint_tensor, f"per-condition checkpoint_tensor_sha256 seed {seed}"
            )
        if locked_routes is not None:
            for record in parsed:
                key = (seed, record["condition"])
                locked = locked_routes.get(key)
                if locked is None:
                    raise LineageError(f"scored route is absent from decision lock: {key}")
                observed_locked_keys.add(key)
                for field in ("kga_decision", "b_hat"):
                    if record[field] != locked[field]:
                        if field == "b_hat" and _close(record[field], locked[field], atol=1e-12):
                            continue
                        raise LineageError(f"scored {field} differs from decision lock: {key}")
                if record["eps_conformal"] != locked["eps_conformal"]:
                    if not (
                        record["eps_conformal"] is not None
                        and locked["eps_conformal"] is not None
                        and _close(record["eps_conformal"], locked["eps_conformal"], atol=1e-12)
                    ):
                        raise LineageError(f"scored radius differs from decision lock: {key}")
        oracle = np.array([record["a_oracle"] for record in parsed], dtype=float)
        routed = np.array([record["a_kbound"] for record in parsed], dtype=float)
        adapted = np.array([record["a_adapted"] for record in parsed], dtype=float)
        frozen = np.array([record["a0"] for record in parsed], dtype=float)
        false_adapt = np.array(
            [
                record["kga_decision"] == "ADAPT" and record["B"] <= 0
                for record in parsed
            ],
            dtype=bool,
        )
        per_seed.append(
            {
                "seed": seed,
                "regret_kga": float(np.mean(oracle - routed)),
                "regret_adapt": float(np.mean(oracle - adapted)),
                "regret_freeze": float(np.mean(oracle - frozen)),
                "fa_u": float(np.mean(false_adapt)),
            }
        )
    if seeds != {int(seed) for seed in aggregate.get("seeds", [])}:
        raise LineageError(f"per-condition seed inventory mismatch: {aggregate_path}")
    if expected_source_keys is not None and observed_source_keys != set(expected_source_keys):
        missing = sorted(set(expected_source_keys) - observed_source_keys)
        extra = sorted(observed_source_keys - set(expected_source_keys))
        raise LineageError(
            "scored rows do not exactly cover the expected candidate/model-seed source rows "
            f"(missing={missing[:3]}, extra={extra[:3]}): {aggregate_path}"
        )
    declared_checkpoints = aggregate.get("checkpoint_sha256_by_seed") or {}
    if checkpoint_by_seed != declared_checkpoints:
        raise LineageError(f"checkpoint lineage mismatch: {aggregate_path}")
    declared_tensor_checkpoints = aggregate.get("checkpoint_tensor_sha256_by_seed") or {}
    if checkpoint_tensor_by_seed != declared_tensor_checkpoints:
        raise LineageError(f"checkpoint tensor-state lineage mismatch: {aggregate_path}")
    if verified_checkpoints is not None:
        expected_checkpoints = {
            str(seed): identity["sha256"]
            for seed, identity in sorted(verified_checkpoints.items())
        }
        if declared_checkpoints != expected_checkpoints:
            raise LineageError(
                f"checkpoint hashes are not bound to verified source artifacts: {aggregate_path}"
            )
        expected_tensor_checkpoints = {
            str(seed): identity["tensor_sha256"]
            for seed, identity in sorted(verified_checkpoints.items())
        }
        if declared_tensor_checkpoints != expected_tensor_checkpoints:
            raise LineageError(
                f"checkpoint tensor hashes are not bound to verified source artifacts: {aggregate_path}"
            )
    if aggregate.get("n_calibration_infeasible_total") != infeasible_total:
        raise LineageError(f"aggregate infeasible-radius count mismatch: {aggregate_path}")
    if aggregate.get("calibration_feasible_all") is not (infeasible_total == 0):
        raise LineageError(f"aggregate calibration-feasibility flag mismatch: {aggregate_path}")
    if sorted(backends) != aggregate.get("kga_backend"):
        raise LineageError(f"aggregate estimator-backend lineage mismatch: {aggregate_path}")
    if aggregate.get("estimator_publication_eligible") is not False:
        raise LineageError(f"aggregate estimator must remain publication-ineligible: {aggregate_path}")
    expected_computational_lock = locked_routes is not None and sorted(backends) == [LOCKED_BACKEND]
    if aggregate.get("estimator_computationally_locked", False) is not expected_computational_lock:
        raise LineageError(f"aggregate computational estimator-lock status mismatch: {aggregate_path}")
    if locked_routes is not None and observed_locked_keys != set(locked_routes):
        missing = sorted(set(locked_routes) - observed_locked_keys)
        raise LineageError(f"decision lock contains unscored or missing routes: {missing[:3]}")
    per_seed.sort(key=lambda row: row["seed"])
    return per_seed


def _parse_locked_routes(routes, *, path, seeds):
    if not isinstance(routes, list) or not routes:
        raise LineageError(f"locked-route inventory is missing: {path}")
    parsed = {}
    for index, route in enumerate(routes):
        if not isinstance(route, dict):
            raise LineageError(f"locked route {index} is not an object: {path}")
        try:
            seed = int(route["seed"])
            condition = str(route["condition"])
            decision = str(route["kga_decision"]).upper()
            bhat = float(route["b_hat"])
            epsilon = float(route["eps_conformal"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LineageError(f"locked route {index} is incomplete: {path}") from exc
        if seed not in seeds or not condition:
            raise LineageError(f"locked route {index} has invalid identity: {path}")
        if not math.isfinite(bhat) or not math.isfinite(epsilon) or epsilon < 0:
            raise LineageError(f"locked route {index} has invalid prediction/radius: {path}")
        expected_decision = (
            "ADAPT" if bhat - epsilon > 0 else
            "FREEZE" if bhat + epsilon < 0 else
            "ABSTAIN"
        )
        if decision != expected_decision:
            raise LineageError(f"locked route {index} violates the decision rule: {path}")
        key = (seed, condition)
        if key in parsed:
            raise LineageError(f"duplicate locked route: {key}")
        parsed[key] = {
            "kga_decision": decision,
            "b_hat": bhat,
            "eps_conformal": epsilon,
        }
    return parsed


def _same_routes(left, right):
    if set(left) != set(right):
        return False
    for key in left:
        for field in ("b_hat", "eps_conformal"):
            if not _close(left[key][field], right[key][field], atol=1e-12):
                return False
        if left[key]["kga_decision"] != right[key]["kga_decision"]:
            return False
    return True


def _validate_decision_lock(aggregate, aggregate_path):
    reference = aggregate.get("decision_lock")
    if not isinstance(reference, dict):
        raise LineageError(f"immutable decision-lock reference is missing: {aggregate_path}")
    lock_path = _safe_child(aggregate_path.parent, reference.get("file"))
    expected_hash = _require_hex_sha256(reference.get("sha256"), "decision_lock.sha256")
    if not lock_path.is_file() or _sha256_file(lock_path) != expected_hash:
        raise LineageError(f"decision-lock hash mismatch: {lock_path}")
    lock = _strict_json_load(lock_path)
    if lock.get("schema") != DECISION_LOCK_SCHEMA:
        raise LineageError(f"unknown decision-lock schema: {lock_path}")

    artifact_common = {
        "dataset": aggregate.get("dataset"),
        "candidate": aggregate.get("candidate"),
        "metric_contract": aggregate.get("metric_contract"),
        "alpha": aggregate.get("alpha"),
        "model_seeds": aggregate.get("seeds"),
        "calibration_partition": aggregate.get("calibration_partition"),
    }
    common = {
        **artifact_common,
        "evaluation_partition": aggregate.get("evaluation_partition"),
        "bootstrap_replicates": aggregate.get("bootstrap_replicates"),
        "bootstrap_seed": aggregate.get("bootstrap_seed"),
    }
    for field, value in common.items():
        if lock.get(field) != value:
            raise LineageError(f"decision-lock {field} mismatch: {lock_path}")
    if lock.get("decisions_locked_before_evaluation") is not True:
        raise LineageError(f"decision lock does not declare pre-evaluation locking: {lock_path}")
    if lock.get("target_opened_before_lock") is not False:
        raise LineageError(f"decision lock records an opened target: {lock_path}")
    if lock.get("estimator_backend") != LOCKED_BACKEND:
        raise LineageError(f"decision lock lacks the validation-locked backend: {lock_path}")

    _, calibration_sha, calibration = _load_bound_artifact(
        lock_path.parent,
        lock.get("calibration_source"),
        field="calibration_source",
        schema=CALIBRATION_SOURCE_SCHEMA,
    )
    _, estimator_sha, estimator = _load_bound_artifact(
        lock_path.parent,
        lock.get("estimator_artifact"),
        field="estimator_artifact",
        schema=LOCKED_ESTIMATOR_SCHEMA,
    )
    _, protocol_sha, protocol = _load_bound_artifact(
        lock_path.parent,
        lock.get("protocol_artifact"),
        field="protocol_artifact",
        schema=LOCKED_PROTOCOL_SCHEMA,
    )
    for artifact_name, artifact in (
        ("calibration source", calibration),
        ("estimator", estimator),
    ):
        for field, value in artifact_common.items():
            if artifact.get(field) != value:
                raise LineageError(f"{artifact_name} {field} is not bound to the decision lock")
    for field, value in common.items():
        if protocol.get(field) != value:
            raise LineageError(f"protocol {field} is not bound to the decision lock")
    if estimator.get("estimator_backend") != LOCKED_BACKEND:
        raise LineageError("locked estimator artifact has an unexpected backend")

    conditions = protocol.get("expected_conditions")
    if not isinstance(conditions, list) or not conditions:
        raise LineageError("locked protocol has an invalid expected-condition inventory")
    try:
        expected_route_keys = {
            (int(entry["seed"]), str(entry["condition"])) for entry in conditions
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise LineageError("locked protocol has an invalid expected-condition inventory") from exc
    if (
        len(expected_route_keys) != len(conditions)
        or any(seed not in aggregate["seeds"] or not condition for seed, condition in expected_route_keys)
    ):
        raise LineageError("locked protocol has duplicate/invalid expected conditions")

    calibration_rows = calibration.get("calibration_rows")
    if not isinstance(calibration_rows, list) or not calibration_rows:
        raise LineageError("calibration source has no route-level residual evidence")
    recomputed = {}
    alpha = float(aggregate["alpha"])
    for index, row in enumerate(calibration_rows):
        try:
            seed = int(row["seed"])
            condition = str(row["condition"])
            bhat = float(row["b_hat"])
            declared_epsilon = float(row["eps_conformal"])
            residuals = np.asarray(row["absolute_residuals"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise LineageError(f"calibration row {index} is incomplete") from exc
        if residuals.ndim != 1 or not len(residuals) or not np.isfinite(residuals).all():
            raise LineageError(f"calibration row {index} has invalid absolute residuals")
        if np.any(residuals < 0) or not math.isfinite(bhat):
            raise LineageError(f"calibration row {index} has invalid values")
        rank = int(math.ceil((len(residuals) + 1) * (1.0 - alpha)))
        if rank > len(residuals):
            raise LineageError(f"calibration row {index} has an infeasible exact-rank radius")
        epsilon = float(np.sort(residuals)[rank - 1])
        decision = (
            "ADAPT" if bhat - epsilon > 0 else
            "FREEZE" if bhat + epsilon < 0 else
            "ABSTAIN"
        )
        declared_decision = row.get("kga_decision")
        if (
            not _close(declared_epsilon, epsilon, atol=1e-12)
            or (
                declared_decision is not None
                and str(declared_decision).upper() != decision
            )
        ):
            raise LineageError(f"calibration row {index} route does not recompute exactly")
        key = (seed, condition)
        if key in recomputed:
            raise LineageError(f"duplicate calibration route: {key}")
        recomputed[key] = {
            "kga_decision": decision,
            "b_hat": bhat,
            "eps_conformal": epsilon,
        }
    if set(recomputed) != expected_route_keys:
        raise LineageError("calibration routes do not cover the locked protocol inventory")

    estimator_routes = _parse_locked_routes(
        estimator.get("locked_routes"), path="locked estimator", seeds=aggregate["seeds"]
    )
    lock_routes_raw = lock.get("locked_routes")
    lock_routes = _parse_locked_routes(
        lock_routes_raw, path=lock_path, seeds=aggregate["seeds"]
    )
    if not _same_routes(recomputed, estimator_routes) or not _same_routes(recomputed, lock_routes):
        raise LineageError("locked routes differ from recomputed calibration evidence")
    routes_sha = _sha256_json(lock_routes_raw)
    if lock.get("locked_routes_sha256") != routes_sha:
        raise LineageError("decision-lock route inventory hash mismatch")

    _, _, receipt = _load_bound_artifact(
        lock_path.parent,
        lock.get("immutable_receipt"),
        field="immutable_receipt",
        schema=PREOPENING_RECEIPT_SCHEMA,
    )
    receipt_expected = {
        **common,
        "calibration_source_sha256": calibration_sha,
        "estimator_artifact_sha256": estimator_sha,
        "protocol_sha256": protocol_sha,
        "locked_routes_sha256": routes_sha,
    }
    for field, value in receipt_expected.items():
        if receipt.get(field) != value:
            raise LineageError(f"pre-opening receipt {field} mismatch")
    if receipt.get("target_opened_at_utc") is not None:
        raise LineageError("pre-opening receipt says the target was already opened")
    _parse_utc_timestamp(receipt.get("locked_at_utc"), "receipt.locked_at_utc")
    provider = receipt.get("provider")
    immutable_uri = receipt.get("immutable_uri")
    allowed_hosts = {"osf": "osf.io", "zenodo": "zenodo.org", "doi": "doi.org"}
    parsed_uri = urlparse(immutable_uri) if isinstance(immutable_uri, str) else None
    if (
        provider not in allowed_hosts
        or parsed_uri is None
        or parsed_uri.scheme != "https"
        or parsed_uri.hostname != allowed_hosts[provider]
        or not parsed_uri.path.strip("/")
    ):
        raise LineageError("pre-opening receipt lacks a recognized immutable external URI")
    return lock_routes


def _validate_recomputed_aggregate(aggregate, per_seed, path, scope):
    """Recompute every plotted/statistical claim from the hashed condition rows."""
    if [row["seed"] for row in per_seed] != sorted(int(seed) for seed in aggregate["seeds"]):
        raise LineageError(f"recomputed seed inventory mismatch: {path}")
    arrays = {
        "regret_kga": np.array([row["regret_kga"] for row in per_seed], dtype=float),
        "regret_adapt": np.array([row["regret_adapt"] for row in per_seed], dtype=float),
        "regret_freeze": np.array([row["regret_freeze"] for row in per_seed], dtype=float),
        "FA_u": np.array([row["fa_u"] for row in per_seed], dtype=float),
    }
    for field in ("regret_kga", "regret_adapt", "regret_freeze"):
        declared = _finite_pair(field, aggregate.get(field))
        expected = (float(arrays[field].mean()), float(arrays[field].std()))
        if not all(_close(actual, target) for actual, target in zip(declared, expected)):
            raise LineageError(f"{field} disagrees with per-condition recomputation: {path}")

    declared_fau = aggregate.get("FA_u_per_seed")
    if not isinstance(declared_fau, list) or len(declared_fau) != len(per_seed):
        raise LineageError(f"FA_u_per_seed inventory is missing: {path}")
    if any(not _close(actual, target) for actual, target in zip(declared_fau, arrays["FA_u"])):
        raise LineageError(f"FA_u_per_seed disagrees with per-condition rows: {path}")
    if not _close(aggregate.get("FA_u_max"), float(arrays["FA_u"].max())):
        raise LineageError(f"FA_u_max disagrees with per-condition rows: {path}")

    try:
        alpha = float(aggregate["alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LineageError(f"aggregate alpha is missing/non-numeric: {path}") from exc
    if not 0.0 < alpha < 1.0:
        raise LineageError(f"aggregate alpha must lie in (0,1): {path}")
    replicates = aggregate.get("bootstrap_replicates")
    bootstrap_seed = aggregate.get("bootstrap_seed")
    gap_adapt = arrays["regret_adapt"] - arrays["regret_kga"]
    gap_freeze = arrays["regret_freeze"] - arrays["regret_kga"]
    ci_adapt = _bootstrap_ci(gap_adapt, replicates=replicates, seed=bootstrap_seed)
    ci_freeze = _bootstrap_ci(gap_freeze, replicates=replicates, seed=bootstrap_seed)
    expected_gaps = {
        "gap_vs_adapt": (float(gap_adapt.mean()), ci_adapt),
        "gap_vs_freeze": (float(gap_freeze.mean()), ci_freeze),
    }
    for field, (mean, ci) in expected_gaps.items():
        declared_mean, declared_low, declared_high = _finite_triplet(field, aggregate.get(field))
        if not _close(declared_mean, mean) or not _close(declared_low, ci[0]) or not _close(declared_high, ci[1]):
            raise LineageError(f"{field} disagrees with per-condition bootstrap recomputation: {path}")

    better = "freeze" if arrays["regret_freeze"].mean() <= arrays["regret_adapt"].mean() else "adapt"
    if aggregate.get("better_policy") != better:
        raise LineageError(f"better_policy disagrees with recomputed regrets: {path}")
    ci_better = ci_freeze if better == "freeze" else ci_adapt
    ci_worse = ci_adapt if better == "freeze" else ci_freeze
    if aggregate.get("gap_vs_better_ci95") != ci_better:
        raise LineageError(f"gap_vs_better_ci95 disagrees with recomputation: {path}")
    if aggregate.get("gap_vs_worse_ci95") != ci_worse:
        raise LineageError(f"gap_vs_worse_ci95 disagrees with recomputation: {path}")

    beats_both = ci_adapt[0] > 0 and ci_freeze[0] > 0
    ties_better = ci_better[0] <= 0 <= ci_better[1]
    beats_worse = ci_worse[0] > 0
    false_adapt_ok = bool(np.all(arrays["FA_u"] <= alpha + 1e-12))
    robust = bool(beats_both and false_adapt_ok)
    if scope == "heldout-candidate":
        expected_code = "WITHHELD_PENDING_INDEPENDENT_AUTHENTICITY_AUDIT"
        if aggregate.get("beats_both_promoted") is not False:
            raise LineageError(f"unverified held-out evidence cannot promote beats-both: {path}")
    else:
        expected_code = (
            "DEVELOPMENT_BEATS_BOTH_DIAGNOSTIC"
            if robust
            else "DEVELOPMENT_STABLE_NO_HARM_DIAGNOSTIC"
            if ties_better and beats_worse and false_adapt_ok
            else "DEVELOPMENT_OTHER_DIAGNOSTIC"
        )
        if aggregate.get("beats_both_promoted") is not False:
            raise LineageError(f"development evidence cannot promote beats-both: {path}")
    if aggregate.get("verdict_code") != expected_code or aggregate.get("verdict") != expected_code:
        raise LineageError(f"verdict disagrees with recomputed evidence: {path}")


def _validate_scope(aggregate, path, scope):
    partition = _normalise_partition(aggregate.get("evaluation_partition"))
    seed_kind = aggregate.get("seed_kind")
    analysis = str(aggregate.get("analysis", "")).lower()
    if scope == "heldout-candidate":
        if seed_kind != "model_seed":
            raise LineageError(f"stream seeds cannot support the held-out candidate audit: {path}")
        if partition not in HELDOUT_PARTITIONS:
            raise LineageError(f"development/opened partition cannot support held-out output: {path}")
        calibration = _normalise_partition(aggregate.get("calibration_partition"))
        if calibration not in DEVELOPMENT_PARTITIONS:
            raise LineageError(f"disjoint development calibration partition is missing: {path}")
        if aggregate.get("decisions_locked_before_evaluation") is not True:
            raise LineageError(f"decisions were not explicitly locked before evaluation: {path}")
        if aggregate.get("target_opened_before_lock") is not False:
            raise LineageError(f"test target was opened before the decision lock: {path}")
        required_true = (
            "sources_computationally_ready",
            "estimator_computationally_locked",
            "calibration_feasible_all",
        )
        missing = [field for field in required_true if aggregate.get(field) is not True]
        if missing:
            raise LineageError(f"held-out candidate lacks computational prerequisites ({missing}): {path}")
        required_false = (
            "external_authenticity_verified",
            "publication_eligible",
            "sources_publication_ready",
            "estimator_publication_eligible",
            "model_seed_ci_eligible",
            "confirmatory_ci_eligible",
            "heldout_promotion_eligible",
            "beats_both_promoted",
        )
        promoted = [field for field in required_false if aggregate.get(field) is not False]
        if promoted:
            raise LineageError(
                f"local held-out candidate must remain non-promotional ({promoted}): {path}"
            )
        if "within_development_partition" in analysis:
            raise LineageError(f"within-partition fitting cannot support a held-out candidate: {path}")
        if aggregate.get("statistical_verdict_withheld") is not True:
            raise LineageError(f"confirmatory verdict must be withheld pending external audit: {path}")
        if aggregate.get("kga_backend") != ["sklearn_gradient_boost_locked_validation_v1"]:
            raise LineageError(f"held-out candidate requires a validation-locked estimator: {path}")
    else:
        if seed_kind != "model_seed":
            raise LineageError(f"stream-seed evidence is excluded even from CI forest output: {path}")
        if partition not in DEVELOPMENT_PARTITIONS:
            raise LineageError(f"development diagnostic requires a development partition: {path}")
        if aggregate.get("model_seed_ci_eligible") is not True:
            raise LineageError(f"development diagnostic lacks independent model-seed CI: {path}")
        forbidden_true = (
            "external_authenticity_verified",
            "publication_eligible",
            "sources_publication_ready",
            "estimator_publication_eligible",
            "confirmatory_ci_eligible",
            "heldout_promotion_eligible",
            "beats_both_promoted",
        )
        mislabeled = [field for field in forbidden_true if aggregate.get(field) is not False]
        if mislabeled:
            raise LineageError(
                f"development diagnostic carries publication/confirmatory flags ({mislabeled}): {path}"
            )


def _validate_aggregate(path, scope):
    path = Path(path).resolve()
    aggregate = _strict_json_load(path)
    if aggregate.get("schema") != AGGREGATE_SCHEMA:
        raise LineageError(f"legacy/unknown aggregate schema: {path}")
    if aggregate.get("lineage_contract") != LINEAGE_CONTRACT:
        raise LineageError(f"aggregate lacks hardened lineage contract: {path}")
    dataset = aggregate.get("dataset")
    candidate = aggregate.get("candidate")
    if not dataset or not candidate:
        raise LineageError(f"dataset/candidate missing: {path}")
    tta_protocol = _validate_tta_protocol(
        aggregate.get("tta_protocol"), candidate=candidate, context=f"aggregate {path}"
    )
    if aggregate.get("tta_protocol_sha256") != _sha256_json(tta_protocol):
        raise LineageError(f"aggregate candidate tta_protocol hash mismatch: {path}")
    heldout = scope == "heldout-candidate"
    if heldout and _normalise_partition(aggregate.get("evaluation_partition")) not in HELDOUT_PARTITIONS:
        raise LineageError(f"development/opened partition cannot support held-out output: {path}")
    expected_metric = _expected_metric_contract(dataset) if heldout else None
    if heldout and aggregate.get("metric_contract") != expected_metric:
        raise LineageError(
            f"aggregate metric_contract is required and must match the reviewed dataset metric: {path}"
        )
    seeds = aggregate.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise LineageError(f"at least two unique seeds are required: {path}")
    if aggregate.get("n_seeds") != len(seeds):
        raise LineageError(f"n_seeds disagrees with seed inventory: {path}")
    if aggregate.get("seed_kind") == "model_seed":
        checkpoints = aggregate.get("checkpoint_sha256_by_seed")
        if not isinstance(checkpoints, dict) or set(checkpoints) != {str(seed) for seed in seeds}:
            raise LineageError(f"complete model-seed checkpoint inventory is required: {path}")
        hashes = list(checkpoints.values())
        if len(set(hashes)) != len(hashes) or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
            for value in hashes
        ):
            raise LineageError(f"checkpoint hashes must be valid and unique by model seed: {path}")
        tensor_checkpoints = aggregate.get("checkpoint_tensor_sha256_by_seed")
        if (
            not isinstance(tensor_checkpoints, dict)
            or set(tensor_checkpoints) != {str(seed) for seed in seeds}
        ):
            raise LineageError(
                f"complete model-seed checkpoint tensor-state inventory is required: {path}"
            )
        tensor_hashes = list(tensor_checkpoints.values())
        if len(set(tensor_hashes)) != len(tensor_hashes) or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
            for value in tensor_hashes
        ):
            raise LineageError(
                f"checkpoint tensor-state hashes must be valid and unique by model seed: {path}"
            )
    _finite_pair("regret_kga", aggregate.get("regret_kga"))
    _finite_pair("regret_adapt", aggregate.get("regret_adapt"))
    _finite_pair("regret_freeze", aggregate.get("regret_freeze"))
    try:
        false_adapt = float(aggregate["FA_u_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LineageError(f"FA_u_max must be numeric: {path}") from exc
    if not math.isfinite(false_adapt) or not 0.0 <= false_adapt <= 1.0:
        raise LineageError(f"FA_u_max must lie in [0,1]: {path}")
    if not isinstance(aggregate.get("verdict"), str) or not aggregate["verdict"].strip():
        raise LineageError(f"verdict is missing: {path}")
    _finite_triplet("gap_vs_adapt", aggregate.get("gap_vs_adapt"))
    _finite_triplet("gap_vs_freeze", aggregate.get("gap_vs_freeze"))

    manifest_path = path.parent / f"extract_manifest_{dataset}.json"
    manifest = _strict_json_load(manifest_path)
    if manifest.get("schema") != EXTRACT_SCHEMA:
        raise LineageError(f"missing current hardened extraction manifest: {manifest_path}")
    if manifest.get("track") != dataset:
        raise LineageError(f"manifest dataset/track mismatch: {manifest_path}")
    if path.name not in manifest.get("aggregates", []):
        raise LineageError(f"aggregate is absent from current manifest inventory: {path}")
    expected_hash = (manifest.get("aggregate_sha256") or {}).get(path.name)
    if expected_hash != _sha256_file(path):
        raise LineageError(f"aggregate hash is absent/stale in manifest: {path}")
    if candidate not in manifest.get("requested_candidates", []):
        raise LineageError(f"candidate is absent from current request manifest: {path}")
    manifest_protocols = manifest.get("validated_tta_protocol_by_candidate")
    if (
        not isinstance(manifest_protocols, dict)
        or set(manifest_protocols) != set(manifest.get("requested_candidates", []))
        or manifest_protocols.get(candidate) != tta_protocol
    ):
        raise LineageError(f"manifest does not preserve the validated candidate tta_protocol: {path}")
    if heldout:
        _validate_serialization_generation(manifest, manifest_path, manifest_protocols)
    manifest_seeds = manifest.get("expected_seeds")
    if (
        not isinstance(manifest_seeds, list)
        or len(manifest_seeds) != len(set(manifest_seeds))
        or {int(seed) for seed in manifest_seeds} != {int(seed) for seed in seeds}
    ):
        raise LineageError(f"manifest seed inventory mismatch: {path}")
    if manifest.get("seed_kind") != aggregate.get("seed_kind"):
        raise LineageError(f"manifest seed-kind mismatch: {path}")
    source_partitions, source_records, verified_checkpoints = _validate_source_hashes(
        manifest,
        manifest_path,
        dataset=dataset,
        expected_metric=expected_metric,
        expected_seeds=seeds,
        require_heldout_candidate=heldout,
    )
    if source_partitions != {_normalise_partition(aggregate.get("evaluation_partition"))}:
        raise LineageError(f"source/aggregate partition lineage mismatch: {path}")
    locked_routes = (
        _validate_decision_lock(aggregate, path)
        if heldout
        else None
    )
    expected_source_keys = (
        _expected_candidate_source_keys(
            source_records,
            dataset=dataset,
            candidate=candidate,
            seeds=seeds,
        )
        if heldout
        else None
    )
    per_seed = _validate_per_condition_files(
        aggregate,
        path,
        source_records=source_records,
        expected_source_keys=expected_source_keys,
        expected_metric=expected_metric,
        verified_checkpoints=verified_checkpoints if heldout else None,
        locked_routes=locked_routes,
    )
    _validate_recomputed_aggregate(aggregate, per_seed, path, scope)
    _validate_scope(aggregate, path, scope)
    return aggregate


def _expand_inputs(patterns):
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        files.extend(matches)
    unique = []
    seen = set()
    for raw in files:
        path = str(Path(raw).resolve())
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        raise LineageError(f"no aggregate JSONs matched: {patterns}")
    return unique


def _load_aggs(patterns, scope="heldout-candidate"):
    aggregates = []
    keys = set()
    for path in _expand_inputs(patterns):
        if not path.endswith(".json") or not Path(path).name.startswith("multiseed_"):
            raise LineageError(f"input is not a multiseed aggregate JSON: {path}")
        aggregate = _validate_aggregate(path, scope)
        key = (aggregate["dataset"], aggregate["candidate"])
        if key in keys:
            raise LineageError(f"duplicate aggregate key in current inputs: {key}")
        keys.add(key)
        aggregates.append((path, aggregate))
    return sorted(aggregates, key=lambda item: (item[1]["dataset"], item[1]["candidate"]))


def _gaps(aggregate):
    dataset = DISPLAY.get(aggregate["dataset"], aggregate["dataset"])
    candidate = aggregate["candidate"]
    prefix = f"{dataset} ({candidate})"
    return (
        prefix,
        _finite_triplet("gap_vs_adapt", aggregate["gap_vs_adapt"]),
        _finite_triplet("gap_vs_freeze", aggregate["gap_vs_freeze"]),
    )


def forest(rows, out_fig, scope):
    if not rows:
        raise LineageError("forest plot requires at least one validated row")
    fig_h = max(3.2, 0.45 * len(rows) + 1.2)
    fig, axis = plt.subplots(figsize=(6.8, fig_h))
    y_values = list(range(len(rows)))[::-1]
    for y_value, (label, mean, lo, hi) in zip(y_values, rows):
        color = GREEN if lo > 0 else GRAY
        axis.plot([lo, hi], [y_value, y_value], color=color, lw=3, solid_capstyle="round")
        axis.plot(mean, y_value, "o", color=color, ms=7)
    axis.axvline(0, color="k", lw=1, ls="--")
    axis.set_yticks(y_values)
    axis.set_yticklabels([row[0] for row in rows], fontsize=8)
    axis.set_xlabel("Regret reduction by KGA (95% declared-seed bootstrap diagnostic)")
    title = (
        "Held-out natural-shift candidate (chronology requires external audit)"
        if scope == "heldout-candidate"
        else "Development-only natural-shift stability diagnostic (not held-out evidence)"
    )
    axis.set_title(title, fontsize=10)
    axis.legend(
        handles=[
            Patch(color=GREEN, label="Diagnostic interval excludes 0"),
            Patch(color=GRAY, label="CI includes 0 (ties / inconclusive)"),
        ],
        fontsize=7.5,
        loc="lower right",
        framealpha=0.9,
    )
    axis.margins(y=0.08)
    fig.tight_layout()
    out_fig = Path(out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{out_fig.stem}.", suffix=out_fig.suffix or ".png", dir=out_fig.parent
    )
    os.close(fd)
    try:
        fig.savefig(temporary, dpi=200)
        os.chmod(temporary, 0o644)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, out_fig)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    finally:
        plt.close(fig)


def _latex_escape(value):
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
    return "".join(replacements.get(character, character) for character in str(value))


def latex_table(aggregates, out_tex, scope, generation_id=None):
    scope_note = (
        "held-out candidate; computational checks passed, chronology externally auditable"
        if scope == "heldout-candidate"
        else "development diagnostic only; not held-out evidence"
    )
    lines = [
        r"% Auto-generated by make_multiseed_natural_forest.py -- do not hand-edit.",
        f"% Generation: {generation_id or 'unspecified'}",
        f"% Scope: {scope_note}",
        r"\begin{tabular}{lcccccl}",
        r"\toprule",
        r"track (candidate) & seeds & KGA regret & adapt & freeze & "
        r"$\mathrm{FA}_{\mathrm u}$ & verdict \\",
        r"\midrule",
    ]
    for _, aggregate in aggregates:
        dataset = DISPLAY.get(aggregate["dataset"], aggregate["dataset"])
        candidate = aggregate["candidate"]
        routed = aggregate["regret_kga"]
        lines.append(
            f"{_latex_escape(dataset)} ({_latex_escape(candidate)}) & {aggregate['n_seeds']} & "
            f"{routed[0]:.4f}$\\pm${routed[1]:.4f} & "
            f"{aggregate['regret_adapt'][0]:.4f} & {aggregate['regret_freeze'][0]:.4f} & "
            f"{aggregate['FA_u_max']:.3f} & {_latex_escape(aggregate['verdict'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    _atomic_text("\n".join(lines), out_tex)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg", nargs="+", required=True, help="current multiseed v2 aggregates")
    parser.add_argument(
        "--scope",
        choices=["heldout-candidate", "development-diagnostic"],
        default="heldout-candidate",
    )
    parser.add_argument("--out-fig", default=str(FIG_DIR / "fig_natural_forest_multiseed.png"))
    parser.add_argument("--out-tex", default=str(FIG_DIR / "tab_multiseed_natural.tex"))
    parser.add_argument(
        "--out-json", default=str(FIG_DIR / "multiseed_natural_forest_payload.json")
    )
    return parser.parse_args(argv)


def verify_committed_generation(payload_path):
    """Verify the payload commit marker binds one complete figure/table generation."""

    payload_path = Path(payload_path).resolve()
    payload = _strict_json_load(payload_path)
    if not isinstance(payload, dict) or payload.get("schema") != "kbound_natural_multiseed_forest_payload_v2":
        raise LineageError(f"unrecognized forest generation payload {payload_path}")
    if payload.get("generation_committed") is not True:
        raise LineageError(f"forest generation is not committed in {payload_path}")
    generation_id = payload.get("generation_id")
    if (
        not isinstance(generation_id, str)
        or len(generation_id) != 64
        or any(character not in "0123456789abcdef" for character in generation_id)
    ):
        raise LineageError(f"invalid forest generation id in {payload_path}")
    for artifact_name in ("figure", "table"):
        descriptor = payload.get(artifact_name)
        if not isinstance(descriptor, dict):
            raise LineageError(f"forest payload lacks {artifact_name} descriptor")
        artifact_path = Path(str(descriptor.get("path", ""))).resolve()
        expected_hash = descriptor.get("sha256")
        if not artifact_path.is_file():
            raise LineageError(f"committed forest {artifact_name} is missing: {artifact_path}")
        if not isinstance(expected_hash, str) or _sha256_file(artifact_path) != expected_hash:
            raise LineageError(
                f"committed forest {artifact_name} hash mismatch for generation {generation_id}"
            )
    table_text = Path(payload["table"]["path"]).read_text(encoding="utf-8")
    if f"% Generation: {generation_id}" not in table_text:
        raise LineageError("forest table generation marker does not match payload")
    return payload


def main(argv=None):
    args = parse_args(argv)
    aggregates = _load_aggs(args.agg, scope=args.scope)
    forest_rows = []
    payload_rows = []
    for path, aggregate in aggregates:
        prefix, adapt_gap, freeze_gap = _gaps(aggregate)
        forest_rows.append((f"{prefix}\nvs always-adapt", *adapt_gap))
        forest_rows.append((f"{prefix}\nvs always-freeze", *freeze_gap))
        payload_rows.append(
            {
                "source": str(Path(path).resolve()),
                "source_sha256": _sha256_file(path),
                "diagnostic": {
                    "dataset": aggregate["dataset"],
                    "candidate": aggregate["candidate"],
                    "declared_seed_count": aggregate["n_seeds"],
                    "regret_kga": aggregate["regret_kga"],
                    "regret_adapt": aggregate["regret_adapt"],
                    "regret_freeze": aggregate["regret_freeze"],
                    "gap_vs_adapt": aggregate["gap_vs_adapt"],
                    "gap_vs_freeze": aggregate["gap_vs_freeze"],
                    "false_adapt_rate_max": aggregate["FA_u_max"],
                    "verdict": aggregate["verdict"],
                    "publication_status": "WITHHELD_PENDING_INDEPENDENT_AUTHENTICITY_AUDIT",
                },
            }
        )
    source_inventory = [
        {"path": str(Path(path).resolve()), "sha256": _sha256_file(path)}
        for path, _ in aggregates
    ]
    generation_id = _sha256_json(
        {"scope": args.scope, "sources": source_inventory, "schema": "forest_generation_v1"}
    )
    out_fig = Path(args.out_fig).resolve()
    out_tex = Path(args.out_tex).resolve()
    out_json = Path(args.out_json).resolve()
    for output in (out_fig, out_tex, out_json):
        output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".forest-generation-", dir=out_json.parent))
    stage_fig = stage / out_fig.name
    stage_tex = stage / out_tex.name
    stage_json = stage / out_json.name
    try:
        forest(forest_rows, stage_fig, args.scope)
        latex_table(aggregates, stage_tex, args.scope, generation_id=generation_id)
        payload = {
            "schema": "kbound_natural_multiseed_forest_payload_v2",
            "generation_id": generation_id,
            "generation_committed": True,
            "scope": args.scope,
            "lineage_verified": False,
            "computational_lineage_verified": True,
            "chronology_independently_verified": False,
            "verification_limit": (
                "Local validation proves hashes, complete row lineage, locked-route consistency, "
                "and recomputed statistics; it does not independently prove wall-clock chronology."
            ),
            "rows": payload_rows,
            "figure": {"path": str(out_fig), "sha256": _sha256_file(stage_fig)},
            "table": {"path": str(out_tex), "sha256": _sha256_file(stage_tex)},
        }
        _atomic_json_dump(payload, stage_json)
        # The payload is the commit marker and is published last. A crash during
        # the preceding replaces leaves the old payload hashes, so mixed output
        # generations fail verification instead of appearing current.
        os.replace(stage_fig, out_fig)
        os.replace(stage_tex, out_tex)
        os.replace(stage_json, out_json)
        verify_committed_generation(out_json)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(f"validated {len(aggregates)} aggregates -> {len(forest_rows)} forest rows")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LineageError as error:
        print(f"LINEAGE ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
