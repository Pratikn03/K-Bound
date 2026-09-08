"""Validate saved claims for the physical-camera study's publication gate.

Primitive/schema checks are not authentication of study artifacts or custody.
This gate does not independently establish the truth of an asserted audit result.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

from kbound_edge.real_manifest import canonical_protocol_hash
from kbound_edge.recording import build_session_checklist

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _mapping(value: Any) -> dict[str, Any]:
    return value if type(value) is dict else {}


def _strings(value: Any) -> bool:
    return type(value) is list and bool(value) and all(type(item) is str for item in value)


def _metric(value: Any) -> float | None:
    # Bounds precede isfinite so a huge native integer cannot overflow coercion.
    if type(value) in (int, float) and 0 <= value <= 1 and math.isfinite(value):
        return float(value)
    return None


def _timestamp(value: Any) -> datetime:
    if type(value) is not str:
        raise ValueError("invalid timestamp")
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError("timestamp requires timezone")
    return result


def _config_shape(cfg: Any) -> bool:
    if type(cfg) is not dict or not _strings(cfg.get("classes")):
        return False
    sessions = cfg.get("sessions")
    if type(sessions) is not dict or not {"S07", "S08", "S09", "S10"}.issubset(sessions):
        return False
    return all(
        type(sid) is str
        and type(row) is dict
        and type(row.get("windows")) is int
        and row["windows"] > 0
        and _strings(row.get("objects"))
        for sid, row in sessions.items()
    )


def evaluate_publication_gate(
    cfg: dict[str, Any],
    *,
    model_card: dict[str, Any],
    split_audit: dict[str, Any],
    inventory: dict[str, Any],
    heldout: dict[str, Any],
    replication: dict[str, Any],
    anti_leakage: dict[str, Any],
) -> dict[str, Any]:
    """Return failed checks for malformed saved values, without coercing claims.

    The configuration is active input; saved artifacts remain unauthenticated.
    Audit names/provenance and exact clip identities need separate binding.
    """

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any) -> None:
        checks.append({"check": name, "passed": passed is True, "observed": observed})

    try:
        if not _config_shape(cfg):
            raise ValueError("invalid configuration")
        expected_hash = canonical_protocol_hash(cfg)
        expected_counts = {sid: len(build_session_checklist(cfg, sid)) for sid in sorted(cfg["sessions"])}
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        add("protocol configuration has valid shape", False, "invalid configuration")
        return {"passed": False, "protocol_hash": None, "checks": checks}

    artifacts = (model_card, split_audit, inventory, heldout, replication, anti_leakage)
    artifact_shapes = all(type(value) is dict for value in artifacts)
    model_card, split_audit, inventory, heldout, replication, anti_leakage = map(_mapping, artifacts)
    metrics_raw = model_card.get("metrics")
    sealed_raw = split_audit.get("sealed_splits")
    clips_raw = inventory.get("clips")
    audit_raw = anti_leakage.get("checks")
    nested_shapes = (
        type(metrics_raw) is dict
        and type(sealed_raw) is dict
        and type(clips_raw) is list
        and all(type(row) is dict for row in clips_raw)
        and type(audit_raw) is list
        and all(type(row) is dict for row in audit_raw)
    )
    if not artifact_shapes or not nested_shapes:
        add("saved study artifact containers have valid shapes", False, "invalid saved artifact container")
    model_metrics = _mapping(metrics_raw)
    sealed_splits = _mapping(sealed_raw)
    clips = [_mapping(row) for row in clips_raw] if type(clips_raw) is list else []
    audit_checks = [_mapping(row) for row in audit_raw] if type(audit_raw) is list else []
    bal = _metric(model_metrics.get("val_balanced_acc"))
    macro = _metric(model_metrics.get("val_macro_f1"))
    command_raw = model_card.get("training_command")
    command = command_raw if type(command_raw) is str else None
    model_hash = model_card.get("protocol_hash")
    model_hash = model_hash if type(model_hash) is str else None
    sealed = sealed_splits.get("calibration_conformal")

    add("protocol hash matches source model", model_hash == expected_hash, model_hash)
    add("development split sealed", sealed is True, {"calibration_conformal": sealed if type(sealed) is bool else None})
    add("source balanced accuracy >= 0.80", bal is not None and bal >= 0.80, bal)
    add("source macro-F1 >= 0.80", macro is not None and macro >= 0.80, macro)
    add(
        "source gate was not bypassed",
        command is not None and bool(command.strip()) and "--bypass-gate" not in command,
        command,
    )

    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in clips:
        sid = row.get("session_id")
        if type(sid) is str:
            by_session.setdefault(sid, []).append(row)

    capture_counts: dict[str, dict[str, int]] = {}
    complete = all(type(row.get("session_id")) is str and row["session_id"] in cfg["sessions"] for row in clips)
    for sid, expected in expected_counts.items():
        observed = len(by_session.get(sid, []))
        capture_counts[sid] = {"expected": expected, "observed": observed}
        complete = complete and observed == expected
    add("all physical session clips present exactly once", complete, capture_counts)
    add(
        "all inventory clips are physical",
        bool(clips)
        and all(type(row.get("capture_mode")) is str and row["capture_mode"] == "physical" for row in clips),
        sorted({row["capture_mode"] if type(row.get("capture_mode")) is str else "invalid" for row in clips}),
    )
    hashes = [row.get("sha256") for row in clips]
    valid_hashes = bool(hashes) and all(type(value) is str and _SHA256.fullmatch(value) is not None for value in hashes)
    add("all clip hashes are present and unique", valid_hashes and len(hashes) == len(set(hashes)), len(hashes))

    sealed_at_raw = split_audit.get("sealed_at")
    heldout_rows = [
        row for row in clips if type(row.get("session_id")) is str and row["session_id"] in {"S07", "S08", "S09", "S10"}
    ]
    try:
        sealed_at = _timestamp(sealed_at_raw)
        capture_times = [_timestamp(row.get("captured_at")) for row in heldout_rows]
        opened_after_seal = bool(capture_times) and all(ts > sealed_at for ts in capture_times)
    except (KeyError, TypeError, ValueError, OverflowError):
        opened_after_seal = False
    add(
        "held-out and replication captures occurred after development seal",
        opened_after_seal,
        {"sealed_at": sealed_at_raw if type(sealed_at_raw) is str else None, "n_test_clips": len(heldout_rows)},
    )

    expected_heldout = sum(cfg["sessions"][sid]["windows"] for sid in ("S07", "S08"))
    expected_replication = sum(cfg["sessions"][sid]["windows"] for sid in ("S09", "S10"))
    n_heldout = heldout.get("n_windows")
    n_replication = replication.get("n_windows")
    add(
        "held-out replay is complete",
        type(n_heldout) is int and n_heldout == expected_heldout,
        n_heldout if type(n_heldout) is int else None,
    )
    add(
        "replication replay is complete",
        type(n_replication) is int and n_replication == expected_replication,
        n_replication if type(n_replication) is int else None,
    )

    add(
        "strict anti-leakage audit passes",
        len(audit_checks) == 8
        and all(
            type(row.get("check")) is str and bool(row["check"].strip()) and row.get("passed") is True
            for row in audit_checks
        ),
        [
            {
                "check": row.get("check") if type(row.get("check")) is str else None,
                "passed": row.get("passed") if type(row.get("passed")) is bool else None,
            }
            for row in audit_checks
        ],
    )

    return {
        "passed": all(row["passed"] for row in checks),
        "protocol_hash": expected_hash,
        "checks": checks,
    }
