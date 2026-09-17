"""Publication gate for the preregistered physical-camera study."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from kbound_edge.real_manifest import canonical_protocol_hash
from kbound_edge.recording import build_session_checklist


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
    """Return a fail-closed release decision from saved study artifacts."""

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "observed": observed})

    expected_hash = canonical_protocol_hash(cfg)
    model_metrics = model_card.get("metrics") or {}
    bal = float(model_metrics.get("val_balanced_acc", 0.0))
    macro = float(model_metrics.get("val_macro_f1", 0.0))
    command = str(model_card.get("training_command", ""))

    add("protocol hash matches source model", model_card.get("protocol_hash") == expected_hash, model_card.get("protocol_hash"))
    add("development split sealed", bool((split_audit.get("sealed_splits") or {}).get("calibration_conformal")), split_audit.get("sealed_splits"))
    add("source balanced accuracy >= 0.80", bal >= 0.80, bal)
    add("source macro-F1 >= 0.80", macro >= 0.80, macro)
    add("source gate was not bypassed", "--bypass-gate" not in command, command)

    clips = inventory.get("clips") or []
    by_session: dict[str, list[dict[str, Any]]] = {}
    for row in clips:
        by_session.setdefault(str(row.get("session_id")), []).append(row)

    capture_counts: dict[str, dict[str, int]] = {}
    complete = True
    for sid in sorted(cfg["sessions"]):
        expected = len(build_session_checklist(cfg, sid))
        observed = len(by_session.get(sid, []))
        capture_counts[sid] = {"expected": expected, "observed": observed}
        complete = complete and observed == expected
    add("all physical session clips present exactly once", complete, capture_counts)
    add(
        "all inventory clips are physical",
        bool(clips) and all(row.get("capture_mode") == "physical" for row in clips),
        sorted({str(row.get("capture_mode", "missing")) for row in clips}),
    )
    hashes = [row.get("sha256") for row in clips]
    add("all clip hashes are present and unique", bool(hashes) and None not in hashes and len(hashes) == len(set(hashes)), len(hashes))

    sealed_at_raw = split_audit.get("sealed_at")
    heldout_rows = [
        row for row in clips if row.get("session_id") in {"S07", "S08", "S09", "S10"}
    ]
    try:
        sealed_at = datetime.fromisoformat(str(sealed_at_raw))
        capture_times = [datetime.fromisoformat(str(row["captured_at"])) for row in heldout_rows]
        opened_after_seal = bool(capture_times) and all(ts > sealed_at for ts in capture_times)
    except (KeyError, TypeError, ValueError):
        opened_after_seal = False
    add(
        "held-out and replication captures occurred after development seal",
        opened_after_seal,
        {"sealed_at": sealed_at_raw, "n_test_clips": len(heldout_rows)},
    )

    expected_heldout = sum(int(cfg["sessions"][sid]["windows"]) for sid in ("S07", "S08"))
    expected_replication = sum(int(cfg["sessions"][sid]["windows"]) for sid in ("S09", "S10"))
    add("held-out replay is complete", heldout.get("n_windows") == expected_heldout, heldout.get("n_windows"))
    add("replication replay is complete", replication.get("n_windows") == expected_replication, replication.get("n_windows"))

    audit_checks = anti_leakage.get("checks") or []
    add(
        "strict anti-leakage audit passes",
        len(audit_checks) == 8 and all(bool(row.get("passed")) for row in audit_checks),
        [{"check": row.get("check"), "passed": row.get("passed")} for row in audit_checks],
    )

    return {
        "passed": all(row["passed"] for row in checks),
        "protocol_hash": expected_hash,
        "checks": checks,
    }
