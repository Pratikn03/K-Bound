"""Byte and portable provenance locks for non-K-Bound historical assertions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "experiments/audit"
RECEIPT = AUDIT / "historical_diagnostic_recovery_receipt.portable.json"
PORTABLE_SHA256 = "90954bae822a8c124278ab2f1d5b780e1649eaf8e42ff6eac807ebe38efb6892"
EXPECTED = {
    "experiments/audit/polarity_diagnostic_log.csv": {
        "bytes": 1_744,
        "sha256": "65d6ffba90ff7ac418b5233a9ee50bea33064dd219390807469f99aa918408ee",
    },
    "experiments/audit/canonical_label_semantics.json": {
        "bytes": 23_650,
        "sha256": "d52a8d83f905c59c3068fb7105b7ddedea0aec23dad3f6ede7d0b6d73bea30c7",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recovered_historical_diagnostic_bytes_are_exact() -> None:
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        assert path.is_file(), f"missing recovered historical assertion: {relative}"
        assert path.stat().st_size == expected["bytes"]
        assert _sha256(path) == expected["sha256"]


def test_recovery_receipt_classifies_scope_without_claiming_recomputation() -> None:
    # The public test consumes the reviewed path-only companion. Original raw
    # receipt identity is retained; the private checkout is not a test input.
    assert _sha256(RECEIPT) == PORTABLE_SHA256
    companion = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert companion["schema"] == "portable-historical-diagnostic-recovery-receipt-v1"
    assert companion["original_receipt"] == {
        "logical_path": "experiments/audit/historical_diagnostic_recovery_receipt.json",
        "sha256": "3e53221320370bb6e62f7e1955e2ecdb37822a2d9613f28960b8cf09d44a27c8",
        "bytes": 1122,
        "original_bytes_included": False,
    }
    receipt = companion["historical_receipt"]
    assert receipt["classification"] == "recovered_historical_assertions"
    assert receipt["scope"] == "non-K-Bound MVTec/VisA diagnostics"
    assert receipt["metric_recomputation_performed"] is False
    assert receipt["kbound_confirmation"] is False
    rows = {row["destination_path"]: row for row in receipt["artifacts"]}
    assert set(rows) == set(EXPECTED)
    for relative, expected in EXPECTED.items():
        assert rows[relative]["bytes"] == expected["bytes"]
        assert rows[relative]["sha256"] == expected["sha256"]
        assert rows[relative]["source_sha256"] == expected["sha256"]
    assert rows["experiments/audit/polarity_diagnostic_log.csv"]["blank_fields_preserved"] is True
