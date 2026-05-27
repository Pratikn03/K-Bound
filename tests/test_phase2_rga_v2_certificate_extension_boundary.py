"""Phase 2.2B.2 — RGA-v2 certificate extension boundary: only G0 rows in v2 CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CERT_V2 = ROOT / "experiments" / "phase2" / "certification" / "switching_certificates_v2.csv"
DECISION = ROOT / "docs" / "research" / "phase2" / "RGA_V2_CERTIFICATE_EXTENSION_DECISION.md"


def test_decision_doc_states_no_admissible_extension():
    t = DECISION.read_text()
    assert "No RGA-v2 partial-failure certificate extension is admissible" in t


def test_v2_certificate_csv_contains_only_g0_rows():
    if not CERT_V2.exists():
        pytest.skip("v2 cert CSV not yet produced")
    with CERT_V2.open() as f:
        for r in csv.DictReader(f):
            assert r["gate_id"].startswith(
                "G0"
            ), f"row gate_id={r['gate_id']!r} — only G0 rows allowed until any RGA-v2 candidate passes C1"
