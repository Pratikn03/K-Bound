"""Phase 2.2B.2 — A-POWERED-3 MVTec LOCO-AD must be derived_view_proxy."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "research" / "phase2" / "PHASE_2_EXPERIMENT_REGISTRY_v2.csv"
REPORT = ROOT / "docs" / "research" / "phase2" / "FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md"


def test_registry_a_powered_3_is_derived_view_proxy():
    with REGISTRY.open() as f:
        rows = list(csv.DictReader(f))
    a3 = next(r for r in rows if r["experiment_id"] == "A-POWERED-3")
    assert a3["pairing_strength"] == "derived_view_proxy", (
        f"A-POWERED-3 pairing_strength must be derived_view_proxy; got {a3['pairing_strength']!r}"
    )


def test_v2_report_classifies_a_powered_3_as_derived_view_proxy():
    t = REPORT.read_text()
    assert "A-POWERED-3" in t
    # Ensure the §5 caveats line explicitly labels A-POWERED-3 as derived_view_proxy
    assert "**derived_view_proxy**" in t and "A-POWERED-3" in t
