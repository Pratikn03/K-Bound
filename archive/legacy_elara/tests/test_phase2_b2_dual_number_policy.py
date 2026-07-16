"""Phase 2.2B.2 — B2 dual-number policy invariants."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B2_CSV = ROOT / "experiments" / "phase2" / "mechanism" / "b2_phase1_vs_phase2_comparability.csv"
POLICY = ROOT / "docs" / "research" / "phase2" / "PHASE_2_B1_B2_INTEGRATION_POLICY.md"


def test_b2_comparability_csv_has_both_phase1_and_phase2_rows():
    with B2_CSV.open() as f:
        rows = list(csv.DictReader(f))
    phases = {r["phase"] for r in rows if r.get("endpoint", "").startswith("B2")}
    assert "Phase-1" in phases and "Phase-2" in phases


def test_policy_doc_locks_dual_number_wording():
    t = POLICY.read_text()
    assert "+0.0319" in t and "+0.0939" in t
    assert "COMPARABLE_BUT_ESTIMATOR_CHANGED_POSITIVE_RESULT" in t


def test_policy_doc_forbids_replacement_phrases():
    t = POLICY.read_text()
    # Forbidden phrases must appear inside the explicit "Forbidden wording" list,
    # not as bare assertions. Crude check: the policy doc contains the marker.
    assert "Forbidden wording" in t
    assert "B2 exact replication" in t  # listed as forbidden
