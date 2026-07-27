"""The committed assumption reports must match what the emitter produces.

A report that has drifted from `NUMBERS_PACK.json` is worse than no report: it looks
like a machine-checked provenance record while carrying a hand-edited number. This
test is the thing that keeps that from happening quietly.

It shells out rather than importing the emitter, because the emitter is deliberately
free of `kga` imports (it must run without numpy) and shelling out is the same path a
release check would take.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EMITTER = REPO / "scripts" / "emit_assumption_reports.py"
PACK = REPO / "docs/research/kbound/panel_review_2026-07-25/NUMBERS_PACK.json"
REPORTS = REPO / "research_lock" / "assumption_reports"

pytestmark = pytest.mark.skipif(
    not (EMITTER.exists() and PACK.exists()),
    reason="assumption-report emitter or NUMBERS_PACK not present in this checkout",
)


def test_committed_reports_match_the_emitter():
    r = subprocess.run(
        [sys.executable, str(EMITTER), "--check"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert r.returncode == 0, (
        "committed assumption reports are stale; re-run "
        f"`python scripts/emit_assumption_reports.py`\n{r.stdout}{r.stderr}"
    )


def test_no_report_claims_theoretical_coverage():
    """The paper states this as a fact about every track. Keep it true."""
    paths = sorted(REPORTS.glob("*.assumption_report.json"))
    assert paths, "no assumption reports found"
    for p in paths:
        d = json.loads(p.read_text())
        assert d["theoretical_coverage_claimed"] is False, p.name
        assert d["coverage_type"] != "theoretical", p.name


def test_every_report_carries_provenance_and_limitations():
    for p in sorted(REPORTS.glob("*.assumption_report.json")):
        d = json.loads(p.read_text())
        assert d["provenance"], f"{p.name} has no provenance block"
        for eid, meta in d["provenance"].items():
            assert meta.get("method"), f"{p.name}: entry {eid} has no method string"
            assert meta.get("artifact_paths"), f"{p.name}: entry {eid} has no artifacts"
        assert d["limitations"], f"{p.name} lists no limitations"


def test_fallback_action_matches_the_gate():
    ladder = {
        "certify": "adapt_freeze_abstain",
        "restricted": "freeze_or_abstain",
        "diagnostic_only": "none",
        "reject": "none",
    }
    for p in sorted(REPORTS.glob("*.assumption_report.json")):
        d = json.loads(p.read_text())
        assert d["fallback_action"] == ladder[d["deployment_gate"]], p.name
