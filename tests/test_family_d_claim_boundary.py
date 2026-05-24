"""Phase 2.1 — Family-D claim boundary must not state that Family-D
success removes the audited-reanalysis status of Family A."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "docs" / "research" / "phase2"


def _docs_to_check() -> list[Path]:
    """Every doc that discusses the Family-D claim boundary."""
    out = []
    for p in PHASE2.glob("FAMILY_D_*"):
        if p.suffix in {".md", ".json", ".csv"}:
            out.append(p)
    for name in (
        "PHASE_2_RESEARCH_CONTRACT_v2.md",
        "PHASE_2_STATISTICAL_POLICY_v2.md",
        "FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT_v2.md",
        "PHASE_2_INTERIM_REPORT_v2.md",
    ):
        out.append(PHASE2 / name)
    return [p for p in out if p.exists()]


FORBIDDEN_CLAIM_PATTERNS = [
    re.compile(r"remov\w+\s+the\s+audited[-\s]reanalysis", re.IGNORECASE),
    re.compile(r"remov\w+\s+the\s+['\"]audited\s+reanalysis['\"]", re.IGNORECASE),
    re.compile(r"family[-\s]?d\s+success\s+removes\s+family[-\s]?a", re.IGNORECASE),
]


def test_no_doc_states_family_d_success_removes_family_a_audited_status():
    offenders = []
    for p in _docs_to_check():
        # The v1 contract may contain this statement (it's what we're invalidating);
        # skip the v1 contract because it is preserved unchanged as historical record.
        if p.name == "FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md":
            continue
        t = p.read_text()
        for pat in FORBIDDEN_CLAIM_PATTERNS:
            if pat.search(t):
                offenders.append((p.name, pat.pattern))
    assert not offenders, f"forbidden Family-D claim-boundary phrases found: {offenders}"


def test_v2_claim_boundary_is_documented_explicitly():
    """v2 design-status / contract-repair docs must explicitly limit what
    Family-D success may say."""
    p = PHASE2 / "FAMILY_D_V2_DESIGN_STATUS.md"
    t = p.read_text().lower()
    # at least one of the disallow-list phrases must be present
    require_any = (
        "does not remove",
        "must not",
        "may not",
        "not entitled",
    )
    assert any(s in t for s in require_any), (
        "v2 design status must explicitly state what Family-D success may not unlock"
    )
