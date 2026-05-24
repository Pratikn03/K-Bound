"""Phase 2.2A QC — the v2 Family-A static-reference report must not
contain competitive-superiority phrasing."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "research" / "phase2" / "FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md"

FORBIDDEN_PHRASES = [
    "beats the best baseline",
    "beats every baseline",
    "strongest comparator",
    "best baseline",
    "ELARA is SOTA",
    "ELARA is universal",
    "production-ready",
    "deployment-ready",
    "clinically validated",
    "Proceed to Phase 3",
]

# These phrases are allowed but must be preceded by "not" / "no"
# / "cannot" within ~60 chars (e.g. "this is not a confirmatory test").
NEEDS_NEGATION_NEAR = [
    "strongest-baseline",
    "competitive superiority",
    "confirmatory",
]


def _live_prose(text: str) -> str:
    """Strip 'Forbidden interpretation' section so the test sees only
    live prose. The forbidden section is bounded by a heading that
    contains 'Forbidden' (case-insensitive) and ends at the next ##
    heading."""
    lines = text.splitlines()
    out, in_forbidden = [], False
    for ln in lines:
        if re.match(r"^\s*#{2,3}\s.*\bforbidden\b", ln, re.IGNORECASE):
            in_forbidden = True
            continue
        if in_forbidden and re.match(r"^\s*#{2,3}\s", ln):
            in_forbidden = False
        if not in_forbidden:
            out.append(ln)
    return "\n".join(out)


def test_report_contains_no_forbidden_competitive_phrase():
    if not REPORT.exists():
        return
    t = _live_prose(REPORT.read_text())
    offenders = [p for p in FORBIDDEN_PHRASES if p in t]
    assert not offenders, (
        f"v2 static-reference report contains forbidden phrases in live prose: {offenders}"
    )


def test_report_negates_phrases_that_require_negation():
    """Each occurrence of a needs-negation phrase must be in a negating
    context. Accepted negating tokens within ~120 chars before/after:
    'not', 'no', 'does not', 'cannot', 'only' (e.g. 'only Family D may
    use confirmatory'), 'reserved for'."""
    if not REPORT.exists():
        return
    t = _live_prose(REPORT.read_text())
    negation_pat = re.compile(
        r"\b(not|no|does\s+not|cannot|only|reserved\s+for)\b", re.IGNORECASE
    )
    for phrase in NEEDS_NEGATION_NEAR:
        for m in re.finditer(re.escape(phrase), t):
            window = t[max(0, m.start() - 120): m.end() + 120]
            assert negation_pat.search(window), (
                f"phrase {phrase!r} appears without negation in live prose: {window!r}"
            )


def test_report_states_static_reference_audit_label_explicitly():
    if not REPORT.exists():
        return
    t = REPORT.read_text()
    assert "PRIMARY_FAMILY_A_CELL_LEVEL_STATIC_REFERENCE_AUDIT" in t
    assert "static-reference" in t.lower() or "static reference" in t.lower()


def test_report_states_not_confirmatory_explicitly():
    if not REPORT.exists():
        return
    t = REPORT.read_text().lower()
    assert any(s in t for s in (
        "not confirmatory",
        "not a confirmatory",
        "is not confirmatory",
    )), "report must explicitly state Family A is not confirmatory"
