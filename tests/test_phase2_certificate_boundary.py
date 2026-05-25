"""Phase 2.2B — risk-dominance + switching certificate code remains
behaviourally correct under the Phase-2.G tests (already passing) and
the boundary phrase "retrospective evaluation certificate" appears in
the driver source."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "src" / "scripts" / "run_phase2_certificate_audit.py"


def test_certificate_driver_states_retrospective_boundary_in_source():
    src = DRIVER.read_text()
    assert "retrospective" in src.lower(), (
        "certificate driver must state the retrospective-evaluation boundary"
    )


def test_certificate_driver_does_not_promise_production_safety():
    """Every occurrence of a deployment/production-safety phrase in the
    driver source must be in a negating context — the boundary notice
    explicitly disclaims these properties."""
    import re
    src = DRIVER.read_text()
    forbidden = (
        "production safety",
        "deployment guarantee",
        "real-world deployment",
        "clinical deployment",
    )
    negation_pat = re.compile(r"\b(not|no|does\s+not|cannot)\b", re.IGNORECASE)
    for phrase in forbidden:
        for m in re.finditer(re.escape(phrase), src, re.IGNORECASE):
            window = src[max(0, m.start() - 80): m.end() + 10]
            assert negation_pat.search(window), (
                f"phrase {phrase!r} appears without negation in driver source: {window!r}"
            )


def test_certificate_code_is_importable():
    from elara.certification import (
        estimate_risk_dominance,
        fired_subset_certificate,
        paired_bootstrap_lcb,
    )
    assert callable(estimate_risk_dominance)
    assert callable(fired_subset_certificate)
    assert callable(paired_bootstrap_lcb)
