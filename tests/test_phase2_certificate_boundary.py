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
    src = DRIVER.read_text().lower()
    for forbidden in (
        "production safety",
        "deployment guarantee",
        "real-world deployment",
        "clinical deployment",
    ):
        # The phrase may legally appear inside a negation like "is not a
        # production safety guarantee" — but bare positive use is forbidden.
        # The simplest invariant: the driver must not assert these in a
        # claim-y context, so we restrict the test to a positive-form
        # substring match. The B-CERT-1 driver currently doesn't quote
        # these phrases at all, which is the desired state.
        assert forbidden not in src, (
            f"driver source contains forbidden phrase {forbidden!r}"
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
