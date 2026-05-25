"""Phase 2.2B.2 — Master status checklist consistency."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "docs" / "research" / "phase2" / "PHASE_2_MASTER_STATUS_CHECKLIST.md"


def test_master_exists():
    assert MASTER.exists()


def test_master_a_powered_3_is_now_derived_view_proxy():
    if not MASTER.exists():
        pytest.skip("master file not present")
    t = MASTER.read_text()
    # Find every line that names A-POWERED-3 plus a pairing strength
    for line in t.splitlines():
        if "A-POWERED-3" in line and "independent_modalities" in line:
            raise AssertionError(
                f"master file row for A-POWERED-3 still says independent_modalities: {line!r}"
            )


def test_master_b1_b2_wording_uses_estimator_change_label():
    if not MASTER.exists():
        pytest.skip("master file not present")
    t = MASTER.read_text()
    # Must not bare-claim "B-MECH-1 REPRODUCED ×2"; must reference estimator change for B2
    assert "REPRODUCED × 2" not in t and "reproduced × 2" not in t
    # B2 wording must reference estimator change or explicit dual-number form
    assert "COMPARABLE_BUT_ESTIMATOR_CHANGED" in t


def test_master_rga_v2_status_is_executed_not_promoted_or_equivalent():
    if not MASTER.exists():
        pytest.skip("master file not present")
    t = MASTER.read_text()
    assert "RGA_V2_EXECUTED_NOT_PROMOTED" in t or "NOT_IMPROVED" in t
