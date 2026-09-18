"""Fail-closed guards for tiny smoke panels in the released decision driver."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "docs" / "research" / "kbound" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from kbound_decide import decide_kga  # noqa: E402


def test_single_cell_smoke_panel_fails_closed_without_fit_crash() -> None:
    """A one-cell smoke run has no leave-one-out fit/calibration pool."""
    z = np.zeros((1, 11), dtype=float)
    b = np.asarray([-0.1], dtype=float)
    bhat, epsilon, decision = decide_kga(z, b, alpha=0.1)
    assert bhat.shape == (1,)
    assert epsilon.shape == (1,)
    assert math.isinf(float(epsilon[0]))
    assert decision.tolist() == ["ABSTAIN"]
