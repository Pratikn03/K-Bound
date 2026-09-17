"""Calibration split integrity -- edge real protocol.

Fix-queue item 8 (test half).  This guard used to point at
``REPO / "docs" / "experiments" / "kbound" / "results" / "edge_real_phone_v1"``,
a path that cannot exist in this tree: the results root is ``REPO/experiments``,
not ``REPO/docs/experiments``.  Every run therefore ended in ``FileNotFoundError``
on a path no one had ever created, which is a broken guard rather than a failing
one -- it told the reader nothing about whether the split is actually clean.

The path is now correct, and the two checks ``skipif`` on the *specific* artifact
each one needs.  A skip here means "the edge artifact is not in this release, so
this property is unverified", which is an honest report; it must not be read as
"the split was checked and is clean".  Set ``KBOUND_REQUIRE_EDGE_ARTIFACTS=1`` to
turn the skips into hard failures for a release gate that does have the
artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# tests/ -> kbound/ -> research/ -> docs/ -> REPO
REPO = Path(__file__).resolve().parents[4]
EDGE_RESULTS = REPO / "experiments" / "kbound" / "results" / "edge_real_phone_v1"

CALIBRATION_SUMMARY = EDGE_RESULTS / "calibration_summary.json"
SPLIT_AUDIT = EDGE_RESULTS / "split_audit.json"

_REQUIRE = os.environ.get("KBOUND_REQUIRE_EDGE_ARTIFACTS", "") not in ("", "0", "false", "False")


def _unusable(path: Path) -> str | None:
    """Return a reason string if ``path`` cannot be read, else ``None``.

    Handles the three ways an artifact is missing in this tree: absent, empty,
    and NUL-filled (an unmaterialised iCloud placeholder -- 142 tracked text
    artifacts are in that state, and a whitespace-only check does not catch it).
    """
    if not path.exists():
        return f"{path} is absent from this release"
    try:
        raw = path.read_bytes()
    except OSError as exc:  # unmaterialised placeholder can raise on read
        return f"{path} is unreadable ({exc.__class__.__name__}: {exc})"
    if len(raw) == 0:
        return f"{path} is empty (0 bytes)"
    if b"\x00" in raw:
        return f"{path} is a NUL-filled placeholder ({len(raw)} bytes, not materialised)"
    return None


def _load(path: Path) -> dict:
    reason = _unusable(path)
    if reason is not None:
        if _REQUIRE:
            pytest.fail(f"KBOUND_REQUIRE_EDGE_ARTIFACTS is set but {reason}")
        pytest.skip(f"edge artifact unavailable, split integrity UNVERIFIED: {reason}")
    return json.loads(path.read_text())


def test_edge_results_dir_is_on_the_real_results_root():
    """The results root is REPO/experiments, never REPO/docs/experiments.

    This assertion has no artifact dependency, so it runs unconditionally and
    catches a regression of the original path bug even in a release where the
    edge artifacts are absent.
    """
    assert EDGE_RESULTS.name == "edge_real_phone_v1"
    assert EDGE_RESULTS.parents[2] == REPO / "experiments", (
        f"edge results must live under {REPO / 'experiments'}, got {EDGE_RESULTS}"
    )
    assert not str(EDGE_RESULTS).startswith(str(REPO / "docs" / "experiments"))


def test_edge_calibration_sessions_disjoint():
    cal = _load(CALIBRATION_SUMMARY)
    fit = set(cal.get("fit_sessions", []))
    conf = set(cal.get("conformal_sessions", []))
    assert fit.isdisjoint(conf), "fit and conformal sessions must be disjoint"
    assert fit, "fit sessions required"
    assert conf, "conformal sessions required"


def test_edge_split_audit_seals_before_heldout():
    audit = _load(SPLIT_AUDIT)
    sealed = audit.get("sealed_splits", {})
    assert sealed.get("calibration_conformal") is True
