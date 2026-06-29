"""Calibration split integrity — edge real protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
EDGE_RESULTS = REPO / "docs" / "experiments" / "kbound" / "results" / "edge_real_phone_v1"


def test_edge_calibration_sessions_disjoint():
  cal = json.loads((EDGE_RESULTS / "calibration_summary.json").read_text())
  fit = set(cal.get("fit_sessions", []))
  conf = set(cal.get("conformal_sessions", []))
  assert fit.isdisjoint(conf), "fit and conformal sessions must be disjoint"
  assert fit, "fit sessions required"
  assert conf, "conformal sessions required"


def test_edge_split_audit_seals_before_heldout():
  audit = json.loads((EDGE_RESULTS / "split_audit.json").read_text())
  sealed = audit.get("sealed_splits", {})
  assert sealed.get("calibration_conformal") is True
