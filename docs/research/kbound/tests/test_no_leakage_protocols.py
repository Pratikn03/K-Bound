"""Tests that protocol configs forbid test-label leakage paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
KBOUND = REPO / "docs" / "research" / "kbound"
LOCK = REPO / "research_lock"


@pytest.mark.parametrize("name", [
    "STRESS_GRID_STRICT_PROTOCOL_A_v2.yaml",
    "mixed_protocol_oof_v2.yaml",
    "assumption_audit_v1.yaml",
])
def test_strict_protocols_forbid_test_tuning(name: str):
  path = LOCK / name
  assert path.exists(), f"missing {path}"
  text = path.read_text()
  assert "forbidden" in text.lower()
  assert "test" in text.lower()


def test_claim_ledger_valid_json():
  p = KBOUND / "claim_ledger.json"
  data = json.loads(p.read_text())
  assert "claims" in data
  ids = [c["claim_id"] for c in data["claims"]]
  assert len(ids) == len(set(ids))


def test_results_source_uses_oof_not_in_sample():
  src = json.loads((KBOUND / "results_source.json").read_text())
  readme = src.get("_README", "")
  assert "OUT-OF-FOLD" in readme or "out-of-fold" in readme.lower()
  assert "in-sample" in readme.lower()
