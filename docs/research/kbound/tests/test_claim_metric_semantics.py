"""Claim metric semantics — FA_u vs FA_c, beta vs epsilon."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

KBOUND = Path(__file__).resolve().parents[1]


def test_fa_c_not_claimed_as_theorem_bounded():
  ledger = json.loads((KBOUND / "claim_ledger.json").read_text())
  fa_c = next(c for c in ledger["claims"] if c["claim_id"] == "KB-CLAIM-004")
  assert fa_c["status"] == "withdrawn"
  assert "FA_c" in fa_c["claim_text"]


def test_fa_u_theorem_claim_supported():
  ledger = json.loads((KBOUND / "claim_ledger.json").read_text())
  fa_u = next(c for c in ledger["claims"] if c["claim_id"] == "KB-CLAIM-003")
  assert fa_u["status"] == "supported"
  assert "FA_u" in fa_u["claim_text"] or "false-adapt" in fa_u["claim_text"].lower()


def test_jackknife_plus_not_claimed_for_stress_grid():
  ledger = json.loads((KBOUND / "claim_ledger.json").read_text())
  jk = next(c for c in ledger["claims"] if c["claim_id"] == "KB-CLAIM-012")
  assert jk["status"] == "withdrawn"
