"""Claim metric semantics — commitment error, FA_c, beta, and epsilon."""

from __future__ import annotations

import json
from pathlib import Path

KBOUND = Path(__file__).resolve().parents[1]


def test_fa_c_not_claimed_as_theorem_bounded():
    ledger = json.loads((KBOUND / "claim_ledger.json").read_text())
    fa_c = next(c for c in ledger["claims"] if c["claim_id"] == "KB-CLAIM-004")
    assert fa_c["status"] == "withdrawn"
    assert "FA_c" in fa_c["claim_text"]


def test_supported_commitment_error_claim_uses_current_semantics():
    ledger = json.loads((KBOUND / "claim_ledger.json").read_text())
    certificate = next(c for c in ledger["claims"] if c["claim_id"] == "KB-CLAIM-003")
    claim_text = certificate["claim_text"].lower()

    assert certificate["status"] == "supported"
    assert "fixed scalar benefit target" in claim_text
    assert "evaluation unit" in claim_text
    assert "valid marginal interval coverage" in claim_text
    assert "unconditional wrong-direction commitment probability" in claim_text
    assert "at most alpha" in claim_text
    assert "does not identify" in claim_text
    assert "observed batch outcome" in claim_text
    assert "population risk" in claim_text


def test_jackknife_plus_not_claimed_for_stress_grid():
    ledger = json.loads((KBOUND / "claim_ledger.json").read_text())
    jk = next(c for c in ledger["claims"] if c["claim_id"] == "KB-CLAIM-012")
    assert jk["status"] == "withdrawn"
