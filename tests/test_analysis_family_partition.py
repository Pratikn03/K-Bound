"""Phase 1.D — analysis family partition is consistent across artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

REGISTRY = Path("experiments/audit/statistical_family_registry.csv")
INFERENCE = Path("experiments/audit/audited_ensemble_inference_results.csv")


@pytest.fixture(scope="module")
def registry():
    if not REGISTRY.exists():
        pytest.skip(f"family registry missing: {REGISTRY}")
    with REGISTRY.open() as f:
        return list(csv.DictReader(f))


def test_registry_has_family_a_b_c_partitions(registry):
    fams = sorted({r["analysis_family"] for r in registry})
    # We always expect Family A; Family B may live in a separate emit; Family C is C-prefixed cells.
    assert "A" in fams
    # Family C cells (Real3D, VisA noise-floor, UNSW held-out) must be present.
    assert "C" in fams


def test_family_A_has_5_confirmatory_and_3_diagnostic(registry):
    a_rows = [r for r in registry if r["analysis_family"] == "A"]
    confirmatory = [r for r in a_rows if r["analysis_status"] == "audited primary reanalysis"]
    diagnostic = [r for r in a_rows if r["analysis_status"] == "protocol-diagnostic"]
    assert len(confirmatory) == 5, f"expected 5 audited-primary Family A cells, got {len(confirmatory)}"
    assert len(diagnostic) == 3, f"expected 3 protocol-diagnostic Family A cells, got {len(diagnostic)}"


def test_holm_family_size_matches_confirmatory_count(registry):
    a_rows = [r for r in registry if r["analysis_family"] == "A"]
    confirmatory = [r for r in a_rows if r["analysis_status"] == "audited primary reanalysis"]
    if not confirmatory:
        pytest.skip("no confirmatory rows")
    # Every confirmatory row's holm_family_size should be 5.
    for r in confirmatory:
        assert (
            str(r.get("holm_family_size")) == "5"
        ), f"{r['cell_id']} has holm_family_size={r['holm_family_size']} but Family A confirmatory K=5."


def test_inference_rows_match_registry_status():
    if not INFERENCE.exists():
        pytest.skip("audited inference results missing")
    with INFERENCE.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        assert (
            r.get("claim_status") == "locked_audited_reanalysis"
        ), f"cell {r['cell_id']} claim_status={r['claim_status']!r} (must be 'locked_audited_reanalysis')"


def test_family_C_holm_size_is_zero(registry):
    for r in registry:
        if r["analysis_family"] == "C":
            assert (
                str(r.get("holm_family_size")) == "0"
            ), f"Family C cell {r['cell_id']} has holm_family_size={r['holm_family_size']} (must be 0)"
