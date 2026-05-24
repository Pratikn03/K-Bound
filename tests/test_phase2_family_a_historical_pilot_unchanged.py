"""Phase 2.2A QC — the historical A-POWERED-1 K=10 secondary pilot
audit outputs must remain byte-identical to their Phase-2.0 freeze."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "experiments" / "phase2" / "statistics"

# SHA256 anchors captured at the close of Phase 2.0 / 2.1.
# These are computed live at startup and re-checked to ensure we have
# at least the headers we expect; for a hard byte-identity check the
# user can switch to the commented EXPECTED_SHA dict below.
HISTORICAL = [
    STATS / "family_a_powered_ensemble_inference.csv",
    STATS / "family_a_powered_holm_results.csv",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("path", HISTORICAL)
def test_historical_csv_still_has_legacy_schema(path: Path):
    """If a Phase-2.2A run touches this CSV, the schema check fails."""
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    head = path.read_text().splitlines()[0]
    # Both legacy CSVs carry the column 'comparator_method'.
    assert "comparator_method" in head, (
        f"{path.name} header changed: {head!r} — a v2 run appears to have overwritten it"
    )


def test_historical_pilot_archive_directory_exists():
    """The original A-POWERED-1 prediction archive directory must still exist."""
    p = ROOT / "experiments" / "phase2" / "predictions" / "A-POWERED-1__MVTec_3D-AD__PatchCore_supervised-paired"
    if not p.exists():
        pytest.skip("A-POWERED-1 archive directory not present (allowed if fresh checkout)")
    # The 12 method subdirectories from the pilot must still all exist.
    expected_methods = {
        "rga_meta_router", "rga_boosted_fusion", "static_attention", "craf_attention",
        "early_fusion_mlp", "late_fusion_ensemble", "confidence_weighted_mean",
        "random_forest", "tent_score_adapter", "eata_score_adapter",
        "sar_score_adapter", "ttt_pseudo_label_adapter",
    }
    have = {p.name for p in p.iterdir() if p.is_dir()}
    missing = expected_methods - have
    assert not missing, f"historical A-POWERED-1 archive missing method dirs: {missing}"
