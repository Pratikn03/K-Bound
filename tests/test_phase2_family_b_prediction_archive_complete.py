"""Phase 2.2B — Family-B prediction archives (if present) must have the
required Family-B columns populated."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
B_MECH_1 = ROOT / "experiments" / "phase2" / "mechanism" / "b_mech_1_prediction_archives"


def test_b_mech_1_archive_directory_either_absent_or_well_formed():
    if not B_MECH_1.exists():
        pytest.skip("B-MECH-1 archive not yet produced (infrastructure-only phase)")
    # If archives exist, every test parquet must carry the failure_type column.
    import pandas as pd
    for cell_dir in B_MECH_1.iterdir():
        # macOS AppleDouble files (._foo) are treated as not-a-directory; skip them
        if cell_dir.name.startswith("._") or not cell_dir.is_dir():
            continue
        for method_dir in cell_dir.iterdir():
            if method_dir.name.startswith("._") or not method_dir.is_dir():
                continue
            for split_dir in method_dir.iterdir():
                if split_dir.name.startswith("._") or not split_dir.is_dir():
                    continue
                for p in split_dir.glob("seed_*.parquet"):
                    if p.name.startswith("._"):
                        continue
                    df = pd.read_parquet(p)
                    assert "failure_type_if_applicable" in df.columns
                    assert "failed_domain_count_if_applicable" in df.columns
                    assert "fault_severity_if_applicable" in df.columns
                    assert "gate_mode" in df.columns
                    assert "selection_used_test_metrics" in df.columns
                    assert (df["selection_used_test_metrics"] == False).all()  # noqa: E712
