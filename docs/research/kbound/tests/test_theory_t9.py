"""Theorem-9 clean-transfer ceiling: headroom and sub-optimality-gap identities."""
from theory.t9_clean_transfer_ceiling import ceiling_headroom, suboptimality_gap

def test_ceiling_headroom():
    assert abs(ceiling_headroom(0.9) - 0.1) < 1e-9

def test_suboptimality_gap_clips_and_computes():
    assert abs(suboptimality_gap(0.95, 0.90) - 0.05) < 1e-9
    assert suboptimality_gap(0.80, 0.90) == 0.0     # clipped at 0
