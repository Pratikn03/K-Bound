from __future__ import annotations

from pathlib import Path


def test_elara_full_form_matches_system_purpose():
    expected = "Evidence-Layered Anomaly Reliability Architecture"
    for path in [
        Path("README.md"),
        Path("docs/research/PAPER_DRAFT_v1.tex"),
        Path("docs/research/THESIS_CHAPTER_v1.tex"),
        Path("src/uais/fusion/attention/__init__.py"),
    ]:
        assert expected in path.read_text(encoding="utf-8")


def test_rga_full_form_remains_reliability_gated_attention():
    expected = "Reliability-Gated Attention"
    for path in [
        Path("README.md"),
        Path("docs/research/PAPER_DRAFT_v1.tex"),
        Path("docs/research/THESIS_CHAPTER_v1.tex"),
        Path("src/uais/fusion/attention/__init__.py"),
    ]:
        assert expected in path.read_text(encoding="utf-8")
