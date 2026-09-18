"""Contracts for the three canonical K-Bound publication artifacts."""

from __future__ import annotations

from pathlib import Path

from docs.research.kbound.scripts import build_release_source_seal, render_pdf_pages, verify_release_checksums

ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "docs/research/kbound/scripts/build_pdfs.sh"
RELEASE_RUNBOOK = ROOT / "docs/research/kbound/runbooks/release_candidate.sh"

CANONICAL_ARTIFACTS = {
    "docs/research/kbound/kbound_short_final_draft.docx",
    "docs/research/kbound/kbound_short_final_draft.pdf",
    "docs/research/kbound/kbound_tmlr.pdf",
}


def test_release_inventory_contains_exactly_three_publication_artifacts() -> None:
    publication_artifacts = {
        path
        for path in verify_release_checksums.REQUIRED_RELEASE_PATHS
        if Path(path).parent.as_posix() == "docs/research/kbound" and Path(path).suffix in {".docx", ".pdf"}
    }

    assert publication_artifacts == CANONICAL_ARTIFACTS


def test_build_driver_has_no_missing_current_pdf_publisher_dependency() -> None:
    build = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert not (BUILD_SCRIPT.parent / "publish_current_pdfs.py").exists()
    assert "publish_current_pdfs.py" not in build
    assert "output/pdf" not in build


def test_release_pdf_mode_builds_only_the_three_canonical_artifacts() -> None:
    runbook = RELEASE_RUNBOOK.read_text(encoding="utf-8")
    pdf_step = runbook.split("step_pdf() {", 1)[1].split("\n}", 1)[0]

    assert "BUILD_LONG_TMLR=1" in pdf_step
    assert "BUILD_DOCX=1" in pdf_step
    for nonrelease_flag in (
        "BUILD_SHORT_MAIN=1",
        "BUILD_SHORT_SUPPLEMENT=1",
    ):
        assert nonrelease_flag not in pdf_step


def test_visual_release_check_covers_only_the_two_canonical_pdfs() -> None:
    assert render_pdf_pages.CURRENT_PDF_NAMES == (
        "kbound_short_final_draft.pdf",
        "kbound_tmlr.pdf",
    )


def test_source_seal_does_not_name_the_absent_pdf_publisher() -> None:
    all_sources = {path for paths in build_release_source_seal.EXPLICIT_FILES.values() for path in paths}

    assert "docs/research/kbound/scripts/publish_current_pdfs.py" not in all_sources
