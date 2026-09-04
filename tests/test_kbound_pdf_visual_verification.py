from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from docs.research.kbound.scripts import (
    compare_pdf_renders,
    render_pdf_pages,
    verify_pdf_structure,
)


def test_renderer_uses_release_default_of_192_dpi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[list[str]] = []
    monkeypatch.setattr(render_pdf_pages, "page_count", lambda _pdf: 1)
    monkeypatch.setattr(render_pdf_pages, "require_binary", lambda name: f"/tools/{name}")

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        Path(command[-1] + "-1.png").write_bytes(b"rendered")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(render_pdf_pages.subprocess, "run", run)
    render_pdf_pages.render(tmp_path / "paper.pdf", tmp_path / "pages")

    assert observed == [
        [
            "/tools/pdftoppm",
            "-png",
            "-r",
            "192",
            str(tmp_path / "paper.pdf"),
            str(tmp_path / "pages/page"),
        ]
    ]


def test_renderer_rejects_nonpositive_dpi(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="DPI must be positive"):
        render_pdf_pages.render(tmp_path / "paper.pdf", tmp_path / "pages", dpi=0)


def test_render_comparison_indexes_mixed_zero_padding(tmp_path: Path) -> None:
    for name in ("page-1.png", "page-02.png", "page-003.png"):
        (tmp_path / name).write_bytes(b"png")

    assert compare_pdf_renders.page_inventory(tmp_path) == {
        1: tmp_path / "page-1.png",
        2: tmp_path / "page-02.png",
        3: tmp_path / "page-003.png",
    }


def test_render_comparison_rejects_duplicate_page_numbers(tmp_path: Path) -> None:
    (tmp_path / "page-1.png").write_bytes(b"one")
    (tmp_path / "page-01.png").write_bytes(b"two")

    with pytest.raises(ValueError, match="duplicate rendered page 1"):
        compare_pdf_renders.page_inventory(tmp_path)


def test_render_comparison_selects_first_largest_and_last_changed_pages() -> None:
    comparisons = [
        compare_pdf_renders.PageComparison(page=1, changed_pixels=5, total_pixels=100),
        compare_pdf_renders.PageComparison(page=2, changed_pixels=80, total_pixels=100),
        compare_pdf_renders.PageComparison(page=3, changed_pixels=0, total_pixels=100),
        compare_pdf_renders.PageComparison(page=4, changed_pixels=20, total_pixels=100),
    ]

    assert compare_pdf_renders.select_preview_pages(comparisons) == (1, 2, 4)


def test_structural_verifier_parses_pdfinfo_and_unembedded_fonts() -> None:
    info = verify_pdf_structure.parse_pdfinfo(
        "Pages: 22\nEncrypted: no\nPage size: 612 x 792 pts (letter)\n"
    )
    fonts = """\
name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
ABC+Embedded                        Type 1            Custom           yes yes yes    10  0
NotEmbedded                         Type 1            Custom           no  no  yes    11  0
"""

    assert info["Pages"] == "22"
    assert info["Encrypted"] == "no"
    assert verify_pdf_structure.unembedded_fonts(fonts) == ("NotEmbedded",)


def test_structural_report_records_exact_per_pdf_commands(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    tools = {
        "gs": "/tools/gs",
        "pdffonts": "/tools/pdffonts",
        "pdfinfo": "/tools/pdfinfo",
        "pdftotext": "/tools/pdftotext",
    }
    result = {
        "role": "paper",
        "pdf": "/release/paper.pdf",
        "status": "PASS",
        "pages": 2,
        "page_size": "612 x 792 pts (letter)",
        "all_fonts_embedded": True,
        "font_records": 3,
        "extracted_characters": 1001,
        "ghostscript_clean": True,
        "sha256": "a" * 64,
    }

    verify_pdf_structure._write_report([result], tools, report)

    rendered = report.read_text(encoding="utf-8")
    assert "`/tools/pdfinfo /release/paper.pdf`" in rendered
    assert "`/tools/pdffonts /release/paper.pdf`" in rendered
    assert "`/tools/pdftotext -layout /release/paper.pdf" in rendered
    assert "`/tools/gs -q -dSAFER -dNOPAUSE -dBATCH -sDEVICE=nullpage /release/paper.pdf`" in rendered
