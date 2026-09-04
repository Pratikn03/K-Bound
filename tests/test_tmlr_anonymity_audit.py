"""Behavioral tests for the anonymous TMLR PDF audit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/research/kbound/scripts/audit_tmlr_anonymity.py"


def _pdf_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_pdf(
    path: Path,
    *,
    body: str = "Anonymous authors",
    author: str = "",
    uri: str | None = None,
    bookmark: str | None = None,
) -> None:
    """Write a tiny valid PDF without depending on a PDF Python package."""

    content = f"BT /F1 12 Tf 72 720 Td ({_pdf_literal(body)}) Tj ET\n"
    catalog_extra = " /Outlines 7 0 R" if bookmark is not None else ""
    page_extra = " /Annots [6 0 R]" if uri is not None else ""
    objects: dict[int, bytes] = {
        1: f"<< /Type /Catalog /Pages 2 0 R{catalog_extra} >>".encode(),
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R"
            f"{page_extra} >>"
        ).encode(),
        4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        5: f"<< /Length {len(content.encode())} >>\nstream\n{content}endstream".encode(),
    }
    if uri is not None:
        objects[6] = (
            "<< /Type /Annot /Subtype /Link /Rect [0 0 10 10] /Border [0 0 0] "
            f"/A << /S /URI /URI ({_pdf_literal(uri)}) >> >>"
        ).encode()
    if bookmark is not None:
        objects[7] = b"<< /Type /Outlines /First 8 0 R /Last 8 0 R /Count 1 >>"
        objects[8] = (f"<< /Title ({_pdf_literal(bookmark)}) /Parent 7 0 R /Dest [3 0 R /Fit] >>").encode()
    info_id = max(objects) + 1
    objects[info_id] = (
        f"<< /Title (Anonymous test manuscript) /Author ({_pdf_literal(author)}) /Subject () /Keywords () >>"
    ).encode()

    pdf = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for object_id in range(1, info_id + 1):
        if object_id not in objects:
            objects[object_id] = b"null"
        offsets[object_id] = len(pdf)
        pdf.extend(f"{object_id} 0 obj\n".encode())
        pdf.extend(objects[object_id])
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {info_id + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for object_id in range(1, info_id + 1):
        pdf.extend(f"{offsets[object_id]:010d} 00000 n \n".encode())
    pdf.extend(
        (
            f"trailer\n<< /Size {info_id + 1} /Root 1 0 R /Info {info_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    path.write_bytes(pdf)


def _run(pdf: Path, report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDIT), "--pdf", str(pdf), "--report", str(report)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_anonymous_pdf_with_scholarly_link_passes(tmp_path: Path) -> None:
    pdf = tmp_path / "anonymous.pdf"
    report = tmp_path / "report.md"
    _write_pdf(
        pdf,
        body="Anonymous authors. See supplemental material.",
        uri="https://doi.org/10.1000/example",
        bookmark="Methods",
    )

    result = _run(pdf, report)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Overall result: **PASS**" in report.read_text(encoding="utf-8")


def test_named_author_metadata_and_personal_repository_url_fail(tmp_path: Path) -> None:
    pdf = tmp_path / "identified.pdf"
    report = tmp_path / "report.md"
    _write_pdf(
        pdf,
        author="Pratik Niroula",
        uri="https://github.com/Pratikn03/K-Bound",
    )

    result = _run(pdf, report)

    output = report.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert "Overall result: **FAIL**" in output
    assert "Pratik Niroula" in output
    assert "personal repository or forge URL" in output


def test_identifier_present_only_in_bookmark_object_fails(tmp_path: Path) -> None:
    pdf = tmp_path / "bookmark.pdf"
    report = tmp_path / "report.md"
    _write_pdf(pdf, bookmark="Appendix by Niroula")

    result = _run(pdf, report)

    assert result.returncode == 1
    output = report.read_text(encoding="utf-8")
    assert "Overall result: **FAIL**" in output
    assert "Niroula" in output
    assert "raw PDF objects" in output
