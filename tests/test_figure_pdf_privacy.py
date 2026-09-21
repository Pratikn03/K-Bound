"""Native PDF regressions for anonymous figure exports and preservation."""

from pathlib import Path
import subprocess

import pytest


def figure(tmp_path: Path, *, body: str = "Example figure", annotations: bool = False) -> Path:
    annotation = (
        r"\pdfannot width 20pt height 10pt depth 0pt {/Subtype /Link /A << /S /URI /URI (https://example.com) >>}"
        if annotations
        else ""
    )
    tex = (
        r"""\pdfoutput=1
\pdfobjcompresslevel=0
\pdfpagewidth=160bp
\pdfpageheight=80bp
\pdfhorigin=0pt
\pdfvorigin=0pt
\pdfinfo{/Author (Example Person) /Creator (Figure exporter)}
\font\testfont=cmr10
\setbox0=\hbox{\testfont """
        + annotation
        + body
        + r"""}
\shipout\box0
\end
"""
    )
    (tmp_path / "figure.tex").write_text(tex)
    subprocess.run(
        ["pdflatex", "-fmt=pdftex", "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", "figure.tex"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path / "figure.pdf"


def test_figure_metadata_scrub_preserves_text_pixels_and_is_idempotent(tmp_path):
    from docs.research.kbound.scripts.sanitize_figure_pdfs import pixels, run, sanitize
    from docs.research.kbound.scripts.build_anonymous_supplement import verify_pdf_anonymity

    source = figure(tmp_path)
    before = source.read_bytes()
    before_text = run(["pdftotext", str(source), "-"])
    before_pixels = pixels(source, tmp_path / "before")
    assert sanitize(source)
    assert source.read_bytes() != before
    assert run(["pdftotext", str(source), "-"]) == before_text
    assert pixels(source, tmp_path / "after") == before_pixels
    verify_pdf_anonymity("figure.pdf", source.read_bytes())
    sanitized = source.read_bytes()
    assert not sanitize(source)
    assert source.read_bytes() == sanitized


def test_visible_private_content_is_not_hidden_or_published(tmp_path):
    from docs.research.kbound.scripts.sanitize_figure_pdfs import sanitize
    from docs.research.kbound.scripts.release_privacy import PrivacyError

    source = figure(tmp_path, body="/Users/alice/private-input.csv")
    before = source.read_bytes()
    with pytest.raises(PrivacyError):
        sanitize(source)
    assert source.read_bytes() == before


@pytest.mark.parametrize("pdf_name", [b"/Annots ", b"/Annots\x00", b"/An#6eots\x00"])
def test_figure_annotations_are_rejected_without_replacement(tmp_path, pdf_name):
    from docs.research.kbound.scripts.sanitize_figure_pdfs import sanitize

    source = figure(tmp_path, annotations=True)
    original = source.read_bytes()
    assert b"/Annots" in original
    source.write_bytes(original.replace(b"/Annots", pdf_name))
    before = source.read_bytes()
    with pytest.raises(ValueError, match="annotations|object streams"):
        sanitize(source)
    assert source.read_bytes() == before
