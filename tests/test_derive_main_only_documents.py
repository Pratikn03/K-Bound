"""Regression tests for lossless prefix derivation and unsafe boundary rejection."""

from __future__ import annotations

import importlib.util
import io
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree

SCRIPT = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/derive_main_only_documents.py"
SPEC = importlib.util.spec_from_file_location("derive_main_only_documents", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def pdf_bytes(pages: list[tuple[str, int]]) -> bytes:
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    page_ids = []
    for text, y in pages:
        page_id = len(objects) + 1
        page_ids.append(page_id)
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands = f"BT /F1 12 Tf 40 {y} Td ({escaped}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>".encode()
        )
        objects.append(b"<< /Length " + str(len(commands)).encode() + b" >>\nstream\n" + commands + b"\nendstream")
    objects[1] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{' '.join(f'{n} 0 R' for n in page_ids)}] >>".encode()
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(output)


def docx_bytes(*, duplicate: bool = False, cross_boundary: bool = False, references: bool = True) -> bytes:
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
    heading = '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>{}</w:t></w:r></w:p>'
    appendix = heading.format("A " + MODULE.APPENDIX_HEADING)
    body = "<w:p><w:r><w:t>Main paper</w:t></w:r><m:oMath><m:r><m:t>x</m:t></m:r></m:oMath><w:drawing><a:blip/></w:drawing></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Table value</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
    if cross_boundary:
        body = '<w:bookmarkStart w:id="7" w:name="crossing"/>' + body
    if references:
        body += heading.format("References")
    body += '<w:p><w:pPr><w:numPr><w:numId w:val="10"/></w:numPr></w:pPr><w:r><w:t>Last citation</w:t></w:r></w:p>'
    body += '<w:bookmarkStart w:id="8" w:name="appendix"/>' + appendix
    body += '<w:p><w:r><w:t>Appendix content</w:t></w:r><m:oMath><m:r><m:t>y</m:t></m:r></m:oMath></w:p><w:bookmarkEnd w:id="8"/>'
    if duplicate:
        body += appendix
    if cross_boundary:
        body += '<w:bookmarkEnd w:id="7"/>'
    body += '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/></w:sectPr>'
    result = io.BytesIO()
    with ZipFile(result, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", f"<w:document {ns}><w:body>{body}</w:body></w:document>")
        for member in (
            "word/styles.xml",
            "word/numbering.xml",
            "word/_rels/document.xml.rels",
            "word/media/image1.png",
        ):
            archive.writestr(member, b"unchanged fixture " + member.encode())
    return result.getvalue()


class MainOnlyDerivationTests(unittest.TestCase):
    def test_pdf_keeps_every_page_through_references(self) -> None:
        source = pdf_bytes(
            [("Main", 750), ("References", 750), ("Final reference", 750), ("A " + MODULE.APPENDIX_HEADING, 750)]
        )
        output, record = MODULE.derive_pdf(source)
        self.assertEqual(record["output_pages"], 3)
        self.assertEqual(record["appendix_start_page_one_based"], 4)
        self.assertTrue(output.startswith(b"%PDF-"))
        self.assertTrue(record["retained_page_text_and_rasters_unchanged"])

    def test_pdf_rejects_appendix_below_final_main_text(self) -> None:
        source = pdf_bytes([("References", 750), ("Last main paragraph. A " + MODULE.APPENDIX_HEADING, 700)])
        with self.assertRaisesRegex(ValueError, "fresh page"):
            MODULE.derive_pdf(source)

    def test_pdf_rejects_low_heading_even_without_prior_text(self) -> None:
        source = pdf_bytes([("References", 750), ("A " + MODULE.APPENDIX_HEADING, 300)])
        with self.assertRaisesRegex(ValueError, "top of"):
            MODULE.derive_pdf(source)

    def test_pdf_rejects_ambiguous_appendix(self) -> None:
        source = pdf_bytes(
            [("References", 750), ("A " + MODULE.APPENDIX_HEADING, 750), ("A " + MODULE.APPENDIX_HEADING, 750)]
        )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            MODULE.derive_pdf(source)

    def test_docx_preserves_math_images_tables_numbering_and_package_parts(self) -> None:
        source = docx_bytes()
        output, record = MODULE.derive_docx(source)
        self.assertEqual(record["output_counts"]["math_objects"], 1)
        self.assertEqual(record["output_counts"]["tables"], 1)
        self.assertEqual(record["output_counts"]["drawings"], 1)
        self.assertEqual(record["output_counts"]["numbered_paragraphs"], 1)
        with ZipFile(io.BytesIO(source)) as before, ZipFile(io.BytesIO(output)) as after:
            for name in before.namelist():
                if name != "word/document.xml":
                    self.assertEqual(before.read(name), after.read(name))
            root = etree.fromstring(after.read("word/document.xml"))
            text = "".join(root.xpath(".//w:t/text()", namespaces=MODULE.NS))
            self.assertIn("Last citation", text)
            self.assertNotIn("Appendix content", text)
            self.assertEqual(root.xpath(".//w:bookmarkStart", namespaces=MODULE.NS), [])

    def test_docx_rejects_duplicate_heading(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            MODULE.derive_docx(docx_bytes(duplicate=True))

    def test_docx_rejects_missing_references(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            MODULE.derive_docx(docx_bytes(references=False))

    def test_docx_rejects_bookmark_crossing_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "unmatched bookmark"):
            MODULE.derive_docx(docx_bytes(cross_boundary=True))


if __name__ == "__main__":
    unittest.main()
