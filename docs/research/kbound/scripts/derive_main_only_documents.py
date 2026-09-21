#!/usr/bin/env python3
"""Derive main-paper-only PDF and DOCX without rebuilding or rewriting their content.

The first appendix must have the known heading after References. PDF extraction
is permitted only when that heading is the first text on a fresh page. DOCX
body-prefix XML is preserved, as are all other ZIP members byte for byte. Unused
appendix media/relationships may remain in the package; no appendix body remains.
Run document authoring with the selected bundled document Python runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from lxml import etree

APPENDIX_HEADING = "A Rigorous Bridge from Cell Coverage to Population Coverage"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}
W = "{" + NS["w"] + "}"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(text: str) -> str:
    """Join printed line-end hyphenation, then normalize whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)).strip()


def heading_match(text: str) -> re.Match[str] | None:
    # TeX's appendix letter and the title's initial article can both be A.
    return re.match(r"^(?:A )?" + re.escape(APPENDIX_HEADING) + r"(?= |$)", normalize(text))


def poppler(command: list[str], *, cwd: Path) -> str:
    binary = shutil.which(command[0])
    if binary is None:
        raise RuntimeError(f"required Poppler executable missing: {command[0]}")
    completed = subprocess.run([binary, *command[1:]], cwd=cwd, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Poppler command failed: {command[0]}: {completed.stderr}")
    return completed.stdout


def pdf_text_pages(path: Path) -> list[str]:
    text = poppler(["pdftotext", "-raw", str(path), "-"], cwd=path.parent)
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return pages


def derive_pdf(source: bytes) -> tuple[bytes, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="kbound-main-only-") as temp_name:
        temp = Path(temp_name)
        source_path = temp / "source.pdf"
        source_path.write_bytes(source)
        metadata = poppler(["pdfinfo", str(source_path)], cwd=temp)
        count_match = re.search(r"(?m)^Pages:\s+(\d+)\s*$", metadata)
        if count_match is None:
            raise ValueError("PDF page count is unavailable")
        source_pages = int(count_match.group(1))
        texts = pdf_text_pages(source_path)
        if len(texts) != source_pages:
            raise ValueError("PDF text extraction page count is inconsistent")
        reference_pages = [i for i, text in enumerate(texts) if re.search(r"(?m)^\s*(?:\d+\s+)?References\s*$", text)]
        if len(reference_pages) != 1:
            raise ValueError("PDF requires one unambiguous References heading")
        candidates = [i for i, text in enumerate(texts) if heading_match(text)]
        if len(candidates) != 1:
            raise ValueError("PDF appendix heading must occur exactly once as the first text on a fresh page")
        cutoff = candidates[0]
        if cutoff <= reference_pages[0]:
            raise ValueError("PDF appendix must begin after the References page")
        if any(APPENDIX_HEADING in normalize(text) for text in texts[:cutoff]):
            raise ValueError("PDF appendix heading also occurs before the detected boundary")

        bbox = poppler(
            ["pdftotext", "-f", str(cutoff + 1), "-l", str(cutoff + 1), "-bbox", str(source_path), "-"], cwd=temp
        )
        geometry = etree.fromstring(bbox.encode(), etree.XMLParser(resolve_entities=False, no_network=True))
        pages = geometry.xpath("//*[local-name()='page']")
        words = geometry.xpath("//*[local-name()='word']")
        if len(pages) != 1 or not words or words[0].text != "A":
            raise ValueError("PDF appendix first text fragment is not the expected heading")
        first_y = float(words[0].get("yMin"))
        height = float(pages[0].get("height"))
        page_info = poppler(["pdfinfo", "-f", str(cutoff + 1), "-l", str(cutoff + 1), str(source_path)], cwd=temp)
        rotation = re.search(r"(?m)^Page\s+\d+ rot:\s+(-?\d+)\s*$", page_info)
        if rotation is None or int(rotation.group(1)) % 360 != 0 or first_y > 0.25 * height:
            raise ValueError("PDF appendix heading is not verified at the top of an unrotated fresh page")

        poppler(["pdfseparate", "-f", "1", "-l", str(cutoff), str(source_path), str(temp / "prefix-%d.pdf")], cwd=temp)
        output_path = temp / "main.pdf"
        poppler(
            ["pdfunite", *[str(temp / f"prefix-{i}.pdf") for i in range(1, cutoff + 1)], str(output_path)], cwd=temp
        )
        if pdf_text_pages(output_path) != texts[:cutoff]:
            raise RuntimeError("PDF retained page text changed")
        # Raster identity checks every retained page, including figures and equations.
        poppler(
            ["pdftoppm", "-f", "1", "-l", str(cutoff), "-r", "72", str(source_path), str(temp / "source-render")],
            cwd=temp,
        )
        poppler(["pdftoppm", "-r", "72", str(output_path), str(temp / "output-render")], cwd=temp)
        before = sorted(temp.glob("source-render-*.ppm"))
        after = sorted(temp.glob("output-render-*.ppm"))
        if len(before) != cutoff or len(after) != cutoff:
            raise RuntimeError("PDF raster verification has an incomplete page set")
        raster_hashes = []
        for i, (original, derived) in enumerate(zip(before, after, strict=True), 1):
            digest = sha256(original.read_bytes())
            if digest != sha256(derived.read_bytes()):
                raise RuntimeError(f"PDF rendered page changed: {i}")
            raster_hashes.append(digest)
        data = output_path.read_bytes()
    return data, {
        "source_pages": source_pages,
        "output_pages": cutoff,
        "retained_pages_one_based": [1, cutoff],
        "references_page_one_based": reference_pages[0] + 1,
        "appendix_start_page_one_based": cutoff + 1,
        "appendix_first_text_y_from_top_points": first_y,
        "appendix_page_height_points": height,
        "fresh_page_verified": True,
        "retained_page_text_and_rasters_unchanged": True,
        "raster_verification_dpi": 72,
        "retained_page_raster_sha256": raster_hashes,
        "note": "PDF structural metadata and internal link behavior may differ; retained page content is text- and raster-identical.",
    }


def xml_counts(element: Any) -> dict[str, int]:
    expressions = {
        "paragraphs": ".//w:p",
        "tables": ".//w:tbl",
        "math_objects": ".//m:oMath",
        "display_math_paragraphs": ".//m:oMathPara",
        "drawings": ".//w:drawing",
        "embedded_image_references": ".//a:blip",
        "numbered_paragraphs": ".//w:numPr",
    }
    return {name: len(element.xpath(expr, namespaces=NS)) for name, expr in expressions.items()}


def body_fingerprint(elements: list[Any]) -> str:
    return sha256(b"".join(etree.tostring(e, method="c14n") for e in elements))


def derive_docx(source: bytes) -> tuple[bytes, dict[str, Any]]:
    with ZipFile(io.BytesIO(source)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("DOCX package contains duplicate ZIP member names")
        members = {name: archive.read(name) for name in names}
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        root = etree.fromstring(members["word/document.xml"], parser)
        body = root.find("w:body", NS)
        if body is None:
            raise ValueError("DOCX has no document body")
        children = list(body)

        def text(element: Any) -> str:
            return normalize("".join(element.xpath(".//w:t/text()", namespaces=NS)))

        def is_heading(element: Any) -> bool:
            styles = element.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
            return bool(element.tag == W + "p" and styles == ["Heading1"])

        refs = [i for i, element in enumerate(children) if text(element) == "References" and is_heading(element)]
        starts = [i for i, element in enumerate(children) if heading_match(text(element)) and is_heading(element)]
        if len(refs) != 1 or len(starts) != 1 or starts[0] <= refs[0]:
            raise ValueError("DOCX needs unique References and appendix Heading1 paragraphs in that order")
        if any(APPENDIX_HEADING in text(element) for element in children[: starts[0]]):
            raise ValueError("DOCX appendix heading also occurs before the detected boundary")
        if not children or children[-1].tag != W + "sectPr":
            raise ValueError("DOCX requires final body section properties")
        cutoff = starts[0]
        # Pandoc places an appendix's bookmarkStart immediately before its heading.
        while cutoff > refs[0] and children[cutoff - 1].tag == W + "bookmarkStart":
            cutoff -= 1
        prefix = children[:cutoff]
        source_counts = xml_counts(body)
        prefix_hash = body_fingerprint(prefix)
        final_section = children[-1]
        for element in children[cutoff:-1]:
            body.remove(element)
        bookmark_starts = body.xpath(".//w:bookmarkStart/@w:id", namespaces=NS)
        bookmark_ends = body.xpath(".//w:bookmarkEnd/@w:id", namespaces=NS)
        if sorted(bookmark_starts) != sorted(bookmark_ends):
            raise ValueError("DOCX retained prefix has an unmatched bookmark across the appendix boundary")
        if body_fingerprint(list(body)[:-1]) != prefix_hash or body[-1] is not final_section:
            raise RuntimeError("DOCX retained prefix or final section properties changed")
        output_counts = xml_counts(body)
        document = etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True)
        output = io.BytesIO()
        with ZipFile(output, "w") as derived:
            derived.comment = archive.comment
            for info in archive.infolist():
                derived.writestr(info, document if info.filename == "word/document.xml" else members[info.filename])
    data = output.getvalue()
    with ZipFile(io.BytesIO(data)) as derived:
        unchanged = [name for name in names if name != "word/document.xml"]
        if derived.namelist() != names or any(derived.read(name) != members[name] for name in unchanged):
            raise RuntimeError("DOCX non-document package parts changed")
        verified_root = etree.fromstring(derived.read("word/document.xml"), parser)
        verified_body = verified_root.find("w:body", NS)
        if verified_body is None or body_fingerprint(list(verified_body)[:-1]) != prefix_hash:
            raise RuntimeError("DOCX serialized main-paper prefix changed")
    return data, {
        "appendix_heading_body_index_zero_based": starts[0],
        "retained_body_elements": cutoff,
        "references_body_index_zero_based": refs[0],
        "source_counts": source_counts,
        "output_counts": output_counts,
        "retained_body_prefix_sha256": prefix_hash,
        "retained_body_prefix_unchanged": True,
        "final_section_properties_unchanged": True,
        "unchanged_non_document_zip_members": len(unchanged),
        "unchanged_media_members": len([name for name in unchanged if name.startswith("word/media/")]),
        "non_document_member_sha256": {name: sha256(members[name]) for name in unchanged},
        "deviations": ["Removed appendix body and its immediately preceding bookmark starts."],
        "package_note": "Unused appendix media and relationships remain unchanged in the ZIP package.",
    }


def derive_documents(pdf_path: Path, docx_path: Path, output_dir: Path) -> dict[str, Any]:
    pdf_source, docx_source = pdf_path.read_bytes(), docx_path.read_bytes()
    pdf_data, pdf_record = derive_pdf(pdf_source)
    docx_data, docx_record = derive_docx(docx_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pdf": output_dir / "KBound_Without_Appendices.pdf",
        "docx": output_dir / "KBound_Without_Appendices.docx",
    }
    manifest_path = output_dir / "main_only_derivation.json"
    if any(path.exists() for path in [*outputs.values(), manifest_path]):
        raise FileExistsError("refusing to overwrite an existing main-only derivation")
    records = {}
    for kind, source_path, source, data, detail in (
        ("pdf", pdf_path, pdf_source, pdf_data, pdf_record),
        ("docx", docx_path, docx_source, docx_data, docx_record),
    ):
        records[kind] = {
            "source_path": str(source_path.resolve()),
            "source_sha256": sha256(source),
            "source_bytes": len(source),
            "output_path": str(outputs[kind].resolve()),
            "output_sha256": sha256(data),
            "output_bytes": len(data),
            **detail,
        }
    receipt = {
        "schema": "kbound-main-only-derivation-v1",
        "appendix_heading": APPENDIX_HEADING,
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "documents": records,
        "visual_qa": "Required separately after rendering; structural checks are not visual inspection.",
    }
    for kind, data in (("pdf", pdf_data), ("docx", docx_data)):
        with outputs[kind].open("xb") as handle:
            handle.write(data)
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-pdf", type=Path, required=True)
    parser.add_argument("--full-docx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = derive_documents(args.full_pdf, args.full_docx, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "pdf_pages": receipt["documents"]["pdf"]["output_pages"]}))


if __name__ == "__main__":
    main()
