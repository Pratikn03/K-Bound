#!/usr/bin/env python3
"""Verify the four current K-Bound PDFs with Poppler and Ghostscript."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

try:
    from docs.research.kbound.scripts.render_pdf_pages import (
        CURRENT_PDF_NAMES,
        require_binary,
    )
except ModuleNotFoundError:  # direct execution from the scripts directory
    from render_pdf_pages import CURRENT_PDF_NAMES, require_binary

ROOT = Path(__file__).resolve().parents[1]


def parse_pdfinfo(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def unembedded_fonts(output: str) -> tuple[str, ...]:
    missing = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 6 or not fields[-1].isdigit() or not fields[-2].isdigit():
            continue
        if fields[-5] != "yes":
            missing.append(fields[0])
    return tuple(missing)


def _font_count(output: str) -> int:
    count = 0
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[-1].isdigit() and fields[-2].isdigit():
            count += 1
    return count


def _run_text(command: list[str], *, accepted: tuple[int, ...] = (0,)) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode not in accepted:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    return result.stdout + result.stderr


def verify_pdf(pdf: Path, output_root: Path, tools: dict[str, str]) -> dict[str, object]:
    payload = pdf.read_bytes()
    if not payload.startswith(b"%PDF-") or b"%%EOF" not in payload[-2048:]:
        raise RuntimeError(f"{pdf}: missing PDF header or terminal EOF marker")

    role = pdf.stem
    info_text = _run_text([tools["pdfinfo"], str(pdf)])
    font_text = _run_text([tools["pdffonts"], str(pdf)])
    ghostscript_text = _run_text(
        [
            tools["gs"],
            "-q",
            "-dSAFER",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=nullpage",
            str(pdf),
        ]
    )
    text_path = output_root / "text" / f"{role}.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    _run_text([tools["pdftotext"], "-layout", str(pdf), str(text_path)])
    extracted = text_path.read_text(encoding="utf-8")

    info = parse_pdfinfo(info_text)
    pages = int(info.get("Pages", "0"))
    failures = []
    if pages <= 0:
        failures.append("nonpositive page count")
    if info.get("Encrypted", "").lower() != "no":
        failures.append("PDF is encrypted")
    if "612 x 792 pts" not in info.get("Page size", ""):
        failures.append(f"unexpected page size: {info.get('Page size', 'missing')}")
    missing_fonts = unembedded_fonts(font_text)
    if missing_fonts:
        failures.append(f"unembedded fonts: {', '.join(missing_fonts)}")
    if len(extracted.strip()) < 1000:
        failures.append("extracted text is unexpectedly short")
    if "\ufffd" in extracted:
        failures.append("extracted text contains Unicode replacement characters")
    formfeeds = extracted.count("\f")
    if formfeeds not in (pages - 1, pages):
        failures.append(f"text page separators {formfeeds} do not match {pages} pages")
    if ghostscript_text.strip():
        failures.append(f"Ghostscript emitted diagnostics: {ghostscript_text.strip()}")

    (output_root / f"{role}.pdfinfo.txt").write_text(info_text, encoding="utf-8")
    (output_root / f"{role}.fonts.txt").write_text(font_text, encoding="utf-8")
    (output_root / f"{role}.ghostscript.txt").write_text(
        ghostscript_text, encoding="utf-8"
    )
    result = {
        "role": role,
        "pdf": str(pdf),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "pages": pages,
        "page_size": info.get("Page size", ""),
        "pdf_version": info.get("PDF version", ""),
        "encrypted": info.get("Encrypted", ""),
        "font_records": _font_count(font_text),
        "all_fonts_embedded": not missing_fonts,
        "extracted_characters": len(extracted),
        "text_page_separators": formfeeds,
        "ghostscript_clean": not ghostscript_text.strip(),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    return result


def _write_report(
    results: list[dict[str, object]], tools: dict[str, str], destination: Path
) -> None:
    lines = [
        "# K-Bound PDF structural verification",
        "",
        "Each current release role was checked with `pdfinfo`, `pdffonts`, `pdftotext -layout`, and a Ghostscript null-device parse.",
        "",
        "| Release role | Status | Pages | Page size | Fonts embedded | Extracted characters | Ghostscript | SHA-256 |",
        "|---|---|---:|---|---|---:|---|---|",
    ]
    for result in results:
        lines.append(
            "| {role} | {status} | {pages} | {page_size} | {fonts} ({font_records}) | "
            "{extracted_characters} | {ghostscript} | `{sha256}` |".format(
                **result,
                fonts="yes" if result["all_fonts_embedded"] else "no",
                ghostscript="clean" if result["ghostscript_clean"] else "diagnostics",
            )
        )
    lines.extend(["", "## Exact tool paths", ""])
    for name in sorted(tools):
        lines.append(f"- `{name}`: `{tools[name]}`")
    lines.extend(["", "## Exact commands", ""])
    for result in results:
        pdf = str(result["pdf"])
        text_output = destination.parent / "text" / f"{result['role']}.txt"
        commands = (
            [tools["pdfinfo"], pdf],
            [tools["pdffonts"], pdf],
            [tools["pdftotext"], "-layout", pdf, str(text_output)],
            [
                tools["gs"],
                "-q",
                "-dSAFER",
                "-dNOPAUSE",
                "-dBATCH",
                "-sDEVICE=nullpage",
                pdf,
            ],
        )
        lines.append(f"### `{result['role']}`")
        lines.append("")
        lines.extend(f"- `{shlex.join(command)}`" for command in commands)
        lines.append("")
    lines.extend(
        [
            "",
            "Detailed Poppler output, extracted text, and Ghostscript output are stored beside this report.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-root", type=Path, default=ROOT / "release/current"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "paper/reports/pdf_structure"
    )
    args = parser.parse_args()
    tools = {name: require_binary(name) for name in ("gs", "pdffonts", "pdfinfo", "pdftotext")}
    output_root = args.output_root.resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    results = [
        verify_pdf(args.pdf_root.resolve() / name, output_root, tools)
        for name in CURRENT_PDF_NAMES
    ]
    (output_root / "pdf_structure_summary.json").write_text(
        json.dumps({"roles": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = output_root / "PDF_STRUCTURAL_VERIFICATION.md"
    _write_report(results, tools, report)
    for result in results:
        if result["status"] != "PASS":
            raise SystemExit(f"FAIL: {result['role']}: {result['failures']}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
