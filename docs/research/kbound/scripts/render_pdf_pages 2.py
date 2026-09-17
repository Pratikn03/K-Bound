#!/usr/bin/env python3
"""Render every release PDF page and verify the rendered page count."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def page_count(pdf: Path) -> int:
    output = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"pdfinfo did not report a page count for {pdf}")


def render(pdf: Path, destination: Path) -> None:
    expected = page_count(pdf)
    destination.mkdir(parents=True, exist_ok=True)
    # The output directory is stable across release runs; remove only this
    # renderer's prior page images so a page-count decrease cannot look stale.
    for stale_page in destination.glob("page-*.png"):
        stale_page.unlink()
    subprocess.run(
        ["pdftoppm", "-png", "-r", "144", str(pdf), str(destination / "page")],
        check=True,
    )
    pages = sorted(destination.glob("page-*.png"))
    if len(pages) != expected:
        raise RuntimeError(f"{pdf.name}: rendered {len(pages)} pages, expected {expected}")
    if any(path.stat().st_size == 0 for path in pages):
        raise RuntimeError(f"{pdf.name}: one or more rendered pages are empty")
    print(f"OK: {pdf.name}: rendered all {expected} pages to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "kbound_release_pdf_pages",
    )
    args = parser.parse_args()
    for tool in ("pdfinfo", "pdftoppm"):
        if shutil.which(tool) is None:
            raise SystemExit(f"ERROR: missing required PDF tool: {tool}")
    output_root = args.output_root.resolve()
    for name in ("kbound_short_final_draft.pdf", "kbound_tmlr.pdf"):
        pdf = ROOT / name
        if not pdf.is_file():
            raise SystemExit(f"ERROR: missing release PDF: {pdf}")
        render(pdf, output_root / pdf.stem)


if __name__ == "__main__":
    main()
