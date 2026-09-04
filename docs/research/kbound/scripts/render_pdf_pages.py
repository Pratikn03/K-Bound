#!/usr/bin/env python3
"""Render every release PDF page and verify the rendered page count."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT_PDF_NAMES = (
    "kbound_short_main.pdf",
    "kbound_short_supplement.pdf",
    "kbound_tmlr.pdf",
    "kbound_full_report.pdf",
)
DEFAULT_DPI = 192


def require_binary(name: str) -> str:
    override_name = "KBOUND_TOOL_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()
    override = os.environ.get(override_name)
    candidate = override if override else shutil.which(name)
    if candidate is None:
        raise RuntimeError(f"missing required PDF tool: {name}")
    if not override:
        return candidate
    unresolved = Path(override).expanduser()
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{override_name} is not a resident executable") from exc
    if (
        not unresolved.is_absolute()
        or unresolved.is_symlink()
        or unresolved != resolved
        or not resolved.is_file()
        or not os.access(resolved, os.X_OK)
    ):
        raise RuntimeError(f"{override_name} must name an absolute non-symlink executable realpath")
    return str(resolved)


def page_count(pdf: Path) -> int:
    output = subprocess.check_output([require_binary("pdfinfo"), str(pdf)], text=True)
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"pdfinfo did not report a page count for {pdf}")


def render(pdf: Path, destination: Path, *, dpi: int = DEFAULT_DPI) -> None:
    if dpi <= 0:
        raise ValueError("DPI must be positive")
    expected = page_count(pdf)
    destination.mkdir(parents=True, exist_ok=True)
    # The output directory is stable across release runs; remove only this
    # renderer's prior page images so a page-count decrease cannot look stale.
    for stale_page in destination.glob("page-*.png"):
        stale_page.unlink()
    subprocess.run(
        [
            require_binary("pdftoppm"),
            "-png",
            "-r",
            str(dpi),
            str(pdf),
            str(destination / "page"),
        ],
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
    parser.add_argument(
        "--pdf-root",
        type=Path,
        default=ROOT / "release/current",
        help="directory containing the four current release-role PDFs",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"rendering resolution in dots per inch (default: {DEFAULT_DPI})",
    )
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    for tool in ("pdfinfo", "pdftoppm"):
        try:
            require_binary(tool)
        except RuntimeError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc
    output_root = args.output_root.resolve()
    pdf_root = args.pdf_root.resolve()
    for name in CURRENT_PDF_NAMES:
        pdf = pdf_root / name
        if not pdf.is_file():
            raise SystemExit(f"ERROR: missing release PDF: {pdf}")
        render(pdf, output_root / pdf.stem, dpi=args.dpi)


if __name__ == "__main__":
    main()
