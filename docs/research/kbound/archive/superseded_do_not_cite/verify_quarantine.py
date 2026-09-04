#!/usr/bin/env python3
"""Verify the K-Bound superseded-PDF quarantine without reading active claims."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys


QUARANTINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = QUARANTINE_DIR.parents[4]

EXPECTED = {
    "kbound_short.pdf": {
        "sha256": "c87cbf8042e160cd8217bc8fa698b7ff3fb77ff6b0f0469c97d2bcb4d2c1834a",
        "pages": 58,
    },
    "kbound.pdf": {
        "sha256": "eb83c197be9a06ef6f9167fe6872daf608836d7ee8de8cd4b58b6f86b677bf42",
        "pages": 52,
    },
    "kbound_submission.pdf": {
        "sha256": "acd938247b08b5f8665dd3b0d7a307d5b157343c8eb841aadb6195e54736d6d4",
        "pages": 35,
    },
}

WATERMARK_LINES = (
    "SUPERSEDED DRAFT",
    "NUMBERS AND CLAIMS DO NOT MATCH",
    "THE CURRENT SUBMISSION",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_poppler_tool(name: str) -> str:
    direct = shutil.which(name)
    if direct:
        return direct

    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        dependencies = Path(pdfinfo).resolve().parents[2]
        for candidate in (
            dependencies / "native" / "poppler" / "poppler" / "bin" / name,
            dependencies / "native" / "poppler" / "bin" / name,
        ):
            if candidate.is_file():
                return str(candidate)
    raise RuntimeError(f"Required Poppler tool not found: {name}")


def pdf_pages(path: Path, pdfinfo: str) -> int:
    output = subprocess.check_output([pdfinfo, str(path)], text=True)
    match = re.search(r"^Pages:\s+(\d+)\s*$", output, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not read page count from {path}")
    return int(match.group(1))


def pdf_text(path: Path, pdftotext: str, *, first_page_only: bool = False) -> str:
    # Raw reading order keeps the words of the rotated vector warning together;
    # the default physical-layout heuristic can interleave individual letters.
    command = [pdftotext, "-raw"]
    if first_page_only:
        command.extend(["-f", "1", "-l", "1"])
    command.extend([str(path), "-"])
    return subprocess.check_output(command, text=True, errors="replace")


def normalized(text: str) -> str:
    return " ".join(text.split())


def verify_public_surfaces() -> list[str]:
    failures: list[str] = []
    surfaces = (
        REPO_ROOT,
        REPO_ROOT / "docs" / "research" / "kbound",
        REPO_ROOT / "docs" / "research" / "kbound" / "release" / "current",
        REPO_ROOT / "docs" / "research" / "kbound" / "output" / "pdf",
    )
    for surface in surfaces:
        for historical_name in EXPECTED:
            candidate = surface / historical_name
            if candidate.exists():
                failures.append(f"superseded name remains on a public surface: {candidate}")
    return failures


def main() -> int:
    pdfinfo = find_poppler_tool("pdfinfo")
    pdftotext = find_poppler_tool("pdftotext")
    failures = verify_public_surfaces()

    for historical_name, expected in EXPECTED.items():
        original = QUARANTINE_DIR / "originals" / historical_name
        warning = QUARANTINE_DIR / f"{Path(historical_name).stem}_SUPERSEDED.pdf"

        if not original.is_file():
            failures.append(f"missing preserved original: {original}")
            continue
        if not warning.is_file():
            failures.append(f"missing warning copy: {warning}")
            continue

        actual_hash = sha256(original)
        if actual_hash != expected["sha256"]:
            failures.append(
                f"original hash mismatch for {historical_name}: {actual_hash}"
            )

        original_pages = pdf_pages(original, pdfinfo)
        warning_pages = pdf_pages(warning, pdfinfo)
        if original_pages != expected["pages"]:
            failures.append(
                f"original page mismatch for {historical_name}: {original_pages}"
            )
        if warning_pages != original_pages:
            failures.append(
                f"warning page mismatch for {historical_name}: "
                f"{warning_pages} != {original_pages}"
            )

        original_text = normalized(pdf_text(original, pdftotext))
        warning_text = normalized(pdf_text(warning, pdftotext))
        first_page = normalized(
            pdf_text(warning, pdftotext, first_page_only=True)
        )
        for line in WATERMARK_LINES:
            if line not in first_page:
                failures.append(
                    f"missing searchable watermark line in {warning.name}: {line}"
                )

        sample = " ".join(original_text.split()[:24])
        if not sample or sample not in warning_text:
            failures.append(
                f"searchable manuscript text was not retained in {warning.name}"
            )

    if failures:
        print("Superseded quarantine verification FAILED:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Superseded quarantine verification passed:")
    for historical_name, expected in EXPECTED.items():
        warning = QUARANTINE_DIR / f"{Path(historical_name).stem}_SUPERSEDED.pdf"
        print(
            f"- {historical_name}: {expected['pages']} pages, "
            f"original sha256={expected['sha256']}, "
            f"warning sha256={sha256(warning)}"
        )
    print("- searchable three-line warning present on every first page")
    print("- superseded filenames absent from default public-release surfaces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
