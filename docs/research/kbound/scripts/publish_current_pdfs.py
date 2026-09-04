#!/usr/bin/env python3
"""Publish the four checksum-sealed current K-Bound PDF roles."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

SOURCE_TO_CURRENT = {
    "kbound_full_report.pdf": "kbound_full_report.pdf",
    "kbound_short_main.pdf": "kbound_short_main.pdf",
    "kbound_short_supplement.pdf": "kbound_short_supplement.pdf",
    "kbound_tmlr.pdf": "kbound_tmlr.pdf",
}
CHECKSUM_NAME = "KBOUND_CURRENT_SHA256SUMS.txt"


def _validated_pdf_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"missing maintained PDF: {path}")
    payload = path.read_bytes()
    if not payload.startswith(b"%PDF-") or b"%%EOF" not in payload[-2048:]:
        raise ValueError(f"not a complete PDF: {path}")
    return payload


def _stage_bytes(output_dir: Path, destination_name: str, payload: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir,
        prefix=f".{destination_name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def publish_current_pdfs(
    paper_dir: Path,
    release_dir: Path,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Replace both current surfaces from one validated four-role payload set."""
    payloads = {
        current_name: _validated_pdf_bytes(paper_dir / source_name)
        for source_name, current_name in SOURCE_TO_CURRENT.items()
    }
    checksum_payload = "".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n"
        for name in sorted(payloads)
    ).encode("utf-8")
    destinations = tuple(dict.fromkeys((release_dir, output_dir)))
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)

    staged: dict[tuple[Path, str], Path] = {}
    try:
        for destination in destinations:
            for current_name, payload in payloads.items():
                staged[(destination, current_name)] = _stage_bytes(
                    destination, current_name, payload
                )
            staged[(destination, CHECKSUM_NAME)] = _stage_bytes(
                destination,
                CHECKSUM_NAME,
                checksum_payload,
            )

        for (destination, destination_name), temporary in staged.items():
            os.replace(temporary, destination / destination_name)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)

    current_names = set(payloads)
    for destination in destinations:
        for path in destination.glob("kbound*.pdf"):
            if path.name not in current_names:
                path.unlink()
        for path in destination.glob("KBOUND_*SHA256SUMS.txt"):
            if path.name != CHECKSUM_NAME:
                path.unlink()

    published = tuple(release_dir / name for name in sorted(current_names))
    for destination in destinations:
        for name in sorted(current_names):
            print(destination / name)
        print(destination / CHECKSUM_NAME)
    return published


def _arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=repository / "docs/research/kbound",
        help="directory containing the four freshly built release-role PDFs",
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=repository / "docs/research/kbound/release/current",
        help="authoritative directory containing the four current release roles",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "output/pdf",
        help="mirrored publication directory for byte-identical current-role copies",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    publish_current_pdfs(
        arguments.paper_dir.resolve(),
        arguments.release_dir.resolve(),
        arguments.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
