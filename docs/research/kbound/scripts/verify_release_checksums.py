#!/usr/bin/env python3
"""Fail-closed verifier for ``KBOUND_RELEASE_SHA256SUMS.txt``."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKSUMS = ROOT / "docs/research/kbound/KBOUND_RELEASE_SHA256SUMS.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum_file(
    checksum_path: Path,
    *,
    root: Path,
    required_paths: tuple[str, ...] = (),
) -> int:
    root = root.resolve()
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("release checksum file is empty")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"malformed checksum line {line_number}")
        digest, relative = match.groups()
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != relative:
            raise ValueError(f"unsafe checksum path on line {line_number}: {relative}")
        if relative in entries:
            raise ValueError(f"duplicate checksum entry: {relative}")
        entries[relative] = digest

    missing_entries = sorted(set(required_paths) - entries.keys())
    if missing_entries:
        raise ValueError("required checksum entries are missing: " + ", ".join(missing_entries))

    for relative, expected in entries.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"checksummed release file is missing or a symlink: {relative}")
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(
                f"release checksum mismatch for {relative}: expected {expected}, got {observed}"
            )
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checksum_file", nargs="?", type=Path, default=DEFAULT_CHECKSUMS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require", action="append", default=[])
    args = parser.parse_args()
    try:
        count = verify_checksum_file(
            args.checksum_file.resolve(),
            root=args.root,
            required_paths=tuple(args.require),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(f"release checksums: PASS ({count} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
