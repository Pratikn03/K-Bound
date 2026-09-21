"""Check active executable portability without rewriting preserved history.

The inventory is an explicit preservation commitment, not authentication of
historical scientific claims. Every script in its bounded roots is registered;
preserved scripts are forbidden in the independently supplied release closure.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterable
from pathlib import Path, PurePosixPath


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("invalid relative source path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(p in {".", ".."} for p in path.parts):
        raise ValueError("noncanonical relative source path")
    return value


def _regular(root: Path, relative: str) -> Path:
    path = root / _relative(relative)
    for component in (path, *path.parents):
        metadata = component.lstat()
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_flags", 0) & getattr(stat, "SF_DATALESS", 0x40000000):
            raise ValueError(f"redirected or unavailable source: {relative}")
        if component == root:
            break
    if not path.is_file():
        raise ValueError(f"source is not a regular file: {relative}")
    return path


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate inventory key: {key}")
        result[key] = value
    return result


def _walk_error(error: OSError) -> None:
    raise error


def check_source_hygiene(
    root: Path,
    source_paths: Iterable[str],
    inventory_path: Path,
    release_paths: Iterable[str],
    banned_fragments: Iterable[str],
    detector_allowlist: Iterable[str] = (),
) -> set[str]:
    """Return active portability violations; invalid preservation raises.

    Callers supply tracked plus new nonignored code, and separately the full
    maintained release closure. Ignored scratch is not silently trusted: any
    scratch path required by that closure is scanned too.
    """
    root = root.absolute()
    inventory = json.loads(
        _regular(root, inventory_path.absolute().relative_to(root).as_posix()).read_text(),
        object_pairs_hook=_pairs,
    )
    if inventory.get("schema") != "kbound-preserved-executables-v1":
        raise ValueError("unsupported preservation inventory")
    roots = inventory["roots"]
    if not isinstance(roots, list) or not roots or len(roots) != len(set(roots)):
        raise ValueError("invalid preservation roots")
    for prefix in roots:
        _relative(prefix)
        if not (prefix == "docs/research/kbound/archive" or prefix.startswith(("archive/", "docs/research/kbound/archive/")) or re.fullmatch(r"audits/integrity_\d{4}-\d{2}-\d{2}", prefix)):
            raise ValueError("preservation cannot exempt active source directories")
    records = inventory["records"]
    expected = set()
    for row in records:
        relative = _relative(row["path"])
        if relative in expected or not any(relative.startswith(prefix + "/") for prefix in roots):
            raise ValueError("duplicate or out-of-root preservation record")
        if Path(relative).suffix not in {".py", ".sh"}:
            raise ValueError("inventory must contain executable source only")
        if type(row["bytes"]) is not int or row["bytes"] < 0 or not re.fullmatch(r"[a-f0-9]{64}", row["sha256"]):
            raise ValueError("invalid preservation identity")
        data = _regular(root, relative).read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"changed preserved bytes: {relative}")
        expected.add(relative)
    observed = set()
    for prefix in roots:
        directory = root / prefix
        # Reject redirected ancestors before walking and every linked child.
        for ancestor in (directory, *directory.parents):
            metadata = ancestor.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("preservation root is redirected or not a directory")
            if ancestor == root:
                break
        for parent, dirs, files in os.walk(directory, followlinks=False, onerror=_walk_error):
            for name in dirs + files:
                child = Path(parent) / name
                if child.is_symlink():
                    raise ValueError("symlink in preservation root")
            dirs[:] = [name for name in dirs if name != ".git"]
            for name in files:
                child = Path(parent) / name
                if child.suffix in {".py", ".sh"}:
                    observed.add(child.relative_to(root).as_posix())
    if observed != expected:
        raise ValueError(f"unregistered or missing preserved scripts: {sorted(observed ^ expected)}")
    release = {_relative(path) for path in release_paths}
    if release & expected:
        raise ValueError(f"preserved code in release source: {sorted(release & expected)}")
    paths = {_relative(path) for path in source_paths} | release
    violations = set()
    allowed = set(detector_allowlist)
    fragments = tuple(banned_fragments)
    for relative in sorted(paths - expected):
        if Path(relative).suffix not in {".py", ".sh"}:
            continue
        data = _regular(root, relative).read_bytes()
        if b"\x00" in data:
            raise ValueError(f"unreadable executable text: {relative}")
        text = data.decode("utf-8")
        if relative not in allowed and any(fragment in text for fragment in fragments):
            violations.add(relative)
    return violations
