"""Small, strict provenance helpers for the prospective CCT-20 campaign."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class IntegrityError(ValueError):
    """Raised when an artifact cannot satisfy the prospective contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value deterministically and reject NaN/Infinity."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_load(path: str | os.PathLike[str]) -> Any:
    """Load ordinary (non-target) JSON while rejecting non-standard numbers."""

    def reject_constant(token: str) -> None:
        raise IntegrityError(f"non-standard JSON constant {token!r} in {path}")

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def atomic_json_dump(path: str | os.PathLike[str], document: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                dict(document),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise IntegrityError(f"{field} must be a lowercase SHA-256 digest")
    lowered = value.lower()
    if value != lowered or any(character not in "0123456789abcdef" for character in value):
        raise IntegrityError(f"{field} must be a lowercase SHA-256 digest")
    return value
