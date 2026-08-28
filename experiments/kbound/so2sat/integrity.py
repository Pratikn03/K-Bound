"""Strict hashing and immutable-artifact helpers for the So2Sat campaign."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


ARTIFACT_RECEIPT_SCHEMA_V1 = "kbound_so2sat_artifact_receipt_v1"
ARTIFACT_RECEIPT_SCHEMA_V2 = "kbound_so2sat_artifact_receipt_v2"
DEFAULT_ARTIFACT_RECEIPT_SCHEMA = ARTIFACT_RECEIPT_SCHEMA_V2

_RECEIPT_COMMON_FIELDS = frozenset(
    {
        "schema",
        "artifact_bytes",
        "artifact_sha256",
        "canonical_document_sha256",
    }
)


class IntegrityError(ValueError):
    """Raised when prospective data or an artifact violates its contract."""


class LabelFirewallError(IntegrityError):
    """Raised when code attempts to cross the target-label firewall."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic ASCII JSON bytes, rejecting NaN and Infinity."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def ordered_records_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash a record stream without materializing the population in memory."""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_bytes(dict(row)))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise IntegrityError(f"{field} must be a lowercase SHA-256 digest")
    if value != value.lower() or any(character not in "0123456789abcdef" for character in value):
        raise IntegrityError(f"{field} must be a lowercase SHA-256 digest")
    return value


def strict_json_load(path: str | os.PathLike[str]) -> Any:
    def reject_constant(token: str) -> None:
        raise IntegrityError(f"non-standard JSON constant {token!r} in {path}")

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def _exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise IntegrityError(f"refusing to overwrite immutable artifact: {path}") from exc
    finally:
        if descriptor is not None:  # pragma: no cover - exceptional cleanup
            os.close(descriptor)


def _require_portable_basename(value: Any, *, field: str) -> str:
    """Require one literal filename, not a path or traversal expression."""

    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or Path(value).name != value
    ):
        raise IntegrityError(f"{field} must be one exact portable artifact basename")
    return value


def _validate_receipt_schema_name(value: Any) -> str:
    if not isinstance(value, str) or not value or any(character.isspace() for character in value):
        raise IntegrityError("receipt_schema must be one non-empty token")
    return value


def _receipt_document(
    destination: Path,
    document: Mapping[str, Any],
    *,
    receipt_schema: str,
) -> dict[str, Any]:
    """Build a receipt for an already-written canonical JSON artifact."""

    schema = _validate_receipt_schema_name(receipt_schema)
    basename = _require_portable_basename(destination.name, field="artifact basename")
    receipt = {
        "schema": schema,
        "artifact_bytes": destination.stat().st_size,
        "artifact_sha256": file_sha256(destination),
        "canonical_document_sha256": stable_sha256(dict(document)),
    }
    if schema == ARTIFACT_RECEIPT_SCHEMA_V1:
        # Legacy v1 remains available only through an explicit schema request.
        # Its absolute path is deliberately verified exactly for compatibility.
        receipt["artifact_path"] = str(destination)
    else:
        receipt["artifact_basename"] = basename
    return receipt


def write_immutable_json_with_receipt(
    path: str | os.PathLike[str],
    document: Mapping[str, Any],
    *,
    receipt_schema: str = DEFAULT_ARTIFACT_RECEIPT_SCHEMA,
) -> dict[str, Any]:
    """Write a create-only JSON artifact and portable create-only receipt.

    New generic receipts default to v2 and bind the exact artifact basename,
    making a verified artifact/receipt pair relocatable as a unit.  Passing the
    legacy generic v1 schema explicitly retains the original absolute-path
    representation for controlled compatibility tests and migrations.  Other
    explicit custom schemas use the portable basename representation.
    """

    destination = Path(path).expanduser().resolve()
    schema = _validate_receipt_schema_name(receipt_schema)
    _require_portable_basename(destination.name, field="artifact basename")
    receipt_path = destination.with_name(destination.name + ".receipt.json")
    if destination.exists() or receipt_path.exists():
        raise IntegrityError(f"refusing to overwrite artifact/receipt pair for {destination}")
    payload = json.dumps(
        dict(document),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    _exclusive_write(destination, payload)
    receipt = _receipt_document(
        destination,
        document,
        receipt_schema=schema,
    )
    if receipt["artifact_bytes"] != len(payload):  # pragma: no cover - defensive race guard
        raise IntegrityError("artifact changed while its receipt was being constructed")
    _exclusive_write(
        receipt_path,
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode("ascii") + b"\n",
    )
    return receipt


def verify_artifact_receipt(
    artifact_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str] | None = None,
    *,
    receipt_schema: str | None = None,
) -> dict[str, Any]:
    """Verify a portable v2, strict legacy v1, or explicit custom receipt.

    With no schema argument, only the two generic schemas are accepted.  A
    custom schema must be requested explicitly and uses the portable basename
    layout.  Legacy v1 verification remains path-exact: moving only one member
    of a v1 artifact/receipt pair is rejected rather than silently rebased.
    """

    artifact = Path(artifact_path).expanduser().resolve()
    artifact_basename = _require_portable_basename(artifact.name, field="artifact basename")
    receipt_file = (
        Path(receipt_path).expanduser().resolve()
        if receipt_path is not None
        else artifact.with_name(artifact.name + ".receipt.json")
    )
    if not artifact.is_file() or not receipt_file.is_file():
        raise IntegrityError(f"artifact/receipt pair is incomplete: {artifact}, {receipt_file}")
    receipt = strict_json_load(receipt_file)
    document = strict_json_load(artifact)
    if not isinstance(receipt, Mapping) or not isinstance(document, Mapping):
        raise IntegrityError("artifact and receipt must both be JSON mappings")
    observed_schema = receipt.get("schema")
    if not isinstance(observed_schema, str):
        raise IntegrityError("unknown So2Sat receipt schema")
    if receipt_schema is None:
        accepted_schemas = {ARTIFACT_RECEIPT_SCHEMA_V1, ARTIFACT_RECEIPT_SCHEMA_V2}
    else:
        accepted_schemas = {_validate_receipt_schema_name(receipt_schema)}
    if observed_schema not in accepted_schemas:
        raise IntegrityError("unknown So2Sat receipt schema")
    if observed_schema == ARTIFACT_RECEIPT_SCHEMA_V1:
        expected_fields = _RECEIPT_COMMON_FIELDS | {"artifact_path"}
        if set(receipt) != expected_fields:
            raise IntegrityError("legacy receipt has unknown or missing fields")
        claimed_path = receipt.get("artifact_path")
        if not isinstance(claimed_path, str) or not Path(claimed_path).is_absolute():
            raise IntegrityError("legacy receipt artifact_path must be absolute")
        if Path(claimed_path).name != artifact_basename:
            raise IntegrityError("legacy receipt artifact basename mismatch")
        if claimed_path != str(artifact):
            raise IntegrityError("legacy receipt artifact_path mismatch")
    else:
        expected_fields = _RECEIPT_COMMON_FIELDS | {"artifact_basename"}
        if set(receipt) != expected_fields:
            raise IntegrityError("portable receipt has unknown or missing fields")
        claimed_basename = _require_portable_basename(
            receipt.get("artifact_basename"),
            field="receipt artifact_basename",
        )
        if claimed_basename != artifact_basename:
            raise IntegrityError("portable receipt artifact_basename mismatch")
    artifact_bytes = receipt.get("artifact_bytes")
    if (
        isinstance(artifact_bytes, bool)
        or not isinstance(artifact_bytes, int)
        or artifact_bytes < 1
    ):
        raise IntegrityError("receipt artifact_bytes must be a positive integer")
    if artifact_bytes != artifact.stat().st_size:
        raise IntegrityError("receipt byte count mismatch")
    artifact_sha256 = require_sha256(receipt.get("artifact_sha256"), field="artifact_sha256")
    if artifact_sha256 != file_sha256(artifact):
        raise IntegrityError("receipt file SHA-256 mismatch")
    canonical_sha256 = require_sha256(
        receipt.get("canonical_document_sha256"),
        field="canonical_document_sha256",
    )
    if canonical_sha256 != stable_sha256(dict(document)):
        raise IntegrityError("receipt canonical document SHA-256 mismatch")
    return dict(receipt)
