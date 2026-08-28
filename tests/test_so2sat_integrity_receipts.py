"""Focused portability and backward-compatibility tests for So2Sat receipts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from experiments.kbound.so2sat.integrity import (
    ARTIFACT_RECEIPT_SCHEMA_V1,
    ARTIFACT_RECEIPT_SCHEMA_V2,
    IntegrityError,
    file_sha256,
    stable_sha256,
    verify_artifact_receipt,
    write_immutable_json_with_receipt,
)


def _receipt_path(artifact: Path) -> Path:
    return artifact.with_name(artifact.name + ".receipt.json")


def _replace_receipt(artifact: Path, receipt: dict[str, Any]) -> None:
    path = _receipt_path(artifact)
    path.chmod(0o644)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def _write_legacy_pair(
    directory: Path,
    basename: str = "legacy.json",
) -> tuple[Path, dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / basename
    document = {"schema": "synthetic_document_v1", "value": 7}
    artifact.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    receipt = {
        "schema": ARTIFACT_RECEIPT_SCHEMA_V1,
        "artifact_path": str(artifact.resolve()),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": file_sha256(artifact),
        "canonical_document_sha256": stable_sha256(document),
    }
    _receipt_path(artifact).write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    return artifact, receipt


def test_default_v2_receipt_is_basename_bound_and_relocatable(tmp_path: Path) -> None:
    original = tmp_path / "original" / "seal.json"
    receipt = write_immutable_json_with_receipt(original, {"schema": "test", "value": 1})
    assert receipt["schema"] == ARTIFACT_RECEIPT_SCHEMA_V2
    assert receipt["artifact_basename"] == "seal.json"
    assert "artifact_path" not in receipt
    assert verify_artifact_receipt(original) == receipt

    relocated = tmp_path / "relocated" / original.name
    relocated.parent.mkdir()
    shutil.copy2(original, relocated)
    shutil.copy2(_receipt_path(original), _receipt_path(relocated))
    assert verify_artifact_receipt(relocated) == receipt


def test_legacy_v1_receipt_verifies_at_exact_original_path_only(tmp_path: Path) -> None:
    artifact, receipt = _write_legacy_pair(tmp_path / "original")
    assert verify_artifact_receipt(artifact) == receipt

    relocated = tmp_path / "relocated" / artifact.name
    relocated.parent.mkdir()
    shutil.copy2(artifact, relocated)
    shutil.copy2(_receipt_path(artifact), _receipt_path(relocated))
    with pytest.raises(IntegrityError, match="artifact_path mismatch"):
        verify_artifact_receipt(relocated)


def test_legacy_v1_can_still_be_requested_explicitly_for_controlled_migration(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "legacy-explicit.json"
    receipt = write_immutable_json_with_receipt(
        artifact,
        {"schema": "test"},
        receipt_schema=ARTIFACT_RECEIPT_SCHEMA_V1,
    )
    assert receipt["artifact_path"] == str(artifact.resolve())
    assert "artifact_basename" not in receipt
    assert verify_artifact_receipt(artifact) == receipt


@pytest.mark.parametrize(
    "malicious_basename",
    ["../seal.json", "subdir/seal.json", r"subdir\seal.json", "/tmp/seal.json", ".."],
)
def test_v2_rejects_traversal_and_non_basename_claims(
    tmp_path: Path,
    malicious_basename: str,
) -> None:
    artifact = tmp_path / "seal.json"
    receipt = write_immutable_json_with_receipt(artifact, {"schema": "test"})
    receipt["artifact_basename"] = malicious_basename
    _replace_receipt(artifact, receipt)
    with pytest.raises(IntegrityError, match="exact portable artifact basename"):
        verify_artifact_receipt(artifact)


def test_v2_rejects_a_different_valid_basename(tmp_path: Path) -> None:
    artifact = tmp_path / "seal.json"
    receipt = write_immutable_json_with_receipt(artifact, {"schema": "test"})
    receipt["artifact_basename"] = "different.json"
    _replace_receipt(artifact, receipt)
    with pytest.raises(IntegrityError, match="artifact_basename mismatch"):
        verify_artifact_receipt(artifact)


def test_receipt_layout_rejects_mixed_or_unknown_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "seal.json"
    receipt = write_immutable_json_with_receipt(artifact, {"schema": "test"})
    receipt["artifact_path"] = str(artifact.resolve())
    _replace_receipt(artifact, receipt)
    with pytest.raises(IntegrityError, match="unknown or missing fields"):
        verify_artifact_receipt(artifact)


def test_receipt_rejects_a_non_string_schema_without_type_errors(tmp_path: Path) -> None:
    artifact = tmp_path / "seal.json"
    receipt = write_immutable_json_with_receipt(artifact, {"schema": "test"})
    receipt["schema"] = [ARTIFACT_RECEIPT_SCHEMA_V2]
    _replace_receipt(artifact, receipt)
    with pytest.raises(IntegrityError, match="unknown So2Sat receipt schema"):
        verify_artifact_receipt(artifact)


def test_custom_schema_is_portable_but_requires_explicit_verification(tmp_path: Path) -> None:
    custom_schema = "kbound_so2sat_custom_receipt_v7"
    artifact = tmp_path / "custom.json"
    receipt = write_immutable_json_with_receipt(
        artifact,
        {"schema": "custom_document"},
        receipt_schema=custom_schema,
    )
    assert receipt["artifact_basename"] == artifact.name
    assert "artifact_path" not in receipt
    with pytest.raises(IntegrityError, match="unknown So2Sat receipt schema"):
        verify_artifact_receipt(artifact)
    assert verify_artifact_receipt(artifact, receipt_schema=custom_schema) == receipt
