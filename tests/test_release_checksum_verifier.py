from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from docs.research.kbound.scripts import verify_release_checksums as verifier


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_checksum_verifier_accepts_exact_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"release bytes\n")
    checksums = tmp_path / "SHA256SUMS.txt"
    checksums.write_text(f"{_digest(artifact)}  artifact.bin\n", encoding="utf-8")
    assert verifier.verify_checksum_file(
        checksums,
        root=tmp_path,
        required_paths=("artifact.bin",),
    ) == 1


def test_release_checksum_verifier_rejects_mismatch_and_missing_entry(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"release bytes\n")
    checksums = tmp_path / "SHA256SUMS.txt"
    checksums.write_text(f"{'0' * 64}  artifact.bin\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verifier.verify_checksum_file(checksums, root=tmp_path)

    checksums.write_text(f"{_digest(artifact)}  artifact.bin\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required checksum entries are missing"):
        verifier.verify_checksum_file(
            checksums,
            root=tmp_path,
            required_paths=("artifact.bin", "missing.json"),
        )


def test_release_runbook_verifies_temp_file_before_atomic_publish() -> None:
    runbook = (
        Path(__file__).resolve().parents[1]
        / "docs/research/kbound/runbooks/release_candidate.sh"
    ).read_text(encoding="utf-8")
    assert "verify_release_checksums.py" in runbook
    assert 'mv -f "$checksum_tmp" "$output"' in runbook
    assert 'tee -a "$checksum_tmp"' in runbook
