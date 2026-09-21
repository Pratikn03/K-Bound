from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from docs.research.kbound.scripts import build_next_phase_evidence as evidence


def _study(tmp_path: Path) -> tuple[Path, str, str]:
    relative = "output/next_phase/study_v1"
    root = tmp_path / relative
    root.mkdir(parents=True)
    (root / "receipt.json").write_text('{"status":"complete"}\n')
    (root / "failed_attempt.log").write_text("retained failure\n")
    return root, relative, relative + "/receipt.json"


def test_archive_binds_all_attempts_and_execution_sources(tmp_path: Path) -> None:
    root, relative, authority = _study(tmp_path)
    (root / "execution.py").write_text("# original executed source\n")
    manifest = evidence.build_manifest(tmp_path, [relative], [authority])
    assert {row["path"] for row in manifest["artifacts"]} == {
        authority,
        relative + "/failed_attempt.log",
        relative + "/execution.py",
    }
    manifest_path = tmp_path / "manifest.json"
    evidence.write_manifest(manifest_path, manifest)
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    evidence.build_archive(tmp_path, manifest_path, first)
    evidence.build_archive(tmp_path, manifest_path, second)
    assert first.read_bytes() == second.read_bytes()
    evidence.verify_archive(manifest_path, first)
    with zipfile.ZipFile(first) as archive:
        assert archive.read(relative + "/execution.py") == b"# original executed source\n"
    with pytest.raises(FileExistsError):
        evidence.build_archive(tmp_path, manifest_path, first)
    with pytest.raises(FileExistsError):
        evidence.write_manifest(manifest_path, manifest)


@pytest.mark.parametrize("change", ["altered", "added", "removed"])
def test_post_freeze_drift_rejects_archive(tmp_path: Path, change: str) -> None:
    root, relative, authority = _study(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    evidence.write_manifest(manifest_path, evidence.build_manifest(tmp_path, [relative], [authority]))
    if change == "added":
        (root / "later.json").write_text("{}")
    elif change == "removed":
        (root / "failed_attempt.log").unlink()
    else:
        (root / "receipt.json").write_text('{"status":"different"}')
    with pytest.raises(ValueError, match="drift"):
        evidence.build_archive(tmp_path, manifest_path, tmp_path / "release.zip")
    assert not (tmp_path / "release.zip").exists()


@pytest.mark.parametrize("relative", ["output/next_phase", "../outside", "/absolute", "output/next_phase/a/../b"])
def test_study_scope_must_be_an_explicit_next_phase_child(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError, match="study"):
        evidence.build_manifest(tmp_path, [relative], [])


def test_symlink_and_missing_authority_rejected(tmp_path: Path) -> None:
    root, relative, authority = _study(tmp_path)
    with pytest.raises(ValueError, match="authority"):
        evidence.build_manifest(tmp_path, [relative], [])
    (root / "redirected").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symlink"):
        evidence.build_manifest(tmp_path, [relative], [authority])


def test_archive_verification_rejects_extra_and_tampered_members(tmp_path: Path) -> None:
    _, relative, authority = _study(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    evidence.write_manifest(manifest_path, evidence.build_manifest(tmp_path, [relative], [authority]))
    archive_path = tmp_path / "release.zip"
    evidence.build_archive(tmp_path, manifest_path, archive_path)
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr("unreviewed.txt", "extra")
    with pytest.raises(ValueError, match="members"):
        evidence.verify_archive(manifest_path, archive_path)


def test_manifest_rejects_duplicate_paths_and_unsafe_artifact(tmp_path: Path) -> None:
    _, relative, authority = _study(tmp_path)
    manifest = evidence.build_manifest(tmp_path, [relative], [authority])
    manifest["artifacts"].append(manifest["artifacts"][0])
    with pytest.raises(ValueError, match="canonical"):
        evidence.validate_manifest(manifest)
    manifest = evidence.build_manifest(tmp_path, [relative], [authority])
    manifest["artifacts"][0]["path"] = "../escape"
    with pytest.raises(ValueError, match="path"):
        evidence.validate_manifest(manifest)


def test_manifest_bytes_are_canonical_and_cannot_silently_change(tmp_path: Path) -> None:
    _, relative, authority = _study(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = evidence.build_manifest(tmp_path, [relative], [authority])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="canonical"):
        evidence.build_archive(tmp_path, manifest_path, tmp_path / "release.zip")
