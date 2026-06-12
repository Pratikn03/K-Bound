"""Tests for the data-integrity manifest (uais.data.manifest).

Hermetic and stdlib-fast: every file operation happens inside ``tmp_path`` -- no
real datasets, no network, no torch.  Covers the build+verify round-trip and
detection of corruption (changed content) and of an added/removed file.
"""

from __future__ import annotations

import json
from pathlib import Path

from uais.data.manifest import (
    MANIFEST_VERSION,
    build_manifest,
    verify_manifest,
)


def _make_tree(base: Path) -> Path:
    """Create a tiny fake data root under ``base`` and return the root path."""
    root = base / "experiments" / "fake_archive"
    (root / "sub").mkdir(parents=True, exist_ok=True)
    (root / "a.bin").write_bytes(b"alpha-bytes-0123456789")
    (root / "b.txt").write_text("hello manifest", encoding="utf-8")
    (root / "sub" / "c.dat").write_bytes(b"\x00\x01\x02\x03nested")
    return root


def test_build_round_trips_and_verifies_ok(tmp_path: Path) -> None:
    """A freshly built manifest verifies cleanly and records every file."""
    _make_tree(tmp_path)
    out = tmp_path / "data" / "MANIFEST.json"

    manifest = build_manifest(roots=["experiments/fake_archive"], output=str(out), base=tmp_path)

    assert out.exists()
    assert manifest["version"] == MANIFEST_VERSION
    # 3 files created in the tree.
    assert len(manifest["entries"]) == 3
    paths = [e["path"] for e in manifest["entries"]]
    assert paths == sorted(paths)  # stable order
    assert "experiments/fake_archive/a.bin" in paths
    assert "experiments/fake_archive/sub/c.dat" in paths
    for entry in manifest["entries"]:
        assert len(entry["sha256"]) == 64
        assert entry["size_bytes"] > 0

    report = verify_manifest(path=str(out), base=tmp_path)
    assert report["ok"] is True
    assert report["empty"] is False
    assert report["n_entries"] == 3
    assert report["n_ok"] == 3
    assert report["missing"] == []
    assert report["changed"] == []


def test_verify_detects_corrupted_file(tmp_path: Path) -> None:
    """Mutating a file's content makes verify report a sha256 change."""
    root = _make_tree(tmp_path)
    out = tmp_path / "data" / "MANIFEST.json"
    build_manifest(roots=["experiments/fake_archive"], output=str(out), base=tmp_path)

    # Corrupt one file's bytes while keeping a *different* size guard honest:
    # change content length too so either the size or hash check trips.
    (root / "b.txt").write_text("hello manifest -- tampered!", encoding="utf-8")

    report = verify_manifest(path=str(out), base=tmp_path)
    assert report["ok"] is False
    changed_paths = {c["path"] for c in report["changed"]}
    assert "experiments/fake_archive/b.txt" in changed_paths


def test_verify_detects_content_change_same_size(tmp_path: Path) -> None:
    """A same-length content edit is caught by the sha256 (not just size)."""
    root = _make_tree(tmp_path)
    out = tmp_path / "data" / "MANIFEST.json"
    build_manifest(roots=["experiments/fake_archive"], output=str(out), base=tmp_path)

    original = (root / "a.bin").read_bytes()
    # Flip the first byte; length is unchanged so only the hash differs.
    tampered = bytes([original[0] ^ 0xFF]) + original[1:]
    assert len(tampered) == len(original)
    (root / "a.bin").write_bytes(tampered)

    report = verify_manifest(path=str(out), base=tmp_path)
    assert report["ok"] is False
    reasons = {c["path"]: c["reason"] for c in report["changed"]}
    assert reasons.get("experiments/fake_archive/a.bin") == "sha256 mismatch"


def test_verify_detects_missing_file(tmp_path: Path) -> None:
    """Deleting a recorded file makes verify report it as missing."""
    root = _make_tree(tmp_path)
    out = tmp_path / "data" / "MANIFEST.json"
    build_manifest(roots=["experiments/fake_archive"], output=str(out), base=tmp_path)

    (root / "sub" / "c.dat").unlink()

    report = verify_manifest(path=str(out), base=tmp_path)
    assert report["ok"] is False
    assert "experiments/fake_archive/sub/c.dat" in report["missing"]


def test_added_file_changes_manifest_entry_count(tmp_path: Path) -> None:
    """Adding a file and rebuilding grows the entry set (detects new inputs)."""
    root = _make_tree(tmp_path)
    out = tmp_path / "data" / "MANIFEST.json"
    m1 = build_manifest(roots=["experiments/fake_archive"], output=str(out), base=tmp_path)
    assert len(m1["entries"]) == 3

    (root / "d_new.bin").write_bytes(b"a-brand-new-input")
    m2 = build_manifest(roots=["experiments/fake_archive"], output=str(out), base=tmp_path)
    assert len(m2["entries"]) == 4
    new_paths = {e["path"] for e in m2["entries"]} - {e["path"] for e in m1["entries"]}
    assert new_paths == {"experiments/fake_archive/d_new.bin"}


def test_missing_root_builds_empty_manifest(tmp_path: Path) -> None:
    """An absent root yields a valid, empty, self-verifying manifest."""
    out = tmp_path / "data" / "MANIFEST.json"
    manifest = build_manifest(roots=["experiments/does_not_exist"], output=str(out), base=tmp_path)
    assert manifest["entries"] == []
    # The empty manifest is still well-formed JSON with a version stamp.
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["version"] == MANIFEST_VERSION

    report = verify_manifest(path=str(out), base=tmp_path)
    assert report["empty"] is True
    assert report["ok"] is True
