from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "660d893caede49c3b7daa8c18e43bb6cbbce5480"
ARCHIVE = Path("docs/research/kbound/archive/stale_publication_builds_2026-09-02")
CANONICAL_OUTPUTS = (
    "docs/research/kbound/kbound_short_final_draft.docx",
    "docs/research/kbound/kbound_short_final_draft.pdf",
    "docs/research/kbound/kbound_tmlr.pdf",
)
STALE_OUTPUTS = (
    "docs/research/kbound/kbound.pdf",
    "docs/research/kbound/kbound_edited.pdf",
    "docs/research/kbound/kbound_long_companion.pdf",
    "docs/research/kbound/kbound_short.docx",
    "docs/research/kbound/kbound_short.pdf",
    "docs/research/kbound/kbound_short_companion.pdf",
    "docs/research/kbound/kbound_short_edited.pdf",
    "docs/research/kbound/manuscript/K-Bound_Manuscript.pdf",
    "docs/research/kbound/manuscript/main.pdf",
    "docs/research/kbound/paper/kbound.pdf",
)


def _git_bytes(relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_active_publication_output(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in {".pdf", ".docx"}
        and not path.name.startswith("~$")
        and "archive" not in path.parts
        and "figures" not in path.parts
    )


def _tracked_publication_outputs() -> set[str]:
    """Read only regular publication blobs from the pinned Git tree."""

    raw = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "HEAD", "--", "docs/research/kbound"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    selected: set[str] = set()
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        metadata, encoded_path = encoded.split(b"\t", 1)
        mode, object_type, _object_id = metadata.decode("ascii").split(" ", 2)
        relative = encoded_path.decode("utf-8")
        path = Path(relative)
        if (
            mode in {"100644", "100755"}
            and object_type == "blob"
            and path.suffix.lower() in {".pdf", ".docx"}
            and not path.name.startswith("~$")
            and "archive" not in path.parts
            and "figures" not in path.parts
        ):
            selected.add(relative)
    return selected


def test_publication_output_classifier_excludes_only_office_lock_prefix(tmp_path: Path) -> None:
    publication_root = tmp_path / "docs/research/kbound"
    publication_root.mkdir(parents=True)
    for name in ("~$draft.pdf", "draft~$copy.pdf", "~draft.pdf"):
        (publication_root / name).write_bytes(b"synthetic")

    assert not _is_active_publication_output(publication_root / "~$draft.pdf")
    assert _is_active_publication_output(publication_root / "draft~$copy.pdf")
    assert _is_active_publication_output(publication_root / "~draft.pdf")


def test_only_three_canonical_publication_outputs_remain_active() -> None:
    for relative_path in CANONICAL_OUTPUTS:
        assert (ROOT / relative_path).is_file(), relative_path
    for relative_path in STALE_OUTPUTS:
        assert not (ROOT / relative_path).exists(), relative_path

    assert _tracked_publication_outputs() == set(CANONICAL_OUTPUTS)


def test_stale_publication_builds_are_byte_preserved_and_manifested() -> None:
    manifest_path = ROOT / ARCHIVE / "MANIFEST.json"
    readme_path = ROOT / ARCHIVE / "README.md"
    assert manifest_path.is_file()
    assert readme_path.is_file()
    assert "not release deliverables" in readme_path.read_text(encoding="utf-8").lower()

    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["schema_version"] == 1
    assert manifest["archive_id"] == ARCHIVE.name
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["canonical_outputs"] == list(CANONICAL_OUTPUTS)
    records = manifest["records"]
    assert [record["original_path"] for record in records] == list(STALE_OUTPUTS)

    for record in records:
        original_path = record["original_path"]
        archive_path = (ARCHIVE / "tree" / original_path).as_posix()
        assert record["archive_path"] == archive_path
        assert record["category"] == "stale_publication_build"
        assert record["replacement_outputs"] == list(CANONICAL_OUTPUTS)
        archived_bytes = (ROOT / archive_path).read_bytes()
        assert archived_bytes == _git_bytes(original_path)
        assert record["bytes"] == len(archived_bytes)
        assert record["sha256"] == _sha256(archived_bytes)

    archived_files = {
        path.relative_to(ROOT).as_posix() for path in (ROOT / ARCHIVE / "tree").rglob("*") if path.is_file()
    }
    assert archived_files == {record["archive_path"] for record in records}

    canonical = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    assert manifest_text == canonical


def test_current_indexes_route_stale_builds_to_archive() -> None:
    archive_id = ARCHIVE.name
    for relative_path in (
        "docs/research/kbound/DOCS_INDEX.md",
        "docs/research/kbound/README.md",
        "docs/research/kbound/RELEASE_CHECKLIST.md",
    ):
        assert archive_id in (ROOT / relative_path).read_text(encoding="utf-8"), relative_path
