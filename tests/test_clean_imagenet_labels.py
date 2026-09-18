"""Clean validation label authentication must never read image contents."""

import builtins
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "docs/research/kbound/scripts/audit_clean_imagenet_labels.py"


@pytest.fixture
def audit():
    assert SCRIPT.is_file(), "The clean validation metadata auditor is not implemented"
    spec = importlib.util.spec_from_file_location("clean_imagenet_labels", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "val"
    for synset, number in (("n00000002", 1), ("n00000001", 2), ("n00000002", 3)):
        directory = root / synset
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"ILSVRC2012_val_{number:08d}.JPEG").write_bytes(b"NOT IMAGE CONTENT")
    (root / "dataset-metadata.json").write_text("{}")
    (root / "._n00000001").write_bytes(b"metadata")
    (root / "n00000001/._ILSVRC2012_val_00000002.JPEG").write_bytes(b"metadata")
    return root


@pytest.fixture
def authority(audit, monkeypatch):
    # Directory behavior uses a small hand-checked authority. The separate
    # digest test exercises the unmodified production authentication gate.
    value = {
        "validation_synsets": ["n00000002", "n00000001", "n00000002"],
        "class_synsets": ["n00000001", "n00000002"],
        "sha256": "synthetic-fixture-only",
        "md5": "synthetic-fixture-only",
    }
    monkeypatch.setattr(audit, "load_authenticated_devkit", lambda path: value)
    return value


def test_valid_labels_record_canonical_manifest_without_opening_images(audit, dataset, authority, tmp_path, monkeypatch):
    # Opening JPEGs, even merely to hash them, violates this audit's boundary.
    for owner, name in ((builtins, "open"), (io, "open"), (os, "open")):
        original = getattr(owner, name)

        def guarded(path, *args, _original=original, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path).endswith(".JPEG"):
                raise AssertionError("image bytes must not be opened")
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(owner, name, guarded)
    output = tmp_path / "audit"
    report = audit.run_audit(dataset, tmp_path / "synthetic-devkit", output)
    expected = (
        "n00000001/ILSVRC2012_val_00000002.JPEG\n"
        "n00000002/ILSVRC2012_val_00000001.JPEG\n"
        "n00000002/ILSVRC2012_val_00000003.JPEG\n"
    )
    assert report["status"] == "LABEL_PATHS_VERIFIED_NOT_IMAGE_CONTENT"
    assert report["image_content_verified"] is False
    assert report["image_count"] == 3
    assert report["ignored_metadata_count"] == 3
    assert (output / "relative_paths.txt").read_text() == expected
    assert report["relative_paths_sha256"] == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    rows = [json.loads(line) for line in (output / "observed_paths.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert {(row["validation_id"], row["synset"]) for row in rows} == {
        (1, "n00000002"), (2, "n00000001"), (3, "n00000002")
    }
    assert json.loads((output / "report.json").read_text())["status"] == report["status"]


@pytest.mark.parametrize("mutation,field", [
    ("missing", "missing_validation_ids"),
    ("duplicate", "duplicate_validation_ids"),
    ("mislabeled", "label_mismatches"),
])
def test_missing_duplicate_or_mislabeled_image_refuses_verification(audit, dataset, authority, tmp_path, mutation, field):
    original = dataset / "n00000002/ILSVRC2012_val_00000003.JPEG"
    if mutation == "missing":
        original.unlink()
    elif mutation == "duplicate":
        (dataset / "n00000001/ILSVRC2012_val_00000001.JPEG").write_bytes(b"not an image")
    else:
        original.rename(dataset / "n00000001" / original.name)
    output = tmp_path / "audit"
    with pytest.raises(audit.AuditError):
        audit.run_audit(dataset, tmp_path / "synthetic-devkit", output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "FAILED_LABEL_PATH_AUDIT"
    assert report[field]
    assert report["image_content_verified"] is False


def test_wrong_devkit_digest_fails_before_target_walk(audit, tmp_path, monkeypatch):
    kit = tmp_path / "ILSVRC2012_devkit_t12.tar.gz"
    kit.write_bytes(b"not the authenticated official kit")

    def forbidden_walk(*args, **kwargs):
        raise AssertionError("target walk occurred before kit authentication")

    monkeypatch.setattr(os, "scandir", forbidden_walk)
    with pytest.raises(audit.AuditError, match="devkit.*digest"):
        audit.run_audit(tmp_path / "val", kit, tmp_path / "audit")


@pytest.mark.parametrize("kind", ["image_symlink", "class_symlink", "metadata_symlink", "unexpected_file", "nested_directory"])
def test_nonregular_or_unexpected_entries_are_rejected_without_following(audit, dataset, authority, tmp_path, kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    if kind == "image_symlink":
        target = dataset / "n00000002/ILSVRC2012_val_00000003.JPEG"
        target.unlink()
        target.symlink_to(outside / "unopened.JPEG")
    elif kind == "class_symlink":
        (dataset / "n00000003").symlink_to(outside, target_is_directory=True)
    elif kind == "metadata_symlink":
        (dataset / "._forbidden").symlink_to(outside, target_is_directory=True)
    elif kind == "unexpected_file":
        (dataset / "n00000001/notes.txt").write_text("unexpected")
    else:
        (dataset / "n00000001/nested").mkdir()
    output = tmp_path / "audit"
    with pytest.raises(audit.AuditError):
        audit.run_audit(dataset, tmp_path / "synthetic-devkit", output)
    assert json.loads((output / "report.json").read_text())["unexpected_entries"]


def test_existing_output_is_never_overwritten(audit, dataset, authority, tmp_path):
    output = tmp_path / "audit"
    output.mkdir()
    sentinel = output / "report.json"
    sentinel.write_text("prior evidence")
    with pytest.raises((audit.AuditError, FileExistsError)):
        audit.run_audit(dataset, tmp_path / "synthetic-devkit", output)
    assert sentinel.read_text() == "prior evidence"


def test_output_inside_dataset_is_rejected(audit, dataset, authority, tmp_path):
    with pytest.raises(audit.AuditError, match="output.*dataset"):
        audit.run_audit(dataset, tmp_path / "synthetic-devkit", dataset / "output")
    assert not (dataset / "output").exists()


def test_final_directory_io_error_cannot_issue_success(audit, dataset, authority, tmp_path, monkeypatch):
    original_open = audit._open_directory
    original_close = os.close
    root_handle = []

    def record_open(path, **kwargs):
        handle = original_open(path, **kwargs)
        if Path(path) == dataset:
            root_handle.append(handle)
        return handle

    def fail_root_close(handle):
        original_close(handle)
        if root_handle and handle == root_handle[0]:
            raise OSError("simulated final directory I/O failure")

    monkeypatch.setattr(audit, "_open_directory", record_open)
    monkeypatch.setattr(os, "close", fail_root_close)
    output = tmp_path / "audit"
    with pytest.raises(audit.AuditError):
        audit.run_audit(dataset, tmp_path / "synthetic-devkit", output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "FAILED_LABEL_PATH_AUDIT"
    assert report["class_mapping_authenticated"] is False
    assert "I/O failure" in report["fatal_error"]
