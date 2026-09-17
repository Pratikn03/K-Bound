#!/usr/bin/env python3
"""Authenticate extracted ImageNet validation filename labels without opening images.

The official devkit is read and authenticated first. Dataset access thereafter is
limited to directory entries and file types, with symlink traversal forbidden.
The CLI has no option to relax the official counts or digests. Successful label
authentication says nothing about image bytes, decoding, or model performance.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import tarfile
import time


DEVKIT_SHA256 = "b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953"
DEVKIT_MD5 = "fa75699e90414af021442c21a62c3abf"
IMAGE_COUNT = 50000
CLASS_COUNT = 1000
IMAGE_NAME = re.compile(r"ILSVRC2012_val_([0-9]{8})\.JPEG\Z")
HASH_RECIPE = (
    "SHA-256 of relative_paths.txt: lexicographically sorted relative POSIX image "
    "paths, UTF-8 encoded, each followed by one LF byte (including the last path); "
    "no BOM. AppleDouble-prefixed regular files and root dataset-metadata.json excluded."
)


class AuditError(RuntimeError):
    """Inputs cannot support the claimed label-path authentication."""


def load_authenticated_devkit(path):
    """Read the pinned devkit only; never extract it or read a dataset image."""
    data = Path(path).read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    md5 = hashlib.md5(data).hexdigest()
    if sha256 != DEVKIT_SHA256 or md5 != DEVKIT_MD5:
        raise AuditError(f"devkit digest mismatch: sha256={sha256}, md5={md5}")

    from scipy.io import loadmat

    prefix = "ILSVRC2012_devkit_t12/data/"
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        def member_bytes(name):
            member = archive.getmember(prefix + name)
            if not member.isfile():
                raise AuditError(f"devkit metadata member is not a regular file: {name}")
            with archive.extractfile(member) as stream:
                return stream.read()

        meta = loadmat(io.BytesIO(member_bytes("meta.mat")), squeeze_me=True)["synsets"]
        leaf_rows = [row for row in meta if int(row[4]) == 0]
        by_id = {int(row[0]): str(row[1]) for row in leaf_rows}
        labels = [int(line) for line in member_bytes("ILSVRC2012_validation_ground_truth.txt").splitlines()]
    if (len(leaf_rows) != CLASS_COUNT or len(set(by_id.values())) != CLASS_COUNT
            or set(by_id) != set(range(1, CLASS_COUNT + 1)) or len(labels) != IMAGE_COUNT
            or not set(labels).issubset(by_id)):
        raise AuditError("authenticated devkit has unexpected class or validation-label structure")
    return {
        "sha256": sha256,
        "md5": md5,
        "validation_synsets": [by_id[label] for label in labels],
        "class_synsets": sorted(by_id.values()),
    }


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _absolute(path):
    # Lexical normalization only: resolving symlinks could leave the allowed root.
    return Path(os.path.abspath(os.fspath(path)))


def _require_real_directories(path):
    for component in [*reversed(path.parents), path]:
        mode = component.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise AuditError(f"directory path contains a symlink or non-directory: {component}")


def _open_directory(path, *, dir_fd=None):
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)


def _json(path, value):
    # Only the fresh output directory is writable; replace this audit's own
    # progress file atomically so readers never observe truncated JSON.
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def run_audit(root, devkit, output_dir):
    root, devkit, output = map(_absolute, (root, devkit, output_dir))
    if root == output or root in output.parents:
        raise AuditError("output directory must be outside the dataset")
    if root == devkit or root in devkit.parents:
        raise AuditError("devkit must be outside the dataset")
    # This gate must run before any target directory walk.
    authority = load_authenticated_devkit(devkit)
    _require_real_directories(root)
    _require_real_directories(output.parent)
    output.mkdir(exist_ok=False)

    started = time.monotonic()
    expected_labels = authority["validation_synsets"]
    expected_classes = set(authority["class_synsets"])
    report = {
        "schema": "kbound-clean-imagenet-label-path-audit-v1",
        "status": "IN_PROGRESS",
        "pid": os.getpid(),
        "started_utc": _utc(),
        "root": str(root),
        "devkit": str(devkit),
        "devkit_sha256": authority["sha256"],
        "devkit_md5": authority["md5"],
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "expected_image_count": len(expected_labels),
        "expected_class_count": len(expected_classes),
        "completed_class_folders": 0,
        "image_count": 0,
        "image_content_verified": False,
        "image_bytes_opened": False,
        "class_mapping_authenticated": False,
        "directory_inventory_complete": False,
        "current_class": None,
        "relative_paths_hash_recipe": HASH_RECIPE,
        "ignored_metadata_paths": [],
        "unexpected_entries": [],
        "duplicate_validation_ids": [],
        "label_mismatches": [],
    }
    paths = []
    seen_ids = {}
    classes = []

    def progress():
        report["updated_utc"] = _utc()
        report["elapsed_seconds"] = time.monotonic() - started
        report["ignored_metadata_count"] = len(report["ignored_metadata_paths"])
        # Arrays can contain tens of thousands of paths; save them once in the
        # final report rather than rewriting them for every completed class.
        compact = {key: value for key, value in report.items() if not isinstance(value, list)}
        for key in ("unexpected_entries", "duplicate_validation_ids", "label_mismatches"):
            compact[key + "_count"] = len(report[key])
        _json(output / "progress.json", compact)

    def unexpected(path, reason):
        report["unexpected_entries"].append({"relative_path": path, "reason": reason})

    def ignored_metadata(entry, relative, *, at_root):
        if entry.name.startswith("._") or (at_root and entry.name == "dataset-metadata.json"):
            if entry.is_file(follow_symlinks=False):
                report["ignored_metadata_paths"].append(relative)
            else:
                unexpected(relative, "metadata-shaped entry is not a regular file")
            return True
        return False

    progress()
    print(json.dumps({"status": "IN_PROGRESS", "pid": os.getpid(), "output_dir": str(output)}), flush=True)
    try:
        root_fd = _open_directory(root)
        try:
            with (output / "observed_paths.jsonl").open("x", encoding="utf-8", newline="\n") as observed:
                with os.scandir(root_fd) as entries:
                    for entry in entries:
                        if entry.is_symlink():
                            unexpected(entry.name, "symlink")
                        elif ignored_metadata(entry, entry.name, at_root=True):
                            continue
                        elif entry.name in expected_classes and entry.is_dir(follow_symlinks=False):
                            classes.append(entry.name)
                        else:
                            unexpected(entry.name, "unexpected root entry or non-directory class")
                for synset in sorted(classes):
                    report["current_class"] = synset
                    progress()
                    class_fd = _open_directory(synset, dir_fd=root_fd)
                    try:
                        with os.scandir(class_fd) as entries:
                            for entry in entries:
                                relative = f"{synset}/{entry.name}"
                                if entry.is_symlink():
                                    unexpected(relative, "symlink")
                                    continue
                                if ignored_metadata(entry, relative, at_root=False):
                                    continue
                                match = IMAGE_NAME.fullmatch(entry.name)
                                if not match or not entry.is_file(follow_symlinks=False):
                                    unexpected(relative, "unexpected filename or nonregular image entry")
                                    continue
                                image_id = int(match.group(1))
                                expected = expected_labels[image_id - 1] if 1 <= image_id <= len(expected_labels) else None
                                paths.append(relative)
                                report["image_count"] += 1
                                if image_id in seen_ids:
                                    report["duplicate_validation_ids"].append({
                                        "validation_id": image_id, "first_path": seen_ids[image_id], "duplicate_path": relative,
                                    })
                                seen_ids[image_id] = relative
                                if expected is None:
                                    unexpected(relative, "validation ID outside official range")
                                elif synset != expected:
                                    report["label_mismatches"].append({"relative_path": relative, "expected_synset": expected})
                                observed.write(json.dumps({
                                    "relative_path": relative, "synset": synset,
                                    "validation_id": image_id, "expected_synset": expected,
                                }, sort_keys=True) + "\n")
                    finally:
                        os.close(class_fd)
                    observed.flush()
                    report["completed_class_folders"] += 1
                    progress()
                    if report["completed_class_folders"] % 25 == 0:
                        print(json.dumps({key: report[key] for key in ("completed_class_folders", "image_count", "elapsed_seconds")}), flush=True)
            report["directory_inventory_complete"] = True
        finally:
            os.close(root_fd)
    except Exception as error:
        report["fatal_error"] = f"{type(error).__name__}: {error}"

    manifest = "".join(path + "\n" for path in sorted(paths)).encode("utf-8")
    with (output / "relative_paths.txt").open("xb") as stream:
        stream.write(manifest)
    report["relative_paths_sha256"] = hashlib.sha256(manifest).hexdigest()
    report["unique_validation_ids"] = len(seen_ids)
    report["missing_validation_ids"] = sorted(set(range(1, len(expected_labels) + 1)) - set(seen_ids))
    report["missing_classes"] = sorted(expected_classes - set(classes))
    report["observed_class_count"] = len(classes)
    report["ignored_metadata_count"] = len(report["ignored_metadata_paths"])
    passed = (
        report["directory_inventory_complete"]
        and "fatal_error" not in report
        and report["image_count"] == len(expected_labels)
        and not any(report[key] for key in (
            "unexpected_entries", "duplicate_validation_ids", "label_mismatches",
            "missing_validation_ids", "missing_classes",
        ))
    )
    report["status"] = "LABEL_PATHS_VERIFIED_NOT_IMAGE_CONTENT" if passed else "FAILED_LABEL_PATH_AUDIT"
    report["class_mapping_authenticated"] = passed
    report["current_class"] = None
    report["finished_utc"] = _utc()
    progress()
    _json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "image_count": report["image_count"], "relative_paths_sha256": report["relative_paths_sha256"]}), flush=True)
    if not passed:
        raise AuditError(f"label-path audit failed; inspect {output / 'report.json'}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Extracted clean ImageNet val directory")
    parser.add_argument("--devkit", type=Path, required=True, help="Pinned official ILSVRC2012_devkit_t12.tar.gz")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory outside the dataset; parent must exist")
    args = parser.parse_args(argv)
    try:
        run_audit(args.root, args.devkit, args.output_dir)
    except (AuditError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
