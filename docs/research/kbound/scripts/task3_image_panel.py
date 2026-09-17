#!/usr/bin/env python3
"""Build a bounded, fixed real-image engineering panel, never benchmark evidence.

Selection uses only authenticated filename inventory. No outcomes, confidences,
or adaptation results are accepted. JPEG bytes are decoded and hashed together;
this verifies local content integrity, not identity with an official image tar.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import stat
import sys
from datetime import datetime, timezone

DEVKIT_SHA256 = "b59243268c0d266621fd587d2018f69e906fb22875aca0e295b48cafaa927953"
NAME = re.compile(r"n[0-9]{8}/ILSVRC2012_val_[0-9]{8}\.JPEG\Z")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_names(names):
    if not names or any(not isinstance(p, str) or not NAME.fullmatch(p) for p in names):
        raise ValueError("invalid relative image path")
    if len(set(names)) != len(names) or len({Path(p).name for p in names}) != len(names):
        raise ValueError("duplicate underlying image ID")


def plan_smoke(names, *, seed, source_count, target_count):
    validate_names(names)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(target_count) is not int or target_count <= 100:
        raise ValueError("target smoke must exceed native 100-sample warmup")
    if type(source_count) is not int or not 2 <= source_count <= len(names) // 4:
        raise ValueError("insufficient disjoint source pool")
    ordered = sorted(names)
    random.Random(seed).shuffle(ordered)
    split = len(ordered) // 4
    if target_count > len(ordered) - split:
        raise ValueError("insufficient target pool")
    return {"source": ordered[:source_count], "target": ordered[split:split + target_count]}


def load_inventory(root, report_path, inventory_path):
    raw_report = Path(report_path).read_bytes()
    report = json.loads(raw_report)
    if (report.get("status") != "LABEL_PATHS_VERIFIED_NOT_IMAGE_CONTENT"
            or report.get("devkit_sha256") != DEVKIT_SHA256
            or report.get("class_mapping_authenticated") is not True
            or report.get("directory_inventory_complete") is not True
            or os.path.abspath(report.get("root", "")) != os.path.abspath(root)):
        raise ValueError("clean metadata authority is not verified for this root")
    data = Path(inventory_path).read_bytes()
    if digest(data) != report.get("relative_paths_sha256"):
        raise ValueError("inventory digest mismatch")
    names = data.decode("utf-8").splitlines()
    validate_names(names)
    if (len(names) != 50000 or len({p.split('/')[0] for p in names}) != 1000
            or names != sorted(names) or data != ("\n".join(names) + "\n").encode()):
        raise ValueError("expected exact sorted 50000-image/1000-class inventory")
    return names, {"report_sha256": digest(raw_report), "inventory_sha256": digest(data),
                   "devkit_sha256": DEVKIT_SHA256,
                   "claim": "labels and relative paths; not official image-byte provenance"}


def _open_directory_chain(path):
    """Open every absolute path component relative to an already open parent."""
    absolute = Path(os.path.abspath(path))
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError("directory chain contains symlink or non-directory") from exc
        raise


def write_fresh_output(path, data, dataset_roots):
    path = Path(os.path.abspath(path))
    for root in map(lambda p: Path(os.path.abspath(p)), dataset_roots):
        if path == root or root in path.parents:
            raise ValueError("output must be outside datasets")
    descriptor = _open_directory_chain(path.parent)
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=descriptor)
        with os.fdopen(fd, "wb") as output:
            output.write(data)
    finally:
        os.close(descriptor)


def _read_image_bytes(root, relative_path):
    """Open through directory descriptors; never follow a substituted symlink."""
    parts = Path(relative_path).parts
    if (not parts or Path(relative_path).is_absolute() or any(p in ("..", ".") for p in parts)):
        raise ValueError("unsafe relative image path")
    descriptor = _open_directory_chain(root)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        image_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        with os.fdopen(image_fd, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 32 * 1024**2:
                raise ValueError("image must be a bounded regular file")
            data = handle.read()
            after = os.fstat(handle.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("image changed during read")
            return data
    finally:
        os.close(descriptor)


def decode(data):
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG" or min(image.size) < 1:
            raise ValueError("not a valid JPEG image")
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        rgb.load()
        return rgb


def inspect_image(root, relative_path):
    data = _read_image_bytes(root, relative_path)
    image = decode(data)
    return {"relative_path": relative_path, "sha256": digest(data), "size_bytes": len(data),
            "width": image.width, "height": image.height, "decoded_rgb": True}


def read_bound_image(root, row):
    data = _read_image_bytes(root, row["relative_path"])
    if digest(data) != row["sha256"] or len(data) != row["size_bytes"]:
        raise ValueError("image digest or size no longer matches locked manifest")
    return decode(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--corruption-root", type=Path, required=True)
    parser.add_argument("--metadata-report", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Fixed engineering scope. No CLI mechanism for result-dependent selection.
    names, authority = load_inventory(args.clean_root, args.metadata_report, args.inventory)
    plan = plan_smoke(names, seed=0, source_count=32, target_count=104)
    if args.output.exists():
        parser.error("fresh output file required")
    for root in (args.clean_root, args.corruption_root):
        if Path(os.path.abspath(root)) in Path(os.path.abspath(args.output)).parents:
            parser.error("output must be outside datasets")
    rows = {"source": [], "target": []}
    for role in rows:
        for path in plan[role]:
            relative = path if role == "source" else "gaussian_noise/5/" + path
            root = args.clean_root if role == "source" else args.corruption_root
            row = inspect_image(root, relative)
            row["sample_id"] = Path(path).stem
            rows[role].append(row)
    record = {"schema": "kbound-real-image-smoke-panel-v1", "scope": "ENGINEERING_SMOKE_NOT_BENCHMARK",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "selection": "Python Random(0) shuffle of sorted verified inventory; first32 from25% source pool, first104 from remaining75%; differs from upstream torch shuffle",
              "seed": 0, "batch_size": 4, "condition": "gaussian_noise/5",
              "clean_root": os.path.abspath(args.clean_root), "corruption_root": os.path.abspath(args.corruption_root),
              "metadata_authority": authority, "source": rows["source"], "target": rows["target"],
              "builder_sha256": digest(Path(__file__).read_bytes()), "python": sys.version,
              "outcomes_read": False, "official_image_bytes_authenticated": False,
              "upstream_full_dataset_protocol": False}
    data = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    write_fresh_output(args.output, data, [args.clean_root, args.corruption_root])
    print(json.dumps({"manifest": str(args.output), "sha256": digest(data), "source": 32, "target": 104,
                      "status": "PASS_LOCAL_SELECTED_IMAGE_INTEGRITY_ONLY"}))


if __name__ == "__main__":
    main()
