"""Strict image identity indexes for the official WILDS source populations.

An index is input evidence, not upstream download authentication or proof of
independent KGA partitions. Target class-label columns are never interpreted
or emitted. Missing/undecodable files cannot be dropped or substituted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import time
from collections import Counter
from pathlib import Path, PurePosixPath

OFFICIAL_COUNTS = {
    "rxrx1": {"train": 40612, "id_test": 40612, "val": 9854, "test": 34432},
    "camelyon17": {"train": 302436, "id_val": 33560, "val": 34904, "test": 85054},
    "iwildcam": {"train": 129809, "id_val": 7314, "id_test": 8154, "val": 14961, "test": 42791},
}
REQUIRED_COLUMNS = {
    "rxrx1": {"dataset", "experiment", "plate", "well", "site"},
    "camelyon17": {"patient", "node", "x_coord", "y_coord", "center", "slide", "split"},
    "iwildcam": {"filename", "split", "location_remapped", "sequence_remapped"},
}


class PopulationError(ValueError):
    """The exact declared input population could not be verified."""


def _no_symlinks(path: Path) -> Path:
    path = path.absolute()
    if ".." in path.parts:
        raise PopulationError(f"path contains parent traversal: {path}")
    for item in (path, *path.parents):
        if item.is_symlink():
            raise PopulationError(f"symlink input/output path is not permitted: {item}")
    return path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token(value: str, field: str, numeric: bool = False) -> str:
    pattern = r"[0-9]+" if numeric else r"[A-Za-z0-9_-]+"
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise PopulationError(f"invalid path/group component {field}: {value!r}")
    return value


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or any(ord(c) < 32 for c in value):
        raise PopulationError("invalid image path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise PopulationError(f"unsafe image path: {value}")
    return value


def metadata_entries(family: str, root: Path):
    """Yield original row IDs and official split/group identities, without labels."""
    if family not in OFFICIAL_COUNTS:
        raise PopulationError(f"unsupported family: {family}")
    root = _no_symlinks(Path(root))
    metadata = _no_symlinks(root / "metadata.csv")
    with metadata.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not REQUIRED_COLUMNS[family].issubset(fields):
            raise PopulationError("metadata columns are missing or duplicated")
        for index, row in enumerate(reader):
            if None in row or any(row.get(k) is None for k in REQUIRED_COLUMNS[family]):
                raise PopulationError(f"malformed metadata row {index}")
            if family == "rxrx1":
                experiment = _token(row["experiment"], "experiment")
                plate = _token(row["plate"], "plate", True)
                well = _token(row["well"], "well")
                site = _token(row["site"], "site", True)
                if site not in {"1", "2"}:
                    raise PopulationError(f"invalid RxRx1 site at row {index}")
                split = row["dataset"]
                if split not in {"train", "val", "test"}:
                    raise PopulationError(f"invalid RxRx1 dataset split at row {index}")
                if split == "train" and site == "2":
                    split = "id_test"
                relative = f"images/{experiment}/Plate{plate}/{well}_s{site}.png"
                group = {"experiment": experiment, "plate": plate, "well": well}
            elif family == "camelyon17":
                patient, node, x, y, center, slide = [
                    _token(row[key], key, True)
                    for key in ("patient", "node", "x_coord", "y_coord", "center", "slide")
                ]
                split = "val" if center == "1" else "test" if center == "2" else {
                    "0": "train", "1": "id_val"
                }.get(row["split"])
                relative = (f"patches/patient_{patient}_node_{node}/"
                            f"patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png")
                group = {"center": center, "slide": slide}
            else:
                relative = "train/" + _relative(row["filename"])
                split = row["split"]
                group = {"location": _token(row["location_remapped"], "location", True),
                         "sequence": _token(row["sequence_remapped"], "sequence", True)}
            if split not in OFFICIAL_COUNTS[family]:
                raise PopulationError(f"invalid official split at row {index}: {split!r}")
            yield {"row_id": index, "path": _relative(relative), "split": split, "group": group}


def _image_identity(path: Path) -> dict:
    # Decode and hash exactly the same bytes; reject truncation instead of PIL's
    # opt-in LOAD_TRUNCATED_IMAGES behavior used by some historical runners.
    from PIL import Image, ImageFile

    try:
        path = _no_symlinks(path)
        before_open = os.lstat(path)
        if not stat.S_ISREG(before_open.st_mode) or before_open.st_size <= 0:
            raise PopulationError(f"image is not a nonempty regular file: {path}")
        fd = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                raise PopulationError(f"image is not a nonempty regular file: {path}")
            if (before_open.st_dev, before_open.st_ino) != (before.st_dev, before.st_ino):
                raise PopulationError(f"image changed before open: {path}")
            data = stream.read()
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino
            ) or len(data) != before.st_size:
                raise PopulationError(f"image changed during read: {path}")
        old_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                original_mode = image.mode
                width, height = image.size
                converted = image.convert("RGB")
                converted.load()
                if converted.mode != "RGB" or converted.size != image.size:
                    raise PopulationError(f"image cannot be converted to reference RGB input: {path}")
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = old_truncated
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise PopulationError(f"image integrity failure at {path}: {exc}") from exc
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "width": width, "height": height, "original_mode": original_mode}


def _write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_population(family: str, root: Path, metadata_sha256: str, output_dir: Path,
                     expected_counts: dict | None = None) -> dict:
    """Index every declared file, failing on any missing, corrupt or duplicate row.

    Custom expected_counts support explicit subpopulation indexes. Only a scan
    matching the complete official profile earns reference_population_complete.
    The command-line entry point always requires the full official profile.
    """
    if family not in OFFICIAL_COUNTS:
        raise PopulationError(f"unsupported family: {family}")
    root, out = _no_symlinks(Path(root)), _no_symlinks(Path(output_dir))
    metadata = _no_symlinks(root / "metadata.csv")
    if (not isinstance(metadata_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", metadata_sha256)
            or not metadata.is_file() or _sha(metadata) != metadata_sha256):
        raise PopulationError("metadata SHA-256 mismatch or missing metadata")
    expected = dict(OFFICIAL_COUNTS[family]) if expected_counts is None else expected_counts
    if (not isinstance(expected, dict) or not expected
            or any(k not in OFFICIAL_COUNTS[family] or type(v) is not int or v < 1
                   for k, v in expected.items())):
        raise PopulationError("invalid expected counts")
    try:
        out.mkdir(parents=False, exist_ok=False)
    except OSError as exc:
        raise PopulationError(f"fresh output directory required: {out}") from exc
    start = time.monotonic()
    counts, seen = Counter(), set()
    receipt = {"schema": "kbound_reference_population_v1", "family": family,
               "root": str(root), "metadata_sha256": metadata_sha256,
               "expected_counts": expected, "complete": False,
               "reference_population_complete": False,
               "upstream_download_authenticated": False,
               "independent_kga_partitions_verified": False}
    index = out / "image-index.jsonl"
    try:
        with index.open("x", encoding="utf-8") as stream:
            for row in metadata_entries(family, root):
                if row["path"] in seen:
                    raise PopulationError(f"duplicate metadata image path: {row['path']}")
                seen.add(row["path"])
                row.update(_image_identity(root / row["path"]))
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                counts[row["split"]] += 1
                if len(seen) % 5000 == 0:
                    stream.flush()
                    print(json.dumps({"indexed": len(seen), "family": family,
                                      "wall_seconds": round(time.monotonic() - start, 2)}), flush=True)
            stream.flush()
            os.fsync(stream.fileno())
        if dict(counts) != expected:
            raise PopulationError(f"population counts mismatch: expected {expected}, observed {dict(counts)}")
        if _sha(metadata) != metadata_sha256:
            raise PopulationError("metadata SHA-256 changed during scan")
        receipt.update(complete=True, reference_population_complete=expected == OFFICIAL_COUNTS[family],
                       index_sha256=_sha(index), index_file="image-index.jsonl")
    except (PopulationError, OSError, csv.Error, UnicodeError) as exc:
        receipt.update(counts=dict(counts), error=str(exc), wall_seconds=time.monotonic() - start)
        _write_new_json(out / "completion.json", receipt)
        raise PopulationError(str(exc)) from exc
    receipt.update(counts=dict(counts), wall_seconds=time.monotonic() - start)
    _write_new_json(out / "completion.json", receipt)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(OFFICIAL_COUNTS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_population(args.family, args.root, args.metadata_sha256, args.output_dir)
    except (PopulationError, OSError) as exc:
        print(json.dumps({"complete": False, "error": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
