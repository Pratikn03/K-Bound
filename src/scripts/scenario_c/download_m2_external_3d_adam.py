#!/usr/bin/env python3
"""Download the sealed external M2 dataset: 3D-ADAM anomalib (Hugging Face).

Default strategy downloads category folders needed for the D4 held-out split
(train + test categories in research_lock/M2_EXTERNAL_SEALED_v1.yaml).

Alternative: --zip downloads and extracts adam3d_cropped.zip (~5.2 GB).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ID = "pmchard/3D-ADAM_anomalib"
ZIP_NAME = "adam3d_cropped.zip"
SEAL_YAML = "research_lock/M2_EXTERNAL_SEALED_v1.yaml"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _load_categories(root: Path) -> tuple[list[str], list[str]]:
    seal = yaml.safe_load((root / SEAL_YAML).read_text(encoding="utf-8"))
    proto = seal["heldout_protocol"]
    return list(proto["train_categories"]), list(proto["test_categories"])


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _category_complete(local_dir: Path, category: str) -> bool:
    base = local_dir / category
    if not base.is_dir():
        return False
    train_rgb = base / "train" / "good" / "rgb"
    return train_rgb.is_dir() and any(train_rgb.glob("*.png"))


def _download_categories(root: Path, local_dir: Path, categories: list[str]) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with: .venv/bin/pip install huggingface_hub"
        ) from exc

    patterns = [f"{cat}/**" for cat in categories]
    print(f"Downloading {len(categories)} categories from {REPO_ID} -> {local_dir}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=patterns,
        local_dir_use_symlinks=False,
    )


def _download_zip(root: Path, local_dir: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub required") from exc

    archive_dir = local_dir.parent / "_downloads"
    archive_dir.mkdir(parents=True, exist_ok=True)
    zip_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=ZIP_NAME,
            local_dir=str(archive_dir),
        )
    )
    print(f"Extracting {zip_path} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(local_dir)
    record = {
        "zip_path": str(zip_path.relative_to(root)),
        "zip_sha256": _sha256_file(zip_path),
        "extracted_to": str(local_dir.relative_to(root)),
    }
    (archive_dir / "zip_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Override local dataset root (default: data/raw/3d_adam_anomalib)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Download adam3d_cropped.zip instead of per-category snapshots",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if categories appear present",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated subset of categories (default: all train+test from seal yaml)",
    )
    args = parser.parse_args()

    root = _repo_root()
    train_cats, test_cats = _load_categories(root)
    all_cats = sorted(set(train_cats) | set(test_cats))
    if args.categories:
        all_cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    local_dir = Path(args.local_dir) if args.local_dir else root / "data/raw/3d_adam_anomalib"

    if not args.force and all(_category_complete(local_dir, c) for c in all_cats):
        print(f"All {len(all_cats)} categories already present under {local_dir}")
        return 0

    if args.zip:
        _download_zip(root, local_dir)
    else:
        missing = [c for c in all_cats if not _category_complete(local_dir, c)]
        _download_categories(root, local_dir, missing)

    still_missing = [c for c in all_cats if not _category_complete(local_dir, c)]
    if still_missing:
        print(f"ERROR: still missing categories after download: {still_missing}", file=sys.stderr)
        return 1

    record = {
        "dataset_id": "m2_3d_adam_anomalib_external",
        "source": REPO_ID,
        "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "local_root": str(local_dir.relative_to(root)),
        "train_categories": train_cats,
        "test_categories": test_cats,
        "n_categories": len(all_cats),
        "method": "zip" if args.zip else "snapshot_categories",
    }
    out = root / "experiments/fusion/m2_external_3d_adam_acquisition.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Wrote acquisition record -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
