#!/usr/bin/env python3
"""Download and extract MulSen-AD for external M2 v2 (Hugging Face)."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ID = "orgjy314159/MulSen_AD"
ZIP_CANDIDATES = ("MulSen_AD.zip", "MulSen_AD_new.zip", "MulSen_AD.rar")
SEAL_YAML = "research_lock/M2_EXTERNAL_SEALED_v2.yaml"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "elara_master_c").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _category_ready(local_dir: Path, category: str) -> bool:
    rgb_train = local_dir / category / "RGB" / "train"
    ir_train = local_dir / category / "Infrared" / "train"
    if not rgb_train.is_dir() or not ir_train.is_dir():
        return False
    return any(rgb_train.glob("*.png")) and any(ir_train.glob("*.png"))


def _load_categories(root: Path) -> list[str]:
    seal = yaml.safe_load((root / SEAL_YAML).read_text(encoding="utf-8"))
    proto = seal["heldout_protocol"]
    return sorted(set(proto["train_categories"]) | set(proto["test_categories"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-name", default=None, help="Override HF archive filename")
    args = parser.parse_args()

    root = _repo_root()
    seal = yaml.safe_load((root / SEAL_YAML).read_text(encoding="utf-8"))
    local_dir = root / seal["source"]["local_root"]
    categories = _load_categories(root)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub required") from exc

    archive_dir = local_dir.parent / "_downloads_mulsen"
    archive_dir.mkdir(parents=True, exist_ok=True)
    zip_name = args.zip_name
    if zip_name is None:
        zip_name = ZIP_CANDIDATES[0]
    print(f"Downloading {zip_name} from {REPO_ID}")
    try:
        archive_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=zip_name,
                local_dir=str(archive_dir),
            )
        )
    except Exception:
        if zip_name == ZIP_CANDIDATES[0] and len(ZIP_CANDIDATES) > 1:
            zip_name = ZIP_CANDIDATES[1]
            print(f"Retrying with {zip_name}")
            archive_path = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=zip_name,
                    local_dir=str(archive_dir),
                )
            )
        else:
            raise

    local_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(local_dir)
    elif archive_path.suffix.lower() == ".rar":
        try:
            import rarfile  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "Install rarfile and unrar, or download MulSen_AD.zip instead of .rar"
            ) from exc
        with rarfile.RarFile(archive_path) as rf:
            rf.extractall(local_dir)
    else:
        raise SystemExit(f"Unsupported archive type: {archive_path}")

    # HF zip may nest MulSen_AD/MulSen_AD/ — flatten one level if needed.
    nested = local_dir / "MulSen_AD"
    if nested.is_dir() and not _category_ready(local_dir, categories[0]):
        for child in nested.iterdir():
            target = local_dir / child.name
            if child.is_dir() and not target.exists():
                child.rename(target)

    missing = [c for c in categories if not _category_ready(local_dir, c)]
    record = {
        "source": REPO_ID,
        "archive": str(archive_path.relative_to(root)),
        "archive_sha256": _sha256_file(archive_path),
        "extracted_to": str(local_dir.relative_to(root)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "categories_expected": categories,
        "categories_missing": missing,
    }
    out = root / seal["artifacts"]["acquisition_record"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    if missing:
        print(f"WARNING: missing categories after extract: {missing}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
