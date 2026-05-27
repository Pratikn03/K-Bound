"""Download a small, auditable MVTec 3D-AD RGB/XYZ subset from Kaggle.

This helper is intentionally scoped to a category subset so a paired benchmark
can run locally without downloading the full ~25GB dataset archive. It preserves
the directory layout expected by ``prepare_mvtec3d_fusion_benchmark.py``:

  data/raw/mvtec3d/<category>/<split>/<defect_type>/{rgb,xyz}/<id>

The Kaggle mirror used by this helper lists an unknown license. The official
MVTec 3D-AD page states CC BY-NC-SA 4.0 terms; use this only for non-commercial
research and cite the MVTec 3D-AD paper.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

DATASET = "gengchensun/mvtec-3d-ad"
KAGGLE_ROOT = "mvtec_3d_anomaly_detection"


def _stem_key(path: str) -> tuple[str, str, str, str, str] | None:
    parts = path.split("/")
    if len(parts) != 6 or parts[0] != KAGGLE_ROOT:
        return None
    _, category, split, defect, modality, filename = parts
    if modality not in {"rgb", "xyz"}:
        return None
    return category, split, defect, filename.rsplit(".", 1)[0], modality


def _sorted_stems(files: Iterable[str], category: str, split: str, defect: str) -> list[str]:
    modalities_by_stem: dict[str, set[str]] = defaultdict(set)
    for path in files:
        key = _stem_key(path)
        if key is None:
            continue
        cat, spl, defect_type, stem, modality = key
        if cat == category and spl == split and defect_type == defect:
            modalities_by_stem[stem].add(modality)
    return sorted(stem for stem, modalities in modalities_by_stem.items() if {"rgb", "xyz"}.issubset(modalities))


def _paired_paths(category: str, split: str, defect: str, stems: Iterable[str]) -> list[str]:
    out: list[str] = []
    for stem in stems:
        out.append(f"{KAGGLE_ROOT}/{category}/{split}/{defect}/rgb/{stem}.png")
        out.append(f"{KAGGLE_ROOT}/{category}/{split}/{defect}/xyz/{stem}.tiff")
    return out


def select_subset_files(
    files: Iterable[str],
    category: str,
    max_train_good: int = 60,
    max_validation_good: int = 12,
    max_test_good: int = 12,
    max_test_per_defect: int = 12,
) -> list[str]:
    """Select paired RGB/XYZ files for a reproducible category subset."""
    file_list = sorted(files)
    selected: list[str] = []

    for split, defect, limit in [
        ("train", "good", max_train_good),
        ("validation", "good", max_validation_good),
        ("test", "good", max_test_good),
    ]:
        stems = _sorted_stems(file_list, category, split, defect)[: max(0, int(limit))]
        selected.extend(_paired_paths(category, split, defect, stems))

    test_defects = sorted(
        {
            key[2]
            for path in file_list
            if (key := _stem_key(path)) is not None and key[0] == category and key[1] == "test" and key[2] != "good"
        }
    )
    for defect in test_defects:
        stems = _sorted_stems(file_list, category, "test", defect)[: max(0, int(max_test_per_defect))]
        selected.extend(_paired_paths(category, "test", defect, stems))
    return selected


def _list_dataset_files(dataset: str) -> list[str]:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    files: list[str] = []
    token = None
    while True:
        response = api.dataset_list_files(dataset, page_token=token, page_size=200)
        files.extend(file.name for file in response.files)
        token = response.next_page_token
        if not token:
            return files


def download_subset(
    output_root: Path,
    category: str,
    dataset: str = DATASET,
    max_train_good: int = 60,
    max_validation_good: int = 12,
    max_test_good: int = 12,
    max_test_per_defect: int = 12,
    force: bool = False,
) -> dict:
    from kaggle.api.kaggle_api_extended import KaggleApi

    files = _list_dataset_files(dataset)
    selected = select_subset_files(
        files,
        category=category,
        max_train_good=max_train_good,
        max_validation_good=max_validation_good,
        max_test_good=max_test_good,
        max_test_per_defect=max_test_per_defect,
    )
    if not selected:
        raise FileNotFoundError(f"No paired files selected for category {category!r}")

    api = KaggleApi()
    api.authenticate()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for file_name in selected:
            relative = Path(*Path(file_name).parts[1:])
            destination = output_root / relative
            if destination.exists() and not force:
                skipped += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            api.dataset_download_file(dataset, file_name, path=tmp, force=True, quiet=True)
            downloaded_file = tmp / Path(file_name).name
            if not downloaded_file.exists():
                raise FileNotFoundError(f"Kaggle did not produce expected file {downloaded_file}")
            shutil.move(str(downloaded_file), destination)
            downloaded += 1

    metadata = {
        "dataset": dataset,
        "source": "Kaggle mirror of MVTec 3D-AD",
        "official_license_note": "Official MVTec 3D-AD data is CC BY-NC-SA 4.0 for non-commercial research.",
        "category": category,
        "selected_files": len(selected),
        "downloaded_files": downloaded,
        "skipped_existing_files": skipped,
        "output_root": str(output_root),
        "limits": {
            "max_train_good": max_train_good,
            "max_validation_good": max_validation_good,
            "max_test_good": max_test_good,
            "max_test_per_defect": max_test_per_defect,
        },
    }
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small paired MVTec 3D-AD subset from Kaggle")
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/mvtec3d"))
    parser.add_argument("--category", default="bagel")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--max-train-good", type=int, default=60)
    parser.add_argument("--max-validation-good", type=int, default=12)
    parser.add_argument("--max-test-good", type=int, default=12)
    parser.add_argument("--max-test-per-defect", type=int, default=12)
    parser.add_argument(
        "--metadata", type=Path, default=Path("experiments/fusion/mvtec3d_download_subset_metadata.json")
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    metadata = download_subset(
        output_root=args.output_root,
        category=args.category,
        dataset=args.dataset,
        max_train_good=args.max_train_good,
        max_validation_good=args.max_validation_good,
        max_test_good=args.max_test_good,
        max_test_per_defect=args.max_test_per_defect,
        force=args.force,
    )
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
