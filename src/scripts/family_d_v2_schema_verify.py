"""Phase 2.2D — Eyecandies schema verification (no label inspection, no model run).

Scans each .tar archive WITHOUT reading anomaly mask file contents.
Counts per-(category, split, sample, modality) presence by parsing file
NAMES only.

Eyecandies file naming convention (per release 1.0.3):
  Eyecandies/<Category>/<split>/data/<NNN>_<kind>.{png,yaml,txt}
Sample-ID = the leading numeric prefix NNN inside the data/ directory.

Modalities detected:
  rgb     : '<NNN>_image_<view>.png' (6 views per sample)
  depth   : '<NNN>_depth.png'
  normal  : '<NNN>_normals.png'
  anomaly_mask (presence only; NEVER opened) :
    '<NNN>_mask.png', '<NNN>_bumps_mask.png', '<NNN>_dents_mask.png',
    '<NNN>_colors_mask.png', '<NNN>_normals_mask.png'

Writes:
  experiments/phase2/family_d/eyecandies_schema_verification.json
  experiments/phase2/family_d/eyecandies_archive_inventory.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT / "data" / "raw" / "eyecandies" / "_archives"
OUT_DIR = ROOT / "experiments" / "phase2" / "family_d"

SPLIT_TOKENS = {"train", "training", "val", "validation",
                 "test", "test_public", "test_private"}


def _split_from_path(parts: list[str]) -> str:
    for p in parts:
        if p.lower() in SPLIT_TOKENS:
            t = p.lower()
            return "val" if t == "validation" else ("train" if t == "training" else t)
    return "unknown"


# Mask filename matchers — anomaly-mask files are explicitly enumerated
# from the Eyecandies file layout. Order matters: most-specific first.
_MASK_PATTERN = re.compile(
    r"^(\d+)_("
    r"mask|bumps_mask|dents_mask|colors_mask|normals_mask"
    r")\.(png|jpg|jpeg|tif|tiff|npy)$",
    re.IGNORECASE,
)
_RGB_PATTERN = re.compile(r"^(\d+)_image_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
_DEPTH_PATTERN = re.compile(r"^(\d+)_depth\.(png|jpg|jpeg|tif|tiff|npy)$", re.IGNORECASE)
_NORMAL_PATTERN = re.compile(r"^(\d+)_normals\.(png|jpg|jpeg|tif|tiff|npy)$", re.IGNORECASE)


def _categorise(basename: str) -> tuple[str | None, str | None]:
    """Return (modality, sample_id) or (None, None) if not a known data file."""
    if _MASK_PATTERN.match(basename):
        return ("anomaly_mask", _MASK_PATTERN.match(basename).group(1))
    m = _RGB_PATTERN.match(basename)
    if m:
        return ("rgb", m.group(1))
    m = _DEPTH_PATTERN.match(basename)
    if m:
        return ("depth", m.group(1))
    m = _NORMAL_PATTERN.match(basename)
    if m:
        return ("normal", m.group(1))
    return (None, None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", default=None)
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.only:
        targets = [ARCHIVE_DIR / f"{c.strip()}.tar" for c in args.only.split(",")]
    else:
        targets = sorted(p for p in ARCHIVE_DIR.glob("*.tar") if not p.name.startswith("._"))

    report = {}
    inventory_rows = []
    for archive_path in targets:
        if not archive_path.exists():
            print(f"[{archive_path.name}] missing; skipping")
            continue
        cat = archive_path.stem
        print(f"[{cat}] scanning archive (no extraction)...", flush=True)

        # split -> { 'rgb_samples', 'depth_samples', 'normal_samples', 'mask_files' }
        per_split = defaultdict(lambda: {
            "rgb_samples": set(),
            "depth_samples": set(),
            "normal_samples": set(),
            "mask_files_NOT_INSPECTED": 0,
            "rgb_views_per_sample": defaultdict(int),
        })
        member_count = 0
        with tarfile.open(archive_path, "r") as tf:
            for m in tf:
                member_count += 1
                if not m.isfile():
                    continue
                parts = m.name.split("/")
                split = _split_from_path(parts)
                basename = parts[-1]
                modality, sample_id = _categorise(basename)
                if modality is None:
                    continue
                if modality == "anomaly_mask":
                    per_split[split]["mask_files_NOT_INSPECTED"] += 1
                    continue
                if modality == "rgb":
                    per_split[split]["rgb_samples"].add(sample_id)
                    per_split[split]["rgb_views_per_sample"][sample_id] += 1
                elif modality == "depth":
                    per_split[split]["depth_samples"].add(sample_id)
                elif modality == "normal":
                    per_split[split]["normal_samples"].add(sample_id)

        splits_out = {}
        for split, agg in per_split.items():
            rgb = agg["rgb_samples"]
            depth = agg["depth_samples"]
            normal = agg["normal_samples"]
            paired = rgb & depth
            unpaired_rgb = rgb - depth
            unpaired_depth = depth - rgb
            views_per_sample = agg["rgb_views_per_sample"]
            views_count_set = sorted(set(views_per_sample.values()))
            splits_out[split] = {
                "rgb_sample_count": len(rgb),
                "depth_sample_count": len(depth),
                "normal_sample_count": len(normal),
                "rgb_depth_paired_count": len(paired),
                "rgb_only_count": len(unpaired_rgb),
                "depth_only_count": len(unpaired_depth),
                "anomaly_mask_file_count_NOT_INSPECTED": agg["mask_files_NOT_INSPECTED"],
                "rgb_views_per_sample_distribution": views_count_set,
            }
            inventory_rows.append({
                "category": cat, "split": split,
                "rgb_samples": len(rgb), "depth_samples": len(depth),
                "normal_samples": len(normal),
                "rgb_depth_paired": len(paired),
                "anomaly_mask_files_present_not_inspected": agg["mask_files_NOT_INSPECTED"],
                "rgb_views_per_sample": ",".join(str(v) for v in views_count_set),
            })
        report[cat] = {
            "tar_member_count_total": member_count,
            "splits": splits_out,
        }
        for split, info in splits_out.items():
            print(f"  {split}: rgb_samples={info['rgb_sample_count']}  depth_samples={info['depth_sample_count']}  "
                  f"paired={info['rgb_depth_paired_count']}  normal={info['normal_sample_count']}  "
                  f"rgb_views_per_sample={info['rgb_views_per_sample_distribution']}  "
                  f"mask_files_not_inspected={info['anomaly_mask_file_count_NOT_INSPECTED']}", flush=True)

    out_json = OUT_DIR / "eyecandies_schema_verification.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_json}")

    out_csv = OUT_DIR / "eyecandies_archive_inventory.csv"
    with out_csv.open("w", newline="") as f:
        fields = ["category", "split", "rgb_samples", "depth_samples", "normal_samples",
                  "rgb_depth_paired", "anomaly_mask_files_present_not_inspected",
                  "rgb_views_per_sample"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in inventory_rows:
            w.writerow(r)
    print(f"wrote {out_csv}")

    # Invariant checks
    #
    # Note on anomaly-mask file presence in train/val:
    # Per official Eyecandies docs train/val are anomaly-free. However the
    # synthetic pipeline emits placeholder mask files for every sample
    # (likely zero-valued in anomaly-free splits) so all samples share the
    # same on-disk layout. The held-out invariant for this protocol is
    # *no protocol step reads anomaly mask files during selection/tuning*,
    # NOT "mask files do not exist on disk". The base RGA primary method
    # does not access mask files; the protocol YAML's
    # `selection_used_test_metrics=false` invariant enforces this at the
    # protocol level. We record mask file COUNTS for transparency without
    # opening any mask file in this script.
    failures = []
    for cat, info in report.items():
        for split, s in info["splits"].items():
            if split in ("train", "val"):
                if s["rgb_sample_count"] == 0 or s["depth_sample_count"] == 0:
                    failures.append(f"{cat}/{split}: empty rgb or depth")
                if s["rgb_depth_paired_count"] != s["rgb_sample_count"]:
                    failures.append(
                        f"{cat}/{split}: rgb={s['rgb_sample_count']} paired={s['rgb_depth_paired_count']}"
                    )
                if s["rgb_views_per_sample_distribution"] != [6]:
                    failures.append(
                        f"{cat}/{split}: expected 6 RGB views per sample; "
                        f"got distribution {s['rgb_views_per_sample_distribution']}"
                    )
    if failures:
        print(f"INVARIANT FAILURES: {len(failures)}")
        for fl in failures[:20]:
            print(f"  FAIL: {fl}")
        return 2
    print("All schema invariants pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
