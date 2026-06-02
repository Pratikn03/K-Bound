#!/usr/bin/env python3
"""Prepare Real-IAD D3 inputs for D15 natural headroom/degradation audit.

This script is protocol-driven:
- all downloaded Real-IAD D3 categories are used by default;
- official train OK samples define one-class references and quality baselines;
- official validation/test splits are preserved;
- no test labels are used for score calibration, threshold selection, or routing.

The output CSV is consumed by `run_realiad_d3_headroom_audit.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from sklearn.metrics import pairwise_distances, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]


MODALITY_PATH_FIELDS = {
    "rgb": "image_path",
    "ps": "ps_path",
    "xyz": "xyz_path",
}


@dataclass(frozen=True)
class D3Row:
    category: str
    sample_id: str
    split: str
    modality: str
    label: int
    anomaly_class: str
    zip_member: str


def _extract_jsons(root: Path) -> Path:
    json_dir = root / "jsons" / "realiad_d3_jsons"
    if json_dir.is_dir() and any(json_dir.glob("*.json")):
        return json_dir
    zip_path = root / "realiad_d3_jsons.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"missing {zip_path}")
    (root / "jsons").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.startswith("realiad_d3_jsons/") and member.endswith(".json") and "/._" not in member:
                zf.extract(member, root / "jsons")
    return json_dir


def _sample_id(category: str, entry: dict[str, Any]) -> str:
    p = str(entry["image_path"])
    return f"{category}/{str(Path(p).with_suffix(''))}"


def _load_rows(
    root: Path,
    categories_arg: list[str] | None,
    exclude_categories_arg: list[str] | None = None,
) -> tuple[list[D3Row], list[str]]:
    json_dir = _extract_jsons(root)
    zip_dir = root / "realiad_d3_raw"
    zips = {p.stem: p for p in zip_dir.glob("*.zip") if not p.name.startswith("._")}
    json_categories = {p.stem for p in json_dir.glob("*.json") if not p.name.startswith("._")}
    categories = sorted(set(zips) & json_categories)
    if categories_arg:
        requested = set(categories_arg)
        missing = sorted(requested - set(categories))
        if missing:
            raise FileNotFoundError(f"requested D3 categories are not downloaded: {missing}")
        categories = sorted(requested)
    if exclude_categories_arg:
        categories = [c for c in categories if c not in set(exclude_categories_arg)]
    if not categories:
        raise FileNotFoundError(f"no completed Real-IAD D3 category zips found under {zip_dir}")

    rows: list[D3Row] = []
    for category in categories:
        doc = json.loads((json_dir / f"{category}.json").read_text(encoding="utf-8"))
        prefix = str(doc.get("meta", {}).get("prefix", f"{category}/"))
        for split in ("train", "validation", "test"):
            for entry in doc.get(split, []):
                anomaly_class = str(entry.get("anomaly_class", ""))
                label = 0 if anomaly_class == "OK" else 1
                sid = _sample_id(category, entry).replace("/", "__")
                for modality, field in MODALITY_PATH_FIELDS.items():
                    path = entry.get(field)
                    if path:
                        rows.append(
                            D3Row(
                                category=category,
                                sample_id=sid,
                                split=split,
                                modality=modality,
                                label=label,
                                anomaly_class=anomaly_class,
                                zip_member=f"{prefix}{path}",
                            )
                        )
    return rows, categories


def _read_rgb_like(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as f:
        data = f.read()
    img = Image.open(BytesIO(data)).convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _read_xyz(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as f:
        arr = tifffile.imread(BytesIO(f.read()))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    # Keep geometric structure but reduce memory and normalize later in features.
    arr = cv2.resize(arr, (96, 96), interpolation=cv2.INTER_AREA)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr.astype(np.float32)


def _hist_u8(gray: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([(gray * 255).astype(np.uint8)], [0], None, [16], [0, 256]).reshape(-1)
    return hist / max(float(hist.sum()), 1.0)


def _image_features(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = img.astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    chans = [rgb[:, :, i] for i in range(3)] + [gray]
    stats: list[float] = []
    for ch in chans:
        stats.extend(
            [
                float(ch.mean()),
                float(ch.std()),
                float(np.percentile(ch, 10)),
                float(np.percentile(ch, 50)),
                float(np.percentile(ch, 90)),
            ]
        )
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    feature = np.concatenate(
        [
            np.asarray(stats, dtype=np.float32),
            _hist_u8(gray).astype(np.float32),
            cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32),
        ]
    )
    quality = np.asarray(
        [
            float(gray.mean()),
            float(gray.std()),
            float(lap.var()),
            float(mag.mean()),
            float(mag.std()),
            float((gray < 0.02).mean()),
            float((gray > 0.98).mean()),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=0.0), np.nan_to_num(
        quality,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )


def _xyz_features(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    valid_ratio = float(finite.mean())
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    flat = x.reshape(-1, x.shape[-1])
    stats: list[float] = [valid_ratio]
    for ci in range(x.shape[-1]):
        ch = flat[:, ci]
        stats.extend(
            [
                float(ch.mean()),
                float(ch.std()),
                float(np.percentile(ch, 10)),
                float(np.percentile(ch, 50)),
                float(np.percentile(ch, 90)),
            ]
        )
    z = x[:, :, min(2, x.shape[-1] - 1)]
    z_min = float(np.percentile(z, 1))
    z_max = float(np.percentile(z, 99))
    z_norm = np.clip((z - z_min) / max(z_max - z_min, 1e-6), 0.0, 1.0)
    gx = cv2.Sobel(z_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z_norm, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    feature = np.concatenate(
        [
            np.asarray(stats, dtype=np.float32),
            _hist_u8(z_norm).astype(np.float32),
            cv2.resize(z_norm, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32),
        ]
    )
    quality = np.asarray(
        [
            valid_ratio,
            float(z_norm.mean()),
            float(z_norm.std()),
            float(mag.mean()),
            float(mag.std()),
            float(np.percentile(mag, 90)),
            float((z_norm <= 0.001).mean()),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=0.0), np.nan_to_num(
        quality,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )


def _score_and_reliability(
    *,
    rows: list[D3Row],
    features: dict[tuple[str, str, str], np.ndarray],
    qualities: dict[tuple[str, str, str], np.ndarray],
) -> tuple[dict[tuple[str, str, str], float], dict[tuple[str, str, str], float], dict[tuple[str, str, str], float]]:
    scores: dict[tuple[str, str, str], float] = {}
    reliabilities: dict[tuple[str, str, str], float] = {}
    quality_distances: dict[tuple[str, str, str], float] = {}
    for category in sorted({r.category for r in rows}):
        for modality in MODALITY_PATH_FIELDS:
            group = [r for r in rows if r.category == category and r.modality == modality]
            train = [r for r in group if r.split == "train" and r.label == 0]
            if not train:
                continue
            train_keys = [(r.category, r.sample_id, r.modality) for r in train]
            eval_keys = [(r.category, r.sample_id, r.modality) for r in group]

            ref = np.stack([features[k] for k in train_keys], axis=0)
            ref_mean = ref.mean(axis=0, keepdims=True)
            ref_std = ref.std(axis=0, keepdims=True) + 1e-6
            ref_z = (ref - ref_mean) / ref_std
            eval_z = np.stack([(features[k] - ref_mean.reshape(-1)) / ref_std.reshape(-1) for k in eval_keys], axis=0)
            d = pairwise_distances(eval_z, ref_z, metric="euclidean", n_jobs=1)
            raw = np.sort(d, axis=1)[:, : min(5, d.shape[1])].mean(axis=1)
            train_raw = raw[: len(train_keys)]
            score_scale = max(
                float(np.percentile(train_raw, 95)),
                float(np.median(train_raw)),
                float(np.std(train_raw)),
                1e-6,
            )
            norm_score = raw / (raw + score_scale)

            qref = np.stack([qualities[k] for k in train_keys], axis=0)
            qmean = qref.mean(axis=0, keepdims=True)
            qstd = qref.std(axis=0, keepdims=True) + 1e-6
            qref_z = (qref - qmean) / qstd
            qeval_z = np.stack([(qualities[k] - qmean.reshape(-1)) / qstd.reshape(-1) for k in eval_keys], axis=0)
            qraw = np.linalg.norm(qeval_z, axis=1)
            qtrain = np.linalg.norm(qref_z, axis=1)
            quality_scale = max(
                float(np.percentile(qtrain, 95)),
                float(np.median(qtrain)),
                float(np.std(qtrain)),
                1e-6,
            )
            rel = quality_scale / (qraw + quality_scale)

            for key, value, qdist, qrel in zip(eval_keys, norm_score, qraw, rel, strict=True):
                scores[key] = float(value)
                quality_distances[key] = float(qdist)
                reliabilities[key] = float(qrel)
    return scores, reliabilities, quality_distances


def _apply_validation_score_orientation(
    rows: list[D3Row],
    scores: dict[tuple[str, str, str], float],
) -> dict[tuple[str, str], dict[str, Any]]:
    orientations: dict[tuple[str, str], dict[str, Any]] = {}
    for category in sorted({r.category for r in rows}):
        for modality in MODALITY_PATH_FIELDS:
            group = [r for r in rows if r.category == category and r.modality == modality]
            val = [r for r in group if r.split == "validation"]
            labels = np.asarray([r.label for r in val if (r.category, r.sample_id, r.modality) in scores], dtype=int)
            values = np.asarray(
                [scores[(r.category, r.sample_id, r.modality)] for r in val if (r.category, r.sample_id, r.modality) in scores],
                dtype=float,
            )
            auc: float | None = None
            direction = "native"
            if len(labels) >= 2 and len(np.unique(labels)) == 2:
                auc = float(roc_auc_score(labels, values))
                if auc < 0.5:
                    direction = "inverted"
                    for row in group:
                        key = (row.category, row.sample_id, row.modality)
                        if key in scores:
                            scores[key] = float(1.0 - scores[key])
            orientations[(category, modality)] = {
                "direction": direction,
                "used_split": "validation",
                "used_test_labels": False,
                "validation_auc_before_orientation": auc,
                "n_validation": int(len(labels)),
            }
    return orientations


def _confidence(score: float) -> float:
    return float(np.clip(0.5 + abs(float(score) - 0.5), 0.05, 1.0))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    rows, categories = _load_rows(args.root, args.categories, args.exclude_categories)
    features: dict[tuple[str, str, str], np.ndarray] = {}
    qualities: dict[tuple[str, str, str], np.ndarray] = {}
    for category in categories:
        zpath = args.root / "realiad_d3_raw" / f"{category}.zip"
        category_rows = [r for r in rows if r.category == category]
        with zipfile.ZipFile(zpath) as zf:
            for row in category_rows:
                key = (row.category, row.sample_id, row.modality)
                if key in features:
                    continue
                if row.modality == "xyz":
                    feat, quality = _xyz_features(_read_xyz(zf, row.zip_member))
                else:
                    feat, quality = _image_features(_read_rgb_like(zf, row.zip_member))
                features[key] = feat.astype(np.float32)
                qualities[key] = quality.astype(np.float32)
    scores, reliabilities, quality_distances = _score_and_reliability(
        rows=rows,
        features=features,
        qualities=qualities,
    )
    orientations = _apply_validation_score_orientation(rows, scores)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (row.category, row.sample_id, row.modality)
        if key not in scores:
            continue
        feat = features[key]
        score = scores[key]
        orientation = orientations.get((row.category, row.modality), {})
        out_rows.append(
            {
                "sample_id": row.sample_id,
                "category": row.category,
                "split": row.split,
                "defect_type": row.anomaly_class,
                "domain": row.modality,
                "label": int(row.label),
                "source_path": f"{args.root / 'realiad_d3_raw' / (row.category + '.zip')}::{row.zip_member}",
                "score": float(score),
                "confidence": _confidence(score),
                "quality_reliability": float(reliabilities[key]),
                "quality_distance": float(quality_distances[key]),
                "score_orientation": orientation.get("direction", "native"),
                "score_orientation_used_split": orientation.get("used_split", "validation"),
                "score_orientation_used_test_labels": bool(orientation.get("used_test_labels", False)),
                "validation_auc_before_orientation": orientation.get("validation_auc_before_orientation"),
                **{f"embedding_{i}": float(v) for i, v in enumerate(feat[:32])},
            }
        )
    df = pd.DataFrame(out_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    manifest = {
        "dataset_id": "realiad_d3_natural_degradation_v1",
        "protocol": "research_lock/D15_NATURAL_DEGRADATION_PROTOCOL_v1.yaml",
        "categories": categories,
        "modalities": sorted(MODALITY_PATH_FIELDS),
        "n_samples": {
            split: int(df[df["split"] == split]["sample_id"].nunique())
            for split in ("train", "validation", "test")
        },
        "test_label_counts": {
            str(k): int(v)
            for k, v in df[df["split"] == "test"].drop_duplicates("sample_id")["label"].value_counts().to_dict().items()
        },
        "score_orientations": {
            f"{category}/{modality}": info
            for (category, modality), info in sorted(orientations.items())
        },
        "input_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/raw/realiad_d3")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/fusion/realiad_d3_headroom_inputs.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "elara_master_c/data/splits/split_hashes/realiad_d3_headroom_v1.json")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--exclude-categories", nargs="*", default=None)
    args = parser.parse_args()
    manifest = prepare(args)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
