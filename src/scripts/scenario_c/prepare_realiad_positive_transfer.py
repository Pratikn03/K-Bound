#!/usr/bin/env python3
"""Prepare Real-IAD 256 C1/C2 natural-transfer inputs for D13.

The script intentionally fixes the protocol before scoring:
- all downloaded Real-IAD 256 categories are used;
- modalities are camera C1 and camera C2;
- official train normals form the normal reference;
- official test samples are split into validation/test by stable sample hash.

It writes the D13 input CSV plus SAR/CW comparator score files expected by
`run_positive_transfer_confirmatory.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uais.fusion.attention.baselines import ConfidenceWeightedMean, SARScoreAdapter  # noqa: E402,I001


CAMERA_RE = re.compile(r"_C([1-5])_")


@dataclass(frozen=True)
class ImageRow:
    category: str
    sample_key: str
    camera: str
    label: int
    anomaly_class: str
    split_source: str
    json_image_path: str
    zip_member: str


def _camera(path: str) -> str:
    m = CAMERA_RE.search(path)
    if not m:
        raise ValueError(f"cannot parse camera from path: {path}")
    return f"C{m.group(1)}"


def _sample_key(category: str, path: str) -> str:
    prefix = path.split("_C", 1)[0]
    return f"{category}/{prefix}"


def _stable_unit(text: str, *, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{text}".encode()).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def _extract_jsons(root: Path) -> Path:
    json_dir = root / "jsons" / "realiad_jsons"
    if json_dir.is_dir() and any(json_dir.glob("*.json")):
        return json_dir
    zip_path = root / "realiad_jsons.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"missing {zip_path}")
    (root / "jsons").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.startswith("realiad_jsons/") and member.endswith(".json") and "/._" not in member:
                zf.extract(member, root / "jsons")
    return json_dir


def _load_rows(root: Path) -> tuple[list[ImageRow], list[str]]:
    json_dir = _extract_jsons(root)
    zip_dir = root / "realiad_256"
    zips = {p.stem: p for p in zip_dir.glob("*.zip") if not p.name.startswith("._")}
    if not zips:
        raise FileNotFoundError(f"no Real-IAD 256 category zips found under {zip_dir}")
    categories = sorted(set(zips) & {p.stem for p in json_dir.glob("*.json") if not p.name.startswith("._")})
    rows: list[ImageRow] = []
    for category in categories:
        doc = json.loads((json_dir / f"{category}.json").read_text(encoding="utf-8"))
        prefix = str(doc.get("meta", {}).get("prefix", f"{category}/"))
        for split_source in ("train", "test"):
            for entry in doc.get(split_source, []):
                image_path = str(entry["image_path"])
                camera = _camera(image_path)
                if camera not in {"C1", "C2"}:
                    continue
                anomaly_class = str(entry.get("anomaly_class", ""))
                label = 0 if anomaly_class == "OK" else 1
                rows.append(
                    ImageRow(
                        category=category,
                        sample_key=_sample_key(category, image_path),
                        camera=camera,
                        label=label,
                        anomaly_class=anomaly_class,
                        split_source=split_source,
                        json_image_path=image_path,
                        zip_member=f"{prefix}{image_path}",
                    )
                )
    return rows, categories


def _read_image_from_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as f:
        data = f.read()
    img = Image.open(BytesIO(data)).convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _image_features(img: np.ndarray) -> np.ndarray:
    rgb = img.astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    chans = [rgb[:, :, i] for i in range(3)]
    stats: list[float] = []
    for ch in chans + [gray]:
        stats.extend(
            [
                float(ch.mean()),
                float(ch.std()),
                float(np.percentile(ch, 10)),
                float(np.percentile(ch, 50)),
                float(np.percentile(ch, 90)),
            ]
        )
    hist = cv2.calcHist([(gray * 255).astype(np.uint8)], [0], None, [16], [0, 256]).reshape(-1)
    hist = hist / max(float(hist.sum()), 1.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    stats.extend(
        [
            float(mag.mean()),
            float(mag.std()),
            float(np.percentile(mag, 90)),
            float((mag > np.percentile(mag, 75)).mean()),
        ]
    )
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1)
    feat = np.concatenate([np.asarray(stats, dtype=np.float32), hist.astype(np.float32), small.astype(np.float32)])
    return np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def _score_features(features: dict[tuple[str, str, str], np.ndarray], rows: list[ImageRow]) -> dict[tuple[str, str, str], float]:
    scores: dict[tuple[str, str, str], float] = {}
    for category in sorted({r.category for r in rows}):
        for camera in ("C1", "C2"):
            train_keys = [
                (r.category, r.sample_key, r.camera)
                for r in rows
                if r.category == category and r.camera == camera and r.split_source == "train" and r.label == 0
            ]
            eval_keys = [(r.category, r.sample_key, r.camera) for r in rows if r.category == category and r.camera == camera]
            ref = np.stack([features[k] for k in train_keys], axis=0)
            ref_mean = ref.mean(axis=0, keepdims=True)
            ref_std = ref.std(axis=0, keepdims=True) + 1e-6
            ref_z = (ref - ref_mean) / ref_std
            eval_z = np.stack([(features[k] - ref_mean.reshape(-1)) / ref_std.reshape(-1) for k in eval_keys], axis=0)
            # Mean distance to the five nearest train-normal references. This is
            # deterministic, validation-free, and one-class with respect to the
            # official train normals.
            d = pairwise_distances(eval_z, ref_z, metric="euclidean", n_jobs=1)
            k = min(5, d.shape[1])
            raw = np.sort(d, axis=1)[:, :k].mean(axis=1)
            lo = float(np.percentile(raw[: len(train_keys)], 5))
            hi = float(np.percentile(raw[: len(train_keys)], 95))
            denom = hi - lo if hi > lo else float(np.std(raw) + 1e-6)
            norm = np.clip((raw - lo) / denom, 0.0, 1.0)
            for key, value in zip(eval_keys, norm, strict=True):
                scores[key] = float(value)
    return scores


def _confidence(score: float) -> float:
    return float(np.clip(0.5 + abs(float(score) - 0.5), 0.05, 1.0))


def _build_frames(
    *,
    rows: list[ImageRow],
    features: dict[tuple[str, str, str], np.ndarray],
    scores: dict[tuple[str, str, str], float],
    validation_fraction: float,
    split_salt: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sample_info: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = sample_info.setdefault(
            row.sample_key,
            {
                "category": row.category,
                "label": row.label,
                "anomaly_class": row.anomaly_class,
                "split_source": row.split_source,
                "cameras": set(),
            },
        )
        entry["cameras"].add(row.camera)
    complete = {
        k: v
        for k, v in sample_info.items()
        if {"C1", "C2"} <= v["cameras"] and (v["split_source"] == "train" or v["label"] in {0, 1})
    }
    realiad_rows: list[dict[str, Any]] = []
    feature_rows: list[tuple[str, np.ndarray, np.ndarray, int, str]] = []
    for sample_key, info in sorted(complete.items()):
        if info["split_source"] == "train":
            split = "train"
        else:
            split = "validation" if _stable_unit(sample_key, salt=split_salt) < validation_fraction else "test"
        per_domain_features = []
        for domain in ("C1", "C2"):
            key = (str(info["category"]), sample_key, domain)
            score = float(scores[key])
            feat = features[key]
            per_domain_features.append(np.concatenate([[score, _confidence(score)], feat[:30]], dtype=np.float32))
            realiad_rows.append(
                {
                    "sample_id": sample_key.replace("/", "__"),
                    "pairing_key": sample_key,
                    "category": info["category"],
                    "split": split,
                    "defect_type": info["anomaly_class"],
                    "domain": "rgb_c1" if domain == "C1" else "rgb_c2",
                    "label": int(info["label"]),
                    "source_path": f"data/raw/realiad/realiad_256/{info['category']}.zip::{info['category']}/...",
                    "score_fit_split": "train",
                    "score_fit_defect_type": "OK",
                    "score": score,
                    "confidence": _confidence(score),
                    **{f"embedding_{i}": float(v) for i, v in enumerate(feat[:32])},
                    "raw_distance_score": score,
                }
            )
        if split in {"validation", "test"}:
            feature_rows.append(
                (
                    sample_key.replace("/", "__"),
                    np.stack(per_domain_features, axis=0),
                    np.zeros(2, dtype=bool),
                    int(info["label"]),
                    split,
                )
            )

    df = pd.DataFrame(realiad_rows)
    val_rows = [r for r in feature_rows if r[4] == "validation"]
    test_rows = [r for r in feature_rows if r[4] == "test"]
    train_feat = np.stack([r[1] for r in val_rows], axis=0)
    train_mask = np.stack([r[2] for r in val_rows], axis=0)
    train_labels = np.asarray([r[3] for r in val_rows], dtype=int)
    test_feat = np.stack([r[1] for r in test_rows], axis=0)
    test_mask = np.stack([r[2] for r in test_rows], axis=0)
    test_ids = [r[0] for r in test_rows]

    sar = SARScoreAdapter(random_seed=42, adaptation_steps=30).fit(train_feat, train_mask, train_labels)
    cw = ConfidenceWeightedMean().fit(train_feat, train_mask, train_labels)
    sar_scores = pd.DataFrame({"sample_id": test_ids, "raw_score": sar.predict_proba(test_feat, test_mask)})
    cw_scores = pd.DataFrame({"sample_id": test_ids, "raw_score": cw.predict_proba(test_feat, test_mask)})
    manifest = {
        "dataset_id": "realiad_256_c1_c2_d13",
        "categories": sorted({str(v["category"]) for v in complete.values()}),
        "modalities": ["rgb_c1", "rgb_c2"],
        "validation_fraction_of_official_test": validation_fraction,
        "split_salt": split_salt,
        "n_samples": {
            "train": int(df[df["split"] == "train"]["sample_id"].nunique()),
            "validation": int(df[df["split"] == "validation"]["sample_id"].nunique()),
            "test": int(df[df["split"] == "test"]["sample_id"].nunique()),
        },
        "test_label_counts": {
            str(k): int(v)
            for k, v in df[df["split"] == "test"].drop_duplicates("sample_id")["label"].value_counts().to_dict().items()
        },
    }
    return df, sar_scores, cw_scores, manifest


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root
    rows, categories = _load_rows(root)
    features: dict[tuple[str, str, str], np.ndarray] = {}
    for category in categories:
        zpath = root / "realiad_256" / f"{category}.zip"
        category_rows = [r for r in rows if r.category == category]
        with zipfile.ZipFile(zpath) as zf:
            for row in category_rows:
                key = (row.category, row.sample_key, row.camera)
                if key not in features:
                    features[key] = _image_features(_read_image_from_zip(zf, row.zip_member))
    scores = _score_features(features, rows)
    df, sar, cw, manifest = _build_frames(
        rows=rows,
        features=features,
        scores=scores,
        validation_fraction=args.validation_fraction,
        split_salt=args.split_salt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    args.sar_scores.parent.mkdir(parents=True, exist_ok=True)
    sar.to_csv(args.sar_scores, index=False)
    cw.to_csv(args.cw_scores, index=False)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest["input_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest["sar_scores_sha256"] = hashlib.sha256(args.sar_scores.read_bytes()).hexdigest()
    manifest["cw_scores_sha256"] = hashlib.sha256(args.cw_scores.read_bytes()).hexdigest()
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/raw/realiad")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/fusion/realiad_256_c1_c2_d13_inputs.csv")
    parser.add_argument("--sar-scores", type=Path, default=ROOT / "experiments/fusion/realiad_256_c1_c2_d13_sar_scores.csv")
    parser.add_argument("--cw-scores", type=Path, default=ROOT / "experiments/fusion/realiad_256_c1_c2_d13_cw_scores.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "elara_master_c/data/splits/split_hashes/realiad_256_c1_c2_d13.json")
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    parser.add_argument("--split-salt", default="D13_REALIAD_256_C1_C2_v1_2026-06-01")
    args = parser.parse_args()
    manifest = prepare(args)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
