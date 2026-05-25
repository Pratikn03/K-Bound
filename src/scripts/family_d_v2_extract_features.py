"""Phase 2.2E — Eyecandies per-modality one-class scoring pipeline.

Implementation choices (documented in execution report; standard PatchCore/PADIM defaults):
- Backbone: ResNet-50 ImageNet-1K v2 weights.
- Feature layer: `layer3` output, adaptively pooled to 1x1 (1024-D feature).
- One-class scoring: cosine distance to nearest train memory-bank entry (PatchCore-style, no coreset for simplicity).
- RGB view aggregation: SINGLE canonical view (image_0). Documented as a chosen default.
- Depth: replicated to 3 channels (gray-grayscale-grayscale) and passed through the same ResNet-50.
- Score normalisation: z-score per (category, modality) against train scores; then sigmoid → [0,1].

NEVER inspects anomaly mask file contents. NEVER reads metadata.yaml anomalous flags
(that happens only at the authorised final metric step, in a separate script).
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT / "data" / "raw" / "eyecandies" / "_archives"
OUT_DIR = ROOT / "experiments" / "phase2" / "family_d"
FEAT_DIR = OUT_DIR / "features"

CATEGORIES = [
    "CandyCane", "ChocolateCookie", "ChocolatePraline", "Confetto", "GummyBear",
    "HazelnutTruffle", "LicoriceSandwich", "Lollipop", "Marshmallow", "PeppermintCandy",
]
SPLITS = ["train", "val", "test_public", "test_private"]


def _device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _make_backbone(device):
    weights = ResNet50_Weights.IMAGENET1K_V2
    m = resnet50(weights=weights)
    m.eval()
    # We want layer3 outputs, then global pool.
    feature_layer = torch.nn.Sequential(
        m.conv1, m.bn1, m.relu, m.maxpool,
        m.layer1, m.layer2, m.layer3,
    ).to(device)
    transforms = weights.transforms()
    return feature_layer, transforms


@torch.no_grad()
def _featurize_image(rgb_bytes: bytes, backbone, transforms, device) -> np.ndarray:
    img = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
    tensor = transforms(img).unsqueeze(0).to(device)
    feat_map = backbone(tensor)
    pooled = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
    return pooled.cpu().numpy().squeeze()


@torch.no_grad()
def _featurize_batch(image_byte_list, backbone, transforms, device, batch_size: int = 32) -> np.ndarray:
    """Process N images in one batched forward pass."""
    feats = []
    for start in range(0, len(image_byte_list), batch_size):
        chunk = image_byte_list[start:start + batch_size]
        tensors = []
        for b in chunk:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            tensors.append(transforms(img))
        batch = torch.stack(tensors).to(device)
        feat_map = backbone(batch)
        pooled = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        feats.append(pooled.cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 1024))


def _process_category(cat: str, backbone, transforms, device, batch_size: int) -> None:
    archive = ARCHIVE_DIR / f"{cat}.tar"
    print(f"[{cat}] streaming tar...", flush=True)
    # Build a map split -> sample_id -> {rgb_bytes, depth_bytes}
    samples = {s: {} for s in SPLITS}
    with tarfile.open(archive, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.split("/")
            # Identify split + sample + modality
            split = None
            for p in parts:
                if p in SPLITS:
                    split = p
                    break
            if split is None:
                continue
            base = parts[-1]
            # Parse <NNN>_<kind>.<ext>
            try:
                sample_id, rest = base.split("_", 1)
            except ValueError:
                continue
            if not sample_id.isdigit():
                continue
            if rest == "image_0.png":
                samples[split].setdefault(sample_id, {})["rgb"] = tf.extractfile(m).read()
            elif rest == "depth.png":
                samples[split].setdefault(sample_id, {})["depth"] = tf.extractfile(m).read()
            # Ignore everything else (other views, normals, masks, metadata) for now.

    # Featurize each split's RGB + depth
    feats_per_split = {}
    for split in SPLITS:
        rgb_pairs = sorted(samples[split].items(), key=lambda kv: int(kv[0]))
        if not rgb_pairs:
            continue
        rgb_bytes = [s.get("rgb") for _, s in rgb_pairs]
        depth_bytes = [s.get("depth") for _, s in rgb_pairs]
        if any(b is None for b in rgb_bytes) or any(b is None for b in depth_bytes):
            missing_ids = [sid for sid, s in rgb_pairs
                           if s.get("rgb") is None or s.get("depth") is None]
            print(f"[{cat}/{split}] WARNING: {len(missing_ids)} samples missing rgb or depth; skipping those")
            keep = [i for i, (sid, s) in enumerate(rgb_pairs)
                    if s.get("rgb") is not None and s.get("depth") is not None]
            rgb_pairs = [rgb_pairs[i] for i in keep]
            rgb_bytes = [rgb_bytes[i] for i in keep]
            depth_bytes = [depth_bytes[i] for i in keep]
        sample_ids = [sid for sid, _ in rgb_pairs]

        print(f"[{cat}/{split}] forwarding {len(rgb_bytes)} RGB images...", flush=True)
        rgb_feats = _featurize_batch(rgb_bytes, backbone, transforms, device, batch_size)
        print(f"[{cat}/{split}] forwarding {len(depth_bytes)} depth images...", flush=True)
        depth_feats = _featurize_batch(depth_bytes, backbone, transforms, device, batch_size)
        feats_per_split[split] = {
            "sample_ids": sample_ids,
            "rgb": rgb_feats,
            "depth": depth_feats,
        }

    out = FEAT_DIR / f"{cat}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **{f"{split}_sample_ids": np.array(v["sample_ids"]) for split, v in feats_per_split.items()},
        **{f"{split}_rgb": v["rgb"] for split, v in feats_per_split.items()},
        **{f"{split}_depth": v["depth"] for split, v in feats_per_split.items()},
    )
    print(f"[{cat}] wrote {out}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", default=None,
                   help="comma-separated category subset")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    device = _device()
    print(f"device: {device}")
    backbone, transforms = _make_backbone(device)

    cats = CATEGORIES if args.only is None else [c.strip() for c in args.only.split(",")]
    for cat in cats:
        feat_path = FEAT_DIR / f"{cat}.npz"
        if feat_path.exists():
            print(f"[{cat}] features already exist; skipping")
            continue
        _process_category(cat, backbone, transforms, device, args.batch_size)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
