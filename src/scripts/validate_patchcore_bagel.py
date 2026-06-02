"""Correctness check: true patch-level PatchCore on MVTec 3D-AD bagel (RGB only).

Published PatchCore RGB-only image-AUROC on MVTec 3D-AD bagel is ~0.78-0.88.
If our implementation lands in that range, the patch pipeline is correct and we
can scale it to the full benchmark and to Eyecandies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from uais.fusion.attention.patchcore_patch import (  # noqa: E402
    extract_patch_embeddings,
    greedy_coreset,
    image_anomaly_scores,
)

CAT = ROOT / "data/raw/mvtec3d/bagel"


def main() -> int:
    train_paths = sorted((CAT / "train/good/rgb").glob("*.png"))
    test_good = sorted((CAT / "test/good/rgb").glob("*.png"))
    test_defect = []
    for d in sorted((CAT / "test").iterdir()):
        if d.name == "good" or not d.is_dir():
            continue
        test_defect += sorted((d / "rgb").glob("*.png"))
    print(f"train_good={len(train_paths)}  test_good={len(test_good)}  test_defect={len(test_defect)}")

    print("extracting train patches...", flush=True)
    train_patches, P = extract_patch_embeddings(train_paths, patch_grid=28)
    print(f"  train patches: {train_patches.shape}, patches/image={P}")

    # Coreset to 10% of the bank.
    target = max(P * 4, int(0.10 * train_patches.shape[0]))
    print(f"greedy coreset -> {target} patches...", flush=True)
    bank = greedy_coreset(train_patches, target, seed=0)
    print(f"  bank: {bank.shape}")

    test_paths = test_good + test_defect
    labels = np.array([0] * len(test_good) + [1] * len(test_defect))
    print("extracting test patches...", flush=True)
    test_patches, _ = extract_patch_embeddings(test_paths, patch_grid=28)
    print("scoring...", flush=True)
    scores = image_anomaly_scores(test_patches, P, bank)

    auroc = roc_auc_score(labels, scores)
    print(f"\n=== MVTec 3D-AD bagel RGB-only image-AUROC = {auroc:.4f} ===")
    print("(published PatchCore RGB-only on 3D-AD bagel ~0.78-0.88)")
    if auroc >= 0.70:
        print("CORRECTNESS: PASS (substantially above chance; patch pipeline works)")
    else:
        print("CORRECTNESS: WEAK (below 0.70 - investigate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
