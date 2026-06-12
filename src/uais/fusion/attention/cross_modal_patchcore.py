"""Cross-modal patch-interaction PatchCore (feature-level fusion).

The Level-4 detector lever. ELARA's published number is *score-level* fusion: each
modality is scored independently and the scalars are combined, so the fusion layer
can never see that an RGB patch which looks normal sits exactly where the depth
patch is anomalous. The leaderboard methods (M3DM, AST) win because they fuse at
the *feature* level — the memory bank stores the JOINT cross-modal patch
appearance, so jointly-anomalous-but-marginally-normal defects become visible.

This module builds that joint detector by reusing the existing, tested PatchCore
primitives:

  per-modality patch features (ResNet-50 layer2+layer3, spatially aligned grid)
    -> concatenate across modalities at each patch location  (the interaction)
    -> single memory bank over joint patch descriptors (train-OK, coreset)
    -> max joint-patch distance = image anomaly score.

Requires spatially aligned modalities (e.g. MVTec 3D-AD rgb + xyz share the same
organized H×W grid). The cross-modal combination is factored out
(`combine_cross_modal_patches`) so it is unit-testable without a backbone.

Why this is the *right* lever and not a gate trick: a richer joint representation
raises A* (the Neyman–Pearson ceiling on the joint scores). T9 caps fusion of a
*fixed* score vector; changing what the scores ARE is the one move T9 leaves open
for clean transfer, and it is the standard route to leaderboard-competitive 3D AD.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from uais.fusion.attention.patchcore_patch import greedy_coreset, image_anomaly_scores
from uais.fusion.attention.realiad_3d_detector import extract_patches_from_images

__all__ = [
    "combine_cross_modal_patches",
    "score_cross_modal_from_features",
    "score_one_class_cross_modal_patchcore",
]


def combine_cross_modal_patches(
    modality_patches: Sequence[np.ndarray], n_patches: int,
) -> np.ndarray:
    """Concatenate per-modality patch features at each aligned patch location.

    Each input is a flat ``[n_images * n_patches, C_m]`` array (the
    ``extract_patches_from_images`` layout). All modalities must share the same
    image count and ``n_patches`` (spatial alignment). Returns the joint
    ``[n_images * n_patches, sum_m C_m]`` array, concatenating features at the
    same (image, patch) index across modalities.
    """
    if not modality_patches:
        raise ValueError("need at least one modality")
    rows = modality_patches[0].shape[0]
    if rows % n_patches != 0:
        raise ValueError("rows not divisible by n_patches (misaligned grids)")
    n_img = rows // n_patches
    blocks = []
    for m in modality_patches:
        if m.shape[0] != rows:
            raise ValueError("modalities have different (n_images*n_patches) lengths")
        blocks.append(m.reshape(n_img, n_patches, -1))
    joint = np.concatenate(blocks, axis=-1)          # [n_img, n_patches, sum C]
    return joint.reshape(n_img * n_patches, -1).astype(np.float32)


def score_cross_modal_from_features(
    train_modality_patches: Sequence[np.ndarray],
    eval_modality_patches: Sequence[np.ndarray],
    n_patches: int,
    *,
    coreset_size: int = 4096,
    seed: int = 0,
) -> np.ndarray:
    """Cross-modal PatchCore scores from pre-extracted per-modality patch features.

    Builds one joint memory bank from the all-normal training patches and returns
    per-image max joint-patch distances for the eval set.
    """
    train_joint = combine_cross_modal_patches(train_modality_patches, n_patches)
    bank = (greedy_coreset(train_joint, coreset_size, seed=seed)
            if train_joint.shape[0] > coreset_size else train_joint)
    eval_joint = combine_cross_modal_patches(eval_modality_patches, n_patches)
    return image_anomaly_scores(eval_joint, n_patches, bank)


def score_one_class_cross_modal_patchcore(
    train_images_by_modality: Sequence[Sequence[Image.Image]],
    eval_images_by_modality: Sequence[Sequence[Image.Image]],
    *,
    coreset_size: int = 4096,
    batch_size: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """One-class cross-modal PatchCore over spatially aligned modality images.

    ``*_images_by_modality`` are parallel lists, one image sequence per modality
    (e.g. [rgb_images, depth_images]); image i must correspond across modalities.
    Returns raw max joint-patch distances (higher = more anomalous); the caller
    applies z-sigmoid normalization, mirroring score_one_class_patchcore.
    """
    train_patches, n_patches = [], None
    for imgs in train_images_by_modality:
        p, np_ = extract_patches_from_images(imgs, batch_size=batch_size)
        train_patches.append(p)
        n_patches = np_
    eval_patches = []
    for imgs in eval_images_by_modality:
        p, _ = extract_patches_from_images(imgs, batch_size=batch_size)
        eval_patches.append(p)
    return score_cross_modal_from_features(
        train_patches, eval_patches, n_patches, coreset_size=coreset_size, seed=seed)
