"""kbound_edge.dataset -- preprocessing, windowing, and synthetic condition builders.

Two responsibilities:

1. Turn raw BGR frames into normalised model tensors and group a stream into
   fixed windows (:func:`preprocess_frame`, :func:`frames_to_tensor`,
   :func:`window_ranges`).

2. Build the SYNTHETIC data used to validate the whole chain offline:
   * :func:`make_training_clip` -- a balanced, clean, *labelled* clip to train f0.
   * :data:`REGIMES` + :func:`make_condition` -- per-window "conditions" under a
     named domain shift, each carrying OFFLINE labels so the true adaptation
     benefit B can be measured during calibration / held-out evaluation.

Labels live on :class:`Condition` objects for OFFLINE use only.  The online
runners (replay/shadow) consume frames/tensors, never ``Condition.labels``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from kbound_edge.capture import SyntheticFrameSource, Frame

# ImageNet normalisation (RGB).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_frame(frame_bgr: Frame, size: int = 64) -> np.ndarray:
    """BGR uint8 frame -> normalised CHW float32 array (ImageNet stats)."""
    import cv2

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != size or rgb.shape[1] != size:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))  # CHW


def frames_to_tensor(frames: Sequence[Frame], size: int = 64):
    """Stack BGR frames into a torch tensor (N, 3, size, size)."""
    import torch

    arr = np.stack([preprocess_frame(f, size) for f in frames], axis=0)
    return torch.from_numpy(arr)


def window_ranges(n_frames: int, window: int, stride: Optional[int] = None) -> List[Tuple[int, int]]:
    """Inclusive-exclusive (start, end) ranges of length ``window`` over a stream."""
    if window <= 0:
        raise ValueError("window must be > 0")
    stride = window if stride is None else stride
    out = []
    start = 0
    while start + window <= n_frames:
        out.append((start, start + window))
        start += stride
    return out


# ---------------------------------------------------------------------------
# Synthetic conditions
# ---------------------------------------------------------------------------

#: Named domain-shift regimes.  Values are SyntheticFrameSource kwargs.
#: ("clean" matches the training distribution; the others are shifts of varying
#: severity.  Which ones end up beneficial / neutral / harmful is *measured*, not
#: assumed -- see scripts/04_generate_calibration_pairs.py.)
REGIMES: Dict[str, dict] = {
    "clean":         dict(brightness=0.00, contrast=1.00, noise=0.05, blur=0),
    "bright_up":     dict(brightness=0.30, contrast=1.00, noise=0.05, blur=0),
    "bright_down":   dict(brightness=-0.25, contrast=1.00, noise=0.05, blur=0),
    "contrast_low":  dict(brightness=0.00, contrast=0.50, noise=0.05, blur=0),
    "noisy":         dict(brightness=0.05, contrast=0.95, noise=0.22, blur=0),
    "blurred":       dict(brightness=0.00, contrast=0.90, noise=0.05, blur=5),
    "severe":        dict(brightness=0.45, contrast=0.55, noise=0.25, blur=3),
    "washed":        dict(brightness=0.05, contrast=0.12, noise=0.30, blur=5),
}

#: Window class-diversity modes.  "single"/"low" diversity is the canonical TENT
#: failure mode: with little intra-batch variation the BatchNorm batch
#: statistics degenerate and entropy minimisation collapses to one class, so
#: adaptation HURTS a competent frozen model (true benefit B < 0 -> FREEZE).
DIVERSITY_MODES = ("multi", "low", "single")


@dataclass
class Condition:
    """One window's worth of frames under a regime (with OFFLINE labels)."""

    cond_id: str
    regime: str
    frames: List[Frame]
    labels: np.ndarray          # OFFLINE ONLY (used to measure benefit B)
    image_size: int
    diversity: str = "multi"

    def tensor(self):
        """Preprocessed model tensor for this window (no labels)."""
        return frames_to_tensor(self.frames, self.image_size)


def _diversity_class_seq(diversity: str, n_frames: int, n_classes: int, seed: int):
    """Per-frame class sequence implementing a diversity mode (None = random)."""
    rng = np.random.default_rng(seed)
    if diversity == "multi":
        return None  # SyntheticFrameSource draws classes uniformly at random
    if diversity == "single":
        c = int(rng.integers(0, n_classes))
        return [c] * n_frames
    if diversity == "low":
        cs = rng.choice(n_classes, size=2, replace=False)
        return [int(cs[i % 2]) for i in range(n_frames)]
    raise KeyError(f"unknown diversity '{diversity}'; choices: {DIVERSITY_MODES}")


def make_condition(
    cond_id: str,
    regime: str,
    n_frames: int,
    image_size: int = 64,
    seed: int = 0,
    n_classes: int = 4,
    diversity: str = "multi",
) -> Condition:
    """Build one synthetic :class:`Condition` under a named regime + diversity mode."""
    if regime not in REGIMES:
        raise KeyError(f"unknown regime '{regime}'; choices: {sorted(REGIMES)}")
    class_seq = _diversity_class_seq(diversity, n_frames, n_classes, seed)
    src = SyntheticFrameSource(
        num_frames=n_frames,
        n_classes=n_classes,
        image_size=image_size,
        seed=seed,
        class_seq=class_seq,
        **REGIMES[regime],
    )
    return Condition(
        cond_id=cond_id,
        regime=regime,
        frames=src.frames,
        labels=src.labels,
        image_size=image_size,
        diversity=diversity,
    )


def make_training_clip(
    n_per_class: int = 64,
    image_size: int = 64,
    seed: int = 0,
    n_classes: int = 4,
) -> Tuple[List[Frame], np.ndarray]:
    """Balanced, clean, LABELLED clip for training the source model f0 (offline)."""
    class_seq = np.repeat(np.arange(n_classes), n_per_class)
    rng = np.random.default_rng(seed)
    rng.shuffle(class_seq)
    src = SyntheticFrameSource(
        num_frames=len(class_seq),
        n_classes=n_classes,
        image_size=image_size,
        seed=seed,
        class_seq=class_seq,
        **REGIMES["clean"],
    )
    return src.frames, src.labels


def build_conditions(
    regime_plan: Sequence[tuple],
    n_frames: int = 16,
    image_size: int = 64,
    seed: int = 0,
    n_classes: int = 4,
    prefix: str = "cond",
) -> List[Condition]:
    """Build a list of conditions from a plan.

    Each plan item is either ``(regime, count)`` (diversity defaults to "multi")
    or ``(regime, diversity, count)``.  Each generated condition gets a distinct
    seed so the windows differ.
    """
    conditions: List[Condition] = []
    k = 0
    for item in regime_plan:
        if len(item) == 3:
            regime, diversity, count = item
        elif len(item) == 2:
            regime, count = item
            diversity = "multi"
        else:
            raise ValueError(f"plan item must be (regime,count) or (regime,diversity,count): {item}")
        for _ in range(int(count)):
            conditions.append(
                make_condition(
                    cond_id=f"{prefix}_{k:04d}_{regime}-{diversity}",
                    regime=regime,
                    n_frames=n_frames,
                    image_size=image_size,
                    seed=seed + 1000 * k + 7,
                    n_classes=n_classes,
                    diversity=diversity,
                )
            )
            k += 1
    return conditions
