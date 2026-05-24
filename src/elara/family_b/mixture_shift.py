"""Phase 2.2B — pure mixture-shift sampler for B-MECH-3.

The goal: construct test-fold-like samples where **category / cohort
proportions** differ between two evaluation pulls but **within-category
score distributions are held constant**. This isolates KS-gate
false-firing that is driven purely by batch-composition changes, not
by actual detector degradation.

The sampler does NOT introduce any score perturbation; the rows
returned are unmodified rows drawn from the source split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MixtureShiftResample:
    """One mixture-shift pull.

    `indices` is a row-index vector into the source split. The driver
    indexes the source features / masks / labels by `indices` to build
    the resampled batch. Within-category score distributions are
    preserved by construction (we draw rows by category without
    score-conditioned filtering).
    """

    name: str
    target_proportions: dict[str, float]
    actual_proportions: dict[str, float]
    indices: np.ndarray
    rng_seed: int


def pure_mixture_shift_resample(
    *,
    categories: np.ndarray,
    target_proportions: dict[str, float],
    n_samples: int,
    rng_seed: int = 0,
    require_within_category_invariance: bool = True,
    scores_for_invariance_check: np.ndarray | None = None,
    invariance_tol_ks_p: float = 0.05,
) -> MixtureShiftResample:
    """Resample row indices to match `target_proportions` over `categories`.

    Args:
      categories: [N] vector of category labels for the source split.
      target_proportions: mapping category -> target fraction in [0, 1].
        Fractions are renormalised to sum to 1.0; any category present
        in ``categories`` but not in this mapping receives 0 weight.
      n_samples: total number of rows to draw (with replacement
        per-category). The result vector has length == ``n_samples``.
      rng_seed: PRNG seed for reproducibility.
      require_within_category_invariance: if True and
        ``scores_for_invariance_check`` is provided, verify per-category
        score distributions in the resample match those in the source
        at KS p >= ``invariance_tol_ks_p``. Raises ValueError on
        failure. (Default True is the B-MECH-3 contract: this sampler
        is "pure mixture-shift" only when this invariance holds.)
      scores_for_invariance_check: optional [N] array used only by the
        invariance check.

    Returns:
      ``MixtureShiftResample``.

    By construction this draws by category without score filtering, so
    the within-category score distribution invariance holds in
    expectation. The invariance check empirically validates that the
    finite-sample draw did not break it.
    """
    cats = np.asarray(categories).astype(str)
    rng = np.random.default_rng(int(rng_seed))

    # Renormalise target proportions
    raw_total = sum(max(float(v), 0.0) for v in target_proportions.values())
    if raw_total <= 0:
        raise ValueError("target_proportions must contain at least one positive entry")
    normed = {k: max(float(v), 0.0) / raw_total for k, v in target_proportions.items()}

    # Per-category integer quotas with the largest-remainder method
    # so the per-category counts sum to exactly n_samples.
    raw_counts = {k: normed[k] * n_samples for k in normed}
    base = {k: int(np.floor(raw_counts[k])) for k in raw_counts}
    deficit = int(n_samples) - sum(base.values())
    remainders = sorted(
        ((raw_counts[k] - base[k], k) for k in raw_counts),
        reverse=True,
    )
    for i in range(deficit):
        _, k = remainders[i % len(remainders)]
        base[k] += 1

    # Draw with replacement within each category from the source rows.
    indices_parts = []
    actual_proportions = {}
    for cat, count in base.items():
        if count <= 0:
            actual_proportions[cat] = 0.0
            continue
        cat_idx = np.where(cats == cat)[0]
        if cat_idx.size == 0:
            raise ValueError(
                f"target_proportions references category {cat!r} but no rows of that "
                "category exist in the source split"
            )
        chosen = rng.choice(cat_idx, size=count, replace=True)
        indices_parts.append(chosen)
        actual_proportions[cat] = float(count) / float(n_samples)

    indices = np.concatenate(indices_parts) if indices_parts else np.array([], dtype=int)
    rng.shuffle(indices)

    if require_within_category_invariance and scores_for_invariance_check is not None:
        from scipy.stats import ks_2samp
        source_scores = np.asarray(scores_for_invariance_check, dtype=float)
        for cat in normed:
            if cat not in actual_proportions or actual_proportions[cat] == 0.0:
                continue
            src_mask = (cats == cat)
            in_cat_idx = indices[cats[indices] == cat]
            if src_mask.sum() < 30 or len(in_cat_idx) < 30:
                # Insufficient samples for a meaningful invariance check; skip.
                continue
            _, ks_p = ks_2samp(source_scores[src_mask], source_scores[in_cat_idx])
            if ks_p < invariance_tol_ks_p:
                raise ValueError(
                    f"within-category invariance violated for {cat!r}: KS p={ks_p:.4g} "
                    f"< tol {invariance_tol_ks_p}; the resample distorted score distribution"
                )

    return MixtureShiftResample(
        name=f"mixture_shift_seed{rng_seed}",
        target_proportions=normed,
        actual_proportions=actual_proportions,
        indices=indices,
        rng_seed=int(rng_seed),
    )


__all__ = ["MixtureShiftResample", "pure_mixture_shift_resample"]
