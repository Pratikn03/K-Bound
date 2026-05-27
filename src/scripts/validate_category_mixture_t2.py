"""Synthetic validation of Theorem T2 (Global-KS mixture confounding).

T2 (thesis appendix): if validation and test share the same per-category
score distributions but differ only in mixture weights pi_c, the global
KS statistic is non-zero while category-conditional KS is zero.

This script constructs a controlled synthetic scenario:
  - C=3 categories, each with its own score distribution P_c
  - Validation mixture (pi)   = (0.6, 0.3, 0.1)
  - Test mixture (pi') ~= flipped (0.1, 0.3, 0.6)
  - Per-category CDFs are identical between val and test

It then computes:
  - Global KS(val vs test) with the default ReliabilityEstimator
  - Per-category KS via the CategoryAwareReliabilityEstimator

T2 predicts: global KS > 0 (false fire), category-conditional KS ~= 0.

Output: JSON with the measured KS distances and a paper-ready
verification verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def _build_mixture(
    rng: np.random.Generator, n: int, pis: list[float], shifts: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Sample n scores from sum_c pi_c * Beta(2 + shift_c, 2 - shift_c)."""
    cats = rng.choice(len(pis), size=n, p=pis)
    scores = np.zeros(n, dtype=np.float32)
    for c, shift in enumerate(shifts):
        mask = cats == c
        alpha = 2.0 + shift
        beta = max(0.1, 2.0 - shift)
        scores[mask] = rng.beta(alpha, beta, size=int(mask.sum()))
    return scores, cats


def run_validation(
    *,
    n_val: int = 3000,
    n_test: int = 3000,
    pi_val: list[float] = (0.6, 0.3, 0.1),
    pi_test: list[float] = (0.1, 0.3, 0.6),
    shifts: list[float] = (-0.7, 0.0, +0.7),
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    val_scores, val_cats = _build_mixture(rng, n_val, list(pi_val), list(shifts))
    test_scores, test_cats = _build_mixture(rng, n_test, list(pi_test), list(shifts))

    # Global KS (this is what a category-blind drift gate would see)
    global_ks = float(stats.ks_2samp(val_scores, test_scores).statistic)

    # Category-conditional KS (what a category-aware gate would see)
    per_cat_ks = {}
    for c in range(len(shifts)):
        v = val_scores[val_cats == c]
        t = test_scores[test_cats == c]
        if len(v) < 5 or len(t) < 5:
            per_cat_ks[str(c)] = None
            continue
        per_cat_ks[str(c)] = float(stats.ks_2samp(v, t).statistic)
    finite_per_cat = [v for v in per_cat_ks.values() if v is not None]
    max_per_cat_ks = max(finite_per_cat) if finite_per_cat else None

    # Verdict: T2 confirmed iff global_ks >> max_per_cat_ks.
    # T2 confirmed iff global KS substantially exceeds the worst
    # per-category KS. We use a 3x ratio with absolute floors to absorb
    # finite-sample noise; the populaton statement of T2 says the ratio
    # diverges as n grows.
    confirmed = (
        global_ks is not None and max_per_cat_ks is not None and global_ks > 0.1 and global_ks > 3.0 * max_per_cat_ks
    )

    return {
        "n_val": int(n_val),
        "n_test": int(n_test),
        "pi_val": list(pi_val),
        "pi_test": list(pi_test),
        "shifts": list(shifts),
        "seed": int(seed),
        "global_ks_distance": float(global_ks),
        "per_category_ks_distance": per_cat_ks,
        "max_per_category_ks": float(max_per_cat_ks) if max_per_cat_ks is not None else None,
        "global_minus_per_category": float(global_ks) - (float(max_per_cat_ks) if max_per_cat_ks is not None else 0.0),
        "t2_confirmed": bool(confirmed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/fusion/category_mixture_t2_validation.json"),
    )
    args = parser.parse_args()

    # Run across a small seed sweep for stability.
    rows = []
    for s in range(5):
        rows.append(run_validation(seed=args.seed + s))
    summary = {
        "rows": rows,
        "mean_global_ks": float(np.mean([r["global_ks_distance"] for r in rows])),
        "mean_max_per_category_ks": float(
            np.mean([r["max_per_category_ks"] for r in rows if r["max_per_category_ks"] is not None])
        ),
        "all_confirmed": all(r["t2_confirmed"] for r in rows),
        "n_seeds": len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
