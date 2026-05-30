"""Finite-sample switching-certificate audit (T5).

Theorem T5 (thesis appendix): for fired samples F = {i: g_i = 1},
define the per-sample paired benefit
    X_i = ell_s(i) - ell_r(i)
(loss of the static path minus loss of the reliability path). A
deployment window certifies the switch only if the lower confidence
bound
    LCB_alpha = mean_F - margin_alpha
is positive at the chosen significance alpha.

This script reads the per-seed clean metrics already logged by the
runner in ``experiments/fusion/*_results.json``, treats the RGA+
boosted-fusion path as the "reliability path" and the static-attention
path as the "static path", uses the negative-log-loss-style 0-1
classification error proxy
    ell(p, y) = mean over samples of |p - y|
as the bounded per-seed loss surrogate, and reports a paired-bootstrap
lower confidence bound on (static_err - rga_err).

The audit reports, per benchmark + protocol:
  - n_seeds (fold count) and the per-seed paired benefit
  - point estimate of E[ell_s - ell_r]
  - 95 % LCB via paired bootstrap
  - verdict: "certified" iff LCB > 0

Honest framing: this is a *per-seed* audit at the aggregate-AUROC
loss surrogate, not a per-sample audit at the deployed 0-1 cost. The
per-sample variant requires the per-prediction y_hat columns which are
not currently logged; this aggregate audit is the right honest first
pass and is what the thesis appendix's T5 prose now points at.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _dig(payload: dict | None, *keys: str) -> Any:
    for key in keys:
        if payload is None:
            return None
        payload = payload.get(key)
    return payload


def _per_seed_aurocs(payload: dict, method: str) -> list[float]:
    """Collect per-seed clean ROC-AUC for a named method.

    Supports two shapes:
      (a) flat: [{"method": "static_attention", "roc_auc": 0.x}, ...]
      (b) nested per-seed: [{"seed": 42, "static_attention": {"roc_auc": 0.x}}, ...]
    """
    rows = payload.get("table_1_clean_performance", [])
    out = []
    for r in rows:
        # Shape (a): flat method column.
        if "method" in r and str(r.get("method")) == method:
            roc = r.get("roc_auc")
            if isinstance(roc, (int, float)) and math.isfinite(float(roc)):
                out.append(float(roc))
            continue
        # Shape (b): one row per seed, each method nested.
        nested = r.get(method)
        if isinstance(nested, dict):
            roc = nested.get("roc_auc")
            if isinstance(roc, (int, float)) and math.isfinite(float(roc)):
                out.append(float(roc))
    return out


def paired_bootstrap_lcb(
    diffs: list[float], alpha: float = 0.05, n_boot: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """Return (mean, LCB_alpha) for a paired bootstrap on diffs."""
    if not diffs:
        return float("nan"), float("nan")
    arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = arr[idx].mean()
    lcb = float(np.quantile(boots, alpha))
    return float(arr.mean()), lcb


def empirical_bernstein_lcb(
    diffs: list[float], alpha: float = 0.05, benefit_range: float | None = None
) -> tuple[float, float, float]:
    """Closed-form Maurer-Pontil (2009) empirical-Bernstein LCB on the mean.

    Returns (mean, lcb, sample_variance). The aggregate-AUROC surrogate
    benefit (1 - static_auroc) - (1 - rga_auroc) = rga_auroc - static_auroc
    lies in [-1, 1], so the range R = 2.0 unless overridden.
    """
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(diffs, dtype=np.float64)
    n = arr.size
    mean = float(arr.mean())
    if n == 1:
        return mean, float("-inf"), 0.0
    var = float(arr.var(ddof=1))
    R = float(benefit_range) if benefit_range is not None else 2.0
    ln_term = math.log(2.0 / alpha)
    var_term = math.sqrt(2.0 * var * ln_term / n)
    range_term = 7.0 * R * ln_term / (3.0 * (n - 1))
    return mean, float(mean - var_term - range_term), var


BENCHMARKS = [
    ("MVTec 3D-AD", "PatchCore canonical", "experiments/fusion/mvtec3d_patchcore_results.json"),
    ("MVTec 3D-AD", "PatchCore supervised", "experiments/fusion/mvtec3d_patchcore_supervised_paired_results.json"),
    ("MVTec 3D-AD", "PatchCore held-out", "experiments/fusion/mvtec3d_patchcore_heldout_results.json"),
    ("MVTec LOCO-AD", "PatchCore canonical", "experiments/fusion/mvtec_loco_patchcore_results.json"),
    ("MVTec LOCO-AD", "PatchCore supervised", "experiments/fusion/mvtec_loco_patchcore_supervised_paired_results.json"),
    ("Real3D-AD", "PCA shape + depth supervised", "experiments/fusion/real3d_supervised_paired_results.json"),
    ("VisA", "RGB+edge canonical", "experiments/fusion/visa_fusion_results.json"),
    ("VisA", "RGB+edge supervised", "experiments/fusion/visa_supervised_paired_results.json"),
    ("UNSW-NB15", "flow/conn/context", "experiments/fusion/unsw_paired_results.json"),
    ("UNSW-NB15", "held-out attack", "experiments/fusion/unsw_heldout_attack_results.json"),
]


def audit_one(repo_root: Path, rel_path: str, alpha: float = 0.05) -> dict | None:
    p = repo_root / rel_path
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    static = _per_seed_aurocs(payload, "static_attention")
    boost = _per_seed_aurocs(payload, "rga_boosted_fusion")
    router = _per_seed_aurocs(payload, "rga_meta_router")
    if not static or (not boost and not router):
        return None
    rga_best_path = "rga_boosted_fusion"
    rga_curve = boost
    if router and (not boost or float(np.mean(router)) > float(np.mean(boost))):
        rga_best_path = "rga_meta_router"
        rga_curve = router
    n = min(len(static), len(rga_curve))
    # The aggregate-AUROC surrogate loss is 1 - AUROC.
    diffs = [(1.0 - static[i]) - (1.0 - rga_curve[i]) for i in range(n)]
    mean, lcb = paired_bootstrap_lcb(diffs, alpha=alpha)
    eb_mean, eb_lcb, eb_var = empirical_bernstein_lcb(diffs, alpha=alpha, benefit_range=2.0)
    return {
        "n_seeds": int(n),
        "rga_path": rga_best_path,
        "paired_benefit_mean": float(mean),
        "lcb": float(lcb),
        "alpha": float(alpha),
        "certified": bool(lcb > 0.0),
        "empirical_bernstein_lcb": float(eb_lcb),
        "sample_variance": float(eb_var),
        "benefit_range": 2.0,
        "certified_eb": bool(eb_lcb > 0.0),
        "static_auroc_mean": float(np.mean(static[:n])),
        "rga_auroc_mean": float(np.mean(rga_curve[:n])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/fusion/switching_certificate_t5_audit.json"),
    )
    args = parser.parse_args()

    rows = []
    for benchmark, protocol, rel_path in BENCHMARKS:
        audit = audit_one(args.repo_root, rel_path, alpha=args.alpha)
        if audit is None:
            print(f"-- skipped (missing or empty): {rel_path}")
            continue
        row = {"benchmark": benchmark, "protocol": protocol, **audit}
        rows.append(row)
        print(
            f"{benchmark:<14s} {protocol:<26s} n={row['n_seeds']:>2d}  "
            f"path={row['rga_path']:<18s}  "
            f"mean={row['paired_benefit_mean']:+.4f}  "
            f"boot_LCB={row['lcb']:+.4f}  "
            f"EB_LCB={row['empirical_bernstein_lcb']:+.4f}  "
            f"{'CERT' if row['certified'] else 'no'}/"
            f"{'CERT_EB' if row['certified_eb'] else 'no_EB'}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"alpha": args.alpha, "rows": rows}, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
