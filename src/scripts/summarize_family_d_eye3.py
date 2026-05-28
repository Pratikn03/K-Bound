"""Aggregate Family-D D-EYE-3 seed results.

Computes test-fold summary statistics from the per-seed CSV:
  - Mean static AUC and bootstrap CI
  - Mean RGA AUC and bootstrap CI
  - Mean delta AUC (RGA - static) and bootstrap CI

Outputs:
  - CSV (single-row summary)
  - JSON (machine-readable)
  - Markdown (human-readable)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _bootstrap_ci(values: np.ndarray, n_iter: int, seed: int, alpha: float) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = float(sample.mean())
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="experiments/phase2/family_d/family_d_d_eye_3_full_test_evaluation_per_seed.csv",
        help="Input per-seed CSV path",
    )
    parser.add_argument(
        "--out-prefix",
        default="experiments/phase2/family_d/family_d_d_eye_3_aggregate_summary",
        help="Output prefix for .csv/.json/.md",
    )
    parser.add_argument("--bootstrap-iter", type=int, default=10000, help="Bootstrap iterations")
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="Bootstrap RNG seed")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided alpha for confidence interval")
    args = parser.parse_args()

    input_path = ROOT / args.input
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    rows: list[dict] = []
    with input_path.open() as f:
        rows = list(csv.DictReader(f))

    test_rows = [r for r in rows if str(r.get("fold", "")).strip().lower() == "test"]
    if not test_rows:
        raise SystemExit("No test rows found in input CSV.")

    static_vals = np.array([_to_float(r.get("static_auc")) for r in test_rows], dtype=object)
    rga_vals = np.array([_to_float(r.get("rga_auc")) for r in test_rows], dtype=object)
    delta_vals = np.array([_to_float(r.get("delta_auc")) for r in test_rows], dtype=object)

    def _clean(arr: np.ndarray) -> np.ndarray:
        out = [float(x) for x in arr if x is not None]
        return np.asarray(out, dtype=float)

    static = _clean(static_vals)
    rga = _clean(rga_vals)
    delta = _clean(delta_vals)

    if len(static) == 0 or len(rga) == 0 or len(delta) == 0:
        raise SystemExit("Missing numeric static_auc/rga_auc/delta_auc values in test rows.")

    if not (len(static) == len(rga) == len(delta)):
        raise SystemExit("Mismatched vector lengths across static/rga/delta test values.")

    ci_static = _bootstrap_ci(static, args.bootstrap_iter, args.bootstrap_seed, args.alpha)
    ci_rga = _bootstrap_ci(rga, args.bootstrap_iter, args.bootstrap_seed, args.alpha)
    ci_delta = _bootstrap_ci(delta, args.bootstrap_iter, args.bootstrap_seed, args.alpha)

    summary = {
        "cell_id": "D-EYE-3",
        "n_test_seeds": int(len(delta)),
        "mean_static_auc": float(static.mean()),
        "mean_rga_auc": float(rga.mean()),
        "mean_delta_auc": float(delta.mean()),
        "ci_alpha": float(args.alpha),
        "bootstrap_iter": int(args.bootstrap_iter),
        "bootstrap_seed": int(args.bootstrap_seed),
        "static_auc_ci_lo": ci_static[0],
        "static_auc_ci_hi": ci_static[1],
        "rga_auc_ci_lo": ci_rga[0],
        "rga_auc_ci_hi": ci_rga[1],
        "delta_auc_ci_lo": ci_delta[0],
        "delta_auc_ci_hi": ci_delta[1],
        "input_csv": str(input_path.relative_to(ROOT)),
    }

    out_prefix = ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_csv = out_prefix.with_suffix(".csv")
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    out_json = out_prefix.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    out_md = out_prefix.with_suffix(".md")
    ci_pct = int(round((1.0 - args.alpha) * 100.0))
    md = [
        "# D-EYE-3 Aggregate Summary",
        "",
        f"Input: `{summary['input_csv']}`",
        "",
        "| Metric | Mean | CI |",
        "|---|---:|---:|",
        (
            f"| Static AUC | {summary['mean_static_auc']:.4f} "
            f"| [{summary['static_auc_ci_lo']:+.4f}, {summary['static_auc_ci_hi']:+.4f}] ({ci_pct}%) |"
        ),
        (
            f"| RGA AUC | {summary['mean_rga_auc']:.4f} "
            f"| [{summary['rga_auc_ci_lo']:+.4f}, {summary['rga_auc_ci_hi']:+.4f}] ({ci_pct}%) |"
        ),
        (
            f"| Delta AUC (RGA - Static) | {summary['mean_delta_auc']:+.4f} "
            f"| [{summary['delta_auc_ci_lo']:+.4f}, {summary['delta_auc_ci_hi']:+.4f}] ({ci_pct}%) |"
        ),
        "",
        f"Seeds: {summary['n_test_seeds']}",
        f"Bootstrap: {summary['bootstrap_iter']} iterations, seed={summary['bootstrap_seed']}",
    ]
    out_md.write_text("\n".join(md) + "\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())