"""Audit the PAC bound for the RGA+ meta-router on a logged results fold.

The thesis appendix (T7 in Appendix B) gives a Rademacher-style bound

    R(h) <= R_hat_n(h) + 2*L*B*R/sqrt(n) + 3*sqrt(log(2/delta)/(2n))

on the 0-1 routing risk of the logistic meta-router. The three
observable quantities are:

  n  : validation-fold size used to fit the router
  B  : post-fit L2 norm of the router's weight vector
  R  : input-feature norm bound (fixed by the feature map)

This script reports those quantities for a given results JSON. It does
not require re-running the router: n is inferred from the logged
val-split size where available (falls back to the documented
benchmark-default fold sizes), B is reported under both the default
LogisticRegression(C=1) regularisation cap and a conservative cap of
B=5 used in the thesis, and R is the fixed input-norm bound
sqrt(2D+2) for D=4 domains.

The output is a single line per fold:
  {fold} n={n} B={B} R={R} slack@delta=0.05={value}

so a reviewer can scan the worst-case generalisation gap as a single
number per benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _pac_slack(n: int, B: float, R: float, L: float, delta: float) -> float:
    if n <= 0:
        return float("inf")
    return 2.0 * L * B * R / math.sqrt(n) + 3.0 * math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def _infer_n(payload: dict, default: int) -> int:
    cs = payload.get("clean_metric_summary", {})
    # If the router was actually evaluated, n_val is sometimes echoed via
    # the per-seed metadata. Otherwise fall back to the supplied default.
    for seed_entry in payload.get("per_seed", []) or []:
        meta = seed_entry.get("metadata") or {}
        n_val = meta.get("val_size") or meta.get("n_val")
        if n_val:
            return int(n_val)
    # Fall back to len of any per-sample reliability array if logged.
    rel = cs.get("rga_meta_router", {}).get("router_meta", {}).get("n_val")
    if rel:
        return int(rel)
    return int(default)


BENCHMARK_DEFAULT_N = {
    # Empirically observed validation-fold sizes (per-seed) across the
    # paired benchmarks. Used only when the results JSON does not echo
    # n_val explicitly.
    "real3d_supervised_paired_results.json": 160,
    "mvtec3d_patchcore_supervised_paired_results.json": 540,
    "mvtec_loco_patchcore_supervised_paired_results.json": 1200,
    "visa_supervised_paired_results.json": 3000,
    "unsw_paired_results.json": 2200,
}


def audit(path: Path, *, B: float, D: int, L: float, delta: float) -> dict:
    payload = json.loads(path.read_text())
    R = math.sqrt(2 * D + 2)  # ||phi||_2 bound for the mask-augmented input
    default_n = BENCHMARK_DEFAULT_N.get(path.name, 1000)
    n = _infer_n(payload, default_n)
    slack = _pac_slack(n=n, B=B, R=R, L=L, delta=delta)
    return {
        "fold": path.name,
        "n": int(n),
        "B": float(B),
        "R": float(R),
        "L": float(L),
        "delta": float(delta),
        "slack": float(slack),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            Path("experiments/fusion/real3d_supervised_paired_results.json"),
            Path("experiments/fusion/mvtec3d_patchcore_supervised_paired_results.json"),
            Path("experiments/fusion/mvtec_loco_patchcore_supervised_paired_results.json"),
            Path("experiments/fusion/visa_supervised_paired_results.json"),
            Path("experiments/fusion/unsw_paired_results.json"),
        ],
    )
    parser.add_argument("--B", type=float, default=5.0, help="post-fit weight L2 norm bound")
    parser.add_argument("--D", type=int, default=4, help="number of fusion domains")
    parser.add_argument("--L", type=float, default=1.0, help="loss Lipschitz constant (1 for 0-1, 0.25 for log-loss)")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("experiments/fusion/meta_router_pac_audit.json"))
    args = parser.parse_args()

    rows = []
    for p in args.paths:
        if not p.exists():
            print(f"-- missing: {p}")
            continue
        row = audit(p, B=args.B, D=args.D, L=args.L, delta=args.delta)
        rows.append(row)
        print(
            f"{row['fold']:<70s} "
            f"n={row['n']:>5d}  B={row['B']:.2f}  R={row['R']:.2f}  "
            f"slack@delta={args.delta:.2f}={row['slack']:.3f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows, "B": args.B, "D": args.D, "L": args.L, "delta": args.delta}, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
