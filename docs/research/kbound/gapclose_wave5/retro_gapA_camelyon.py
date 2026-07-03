"""Gap A retro-run on the REAL logged Camelyon17 n256 grid (frozen acceptance).

Data: experiments/kbound/results/wilds_kbound_debug_mps/result_73add410.json
(sha8 5d286065; 432 records, seeds 0-3, n_eval = 256) — the exact file behind
the published bias-variance diagnostic (eps_256 = 0.1127, floor 0.0441, 2.55x).

Replication + repair, leave-one-seed cross-fit (identical protocol to diag):
  V0_baseZ   published pipeline (symmetric |resid| Q90 on evidence-only Z)
  V1_baseZ   signed asymmetric quantiles, evidence-only Z
  V1/V2/V3_augZ  signed (+ ridge orthogonalization / Mondrian) on Z augmented
             with the OBSERVABLE condition metadata (comp, domain, aggr, mode,
             candidate) the published estimator never saw.
Acceptance (frozen): best ratio80 < 1.5 AND FA <= 0.10 + 2 MC-se AND
per-direction coverage >= 0.88. Honest negative reported as FAILS.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from radius_v2 import Z80, Z90, evaluate_variant  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DATA = os.path.join(
    REPO, "experiments/kbound/results/wilds_kbound_debug_mps/result_73add410.json")
ALPHA, N_EVAL = 0.10, 256


def load():
    d = json.load(open(DATA))
    recs = d["records"] if isinstance(d, dict) and "records" in d else d
    if isinstance(recs, dict):
        for k in ("records", "results", "cells"):
            if k in recs:
                recs = recs[k]
                break
    Z = np.array([r["Z"] for r in recs], dtype=float)
    B = np.array([r["B"] for r in recs], dtype=float)
    g = np.array([r["seed"] for r in recs])
    a0 = np.array([r["a0"] for r in recs], dtype=float)
    aa = np.array([r["aa"] for r in recs], dtype=float)
    sig = np.sqrt(a0 * (1 - a0) / N_EVAL + aa * (1 - aa) / N_EVAL)
    # observable condition metadata -> one-hot
    cats = []
    for key in ("comp", "domain", "aggr", "mode", "candidate"):
        vals = sorted({str(r.get(key)) for r in recs})
        if 1 < len(vals) <= 12:
            M = np.array([[1.0 if str(r.get(key)) == v else 0.0 for v in vals]
                          for r in recs])
            cats.append(M)
    Za = np.hstack([Z] + cats) if cats else Z
    return Z, Za, B, g, sig, len(recs)


def main() -> int:
    Zb, Za, B, g, sig, n = load()
    print(f"loaded {n} records | Z dim {Zb.shape[1]} -> augmented {Za.shape[1]} "
          f"| seeds {sorted(set(g.tolist()))} | harmful rate {(B<=0).mean():.3f}",
          file=sys.stderr)
    results = {}
    results["V0_baseZ"] = evaluate_variant(Zb, B, g, ALPHA, "V0", sigma_meas=sig)
    results["V1_baseZ"] = evaluate_variant(Zb, B, g, ALPHA, "V1", sigma_meas=sig)
    for v in ("V1", "V2", "V3", "V4"):
        results[f"{v}_augZ"] = evaluate_variant(Za, B, g, ALPHA, v, sigma_meas=sig)

    # published-diag replication numbers for continuity
    sm = float(np.mean(sig))
    repl = dict(eps_meas_legacy=Z90 * sm, w_meas_80=Z80 * sm,
                published=dict(eps_256_observed=0.11266, eps_meas=0.04411,
                               ratio=2.554))

    cand = {k: r for k, r in results.items() if k != "V0_baseZ"}
    best = min(cand, key=lambda k: cand[k]["ratio80"])
    r = cand[best]
    accept = (r["ratio80"] < 1.5
              and r["fa_emp"] <= ALPHA + 2 * r["fa_mc_se"]
              and r["cov_lo"] >= 0.88 and r["cov_hi"] >= 0.88)
    out = dict(data=os.path.relpath(DATA, REPO), n_records=n,
               replication=repl, results=results, best_variant=best,
               ACCEPTANCE_ratio_lt_1p5=bool(accept))
    print(json.dumps(out, indent=1, default=float))
    with open(os.path.join(HERE, "retro_gapA_results.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
