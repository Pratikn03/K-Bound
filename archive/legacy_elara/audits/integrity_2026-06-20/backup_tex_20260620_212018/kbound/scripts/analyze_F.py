"""
analyze_F.py — Protocol F post-hoc estimator/analysis for rich-evidence Camelyon17.

Pre-registered in research_lock/RICH_EVIDENCE_CAMELYON_PROTOCOL_F_v1.yaml.
This is the CHEAP held-out estimator step: it consumes the per-cell x seed records
serialized by run_wilds_camelyon17.py (--evidence-panel rich) and applies the locked
estimator pipeline. The GPU run only does forward passes + serialization; this script
does the (torch-free) debias + conditional-conformal decision and held-out evaluation.

INTEGRITY (locked, byte-identical to Protocol B where shared):
  - alpha = 0.10 FIXED. tau* / decision-rule NOT tuned.
  - Decision rule UNCHANGED: adapt if (lower bound) > 0; freeze if (upper bound) < 0;
    else abstain. Implemented as Bhat - eps > 0 / Bhat + eps < 0 for the global-eps /
    Mondrian point estimators, and via conditional quantiles for CQR.
  - Split BY SEED: DEV = {0,1} (estimator finalized on DEV only), TEST = {2,3,4}
    evaluated ONCE. (Matches analysis_plan_locked in the protocol.)

ESTIMATOR (the only NEW knob, on the NEW rich Z):
  - PPI / doubly-robust debias of B_hat using LABELED SOURCE residuals on D
    (arXiv:2301.09633): fit a point estimator B_hat(Z), then regress the CAL/DEV
    residual (B - B_hat) on Z and subtract the predicted residual (rectifier).
  - Conditional conformal radius instead of one global eps:
      * mondrian : per-composition (cell) eps  (arXiv:2305.12616)
      * cqr      : conformalized quantile regression radius on Z  (arXiv:1905.03222)

USAGE:
  python docs/research/kbound/scripts/analyze_F.py \\
      --records experiments/kbound/results/camelyon17_richZ_F_v1/wilds_camelyon17_kga.json \\
      --output-dir experiments/kbound/results/camelyon17_richZ_F_v1 \\
      --estimator ppi_debias --conformal mondrian \\
      --dev-seeds 0 1 --test-seeds 2 3 4
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

ALPHA = 0.10  # FIXED — never tuned


# ─── record loading ──────────────────────────────────────────────────────────

def _one_record(r, candidate=None):
    aa = r.get("aa", r.get("a_adapted"))
    if aa is None:
        raise KeyError("record missing aa / a_adapted")
    return {
        "seed": int(r["seed"]),
        "Z": list(r["Z"]),
        "B": float(r["B"]),
        "a0": float(r["a0"]),
        "aa": float(aa),
        "comp": r.get("comp", r.get("condition", r.get("cell", r.get("method", "unknown")))),
        "candidate": r.get("candidate", r.get("method", candidate or "unknown")),
    }


def load_records(path, candidate=None):
    """Read run_wilds_camelyon17.py output JSON -> flat list of cell x seed records.
    Each record carries Z (the FULL rich Z), B, a0, aa, seed, method (cell).
    Also accepts CIFAR-10.1 per_condition JSON (a_adapted field).
    If candidate is set, keep only records with that adapter name (e.g. eata_online)."""
    d = json.load(open(path))
    recs = []
    if d.get("records"):
        panel = d.get("evidence_panel", d.get("config", {}).get("evidence_panel",
                d.get("benchmark", "unknown")))
        for r in d.get("records", []):
            recs.append(_one_record(r, candidate=candidate))
        if candidate:
            recs = [r for r in recs if r.get("candidate") == candidate]
        return recs, panel
    for method, entry in d.get("methods", {}).items():
        for r in entry.get("records", []):
            recs.append({
                "seed": int(r["seed"]),
                "Z": list(r["Z"]),
                "B": float(r["B"]),
                "a0": float(r["a0"]),
                "aa": float(r["aa"]),
                "comp": r.get("cell", r.get("method", method)),
            })
    if candidate:
        recs = [r for r in recs if r.get("candidate") == candidate]
    return recs, d.get("evidence_panel", "unknown")


def arrays(records):
    Z = np.array([r["Z"] for r in records], float)
    B = np.array([r["B"] for r in records], float)
    a0 = np.array([r["a0"] for r in records], float)
    aa = np.array([r["aa"] for r in records], float)
    sd = np.array([r["seed"] for r in records])
    comp = np.array([r["comp"] for r in records])
    return Z, B, a0, aa, sd, comp


# ─── decision + metrics (decision rule UNCHANGED) ─────────────────────────────

def decide_global(Bhat_t, eps):
    return np.where(Bhat_t - eps > 0, "ADAPT",
           np.where(Bhat_t + eps < 0, "FREEZE", "ABSTAIN"))


def metrics(dec, Bt, a0t, aat):
    adapt = dec == "ADAPT"
    commit = dec != "ABSTAIN"
    kga = np.where(adapt, aat, a0t)
    oracle = np.maximum(a0t, aat)
    return {
        "commit_rate": float(commit.mean()),
        "coverage": float(commit.mean()),
        "adapt_rate": float(adapt.mean()),
        "false_adapt": float(np.mean(Bt[adapt] < 0)) if adapt.any() else 0.0,
        "regret_kga": float((oracle - kga).mean()),
        "regret_adapt": float((oracle - aat).mean()),
        "regret_freeze": float((oracle - a0t).mean()),
        "n_test": int(len(Bt)),
    }


# ─── estimator: PPI / doubly-robust debias + conditional conformal ────────────

def fit_point(Zc, Bc):
    """Base point estimator B_hat(Z): GBR identical-spec to decide_kga's learner."""
    return GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=0).fit(Zc, Bc)


def ppi_debias(Bhat_c, Bc, Zc, Zt, Bhat_t):
    """PPI / doubly-robust rectifier: regress CAL residual (B - Bhat) on Z and
    subtract predicted residual from both CAL and TEST predictions. arXiv:2301.09633."""
    resid = Bc - Bhat_c
    db = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                   subsample=0.8, random_state=1).fit(Zc, resid)
    return Bhat_c + db.predict(Zc), Bhat_t + db.predict(Zt)


def run_split(records, cal_seeds, test_seeds, estimator="ppi_debias", conformal="mondrian",
              frozen_eps=None):
    Z, B, a0, aa, sd, comp = arrays(records)
    cal = np.isin(sd, cal_seeds); tst = np.isin(sd, test_seeds)
    if cal.sum() < 2 or tst.sum() == 0:
        return None
    Zc, Bc = Z[cal], B[cal]
    Zt, Bt, a0t, aat = Z[tst], B[tst], a0[tst], aa[tst]
    compc, compt = comp[cal], comp[tst]

    # ── CQR: conditional conformal via conformalized quantile regression on Z ──
    if conformal == "cqr":
        sc = StandardScaler().fit(Zc); Zcs = sc.transform(Zc); Zts = sc.transform(Zt)
        qlo = QuantileRegressor(quantile=ALPHA, alpha=1e-3, solver="highs").fit(Zcs, Bc)
        qhi = QuantileRegressor(quantile=1 - ALPHA, alpha=1e-3, solver="highs").fit(Zcs, Bc)
        # conformity score E = max(qlo - B, B - qhi) on CAL; widen by its (1-alpha) quantile
        e = np.maximum(qlo.predict(Zcs) - Bc, Bc - qhi.predict(Zcs))
        q = float(np.quantile(e, 1 - ALPHA))
        Blo_t = qlo.predict(Zts) - q; Bhi_t = qhi.predict(Zts) + q
        # decision rule UNCHANGED in spirit: adapt if lower bound > 0; freeze if upper < 0
        dec = np.where(Blo_t > 0, "ADAPT", np.where(Bhi_t < 0, "FREEZE", "ABSTAIN"))
        return metrics(dec, Bt, a0t, aat)

    # ── point estimator (+ optional PPI debias) + conformal eps ──
    m = fit_point(Zc, Bc)
    Bhat_c = m.predict(Zc); Bhat_t = m.predict(Zt)
    if estimator == "ppi_debias":
        Bhat_c, Bhat_t = ppi_debias(Bhat_c, Bc, Zc, Zt, Bhat_t)
    elif estimator != "gbr":
        raise ValueError(estimator)

    if conformal == "global":
        eps = float(np.quantile(np.abs(Bhat_c - Bc), 1 - ALPHA))
        dec = decide_global(Bhat_t, eps)
    elif conformal == "mondrian":
        # per-composition (cell) eps; fall back to global eps for tiny/unseen groups
        eps_glob = float(np.quantile(np.abs(Bhat_c - Bc), 1 - ALPHA))
        dec = np.array(["ABSTAIN"] * len(Bhat_t), dtype=object)
        groups = set(compc.tolist())
        for g in groups:
            mc = compc == g
            epsg = (float(np.quantile(np.abs(Bhat_c[mc] - Bc[mc]), 1 - ALPHA))
                    if mc.sum() >= 5 else eps_glob)
            mt = compt == g
            dec[mt] = decide_global(Bhat_t[mt], epsg)
        unseen = ~np.isin(compt, list(groups))
        dec[unseen] = decide_global(Bhat_t[unseen], eps_glob)
    elif conformal == "frozen":
        # Globally frozen eps (e.g. synthetic-grid transplant); alpha/tau* untouched.
        if frozen_eps is None:
            raise ValueError("conformal='frozen' requires frozen_eps")
        dec = decide_global(Bhat_t, float(frozen_eps))
    else:
        raise ValueError(conformal)
    return metrics(dec, Bt, a0t, aat)


# ─── self-test (torch-free, tiny synthetic records) ───────────────────────────

def self_test():
    rng = np.random.default_rng(0)
    recs = []
    for s in range(5):
        for c in ("tent", "eata"):
            z = rng.normal(size=16).tolist()
            b = float(rng.normal(0.02, 0.05))
            recs.append({"seed": s, "Z": z, "B": b,
                         "a0": float(rng.uniform(.6, .8)),
                         "aa": float(rng.uniform(.6, .8)), "comp": c})
    for est, con in [("gbr", "global"), ("ppi_debias", "mondrian"), ("ppi_debias", "cqr")]:
        m = run_split(recs, [0, 1], [2, 3, 4], estimator=est, conformal=con)
        assert m is not None and "false_adapt" in m, (est, con)
        print(f"  self-test [{est:11s}+{con:8s}]: OK  "
              f"FA={m['false_adapt']} commit={round(m['commit_rate'],3)} n={m['n_test']}")
    print("analyze_F self-test PASSED")


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Protocol F post-hoc estimator/analysis")
    p.add_argument("--records", nargs="+", help="One or more record JSON paths (wilds or per_condition)")
    p.add_argument("--candidate", default=None,
                   help="Optional adapter filter (e.g. eata_online for Protocol G)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--estimator", choices=["gbr", "ppi_debias"], default="ppi_debias")
    p.add_argument("--conformal", choices=["global", "mondrian", "cqr", "frozen"], default="mondrian")
    p.add_argument("--frozen-eps", type=float, default=None, dest="frozen_eps",
                   help="When --conformal frozen: globally fixed eps (e.g. synthetic-grid transplant)")
    p.add_argument("--dev-seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--test-seeds", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--self-test", action="store_true",
                   help="Run torch-free self-test on synthetic records (no real data needed)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test(); return
    if not args.records:
        print("ERROR: --records required (or use --self-test)"); sys.exit(2)

    recs, panel = [], "unknown"
    for rp in args.records:
        part, panel = load_records(rp, candidate=args.candidate)
        recs.extend(part)
    seeds = sorted(set(r["seed"] for r in recs))
    out = {"alpha": ALPHA, "evidence_panel": panel,
           "candidate": args.candidate,
           "estimator": args.estimator, "conformal": args.conformal,
           "dev_seeds": args.dev_seeds, "test_seeds": args.test_seeds,
           "n_records": len(recs), "seeds_present": seeds,
           "Z_dim": (len(recs[0]["Z"]) if recs else None)}

    # held-out TEST (evaluated ONCE): fit on DEV, evaluate on TEST.
    out["test_locked"] = run_split(recs, args.dev_seeds, args.test_seeds,
                                   estimator=args.estimator, conformal=args.conformal,
                                   frozen_eps=args.frozen_eps)
    # reference: legacy global-eps GBR on the SAME split (context, not the locked choice)
    out["test_baseline_gbr_global"] = run_split(recs, args.dev_seeds, args.test_seeds,
                                                estimator="gbr", conformal="global",
                                                frozen_eps=args.frozen_eps)
    if args.conformal == "frozen":
        out["frozen_eps"] = args.frozen_eps

    print(f"=== Protocol F analysis  (panel={panel}, Z_dim={out['Z_dim']}) ===")
    print(f"  estimator={args.estimator}  conformal={args.conformal}  alpha={ALPHA} (FIXED)")
    print(f"  DEV={args.dev_seeds}  TEST={args.test_seeds}  (TEST evaluated ONCE)")
    print(f"  locked   : {out['test_locked']}")
    print(f"  baseline : {out['test_baseline_gbr_global']}")
    tl = out["test_locked"]
    if tl:
        bb = (tl["regret_kga"] < tl["regret_adapt"] and tl["regret_kga"] < tl["regret_freeze"])
        out["beats_both"] = bool(bb)
        fa_ok = tl["false_adapt"] is not None and tl["false_adapt"] <= ALPHA
        win = bool(fa_ok and bb)
        print(f"  beats_both: {bb}  (FA<=alpha: {fa_ok})")
        print(f"  VERDICT (Tier-B headline): {'WIN' if win else 'not-cleared'}")
        out["verdict_win"] = win

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        op = os.path.join(args.output_dir, "analyze_F_results.json")
        json.dump(out, open(op, "w"), indent=2)
        print(f"Saved {op}")


if __name__ == "__main__":
    main()
