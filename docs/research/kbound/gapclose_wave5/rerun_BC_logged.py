"""WIN_HUNT_v4 re-analysis runner — arms B/C on LOGGED records (bars per YAML).

Applies estimator_v2 (arm B) / tau_adaptive (arm C) under the ORIGINAL
leave-one-seed splits; scores vs the incumbent KGA (global tau' gate + single-GBR
signed radius) with paired-bootstrap 10^4 improvement CIs; writes per-yaml
WIN / CI_ROBUST_IMPROVEMENT flags. Fits NOTHING on held-out labels. Arm B:
CI_ROBUST_IMPROVEMENT iff FA_u<=a, regret_v2<=incumbent, improvement CI>0. Arm C
(camelyon v2): abstention must drop below 0.595 (FA_u<=a); WIN iff regret_v2 beats
always-adapt with the improvement CI excluding zero. Schema-defensive (exit 3;
assumptions in JSON "field_notes"). Uses single blank separators to stay <=230 ln.
Run: .venv/bin/python .../rerun_BC_logged.py --arm C --dataset camelyon17 --run-dir D
"""
from __future__ import annotations
import argparse, glob, json, os, sys  # noqa: E401
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
from estimator_v2 import CrossFitBenefitEstimator  # noqa: E402
from radius_v2 import crossfit_oof, signed_bounds  # noqa: E402
from tau_adaptive import calibrate_dev_terciles, tau_adaptive  # noqa: E402
from tau_selfnorm import tau_selfnorm  # noqa: E402

ALPHA = 0.10          # radius / FA level (natural protocol)
GATE_ALPHA = 0.05     # CEI (tau') test level
MIN_M = 20            # minimum panel size to run the gate
CAMELYON_ABST = 0.595
NBOOT = 10000

def _fail(msg):
    print(f"SCHEMA ERROR: {msg}", file=sys.stderr)
    sys.exit(3)

def _get(r, *keys):
    for k in keys:
        if r.get(k) is not None:
            return r[k]
    return None

def load_rows(run_dir, dataset):
    pat = os.path.join(run_dir, f"per_condition_{dataset}_*_seed*.json")
    files = [f for f in sorted(glob.glob(pat))
             if not os.path.basename(f).startswith("._")]
    if not files:  # retro-style single result_*.json fallback
        files = [f for f in sorted(glob.glob(os.path.join(run_dir, "*.json")))
                 if not os.path.basename(f).startswith("._")]
    if not files:
        _fail(f"no logged json under {run_dir} for {dataset}")
    rows = []
    for f in files:
        d = json.load(open(f))
        recs = d.get("records", d) if isinstance(d, dict) else d
        if isinstance(recs, dict):
            recs = recs.get("records") or recs.get("results") or recs.get("cells") or []
        rows += [r for r in recs if isinstance(r, dict)]
    if not rows:
        _fail("logged files contained no record dicts")
    return rows, [os.path.basename(f) for f in files]

def to_arrays(rows, need_gate):
    def req(name, keys, cast):
        vals = [_get(r, *keys) for r in rows]
        if any(v is None for v in vals):
            _fail(f"field {name} missing in some records")
        return np.array([cast(v) for v in vals])
    a0 = req("a0", ["a0"], float)
    aa = req("a_adapted/aa", ["a_adapted", "aa"], float)
    g = req("seed", ["seed"], int)
    Zl = [_get(r, "Z") for r in rows]
    if any(z is None for z in Zl):
        _fail("field Z missing in some records")
    B = np.array([float(_get(r, "B")) if _get(r, "B") is not None
                  else float(av - a0v) for r, av, a0v in zip(rows, aa, a0)])
    has_ev2 = any(r.get("Z_ev2") for r in rows)
    dim = len(next((r["Z_ev2"] for r in rows if r.get("Z_ev2")), [])) if has_ev2 else 0
    Z = np.array([list(map(float, z))
                  + (list(map(float, r["Z_ev2"])) if r.get("Z_ev2")
                     else [0.0] * dim) for z, r in zip(Zl, rows)])
    cond = [str(_get(r, "condition", "cond") or i) for i, r in enumerate(rows)]
    # gate inputs ALWAYS loaded (soft): the global tau' gate is part of the
    # incumbent KGA and of arm B. Missing c_ij => that record not gated (no-op);
    # only arm C HARD-FAILS when NO record carries c_ij.
    C = [np.asarray(_get(r, "c_ij"), float) if _get(r, "c_ij") is not None
         else None for r in rows]
    nD = [int(_get(r, "n_D")) if _get(r, "n_D") is not None else 0 for r in rows]
    if need_gate and all(c is None for c in C):
        _fail("arm C needs c_ij on records (drift-conditioned gate); none present")
    return dict(Z=Z, a0=a0, aa=aa, B=B, g=g, C=C, nD=nD, cond=cond, has_ev2=has_ev2)

def incumbent_radius_dec(Z, B, g):
    """Incumbent radius: single-GBR + signed leave-one-seed conformal bounds."""
    bhat = crossfit_oof(Z, B, g)
    resid = B - bhat
    dec = np.zeros(len(B), int)
    for s in np.unique(g):
        cal, te = g != s, g == s
        lo, hi = signed_bounds(resid[cal], ALPHA)
        d = np.zeros(int(te.sum()), int)
        d[bhat[te] + lo > 0] = 1
        d[bhat[te] + hi < 0] = -1
        dec[te] = d
    return dec

def v2_radius_dec(Z, B, g):
    return CrossFitBenefitEstimator(alpha=ALPHA).fit(Z, B, groups=g).oof_decide()[0]

def gate_reject(A, nsim, adaptive):
    """Per-record CEI rejection (True = CEI violated => gate abstains). Arm C
    freezes tercile thresholds per held-out seed on the OTHER seeds (dev-only)."""
    C, nD, g, cond = A["C"], A["nD"], A["g"], A["cond"]
    n = len(g)
    reject, gated, calib_by_seed = np.zeros(n, bool), np.zeros(n, bool), {}
    if adaptive:
        for s in np.unique(g):
            idx = [i for i in range(n)
                   if g[i] != s and C[i] is not None and nD[i] >= MIN_M]
            dc, dm = [C[i] for i in idx], [nD[i] for i in idx]
            calib_by_seed[int(s)] = (
                calibrate_dev_terciles(dc, dm, GATE_ALPHA, n_sim=nsim,
                                       seed=int(s) + 13) if len(dc) >= 6 else None)
    cache = {}
    for i in range(n):
        if C[i] is None or nD[i] < MIN_M:
            continue
        gated[i] = True
        key = (cond[i], int(g[i]), adaptive)
        if key not in cache:
            sd = abs(hash(key)) % (1 << 30)
            if adaptive:
                res = tau_adaptive(C[i], int(nD[i]), alpha=GATE_ALPHA, n_sim=nsim,
                                   seed=sd, calib=calib_by_seed.get(int(g[i])))
            else:
                res = tau_selfnorm(C[i], int(nD[i]), alpha=GATE_ALPHA,
                                   n_sim=nsim, seed=sd)
            cache[key] = bool(res["reject_H"])
        reject[i] = cache[key]
    return reject, gated, calib_by_seed

def realized(dec_radius, reject, a0, aa):
    dec = dec_radius.copy()
    dec[reject] = 0  # gate rejects CEI -> cannot certify -> ABSTAIN (keep source)
    return np.where(dec == 1, aa, a0), dec

def paired_ci(diff, rng):
    idx = rng.integers(0, len(diff), (NBOOT, len(diff)))
    m = diff[idx].mean(1)
    return (float(diff.mean()), float(np.percentile(m, 2.5)),
            float(np.percentile(m, 97.5)))

def score_arm(arm, A, nsim):
    a0, aa, B, g = A["a0"], A["aa"], A["B"], A["g"]
    oracle = np.maximum(a0, aa)
    rng = np.random.default_rng(20260704)
    inc_gate, gated, _ = gate_reject(A, nsim, adaptive=False)
    inc_acc, inc_dec = realized(incumbent_radius_dec(A["Z"], B, g), inc_gate, a0, aa)
    if arm == "B":  # estimator upgrade; gate unchanged
        v2_acc, v2_dec = realized(v2_radius_dec(A["Z"], B, g), inc_gate, a0, aa)
        calib_used = None
    else:           # arm C: gate upgrade; radius unchanged
        v2_gate, _, calib_used = gate_reject(A, nsim, adaptive=True)
        v2_acc, v2_dec = realized(incumbent_radius_dec(A["Z"], B, g), v2_gate, a0, aa)
    r_inc, r_v2, r_adapt = oracle - inc_acc, oracle - v2_acc, oracle - aa
    fa_u = float(np.mean((v2_dec == 1) & (B <= 0)))
    abst = float(np.mean(v2_dec == 0))
    imp_inc = paired_ci(r_inc - r_v2, rng)
    imp_adapt = paired_ci(r_adapt - r_v2, rng)
    out = dict(
        n_records=len(B), n_seeds=int(len(np.unique(g))), n_gated=int(gated.sum()),
        regret=dict(v2=float(r_v2.mean()), incumbent=float(r_inc.mean()),
                    always_adapt=float(r_adapt.mean()),
                    always_freeze=float((oracle - a0).mean())),
        FA_u=fa_u, alpha=ALPHA,
        abstention=dict(v2=abst, incumbent=float(np.mean(inc_dec == 0))),
        improvement_vs_incumbent=dict(mean=imp_inc[0], ci95=[imp_inc[1], imp_inc[2]]),
        improvement_vs_always_adapt=dict(mean=imp_adapt[0],
                                         ci95=[imp_adapt[1], imp_adapt[2]]))
    if arm == "B":
        ci_robust = fa_u <= ALPHA and r_v2.mean() <= r_inc.mean() and imp_inc[1] > 0
        out["CI_ROBUST_IMPROVEMENT"] = bool(ci_robust)
        out["VERDICT"] = ("CI_ROBUST_IMPROVEMENT" if ci_robust else "NO_HARM"
                          if (fa_u <= ALPHA and r_v2.mean() <= r_inc.mean()) else "FAIL")
    else:
        drop = abst < CAMELYON_ABST
        win = drop and fa_u <= ALPHA and r_v2.mean() < r_adapt.mean() and imp_adapt[1] > 0
        out["abstention_below_0p595"] = bool(drop)
        out["WIN"] = bool(win)
        out["CI_ROBUST_IMPROVEMENT"] = bool(fa_u <= ALPHA and imp_inc[1] > 0)
        out["VERDICT"] = ("WIN" if win else "ABSTENTION_DROP_NO_WIN"
                          if (drop and fa_u <= ALPHA) else "NO_WIN")
        out["calib_frozen_per_heldout_seed"] = {
            str(k): (v and dict(cuts=v["cuts"], mult=v["mult"]))
            for k, v in (calib_used or {}).items()}
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["B", "C", "BC"], required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--nsim", type=int, default=250, help="tau bootstrap draws")
    args = ap.parse_args()
    arms = ["B", "C"] if args.arm == "BC" else [args.arm]
    rows, files = load_rows(args.run_dir, args.dataset)
    field_notes = dict(
        a_adapted_key="r['a_adapted'] else r['aa']",
        benefit_key="r['B'] else (a_adapted - a0)",
        incumbent_kga="global tau_selfnorm gate + single-GBR signed LOO radius; "
                      "reject => ABSTAIN",
        arm_C_calibration="terciles frozen per held-out seed on other seeds",
        gate_min_panel=MIN_M, camelyon_abstention_bar=CAMELYON_ABST)
    for arm in arms:
        A = to_arrays(rows, need_gate=(arm == "C"))
        out = dict(protocol=f"WIN_HUNT_v4_ARM_{arm}",
                   registered="research_lock/WIN_HUNT_v4_PROTOCOL.yaml",
                   dataset=args.dataset, run_dir=args.run_dir, files=files,
                   used_ev2=A["has_ev2"], nsim=args.nsim,
                   field_notes=field_notes, **score_arm(arm, A, args.nsim))
        print(json.dumps({k: v for k, v in out.items()
                          if k != "calib_frozen_per_heldout_seed"}, indent=1))
        p = os.path.join(REPO, "research_lock",
                         f"WIN_HUNT_v4_ARM_{arm}_{args.dataset}_result.json")
        with open(p, "w") as f:
            json.dump(out, f, indent=1)
        print(f"saved {p}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
