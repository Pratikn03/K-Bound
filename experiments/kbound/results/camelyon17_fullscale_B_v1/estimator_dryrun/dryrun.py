"""
EXPLORATORY estimator dry-run for the Camelyon17 adapt/freeze/abstain certificate.
QUESTION: can a better estimator / better use of logged Z reduce B_hat(Z) bias enough
that the certificate validly fires (false-adapt <= alpha=0.10 at commit >= 0.3) on a
HELD-OUT seed split, WITHOUT new evidence features?

INTEGRITY:
  - alpha = 0.10 FIXED. tau* / alpha NOT tuned.
  - Split BY SEED. DEV = seeds {0,1} (explore here). TEST = seeds {2,3} (evaluate ONCE).
  - The decide logic is the exact eps-recal from _locked_B_analysis.py:eps_recal_by_seed
    (adapt if Bhat-eps>0, freeze if Bhat+eps<0, else abstain; eps = (1-alpha) quantile
    of |Bhat-B| residuals on CAL). Mondrian variant uses per-group eps. Quantile-reg
    variant predicts a direct upper-bound radius instead of one global eps.
  - CIFAR sanity uses an analogous DEV/TEST seed split to confirm no regression.
"""
# --- defect D8: portable roots (docs/research/kbound/EXTERNAL_STORAGE_POLICY.md bans
# --- machine-local absolute paths in tracked code). This file previously hard-coded a
# --- Cowork *session sandbox* mount, which is worse than a
# --- home directory: it is valid only inside one ephemeral container.
import os as _kb_os
from pathlib import Path as _KbPath


def _kb_repo_root() -> str:
    override = _kb_os.environ.get("KBOUND_REPO_ROOT", "").strip()
    if override:
        return str(_KbPath(override).expanduser().resolve())
    here = _KbPath(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return str(candidate)
    raise RuntimeError(f"repository root not found above {here}; set KBOUND_REPO_ROOT")


KB_REPO_ROOT = _kb_repo_root()

import json, itertools, numpy as np, os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

ALPHA = 0.10
ROOT = KB_REPO_ROOT
CAM = f"{ROOT}/experiments/kbound/results/wilds_kbound_debug_mps/result_73add410.json"
CIFAR_DIR = f"{ROOT}/experiments/kbound/results/stress_grid_multiseed_v1"
OUTDIR = f"{ROOT}/experiments/kbound/results/camelyon17_fullscale_B_v1/estimator_dryrun"

DEV_SEEDS = [0, 1]
TEST_SEEDS = [2, 3]

# ---------------- data loaders ----------------
def load_camelyon():
    d = json.load(open(CAM))
    return d["records"]

def load_cifar():
    recs = []
    for s in range(5):
        for meth in ["tent", "sar", "eata"]:
            p = f"{CIFAR_DIR}/seed{s}/per_condition_cifar10c_{meth}_seed{s}.json"
            if not os.path.exists(p):
                continue
            dd = json.load(open(p))
            for r in dd["records"]:
                recs.append({"seed": r["seed"], "Z": r["Z"], "B": r["B"],
                             "a0": r["a0"], "aa": r["a_adapted"], "comp": r.get("condition","")})
    return recs

def arrays(records):
    Z = np.array([r["Z"] for r in records], float)
    B = np.array([r["B"] for r in records], float)
    a0 = np.array([r["a0"] for r in records], float)
    aa = np.array([r["aa"] for r in records], float)
    sd = np.array([r["seed"] for r in records])
    comp = np.array([r.get("comp","") for r in records])
    return Z, B, a0, aa, sd, comp

# ---------------- estimator builders ----------------
def make_point(variant):
    """Returns (fit(Z,B)->model, predict(model,Z)->Bhat) for point estimators."""
    if variant == "gbr_baseline":
        return GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                         subsample=0.8, random_state=0)
    if variant == "ridge_std":
        return Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    if variant == "ridge_poly":
        return Pipeline([("sc", StandardScaler()),
                         ("poly", PolynomialFeatures(2, include_bias=False)),
                         ("m", Ridge(alpha=1.0))])
    if variant == "rf":
        return RandomForestRegressor(n_estimators=300, max_depth=4, random_state=0)
    if variant == "mlp":
        return Pipeline([("sc", StandardScaler()),
                         ("m", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000,
                                            alpha=1e-2, random_state=0))])
    if variant == "isotonic_gbr":
        return "isotonic_gbr"  # special-cased
    raise ValueError(variant)

# ---------------- core decision (global eps) ----------------
def decide_global(Bhat_t, eps):
    return np.where(Bhat_t - eps > 0, "ADAPT",
           np.where(Bhat_t + eps < 0, "FREEZE", "ABSTAIN"))

def metrics(dec, Bt, a0t, aat):
    adapt = dec == "ADAPT"
    kga = np.where(adapt, aat, a0t)
    oracle = np.maximum(a0t, aat)
    return {
        "commit_rate": float(adapt.mean()),
        "coverage": float(np.mean(dec != "ABSTAIN")),
        "false_adapt": float(np.mean(Bt[adapt] < 0)) if adapt.any() else None,
        "regret_kga": float((oracle - kga).mean()),
        "regret_adapt": float((oracle - aat).mean()),
        "regret_freeze": float((oracle - a0t).mean()),
    }

# ---------------- one split, one variant ----------------
def run_split(records, cal_seeds, test_seeds, variant, debias=False, mondrian=None):
    Z, B, a0, aa, sd, comp = arrays(records)
    cal = np.isin(sd, cal_seeds); tst = np.isin(sd, test_seeds)
    if cal.sum() < 2 or tst.sum() == 0:
        return None
    Zc, Bc = Z[cal], B[cal]
    Zt, Bt, a0t, aat = Z[tst], B[tst], a0[tst], aa[tst]

    if variant.startswith("qr_"):  # quantile regression: direct heteroskedastic radius
        # median predictor + upper/lower conditional quantiles -> per-point asym radius
        sc = StandardScaler().fit(Zc); Zcs = sc.transform(Zc); Zts = sc.transform(Zt)
        med = QuantileRegressor(quantile=0.5, alpha=1e-3, solver="highs").fit(Zcs, Bc)
        # one-sided: to bound false-adapt we need a LOWER quantile of B given Z to be > 0.
        qlo = QuantileRegressor(quantile=ALPHA, alpha=1e-3, solver="highs").fit(Zcs, Bc)
        qhi = QuantileRegressor(quantile=1-ALPHA, alpha=1e-3, solver="highs").fit(Zcs, Bc)
        Blo_t = qlo.predict(Zts); Bhi_t = qhi.predict(Zts)
        # adapt if lower 0.10-quantile of B>0; freeze if upper 0.90-quantile <0
        dec = np.where(Blo_t > 0, "ADAPT", np.where(Bhi_t < 0, "FREEZE", "ABSTAIN"))
        return metrics(dec, Bt, a0t, aat)

    # point estimator + (optional debias) + conformal eps
    if variant == "isotonic_gbr":
        base = GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                         subsample=0.8, random_state=0).fit(Zc, Bc)
        raw_c = base.predict(Zc)
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_c, Bc)
        Bhat_c = iso.predict(raw_c)
        Bhat_t = iso.predict(base.predict(Zt))
    else:
        m = make_point(variant)
        m.fit(Zc, Bc)
        Bhat_c = m.predict(Zc); Bhat_t = m.predict(Zt)

    if debias:  # regress residual on Z (DEV/CAL), subtract predicted residual
        resid = Bc - Bhat_c
        db = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                       subsample=0.8, random_state=1).fit(Zc, resid)
        Bhat_c = Bhat_c + db.predict(Zc)
        Bhat_t = Bhat_t + db.predict(Zt)

    if mondrian is None:
        eps = float(np.quantile(np.abs(Bhat_c - Bc), 1 - ALPHA))
        dec = decide_global(Bhat_t, eps)
    else:
        # per-group eps. group by comp (composition) or by |Bhat| bin.
        if mondrian == "comp":
            gc = comp[cal]; gt = comp[tst]
        else:  # 'bhatbin' : quantile bins of |Bhat_c|
            ab = np.abs(Bhat_c)
            edges = np.quantile(ab, [0, .33, .66, 1.0]); edges[-1] += 1e-9
            gc = np.digitize(ab, edges[1:-1])
            gt = np.digitize(np.abs(Bhat_t), edges[1:-1])
        dec = np.array(["ABSTAIN"] * len(Bhat_t), dtype=object)
        groups = set(gc.tolist())
        # global fallback eps for groups too small in CAL
        eps_glob = float(np.quantile(np.abs(Bhat_c - Bc), 1 - ALPHA))
        for g in groups:
            mc = gc == g
            if mc.sum() >= 5:
                epsg = float(np.quantile(np.abs(Bhat_c[mc] - Bc[mc]), 1 - ALPHA))
            else:
                epsg = eps_glob
            mt = gt == g
            dec[mt] = decide_global(Bhat_t[mt], epsg)
        # any test group unseen in cal -> global eps
        unseen = ~np.isin(gt, list(groups))
        dec[unseen] = decide_global(Bhat_t[unseen], eps_glob)
    return metrics(dec, Bt, a0t, aat)

# ---------------- DEV exploration ----------------
VARIANTS = [
    ("gbr_baseline", dict()),
    ("ridge_std", dict()),
    ("ridge_poly", dict()),
    ("rf", dict()),
    ("mlp", dict()),
    ("isotonic_gbr", dict()),
    ("qr_hetero", dict()),                                  # quantile-reg direct radius
    ("gbr_baseline", dict(debias=True)),                   # debiasing
    ("gbr_baseline", dict(mondrian="comp")),               # Mondrian by composition
    ("gbr_baseline", dict(mondrian="bhatbin")),            # Mondrian by |Bhat| bin
    ("ridge_poly", dict(mondrian="bhatbin")),
]

def vname(v, kw):
    tag = v
    if kw.get("debias"): tag += "+debias"
    if kw.get("mondrian"): tag += f"+mondrian[{kw['mondrian']}]"
    return tag

def explore_dev(records):
    rows = []
    for v, kw in VARIANTS:
        try:
            m = run_split(records, DEV_SEEDS, DEV_SEEDS, v, **kw)  # DEV in-sample fit, but
            # honest DEV eval: fit on one dev seed, test on the other (within-DEV holdout)
            m_a = run_split(records, [0], [1], v, **kw)
            m_b = run_split(records, [1], [0], v, **kw)
            fa = [x["false_adapt"] for x in (m_a, m_b) if x and x["false_adapt"] is not None]
            cr = [x["commit_rate"] for x in (m_a, m_b) if x]
            rk = [x["regret_kga"] for x in (m_a, m_b) if x]
            rows.append({"variant": vname(v, kw),
                         "dev_false_adapt": float(np.mean(fa)) if fa else None,
                         "dev_commit_rate": float(np.mean(cr)) if cr else None,
                         "dev_regret_kga": float(np.mean(rk)) if rk else None})
        except Exception as e:
            rows.append({"variant": vname(v, kw), "error": str(e)})
    return rows

def pick_best(rows):
    # valid = commit >= 0.3 AND false_adapt <= alpha ; minimise false_adapt then regret
    valid = [r for r in rows if r.get("dev_false_adapt") is not None
             and r.get("dev_commit_rate", 0) is not None
             and r["dev_commit_rate"] >= 0.3 and r["dev_false_adapt"] <= ALPHA]
    if valid:
        valid.sort(key=lambda r: (r["dev_false_adapt"], r.get("dev_regret_kga", 1e9)))
        return valid[0]["variant"], True
    # none valid -> pick lowest dev_false_adapt among commit>=0.3 (closest to passing)
    cand = [r for r in rows if r.get("dev_false_adapt") is not None
            and r.get("dev_commit_rate", 0) >= 0.3]
    if cand:
        cand.sort(key=lambda r: r["dev_false_adapt"])
        return cand[0]["variant"], False
    rows2 = [r for r in rows if r.get("dev_false_adapt") is not None]
    rows2.sort(key=lambda r: r["dev_false_adapt"])
    return (rows2[0]["variant"] if rows2 else None), False

def variant_from_name(name):
    for v, kw in VARIANTS:
        if vname(v, kw) == name:
            return v, kw
    raise ValueError(name)

# ---------------- main ----------------
def main():
    cam = load_camelyon()
    out = {"alpha": ALPHA, "dev_seeds": DEV_SEEDS, "test_seeds": TEST_SEEDS}

    dev_rows = explore_dev(cam)
    out["dev_table"] = dev_rows
    best_name, dev_passes = pick_best(dev_rows)
    out["locked_variant"] = best_name
    out["dev_passes"] = dev_passes
    v, kw = variant_from_name(best_name)

    # TEST: fit on DEV seeds {0,1}, evaluate on TEST seeds {2,3} -- ONCE
    test_best = run_split(cam, DEV_SEEDS, TEST_SEEDS, v, **kw)
    test_base = run_split(cam, DEV_SEEDS, TEST_SEEDS, "gbr_baseline")
    out["camelyon_test_locked"] = test_best
    out["camelyon_test_baseline"] = test_base

    # also: full eps-recal baseline reference over all C(4,2) splits (for context)
    Z, B, a0, aa, sd, comp = arrays(cam)
    out["camelyon_grid_seeds"] = sorted(set(sd.tolist()))

    # CIFAR sanity: apply locked variant, DEV/TEST split, must keep false-adapt<=alpha
    cif = load_cifar()
    cs = sorted(set(r["seed"] for r in cif))
    cdev = cs[:2]; ctest = cs[2:4] if len(cs) >= 4 else cs[2:]
    cif_locked = run_split(cif, cdev, ctest, v, **kw)
    cif_base = run_split(cif, cdev, ctest, "gbr_baseline")
    out["cifar_dev_seeds"] = cdev; out["cifar_test_seeds"] = ctest
    out["cifar_test_locked"] = cif_locked
    out["cifar_test_baseline"] = cif_base

    # CHARACTERIZATION ONLY (not used to choose the lock): does ANY variant pass on TEST?
    # This tells overfit-lock (some variant would pass) vs bias-in-Z (none pass).
    test_all = []
    for vn, kw2 in VARIANTS:
        try:
            mm = run_split(cam, DEV_SEEDS, TEST_SEEDS, vn, **kw2)
            test_all.append({"variant": vname(vn, kw2), **(mm or {})})
        except Exception as e:
            test_all.append({"variant": vname(vn, kw2), "error": str(e)})
    out["camelyon_test_all_variants_CHARACTERIZATION"] = test_all
    any_pass = any(r.get("false_adapt") is not None and r["false_adapt"] <= ALPHA
                   and r.get("commit_rate", 0) >= 0.3 for r in test_all)
    out["any_variant_passes_on_test"] = any_pass

    json.dump(out, open(f"{OUTDIR}/dryrun_results.json", "w"), indent=2)

    # console summary
    print("=== DEV TABLE (Camelyon, within-DEV holdout 0<->1) ===")
    for r in dev_rows:
        if "error" in r:
            print(f"  {r['variant']:34s} ERROR {r['error'][:50]}")
        else:
            fa = r["dev_false_adapt"]; cr = r["dev_commit_rate"]
            print(f"  {r['variant']:34s} FA={fa if fa is None else round(fa,3)}  "
                  f"commit={None if cr is None else round(cr,3)}  "
                  f"regret={None if r['dev_regret_kga'] is None else round(r['dev_regret_kga'],4)}")
    print(f"\nLOCKED: {best_name}  (dev_passes={dev_passes})")
    print("\n=== TEST (Camelyon seeds {2,3}) ===")
    print(" locked  :", test_best)
    print(" baseline:", test_base)
    print("\n=== TEST: ALL variants (characterization only) ===")
    for r in test_all:
        if "error" in r: continue
        fa = r.get("false_adapt"); cr = r.get("commit_rate")
        print(f"  {r['variant']:34s} FA={fa if fa is None else round(fa,3)}  commit={None if cr is None else round(cr,3)}")
    print(f" ANY variant passes on TEST (FA<=0.10 & commit>=0.3)? {any_pass}")
    print("\n=== CIFAR sanity ===")
    print(" locked  :", cif_locked)
    print(" baseline:", cif_base)

if __name__ == "__main__":
    main()
