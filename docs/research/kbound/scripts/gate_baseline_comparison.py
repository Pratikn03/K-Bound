#!/usr/bin/env python3
"""
gate_baseline_comparison.py  --  decision-baseline comparison for K-Bound (reviewer item 1).

Compares six LABEL-FREE adapt/freeze(/abstain) decision rules on the SAME locked per-cell
CIFAR-10-C stress data the KGA runner (cifar_tent_mps_v2.py) already produces:

  1. confidence gate    : adapt iff adaptation RAISED mean confidence   (post_conf > pre_conf)
  2. entropy gate       : adapt iff adaptation LOWERED mean entropy      (entropy_drop > 0)   [Tent's own objective]
  3. drift / KL gate    : freeze iff marginal-prediction KL exceeds a dev-calibrated threshold
  4. ATC-style gate     : adapt iff ATC-predicted adapted acc > predicted frozen acc (isotonic conf->acc)
  5. KGA (no radius)    : adapt iff Bhat > 0   (GBR point estimate, NO conformal margin, NO abstain)
  6. KGA (certificate)  : adapt / freeze / abstain via Bhat +/- eps      (the full method)

The point of the table: only the certificate (6) keeps a low *unconditional* false-adapt rate
(FA_u) while staying near-oracle on regret. The naive confidence/entropy gates false-adapt
because, under collapse, a harmful adaptation can look MORE confident / lower entropy.

===============================================================================
BASELINE PARITY -- READ BEFORE CITING THIS TABLE   (fix-queue item 18 / F3-10)
===============================================================================
The previous version of this docstring said every gate is calibrated
"LEAVE-ONE-TASK-OUT (task = corruption), exactly like KGA, so the comparison is
apples-to-apples".  That sentence was false on BOTH sides:

  gate 1 (confidence)  : NO calibration at all. It is the unfitted sign rule
                         `post_conf > pre_conf`. Zero fitted parameters.
  gate 2 (entropy)     : NO calibration at all. Unfitted sign rule `ent_drop > 0`.
  gate 3 (drift/KL)    : leave-one-corruption-out, 6 folds. One threshold.
  gate 4 (ATC)         : leave-one-corruption-out, 6 folds. One isotonic map.
  gates 5-6 (KGA)      : leave-one-CELL-out -- 431 gradient-boosted fits per cell,
                         a strictly larger calibration budget than any gate above,
                         and NOT leave-one-task-out.

So the published `tab:gates` compares two unfitted sign rules and two one-parameter
task-level rules against a 431-fit cell-level estimator.  Two rows are therefore
added here (`KGA (certificate, LOCO)` and `KGA (no radius, LOCO)`) which run KGA at
the gates' OWN budget -- leave-one-corruption-out.  At equal budget the ATC gate has
LOWER regret than the certificate (0.0041 vs 0.0059 on the committed head-to-head
dumps) but BREAKS the declared budget (FA_u 0.116 > alpha = 0.10).  That is the
honest comparison, and it is still favourable to KGA -- on the safety axis, not the
regret axis.  Say that in the caption instead of claiming parity that does not exist.

WHAT THE RADIUS BUYS (fix-queue item 18 / F3-11).  The radius-free variant MEETS the
declared budget (FA_u 0.038 < alpha = 0.10) at 4.4x lower regret and full coverage,
so the aggregate FA_u is NOT the argument for the radius.  The argument is the
harmful-cell column: 11.7% of harmful cells adapt without the radius (83 false adapts
over 5 seeds) versus 0.0% (0 false adapts) with it.  The `harmful_subset` block below
is that column; make it the argument.

INPUT
  --from-percondition (DEFAULT, fix-queue item 8): read the committed per-condition
        dumps directly.  `--in cifar10c_percell.json` used to be the only path and
        that file EXISTS NOWHERE in the tree, so `REVIEWER_REPRO_PACKET.md:135`'s
        instruction crashed with a TypeError at line 213.  The per-condition dumps
        carry the same fields under different names (`a_adapted` for `aa`), so the
        conversion is mechanical and is done by `rows_from_percondition()`.
  --in <file>         : the legacy rows JSON, if you have one.
  --selftest          : pure synthetic, no data needed.

OUTPUT : <out>.json and <out>.md with, per gate, regret-to-oracle, FA_u (unconditional),
         FA_c (conditional among adapts), coverage, adapt-rate, AND the calibration
         budget each rule actually received -- on the full grid and on the harmful
         subset.

  Real run:   python gate_baseline_comparison.py --from-percondition --seed 1 --out gate_comparison
  Self-test:  python gate_baseline_comparison.py --selftest
"""
import argparse, glob, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kbound_decide import (  # noqa: E402
    conformal_radius, decide as _kb_decide, loo_bhat, radii_loo, records, results_root,
)

# Evidence layout -- MUST match cifar_tent_mps_v2.EVIDENCE_NAMES
(PRE_ENT, PRE_CONF, PRE_PBAL, POST_ENT, POST_CONF, POST_PBAL,
 PBAL_DROP, ENT_DROP, FRAC_HI, MKL, UPD) = range(11)
ALPHA = 0.1

RUNNER_HOOK = r"""
# --- OPTIONAL, and no longer needed: the runner hook that used to be required ---
# (fix-queue item 8) The committed per-condition dumps already carry every field
# this script needs, so `--from-percondition` (the default) reads them directly.
# Keep this only if you want a standalone rows file from a live run:
# import json as _json
# _json.dump(rows, open("cifar10c_percell.json", "w"))
# then:  python gate_baseline_comparison.py --in cifar10c_percell.json
"""

# Ranked search order for the per-condition input (fix-queue item 8).  Every entry
# is a real path in the release; the first that exists wins.
PERCOND_CANDIDATES = [
    "mixed_headtohead_v1/per_condition_cifar10c_tent_primary_kga_seed{seed}.json",
    "stress_grid_multiseed_v1/seed{seed}/per_condition_cifar10c_tent_seed{seed}.json",
]


def rows_from_percondition(seed=0, root=None, patterns=None):
    """Build the gate script's ``rows`` from a committed per-condition dump.

    FIX-QUEUE ITEM 8.  ``--in cifar10c_percell.json`` was the only input path and
    that file exists nowhere in this tree, so the documented invocation in
    ``REVIEWER_REPRO_PACKET.md:135`` crashed (``json.load(open(None))``).  The
    per-condition dumps carry the same quantities under different names --
    ``a_adapted`` for ``aa`` -- so the conversion is mechanical.

    Raises with the list of paths tried, rather than a bare FileNotFoundError.
    """
    root = root or results_root()
    tried = []
    for pat in (patterns or PERCOND_CANDIDATES):
        p = os.path.join(root, pat.format(seed=seed))
        tried.append(p)
        if os.path.exists(p):
            recs = records(p)                      # names the file if it is a placeholder
            rows = [{"Z": r["Z"], "a0": float(r["a0"]), "aa": float(r["a_adapted"]),
                     "condition": r["condition"],
                     "regime": r.get("regime", "unknown")} for r in recs]
            print(f"[gates] {len(rows)} cells <- {p}")
            return rows, p
    raise FileNotFoundError(
        f"No per-condition CIFAR-10-C dump for seed={seed}.\n"
        "Tried, in order:\n  " + "\n  ".join(tried) + "\n"
        "  -> pass --input-root / --seed, or see docs/research/kbound/STORAGE_MANIFEST.json.\n"
        "  -> NOTE: stress_grid_multiseed_v1/seed0/ has no per-condition dump at all\n"
        "     (fix-queue item 8 / F4-7); seed 0 is only available in the head-to-head tree."
    )

# ----------------------------------------------------------------------------- gates
def gate_confidence(Z, B, tasks, a0, aa, alpha):
    return np.where(Z[:, POST_CONF] > Z[:, PRE_CONF], "ADAPT", "FREEZE")

def gate_entropy(Z, B, tasks, a0, aa, alpha):
    return np.where(Z[:, ENT_DROP] > 0, "ADAPT", "FREEZE")

def _drift_tau(mkl_dev, B_dev, alpha):
    """largest KL threshold tau s.t. dev false-adapt (adapt & B<0) <= alpha (max coverage under budget)."""
    best = -np.inf
    for tau in np.sort(np.unique(mkl_dev)):
        adapt = mkl_dev <= tau
        if adapt.sum() == 0:
            continue
        if np.mean(B_dev[adapt] < 0) <= alpha:
            best = tau
    return best

def gate_drift(Z, B, tasks, a0, aa, alpha):
    mkl = Z[:, MKL]; dec = np.empty(len(B), dtype=object)
    for t in np.unique(tasks):
        te = tasks == t; dv = ~te
        tau = _drift_tau(mkl[dv], B[dv], alpha)
        dec[te] = np.where(mkl[te] <= tau, "ADAPT", "FREEZE")
    return dec

def gate_atc(Z, B, tasks, a0, aa, alpha):
    from sklearn.isotonic import IsotonicRegression
    dec = np.empty(len(B), dtype=object)
    for t in np.unique(tasks):
        te = tasks == t; dv = ~te
        conf_dev = np.concatenate([Z[dv, PRE_CONF], Z[dv, POST_CONF]])
        acc_dev = np.concatenate([a0[dv], aa[dv]])
        ir = IsotonicRegression(out_of_bounds="clip").fit(conf_dev, acc_dev)
        dec[te] = np.where(ir.predict(Z[te, POST_CONF]) > ir.predict(Z[te, PRE_CONF]),
                           "ADAPT", "FREEZE")
    return dec

def _gbr():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=250, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=0)


def _bhat_loco(Z, B, tasks):
    """Leave-one-CORRUPTION-out benefit estimate: the gates' own calibration budget.

    Fix-queue item 18 / F3-10.  Six folds, one fit each -- exactly what the
    drift/KL and ATC gates get -- instead of KGA's 431 leave-one-cell-out fits.
    """
    Bhat = np.zeros(len(B))
    for t in np.unique(tasks):
        te = tasks == t
        m = _gbr().fit(Z[~te], B[~te])
        Bhat[te] = m.predict(Z[te])
    return Bhat


def _radii(resid, calibration, alpha):
    """Per-cell radii.  ``loo`` (fix-queue item 4 default) excludes the scored
    cell from its own pool; ``in_pool`` reproduces the archived leaky rule;
    ``in_pool_interp`` additionally reproduces np.quantile's interpolation, which
    is what the published `tab:gates` used."""
    resid = np.asarray(resid, float)
    if calibration == "loo":
        return radii_loo(resid, alpha=alpha)
    if calibration == "in_pool":
        return np.full(resid.size, conformal_radius(resid, alpha), float)
    if calibration == "in_pool_interp":
        return np.full(resid.size, float(np.quantile(resid, 1 - alpha)), float)
    raise ValueError(f"unknown calibration {calibration!r}")


def build_decisions(Z, B, tasks, a0, aa, alpha=ALPHA, calibration="loo"):
    """Return ``{rule_name: (decisions, calibration-budget-string)}``.

    FIX-QUEUE ITEM 18 (F3-10) -- the two ``LOCO`` rows exist so the table has at
    least one apples-to-apples comparison; the four original rows keep their real
    (and unequal) budgets, now stated per row instead of claimed equal in prose.

    FIX-QUEUE ITEM 4 -- the KGA radii are leave-one-out-of-pool by default and
    the rule is the exact-rank quantile.  The old ``_kga_bhat`` computed
    ``float(np.quantile(np.abs(Bhat - B), 1 - ALPHA))`` over all N residuals and
    scored the same N cells, i.e. the KGA arm of this baseline comparison was the
    only arm whose threshold saw the test labels.
    """
    out = {}
    n = len(B)
    out["confidence gate"] = (
        np.asarray(gate_confidence(Z, B, tasks, a0, aa, alpha), dtype=object),
        "NONE -- unfitted sign rule (post_conf > pre_conf), zero fitted parameters")
    out["entropy gate"] = (
        np.asarray(gate_entropy(Z, B, tasks, a0, aa, alpha), dtype=object),
        "NONE -- unfitted sign rule (entropy_drop > 0), zero fitted parameters")
    out["drift/KL gate"] = (
        np.asarray(gate_drift(Z, B, tasks, a0, aa, alpha), dtype=object),
        "leave-one-CORRUPTION-out, 6 folds, 1 threshold")
    out["ATC-style gate"] = (
        np.asarray(gate_atc(Z, B, tasks, a0, aa, alpha), dtype=object),
        "leave-one-CORRUPTION-out, 6 folds, 1 isotonic map")

    Bhat = loo_bhat(Z, B)                              # n-1 fits per cell
    eps = _radii(np.abs(Bhat - B), calibration, alpha)
    out["KGA (no radius)"] = (
        np.asarray(np.where(Bhat > 0, "ADAPT", "FREEZE"), dtype=object),
        f"leave-one-CELL-out GBR ({n - 1} fits per cell), NO radius, NO abstain")
    out["KGA (certificate)"] = (
        np.asarray(_kb_decide(Bhat, eps, alpha=alpha), dtype=object),
        f"leave-one-CELL-out GBR ({n - 1} fits per cell) + exact-rank radius "
        f"[{calibration}], mean eps={float(np.mean(eps)):.5f}")

    Bhat_c = _bhat_loco(Z, B, tasks)                   # 6 fits total
    eps_c = _radii(np.abs(Bhat_c - B), calibration, alpha)
    out["KGA (no radius, LOCO)"] = (
        np.asarray(np.where(Bhat_c > 0, "ADAPT", "FREEZE"), dtype=object),
        "leave-one-CORRUPTION-out GBR (6 folds), NO radius -- gates' own budget")
    out["KGA (certificate, LOCO)"] = (
        np.asarray(_kb_decide(Bhat_c, eps_c, alpha=alpha), dtype=object),
        f"leave-one-CORRUPTION-out GBR (6 folds) + exact-rank radius [{calibration}], "
        f"mean eps={float(np.mean(eps_c)):.5f} -- SAME budget as gates 3-4")
    return out


# Legacy shim: the old module-level GATES list, so anything importing it still works.
GATES = [("confidence gate", gate_confidence), ("entropy gate", gate_entropy),
         ("drift/KL gate", gate_drift), ("ATC-style gate", gate_atc)]

RULE_ORDER = ["confidence gate", "entropy gate", "drift/KL gate", "ATC-style gate",
              "KGA (no radius)", "KGA (certificate)",
              "KGA (no radius, LOCO)", "KGA (certificate, LOCO)"]

# ----------------------------------------------------------------------------- scoring
def score(dec, a0, aa, B, idx=None):
    """FIX-QUEUE ITEM 28 -- one definition of false-adapt, both rates named.

    ``FA_u`` is now the marginal rate with the WEAK inequality,
    ``Pr[ADAPT and B <= 0]`` -- the quantity ``thm:certificate`` bounds and what
    ``_locked_analysis_script.py:43`` computes.  The strict variant this function
    used to return is kept as ``FA_u_strict`` because the published `tab:gates`
    was computed with it; the two differ wherever a cell has ``B`` exactly 0.
    """
    dec = np.asarray(dec, dtype=object)
    if idx is not None:
        dec, a0, aa, B = dec[idx], a0[idx], aa[idx], B[idx]
    adapt = dec == "ADAPT"
    realized = np.where(adapt, aa, a0)             # abstain / freeze -> keep frozen
    oracle = np.maximum(a0, aa)
    n_adapt = int(adapt.sum())
    n_fa = int(np.sum(adapt & (B <= 0)))
    return {
        "n": int(len(B)),
        "regret": float((oracle - realized).mean()),
        "FA_u": float(n_fa / len(B)) if len(B) else None,              # P(ADAPT and B<=0)
        "FA_c": float(n_fa / n_adapt) if n_adapt else None,            # P(B<=0 | ADAPT)
        "n_adapt": n_adapt,
        "n_false_adapt": n_fa,
        # DEPRECATED strict variants: what the published table used.
        "FA_u_strict": float(np.mean(adapt & (B < 0))),
        "FA_c_strict": float(np.mean(B[adapt] < 0)) if n_adapt else None,
        "coverage": float(np.mean(dec != "ABSTAIN")),
        "adapt_rate": float(adapt.mean()),
        "mean_acc": float(realized.mean()),
    }

def run_gate_comparison(rows, alpha=ALPHA, calibration="loo"):
    Z = np.array([r["Z"] for r in rows], float)
    a0 = np.array([r["a0"] for r in rows], float)
    aa = np.array([r["aa"] for r in rows], float)
    B = aa - a0
    tasks = np.array([r["condition"].split("|")[0] for r in rows])
    harmful = B < 0
    out = {"n": len(B), "n_harmful": int(harmful.sum()), "alpha": alpha,
           "calibration": calibration,
           "false_adapt_definition":
               "FA_u = Pr[ADAPT and B <= 0] (marginal, weak); FA_c = Pr[B <= 0 | ADAPT]. "
               "FA_*_strict use B < 0 and reproduce the published tab:gates.",
           "gates": {}}
    dec_by_rule = build_decisions(Z, B, tasks, a0, aa, alpha=alpha, calibration=calibration)
    for name in RULE_ORDER:
        dec, budget = dec_by_rule[name]
        out["gates"][name] = {"calibration_budget": budget,
                              "all": score(dec, a0, aa, B),
                              "harmful_subset": score(dec, a0, aa, B, idx=harmful)}
    # what the radius buys (fix-queue item 18 / F3-11)
    nr = out["gates"]["KGA (no radius)"]
    ce = out["gates"]["KGA (certificate)"]
    out["radius_value"] = {
        "no_radius_meets_budget": bool(nr["all"]["FA_u"] <= alpha),
        "no_radius_FA_u": nr["all"]["FA_u"],
        "certificate_FA_u": ce["all"]["FA_u"],
        "regret_ratio_certificate_over_no_radius":
            float(ce["all"]["regret"] / nr["all"]["regret"]) if nr["all"]["regret"] > 0 else None,
        "harmful_cell_adapt_rate_no_radius": nr["harmful_subset"]["adapt_rate"],
        "harmful_cell_adapt_rate_certificate": ce["harmful_subset"]["adapt_rate"],
        "n_false_adapt_no_radius": nr["all"]["n_false_adapt"],
        "n_false_adapt_certificate": ce["all"]["n_false_adapt"],
        "argument":
            "The aggregate FA_u is NOT the argument for the radius -- the radius-free "
            "variant also meets the declared budget, at lower regret and full coverage. "
            "The argument is the harmful-cell column: harmful cells adapted without the "
            "radius, none with it.",
    }
    return out

def to_markdown(res):
    L = [f"# Decision-gate comparison (CIFAR-10-C stress; n={res['n']}, harmful={res['n_harmful']}, "
         f"alpha={res['alpha']}, calibration={res.get('calibration', 'loo')})", "",
         "Lower regret and lower FA_u are better.  **Read the last column before comparing "
         "rows**: the rules do NOT receive equal calibration (fix-queue item 18 / F3-10). "
         "FA_u = Pr[ADAPT and B <= 0]; FA_c = Pr[B <= 0 | ADAPT].", "",
         "| Decision rule | regret | FA_u | FA_c | coverage | adapt-rate | calibration budget |",
         "|---|---:|---:|---:|---:|---:|---|"]
    def _f(v, w=3):
        return "n/a" if v is None else f"{v:.{w}f}"
    for name in RULE_ORDER:
        g = res["gates"][name]["all"]
        L.append(f"| {name} | {_f(g['regret'], 4)} | {_f(g['FA_u'])} | {_f(g['FA_c'])} | "
                 f"{_f(g['coverage'], 2)} | {_f(g['adapt_rate'], 2)} | "
                 f"{res['gates'][name]['calibration_budget']} |")
    L += ["", "## On the harmful subset only (where naive gates fail)", "",
          "| Decision rule | regret | FA_u | FA_c | adapt-rate |", "|---|---:|---:|---:|---:|"]
    for name in RULE_ORDER:
        g = res["gates"][name]["harmful_subset"]
        L.append(f"| {name} | {_f(g['regret'], 4)} | {_f(g['FA_u'])} | {_f(g['FA_c'])} | "
                 f"{_f(g['adapt_rate'], 2)} |")
    rv = res.get("radius_value")
    if rv:
        ratio = rv["regret_ratio_certificate_over_no_radius"]
        ratio_s = "n/a" if ratio is None else f"{ratio:.2f}x"
        L += ["", "## What the radius buys (fix-queue item 18 / F3-11)", "",
              f"- radius-free FA_u = {_f(rv['no_radius_FA_u'])} "
              f"(meets the declared budget: {rv['no_radius_meets_budget']}); "
              f"certificate FA_u = {_f(rv['certificate_FA_u'])}",
              f"- certificate regret is {ratio_s} the radius-free variant's",
              f"- harmful-cell adapt rate: {_f(rv['harmful_cell_adapt_rate_no_radius'])} without "
              f"the radius ({rv['n_false_adapt_no_radius']} false adapts) -> "
              f"{_f(rv['harmful_cell_adapt_rate_certificate'])} with it "
              f"({rv['n_false_adapt_certificate']} false adapts)",
              "", rv["argument"]]
    return "\n".join(L)

# ----------------------------------------------------------------------------- self-test
def _synthetic_rows(n=300, seed=0):
    """Honest synthetic stress grid: harmful cells COLLAPSE (look confident / low entropy),
    so confidence/entropy gates false-adapt; only frac_highconf + marginal_KL reveal the harm,
    which the GBR (KGA) learns. This reproduces the qualitative failure the paper describes."""
    rng = np.random.default_rng(seed)
    corrs = ["gaussian_noise", "shot_noise", "motion_blur", "fog", "contrast", "jpeg"]
    rows = []
    for _ in range(n):
        u = rng.random()
        regime = "helpful" if u < 0.40 else ("harmful" if u < 0.75 else "marginal")
        if regime == "helpful":
            B = rng.uniform(0.02, 0.20)
        elif regime == "harmful":
            B = -rng.uniform(0.02, 0.20)
        else:
            B = rng.uniform(-0.02, 0.02)
        a0 = float(rng.uniform(0.40, 0.80)); aa = float(np.clip(a0 + B, 0, 1))
        pre_ent = rng.uniform(0.8, 1.4); pre_conf = rng.uniform(0.55, 0.75)
        # BOTH helpful and harmful look "better" on entropy/confidence (the confound):
        ent_drop = rng.uniform(0.05, 0.35); conf_rise = rng.uniform(0.03, 0.20)
        post_ent = pre_ent - ent_drop; post_conf = min(0.999, pre_conf + conf_rise)
        if regime == "harmful":                 # collapse signature, only visible in these two:
            frac_hi = rng.uniform(0.75, 0.99); mkl = rng.uniform(0.30, 1.20)
        elif regime == "helpful":
            frac_hi = rng.uniform(0.10, 0.45); mkl = rng.uniform(0.00, 0.12)
        else:
            frac_hi = rng.uniform(0.30, 0.70); mkl = rng.uniform(0.05, 0.40)
        Z = [pre_ent, pre_conf, 0.9, post_ent, post_conf, 0.85,
             0.05, ent_drop, frac_hi, mkl, rng.uniform(0.1, 2.0)]
        rows.append({"condition": f"{rng.choice(corrs)}|s5|x|iid|mild|r0",
                     "Z": Z, "a0": a0, "aa": aa, "regime": regime})
    return rows

def _selftest():
    rows = _synthetic_rows()
    res = run_gate_comparison(rows)
    md = to_markdown(res)
    print(md)
    g = res["gates"]
    cert = g["KGA (certificate)"]["all"]; conf = g["confidence gate"]["all"]; ent = g["entropy gate"]["all"]
    # sanity assertions: the certificate must dominate the naive gates on FA_u and not lose on regret
    assert cert["FA_u"] <= ALPHA + 1e-9, f"cert FA_u {cert['FA_u']} should be <= alpha"
    assert cert["FA_u"] < conf["FA_u"] and cert["FA_u"] < ent["FA_u"], "cert should false-adapt less than naive gates"
    assert cert["regret"] <= conf["regret"] + 1e-9, "cert regret should not exceed confidence gate"
    print("\n[selftest] PASS: certificate keeps FA_u<=alpha and beats naive gates on false-adapt.")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-percondition", action="store_true", default=None,
                    help="read a committed per-condition dump (DEFAULT unless --in "
                         "or --selftest is given). fix-queue item 8.")
    ap.add_argument("--seed", type=int, default=0,
                    help="which seed's per-condition dump to read (default 0)")
    ap.add_argument("--input-root", default=None,
                    help="results tree holding the dumps (default: repo "
                         "experiments/kbound/results, or $KBOUND_RESULTS_ROOT)")
    ap.add_argument("--in", dest="inp",
                    help="LEGACY per-cell rows JSON. The path the docs used to name "
                         "(cifar10c_percell.json) exists nowhere in this tree.")
    ap.add_argument("--calibration", choices=["loo", "in_pool", "in_pool_interp"],
                    default="loo",
                    help="radius pool. loo (default) excludes the scored cell "
                         "(fix-queue item 4); in_pool_interp reproduces the published "
                         "tab:gates numbers.")
    ap.add_argument("--out", default="gate_comparison")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return
    if a.inp:
        if not os.path.exists(a.inp):
            raise SystemExit(
                f"--in {a.inp}: no such file.\n"
                "  -> the legacy rows JSON is not in this release. Use "
                "--from-percondition (the default), which reads the committed "
                "per-condition dumps directly.")
        rows = json.load(open(a.inp))
        src = a.inp
    else:
        rows, src = rows_from_percondition(seed=a.seed, root=a.input_root)
    res = run_gate_comparison(rows, alpha=a.alpha, calibration=a.calibration)
    res["source_artifact"] = src
    json.dump(res, open(a.out + ".json", "w"), indent=2)
    open(a.out + ".md", "w").write(to_markdown(res))
    print(to_markdown(res))
    print(f"\nwrote {a.out}.json and {a.out}.md")

if __name__ == "__main__":
    main()
