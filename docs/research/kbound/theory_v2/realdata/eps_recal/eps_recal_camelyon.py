#!/usr/bin/env python3
"""eps_recal_camelyon.py
=======================
epsilon-recalibration of the single-candidate conformal certificate (Route 1)
on the WILDS Camelyon17 composition-stress grid, HELD OUT BY SEED.

WHY (exchangeability rationale, not a hack)
-------------------------------------------
The Route-1 certificate is  Delta_hat +/- epsilon  where epsilon is a split-
conformal radius:  epsilon = quantile(|Delta_hat - Delta|, 1 - alpha).  Its
coverage guarantee (Thm 2) requires the calibration residuals and the test
residual to be EXCHANGEABLE.  The deployed operating point used an epsilon
calibrated on the SYNTHETIC composition grids; transplanted onto Camelyon17 it
violates that exchangeability premise (different residual law), so the
certificate is no longer level-alpha valid there.  RE-ESTIMATING epsilon from a
Camelyon17 calibration split (held out by seed) RESTORES the premise: CAL and
TEST residuals are then drawn from the same Camelyon17 law, exchangeable across
the seed partition.  tau* is left ALONE (we do not touch the CEI gate).

DATA
----
experiments/kbound/results/wilds_kbound_debug_mps/result_73add410.json
  records[]: per (seed, condition, candidate) cell with
     Z  : 10-dim label-free evidence (entropy/conf statistics of the adapt batch)
     a0 : f0 balanced accuracy,  aa : adapted balanced accuracy
     B  : TRUE benefit Delta = aa - a0  (>0 => adapt helps)
     seed, candidate, domain, comp, regime, aggr
  72 conditions x 6 TTA candidates = 432 cells; seeds {0,1,2,3}.

CERTIFICATE (canonical, mirrors run_wilds_camelyon17.decide_kga)
  Delta_hat = GradientBoostingRegressor(Z -> B), ne=250, depth=2, lr=0.05, sub=0.8
  decision: ADAPT if Dhat-eps>0 ; FREEZE if Dhat+eps<0 ; else ABSTAIN.
  Sanity: pooled cross-fit of this estimator reproduces the paper's
  detectability harm-AUC(-Bhat)=0.912 and certificate_eps=0.0598.

PROCEDURE (STEP 2)
  alpha = 0.10 FIXED (never tuned).
  Splits BY SEED: all (2 CAL + 2 TEST) seed assignments => C(4,2) = 6 splits.
  (b) on CAL cells only: fit the GBR benefit model on CAL, predict on CAL,
      eps = quantile(|Delta_hat - B|, 1 - alpha) over CAL residuals.
  (c) freeze eps AND the CAL-fitted model; predict Delta_hat on TEST cells;
      apply ADAPT/FREEZE/ABSTAIN.  abstain => keep f0.
  (d) per TEST split: KGA regret-to-oracle, always-adapt regret, always-freeze
      regret, false-adapt rate (ADAPT & B<=0), false-freeze rate (FREEZE & B>0),
      coverage (commit fraction), eps.
  (e) aggregate over the 6 splits: mean +/- 95% CI (t over splits); eps
      stability (range / CV across the CAL splits).

VERDICT (pre-stated, STEP 3)
  WIN          = KGA regret < BOTH trivial policies on TEST with CIs clear of
                 both, AND false-adapt <= alpha, AND eps stable.
  PRECISE NEG  = false-adapt > alpha when eps shrinks enough to commit, OR
                 unstable eps, OR CIs overlap a trivial policy => "harm is
                 detectable (AUC 0.91) but not certifiable at level alpha at
                 this sample size."
"""
import json, itertools, math, os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = "/sessions/peaceful-blissful-ptolemy/mnt/uav/AutoML_Flagship_V8/experiments/kbound/results/wilds_kbound_debug_mps/result_73add410.json"
if not os.path.exists(RESULT):  # host path fallback
    RESULT = "/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/results/wilds_kbound_debug_mps/result_73add410.json"

ALPHA = 0.10                 # FIXED. never tuned.
GBR_KW = dict(n_estimators=250, max_depth=2, learning_rate=0.05,
              subsample=0.8, random_state=0)

# ---------------------------------------------------------------- load ----
def load_cells():
    d = json.load(open(RESULT))
    recs = d["records"]
    Z = np.array([r["Z"] for r in recs], float)
    a0 = np.array([r["a0"] for r in recs], float)
    aa = np.array([r["aa"] for r in recs], float)
    B = np.array([r["B"] for r in recs], float)
    seed = np.array([r["seed"] for r in recs], int)
    cand = np.array([r["candidate"] for r in recs])
    # consistency: B == aa - a0
    assert np.max(np.abs(B - (aa - a0))) < 1e-9
    return d, Z, a0, aa, B, seed, cand

# ----------------------------------------------------- sanity: AUC 0.91 ----
def auc_neg(score, label):
    pos = score[label == 1]; neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1
    U = r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2
    return float(U/(len(pos)*len(neg)))

def sanity_detectability(Z, B):
    """5-fold pooled cross-fit Delta_hat; reproduce harm-AUC(-Bhat) & pooled eps."""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    Bhat = np.zeros(len(B))
    for tr, te in kf.split(Z):
        m = GradientBoostingRegressor(**GBR_KW); m.fit(Z[tr], B[tr])
        Bhat[te] = m.predict(Z[te])
    lab = (B < 0).astype(int)
    return {"reproduced_harm_AUC_negBhat": auc_neg(-Bhat, lab),
            "reproduced_pooled_eps": float(np.quantile(np.abs(Bhat - B), 1 - ALPHA)),
            "n_harmful_B<0": int(lab.sum()), "n_cells": int(len(B))}

# --------------------------------------------------- policy bookkeeping ----
def policy_block(dec, a0, aa, B, eps):
    adapt = dec == "ADAPT"; freeze = dec == "FREEZE"
    kga = np.where(adapt, aa, a0)            # ADAPT->fa, FREEZE/ABSTAIN->f0
    oracle = np.maximum(a0, aa)
    reg_kga = float((oracle - kga).mean())
    reg_aa = float((oracle - aa).mean())
    reg_af = float((oracle - a0).mean())
    return {
        "n_test": int(len(B)),
        "eps": float(eps),
        "decision_counts": {d: int((dec == d).sum()) for d in ["ADAPT", "FREEZE", "ABSTAIN"]},
        "coverage_commit_frac": float(np.mean(dec != "ABSTAIN")),
        "regret_KGA": reg_kga,
        "regret_always_adapt": reg_aa,
        "regret_always_freeze": reg_af,
        "false_adapt_rate": (float(np.mean(B[adapt] <= 0)) if adapt.any() else None),  # ADAPT & B<=0
        "false_freeze_rate": (float(np.mean(B[freeze] > 0)) if freeze.any() else None), # FREEZE & B>0
        "n_adapt": int(adapt.sum()), "n_freeze": int(freeze.sum()),
        "KGA_beats_both": bool(reg_kga < reg_aa - 1e-12 and reg_kga < reg_af - 1e-12),
    }

# -------------------------------------------------- seed-split recal -------
def run_seed_splits(Z, a0, aa, B, seed):
    seeds = sorted(set(seed.tolist()))
    splits = []
    for cal in itertools.combinations(seeds, len(seeds)//2):   # 2 CAL of 4
        cal = set(cal); test = set(seeds) - cal
        cal_m = np.isin(seed, list(cal)); test_m = np.isin(seed, list(test))
        # (b) fit benefit model on CAL, residual quantile -> eps
        m = GradientBoostingRegressor(**GBR_KW)
        m.fit(Z[cal_m], B[cal_m])
        Bhat_cal = m.predict(Z[cal_m])
        eps = float(np.quantile(np.abs(Bhat_cal - B[cal_m]), 1 - ALPHA))
        # (c) freeze eps + model; apply on TEST
        Bhat_te = m.predict(Z[test_m])
        dec = np.where(Bhat_te - eps > 0, "ADAPT",
                       np.where(Bhat_te + eps < 0, "FREEZE", "ABSTAIN"))
        pb = policy_block(dec, a0[test_m], aa[test_m], B[test_m], eps)
        pb["cal_seeds"] = sorted(cal); pb["test_seeds"] = sorted(test)
        pb["n_cal"] = int(cal_m.sum())
        pb["cal_residual_eps"] = eps
        pb["test_base_rate_harmful_B<=0"] = float(np.mean(B[test_m] <= 0))
        splits.append(pb)
    return splits

# ----------------------------------------------------------- aggregate -----
def t_crit(df, conf=0.95):
    # two-sided t critical via inverse-CDF approx (no scipy). df<=0 -> nan.
    # Use a small lookup for the dfs we need (5 splits => df=5).
    table = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,
             8:2.306,9:2.262,10:2.228,15:2.131,20:2.086,29:2.045}
    if df in table: return table[df]
    keys = sorted(table);
    for k in keys:
        if df <= k: return table[k]
    return 1.96

def agg(vals):
    v = np.array([x for x in vals if x is not None], float)
    n = len(v)
    if n == 0: return {"mean": None, "ci95": None, "n": 0}
    mean = float(v.mean())
    if n == 1: return {"mean": mean, "ci95": [mean, mean], "n": 1, "sd": 0.0}
    sd = float(v.std(ddof=1)); se = sd/math.sqrt(n)
    h = t_crit(n-1)*se
    return {"mean": mean, "ci95": [mean-h, mean+h], "n": n, "sd": sd,
            "min": float(v.min()), "max": float(v.max())}

def aggregate(splits):
    keys = ["regret_KGA", "regret_always_adapt", "regret_always_freeze",
            "false_adapt_rate", "false_freeze_rate", "coverage_commit_frac", "eps"]
    out = {k: agg([s[k] for s in splits]) for k in keys}
    eps_vals = np.array([s["eps"] for s in splits], float)
    out["eps_stability"] = {
        "values": eps_vals.tolist(),
        "mean": float(eps_vals.mean()),
        "min": float(eps_vals.min()), "max": float(eps_vals.max()),
        "range": float(eps_vals.max() - eps_vals.min()),
        "cv": float(eps_vals.std(ddof=1)/eps_vals.mean()) if eps_vals.mean() else None,
    }
    # commit-conditional false-adapt: pool TEST adapts across splits is not valid
    # (overlapping seeds); we report per-split and the mean above.
    return out

# ----------------------------------------------------------- verdict -------
def verdict(ag):
    rk = ag["regret_KGA"]; ra = ag["regret_always_adapt"]; rf = ag["regret_always_freeze"]
    fa = ag["false_adapt_rate"]
    eps_cv = ag["eps_stability"]["cv"]
    eps_stable = (eps_cv is not None and eps_cv <= 0.25)   # <=25% CV across CAL splits
    # CI clear of BOTH trivials: KGA upper CI < each trivial's lower CI
    clear_adapt = (rk["ci95"][1] < ra["ci95"][0]) if ra["mean"] is not None else False
    clear_freeze = (rk["ci95"][1] < rf["ci95"][0]) if rf["mean"] is not None else False
    fa_ok = (fa["mean"] is None) or (fa["mean"] <= ALPHA)
    commits = ag["coverage_commit_frac"]["mean"] is not None and ag["coverage_commit_frac"]["mean"] > 0
    win = clear_adapt and clear_freeze and fa_ok and eps_stable and commits
    return {
        "WIN": bool(win),
        "kga_ci_clear_of_always_adapt": bool(clear_adapt),
        "kga_ci_clear_of_always_freeze": bool(clear_freeze),
        "false_adapt_le_alpha": bool(fa_ok),
        "eps_stable_cv_le_0.25": bool(eps_stable),
        "commits_at_all": bool(commits),
        "label": ("WIN" if win else "PRECISE_NEGATIVE"),
        "precise_negative_reading":
            ("harm is detectable (pooled cross-fit AUC ~0.91) but NOT certifiable "
             "at level alpha=0.10 at this debug-scale sample size: " +
             ("certificate abstains (coverage~0) so no committed win; "
              if not commits else "") +
             ("false-adapt exceeds alpha once it commits; " if not fa_ok else "") +
             ("eps unstable across seed splits; " if not eps_stable else "") +
             ("KGA-regret CI overlaps a trivial policy; " if not (clear_adapt and clear_freeze) else "")
             ).strip(),
    }

# ------------------------------------- robustness: per-candidate splits ----
def run_per_candidate(Z, a0, aa, B, seed, cand):
    res = {}
    for c in sorted(set(cand.tolist())):
        m = cand == c
        sp = run_seed_splits(Z[m], a0[m], aa[m], B[m], seed[m])
        ag = aggregate(sp)
        res[c] = {"aggregate": {k: ag[k] for k in
                  ["regret_KGA","regret_always_adapt","regret_always_freeze",
                   "false_adapt_rate","coverage_commit_frac","eps"]},
                  "eps_stability": ag["eps_stability"]}
    return res

def main():
    d, Z, a0, aa, B, seed, cand = load_cells()
    sanity = sanity_detectability(Z, B)
    splits = run_seed_splits(Z, a0, aa, B, seed)        # POOLED over 6 candidates
    ag = aggregate(splits)
    vd = verdict(ag)
    per_cand = run_per_candidate(Z, a0, aa, B, seed, cand)

    out = {
        "data_file": RESULT,
        "schema": d.get("schema"),
        "config_sha8": d.get("config_sha8"),
        "alpha_FIXED": ALPHA,
        "estimator": "GradientBoostingRegressor " + str(GBR_KW),
        "tau_star_left_alone": d["config"].get("tau_star"),
        "split_design": "by-seed; C(4,2)=6 assignments of 2 CAL + 2 TEST seeds; "
                        "estimator fit on CAL, eps=quantile(|Dhat-B|,0.9) on CAL, "
                        "frozen and applied on TEST. cells pooled over 6 TTA candidates.",
        "fallback_used": False,
        "synthetic_calibrated_eps_reference": {
            "per_candidate_eps_conformal_reported": {
                k: d["routing_a_single_candidate"][k]["kga"]["eps_conformal"]
                for k in d["routing_a_single_candidate"]},
            "detectability_pooled_certificate_eps_reported": d["detectability"]["certificate_eps"],
            "detectability_harm_AUC_negBhat_reported": d["detectability"]["certificate_harm_AUC_negBhat"],
        },
        "sanity_reproduction": sanity,
        "per_split": splits,
        "aggregate_over_splits": ag,
        "per_candidate_robustness": per_cand,
        "verdict": vd,
    }
    op = os.path.join(HERE, "eps_recal_results.json")
    json.dump(out, open(op, "w"), indent=2)
    # console summary
    print("=== SANITY (reproduce paper detectability) ===")
    print("  harm-AUC(-Bhat): reported %.4f  reproduced %.4f"
          % (d["detectability"]["certificate_harm_AUC_negBhat"], sanity["reproduced_harm_AUC_negBhat"]))
    print("  pooled cert eps: reported %.4f  reproduced %.4f"
          % (d["detectability"]["certificate_eps"], sanity["reproduced_pooled_eps"]))
    print("\n=== eps stability across 6 CAL seed-splits ===")
    print("  eps:", ["%.4f"%e for e in ag["eps_stability"]["values"]],
          "range=%.4f cv=%.3f" % (ag["eps_stability"]["range"], ag["eps_stability"]["cv"]))
    def line(name, k):
        a = ag[k]
        if a["mean"] is None: print("  %-22s n=0 (never triggered)" % name); return
        print("  %-22s mean=%.4f  95%%CI=[%.4f, %.4f]" % (name, a["mean"], a["ci95"][0], a["ci95"][1]))
    print("\n=== aggregate over TEST splits (mean +/- 95%% CI) ===")
    line("regret_KGA", "regret_KGA")
    line("regret_always_adapt", "regret_always_adapt")
    line("regret_always_freeze", "regret_always_freeze")
    line("false_adapt_rate", "false_adapt_rate")
    line("coverage_commit_frac", "coverage_commit_frac")
    print("\n=== VERDICT ===")
    print("  ", vd["label"])
    for k in ["kga_ci_clear_of_always_adapt","kga_ci_clear_of_always_freeze",
              "false_adapt_le_alpha","eps_stable_cv_le_0.25","commits_at_all"]:
        print("    %-32s %s" % (k, vd[k]))
    if vd["label"] != "WIN":
        print("   reading:", vd["precise_negative_reading"])
    print("\nwrote", op)
    return out

if __name__ == "__main__":
    main()
