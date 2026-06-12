"""Reach-table unification (Part 1C) -- consistency validation.

Folds the 1A multi-candidate tau-route and the 1B smooth-drift route into the
ambiguity-reach framework (Def def:reach / Prop thm:unify, Prop prop:reach-table):
both are evaluations of the SAME reach rho(P)=1/2(sup-inf of the benefit interval
I(P)), and the SINGLE certificate "commit sign(c) iff |c|>rho" governs both.
Reuses the 1A and 1B validators as modules (no re-derivation).  CPU only; does
NOT modify the paper or any SAR/GPU harness.

Checks, per family:
  (i)  closed-form reach == brute-force reach over admissible equivalent worlds;
  (ii) the unified margin>reach certificate makes ZERO false certifications when
       the family's structural modulus holds, and the converse (sign flips among
       equivalent worlds) appears exactly when margin < reach.
For the 1A row we ALSO surface the honest caveat that the OBSERVABLE residual tau
bounds only the reach normal to the rank-one variety: a tangential error
correlation gives tau~0 yet a nonzero true reach (the Theorem thm:imp blind spot).
"""

import json
import os
import numpy as np

import val_multicandidate_residual as MC      # 1A
import val_smooth_drift as SD                  # 1B

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "results_reach_unification.json")
SEED = 20260609
RNG = np.random.default_rng(SEED)


# ====================================================================== #
# Family A -- bounded/smooth drift (1B):  c = U-2T_S,  reach = 2 L d W
# ====================================================================== #
def drift_family(L=0.6, n=40000, n_bf=400, n_worlds=60):
    """closed-form reach vs brute-force reach over admissible drifts; certificate."""
    rows = []
    false_cert = 0
    converse_has_both = 0
    converse_cases = 0
    ratios = []
    for _ in range(n_worlds):
        mu_t = RNG.uniform(0.05, 1.2)
        sd_t = 1.0 + 0.3 * mu_t
        xt = SD.make_target_X(mu_t, sd_t, n, RNG)
        U, W = SD.observables(xt)
        TS = float(np.mean(SD.diff(xt) * SD.gS(xt)))      # true T_S
        c = U - 2.0 * TS
        d = SD.w2_gaussian(0.0, 1.0, mu_t, sd_t)
        B = L * d
        reach_closed = 2.0 * B * W
        # brute force the benefit interval over admissible b (||b||_inf <= B)
        deltas = [SD.true_delta(xt, SD.concept_drift(xt, k, B))
                  for k in ("adversarial", "aligned", "zero")]
        for _ in range(n_bf):
            deltas.append(SD.true_delta(xt, RNG.uniform(-B, B, size=xt.shape)))
        lo, hi = min(deltas), max(deltas)
        reach_bf = 0.5 * (hi - lo)
        ratios.append(reach_bf / reach_closed if reach_closed > 1e-9 else 1.0)
        # unified certificate
        committed = abs(c) > reach_closed
        if committed:
            # any admissible world with opposite sign of Delta == false cert
            if any(np.sign(D) == -np.sign(c) and np.sign(D) != 0 for D in deltas):
                false_cert += 1
        else:
            converse_cases += 1
            signs = set(np.sign(D) for D in deltas if np.sign(D) != 0)
            if len(signs) > 1:
                converse_has_both += 1
        rows.append({"mu": mu_t, "c": c, "reach_closed": reach_closed,
                     "reach_bf": reach_bf, "committed": bool(committed)})
    return {
        "reach_ratio_bf_over_closed_mean": float(np.mean(ratios)),
        "reach_ratio_min": float(np.min(ratios)),
        "reach_ratio_max": float(np.max(ratios)),
        "false_cert_count": int(false_cert),
        "converse_both_signs_rate": (converse_has_both / converse_cases
                                     if converse_cases else None),
        "n_worlds": n_worlds,
    }


# ====================================================================== #
# Family B -- multi-candidate (1A):  reach in advantage-space
# ====================================================================== #
def multiview_exact_reach(M=5, beta=0.25, n_D=8000, n_worlds=60):
    """Under EXACT conditional independence (rho=0): the agreement-preserving
    equivalence class collapses to the anchored b -> reach ~ 0 (sampling floor).
    Reproduces the 'reach 0 under moment non-degeneracy' table row."""
    errs = []
    for _ in range(n_worlds):
        b = MC.make_advantages(M, beta, RNG)
        a = (1.0 + b) / 2.0
        C = MC.simulate_agreements(a, 0.0, n_D, RNG)
        bt = MC.minor_estimator(C)
        errs.append(float(np.abs(bt - b).max()))
    return {"mean_reach_exact": float(np.mean(errs)),
            "max_reach_exact": float(np.max(errs))}


def multiview_normal_reach_vs_tau(M=5, beta=0.25, n_D=8000, n_worlds=80):
    """Generic (non-tangential) approximate independence: the recovery half-spread
    (reach normal to the rank-one variety) tracks the OBSERVABLE residual tau."""
    from scipy.stats import spearmanr
    rhos = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    taus, reaches = [], []
    for rho in rhos:
        tt, ee = [], []
        for _ in range(n_worlds):
            b = MC.make_advantages(M, beta, RNG)
            a = (1.0 + b) / 2.0
            C = MC.simulate_agreements(a, rho, n_D, RNG)
            _, tau = MC.rankone_fit_offdiag(C)
            bt = MC.minor_estimator(C)
            tt.append(tau)
            ee.append(float(np.abs(bt - b).max()))
        taus.append(float(np.mean(tt)))
        reaches.append(float(np.mean(ee)))
    sp = float(spearmanr(taus, reaches).statistic)
    return {"rhos": rhos, "tau": taus, "normal_reach": reaches,
            "spearman_tau_reach": sp}


def multiview_tangential_blindspot(M=5, beta=0.25):
    """Honest caveat (Theorem thm:imp): a TANGENTIAL error correlation makes the
    observed agreements EXACTLY rank-one (tau~0) for the WRONG advantage vector.
    True world = b; observed agreements = off(b_alt b_alt^T) with the weakest
    candidate's sign flipped.  tau~0 (looks clean) yet recovered sign is wrong and
    the TRUE misspecification eta > 0 -- so tau does NOT bound this reach."""
    b_true = MC.make_advantages(M, beta, RNG)
    # flip the weakest NON-anchor candidate (candidate 0 is the fixed anchor);
    # use a small opposite-sign value so the implied error-correlation stays
    # within the realizable correctness-covariance bound.
    i0 = 1 + int(np.argmin(np.abs(b_true[1:])))
    b_alt = b_true.copy()
    b_alt[i0] = -np.sign(b_true[i0]) * 0.05
    C_tan = np.outer(b_alt, b_alt)
    np.fill_diagonal(C_tan, 0.0)
    _, tau_tan = MC.rankone_fit_offdiag(C_tan)
    bt = MC.minor_estimator(C_tan)
    off = ~np.eye(M, dtype=bool)
    E = (C_tan - np.outer(b_true, b_true))
    eta_tan = float(np.sqrt((E[off] ** 2).sum()))
    # realizability: |Cov(s_i,s_j)| = |E_ij|/4 must be <= sqrt(Var_i Var_j)
    a = (1.0 + b_true) / 2.0
    var = a * (1.0 - a)
    cov_bound = np.sqrt(np.outer(var, var))
    realizable = bool(np.all(np.abs(E)[off] / 4.0 <= cov_bound[off] + 1e-12))
    sign_wrong = bool(np.sign(bt[i0]) != np.sign(b_true[i0]))
    return {"flipped_candidate": i0, "b_true_i0": float(b_true[i0]),
            "tau_tangential": float(tau_tan), "eta_tangential": eta_tan,
            "error_corr_realizable": realizable,
            "recovered_sign_wrong": sign_wrong}


def main():
    print("=" * 78)
    print("Reach-table unification (Part 1C)  seed =", SEED, " CPU only")
    print("=" * 78)

    drift = drift_family()
    mv_exact = multiview_exact_reach()
    mv_normal = multiview_normal_reach_vs_tau()
    mv_tan = multiview_tangential_blindspot()

    headline = {
        "drift_reach_ratio_mean": drift["reach_ratio_bf_over_closed_mean"],
        "drift_reach_ratio_range": [drift["reach_ratio_min"], drift["reach_ratio_max"]],
        "drift_false_cert_count": drift["false_cert_count"],
        "drift_converse_both_signs_rate": drift["converse_both_signs_rate"],
        "multiview_reach_exact_indep": mv_exact["mean_reach_exact"],
        "multiview_spearman_tau_reach": mv_normal["spearman_tau_reach"],
        "tangential_tau": mv_tan["tau_tangential"],
        "tangential_eta": mv_tan["eta_tangential"],
        "tangential_realizable": mv_tan["error_corr_realizable"],
        "tangential_sign_wrong": mv_tan["recovered_sign_wrong"],
        # ---- PASS flags ----
        "PASS_drift_reach_matches_closed_form":
            bool(0.92 <= drift["reach_ratio_bf_over_closed_mean"] <= 1.08),
        "PASS_drift_certificate_sound":
            bool(drift["false_cert_count"] == 0
                 and (drift["converse_both_signs_rate"] or 0) >= 0.5),
        "PASS_multiview_exact_indep_reach_zero":
            bool(mv_exact["mean_reach_exact"] < 0.05),
        "PASS_multiview_normal_reach_tracks_tau":
            bool(mv_normal["spearman_tau_reach"] > 0.9),
        "PASS_tangential_blindspot_demonstrated":
            bool(mv_tan["tau_tangential"] < 0.02
                 and mv_tan["eta_tangential"] > 0.1
                 and mv_tan["error_corr_realizable"]
                 and mv_tan["recovered_sign_wrong"]),
    }
    headline["ALL_PASS"] = bool(all(v for k, v in headline.items()
                                    if k.startswith("PASS_")))

    results = {
        "description": "Reach-table unification (Part 1C): 1A tau-route and 1B "
                       "drift-route as two evaluations of the same reach rho(P)",
        "seed": SEED,
        "drift_family": drift,
        "multiview_exact": mv_exact,
        "multiview_normal_vs_tau": mv_normal,
        "multiview_tangential_blindspot": mv_tan,
        "headline": headline,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n-- Family A (drift, 1B): reach 2LdW vs brute-force I(P) --")
    print(f"   reach ratio bf/closed: mean={drift['reach_ratio_bf_over_closed_mean']:.3f}"
          f"  range=[{drift['reach_ratio_min']:.3f},{drift['reach_ratio_max']:.3f}]")
    print(f"   unified certificate false-certifications: {drift['false_cert_count']}"
          f"   converse (both signs when |c|<reach): "
          f"{drift['converse_both_signs_rate']}")

    print("\n-- Family B (multi-view, 1A) --")
    print(f"   exact independence reach (should be ~0): "
          f"mean={mv_exact['mean_reach_exact']:.4f} max={mv_exact['max_reach_exact']:.4f}")
    print(f"   normal reach vs observable tau (Spearman): "
          f"{mv_normal['spearman_tau_reach']:.3f}")
    for r, t, e in zip(mv_normal["rhos"], mv_normal["tau"], mv_normal["normal_reach"]):
        print(f"      rho={r:.2f}  tau={t:.4f}  normal_reach={e:.4f}")
    print(f"   TANGENTIAL blind spot (cand {mv_tan['flipped_candidate']}): "
          f"tau={mv_tan['tau_tangential']:.4f} (~0)  "
          f"eta={mv_tan['eta_tangential']:.3f} (>0)  "
          f"realizable={mv_tan['error_corr_realizable']}  "
          f"sign_wrong={mv_tan['recovered_sign_wrong']}")
    print("   => observable tau bounds the NORMAL reach only; the tangential")
    print("      component is the irreducible Thm thm:imp blind spot.")

    print("\n" + "=" * 78)
    print(json.dumps(headline, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print("\nALL_PASS =", headline["ALL_PASS"])
    return 0 if headline["ALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
