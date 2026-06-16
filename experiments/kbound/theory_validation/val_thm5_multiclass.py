"""val_thm5_multiclass.py -- Numerical validation of the MULTICLASS and REGRESSION
extension of Theorem 5 (sign-of-difference on the disagreement region).

This closes the CHARACTERIZATION half of Conjecture 1 (`conj:gen` in
docs/research/kbound/kbound.tex): it extends Theorem `thm:disagree` (the binary
sign-of-difference identity restricted to the disagreement region) to (A) the
multiclass 0/1 case and (B) the squared-error regression case, and validates the
two algebraic identities to numerical precision.

==============================================================================
SETTING (identical conventions to the paper).
    Frozen predictor f0, adapted/gated predictor fa. Both are KNOWN maps of X,
    so the disagreement region
        D = { x : f0(x) != fa(x) }
    is OBSERVABLE (no labels needed to evaluate membership).
    Per-sample benefit (note the sign: f0 first, so Delta>0 == adapting helps):
        delta(x,y) = ell(f0(x), y) - ell(fa(x), y).
    Population benefit  Delta = E_{P_T}[ delta(X,Y) ] = R_T(f0) - R_T(fa).

------------------------------------------------------------------------------
THEOREM A (multiclass, 0/1 loss, Y in {1,...,K}).
    Off D the predictors agree => delta = 0. On D,
        E[delta | D] = p_a - p_0,
        p_a := P_T(fa(X)=Y | X in D),  p_0 := P_T(f0(X)=Y | X in D).
    Hence
        Delta = P_T(D) * (p_a - p_0)   and   sign(Delta) = sign(p_a - p_0).
    Proof.  delta = 1[f0!=Y] - 1[fa!=Y]. Off D both indicators coincide so
    delta=0. On D, E[delta|D] = (1 - p_0) - (1 - p_a) = p_a - p_0. Multiplying by
    P_T(D) (and using that off-D contributes 0) gives Delta = P_T(D)(p_a - p_0).
    Since P_T(D) >= 0 the sign claim follows.  []
    NOTE vs binary: in binary exactly one of f0,fa is correct on D, so
    p_0 = 1 - p_a and p_a - p_0 = 2 p_a - 1 (the paper's `2 a_a^D - 1`). In
    multiclass BOTH can be wrong on D (predicting two different wrong classes),
    so p_0 + p_a <= 1 in general -- yet the identity E[delta|D]=p_a-p_0 STILL
    holds, because it only counts who is correct, not whether the other is wrong.

------------------------------------------------------------------------------
THEOREM B (regression, squared loss, Y in R).
    ell(t, y) = (t - y)^2. Then pointwise
        delta(x,y) = (f0-y)^2 - (fa-y)^2 = (f0 - fa) * (f0 + fa - 2y).
    Off D, f0 = fa => delta = 0. On D,
        E[delta | D] = E[(f0 - fa)(f0 + fa) | D] - 2 E[(f0 - fa) Y | D].
    The FIRST term is fully OBSERVABLE (a function of the known maps f0,fa and
    P_T(X), estimable from unlabeled target). The SECOND term, 2 E[(f0-fa)Y|D],
    is the only label-coupled object; sign(Delta) is identifiable label-free iff
    the sign contribution of E[(f0-fa)Y|D] can be certified without target labels.
        * Under COVARIATE SHIFT (P_S(Y|X)=P_T(Y|X)) it CAN: E[Y|X] transfers from
          source, so an importance-weighted source plug-in
          E_T[(f0-fa)Y | D] = E_T[(f0-fa) m(X) | D],  m(X)=E[Y|X],
          recovers the correct sign (m estimated on labeled source, the (f0-fa)
          and D parts evaluated on target).
        * Under CONCEPT SHIFT (E_T[Y|X] != E_S[Y|X]) it CANNOT: the source
          m_S(X) is the wrong regression function, the plug-in can flip sign, and
          the regime re-enters the unknowable region (ties to Theorem `thm:imp`).

------------------------------------------------------------------------------
HONEST RESIDUAL (the standing assumption, unchanged from the binary case).
    Theorem A reduces sign(Delta) to the ORDINAL accuracy comparison "is fa more
    accurate than f0 on D" (p_a vs p_0); Theorem B reduces it to certifying the
    sign of E[(f0-fa)Y|D]. Both reductions are EXACT and label-free-OBSERVABLE up
    to that one comparison. But the label-free BRACKETING of p_a vs p_0
    (multiclass) / of E[(f0-fa)Y|D] (regression) remains an ASSUMPTION -- a
    reliability/calibration model -- exactly as bracketing a_a^D around 1/2 was an
    assumption in the binary Theorem. So this CLOSES the characterization (the
    sign equals an ordinal accuracy comparison on D), but the label-free
    estimability of that comparison stays the standing assumption. The ONLY
    remaining open piece is the label-free bracketing itself.

------------------------------------------------------------------------------
WHAT THIS SCRIPT CHECKS.
    (i)  MULTICLASS: random K-class problems with random f0, fa. Compute Delta two
         ways -- directly with labels (mean of delta over the sample), and via the
         decomposition P(D)*(p_a - p_0). Confirm EQUAL to numerical precision over
         many trials, and that sign(Delta) == sign(p_a - p_0) on 100% of trials.
    (ii) REGRESSION: confirm the pointwise identity
         delta = (f0-fa)(f0+fa-2y) to machine precision; confirm
         sign(Delta) == sign(E[delta|D]); show a COVARIATE-SHIFT case where an
         importance-weighted SOURCE estimate of E[(f0-fa)Y|D] recovers the correct
         sign of Delta, and a CONCEPT-SHIFT case where the same estimator FAILS
         (flips sign), so the label-free certificate is no longer valid.

Run:  python3 val_thm5_multiclass.py
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# Part (i): MULTICLASS, 0/1 loss  --  Theorem A
# ============================================================================

def zero_one_benefit(f0_pred: np.ndarray, fa_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-sample 0/1 benefit delta = 1[f0!=y] - 1[fa!=y]  (== ell(f0,y)-ell(fa,y))."""
    loss0 = (f0_pred != y).astype(np.float64)
    lossa = (fa_pred != y).astype(np.float64)
    return loss0 - lossa


def run_multiclass_trial(rng: np.random.Generator, K: int, n: int):
    """One synthetic K-class trial with random f0, fa, and a random label law.

    We deliberately make f0 and fa correlated with Y to varying, random degrees
    (so Delta takes both signs across trials), and we let them disagree on a
    nontrivial region so D is non-empty and (in multiclass) both can be wrong.
    """
    # Random true labels.
    y = rng.integers(0, K, size=n)

    # Build f0, fa as noisy copies of y with random, independent accuracies, so
    # that on the disagreement region BOTH may be wrong (the multiclass crux).
    acc0 = rng.uniform(0.05, 0.95)
    acca = rng.uniform(0.05, 0.95)

    def noisy_predictor(acc: float) -> np.ndarray:
        keep = rng.random(n) < acc            # correct on these
        wrong = rng.integers(0, K, size=n)    # a (possibly-coincidentally-correct) guess
        pred = np.where(keep, y, wrong)
        return pred

    f0_pred = noisy_predictor(acc0)
    fa_pred = noisy_predictor(acca)

    # --- Direct Delta (uses labels): mean of per-sample benefit over the sample.
    delta_samples = zero_one_benefit(f0_pred, fa_pred, y)
    Delta_direct = float(delta_samples.mean())

    # --- Decomposition Delta = P(D) * (p_a - p_0).
    D = f0_pred != fa_pred                 # observable disagreement mask
    pD = float(D.mean())
    if D.any():
        p0 = float((f0_pred[D] == y[D]).mean())   # f0 accuracy ON D
        pa = float((fa_pred[D] == y[D]).mean())   # fa accuracy ON D
    else:
        p0 = pa = 0.0
    Delta_decomp = pD * (pa - p0)

    # --- Also verify the off-D mass contributes exactly 0 to Delta (sanity).
    offD_contrib = float(delta_samples[~D].sum()) if (~D).any() else 0.0

    return {
        "K": K,
        "pD": pD,
        "p0": p0,
        "pa": pa,
        "both_wrong_frac_on_D": (
            float(((f0_pred[D] != y[D]) & (fa_pred[D] != y[D])).mean()) if D.any() else 0.0
        ),
        "Delta_direct": Delta_direct,
        "Delta_decomp": Delta_decomp,
        "abs_err": abs(Delta_direct - Delta_decomp),
        "offD_contrib": offD_contrib,
        "sign_direct": int(np.sign(Delta_direct)),
        "sign_decomp": int(np.sign(Delta_decomp)),
        "sign_pa_minus_p0": int(np.sign(pa - p0)),
    }


def validate_multiclass(n_trials: int = 4000, seed: int = 0):
    print("=" * 78)
    print("PART (i)  MULTICLASS 0/1 LOSS  --  Theorem A")
    print("  Check:  Delta_direct (with labels)  ==  P(D) * (p_a - p_0)")
    print("  Check:  sign(Delta) == sign(p_a - p_0)  on every trial")
    print("=" * 78)
    rng = np.random.default_rng(seed)

    max_abs_err = 0.0
    max_offD = 0.0
    sign_matches = 0          # sign(Delta_direct) == sign(p_a - p_0)
    sign_eligible = 0         # trials with Delta != 0 (where sign is defined)
    both_wrong_seen = False
    K_used = set()
    n_with_both_wrong = 0

    for _ in range(n_trials):
        K = int(rng.integers(3, 12))        # >= 3 classes (genuinely multiclass)
        n = int(rng.integers(2000, 8000))
        r = run_multiclass_trial(rng, K, n)
        K_used.add(K)
        max_abs_err = max(max_abs_err, r["abs_err"])
        max_offD = max(max_offD, abs(r["offD_contrib"]))
        if r["both_wrong_frac_on_D"] > 0:
            both_wrong_seen = True
            n_with_both_wrong += 1
        # Sign-match check on the IDENTITY sign(Delta)=sign(p_a-p_0):
        if r["sign_direct"] != 0:
            sign_eligible += 1
            if r["sign_direct"] == r["sign_pa_minus_p0"]:
                sign_matches += 1

    sign_rate = sign_matches / sign_eligible if sign_eligible else float("nan")
    print(f"  trials                          : {n_trials}")
    print(f"  K values exercised              : {sorted(K_used)}")
    print(f"  trials with BOTH-wrong on D     : {n_with_both_wrong}/{n_trials} "
          f"(multiclass-specific regime present: {both_wrong_seen})")
    print(f"  max |Delta_direct - P(D)(pa-p0)|: {max_abs_err:.3e}")
    print(f"  max |off-D contribution to Delta|: {max_offD:.3e}")
    print(f"  sign-match rate sign(Delta)=sign(pa-p0): "
          f"{sign_matches}/{sign_eligible} = {100.0*sign_rate:.4f}%")
    equality_ok = max_abs_err < 1e-9
    sign_ok = (sign_matches == sign_eligible)
    print(f"  --> EQUALITY to numerical precision (<1e-9): {equality_ok}")
    print(f"  --> SIGN MATCH 100%%                        : {sign_ok}")
    return {
        "max_abs_err": max_abs_err,
        "max_offD": max_offD,
        "sign_rate": sign_rate,
        "sign_matches": sign_matches,
        "sign_eligible": sign_eligible,
        "n_with_both_wrong": n_with_both_wrong,
        "equality_ok": equality_ok,
        "sign_ok": sign_ok,
    }


# ============================================================================
# Part (ii): REGRESSION, squared loss  --  Theorem B
# ============================================================================

def squared_benefit(f0: np.ndarray, fa: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-sample squared-loss benefit delta = (f0-y)^2 - (fa-y)^2."""
    return (f0 - y) ** 2 - (fa - y) ** 2


def validate_regression_identity(n_trials: int = 3000, seed: int = 1):
    print()
    print("=" * 78)
    print("PART (ii-a)  REGRESSION SQUARED LOSS  --  Theorem B, pointwise identity")
    print("  Check:  (f0-y)^2 - (fa-y)^2  ==  (f0-fa)(f0+fa-2y)   [machine precision]")
    print("  Check:  sign(Delta) == sign(E[delta|D])")
    print("=" * 78)
    rng = np.random.default_rng(seed)

    max_pointwise_err = 0.0
    max_decomp_err = 0.0
    sign_matches = 0
    sign_eligible = 0

    for _ in range(n_trials):
        n = int(rng.integers(2000, 8000))
        # Arbitrary continuous f0, fa, y (scale/shift randomized).
        scale = rng.uniform(0.2, 5.0)
        y = rng.normal(rng.uniform(-3, 3), scale, size=n)
        f0 = rng.normal(0, scale, size=n) + rng.uniform(-2, 2)
        fa = f0 + rng.normal(0, rng.uniform(0.1, 3.0), size=n)  # disagree a.e.

        delta = squared_benefit(f0, fa, y)
        identity = (f0 - fa) * (f0 + fa - 2.0 * y)
        max_pointwise_err = max(max_pointwise_err, float(np.max(np.abs(delta - identity))))

        Delta_direct = float(delta.mean())

        # Decomposition restricted to D (here D is a.e. the whole space since
        # fa != f0 a.e., but we apply the mask exactly as in the theorem).
        D = f0 != fa
        pD = float(D.mean())
        cond = float(delta[D].mean()) if D.any() else 0.0   # E[delta|D]
        Delta_decomp = pD * cond
        max_decomp_err = max(max_decomp_err, abs(Delta_direct - Delta_decomp))

        if np.sign(Delta_direct) != 0:
            sign_eligible += 1
            if np.sign(Delta_direct) == np.sign(cond):     # sign(Delta)=sign(E[delta|D])
                sign_matches += 1

    print(f"  trials                                  : {n_trials}")
    print(f"  max |delta - (f0-fa)(f0+fa-2y)|         : {max_pointwise_err:.3e}")
    print(f"  max |Delta - P(D)*E[delta|D]|           : {max_decomp_err:.3e}")
    print(f"  sign-match sign(Delta)=sign(E[delta|D]) : "
          f"{sign_matches}/{sign_eligible} = "
          f"{100.0*sign_matches/sign_eligible if sign_eligible else float('nan'):.4f}%")
    identity_ok = max_pointwise_err < 1e-9
    sign_ok = (sign_matches == sign_eligible)
    print(f"  --> POINTWISE IDENTITY (machine precision): {identity_ok}")
    print(f"  --> SIGN MATCH 100%%                       : {sign_ok}")
    return {
        "max_pointwise_err": max_pointwise_err,
        "max_decomp_err": max_decomp_err,
        "sign_matches": sign_matches,
        "sign_eligible": sign_eligible,
        "identity_ok": identity_ok,
        "sign_ok": sign_ok,
    }


def _gaussian_kernel_regress(x_src, y_src, x_eval, bw, chunk=2000):
    """Tiny Nadaraya-Watson 1-D estimate of m(x)=E[Y|X] from SOURCE data.

    Returns m_hat evaluated at x_eval. (1-D X keeps the demo transparent and the
    label-free vs label-coupled distinction crisp.) Chunked-vectorized over the
    eval points to stay memory-light while avoiding a per-point Python loop.
    """
    x_src = np.asarray(x_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    out = np.empty(x_eval.shape[0], dtype=np.float64)
    for s in range(0, x_eval.shape[0], chunk):
        xe = x_eval[s:s + chunk][:, None]            # (c, 1)
        w = np.exp(-0.5 * ((x_src[None, :] - xe) / bw) ** 2)  # (c, n_src)
        sw = w.sum(axis=1)
        num = w @ y_src
        out[s:s + chunk] = np.where(sw > 0, num / np.where(sw > 0, sw, 1.0), 0.0)
    return out


def regression_shift_demo(seed: int = 7):
    """Covariate-shift (label-free certificate WORKS) vs concept-shift (FAILS).

    Setup (1-D for transparency):
      X_S ~ N(0,1) (source),  X_T ~ N(mu_shift, 1) (target covariate shift).
      Source label law:        Y = m_S(X) + noise.
      Frozen f0, adapted fa are two fixed functions of X with f0 != fa.
      Truth on TARGET uses the TARGET label law:
         - covariate shift: m_T = m_S  (same Y|X)  -> source plug-in is correct.
         - concept shift:   m_T != m_S (Y|X changed) -> source plug-in is wrong.

      We estimate E_T[(f0-fa)Y | D] two ways:
         (1) ORACLE / true:  using actual target labels (ground truth sign).
         (2) LABEL-FREE certificate:  plug in m_hat(X)=E_hat[Y|X] learned on
             SOURCE (the only labels available), evaluated on target X in D.
      Theorem B predicts: (2) recovers sign(Delta) under covariate shift and can
      flip it under concept shift.
    """
    print()
    print("=" * 78)
    print("PART (ii-b)  REGRESSION: label-free sign certificate under shift")
    print("  Covariate shift  -> source-plug-in E[(f0-fa)Y|D] recovers sign(Delta)")
    print("  Concept  shift   -> same estimator FLIPS sign (certificate invalid)")
    print("=" * 78)
    rng = np.random.default_rng(seed)

    n_src, n_tgt = 40000, 40000
    mu_shift = 2.0                       # covariate shift in X
    noise_sd = 0.5

    # Source label function m_S(x) and a DIFFERENT target function m_T(x).
    # m_S linear; under concept shift m_T tilts the slope sign in the shifted region.
    def m_S(x):
        return 1.0 * x
    def m_T_concept(x):
        # Same near 0 (where source lives) but opposite trend out at the target
        # support -> source extrapolation has the WRONG sign exactly where target X is.
        return 1.0 * x - 1.6 * x

    # Fixed predictors (known maps), chosen so f0 != fa everywhere and so that
    # (f0 - fa) correlates with x (so E[(f0-fa)Y|D] depends on the Y|X trend).
    def f0_map(x):
        return 0.5 * x + 0.3
    def fa_map(x):
        return 0.5 * x - 0.4 + 0.25 * x   # fa - f0 = 0.25 x - 0.7 (sign varies with x)

    def eval_case(concept: bool):
        x_src = rng.normal(0.0, 1.0, size=n_src)
        x_tgt = rng.normal(mu_shift, 1.0, size=n_tgt)

        # Labels.
        y_src = m_S(x_src) + rng.normal(0, noise_sd, size=n_src)
        m_T = m_T_concept if concept else m_S
        y_tgt = m_T(x_tgt) + rng.normal(0, noise_sd, size=n_tgt)

        f0_t, fa_t = f0_map(x_tgt), fa_map(x_tgt)
        D = f0_t != fa_t                      # observable; here all-True
        diff = (f0_t - fa_t)[D]

        # --- TRUTH: Delta on target using target labels.
        delta_t = squared_benefit(f0_t, fa_t, y_tgt)
        Delta_true = float(delta_t.mean())

        # E_T[(f0-fa)Y|D] term, true (oracle uses target labels):
        EfdY_true = float((diff * y_tgt[D]).mean())
        # First (observable) term E[(f0-fa)(f0+fa)|D]:
        Efsum = float((diff * (f0_t + fa_t)[D]).mean())
        Delta_from_decomp = float(D.mean()) * (Efsum - 2.0 * EfdY_true)

        # --- LABEL-FREE certificate: estimate m(x)=E[Y|X] from SOURCE labels,
        #     evaluate on target X in D. (No target labels used.)
        bw = 0.3
        # subsample source for the NW estimator (speed); still source-only labels
        idx = rng.choice(n_src, size=4000, replace=False)
        m_hat_tgt = _gaussian_kernel_regress(x_src[idx], y_src[idx], x_tgt[D], bw)
        EfdY_hat = float((diff * m_hat_tgt).mean())
        Delta_hat = float(D.mean()) * (Efsum - 2.0 * EfdY_hat)   # observable Efsum + plug-in

        return {
            "Delta_true": Delta_true,
            "Delta_from_decomp": Delta_from_decomp,
            "decomp_err": abs(Delta_true - Delta_from_decomp),
            "EfdY_true": EfdY_true,
            "EfdY_hat": EfdY_hat,
            "Delta_hat": Delta_hat,
            "sign_true": int(np.sign(Delta_true)),
            "sign_hat": int(np.sign(Delta_hat)),
            "cert_correct": int(np.sign(Delta_true)) == int(np.sign(Delta_hat)),
        }

    cov = eval_case(concept=False)
    con = eval_case(concept=True)

    def _report(tag, r):
        print(f"  [{tag}]")
        print(f"     Delta_true (target labels)          : {r['Delta_true']:+.5f}  "
              f"(sign {r['sign_true']:+d})")
        print(f"     Delta via decomposition (oracle term): {r['Delta_from_decomp']:+.5f}  "
              f"(|err| {r['decomp_err']:.2e})")
        print(f"     E_T[(f0-fa)Y|D]  true vs source-plug : {r['EfdY_true']:+.4f}  vs  "
              f"{r['EfdY_hat']:+.4f}")
        print(f"     Delta_hat (LABEL-FREE certificate)   : {r['Delta_hat']:+.5f}  "
              f"(sign {r['sign_hat']:+d})")
        print(f"     --> certificate sign CORRECT?        : {bool(r['cert_correct'])}")

    _report("COVARIATE SHIFT  (Y|X unchanged)", cov)
    _report("CONCEPT  SHIFT   (Y|X changed)  ", con)

    print()
    print("  EXPECTED per Theorem B:")
    print("     covariate shift : certificate CORRECT  (label-free sign identifiable)")
    print("     concept  shift  : certificate WRONG    (re-enters unknowable, thm:imp)")
    cov_ok = bool(cov["cert_correct"])
    con_fails = not bool(con["cert_correct"])
    decomp_ok = max(cov["decomp_err"], con["decomp_err"]) < 1e-9
    print(f"  OBSERVED: covariate-correct={cov_ok}, concept-fails={con_fails}, "
          f"decomp-identity-exact={decomp_ok}")
    return {
        "cov": cov, "con": con,
        "cov_ok": cov_ok, "con_fails": con_fails, "decomp_ok": decomp_ok,
    }


# ============================================================================
def main():
    print("\nVALIDATION: Theorem 5 extension to MULTICLASS + REGRESSION "
          "(closes characterization of Conjecture 1)\n")
    mc = validate_multiclass()
    reg_id = validate_regression_identity()
    reg_shift = regression_shift_demo()

    print()
    print("#" * 78)
    print("SUMMARY")
    print("#" * 78)
    print(f"[A multiclass]  equality (Delta == P(D)(pa-p0)) to <1e-9 : {mc['equality_ok']} "
          f"(max err {mc['max_abs_err']:.2e})")
    print(f"[A multiclass]  sign(Delta)==sign(pa-p0) 100%%           : {mc['sign_ok']} "
          f"({mc['sign_matches']}/{mc['sign_eligible']})")
    print(f"[A multiclass]  trials exhibiting BOTH-wrong-on-D       : {mc['n_with_both_wrong']} "
          f"(genuinely beyond binary)")
    print(f"[B regression]  pointwise identity to machine precision : {reg_id['identity_ok']} "
          f"(max err {reg_id['max_pointwise_err']:.2e})")
    print(f"[B regression]  sign(Delta)==sign(E[delta|D]) 100%%      : {reg_id['sign_ok']} "
          f"({reg_id['sign_matches']}/{reg_id['sign_eligible']})")
    print(f"[B regression]  covariate-shift certificate CORRECT     : {reg_shift['cov_ok']}")
    print(f"[B regression]  concept-shift  certificate FAILS        : {reg_shift['con_fails']}")
    print(f"[B regression]  decomposition identity exact (<1e-9)    : {reg_shift['decomp_ok']}")

    all_ok = (
        mc["equality_ok"] and mc["sign_ok"]
        and reg_id["identity_ok"] and reg_id["sign_ok"]
        and reg_shift["cov_ok"] and reg_shift["con_fails"] and reg_shift["decomp_ok"]
    )
    print()
    print(f"ALL CHECKS PASS: {all_ok}")
    print()

    import os, json
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "results_thm5_multiclass.json")
    with open(out_path, "w") as f:
        json.dump({
            "multiclass": {
                "equality_ok": mc["equality_ok"],
                "max_abs_err": mc["max_abs_err"],
                "sign_ok": mc["sign_ok"],
                "sign_matches": mc["sign_matches"],
                "sign_eligible": mc["sign_eligible"],
                "n_with_both_wrong": mc["n_with_both_wrong"]
            },
            "regression_identity": {
                "identity_ok": reg_id["identity_ok"],
                "max_pointwise_err": reg_id["max_pointwise_err"],
                "sign_ok": reg_id["sign_ok"],
                "sign_matches": reg_id["sign_matches"],
                "sign_eligible": reg_id["sign_eligible"]
            },
            "regression_shift": reg_shift,
            "all_ok": all_ok
        }, f, indent=2)
    print(f"Wrote machine-readable results to {out_path}\n")

    print("HONEST RESIDUAL (the only remaining open piece):")
    print("  The reductions above are EXACT. sign(Delta) equals an ordinal accuracy")
    print("  comparison on D (p_a vs p_0, multiclass) / the sign of E[(f0-fa)Y|D]")
    print("  (regression). The LABEL-FREE BRACKETING of that comparison remains a")
    print("  reliability-model assumption -- exactly as in the binary case. The")
    print("  characterization is CLOSED; the label-free estimability of the ordinal")
    print("  comparison is the standing assumption (multiclass bracketing = open piece).")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
