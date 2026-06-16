#!/usr/bin/env python3
"""bias_variance_diag/diag.py  — Protocol B "Path 1" diagnostic.

Decide whether the conformal certificate radius (eps) on Camelyon17 is
VARIANCE-limited (shrinks ~1/sqrt(n_eval); a fair re-run would help) or
BIAS-limited (dominated by B_hat(Z) model error / calibration-drift bias; more
eval samples cannot help). This explains WHY Protocol B's 1/sqrt(n) prediction
for n_eval=1024 failed.

DECOMPOSITION
  eps-recal residual: r_i = |B_hat(Z_i) - B_i|, alpha=0.10, eps = 0.9-quantile|r|.
  B_hat = GradientBoostingRegressor(Z->B), SAME config as eps_recal_camelyon.py.
  Total residual variance = Var(model error of B_hat) + Var(measurement noise B_i).
  Measurement floor (binomial, per record, n=eval count):
     sigma_meas^2(n) = a0(1-a0)/n + aa(1-aa)/n
     eps_meas(n) = 0.9-quantile of half-normal(sigma_meas) = 1.6449*sigma_meas.
  frac_meas = mean(sigma_meas^2) / Var(r).  ~1 => variance-limited; ~0 => bias.

Reuses the exact GBR_KW from theory_v2/realdata/eps_recal/eps_recal_camelyon.py.
"""
import json, os, math
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.10
HN_Q90 = 1.6448536269514722          # 0.9-quantile of half-normal / sigma
GBR_KW = dict(n_estimators=250, max_depth=2, learning_rate=0.05,
              subsample=0.8, random_state=0)   # EXACT, from eps_recal_camelyon.py

ROOT = "/sessions/peaceful-blissful-ptolemy/mnt/uav/AutoML_Flagship_V8"
if not os.path.isdir(ROOT):
    ROOT = "/Volumes/T9/uav/AutoML_Flagship_V8"
F256 = ROOT + "/experiments/kbound/results/wilds_kbound_debug_mps/result_73add410.json"
F1024 = ROOT + "/experiments/kbound/results/camelyon17_fullscale_B_v1/wilds_camelyon17_kga.json"
OUTDIR = ROOT + "/experiments/kbound/results/camelyon17_fullscale_B_v1/bias_variance_diag"


def quant(x, q):
    return float(np.quantile(np.asarray(x, float), q))


# ----------------------------------------------------------------- n=256 ----
def load_256():
    d = json.load(open(F256))
    recs = d["records"]
    Z = np.array([r["Z"] for r in recs], float)
    a0 = np.array([r["a0"] for r in recs], float)
    aa = np.array([r["aa"] for r in recs], float)
    B = np.array([r["B"] for r in recs], float)
    seed = np.array([r["seed"] for r in recs], int)
    assert np.max(np.abs(B - (aa - a0))) < 1e-9
    return Z, a0, aa, B, seed


def leave_one_seed_residuals(Z, B, seed):
    """LEAVE-ONE-SEED cross-fit (held out by seed, as eps-recal does): for each
    seed s, fit GBR on the other seeds, predict on s. r_i = |Bhat_i - B_i|."""
    Bhat = np.full(len(B), np.nan)
    for s in sorted(set(seed.tolist())):
        te = seed == s
        tr = ~te
        m = GradientBoostingRegressor(**GBR_KW)
        m.fit(Z[tr], B[tr])
        Bhat[te] = m.predict(Z[te])
    r = np.abs(Bhat - B)
    return Bhat, r


def main():
    Z, a0, aa, B, seed = load_256()
    n256 = 256

    # --- STEP 1: observed eps_256 from leave-one-seed cross-fit residuals ---
    Bhat, r = leave_one_seed_residuals(Z, B, seed)
    eps_256_obs = quant(r, 1 - ALPHA)
    resid_std = float(r.std(ddof=1))          # std of |r|
    signed = Bhat - B
    var_r = float(signed.var(ddof=1))         # variance of the SIGNED residual
    var_absr = float(r.var(ddof=1))

    # --- STEP 2: measurement-noise floor (binomial), per record, n=256 ---
    # In the 256 pool a0*256 and aa*256 are integers => balanced acc == raw acc
    # over n=256, so n_eff = 256 (no disagreement-region reduction here).
    a0_int = np.allclose(a0 * n256, np.round(a0 * n256))
    aa_int = np.allclose(aa * n256, np.round(aa * n256))
    sig2_meas = a0 * (1 - a0) / n256 + aa * (1 - aa) / n256
    mean_sig2 = float(sig2_meas.mean())
    # predicted measurement-only eps = 0.9-quantile of half-normal w/ that sigma.
    # Use the typical (mean) sigma to form a scalar prediction; also report the
    # 0.9-quantile over the per-record half-normal predictions for robustness.
    eps_meas_256 = float(HN_Q90 * math.sqrt(mean_sig2))
    eps_meas_256_perrec_q90 = quant(HN_Q90 * np.sqrt(sig2_meas), 0.90)
    eps_meas_1024 = eps_meas_256 / 2.0        # 1/sqrt(4) shrink prediction

    # --- STEP 4: fraction of residual variance from measurement noise ---
    frac_meas = mean_sig2 / var_r
    frac_meas_absvar = mean_sig2 / var_absr
    # predicted bias-limited radius at 1024 == eps_256_obs (model err invariant)
    eps_bias_pred_1024 = eps_256_obs

    # --------------------------------------------------- n=1024 cross-check ---
    e = json.load(open(F1024))
    n1024_recs = []
    for mth in e["methods"]:
        for rr in e["methods"][mth]["records"]:
            n1024_recs.append((mth, rr))
    B1024 = np.array([rr["B"] for _, rr in n1024_recs], float)
    a01024 = np.array([rr["a0"] for _, rr in n1024_recs], float)
    aa1024 = np.array([rr["aa"] for _, rr in n1024_recs], float)
    n_used = int(n1024_recs[0][1].get("n_eval_used", 1024))
    # measurement floor empirically at 1024 (per-record binomial sigma)
    sig2_1024 = a01024 * (1 - a01024) / n_used + aa1024 * (1 - aa1024) / n_used
    eps_meas_1024_empirical = float(HN_Q90 * math.sqrt(sig2_1024.mean()))
    # |B| spread and B std at 1024 vs 256 (cross-check; n=10 is weak)
    B256_std = float(B.std(ddof=1))
    B1024_std = float(B1024.std(ddof=1))
    absB256 = float(np.mean(np.abs(B)))
    absB1024 = float(np.mean(np.abs(B1024)))

    # ----- VERDICT logic -----
    ratio = eps_256_obs / eps_meas_256
    if frac_meas < 0.33 and ratio > 1.8:
        verdict = "BIAS-LIMITED"
        verdict_reason = (
            "Observed eps_256 (%.4f) >> measurement floor eps_meas(256)=%.4f "
            "(ratio %.1fx); only %.0f%% of residual variance is measurement "
            "noise. The radius is dominated by B_hat(Z) model error "
            "(calibration-drift bias), which does NOT shrink with eval n. "
            "Protocol B's 1/sqrt(n) prediction for eps(1024) is therefore "
            "unattainable; a fair re-run at n_eval=1024 cannot close the radius."
            % (eps_256_obs, eps_meas_256, ratio, 100 * frac_meas))
    elif frac_meas > 0.66 and ratio < 1.3:
        verdict = "VARIANCE-LIMITED"
        verdict_reason = (
            "Observed eps_256 (%.4f) ~ measurement floor eps_meas(256)=%.4f "
            "(ratio %.1fx); %.0f%% of residual variance is measurement noise. "
            "A fair re-run at n_eval=1024 should shrink eps toward %.4f."
            % (eps_256_obs, eps_meas_256, ratio, 100 * frac_meas, eps_meas_1024))
    else:
        verdict = "INCONCLUSIVE"
        verdict_reason = (
            "frac_meas=%.2f, eps ratio=%.1fx fall in the ambiguous band; "
            "the data cannot cleanly separate bias from variance." % (frac_meas, ratio))

    out = {
        "reused_estimator": "GradientBoostingRegressor " + str(GBR_KW) +
                            " (EXACT from eps_recal_camelyon.py)",
        "alpha_FIXED": ALPHA,
        "n256_file": F256, "n1024_file": F1024,
        "n256_records": int(len(B)), "seeds_256": sorted(set(seed.tolist())),
        "residual_method": "leave-one-seed cross-fit (held out by seed)",
        "step1_model_error": {
            "eps_256_observed": eps_256_obs,
            "residual_abs_std": resid_std,
            "var_signed_residual": var_r,
            "var_abs_residual": var_absr,
        },
        "step2_measurement_floor": {
            "n_per_cell": n256,
            "a0_times_n_integer": bool(a0_int), "aa_times_n_integer": bool(aa_int),
            "n_eff_note": ("a0*256 and aa*256 are integers => balanced-acc equals "
                           "raw acc over n=256 in this pool; no disagreement-region "
                           "reduction => n_eff = 256."),
            "mean_sigma_meas_sq": mean_sig2,
            "mean_sigma_meas": float(math.sqrt(mean_sig2)),
            "eps_meas_256": eps_meas_256,
            "eps_meas_256_perrec_q90": eps_meas_256_perrec_q90,
        },
        "step3_key_numbers": {
            "eps_256_observed": eps_256_obs,
            "eps_meas_256": eps_meas_256,
            "ratio_obs_over_meas": ratio,
            "eps_meas_1024_variance_limited_prediction": eps_meas_1024,
            "eps_bias_limited_prediction_1024": eps_bias_pred_1024,
        },
        "step4_variance_decomposition": {
            "frac_meas_signedvar": frac_meas,
            "frac_meas_absvar": frac_meas_absvar,
            "interpretation": "frac_meas near 1 => variance-limited; near 0 => bias-limited",
        },
        "n1024_crosscheck": {
            "n_points": len(B1024), "n_eval_used": n_used,
            "WEAK_n_warning": "n=10 points (5 seeds x 2 methods); underpowered.",
            "B_std_256": B256_std, "B_std_1024": B1024_std,
            "mean_absB_256": absB256, "mean_absB_1024": absB1024,
            "eps_meas_1024_empirical": eps_meas_1024_empirical,
            "B_std_ratio_1024_over_256": B1024_std / B256_std,
            "note": ("If B-spread at 1024 ~ B-spread at 256 (ratio ~1), the spread "
                     "is NOT shrinking ~1/sqrt(4)=0.5 as variance-limited would "
                     "require -> consistent with bias-limited."),
        },
        "VERDICT": verdict,
        "verdict_reason": verdict_reason,
    }
    json.dump(out, open(OUTDIR + "/diag_results.json", "w"), indent=2)

    # ----------------------------------------------------------- the PNG ----
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.hist(r, bins=24, color="#4C72B0", alpha=0.80,
            edgecolor="white", label="|r| = |B_hat(Z)-B|  (n=256, LOSO cross-fit)")
    ax.axvline(eps_256_obs, color="#C44E52", lw=2.4,
               label="eps_256 observed = %.4f" % eps_256_obs)
    ax.axvline(eps_meas_256, color="#55A868", lw=2.4, ls="--",
               label="eps_meas(256) floor = %.4f" % eps_meas_256)
    ax.axvline(eps_meas_1024, color="#8172B3", lw=2.0, ls=":",
               label="eps_meas(1024) var-pred = %.4f" % eps_meas_1024)
    ax.set_xlabel("absolute eps-recal residual |r|")
    ax.set_ylabel("count")
    ax.set_title("Camelyon17 certificate radius: model error vs measurement floor\n"
                 "VERDICT: %s  (frac_meas=%.2f, eps obs/floor=%.1fx)"
                 % (verdict, frac_meas, ratio), fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTDIR + "/residual_vs_measurement_floor.png", dpi=140)

    # console
    print("eps_256_observed      = %.4f" % eps_256_obs)
    print("eps_meas(256) floor   = %.4f  (ratio obs/floor = %.2fx)" % (eps_meas_256, ratio))
    print("eps_meas(1024) var-pred = %.4f   eps_bias-pred(1024) = %.4f"
          % (eps_meas_1024, eps_bias_pred_1024))
    print("frac_meas (signed var) = %.3f" % frac_meas)
    print("1024 B-std/256 B-std   = %.3f  (var-limited would be ~0.50)"
          % (B1024_std / B256_std))
    print("VERDICT:", verdict)
    print(verdict_reason)
    return out


if __name__ == "__main__":
    main()
