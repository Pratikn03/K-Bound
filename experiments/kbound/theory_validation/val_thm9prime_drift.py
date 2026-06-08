"""Theorem 9' : anytime-valid certificate under beta-bounded non-stationarity -- validation.

Validates Theorem ``thm:anytime-drift`` in
docs/research/kbound/paper/sections/theory_extensions_v2.tex. Does NOT modify the paper.

------------------------------------------------------------------------------
RELAXATION
------------------------------------------------------------------------------
The appendix e-process (thm:anytime) assumes a CONSTANT conditional mean
E[X_t | F_{t-1}] = Delta. We relax this to a beta-bounded NON-STATIONARITY:

        | E[X_t | F_{t-1}] - Delta | <= beta     for all t.                          (NS)

So the per-step conditional mean mu_t := E[X_t|F_{t-1}] satisfies mu_t <= Delta + beta.
Under H0 (Delta <= 0) we therefore only have mu_t <= beta, NOT mu_t <= 0; the betting
wealth E_t^+ = prod_{i<=t} (1 + lambda_i X_i) is NO LONGER a supermartingale.

------------------------------------------------------------------------------
CORRECTION (the theorem)
------------------------------------------------------------------------------
With predictable lambda_i in [0, c/(-a)] (so factors stay positive), under (NS)+H0,

    E[E_t^+ | F_{t-1}] = E_{t-1}^+ (1 + lambda_t mu_t) <= E_{t-1}^+ (1 + lambda_t beta).

Hence the DISCOUNTED process

        M_t := E_t^+ / D_t,     D_t := prod_{i<=t} (1 + lambda_i beta),     D_0 = 1,

is a nonnegative supermartingale (E[M_t|F_{t-1}] <= M_{t-1}, E[M_0]=1). Ville gives

        Pr( exists t : M_t >= 1/alpha )  <= alpha
   <=>  Pr( exists t : E_t^+ >= (1/alpha) * D_t )  <= alpha   under any Delta <= 0.

So the decision threshold is INFLATED by the drift correction D_t. Since
log(1+lambda_i beta) <= lambda_i beta, the log-wealth drifts by at most

        log D_t = sum_{i<=t} log(1 + lambda_i beta)  <=  beta * sum_{i<=t} lambda_i,

i.e. PER-STEP drift <= lambda_t beta and CUMULATIVE log-drift <= beta * sum lambda_i.
Equivalently a conservative threshold is (1/alpha) * exp( beta * sum_{i<=t} lambda_i ).

DECISION (corrected): ADAPT the first t with  E_t^+ >= (1/alpha) D_t.
FALSE-ADAPT (Delta <= 0) is then <= alpha for ANY drift obeying (NS).

TIGHTNESS: the correction is necessary. If one keeps the UNCORRECTED threshold 1/alpha
while beta > 0, false-adapt EXCEEDS alpha (the wealth has positive drift under H0). We
demonstrate both: corrected threshold controls error for all tested beta; the uncorrected
threshold is violated once beta is large.

------------------------------------------------------------------------------
SIMULATION
------------------------------------------------------------------------------
H0 streams with Delta <= 0 and adversarial-ish bounded drift: we set the per-step mean
mu_t = min(Delta + beta, beta) (the WORST case allowed by (NS) under H0, pushing wealth
up), draw X_t in [a,b] with that mean (shifted/clipped Gaussian), and run the predictable
truncated-GRAPA bet. We report, over many runs, the fraction that EVER cross (i) the
corrected threshold and (ii) the uncorrected 1/alpha threshold.
"""

import json
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results_thm9prime_drift.json")
SEED = 20260606


def draw_bounded(mu, a, b, sigma, rng):
    """One X in [a,b] with mean approx mu (clipped Gaussian; mean shift is small near
    the interior so the realized mean ~ mu)."""
    x = mu + sigma * rng.standard_normal()
    return float(np.clip(x, a, b))


def run_stream(Delta, beta, a, b, T, alpha, c, sigma, rng):
    """Run one H0 stream of length T under worst-case bounded drift.

    Returns (crossed_corrected, crossed_uncorrected, t_cross_corr).
    """
    # worst-case conditional mean allowed under (NS) and H0 (maximizes upward drift)
    mu_t = min(Delta + beta, beta)
    lam_cap = c / (-a)
    E = 1.0           # E_t^+
    D = 1.0           # drift discount prod (1 + lambda_i beta)
    # predictable bet state (truncated GRAPA against mu0 = 0)
    s = 0.0           # sum X
    s2 = 0.0          # sum X^2
    nseen = 0
    crossed_corr = False
    crossed_unc = False
    t_cross = None
    for t in range(1, T + 1):
        # predictable bet from PAST data only
        if nseen == 0:
            lam = 0.0
        else:
            mu_hat = s / nseen
            var_hat = max(s2 / nseen - mu_hat ** 2, 1e-6)
            lam = mu_hat / var_hat
            lam = float(np.clip(lam, 0.0, lam_cap))
        x = draw_bounded(mu_t, a, b, sigma, rng)
        # update wealth and drift discount with the bet chosen BEFORE seeing x
        E *= (1.0 + lam * x)
        D *= (1.0 + lam * beta)
        # decisions
        if not crossed_corr and E >= (1.0 / alpha) * D:
            crossed_corr = True
            t_cross = t
        if not crossed_unc and E >= (1.0 / alpha):
            crossed_unc = True
        # advance estimator state
        s += x
        s2 += x * x
        nseen += 1
    return crossed_corr, crossed_unc, t_cross


def sweep_beta(Delta=0.0, betas=None, a=-1.0, b=1.0, T=300, alpha=0.10,
               c=0.5, sigma=0.6, n_runs=2000):
    if betas is None:
        betas = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
    rng = np.random.default_rng(SEED)
    rows = []
    for beta in betas:
        n_corr = 0
        n_unc = 0
        for _ in range(n_runs):
            cc, cu, _ = run_stream(Delta, beta, a, b, T, alpha, c, sigma, rng)
            n_corr += int(cc)
            n_unc += int(cu)
        rows.append({
            "beta": beta, "Delta": Delta, "alpha": alpha,
            "false_adapt_corrected": n_corr / n_runs,
            "false_adapt_uncorrected": n_unc / n_runs,
            "corrected_controls": bool(n_corr / n_runs <= alpha + 0.02),  # +MC slack
        })
    return rows


def sweep_delta_negative(betas_fixed=0.05, Deltas=None, a=-1.0, b=1.0, T=300,
                         alpha=0.10, c=0.5, sigma=0.6, n_runs=1500):
    """Confirm: for genuinely negative Delta (well inside H0), corrected false-adapt is
    even smaller. Uses a fixed beta."""
    if Deltas is None:
        Deltas = [0.0, -0.05, -0.10, -0.20]
    rng = np.random.default_rng(SEED + 7)
    rows = []
    for Delta in Deltas:
        n_corr = 0
        for _ in range(n_runs):
            cc, _, _ = run_stream(Delta, betas_fixed, a, b, T, alpha, c, sigma, rng)
            n_corr += int(cc)
        rows.append({"Delta": Delta, "beta": betas_fixed,
                     "false_adapt_corrected": n_corr / n_runs})
    return rows


def power_under_alt(Delta=0.30, beta=0.05, a=-1.0, b=1.0, T=300, alpha=0.10,
                    c=0.5, sigma=0.6, n_runs=1000):
    """Sanity: under a true positive Delta (adapt is correct), the corrected rule still
    has high detection power and finite stopping time -- the correction does not destroy
    usefulness. (Here drift HELPS the alternative; we use mu_t = Delta - beta, the
    pessimistic within-(NS) value, to be conservative about power.)"""
    rng = np.random.default_rng(SEED + 13)
    n_detect = 0
    stops = []
    lam_cap = c / (-a)
    mu_t = Delta - beta
    for _ in range(n_runs):
        E = 1.0
        D = 1.0
        s = 0.0
        s2 = 0.0
        nseen = 0
        detected = False
        for t in range(1, T + 1):
            if nseen == 0:
                lam = 0.0
            else:
                mu_hat = s / nseen
                var_hat = max(s2 / nseen - mu_hat ** 2, 1e-6)
                lam = float(np.clip(mu_hat / var_hat, 0.0, lam_cap))
            x = draw_bounded(mu_t, a, b, sigma, rng)
            E *= (1.0 + lam * x)
            D *= (1.0 + lam * beta)
            if not detected and E >= (1.0 / alpha) * D:
                detected = True
                stops.append(t)
            s += x
            s2 += x * x
            nseen += 1
        n_detect += int(detected)
    return {"Delta": Delta, "beta": beta, "alpha": alpha,
            "detection_power_corrected": n_detect / n_runs,
            "mean_stop_time": float(np.mean(stops)) if stops else None}


def main():
    results = {
        "description": "Thm 9': anytime-valid certificate under beta-bounded drift (thm:anytime-drift)",
        "seed": SEED,
        "alpha": 0.10,
        "beta_sweep_at_Delta0": sweep_beta(),
        "delta_negative_sweep": sweep_delta_negative(),
        "power_under_alt": power_under_alt(),
    }
    bs = results["beta_sweep_at_Delta0"]
    alpha = results["alpha"]
    results["headline"] = {
        "alpha": alpha,
        "max_false_adapt_corrected_over_all_beta":
            float(max(r["false_adapt_corrected"] for r in bs)),
        "corrected_controls_all_beta":
            bool(all(r["corrected_controls"] for r in bs)),
        "uncorrected_false_adapt_at_beta_0.20":
            float([r for r in bs if abs(r["beta"] - 0.20) < 1e-9][0]["false_adapt_uncorrected"]),
        "uncorrected_false_adapt_at_beta_0.40":
            float([r for r in bs if abs(r["beta"] - 0.40) < 1e-9][0]["false_adapt_uncorrected"]),
        "uncorrected_violates_alpha_at_large_beta":
            bool(max(r["false_adapt_uncorrected"] for r in bs) > alpha),
        "detection_power_corrected": results["power_under_alt"]["detection_power_corrected"],
        "PASS_corrected_controls": bool(all(r["corrected_controls"] for r in bs)),
        "PASS_uncorrected_breaks": bool(max(r["false_adapt_uncorrected"] for r in bs) > alpha),
        "PASS_power_retained": bool(results["power_under_alt"]["detection_power_corrected"] >= 0.5),
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["headline"], indent=2))
    print("\n-- beta sweep at Delta=0 (alpha=0.10) --")
    for r in bs:
        print(f"beta={r['beta']:.2f} corrected_FA={r['false_adapt_corrected']:.4f} "
              f"uncorrected_FA={r['false_adapt_uncorrected']:.4f} "
              f"corrected_ok={int(r['corrected_controls'])}")
    print("\n-- negative Delta sweep (beta=0.05) --")
    for r in results["delta_negative_sweep"]:
        print(f"Delta={r['Delta']:.2f} corrected_FA={r['false_adapt_corrected']:.4f}")
    print("\n-- power under alt --")
    print(json.dumps(results["power_under_alt"], indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
