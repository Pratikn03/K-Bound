"""Anytime-valid (e-value / testing-by-betting) K-Bound certificate -- numerical validation.

This script validates the ANYTIME-VALID complement to the batch finite-sample
certificate (Theorem 3, ``thm:cert``, in docs/research/kbound/kbound.tex). It does
NOT modify the paper.

------------------------------------------------------------------------------
SETTING (identical to the batch certificate)
------------------------------------------------------------------------------
Paired benefits  X_i = ell(f0(x_i), y_i) - ell(fa(x_i), y_i),  bounded in [a, b].
    Delta = E[X_i].   Adapting helps iff Delta > 0.
The batch Theorem 3 uses the Maurer-Pontil empirical-Bernstein lower confidence
bound on a FIXED sample of size n to certify sign(Delta) (see
vendored_from_elara/certification/switching_certificate.py). Here we want a rule
that may LOOK AT THE DATA AS IT ARRIVES and STOP WHENEVER it has enough evidence,
while still controlling the error probability simultaneously over ALL stopping
times -- i.e. an anytime-valid certificate.

------------------------------------------------------------------------------
THE e-PROCESS (one-sided, for H0: Delta <= 0)
------------------------------------------------------------------------------
With the H0 boundary mu0 = 0, pick PREDICTABLE bets lambda_i (functions of
X_1, ..., X_{i-1} only) and form the betting wealth

        E_t^+ = prod_{i=1}^t ( 1 + lambda_i * (X_i - mu0) ),   E_0^+ = 1.

Positivity: with X_i in [a, b] and mu0 = 0, each factor stays > 0 provided
lambda_i in [0, 1/(-a)) when a < 0 (the binding constraint is X_i = a). We bet
in the nonnegative range [0, c/(-a)] with c < 1, so positivity holds with margin.

Supermartingale under H0. Because lambda_i is F_{i-1}-measurable and X_i is drawn
fresh,
        E[E_t^+ | F_{t-1}] = E_{t-1}^+ * ( 1 + lambda_i * (E[X_i] - mu0) )
                           = E_{t-1}^+ * ( 1 + lambda_i * Delta )
                           <= E_{t-1}^+        (since lambda_i >= 0 and Delta <= 0).
So (E_t^+) is a nonnegative supermartingale with E[E_0^+] = 1. By VILLE'S
INEQUALITY, for any P with Delta <= 0,
        P( exists t >= 1 : E_t^+ >= 1/alpha ) <= alpha.

Symmetric e-process for H0': Delta >= 0 uses the centered values (mu0 - X_i) with
nonnegative bets:
        E_t^- = prod_{i=1}^t ( 1 + nu_i * (mu0 - X_i) ),   nu_i in [0, c/b].

Anytime-valid decision (run both processes online):
    ADAPT   the first time  E_t^+ >= 1/alpha   (reject Delta <= 0),
    FREEZE  the first time  E_t^- >= 1/alpha   (reject Delta >= 0),
    else keep sampling (ABSTAIN).
Ville controls the anytime FALSE-ADAPT probability at <= alpha under any
Delta <= 0, and the anytime FALSE-FREEZE probability at <= alpha under any
Delta >= 0 -- SIMULTANEOUSLY over all t (no fixed-n, no multiplicity correction
for the repeated looks).

------------------------------------------------------------------------------
PREDICTABLE BETTING RULE (truncated aGRAPA, Waudby-Smith & Ramdas 2024)
------------------------------------------------------------------------------
The growth-rate-optimal-in-hindsight bet against mu0 is lambda* = Delta /
E[(X - mu0)^2]. We do not know Delta, so we plug in the running (predictable)
estimates from the first t-1 samples:

        mu_hat_{t-1}    = running mean of X_1..X_{t-1},
        sigma2_hat_{t-1}= running 2nd moment of (X - mu0) with a small prior,
        lambda_t        = clip( mu_hat_{t-1} / sigma2_hat_{t-1}, 0, c/(-a) ).

This is PREDICTABLE (depends only on the past), nonnegative (one-sided test of
H0: Delta <= 0), and bounded (so the wealth factors stay positive). Validity of
the certificate needs ONLY predictability + boundedness; the *specific* bet
affects power and stopping time, not validity.

------------------------------------------------------------------------------
WHAT THIS SCRIPT MEASURES
------------------------------------------------------------------------------
1. Anytime false-adapt rate under H0 (Delta = 0, the hardest H0 boundary, and
   Delta < 0): fraction of runs in which E_t^+ EVER crosses 1/alpha within a long
   horizon. Theory: <= alpha for every Delta <= 0.
2. Detection power and mean/median stopping time under H1 (Delta > 0).
3. A sanity check that the e-process is a supermartingale: the Monte-Carlo mean
   of E_t^+ under Delta = 0 stays <= 1 (up to MC noise) for all t.

We sweep several bounded benefit distributions (two-point, Beta-shifted,
clipped-Gaussian) to show validity does not depend on the shape, only on the
[a, b] bound and Delta.

The e-process is run VECTORIZED across all Monte-Carlo runs at once: streams are
drawn as an (n_runs, horizon) matrix and the recursive wealth update is stepped
with one Python loop over t and numpy ops over the run axis.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

# Benefit range [a, b]; matches the |p - y| paired-loss benefit range in
# vendored_from_elara/certification/switching_certificate.py (each loss in
# [0,1] => X = l_static - l_gated in [-1, 1]).
A_LO, B_HI = -1.0, 1.0


# ----------------------------------------------------------------------------
# Bounded paired-benefit stream generators -> (n_runs, horizon) matrices.
# ----------------------------------------------------------------------------
def stream_twopoint(rng: np.random.Generator, shape, delta: float) -> np.ndarray:
    """X in {a, b} with P(X=b) so E[X] = delta. Extreme two-endpoint mass: the
    worst case for the boundedness/positivity argument."""
    a, b = A_LO, B_HI
    p_b = min(max((delta - a) / (b - a), 0.0), 1.0)  # E[X] = a + p_b (b-a) = delta
    return np.where(rng.random(shape) < p_b, b, a).astype(np.float64)


def stream_beta(rng: np.random.Generator, shape, delta: float) -> np.ndarray:
    """X = a + (b-a) Beta(k1,k2), mean shaped to delta. Smooth, full support."""
    a, b = A_LO, B_HI
    m = min(max((delta - a) / (b - a), 1e-3), 1 - 1e-3)
    conc = 4.0
    z = rng.beta(m * conc, (1 - m) * conc, size=shape)
    return (a + (b - a) * z).astype(np.float64)


# Cache of clipped-Gaussian location params so we do the (expensive) mean
# correction once per delta, not once per run.
_CLIPGAUSS_MU: dict[float, float] = {}


def _clipgauss_mu(delta: float, sd: float, rng: np.random.Generator) -> float:
    key = round(delta, 6)
    if key in _CLIPGAUSS_MU:
        return _CLIPGAUSS_MU[key]
    a, b = A_LO, B_HI
    mu = delta
    for _ in range(4):  # a few correction steps so E[clip(N(mu,sd))] ~ delta
        big = rng.standard_normal(400_000) * sd + mu
        mu += (delta - np.clip(big, a, b).mean())
    _CLIPGAUSS_MU[key] = float(mu)
    return float(mu)


def stream_clipgauss(rng: np.random.Generator, shape, delta: float) -> np.ndarray:
    """N(delta, sd) clipped to [a, b], re-centered so the clipped mean ~ delta."""
    a, b = A_LO, B_HI
    sd = 0.5
    mu = _clipgauss_mu(delta, sd, rng)
    return np.clip(rng.standard_normal(shape) * sd + mu, a, b).astype(np.float64)


STREAMS = {
    "twopoint": stream_twopoint,
    "beta": stream_beta,
    "clipgauss": stream_clipgauss,
}


# ----------------------------------------------------------------------------
# Vectorized one-sided e-process E_t^+ over a batch of runs.
# ----------------------------------------------------------------------------
def run_eprocess_plus_batch(
    X: np.ndarray,          # (n_runs, horizon) bounded benefits in [a, b]
    *,
    alpha: float,
    a: float = A_LO,
    bet_cap_frac: float = 0.5,
    prior_var: float = 0.25,
    prior_weight: float = 1.0,
    record_times: list[int] | None = None,
):
    """Run E_t^+ = prod (1 + lambda_i (X_i - 0)) for H0: Delta <= 0 over every run
    simultaneously, with the truncated-aGRAPA PREDICTABLE bet.

    lambda_i is a function of X[:, :i] only (predictable). lambda_i in
    [0, bet_cap_frac/(-a)] guarantees the worst factor 1 + lambda_i*a >=
    1 - bet_cap_frac > 0.

    Returns dict with:
      crossed       (n_runs,) bool : did logW ever reach log(1/alpha)?
      stop_time     (n_runs,) int  : first crossing t (horizon+1 if never)
      log_final     (n_runs,)      : terminal log-wealth
      wealth_at     {t: (n_runs,)} : raw wealth at requested record_times (for the
                                     supermartingale check), if record_times given.
    """
    mu0 = 0.0
    n_runs, horizon = X.shape
    lam_max = bet_cap_frac / (-a)          # a < 0
    log_thr = math.log(1.0 / alpha)
    record_set = sorted(set(record_times)) if record_times else []

    log_w = np.zeros(n_runs)
    crossed = np.zeros(n_runs, dtype=bool)
    stop_time = np.full(n_runs, horizon + 1, dtype=np.int64)
    wealth_at: dict[int, np.ndarray] = {}

    # Predictable running stats over the PAST (exclude current column).
    s1 = np.zeros(n_runs)                                   # sum X_j, j < i
    s2 = np.full(n_runs, prior_weight * prior_var)          # sum (X_j - mu0)^2 + prior
    cnt = 0.0
    cnt_var = float(prior_weight)

    ri = 0
    for i in range(horizon):
        mu_hat = s1 / cnt if cnt > 0 else np.zeros(n_runs)
        sig2_hat = s2 / cnt_var
        lam = np.where(sig2_hat > 0, mu_hat / sig2_hat, 0.0)
        np.clip(lam, 0.0, lam_max, out=lam)                # nonneg + bounded

        xi = X[:, i]
        log_w += np.log(np.maximum(1.0 + lam * (xi - mu0), 1e-300))

        newly = (~crossed) & (log_w >= log_thr)
        stop_time[newly] = i + 1
        crossed |= newly

        # update predictable stats for the NEXT step
        s1 += xi
        s2 += (xi - mu0) ** 2
        cnt += 1.0
        cnt_var += 1.0

        if ri < len(record_set) and (i + 1) == record_set[ri]:
            wealth_at[record_set[ri]] = np.exp(log_w)
            ri += 1

    return {
        "crossed": crossed,
        "stop_time": stop_time,
        "log_final": log_w,
        "wealth_at": wealth_at,
    }


# ----------------------------------------------------------------------------
# Monte-Carlo drivers.
# ----------------------------------------------------------------------------
def mc_run(
    *, stream_name: str, delta: float, alpha: float, horizon: int, n_runs: int,
    rng: np.random.Generator, bet_cap_frac: float, record_times=None,
) -> dict:
    X = STREAMS[stream_name](rng, (n_runs, horizon), delta)
    out = run_eprocess_plus_batch(
        X, alpha=alpha, a=A_LO, bet_cap_frac=bet_cap_frac, record_times=record_times)
    crossed = out["crossed"]
    stopped = out["stop_time"][crossed]
    res = {
        "stream": stream_name,
        "delta": delta,
        "alpha": alpha,
        "horizon": horizon,
        "n_runs": n_runs,
        "cross_rate": float(crossed.mean()),     # = false-adapt rate when delta<=0
        "n_crossed": int(crossed.sum()),
        "mean_stop_time": float(stopped.mean()) if stopped.size else float("nan"),
        "median_stop_time": float(np.median(stopped)) if stopped.size else float("nan"),
        "p90_stop_time": float(np.percentile(stopped, 90)) if stopped.size else float("nan"),
        "mean_log_wealth_final": float(out["log_final"].mean()),
    }
    if record_times:
        res["E_wealth_mean"] = {str(t): float(out["wealth_at"][t].mean()) for t in record_times}
        res["E_wealth_sem"] = {
            str(t): float(out["wealth_at"][t].std(ddof=1) / math.sqrt(n_runs))
            for t in record_times
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Anytime-valid e-value K-Bound validation.")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--horizon", type=int, default=2000,
                    help="max samples per stream before forced stop")
    ap.add_argument("--n_runs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bet_cap_frac", type=float, default=0.5,
                    help="lambda <= bet_cap_frac/|a|; <1 guarantees positivity")
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "val_thm3_evalue_results.json"))
    args = ap.parse_args()

    alpha = args.alpha
    rng = np.random.default_rng(args.seed)

    print("=" * 80)
    print("ANYTIME-VALID K-BOUND CERTIFICATE (e-value / testing-by-betting)")
    print("Complement to the batch empirical-Bernstein certificate (Theorem 3).")
    print(f"alpha={alpha}  1/alpha={1/alpha:.1f}  horizon={args.horizon}  "
          f"n_runs={args.n_runs}  benefit range [a,b]=[{A_LO},{B_HI}]")
    print(f"bet cap: lambda in [0, {args.bet_cap_frac}/|a|] = "
          f"[0, {args.bet_cap_frac/(-A_LO):.3f}] (guarantees wealth factors > 0)")
    print("=" * 80)

    streams = list(STREAMS.keys())
    report: dict = {"config": vars(args), "results": {}}

    # ---- (A) Anytime FALSE-ADAPT under H0: Delta <= 0 -----------------------
    print(f"\n[A] ANYTIME FALSE-ADAPT under H0 (Delta <= 0). Theory: <= alpha = {alpha}.")
    print(f"{'stream':<11}{'Delta':>8}{'false_adapt':>13}{'mean_logW':>11}  flag")
    h0_results, worst = [], 0.0
    for sname in streams:
        for d in [-0.20, -0.05, 0.0]:   # 0.0 is the hardest boundary case
            r = mc_run(stream_name=sname, delta=d, alpha=alpha, horizon=args.horizon,
                       n_runs=args.n_runs, rng=rng, bet_cap_frac=args.bet_cap_frac)
            h0_results.append(r)
            worst = max(worst, r["cross_rate"])
            flag = "OK" if r["cross_rate"] <= alpha + 1e-9 else "VIOLATION"
            print(f"{sname:<11}{d:>8.2f}{r['cross_rate']:>13.4f}"
                  f"{r['mean_log_wealth_final']:>11.3f}  {flag}")
    report["results"]["false_adapt_h0"] = h0_results
    report["results"]["worst_case_false_adapt_h0"] = worst
    print(f"  -> WORST-CASE anytime false-adapt across all H0 settings: {worst:.4f} "
          f"(must be <= {alpha})")

    # ---- (B) Supermartingale sanity at Delta = 0 ----------------------------
    print("\n[B] SUPERMARTINGALE CHECK at Delta = 0: Monte-Carlo E[E_t^+] should "
          "stay <= 1 (up to MC noise).")
    rec = [1, 5, 25, 100, 500, args.horizon]
    sm_results = []
    for sname in streams:
        r = mc_run(stream_name=sname, delta=0.0, alpha=alpha, horizon=args.horizon,
                   n_runs=args.n_runs, rng=rng, bet_cap_frac=args.bet_cap_frac,
                   record_times=rec)
        ok = all(r["E_wealth_mean"][str(t)] <= 1.0 + 3.0 * r["E_wealth_sem"][str(t)]
                 for t in rec)
        r["all_leq_one_within_3sem"] = bool(ok)
        sm_results.append(r)
        means = "  ".join(f"t={t}:{r['E_wealth_mean'][str(t)]:.3f}" for t in rec)
        print(f"  {sname:<11} E[E_t^+]:  {means}  | all<=1(3sem):{ok}")
    report["results"]["supermartingale_check"] = sm_results

    # ---- (C) Detection POWER and STOPPING TIME under H1: Delta > 0 ----------
    print("\n[C] DETECTION POWER and STOPPING TIME under H1 (Delta > 0).")
    print(f"{'stream':<11}{'Delta':>8}{'power':>9}{'mean_stop':>11}"
          f"{'median_stop':>13}{'p90_stop':>10}")
    h1_results = []
    for sname in streams:
        for d in [0.05, 0.10, 0.20, 0.40]:
            r = mc_run(stream_name=sname, delta=d, alpha=alpha, horizon=args.horizon,
                       n_runs=args.n_runs, rng=rng, bet_cap_frac=args.bet_cap_frac)
            h1_results.append(r)
            print(f"{sname:<11}{d:>8.2f}{r['cross_rate']:>9.3f}"
                  f"{r['mean_stop_time']:>11.1f}{r['median_stop_time']:>13.1f}"
                  f"{r['p90_stop_time']:>10.1f}")
    report["results"]["power_h1"] = h1_results

    # ---- summary ------------------------------------------------------------
    verdict = "CONTROLLED" if worst <= alpha + 1e-9 else "NOT CONTROLLED -- FIX BET"
    print("\n" + "=" * 80)
    print(f"VERDICT: anytime false-adapt {verdict} (worst {worst:.4f} vs alpha {alpha}).")
    print("=" * 80)
    report["verdict"] = {
        "worst_case_false_adapt_h0": worst,
        "alpha": alpha,
        "controlled": bool(worst <= alpha + 1e-9),
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote machine-readable results to {args.out}")


if __name__ == "__main__":
    main()
