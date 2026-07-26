#!/usr/bin/env python3
"""
frontier_validation.py -- ILLUSTRATION (not a test) of the K-Bound benefit-sign frontier.

CPU-ONLY. Needs numpy + scikit-learn + matplotlib. No GPU, no datasets, ~3-5 min.

=============================================================================
WHAT THIS SCRIPT IS -- AND WHAT IT IS NOT   (fix-queue item 14 / F5-4 / F1-4)
=============================================================================
This script is an ILLUSTRATION that the decision machinery behaves as the algebra
says it must.  It is NOT a test of the frontier, because its data-generating
process makes the result true by construction.  Concretely, in
``gen_circular_world`` below:

    M     ~ U(m_lo, m_hi)                      observable margin
    gamma ~ U(-beta, beta)                     unobserved drift, |gamma| <= beta
    B     = M + gamma                          true benefit
    Z     = four noisy copies of M             evidence

Z is FOUR NOISY COPIES OF M.  So any consistent regressor recovers Bhat ~ M, the
residual ``Bhat - B`` is therefore ``-gamma`` up to observation noise, and gamma
was drawn U(-beta, beta) by us.  Three consequences follow *by algebra*, not by
measurement:

  1. ``eps -> 0.9 * beta``.  It is the 0.9-quantile of |gamma| ~ U(0, beta).
     The "eps self-calibrates to ~beta" panel is arithmetic on our own draw.
  2. The commit/abstain transition lands at |M| ~ eps ~ 0.9*beta.  The "frontier
     lands at |M| = beta" claim is the same statement restated.
  3. The reported "90.0% empirical coverage" is the DEFINITION of the empirical
     quantile at the sample sizes used: with an in-pool radius,
     ``mean(|Bhat - B| <= np.quantile(|Bhat - B|, 0.9))`` is 0.90 at n = 400 and
     at n = 220 for arbitrary data.  It measures nothing about the estimator.

A real test of the frontier needs (a) Z that is not a noisy copy of M, (b) gamma
whose 0.9-quantile is not beta, and (c) a held-out calibration set.  That
experiment is scaffolded in ``frontier_sweep.py`` and has NOT been run; do not
cite this file's numbers as evidence for the frontier claim in the paper.  The
honest caption for these three figures is "the certificate behaves as the algebra
predicts on data we generated to satisfy the algebra".

The circularity is asserted at runtime (see ``_assert_circular``) so the label
cannot silently drift away from the code.

MODEL (paper's decomposition sign(Delta)=sign(M+gamma), |gamma|<=beta) -- see above.

The decision path is imported from ``kbound_decide`` (fix-queue item 15), which
calls ``kga.certificate`` / ``kga.policy``, so this exercises the shipped library
rather than a seventh copy-pasted fork.  Radii are leave-one-out-of-pool
(fix-queue item 4).
"""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.10
SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kbound_decide import decide_kga as _decide_kga, radii_in_pool  # noqa: E402

FIGDIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
RESJSON = os.path.normpath(os.path.join(HERE, "..", "frontier_validation_results.json"))


def decide_kga(Z, B, alpha=ALPHA, n_estimators=250, max_depth=2, lr=0.05, seed=SEED,
               calibration="loo"):
    """The shipped decision path. Returns (Bhat, eps_per_cell, dec)."""
    return _decide_kga(Z, B, alpha=alpha, n_estimators=n_estimators, max_depth=max_depth,
                       lr=lr, seed=seed, calibration=calibration)


def gen_circular_world(n, beta, rng, m_lo=-0.40, m_hi=0.40, obs_noise=0.02, n_evidence=4):
    """Generate the CIRCULAR world described in the module docstring.

    Z is n_evidence noisy copies of M, so Bhat ~ M, the residual is -gamma, and
    eps -> 0.9*beta by algebra.  Renamed from ``gen_world`` to make the property
    unmissable at every call site: nothing produced from this function tests the
    frontier, it only exhibits it.
    """
    M = rng.uniform(m_lo, m_hi, n)
    gamma = rng.uniform(-beta, beta, n)                 # unobserved; |gamma| <= beta
    B = M + gamma                                        # true benefit; sign(Delta)=sign(M+gamma)
    Z = np.column_stack([M + rng.normal(0, obs_noise, n) for _ in range(n_evidence)])
    return M, gamma, B, Z


# Backwards-compatible alias; the old name hid the defect.
gen_world = gen_circular_world


def _assert_circular(M, Z, thresh=0.99):
    """Fail loudly if a future edit makes Z something other than a copy of M.

    This assertion exists to keep the file's ILLUSTRATION label honest: if Z ever
    stops being a near-deterministic function of M, the circularity argument in
    the docstring no longer applies and the caption must be rewritten (probably
    upgraded).  Either way the label must not drift away from the code.
    """
    r = max(abs(float(np.corrcoef(M, Z[:, j])[0, 1])) for j in range(Z.shape[1]))
    assert r >= thresh, (
        f"Z is no longer a noisy copy of M (max |corr| = {r:.4f} < {thresh}).\n"
        "This script's ILLUSTRATION disclaimer was written for the circular DGP. "
        "If you changed the DGP on purpose, update the module docstring and the "
        "figure captions -- and see frontier_sweep.py for the real experiment."
    )
    return r


def _quantile_identity_coverage(n, alpha=ALPHA):
    """The coverage an IN-POOL empirical quantile returns for ANY data at size n.

    ``mean(r <= np.quantile(r, 1-alpha))`` is a function of n alone.  Printing it
    next to the "empirical coverage" number is what stops that number from being
    read as evidence.
    """
    r = np.arange(n, dtype=float)   # arbitrary distinct values
    return float(np.mean(r <= float(np.quantile(r, 1 - alpha))))

def run_illustration():
    """Run the circular-world illustration. Renamed from ``main`` (fix-queue item 14)
    so that no caller can invoke it thinking it validates anything."""
    os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    R = {"alpha": ALPHA,
         "status": "ILLUSTRATION -- NOT A TEST OF THE FRONTIER",
         "note": "Z is four noisy copies of M, so the residual is exactly -gamma and "
                 "eps -> 0.9*beta by algebra. The frontier location and the reported "
                 "coverage are consequences of the data-generating process, not "
                 "measurements. See frontier_sweep.py for the real experiment.",
         "circularity_asserted": True}

    # =================== (A) estimator recovery + conformal coverage ===================
    beta = 0.10
    M, gamma, B, Z = gen_circular_world(400, beta, rng)
    R["max_abs_corr_Z_M"] = round(_assert_circular(M, Z), 6)
    Bhat, eps, dec = decide_kga(Z, B)
    eps = np.asarray(eps, float)
    cov = float(np.mean(np.abs(Bhat - B) <= eps))
    R["recovery"] = {"n": 400, "beta": beta,
                     "eps_mean": round(float(eps.mean()), 4),
                     "eps_min": round(float(eps.min()), 4),
                     "eps_max": round(float(eps.max()), 4),
                     "eps_over_beta": round(float(eps.mean()) / beta, 3),
                     "eps_predicted_by_algebra": round(0.9 * beta, 4),
                     "empirical_coverage": round(cov, 4), "target_coverage": 1 - ALPHA,
                     "in_pool_quantile_identity_coverage_at_n": round(
                         _quantile_identity_coverage(400), 4),
                     "coverage_caveat":
                         "with the archived IN-POOL interpolated radius this coverage is a "
                         "function of n alone (0.90 at n=400) and carries no information; the "
                         "leave-one-out radius used here at least lets it vary.",
                     "mae": round(float(np.mean(np.abs(Bhat - B))), 4)}
    eps = float(eps.mean())   # the figures below draw a single band
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(B, Bhat, s=10, alpha=0.5, color="#2b6cb0")
    lo, hi = B.min(), B.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    ax.fill_between([lo, hi], [lo - eps, hi - eps], [lo + eps, hi + eps], color="#2b6cb0", alpha=0.12,
                    label=fr"$\pm\varepsilon$ ({cov*100:.0f}% cover)")
    ax.set_xlabel("true benefit $B=M+\\gamma$"); ax.set_ylabel(r"estimate $\widehat{B}$")
    ax.set_title(f"Estimator recovery, circular DGP (coverage {cov*100:.0f}%)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_frontier_recovery.png"), dpi=150); plt.close(fig)

    # =================== (B) frontier transition: commit rate vs |M|, band = beta ===================
    absM = np.abs(M); commit = (dec != "ABSTAIN")
    edges = np.linspace(0, 0.40, 17); ctr = 0.5 * (edges[:-1] + edges[1:])
    rate = np.array([commit[(absM >= edges[k]) & (absM < edges[k+1])].mean()
                     if ((absM >= edges[k]) & (absM < edges[k+1])).any() else np.nan
                     for k in range(len(edges) - 1)])
    # sign correctness among committed cells (should be ~1: KGA only commits when sign is knowable)
    signdec = np.where(dec == "ADAPT", 1, np.where(dec == "FREEZE", -1, 0))
    sign_ok = float(np.mean(signdec[commit] == np.sign(B[commit]))) if commit.any() else float("nan")
    R["frontier"] = {"beta": beta, "eps": round(eps, 4),
                     "commit_rate_below_beta": round(float(np.nanmean(rate[ctr < beta])), 3),
                     "commit_rate_above_beta": round(float(np.nanmean(rate[ctr > beta])), 3),
                     "sign_correct_among_committed": round(sign_ok, 4),
                     "why_this_is_not_evidence":
                         "the transition sits at |M| ~ eps and eps ~ 0.9*beta because the "
                         "residual IS gamma ~ U(-beta,beta) by construction; the plot restates "
                         "the DGP."}
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot(ctr, rate, "-o", ms=4, color="#c05621", label="commit rate (adapt/freeze)")
    ax.axvline(beta, color="k", ls="--", lw=1, label=fr"$\beta={beta}$")
    ax.axvline(eps, color="#2b6cb0", ls=":", lw=1.2, label=fr"$\varepsilon\approx{eps:.2f}$")
    ax.axvspan(0, beta, color="grey", alpha=0.12, label="predicted abstain band")
    ax.set_xlabel("observable margin $|M|$"); ax.set_ylabel("fraction committed")
    ax.set_title("Frontier transition at $|M|=\\beta$ (illustration)"); ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_frontier_transition.png"), dpi=150); plt.close(fig)

    # =================== (C) FA_u <= alpha and eps ~ beta, across beta ===================
    betas = [0.05, 0.10, 0.15, 0.20]; REP = 8; fa_by_beta = {}
    for b in betas:
        fa, ff, cv, ep = [], [], [], []
        for r in range(REP):
            Mr, gr, Br, Zr = gen_circular_world(220, b, np.random.default_rng(1000 + r))
            _assert_circular(Mr, Zr)
            Bh, e, d = decide_kga(Zr, Br)
            e = float(np.mean(np.asarray(e, float)))
            fa.append(float(np.mean((d == "ADAPT")  & (Br <= 0))))   # false-adapt
            ff.append(float(np.mean((d == "FREEZE") & (Br >= 0))))   # false-freeze
            cv.append(float(np.mean(np.abs(Bh - Br) <= e))); ep.append(e)
        fa_by_beta[str(b)] = {"mean_FA_u": round(float(np.mean(fa)), 4), "max_FA_u": round(float(np.max(fa)), 4),
                              "mean_FF_u": round(float(np.mean(ff)), 4),
                              "mean_eps": round(float(np.mean(ep)), 4), "mean_coverage": round(float(np.mean(cv)), 4)}
    R["fa_by_beta"] = fa_by_beta
    R["fa_by_beta_caveat"] = (
        "eps ~ 0.9*beta in every row because gamma ~ U(-beta,beta) was drawn that way. "
        "The 'eps self-calibrates to beta' panel is a plot of our own generator.")
    bs = np.array(betas)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    fau = [fa_by_beta[str(b)]["mean_FA_u"] for b in betas]
    faumax = [fa_by_beta[str(b)]["max_FA_u"] for b in betas]
    axes[0].plot(bs, fau, "-o", color="#2b6cb0", label="mean $FA_u$")
    axes[0].plot(bs, faumax, "--^", color="#2b6cb0", alpha=0.6, label="max $FA_u$")
    axes[0].axhline(ALPHA, color="red", ls="--", lw=1, label=fr"target $\alpha={ALPHA}$")
    axes[0].set_xlabel(r"drift budget $\beta$"); axes[0].set_ylabel("false-adapt rate $FA_u$")
    axes[0].set_title("$FA_u \\leq \\alpha$ on the circular DGP"); axes[0].set_ylim(0, max(ALPHA*1.6, max(faumax)*1.3))
    axes[0].legend(fontsize=8)
    eps_b = [fa_by_beta[str(b)]["mean_eps"] for b in betas]
    axes[1].plot(bs, eps_b, "-o", color="#c05621", label=r"conformal $\varepsilon$")
    axes[1].plot(bs, bs, "k--", lw=1, label=r"$\varepsilon=\beta$")
    axes[1].set_xlabel(r"drift budget $\beta$"); axes[1].set_ylabel(r"conformal radius $\varepsilon$")
    axes[1].set_title(r"$\varepsilon\to 0.9\beta$ (algebraic, not measured)"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_frontier_fa_coverage.png"), dpi=150); plt.close(fig)

    json.dump(R, open(RESJSON, "w"), indent=2)
    print("=" * 68)
    print("FRONTIER ILLUSTRATION -- NOT A TEST (fix-queue item 14)")
    print("  Z is four noisy copies of M => residual == -gamma => eps -> 0.9*beta by algebra.")
    print("  Nothing below is evidence for the frontier claim. See frontier_sweep.py.")
    print(f"  (A) recovery: coverage {R['recovery']['empirical_coverage']:.2%} "
          f"(an in-pool quantile would return "
          f"{R['recovery']['in_pool_quantile_identity_coverage_at_n']:.2%} for ANY data at n=400)  "
          f"MAE {R['recovery']['mae']}   eps {R['recovery']['eps_mean']} "
          f"(algebra predicts {R['recovery']['eps_predicted_by_algebra']})")
    print(f"  (B) transition: commit-rate below beta = {R['frontier']['commit_rate_below_beta']}, "
          f"above beta = {R['frontier']['commit_rate_above_beta']}; "
          f"sign-correct among committed = {R['frontier']['sign_correct_among_committed']:.2%}")
    print(f"  (C) FA_u <= alpha across beta:")
    for b in betas:
        d = fa_by_beta[str(b)]
        print(f"      beta={b}: mean_FA_u={d['mean_FA_u']} max_FA_u={d['max_FA_u']} "
              f"(alpha={ALPHA})  eps~{d['mean_eps']}  cover={d['mean_coverage']:.2%}")
    print(f"\n  figures -> {FIGDIR}/fig_frontier_recovery.png, fig_frontier_transition.png, fig_frontier_fa_coverage.png")
    print(f"  results -> {RESJSON}")
    print("=" * 68)

# Backwards-compatible alias.
main = run_illustration

if __name__ == "__main__":
    run_illustration()
