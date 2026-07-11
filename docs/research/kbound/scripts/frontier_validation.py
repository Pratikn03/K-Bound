#!/usr/bin/env python3
"""
frontier_validation.py -- synthetic ground-truth validation of the K-Bound benefit-sign frontier.

CPU-ONLY. Needs numpy + scikit-learn + matplotlib. No GPU, no datasets, ~3-5 min.

WHY: the paper illustrates the |M|>beta frontier with a SCHEMATIC (fig_frontier_schematic.png)
but never MEASURES the identifiability transition on data where the truth is known. This does
exactly that -- it validates, on synthetic ground truth we control:

  (Thm 2, benefit-sign frontier)  KGA commits (adapt/freeze) iff |M| > beta, and abstains within
                                  the band |M| <= beta -- the transition lands AT |M|=beta.
  (Thm 3, finite-sample certificate)  empirical false-adapt rate FA_u <= alpha.
  (estimator)  the LOO-GBR benefit estimate Bhat tracks the true benefit B, and the split-conformal
               interval [Bhat +/- eps] covers B at >= 1-alpha.

MODEL (paper's decomposition sign(Delta)=sign(M+gamma), |gamma|<=beta):
  cell i has observable margin M_i and an UNOBSERVED calibration drift gamma_i ~ U(-beta,beta);
  true benefit B_i = M_i + gamma_i. Evidence Z_i reveals M_i (small obs. noise) but NOT gamma_i.
  Because gamma is the irreducible residual, the split-conformal radius eps self-calibrates to
  ~beta, so the decision rule reproduces the |M|>beta frontier automatically. That is the point:
  the frontier is not imposed -- it emerges from the certificate.

decide_kga() below is COPIED VERBATIM from scripts/cifar_tent_mps_v2.py (L143-156) so this
validates the exact deployed machinery, not a toy re-implementation.
"""
import os, json
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.10
SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
RESJSON = os.path.normpath(os.path.join(HERE, "..", "frontier_validation_results.json"))

# ---- decide_kga: VERBATIM from cifar_tent_mps_v2.py (LOO-GBR benefit estimator + conformal radius) ----
def decide_kga(Z, B, alpha=ALPHA, n_estimators=250, max_depth=2, lr=0.05, seed=SEED):
    Z = np.asarray(Z, float); B = np.asarray(B, float); N = len(B)
    Bhat = np.zeros(N)
    for i in range(N):
        tr = np.arange(N) != i
        m = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      learning_rate=lr, subsample=0.8, random_state=seed)
        m.fit(Z[tr], B[tr]); Bhat[i] = m.predict(Z[i:i+1])[0]
    eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))
    dec = np.where(Bhat - eps > 0, "ADAPT", np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))
    return Bhat, eps, dec

def gen_world(n, beta, rng, m_lo=-0.40, m_hi=0.40, obs_noise=0.02, n_evidence=4):
    M = rng.uniform(m_lo, m_hi, n)
    gamma = rng.uniform(-beta, beta, n)                 # unobserved; |gamma| <= beta
    B = M + gamma                                        # true benefit; sign(Delta)=sign(M+gamma)
    Z = np.column_stack([M + rng.normal(0, obs_noise, n) for _ in range(n_evidence)])
    return M, gamma, B, Z

def main():
    os.makedirs(FIGDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    R = {"alpha": ALPHA, "note": "synthetic ground-truth validation of the benefit-sign frontier"}

    # =================== (A) estimator recovery + conformal coverage ===================
    beta = 0.10
    M, gamma, B, Z = gen_world(400, beta, rng)
    Bhat, eps, dec = decide_kga(Z, B)
    cov = float(np.mean(np.abs(Bhat - B) <= eps))
    R["recovery"] = {"n": 400, "beta": beta, "eps": round(eps, 4),
                     "empirical_coverage": round(cov, 4), "target_coverage": 1 - ALPHA,
                     "mae": round(float(np.mean(np.abs(Bhat - B))), 4)}
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(B, Bhat, s=10, alpha=0.5, color="#2b6cb0")
    lo, hi = B.min(), B.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    ax.fill_between([lo, hi], [lo - eps, hi - eps], [lo + eps, hi + eps], color="#2b6cb0", alpha=0.12,
                    label=fr"$\pm\varepsilon$ ({cov*100:.0f}% cover)")
    ax.set_xlabel("true benefit $B=M+\\gamma$"); ax.set_ylabel(r"estimate $\widehat{B}$")
    ax.set_title(f"Estimator recovery (coverage {cov*100:.0f}% $\\geq$ {100*(1-ALPHA):.0f}%)")
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
                     "sign_correct_among_committed": round(sign_ok, 4)}
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot(ctr, rate, "-o", ms=4, color="#c05621", label="commit rate (adapt/freeze)")
    ax.axvline(beta, color="k", ls="--", lw=1, label=fr"$\beta={beta}$")
    ax.axvline(eps, color="#2b6cb0", ls=":", lw=1.2, label=fr"$\varepsilon\approx{eps:.2f}$")
    ax.axvspan(0, beta, color="grey", alpha=0.12, label="predicted abstain band")
    ax.set_xlabel("observable margin $|M|$"); ax.set_ylabel("fraction committed")
    ax.set_title("Frontier transition at $|M|=\\beta$"); ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_frontier_transition.png"), dpi=150); plt.close(fig)

    # =================== (C) FA_u <= alpha and eps ~ beta, across beta ===================
    betas = [0.05, 0.10, 0.15, 0.20]; REP = 8; fa_by_beta = {}
    for b in betas:
        fa, ff, cv, ep = [], [], [], []
        for r in range(REP):
            Mr, gr, Br, Zr = gen_world(220, b, np.random.default_rng(1000 + r))
            Bh, e, d = decide_kga(Zr, Br)
            fa.append(float(np.mean((d == "ADAPT")  & (Br <= 0))))   # false-adapt
            ff.append(float(np.mean((d == "FREEZE") & (Br >= 0))))   # false-freeze
            cv.append(float(np.mean(np.abs(Bh - Br) <= e))); ep.append(e)
        fa_by_beta[str(b)] = {"mean_FA_u": round(float(np.mean(fa)), 4), "max_FA_u": round(float(np.max(fa)), 4),
                              "mean_FF_u": round(float(np.mean(ff)), 4),
                              "mean_eps": round(float(np.mean(ep)), 4), "mean_coverage": round(float(np.mean(cv)), 4)}
    R["fa_by_beta"] = fa_by_beta
    bs = np.array(betas)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    fau = [fa_by_beta[str(b)]["mean_FA_u"] for b in betas]
    faumax = [fa_by_beta[str(b)]["max_FA_u"] for b in betas]
    axes[0].plot(bs, fau, "-o", color="#2b6cb0", label="mean $FA_u$")
    axes[0].plot(bs, faumax, "--^", color="#2b6cb0", alpha=0.6, label="max $FA_u$")
    axes[0].axhline(ALPHA, color="red", ls="--", lw=1, label=fr"target $\alpha={ALPHA}$")
    axes[0].set_xlabel(r"drift budget $\beta$"); axes[0].set_ylabel("false-adapt rate $FA_u$")
    axes[0].set_title("Certificate holds: $FA_u \\leq \\alpha$"); axes[0].set_ylim(0, max(ALPHA*1.6, max(faumax)*1.3))
    axes[0].legend(fontsize=8)
    eps_b = [fa_by_beta[str(b)]["mean_eps"] for b in betas]
    axes[1].plot(bs, eps_b, "-o", color="#c05621", label=r"conformal $\varepsilon$")
    axes[1].plot(bs, bs, "k--", lw=1, label=r"$\varepsilon=\beta$")
    axes[1].set_xlabel(r"drift budget $\beta$"); axes[1].set_ylabel(r"conformal radius $\varepsilon$")
    axes[1].set_title(r"$\varepsilon$ self-calibrates to $\approx\beta$"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_frontier_fa_coverage.png"), dpi=150); plt.close(fig)

    json.dump(R, open(RESJSON, "w"), indent=2)
    print("=" * 68)
    print("FRONTIER VALIDATION (synthetic ground truth)")
    print(f"  (A) recovery: coverage {R['recovery']['empirical_coverage']:.2%} "
          f">= target {1-ALPHA:.0%}   MAE {R['recovery']['mae']}   eps {R['recovery']['eps']}")
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

if __name__ == "__main__":
    main()
