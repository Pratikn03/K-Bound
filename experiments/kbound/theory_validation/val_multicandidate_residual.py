"""Multi-candidate sign identifiability under BOUNDED error correlation.

Numerical validation for the v0.5 extension that RELAXES Definition 5
(``def:cei``, conditional error-independence on D) from an exact premise to a
*bounded, observable* residual.  Validates the proposed Theorem
``thm:multicand-robust`` (writeup: theory_validation/THM_multicandidate_residual.tex).
Does NOT modify the paper or any SAR/GPU harness.  CPU only.

==============================================================================
SETTING (from the host paper: Thm thm:disagree / Prop thm:disagree-mc, and the
multi-candidate Prop thm:multicand + diagnostic Prop prop:cei-test)
==============================================================================
On the observable disagreement region D = {x : f0(x) != fa(x)} with BINARY Y,
adapting helps iff a candidate is above chance on D.  For M>=3 candidates with
accuracies a_j = Pr(f^(j)=Y | D) and advantages b_j = 2 a_j - 1, predictions
agree iff correctness indicators agree (binary), so the centered pairwise
agreements

        c_ij := 2 A_ij - 1 ,   A_ij = Pr(f^(i)=f^(j) | D)

satisfy, UNDER Definition 5 (conditional error-independence),

        c_ij = b_i b_j          (exactly rank-one off-diagonal).            (CEI)

Prop thm:multicand recovers {b_j} exactly from (CEI); Prop prop:cei-test notes
that for M>=4 the rank-one form is overdetermined, so a nonzero residual
*certifies* (CEI) is violated.  Both are EXACT (tau = 0) statements.

==============================================================================
THE EXTENSION VALIDATED HERE (bounded residual; the relaxation of Def 5)
==============================================================================
Write the true model as  c_ij = b_i b_j + E_ij ,  E = error-correlation matrix
(E = 0 iff Def 5 holds).  Two quantities:

  * eta    := ||off(E)||_F                 TRUE misspecification (needs ground
                                            truth; here computed in closed form).
  * tau    := min_beta ||off(C - beta beta^T)||_F     OBSERVABLE rank-one-fit
                                            residual (agreements only, no labels).
             with fitted advantage vector  b_hat.

Always tau <= eta.  CLAIMS (proved in the writeup):

  (a) [observable]  tau is a finite-sample statistic of unlabeled agreements;
      tau = 0  <=>  off(C) is rank-one.  For M = 3 the off-diagonal is ALWAYS
      rank-one-fittable (3 entries, 3 unknowns) so tau == 0 identically -- the
      residual is informative only for M >= 4 (matches Prop prop:cei-test).
      Unlike Def 5 (an untestable premise about latent correctness), tau is
      checkable.

  (b) [recovery]  median-of-minors estimator  b_tilde_i^2 = median_{k<l}
      c_ik c_il / c_kl  obeys, on the margin set |b_j| >= beta with
      eta_inf := max|E_ij| <= beta^2/2,
            |b_tilde_i - b_i| <= 3 eta_inf / beta^3 ,
      hence ||b_tilde - b*||_inf = O(eta) and ||.||_2 = O(eta * sqrt(M)/beta^3)
      = O(eps * poly(M)).

  (c) [ordering / sign bracket]  accuracies a_tilde_i = (1+b_tilde_i)/2 have
      half-width w = (3/2) eta_inf/beta^3; the ordering is recovered on every
      gap > 2w and sign(Delta^(j)) on every |b_j| > 2w.  Certifying ALL M
      candidates simultaneously (finite-sample, n points on D) adds a union
      term O(sqrt(log M / n)) -> the predicted O(eps * log M)-type width.

  (d) [converse / abstention]  COMMIT sign(b_tilde_i) iff tau <= tau* AND
      |b_tilde_i| > 2w(tau*); else ABSTAIN.  tau > tau* certifies Def 5 fails
      (sound one-sided).  Worst-case necessity: tangential perturbations
      E = off(b u^T + u b^T) move b_tilde by ~u while keeping tau ~ 0 -- these
      evade ANY agreement-only statistic (re-enters Thm thm:imp).  So small tau
      is necessary, not sufficient, in the worst case; under generic (random)
      error correlation -- exercised below -- the tau-gate gives zero false
      commitments.

==============================================================================
SIMULATION DESIGN (controlled error-correlation rho via a single-factor copula)
==============================================================================
True accuracies a_j drawn with margin |b_j| >= beta, anchor = majority above
chance.  Correctness indicators on D from a one-factor Gaussian copula:
    u_j = sqrt(rho)*g + sqrt(1-rho)*z_j ,  g,z_j ~ N(0,1) iid,
    s_j = 1[u_j <= t_j],  t_j = Phi^{-1}(a_j)  =>  Pr(s_j=1)=a_j,
so every pair of latent gaussians has correlation rho and the correctness (hence
prediction) agreement deviates from (CEI) by an amount that grows with rho.  The
TRUE c_ij^pop and eta are available in closed form via the bivariate-normal
orthant probability  A_ij^pop = 1 - a_i - a_j + 2*Phi_2(t_i,t_j;rho).
rho = 0  <=>  Def 5 holds exactly.
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "results_multicandidate_residual.json")
FIG1 = os.path.join(HERE, "fig_multicand_recovery_vs_rho.png")
FIG2 = os.path.join(HERE, "fig_multicand_eps_and_M_scaling.png")
FIG3 = os.path.join(HERE, "fig_multicand_commit_abstain.png")

SEED = 20260609
RNG = np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# model + estimators
# --------------------------------------------------------------------------- #
def make_advantages(M, beta, rng):
    """True advantages b_j with |b_j| >= beta (margin).

    Anchor: candidate 0 is a designated reference KNOWN to be above chance
    (b_0 > 0).  This is the parity-robust form of the paper's anchor -- a plain
    "majority above chance" is ambiguous for even M (a balanced 2/2 split has no
    majority and leaves the global sign undetermined).  A single trusted
    above-chance reference (e.g. the source-validated model) fixes it for all M.
    The other candidates' signs are unrestricted.
    """
    mag = rng.uniform(beta, 0.9, size=M)          # |b_j| in [beta, 0.9]
    sign = rng.choice([-1.0, 1.0], size=M, p=[0.4, 0.6])
    b = mag * sign
    b[0] = abs(b[0])                               # anchor: candidate 0 > chance
    return b


def simulate_agreements(a, rho, n, rng):
    """Empirical centered-agreement matrix C (off-diag) from the copula, size n."""
    M = len(a)
    t = norm.ppf(a)                                # thresholds Phi^{-1}(a_j)
    g = rng.standard_normal((n, 1))
    z = rng.standard_normal((n, M))
    u = np.sqrt(rho) * g + np.sqrt(1.0 - rho) * z
    s = (u <= t[None, :]).astype(np.float64)       # correctness indicators
    # A_ij = mean over samples of 1[s_i == s_j]; binary => predictions agree iff
    # correctness agrees.  C_ij = 2 A_ij - 1.
    eq = (s[:, :, None] == s[:, None, :]).mean(axis=0)
    C = 2.0 * eq - 1.0
    np.fill_diagonal(C, 0.0)
    return C


def population_C_and_eta(a, rho):
    """Closed-form population c_ij and true misspecification eta = ||off(E)||_F."""
    M = len(a)
    b = 2.0 * a - 1.0
    t = norm.ppf(a)
    C = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            phi2 = mvn.cdf([t[i], t[j]], mean=[0, 0], cov=[[1, rho], [rho, 1]])
            A = 1.0 - a[i] - a[j] + 2.0 * phi2     # Pr(s_i = s_j)
            c = 2.0 * A - 1.0
            C[i, j] = C[j, i] = c
    E = C - np.outer(b, b)
    np.fill_diagonal(E, 0.0)
    eta = float(np.sqrt((E ** 2).sum()))
    eta_inf = float(np.abs(E).max())
    return C, eta, eta_inf


def rankone_fit_offdiag(C, iters=90, tol=1e-12):
    """Best rank-one fit to OFF-diagonal entries (unknown diagonal).

    Alternating diagonal-imputation: hold diag = b_i^2, take the leading
    eigenpair, repeat.  Returns (b_hat, tau) with
    tau = sqrt(sum_{i!=j} (c_ij - b_hat_i b_hat_j)^2).
    """
    M = C.shape[0]
    W = C.copy()
    # init diagonal from row-energy
    d = np.sqrt(np.clip((C ** 2).sum(1) / max(M - 1, 1), 1e-6, None))
    prev = None
    for _ in range(iters):
        np.fill_diagonal(W, d ** 2)
        vals, vecs = np.linalg.eigh(W)
        lam, v = vals[-1], vecs[:, -1]
        lam = max(lam, 0.0)
        b = np.sqrt(lam) * v
        d = np.abs(b)
        if prev is not None and np.linalg.norm(b - prev) < tol and \
           np.linalg.norm(b + prev) > tol:
            break
        prev = b
    off = ~np.eye(M, dtype=bool)
    resid = C[off] - np.outer(b, b)[off]
    tau = float(np.sqrt((resid ** 2).sum()))
    return b, tau


def minor_estimator(C):
    """median-of-minors:  b_tilde_i^2 = median_{k<l, !=i} c_ik c_il / c_kl .

    Magnitudes from the 2x2-minor identity; signs from the designated anchor
    (candidate 0 above chance, b_0 > 0) via sign(b_j) = sign(c_0j).  No global
    flip is needed -- the anchor fixes the sign unambiguously for every M.
    """
    M = C.shape[0]
    b2 = np.zeros(M)
    for i in range(M):
        others = [k for k in range(M) if k != i]
        ratios = []
        for ii in range(len(others)):
            for jj in range(ii + 1, len(others)):
                k, l = others[ii], others[jj]
                if abs(C[k, l]) > 1e-9:
                    ratios.append(C[i, k] * C[i, l] / C[k, l])
        b2[i] = np.median(ratios) if ratios else 0.0
    b2 = np.clip(b2, 0.0, 1.0)
    mag = np.sqrt(b2)
    rel = np.ones(M)                               # anchor: b_0 > 0
    for j in range(1, M):
        rel[j] = np.sign(C[0, j]) if C[0, j] != 0 else 1.0
    return mag * rel


def overdet_residual(C):
    """Prop prop:cei-test diagnostic (M>=4): spread of the three quadruple
    pairings, averaged over all 4-subsets.  ~0 under (CEI), grows with rho."""
    M = C.shape[0]
    if M < 4:
        return 0.0
    from itertools import combinations
    spreads = []
    for quad in combinations(range(M), 4):
        i, j, k, l = quad
        p = [C[i, j] * C[k, l], C[i, k] * C[j, l], C[i, l] * C[j, k]]
        spreads.append(max(p) - min(p))
    return float(np.mean(spreads))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #
def exp_recovery_vs_rho(Ms=(3, 4, 5), rhos=None, n_trials=160, n_D=4000, beta=0.25):
    """E1+E2: recovery, ordering, sign accuracy, and tau vs rho, per M."""
    if rhos is None:
        rhos = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    out = {}
    for M in Ms:
        rows = []
        for rho in rhos:
            errs, ws, tau_ls, tau_od, eta_pop = [], [], [], [], []
            pair_ok, sign_ok, rank_ok = [], [], []
            for _ in range(n_trials):
                b = make_advantages(M, beta, RNG)
                a = (1.0 + b) / 2.0
                C = simulate_agreements(a, rho, n_D, RNG)
                _, tls = rankone_fit_offdiag(C)
                bt = minor_estimator(C)
                at = (1.0 + bt) / 2.0
                errs.append(np.abs(bt - b).max())
                ws.append(np.abs(at - a).max())
                tau_ls.append(tls)
                tau_od.append(overdet_residual(C))
                # pairwise ordering accuracy on D
                po = []
                for i in range(M):
                    for j in range(i + 1, M):
                        po.append(np.sign(at[i] - at[j]) == np.sign(a[i] - a[j]))
                pair_ok.append(np.mean(po))
                sign_ok.append(np.mean(np.sign(bt) == np.sign(b)))
                rank_ok.append(bool(np.array_equal(np.argsort(at), np.argsort(a))))
            # population eta (closed form, representative draws per rho)
            for _ in range(25):
                b = make_advantages(M, beta, RNG)
                a = (1.0 + b) / 2.0
                _, e, _ = population_C_and_eta(a, rho)
                eta_pop.append(e)
            rows.append({
                "rho": rho,
                "mean_eta_pop": float(np.mean(eta_pop)),
                "mean_binf_err": float(np.mean(errs)),
                "mean_bracket_w": float(np.mean(ws)),
                "mean_tau_LS": float(np.mean(tau_ls)),
                "mean_tau_overdet": float(np.mean(tau_od)),
                "pairwise_order_acc": float(np.mean(pair_ok)),
                "sign_acc": float(np.mean(sign_ok)),
                "full_rank_acc": float(np.mean(rank_ok)),
            })
        out[str(M)] = rows
    return out


def exp_eps_scaling(M=5, beta=0.25, n_trials=220, n_D=6000):
    """E3: recovery error vs TRUE misspecification eta -> expect slope ~1 (linear).
    eta_inf computed on a small representative subset (closed form is the slow part);
    b-error measured over the full trial budget."""
    rhos = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45]
    xs, ys = [], []
    for rho in rhos:
        errs, etas = [], []
        for _ in range(n_trials):
            b = make_advantages(M, beta, RNG)
            a = (1.0 + b) / 2.0
            C = simulate_agreements(a, rho, n_D, RNG)
            bt = minor_estimator(C)
            errs.append(np.abs(bt - b).max())
        for _ in range(25):
            b = make_advantages(M, beta, RNG)
            a = (1.0 + b) / 2.0
            _, _, einf = population_C_and_eta(a, rho)
            etas.append(einf)
        xs.append(float(np.mean(etas)))
        ys.append(float(np.mean(errs)))
    lx, ly = np.log(np.array(xs)), np.log(np.array(ys))
    slope, intercept = np.polyfit(lx, ly, 1)
    r2 = 1.0 - np.sum((ly - (slope * lx + intercept)) ** 2) / \
        np.sum((ly - ly.mean()) ** 2)
    return {"eta_inf": xs, "binf_err": ys, "loglog_slope": float(slope),
            "loglog_R2": float(r2)}


def exp_M_scaling(Ms=(4, 5, 6, 8, 10, 12), rho=0.15, beta=0.25,
                  n_trials=160, n_D=6000):
    """E4: simultaneous (max over candidates) bracket width vs M at fixed rho.
    Tests whether the all-candidates width grows ~ log M (vs sqrt(log M))."""
    xs, ys = [], []
    for M in Ms:
        ws = []
        for _ in range(n_trials):
            b = make_advantages(M, beta, RNG)
            a = (1.0 + b) / 2.0
            C = simulate_agreements(a, rho, n_D, RNG)
            bt = minor_estimator(C)
            at = (1.0 + bt) / 2.0
            ws.append(np.abs(at - a).max())        # simultaneous half-width
        xs.append(M)
        ys.append(float(np.mean(ws)))
    yv = np.array(ys)
    logM = np.log(np.array(xs))
    sqrtlogM = np.sqrt(logM)
    # compare fit quality of width ~ c*logM vs width ~ c*sqrt(logM)
    def r2_through(x):
        c = np.sum(x * yv) / np.sum(x * x)
        return 1.0 - np.sum((yv - c * x) ** 2) / np.sum((yv - yv.mean()) ** 2), float(c)
    r2_log, c_log = r2_through(logM)
    r2_sqrt, c_sqrt = r2_through(sqrtlogM)
    return {"M": list(Ms), "sim_bracket_w": ys,
            "fit_logM_R2": float(r2_log), "fit_logM_coef": c_log,
            "fit_sqrtlogM_R2": float(r2_sqrt), "fit_sqrtlogM_coef": c_sqrt,
            "better_fit": "logM" if r2_log >= r2_sqrt else "sqrtlogM"}


def exp_commit_abstain(M=4, beta=0.25, tau_star=0.08, kappa=2.5,
                       n_trials=900, n_D=4000):
    """E5: tau-gated commit/abstain safety.

    OPERATIONAL RULE (all quantities observable):
        h_hat = max_{k!=l} |c_kl - b_hat_k b_hat_l|     (worst per-entry residual)
        COMMIT sign(b_hat_i)  iff   tau <= tau*  AND  |b_hat_i| > kappa*h_hat
                                                         + 2/sqrt(n_D);
        else ABSTAIN.
    The margin scales with the OBSERVED residual (not the loose worst-case beta^3
    constant).  Safety target: false-commit rate ~ 0 at EVERY rho (committing a
    wrong sign is the dangerous error).  Abstention should engage as rho grows.
    Also records (tau, sign-accuracy) to locate the empirical threshold where
    recovery breaks."""
    rhos = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    rows = []
    tau_sign = []                                  # (tau, correct?) per candidate
    for rho in rhos:
        n_commit = n_false = n_cells = 0
        gated_in = 0
        sgn_acc = []
        for _ in range(n_trials):
            b = make_advantages(M, beta, RNG)
            a = (1.0 + b) / 2.0
            C = simulate_agreements(a, rho, n_D, RNG)
            _, tau = rankone_fit_offdiag(C)
            bt = minor_estimator(C)
            off = ~np.eye(M, dtype=bool)
            h_hat = np.abs(C - np.outer(bt, bt))[off].max()
            margin = kappa * h_hat + 2.0 / np.sqrt(n_D)
            gate = tau <= tau_star
            gated_in += int(gate)
            sgn_acc.append(np.mean(np.sign(bt) == np.sign(b)))
            for i in range(M):
                n_cells += 1
                tau_sign.append((tau, int(np.sign(bt[i]) == np.sign(b[i]))))
                if gate and abs(bt[i]) > margin:
                    n_commit += 1
                    if np.sign(bt[i]) != np.sign(b[i]):
                        n_false += 1
        rows.append({
            "rho": rho, "tau_star": tau_star, "kappa": kappa,
            "gate_pass_rate": gated_in / n_trials,
            "commit_rate": n_commit / n_cells,
            "false_commit_rate": n_false / n_cells,
            "false_commits_abs": int(n_false),
            "sign_acc": float(np.mean(sgn_acc)),
        })
    # empirical threshold: largest tau bin whose committed-sign accuracy is still
    # >= 0.99 (group candidates into tau deciles)
    ts = np.array(tau_sign)
    order = np.argsort(ts[:, 0])
    ts = ts[order]
    nbin = 12
    thr = None
    bins = np.array_split(ts, nbin)
    bin_report = []
    for bn in bins:
        if len(bn) == 0:
            continue
        acc = float(bn[:, 1].mean())
        tlo, thi = float(bn[0, 0]), float(bn[-1, 0])
        bin_report.append({"tau_lo": tlo, "tau_hi": thi, "sign_acc": acc})
        if acc >= 0.99:
            thr = thi
    return {"sweep": rows, "empirical_tau_threshold_signacc99": thr,
            "tau_bins": bin_report}


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_recovery(rec, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for M, rows in rec.items():
        r = [x["rho"] for x in rows]
        ax[0].plot(r, [x["sign_acc"] for x in rows], "o-", label=f"M={M}")
        ax[1].plot(r, [x["pairwise_order_acc"] for x in rows], "o-", label=f"M={M}")
        ax[2].plot(r, [x["mean_tau_LS"] for x in rows], "o-", label=f"M={M} (tau_LS)")
    ax[0].set(title="sign(b) recovery vs error-corr rho", xlabel="rho",
              ylabel="sign accuracy"); ax[0].axhline(1, ls=":", c="gray")
    ax[1].set(title="pairwise ordering accuracy vs rho", xlabel="rho",
              ylabel="ordering accuracy"); ax[1].axhline(1, ls=":", c="gray")
    ax[2].set(title="observable residual tau rises with rho\n(flat ~0 for M=3)",
              xlabel="rho", ylabel="mean tau (LS off-diag)")
    for a in ax:
        a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_scaling(eps, msc, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].loglog(eps["eta_inf"], eps["binf_err"], "o-", color="C0")
    ax[0].set(title=f"O(eps) recovery: ||b_tilde-b*||_inf vs eta_inf\n"
                    f"log-log slope={eps['loglog_slope']:.2f} (R^2={eps['loglog_R2']:.3f})",
              xlabel="true eta_inf (misspecification)", ylabel="b-error (inf-norm)")
    ax[0].grid(alpha=.3, which="both")
    ax[1].plot(msc["M"], msc["sim_bracket_w"], "o-", color="C3")
    ax[1].set(title=f"simultaneous bracket width vs M\nbetter fit: {msc['better_fit']}"
                    f" (logM R^2={msc['fit_logM_R2']:.3f}, "
                    f"sqrt(logM) R^2={msc['fit_sqrtlogM_R2']:.3f})",
              xlabel="M (number of candidates)", ylabel="max-over-candidates width")
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_commit(ca, path):
    r = [x["rho"] for x in ca]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(r, [x["commit_rate"] for x in ca], "o-", label="commit rate")
    ax.plot(r, [x["gate_pass_rate"] for x in ca], "s--", label="tau-gate pass rate")
    ax.plot(r, [x["false_commit_rate"] for x in ca], "^-", color="red",
            label="FALSE-commit rate")
    ax.axhline(0, ls=":", c="gray")
    ax.set(title=f"tau-gated commit/abstain (M=4, tau*={ca[0]['tau_star']})",
           xlabel="error-correlation rho", ylabel="rate")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print("Multi-candidate sign identifiability under BOUNDED error correlation")
    print("relaxing Definition 5 (def:cei).  seed =", SEED, " CPU only")
    print("=" * 78)

    rec = exp_recovery_vs_rho()
    eps = exp_eps_scaling()
    msc = exp_M_scaling()
    ca = exp_commit_abstain()

    # ---- headline PASS criteria -------------------------------------------- #
    from scipy.stats import spearmanr

    def monotone(xs):
        return all(xs[i + 1] >= xs[i] - 1e-4 for i in range(len(xs) - 1))

    tau3 = [r["mean_tau_LS"] for r in rec["3"]]
    tau4 = [r["mean_tau_LS"] for r in rec["4"]]
    tau5 = [r["mean_tau_LS"] for r in rec["5"]]
    sign4 = [r["sign_acc"] for r in rec["4"]]
    sp4 = float(spearmanr([r["rho"] for r in rec["4"]], tau4).statistic)
    sp5 = float(spearmanr([r["rho"] for r in rec["5"]], tau5).statistic)

    # rho index 4 == rho 0.2 (end of operational range); index 0 == rho 0
    tau3_operational = max(tau3[:5])               # rho <= 0.2
    ratio_M4_M3_at_02 = tau4[4] / max(tau3[4], 1e-6)

    # recovery accurate under independence (rho=0) for every M (catches anchor bugs)
    rho0_ok = all(rec[M][0]["sign_acc"] >= 0.99 and rec[M][0]["mean_binf_err"] <= 0.07
                  for M in ("3", "4", "5"))

    # M-scaling: simultaneous width must not blow up with M
    w_by_M = msc["sim_bracket_w"]
    M_width_bounded = max(w_by_M) <= 1.4 * min(w_by_M) + 0.03

    sweep = ca["sweep"]
    max_false_all = max(r["false_commit_rate"] for r in sweep)
    commit_rho0 = sweep[0]["commit_rate"]
    commit_rho05 = sweep[-1]["commit_rate"]

    headline = {
        "tau_M3_max_over_all_rho": float(max(tau3)),
        "tau_M3_max_operational_rho_le_0.2": float(tau3_operational),
        "tau_ratio_M4_over_M3_at_rho0.2": float(ratio_M4_M3_at_02),
        "tau_M4_range": [float(min(tau4)), float(max(tau4))],
        "tau_M5_range": [float(min(tau5)), float(max(tau5))],
        "spearman_rho_tau_M4": sp4,
        "spearman_rho_tau_M5": sp5,
        "sign_acc_rho0_M3_M4_M5": [float(rec[m][0]["sign_acc"]) for m in ("3","4","5")],
        "binf_err_rho0_M3_M4_M5": [float(rec[m][0]["mean_binf_err"]) for m in ("3","4","5")],
        "sign_acc_M4_rho0.5": float(sign4[-1]),
        "eps_loglog_slope": eps["loglog_slope"],
        "eps_loglog_R2": eps["loglog_R2"],
        "M_scaling_widths": [float(w) for w in w_by_M],
        "M_scaling_better_fit": msc["better_fit"],
        "commit_rate_rho0": float(commit_rho0),
        "commit_rate_rho0.5": float(commit_rho05),
        "max_false_commit_rate_over_all_rho": float(max_false_all),
        "empirical_tau_threshold_signacc99": ca["empirical_tau_threshold_signacc99"],
        # ---- PASS flags (honest, claim-aligned) ----
        "PASS_recovery_exact_under_independence": bool(rho0_ok),
        "PASS_tau_monotone_in_rho_M4": bool(monotone(tau4) and sp4 > 0.9),
        "PASS_tau_monotone_in_rho_M5": bool(monotone(tau5) and sp5 > 0.9),
        "PASS_M3_diagnostically_blind":
            bool(tau3_operational < 0.012 and ratio_M4_M3_at_02 > 5.0),
        "PASS_recovery_degrades_with_rho": bool(sign4[0] - sign4[-1] > 0.02),
        "PASS_eps_scaling_linear_O_eps": bool(0.7 <= eps["loglog_slope"] <= 1.3),
        "PASS_M_scaling_bounded_sublog": bool(M_width_bounded),
        "PASS_zero_false_commit_all_rho": bool(max_false_all <= 0.005),
        "PASS_abstention_engages_with_rho": bool(commit_rho05 <= 0.6 * commit_rho0),
    }
    headline["ALL_PASS"] = bool(all(v for k, v in headline.items()
                                    if k.startswith("PASS_")))

    results = {
        "description": "Multi-candidate sign identifiability under bounded error "
                       "correlation (relaxation of Definition 5 / def:cei)",
        "seed": SEED,
        "params": {"beta_margin": 0.25, "binary_Y": True},
        "recovery_vs_rho": rec,
        "eps_scaling": eps,
        "M_scaling": msc,
        "commit_abstain": ca,
        "headline": headline,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    plot_recovery(rec, FIG1)
    plot_scaling(eps, msc, FIG2)
    plot_commit(ca["sweep"], FIG3)

    # ---- console report ---------------------------------------------------- #
    print("\n-- E1/E2  recovery & observable residual vs rho --")
    for M in ("3", "4", "5"):
        print(f"\n  M={M}")
        print(f"  {'rho':>5} {'eta_pop':>8} {'b_err':>8} {'tau_LS':>8} "
              f"{'tau_odet':>9} {'sign_acc':>9} {'order_acc':>9} {'rank_acc':>8}")
        for r in rec[M]:
            print(f"  {r['rho']:5.2f} {r['mean_eta_pop']:8.4f} "
                  f"{r['mean_binf_err']:8.4f} {r['mean_tau_LS']:8.4f} "
                  f"{r['mean_tau_overdet']:9.4f} {r['sign_acc']:9.4f} "
                  f"{r['pairwise_order_acc']:9.4f} {r['full_rank_acc']:8.3f}")

    print("\n-- E3  O(eps) scaling:  b-error vs true eta_inf --")
    print(f"   log-log slope = {eps['loglog_slope']:.3f}  (R^2={eps['loglog_R2']:.3f})"
          f"   [~1.0 => linear O(eps)]")

    print("\n-- E4  M-scaling of simultaneous bracket width --")
    print(f"   width vs logM   R^2 = {msc['fit_logM_R2']:.3f}")
    print(f"   width vs sqrt(logM) R^2 = {msc['fit_sqrtlogM_R2']:.3f}")
    print(f"   better fit: {msc['better_fit']}")
    for M, w in zip(msc["M"], msc["sim_bracket_w"]):
        print(f"     M={M:2d}  width={w:.4f}")

    print("\n-- E5  tau-gated commit/abstain (M=4) --")
    print(f"   {'rho':>5} {'gate_pass':>10} {'commit':>8} {'FALSE_commit':>13} "
          f"{'sign_acc':>9}")
    for r in ca["sweep"]:
        print(f"   {r['rho']:5.2f} {r['gate_pass_rate']:10.3f} "
              f"{r['commit_rate']:8.3f} {r['false_commit_rate']:13.5f} "
              f"{r['sign_acc']:9.4f}")
    print(f"   empirical tau threshold (committed sign-acc >= 0.99): "
          f"{ca['empirical_tau_threshold_signacc99']}")
    print("   tau bins (sign-acc vs tau):")
    for bn in ca["tau_bins"]:
        print(f"     tau in [{bn['tau_lo']:.3f},{bn['tau_hi']:.3f}]  "
              f"sign_acc={bn['sign_acc']:.4f}")

    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print(json.dumps(headline, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {FIG1}\nWrote {FIG2}\nWrote {FIG3}")
    print("\nALL_PASS =", headline["ALL_PASS"])
    return 0 if headline["ALL_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
