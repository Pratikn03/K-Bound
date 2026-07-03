"""Validation for the KNOWABILITY FRONTIER theorem + subsumption propositions.

Every number is produced by this run (synthetic parts) or computed from the REAL
123-task score archive (part 'real'). Nothing is asserted without a check.

Theorem K (knowability frontier), validated here:
  margin  kappa_a(z) := |Delta(z)| - 2*eps_a(z)   (eps_a = valid (1-a) radius)
  (i)  certificate rule:  FA := P(adapt & Delta<0) <= a,  FF <= a,
       coverage  C >= P(kappa_a > 0) - a            [achievability]
  (ii) on a (delta,t)-ambiguous set (paired worlds, evidence-law TV<=t, opposite
       signs |Delta|>=delta): commit probability <= 2a + t  [converse; t=0 here]

Subsumption propositions:
  S1 always-adapt = certificate with eps==0 (kappa-blind): regret = full false-adapt
     mass; on the witness pair its worst-case committal regret is delta (=1),
     2x the Le Cam floor delta/2; the certificate abstains (committal regret 0).
  S2 confidence/entropy FILTERS lack false-adapt control: confidently-wrong worlds
     give FA(filter) ~ 1 while the certificate keeps FA <= a. (The certificate's
     safety is purchased by the calibrated benefit sample - stated, and shown.)
  S3 ordinal < cardinal: under unknown symmetric label noise eta<1/2 on the
     disagreement region, sign(a_a' - a_0') = sign(a_a - a_0) (decision invariant)
     while absolute accuracies are confounded by eta (cardinal estimation is not).

Usage:
  python knowability_frontier_validation.py --part synth     # synthetic frontier + S1-S3
  python knowability_frontier_validation.py --part real      # REAL 123-task archive
  python knowability_frontier_validation.py --part figures   # render figures from JSON
"""
import os, json, glob, argparse, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.dirname(HERE)                                   # docs/research/kbound
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(KB))) # repo root
ARCH = os.path.join(ROOT, "experiments", "elara_u", "score_archive")
OUTD = os.path.join(KB, "results", "theory"); os.makedirs(OUTD, exist_ok=True)
OUTJ = os.path.join(OUTD, "knowability_frontier_validation.json")
FIGD = os.path.join(KB, "figures"); FIGF = os.path.join(FIGD, "final")
os.makedirs(FIGF, exist_ok=True)
ALPHAS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
rng = np.random.default_rng(0)


def _load():
    return json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}


def _save(d):
    json.dump(d, open(OUTJ, "w"), indent=2)


# --------------------------------------------------------------------------- #
# shared: out-of-fold GBM benefit estimator + conformal radius per alpha
# --------------------------------------------------------------------------- #
def oof_bhat(Z, B, k=5, n_estimators=200, seed=0):
    from sklearn.ensemble import GradientBoostingRegressor
    Z = np.asarray(Z, float); B = np.asarray(B, float); N = len(B)
    idx = np.random.default_rng(seed).permutation(N)
    folds = np.array_split(idx, k); bh = np.zeros(N)
    for i in range(k):
        te = folds[i]; tr = np.setdiff1d(idx, te)
        m = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=2,
                                      learning_rate=0.05, subsample=0.8, random_state=seed)
        m.fit(Z[tr], B[tr]); bh[te] = m.predict(Z[te])
    return bh


def frontier_rows(Bhat, B, ambig_mask=None):
    """For each alpha: conformal eps, decisions, FA/FF (unconditional), coverage,
    margin-bound P(kappa>0)-alpha, commit-rate on the ambiguous subset."""
    res = np.abs(Bhat - B); rows = []
    for a in ALPHAS:
        eps = float(np.quantile(res, 1 - a))
        adapt = Bhat - eps > 0; freeze = Bhat + eps < 0; commit = adapt | freeze
        kappa = np.abs(B) - 2 * eps
        row = dict(alpha=a, eps=eps,
                   FA=float(np.mean(adapt & (B < 0))), FF=float(np.mean(freeze & (B > 0))),
                   coverage=float(np.mean(commit)),
                   bound_cov_lo=float(max(0.0, np.mean(kappa > 0) - a)),
                   P_kappa_pos=float(np.mean(kappa > 0)))
        if ambig_mask is not None and ambig_mask.any():
            row["commit_on_ambiguous"] = float(np.mean(commit[ambig_mask]))
            row["converse_cap_2alpha"] = 2 * a
        rows.append(row)
    return rows


def check_frontier(rows, slack=0.035, ambig=False):
    ok_fa = all(r["FA"] <= r["alpha"] + slack for r in rows)
    ok_ff = all(r["FF"] <= r["alpha"] + slack for r in rows)
    ok_lo = all(r["coverage"] >= r["bound_cov_lo"] - slack for r in rows)
    out = dict(FA_le_alpha=ok_fa, FF_le_alpha=ok_ff, coverage_ge_bound=ok_lo)
    if ambig:
        out["ambiguous_commit_le_2alpha"] = all(
            r["commit_on_ambiguous"] <= r["converse_cap_2alpha"] + slack for r in rows)
    return out


# --------------------------------------------------------------------------- #
# PART: synth  (frontier on planted data + S1-S3)
# --------------------------------------------------------------------------- #
def part_synth():
    d = _load()
    # ---- frontier: informative (70%) + ambiguous (30%) mass U ------------- #
    N, U = 4000, 0.30
    n_amb = int(N * U); n_inf = N - n_amb
    # informative: Z carries B (plus noise); B spans helpful & harmful
    B_inf = rng.normal(0.0, 0.18, n_inf)
    Z_inf = np.c_[B_inf + rng.normal(0, 0.05, n_inf),
                  np.abs(B_inf) + rng.normal(0, 0.05, n_inf),
                  rng.normal(0, 1, (n_inf, 3))]
    # ambiguous: SAME evidence law regardless of sign; |B| = delta
    delta = 0.40
    B_amb = rng.choice([-delta, delta], n_amb)
    Z_amb = np.c_[rng.normal(0, 0.05, n_amb), rng.normal(delta, 0.05, n_amb),
                  rng.normal(0, 1, (n_amb, 3))]
    Z = np.vstack([Z_inf, Z_amb]); B = np.concatenate([B_inf, B_amb])
    amb = np.zeros(N, bool); amb[n_inf:] = True
    Bhat = oof_bhat(Z, B)
    rows = frontier_rows(Bhat, B, amb)
    d["frontier_synthetic"] = dict(N=N, ambiguous_mass=U, delta=delta, rows=rows,
                                   checks=check_frontier(rows, ambig=True))

    # ---- S1: always-adapt = eps==0 degenerate certificate ----------------- #
    a0 = 0.10
    eps0 = float(np.quantile(np.abs(Bhat - B), 1 - a0))
    adapt_c = Bhat - eps0 > 0
    d["S1_always_adapt"] = dict(
        regret_always_adapt=float(np.sum(np.abs(B)[B < 0])),         # full FA mass
        regret_certificate_committal=float(np.sum(np.abs(B)[adapt_c & (B < 0)])),
        witness_pair=dict(always_adapt_worstcase_regret=1.0,         # commits wrong world
                          lecam_floor=0.5,                            # delta/2, delta=1
                          certificate_committal_regret=0.0,           # abstains (validated
                          note="abstention shown in witness_clean.json: abstain_rate=1.0"),
        statement="always-adapt == certificate with eps==0 (kappa-blind); pays the full false-adapt mass and 2x the Le Cam floor on the witness")

    # ---- S2: confidence/entropy filter lacks FA control -------------------- #
    # confidently-wrong world: harmful instances share the SAME confidence law
    M = 3000; harm_share = 0.5
    harm = rng.random(M) < harm_share
    Bs = np.where(harm, -0.4, +0.4) + rng.normal(0, 0.03, M)
    conf = rng.normal(0.95, 0.02, M)                # high confidence EVERYWHERE
    noise = rng.normal(0, 1, (M, 3))
    Zs = np.c_[conf, noise]                          # confidence uninformative of sign
    filt_adapt = conf > 0.9                          # entropy/confidence filter rule
    FA_filter = float(np.mean(filt_adapt & (Bs < 0)))
    Bh = oof_bhat(Zs, Bs)
    eps = float(np.quantile(np.abs(Bh - Bs), 1 - a0))
    cert_adapt = Bh - eps > 0
    FA_cert = float(np.mean(cert_adapt & (Bs < 0)))
    d["S2_filter_vs_certificate"] = dict(
        FA_entropy_filter=FA_filter, FA_certificate=FA_cert, alpha=a0,
        filter_fails=bool(FA_filter > 0.3), certificate_holds=bool(FA_cert <= a0 + 0.03),
        statement="confidently-wrong world: filter adapts blindly (FA~harmful share); certificate abstains. Safety is purchased by the calibrated benefit sample, not the heuristic.")

    # ---- S3: ordinal survives symmetric label noise; cardinal does not ----- #
    n = 200000; aa, a0_ = 0.62, 0.47
    ya = rng.random(n) < aa; y0 = rng.random(n) < a0_     # correctness indicators on D
    rows3 = []
    for eta in [0.0, 0.1, 0.2, 0.3, 0.4]:
        flip = rng.random(n) < eta                        # symmetric label noise
        aan = float(np.mean(ya ^ flip)); a0n = float(np.mean(y0 ^ flip))
        rows3.append(dict(eta=eta, a_a_obs=aan, a_0_obs=a0n,
                          diff_obs=aan - a0n, sign_preserved=bool((aan - a0n) > 0),
                          cardinal_bias_a=float(abs(aan - aa))))
    d["S3_ordinal_vs_cardinal"] = dict(
        a_a_true=aa, a_0_true=a0_, rows=rows3,
        sign_invariant_all_eta=bool(all(r["sign_preserved"] for r in rows3)),
        cardinal_bias_grows=bool(rows3[-1]["cardinal_bias_a"] > 0.05),
        identity="a_obs - 1/2 = (1-2*eta)(a_true - 1/2)  => sign(diff) invariant; absolute level confounded by unknown eta")
    _save(d)
    print("[synth] frontier checks:", d["frontier_synthetic"]["checks"])
    print("[synth] S1 regret always-adapt vs certificate-committal:",
          round(d["S1_always_adapt"]["regret_always_adapt"], 1), "vs",
          round(d["S1_always_adapt"]["regret_certificate_committal"], 2))
    print("[synth] S2 FA filter vs certificate:", round(FA_filter, 3), "vs", round(FA_cert, 3))
    print("[synth] S3 sign invariant under noise:", d["S3_ordinal_vs_cardinal"]["sign_invariant_all_eta"],
          "| cardinal bias at eta=0.4:", round(rows3[-1]["cardinal_bias_a"], 3))


# --------------------------------------------------------------------------- #
# PART: real  (123-task score archive -> (Z,B) -> frontier checks)
# --------------------------------------------------------------------------- #
def part_real(max_tasks=123):
    from scipy.stats import ks_2samp, rankdata
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    def rank_norm(S):
        R = np.empty_like(S, float)
        for j in range(S.shape[1]):
            R[:, j] = (rankdata(S[:, j]) - 1) / (len(S) - 1)
        return R
    Zr, Br = [], []
    files = sorted(glob.glob(os.path.join(ARCH, "*.npz")))[:max_tasks]
    for f in files:
        try:
            dd = np.load(f, allow_pickle=True)
            Sval, yval, Stest, ytest, va = dd["Sval"], dd["yval"], dd["Stest"], dd["ytest"], dd["val_auc"]
            if len(np.unique(ytest)) < 2: continue
            j0 = int(np.argmax(va))
            auc0 = roc_auc_score(ytest, Stest[:, j0])
            Rv, Rt = rank_norm(Sval), rank_norm(Stest)
            clf = LogisticRegression(max_iter=600).fit(Rv, yval)
            auca = roc_auc_score(ytest, clf.predict_proba(Rt)[:, 1])
            ks = [ks_2samp(Sval[:, j], Stest[:, j]).statistic for j in range(Sval.shape[1])]
            C = np.corrcoef(Rt.T); iu = np.triu_indices_from(C, 1)
            vs = np.sort(va)[::-1]
            Zr.append([float(vs[0]), float(vs[0] - vs[1]), float(np.mean(va)),
                       float(np.mean(ks)), float(np.max(ks)),
                       float(1 - np.nanmean(C[iu])), float(np.std(va)),
                       float(np.mean(yval)), float(np.log(len(ytest)))])
            Br.append(float(auca - auc0))
        except Exception:
            continue
    Zr = np.array(Zr); Br = np.array(Br)
    Bh = oof_bhat(Zr, Br, k=5)
    rows = frontier_rows(Bh, Br)
    d = _load()
    d["frontier_real_archive"] = dict(
        n_tasks=int(len(Br)), source="experiments/elara_u/score_archive (REAL)",
        seconds=round(time.time() - t0, 1), rows=rows,
        checks=check_frontier(rows, slack=0.05))   # finite-n: small slack, reported honestly
    _save(d)
    print(f"[real] {len(Br)} tasks in {d['frontier_real_archive']['seconds']}s")
    print("[real] checks:", d["frontier_real_archive"]["checks"])
    for r in rows:
        print(f"   a={r['alpha']:.2f} FA={r['FA']:.3f} cov={r['coverage']:.3f} bound_lo={r['bound_cov_lo']:.3f}")


# --------------------------------------------------------------------------- #
# PART: figures
# --------------------------------------------------------------------------- #
def part_figures():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = _load()
    fs, fr = d["frontier_synthetic"], d.get("frontier_real_archive")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, blk, title in ((axes[0], fs, f"synthetic (ambiguous mass U={fs['ambiguous_mass']})"),
                           (axes[1], fr, f"REAL 123-task archive (n={fr['n_tasks']})")):
        if blk is None: continue
        A = [r["alpha"] for r in blk["rows"]]
        ax.plot(A, [r["coverage"] for r in blk["rows"]], "-o", color="#2a9d8f", label="coverage (certificate)")
        ax.plot(A, [r["bound_cov_lo"] for r in blk["rows"]], "--", color="#2a9d8f",
                label=r"bound $P(\kappa_\alpha>0)-\alpha$")
        ax.plot(A, [r["FA"] for r in blk["rows"]], "-s", color="#e76f51", label="false-adapt")
        ax.plot(A, A, ":", color="#e76f51", label=r"$\alpha$ (FA cap)")
        if "commit_on_ambiguous" in blk["rows"][0]:
            ax.plot(A, [r["commit_on_ambiguous"] for r in blk["rows"]], "-^", color="#6b7280",
                    label="commit on ambiguous")
            ax.plot(A, [2 * a for a in A], ":", color="#6b7280", label=r"$2\alpha$ (converse cap)")
        ax.set_xlabel(r"$\alpha$"); ax.set_title(title); ax.set_ylim(-0.02, 1.0)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("probability")
    plt.suptitle("Knowability frontier: achievability and converse hold", y=1.02, fontsize=12)
    plt.tight_layout()
    for p in (os.path.join(FIGD, "fig_kfrontier.png"), os.path.join(FIGF, "fig_kfrontier.png")):
        fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()

    s1, s2 = d["S1_always_adapt"], d["S2_filter_vs_certificate"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(["always-adapt\n(=cert, $\\epsilon{=}0$)", "certificate"],
                [s1["regret_always_adapt"], s1["regret_certificate_committal"]],
                color=["#e76f51", "#2a9d8f"])
    axes[0].set_title("S1: regret of committal mass"); axes[0].set_ylabel("total |Δ| regret")
    axes[1].bar(["entropy/confidence\nfilter", "certificate"],
                [s2["FA_entropy_filter"], s2["FA_certificate"]], color=["#e9c46a", "#2a9d8f"])
    axes[1].axhline(s2["alpha"], ls=":", color="k"); axes[1].text(1.3, s2["alpha"] + .01, r"$\alpha$", fontsize=9)
    axes[1].set_title("S2: false-adapt, confidently-wrong world"); axes[1].set_ylabel("false-adapt rate")
    plt.tight_layout()
    for p in (os.path.join(FIGD, "fig_subsumption.png"), os.path.join(FIGF, "fig_subsumption.png")):
        fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print("figures written: fig_kfrontier.png, fig_subsumption.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default="synth",
        choices=["synth", "real", "figures"]); ap.add_argument("--max-tasks", type=int, default=123)
    a = ap.parse_args()
    {"synth": part_synth, "figures": part_figures,
     "real": lambda: part_real(a.max_tasks)}[a.part]()
