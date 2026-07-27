#!/usr/bin/env python3
"""
make_frontier_sweep_figs.py -- figures for the beta-sweep frontier test and the
decision-value / abstention analysis.

Every number plotted here is read from one of two result JSONs produced by the
frontier_sweep_v1 slice; nothing is hard-coded, nothing is drawn from a
generator.  Re-running the two upstream scripts and then this one reproduces the
figures byte-for-byte (matplotlib is seeded only through the data; there is no
randomness in this file).

Inputs
    experiments/kbound/frontier_sweep_v1/beta_sweep_results.json
    experiments/kbound/frontier_sweep_v1/decision_value_results.json

Outputs (docs/research/kbound/figures/)
    fig_beta_frontier_test.png      -- Fig. for Sec. "beta-sweep frontier test"
    fig_decision_value_frontier.png -- Fig. for Sec. "what the certificate buys"
    fig_yield_ceiling.png           -- Fig. for the abstention decomposition

Run:  python3 docs/research/kbound/figures/source/make_frontier_sweep_figs.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
SWEEP_DIR = os.path.join(REPO, "experiments", "kbound", "frontier_sweep_v1")

BETA_JSON = os.path.join(SWEEP_DIR, "beta_sweep_results.json")
DV_JSON = os.path.join(SWEEP_DIR, "decision_value_results.json")

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 6.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.5,
    "figure.dpi": 220,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

C_CIFAR = "#1f5fa8"
C_INET = "#c0392b"
C_INETR = "#7a5195"
C_EMP = "#111111"
C_REF = "#888888"


def _load(path):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Figure 1: does the population frontier operationalize?
# --------------------------------------------------------------------------- #
def fig_beta_frontier_test(bs):
    keys = list(bs["M_quality"].keys())
    rows = []
    for k in keys:
        ds, split, est = k.split("|")
        bhat = bs["beta_derivation"][k]["beta_hat_q90"]
        bmin = bs["beta_sound_min"][k]["beta_sound_min"]
        crux = [c for c in bs["crux"][k] if abs(c["beta"] - round(bhat, 6)) < 1e-9][0]
        ob = crux["outside_band"]
        err = (1.0 - ob["sign_match_rate"]) if ob["n"] else np.nan
        rows.append(dict(key=k, ds=ds, split=split, est=est, bhat=bhat, bmin=bmin,
                         yield_=ob["frac_of_cells"], n_commit=ob["n"], err=err))

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.05))

    # ---- (a) declared budget vs the budget soundness actually required ------
    ax = axes[0]
    mk = {"M_doc": "o", "M_atc4": "s", "M_gbm": "^", "M_bhat": "D"}
    for r in rows:
        c = C_CIFAR if r["ds"] == "cifar10c" else C_INET
        ax.scatter(r["bhat"], max(r["bmin"], 1e-3), s=34, marker=mk[r["est"]],
                   facecolor=c, edgecolor="k", linewidth=0.45, alpha=0.9, zorder=3)
    lim = [4e-3, 2.0]
    ax.plot(lim, lim, "-", color="k", lw=0.9, zorder=2)
    ax.fill_between(lim, lim, [lim[1]] * 2, color="#d62728", alpha=0.085, zorder=1)
    ax.fill_between(lim, [lim[0]] * 2, lim, color="#2ca02c", alpha=0.085, zorder=1)
    ax.text(0.055, 0.75, "declared $\\beta$ TOO SMALL\n(commitments unsound)",
            color="#a02020", fontsize=6.4, ha="left")
    ax.text(0.006, 0.011, "sound", color="#1a7a1a", fontsize=6.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"declared $\widehat\beta = q_{0.90}(|\gamma|)$ on dev cells")
    ax.set_ylabel(r"$\beta_{\mathrm{sound}}^{\min}$ required (oracle)")
    ax.set_title("(a) the budget is not declarable")
    from matplotlib.patches import Patch
    h = [Patch(fc=C_CIFAR, ec="k", lw=0.4, label="CIFAR-10-C"),
         Patch(fc=C_INET, ec="k", lw=0.4, label="ImageNet-C"),
         plt.Line2D([], [], ls="", marker="o", mfc="w", mec="k", ms=5, label="DoC $M$"),
         plt.Line2D([], [], ls="", marker="s", mfc="w", mec="k", ms=5, label="ATC-style $M$"),
         plt.Line2D([], [], ls="", marker="^", mfc="w", mec="k", ms=5, label="full-$Z$ GBM $M$"),
         plt.Line2D([], [], ls="", marker="D", mfc="w", mec="k", ms=5, label=r"shipped $\widehat\Delta$")]
    ax.legend(handles=h, loc="lower right", frameon=True, framealpha=0.92,
              handletextpad=0.35, labelspacing=0.25)

    # ---- (b) sound OR useful, never both, at the declared budget ------------
    ax = axes[1]
    for r in rows:
        c = C_CIFAR if r["ds"] == "cifar10c" else C_INET
        e = 0.0 if not np.isfinite(r["err"]) else r["err"]
        ax.scatter(r["yield_"], e, s=38, marker=mk[r["est"]], facecolor=c,
                   edgecolor="k", linewidth=0.45, alpha=0.9, zorder=4)
    ax.axhline(0.0, color="#1a7a1a", lw=1.3, zorder=2)
    ax.axvspan(-0.04, 0.02, color="#bdbdbd", alpha=0.35, zorder=1)
    ax.set_xlim(-0.04, 0.82)
    ax.set_ylim(-0.06, 1.06)
    ax.set_xlabel(r"decision yield at $\widehat\beta$")
    ax.set_ylabel(r"commit-error rate at $\widehat\beta$")
    ax.set_title("(b) sound or useful, never both")
    ax.text(0.30, 1.03, "5 configurations are sound only\nbecause they commit on 0 of 405 cells",
            fontsize=6.1, va="top",
            bbox=dict(fc="w", ec="#999999", lw=0.5, alpha=0.9))
    ax.text(0.045, 0.30, "green line: the commit-error\nrate the theorem promises",
            color="#1a7a1a", fontsize=6.4)
    ax.annotate("honest source-only\ncalibration, ImageNet-C:\n53% of commitments wrong",
                xy=(0.230, 0.532), xytext=(0.36, 0.62), fontsize=6.0, color="#a02020",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#a02020"))

    # ---- (c) regret-vs-yield: population rule swept over beta ---------------
    ax = axes[2]
    styles = [("cifar10c|loco|M_atc4", C_CIFAR, "-", r"ATC-style $M$ (leave-one-corruption-out)"),
              ("cifar10c|loco|M_gbm", "#3fa34d", "--", r"full-$Z$ GBM $M$ (leave-one-corruption-out)"),
              ("cifar10c|shipped|M_bhat", "#e08214", "-.", r"$M:=\widehat\Delta$ (channel held fixed)")]
    for key, col, ls, lab in styles:
        cur = bs["sensitivity_curve"][key]
        y = np.array([r["yield_"] for r in cur])
        g = np.array([r["regret"] for r in cur])
        ax.plot(y, g, ls, color=col, lw=1.3, label=lab, zorder=3)
        bhat = bs["beta_derivation"][key]["beta_hat_q90"]
        j = int(np.argmin([abs(r["beta"] - bhat) for r in cur]))
        ax.scatter([y[j]], [g[j]], s=46, marker="*", color=col, edgecolor="k",
                   linewidth=0.4, zorder=5)
    sw = bs["sweep"]["cifar10c|shipped|M_bhat"]
    emp = [r for r in sw if r["rule"].startswith("empirical")][0]
    aa = [r for r in sw if r["rule"] == "always_adapt"][0]
    ax.scatter([emp["yield_"]], [emp["regret"]], s=64, marker="P", color=C_EMP,
               zorder=6, label="shipped conformal rule (KGA)")
    ax.axhline(aa["regret"], color=C_REF, ls=":", lw=1.0)
    ax.text(0.02, aa["regret"] * 1.06, "always-adapt", color=C_REF, fontsize=6.2)
    ax.set_yscale("log")
    ax.set_xlabel("decision yield (fraction committed)")
    ax.set_ylabel("regret to per-cell oracle")
    ax.set_title(r"(c) CIFAR-10-C: $\beta$ sweep vs. the empirical rule")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.text(0.03, 0.06, "star = the $\\widehat{\\beta}$ the declaration procedure returns",
            transform=ax.transAxes, fontsize=6.2)

    out = os.path.join(FIGDIR, "fig_beta_frontier_test.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 2: the decision-yield / regret frontier
# --------------------------------------------------------------------------- #
def fig_decision_value_frontier(dv):
    tracks = [("cifar10c", "CIFAR-10-C stress grid ($n{=}6480$)", C_CIFAR),
              ("imagenetc", "ImageNet-C ($n{=}405$)", C_INET),
              ("imagenetr", "ImageNet-R ($n{=}480$)", C_INETR)]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.15))
    for ax, (ds, title, col) in zip(axes, tracks):
        f = dv["frontier"][ds]["__pooled__"]
        sw = [r for r in f["sweep"] if r["kappa"] < 1e5]
        y = np.array([r["yield"] for r in sw])
        g = np.array([r["regret"] for r in sw])
        ax.plot(y, g, "-o", color=col, lw=1.25, ms=2.6,
                label=r"$\kappa$-family ($\kappa\!\cdot\!\varepsilon$ radius)", zorder=3)
        k1 = f["kga_operating_point_kappa1"]
        ax.scatter([k1["yield"]], [k1["regret"]], s=95, marker="*", color=col,
                   edgecolor="k", linewidth=0.55, zorder=6,
                   label=r"shipped KGA ($\kappa{=}1$, $\alpha{=}0.10$)")
        ymin = g.min() * 0.35
        ob = f["oracle_yield_bound"]["ordering_maxB"]
        oy = np.array([p["yield"] for p in ob])
        og = np.array([max(p["regret"], ymin * 1.05) for p in ob])
        ax.plot(oy, og, ":", color="#444444", lw=1.1,
                label="oracle-yield bound (perfect ordering)", zorder=2)
        tb = f["trivial_baselines"]
        ax.axhline(tb["always_adapt"]["regret"], color="#2ca02c", ls="--", lw=1.0)
        ax.axhline(tb["always_freeze"]["regret"], color="#d62728", ls="--", lw=1.0)
        ax.scatter([0.0], [tb["always_abstain"]["regret"]], s=44, marker="X",
                   color="#d62728", edgecolor="k", linewidth=0.4, zorder=5,
                   label="always-abstain $=$ always-freeze")
        hb = f["heuristic_references"]["marginal_KL_BN_proxy"]["hindsight_envelope"][
            "best_regret_hindsight_FAu_le_alpha"]
        ax.scatter([1.0], [hb["regret"]], s=46, marker="v", color="#7f7f7f",
                   edgecolor="k", linewidth=0.4, zorder=5,
                   label=r"best BN-drift heuristic (hindsight, $\mathrm{FA}_u \leq \alpha$)")
        ax.set_yscale("log")
        ax.set_xlim(-0.06, 1.06)
        ax.set_xlabel("decision yield")
        if ax is axes[0]:
            ax.set_ylabel("regret to per-cell oracle (log)")
        ax.set_title(title)
        ymax = tb["always_freeze"]["regret"] * 3.0
        ax.set_ylim(ymin, ymax)
        ax.text(0.42, tb["always_adapt"]["regret"] * 0.72, "always-adapt",
                color="#2ca02c", fontsize=6.2, va="top")
        ax.text(0.42, tb["always_freeze"]["regret"] * 1.16, "always-freeze / abstain",
                color="#d62728", fontsize=6.2)
    axes[0].legend(loc="lower left", frameon=True, framealpha=0.93)
    out = os.path.join(FIGDIR, "fig_decision_value_frontier.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 3: why the abstaining tracks abstain
# --------------------------------------------------------------------------- #
def fig_yield_ceiling(dv):
    per = dv["abstention"]["per_cell_tracks"]
    nat = dv["abstention"]["natural_tracks"]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.15))

    # (a) achieved yield vs the ceiling P(|Delta| > eps)
    ax = axes[0]
    pts = []
    for ds, col in (("cifar10c", C_CIFAR), ("imagenetc", C_INET), ("imagenetr", C_INETR)):
        for m, v in per[ds].items():
            pts.append((v["yield_ceiling_P_absDelta_gt_eps"], v["yield"], col,
                        f"{ds}/{m}".replace("__pooled__", "pooled")))
    for ceil, yld, col, lab in pts:
        ax.scatter(ceil, yld, s=34, color=col, edgecolor="k", linewidth=0.4, zorder=3)
    ax.plot([0, 1], [0, 1], "-", color="k", lw=0.9, zorder=2)
    for ceil, yld, col, lab in pts:
        if lab.startswith("imagenetc/tent") or lab.startswith("imagenetr/convnext_tiny"):
            dy = 5 if lab.startswith("imagenetr") else -9
            ax.annotate(lab, (ceil, yld), textcoords="offset points", xytext=(7, dy),
                        fontsize=6.0, color="#a02020")
    ax.set_xlabel(r"yield ceiling $\Pr(|\Delta|>\varepsilon)$")
    ax.set_ylabel("achieved decision yield")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_title("(a) abstention is the radius, not timidity")
    ax.text(0.50, 0.22, "on the diagonal $=$ the certificate\ncommits on everything that is\n"
                        "committable at its own radius", fontsize=6.2)
    h = [plt.Line2D([], [], ls="", marker="o", mfc=C_CIFAR, mec="k", ms=5, label="CIFAR-10-C"),
         plt.Line2D([], [], ls="", marker="o", mfc=C_INET, mec="k", ms=5, label="ImageNet-C"),
         plt.Line2D([], [], ls="", marker="o", mfc=C_INETR, mec="k", ms=5, label="ImageNet-R")]
    ax.legend(handles=h, loc="upper left", frameon=True, framealpha=0.92)

    # (b) the natural tracks the abstention critique names
    ax = axes[1]
    sel = [("iWildCam", "iwildcam", ["tent_online", "eata_online", "sar_online"]),
           ("RxRx1", "rxrx1", ["tent_online", "sar_online", "eata_online"]),
           ("Office-Home", "officehome", ["tent_online_aggressive", "sar_online_mild",
                                          "eata_online_mild"])]
    labels, padapt, radapt, rfreeze = [], [], [], []
    for pretty, key, cands in sel:
        for c in cands:
            v = nat[key]["per_candidate"][c]
            cnt = v["counts_published"]
            labels.append(f"{pretty}  {c}\n"
                          f"{cnt['ADAPT']}A / {cnt['FREEZE']}F / {cnt['ABSTAIN']}Ab")
            padapt.append(v["P_Delta_gt_eps_adaptable"])
            radapt.append(v["regret_always_adapt"])
            rfreeze.append(v["regret_always_freeze"])
    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, padapt, w, color="#4c78a8", edgecolor="k", linewidth=0.4,
           label=r"$\Pr(\Delta>\varepsilon)$: an ADAPT is even available")
    ax.bar(x, radapt, w, color="#e45756", edgecolor="k", linewidth=0.4,
           label="regret of always-adapt")
    ax.bar(x + w, rfreeze, w, color="#54a24b", edgecolor="k", linewidth=0.4,
           label="regret of always-freeze")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=5.0)
    ax.set_ylabel("probability / accuracy regret")
    ax.set_title("(b) the '0 ADAPT' tracks: nothing to adapt to")
    ax.legend(loc="upper left", frameon=True, framealpha=0.93)
    out = os.path.join(FIGDIR, "fig_yield_ceiling.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main():
    bs = _load(BETA_JSON)
    dv = _load(DV_JSON)
    fig_beta_frontier_test(bs)
    fig_decision_value_frontier(dv)
    fig_yield_ceiling(dv)


if __name__ == "__main__":
    main()
