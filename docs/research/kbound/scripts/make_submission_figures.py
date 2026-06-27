#!/usr/bin/env python3
"""Regenerate the K-Bound *short paper* figures from result artifacts.

Reproducible: run from repo root after experiments are (re)run; it reads the
verified result JSONs and rewrites the PNGs in docs/research/kbound/figures/.

    python docs/research/kbound/scripts/make_submission_figures.py

Outputs:
  figures/fig_natural_forest.png     <- research_lock/KBOUND_WIN_BOOTSTRAP_CIS.json
  figures/fig_frontier_schematic.png <- conceptual (no data)
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
FIG  = os.path.join(ROOT, "docs/research/kbound/figures")
CIS  = os.path.join(ROOT, "research_lock/KBOUND_WIN_BOOTSTRAP_CIS.json")
GREEN, GRAY = "#1b7837", "#888888"
# Honest scope: only held-out natural shifts with legitimate CIs. Camelyon's CI in
# this file is the WITHDRAWN id_val-pooled result and is intentionally excluded.
KEEP = ["Office-Home", "iWildCam"]

def forest():
    W = [w for w in json.load(open(CIS))["wins"] if w["name"] in KEEP]
    W = sorted(W, key=lambda w: KEEP.index(w["name"]))
    rows, labels, colors = [], [], []
    for w in W:
        for comp, tag in [("kga_vs_adapt","vs always-adapt"), ("kga_vs_freeze","vs always-freeze")]:
            d = w[comp]; lo, hi = d["ci95"]
            rows.append((d["mean"], lo, hi)); labels.append(f"{w['name']}\n{tag}")
            colors.append(GREEN if lo > 0 else GRAY)
    fig, ax = plt.subplots(figsize=(6.4, 3.2)); y = np.arange(len(rows))[::-1]
    for yi,(m,lo,hi),c in zip(y, rows, colors):
        ax.plot([lo,hi],[yi,yi],color=c,lw=3,solid_capstyle="round"); ax.plot(m,yi,"o",color=c,ms=7)
    ax.axvline(0,color="k",lw=1,ls="--"); ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=8.5)
    ax.set_xlabel("Regret reduction by KGA  (positive $\\rightarrow$ KGA better)  [95% bootstrap CI]")
    ax.set_title("Natural-shift safety: KGA vs each fixed policy", fontsize=11)
    ax.legend(handles=[Patch(color=GREEN,label="CI excludes 0 (beats policy)"),
                       Patch(color=GRAY,label="CI includes 0 (ties policy)")],
              fontsize=7.5, loc="lower right", framealpha=0.9)
    ax.margins(y=0.12); fig.tight_layout(); fig.savefig(f"{FIG}/fig_natural_forest.png", dpi=200); plt.close(fig)
    print("wrote fig_natural_forest.png")

def frontier():
    fig, ax = plt.subplots(figsize=(6.4, 2.5)); b=1.0
    ax.axvspan(-b,b,color="#f0a000",alpha=0.25); ax.axvspan(b,3.2,color=GREEN,alpha=0.18); ax.axvspan(-3.2,-b,color="#d6604d",alpha=0.18)
    for x in (-b,b): ax.axvline(x,color="k",lw=1.2)
    ax.axvline(0,color="k",lw=0.6,ls=":")
    ax.text(2.1,0.5,"ADAPT\n(knowably helpful)",ha="center",va="center",fontsize=9,color=GREEN)
    ax.text(-2.1,0.5,"FREEZE\n(knowably harmful)",ha="center",va="center",fontsize=9,color="#d6604d")
    ax.text(0,0.5,"ABSTAIN\n(unknowable)",ha="center",va="center",fontsize=9,color="#9a6b00")
    ax.annotate("",xy=(b,-0.28),xytext=(-b,-0.28),arrowprops=dict(arrowstyle="<->",color="k"))
    ax.text(0,-0.43,r"calibration-drift budget  $2\beta$",ha="center",va="top",fontsize=8.5)
    ax.set_xlim(-3.2,3.2); ax.set_ylim(-0.6,1.05); ax.set_yticks([])
    ax.set_xlabel(r"observable benefit margin  $\widehat{M}$")
    ax.set_title(r"Benefit-sign frontier: sign identifiable iff $|\widehat{M}|>\beta$", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_frontier_schematic.png", dpi=200); plt.close(fig)
    print("wrote fig_frontier_schematic.png")

if __name__ == "__main__":
    forest(); frontier(); print("submission figures regenerated.")
