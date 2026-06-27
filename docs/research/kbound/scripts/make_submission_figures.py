#!/usr/bin/env python3
"""Regenerate the K-Bound short-paper figures (reproducible).

    python docs/research/kbound/scripts/make_submission_figures.py

fig_natural_forest.png     : per-dataset regret reduction vs each fixed policy,
                             VALID OUT-OF-FOLD radius (matches the paper's honest
                             no-harm result). We do NOT use KBOUND_WIN_BOOTSTRAP_CIS.json
                             for Office-Home: that file's CI is an in-sample radius that
                             overstates the result as a beats-both. Honest out-of-fold:
                             both Office-Home and iWildCam beat always-adapt and TIE
                             always-freeze (no-harm / damage-prevention).
fig_frontier_schematic.png : conceptual (no data).
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
FIG  = os.path.join(ROOT, "docs/research/kbound/figures")
GREEN, GRAY = "#1b7837", "#888888"

# Out-of-fold regret reductions (gap, lo, hi) matching the paper's per-dataset tables.
ROWS = [
    ("Office-Home\nvs always-adapt",  0.031,  0.004, 0.062),
    ("Office-Home\nvs always-freeze", 0.0001, 0.0,   0.0003),
    ("iWildCam\nvs always-adapt",     0.099,  0.080, 0.119),
    ("iWildCam\nvs always-freeze",    0.0004, 0.0,   0.0013),
]

def forest():
    fig, ax = plt.subplots(figsize=(6.4, 3.2)); y = np.arange(len(ROWS))[::-1]
    for yi,(lab,m,lo,hi) in zip(y, ROWS):
        c = GREEN if lo > 0 else GRAY
        ax.plot([lo,hi],[yi,yi],color=c,lw=3,solid_capstyle="round"); ax.plot(m,yi,"o",color=c,ms=7)
    ax.axvline(0,color="k",lw=1,ls="--")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in ROWS], fontsize=8.5)
    ax.set_xlabel("Regret reduction by KGA  (positive $\\rightarrow$ KGA better)  [95% bootstrap CI, out-of-fold]")
    ax.set_title("Natural-shift safety: KGA vs each fixed policy", fontsize=11)
    ax.legend(handles=[Patch(color=GREEN,label="CI excludes 0 (beats policy)"),
                       Patch(color=GRAY,label="CI includes 0 (ties policy)")],
              fontsize=7.5, loc="lower right", framealpha=0.9)
    ax.margins(y=0.12); fig.tight_layout(); fig.savefig(f"{FIG}/fig_natural_forest.png", dpi=200); plt.close(fig)
    print("wrote fig_natural_forest.png (out-of-fold; Office-Home + iWildCam tie freeze)")

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
    forest(); frontier(); print("done")
