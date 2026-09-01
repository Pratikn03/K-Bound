#!/usr/bin/env python3
"""Render the maintained population schematic or historical figure variants.

Maintained, data-free path:
    python docs/research/kbound/scripts/make_submission_figures.py --frontier-only

The no-argument path retains historical forest/phase plots for reproducibility.
Its hard-coded experiment values are not current release evidence and must not
be used by a manuscript build. The frontier-only path writes just the schematic,
with a declared illustrative beta and M inside its feasible [-1/2, 1/2] range.
"""
import argparse
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ROOT = REPO_ROOT if os.path.isdir(os.path.join(REPO_ROOT, "docs/research/kbound")) else os.path.dirname(HERE)
FIG = os.path.join(ROOT, "docs/research/kbound/figures") if ROOT == REPO_ROOT else os.path.join(ROOT, "figures")
GREEN, GRAY = "#1b7837", "#888888"

# Out-of-fold regret reductions (gap, lo, hi) matching the paper's per-dataset tables.
ROWS = [
    ("Office-Home\nvs always-adapt",  0.031113,  0.003843, 0.061868),
    ("Office-Home\nvs always-freeze", 0.000110, 0.0,      0.000330),
    ("iWildCam\nvs always-adapt",     0.098695,  0.079503, 0.118212),
    ("iWildCam\nvs always-freeze",    0.0,       0.0,      0.0),
]

def forest():
    fig, ax = plt.subplots(figsize=(6.4, 3.2)); y = np.arange(len(ROWS))[::-1]
    for yi,(lab,m,lo,hi) in zip(y, ROWS):
        c = GREEN if lo > 0 else GRAY
        ax.plot([lo,hi],[yi,yi],color=c,lw=3,solid_capstyle="round"); ax.plot(m,yi,"o",color=c,ms=7)
    ax.axvline(0,color="k",lw=1,ls="--")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in ROWS], fontsize=8.5)
    ax.set_xlabel("Regret reduction by KGA (95% bootstrap CI)")
    ax.set_title("Natural-shift safety: KGA vs each fixed policy", fontsize=11)
    ax.legend(handles=[Patch(color=GREEN,label="CI excludes 0 (beats policy)"),
                       Patch(color=GRAY,label="CI includes 0 (ties policy)")],
              fontsize=7.5, loc="lower right", framealpha=0.9)
    ax.margins(y=0.12); fig.tight_layout(); fig.savefig(f"{FIG}/fig_natural_forest.png", dpi=200); plt.close(fig)
    print("wrote fig_natural_forest.png (out-of-fold; Office-Home + iWildCam tie freeze)")

def frontier():
    # Illustrative values, not fitted parameters or experimental measurements.
    b, limit = 0.1, 0.32
    assert 0 < b < limit <= 0.5
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ax.axvspan(-b,b,color="#71717A",alpha=0.16)
    ax.axvspan(b,limit,color="#1F5A85",alpha=0.18)
    ax.axvspan(-limit,-b,color="#C68B2B",alpha=0.20)
    for x in (-b,b): ax.axvline(x,color="k",lw=1.2)
    ax.axvline(0,color="k",lw=0.6,ls=":")
    ax.text(0.21,0.5,"ADAPT\n(knowably helpful)",ha="center",va="center",fontsize=9,color="#1F5A85")
    ax.text(-0.21,0.5,"FREEZE\n(knowably harmful)",ha="center",va="center",fontsize=9,color="#8A5A0B")
    ax.text(0,0.5,"ABSTAIN\n(no sound strict\ncommitment)",ha="center",va="center",fontsize=9,color="#454545")
    ax.annotate("",xy=(b,-0.28),xytext=(-b,-0.28),arrowprops=dict(arrowstyle="<->",color="k"))
    ax.text(0,-0.43,r"unresolved band width  $2\beta$",ha="center",va="top",fontsize=8.5)
    ax.set_xlim(-limit,limit); ax.set_ylim(-0.6,1.05); ax.set_yticks([])
    ax.set_xticks((-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3))
    ax.set_xlabel(r"Population evidence margin $M$ (illustration: $\beta=0.1$)")
    ax.set_title(r"Population strict-commitment frontier: commit iff $|M|>\beta$", fontsize=10.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_frontier_schematic.png", dpi=200); plt.close(fig)
    print("wrote fig_frontier_schematic.png")

def phase():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axvspan(0, 0.34, color="#bfc5ca", alpha=0.70)
    ax.fill_between([0.34, 1], 0.22, 1, color="#9bd3c3", alpha=0.52)
    ax.fill_between([0.34, 1], -0.22, 0.22, color="#f1cf72", alpha=0.52)
    ax.fill_between([0.34, 1], -1, -0.22, color="#e7a096", alpha=0.52)
    ax.axvline(0.34, color="#65737e", ls="--", lw=1.2)
    ax.axhline(0, color="#65737e", lw=0.8)
    ax.text(0.17, 0, "Weak evidence /\nunsupported commitment", ha="center", va="center")
    ax.text(0.70, 0.62, "Helpful-dominated", ha="center", va="center", color="#116149")
    ax.text(0.70, 0, "Mixed and detectable", ha="center", va="center", color="#8a6200")
    ax.text(0.70, -0.62, "Harmful-dominated", ha="center", va="center", color="#a73d2d")
    ax.set_xlim(0, 1); ax.set_ylim(-1, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Evidence distinguishability (low to high)")
    ax.set_ylabel("Signed adaptation benefit\n(harmful to helpful)")
    ax.set_title("Conceptual regime geometry")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_phase_diagram.png", dpi=200); plt.close(fig)
    print("wrote fig_phase_diagram.png (conceptual; no numeric coordinates)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-only", action="store_true")
    args = parser.parse_args()
    if args.frontier_only:
        frontier()
    else:
        forest(); frontier(); phase(); print("done")
