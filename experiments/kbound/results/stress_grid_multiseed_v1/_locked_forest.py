#!/usr/bin/env python3
"""Forest plot: 95% paired-bootstrap CIs for the 6 KGA-vs-trivial comparisons (mean regret diff)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/sessions/peaceful-blissful-ptolemy/mnt/uav/AutoML_Flagship_V8/experiments/kbound/results/stress_grid_multiseed_v1"
r = json.load(open(os.path.join(RES, "LOCKED_ANALYSIS_RESULTS.json")))
rows = r["comparisons"]
# order: tent(adapt,freeze), eata(adapt,freeze), sar(adapt,freeze) -> top to bottom
order = ["tent vs always-adapt", "tent vs always-freeze",
         "eata vs always-adapt", "eata vs always-freeze",
         "sar vs always-adapt", "sar vs always-freeze"]
rows = sorted(rows, key=lambda x: order.index(x["label"]))
ys = list(range(len(rows)))[::-1]  # top label at top

fig, ax = plt.subplots(figsize=(8.6, 4.8))
for y, row in zip(ys, rows):
    lo, hi, pt = row["ci95_lo"], row["ci95_hi"], row["mean_diff_kga_minus_trivial"]
    surv = row["survives_holm"]
    color = "#1a7d3c" if surv else "#b03030"
    ax.plot([lo, hi], [y, y], color=color, lw=2.4, solid_capstyle="round", zorder=2)
    ax.plot([pt], [y], "o", color=color, ms=8, zorder=3, markeredgecolor="black", markeredgewidth=0.6)
    # annotation
    txt = f"{pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]   p_Holm={row['p_holm']:.1e}  {'PASS' if surv else 'n.s.'}"
    ax.text(0.02, y + 0.22, txt, transform=ax.get_yaxis_transform(), fontsize=8.0,
            ha="left", va="bottom", color=color)

ax.axvline(0, color="black", lw=1.0, ls="--", zorder=1)
ax.text(0, len(rows) - 0.35, "  KGA better (lower regret)  <-- | -->  KGA worse",
        fontsize=8, ha="center", va="bottom", color="#555")
ax.set_yticks(ys)
ax.set_yticklabels([row["label"] for row in rows], fontsize=10)
ax.set_xlabel("Mean per-condition regret difference  (KGA - trivial),  95% paired-bootstrap CI (10^4)", fontsize=9.5)
ax.set_title("Protocol-A (locked): KGA vs trivial policies — 5-seed, per-condition, Holm-corrected\n"
             "CIFAR-10-C stress grid (432 conditions). Green = survives Holm (beats trivial); Red = does not.",
             fontsize=9.5)
ax.set_xlim(-0.165, 0.02)
ax.grid(axis="x", alpha=0.25)
ax.set_ylim(-0.6, len(rows) - 0.1)
fig.tight_layout()
out = os.path.join(RES, "LOCKED_forest_plot.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("WROTE", out)
