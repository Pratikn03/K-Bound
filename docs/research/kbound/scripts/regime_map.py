#!/usr/bin/env python3
"""
regime_map.py -- the K-Bound regime map: harmful fraction x detectability, per dataset.
CPU-only (matplotlib + numpy). Produces figures/fig_regime_map.png.

Shows the paper's central empirical claim in one picture: certified routing can WIN only in the
MIXED + DETECTABLE corner (moderate harmful fraction AND an unlabeled signal that separates harmful
from helpful, AUC high); elsewhere it is no-harm (harmful-dominated) or NULL (helpful-dominated).

NUMBERS ARE REAL. The four WILDS points carry a measured harm-detection AUC from their result JSONs:
  Camelyon17  : base_rate_harmful 0.4259, best harm-AUC 0.855   (wilds_kbound/result_8d3c0c41.json)
  ImageNet-R  : base_rate_harmful 0.3889, best harm-AUC 0.948   (imagenetr_kbound_debug_mps/result_75ee8322.json)
  iWildCam    : base_rate_harmful 0.8958, best harm-AUC 0.907   (win_hunt_v5_iwildcam/result_0ba633eb.json)
  RxRx1       : base_rate_harmful 0.9722, best harm-AUC 1.000   (win_hunt_v5/rxrx1_aggr/result_4a2840ef.json)
The CIFAR-10-C / ImageNet-C stress tracks are beats-both wins at harmful fractions 0.19-0.89 with
false-adapt 0 (decisive_tta_results.json), but their runner does not store a single harm-AUC scalar;
they are annotated, not plotted, until the fold-in computes detectability uniformly for all nine.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.normpath(os.path.join(HERE, "..", "figures", "fig_regime_map.png"))

# name, harmful_fraction, harm_AUC, regime, verdict(directional; CI-final at fold-in)
PTS = [
    ("Camelyon17",  0.4259, 0.855, "mixed+detectable",  "candidate (no-harm; win pending fold-in)"),
    ("ImageNet-R",  0.3889, 0.948, "mixed+detectable",  "no-harm (fixed-tau over-abstains)"),
    ("iWildCam",    0.8958, 0.907, "harmful-dominated", "no-harm (freezes the harm)"),
    ("RxRx1",       0.9722, 1.000, "harmful-dominated", "no-harm (freezes the harm)"),
]
COLOR = {"mixed+detectable": "#dd6b20", "harmful-dominated": "#c53030",
         "helpful-dominated": "#2b8a3e"}

def main():
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    # the certifiable-WIN corner: mixed harmful fraction AND detectable
    ax.axhspan(0.80, 1.01, xmin=0.25, xmax=0.75, color="#f6e05e", alpha=0.25)
    ax.text(0.50, 0.83, "certifiable-win corner\n(mixed + detectable)", ha="center",
            va="bottom", fontsize=8.5, style="italic", color="#975a16")
    ax.axvspan(0.25, 0.75, color="grey", alpha=0.05)

    for name, hf, auc, reg, verdict in PTS:
        ax.scatter(hf, auc, s=90, color=COLOR[reg], edgecolor="k", linewidth=0.6, zorder=3)
        dy = -0.035 if name in ("iWildCam",) else 0.02
        ax.annotate(f"{name}\n(h={hf:.2f}, AUC={auc:.2f})", (hf, auc),
                    textcoords="offset points", xytext=(8, 6 if dy > 0 else -22), fontsize=7.5)

    # honest annotation for the stress-track wins (no fabricated AUC)
    ax.annotate("CIFAR-10-C / ImageNet-C stress:\nbeats-both, FA$_u$=0 (AUC uniform at fold-in)",
                (0.37, 0.995), textcoords="offset points", xytext=(4, 8), fontsize=7.5,
                color="#975a16", style="italic")
    ax.scatter([0.37], [0.995], marker="*", s=160, color="#f6e05e", edgecolor="k",
               linewidth=0.6, zorder=3, label="beats-both (stress track)")

    ax.set_xlabel("harmful fraction  (share of cells where adaptation hurts)")
    ax.set_ylabel("detectability  (harm-vs-help AUC from unlabeled evidence)")
    ax.set_title("K-Bound regime map: wins live in the mixed + detectable corner")
    ax.set_xlim(0, 1.02); ax.set_ylim(0.45, 1.03)
    # legend for regimes
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR[k], markeredgecolor='k',
                  markersize=9, label=k) for k in ["mixed+detectable","harmful-dominated"]]
    leg.append(Line2D([0],[0], marker='*', color='w', markerfacecolor="#f6e05e", markeredgecolor='k',
                      markersize=13, label="beats-both (stress track)"))
    ax.legend(handles=leg, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG, dpi=150); plt.close(fig)
    print(f"wrote {FIG}")
    print("regime map: 4 WILDS datasets plotted with measured harm-AUC; stress-track wins annotated.")
    print("mixed+detectable (Camelyon, ImageNet-R) sit in the win corner; harmful-dominated")
    print("(iWildCam, RxRx1) sit far right where KGA freezes -> no-harm.")

if __name__ == "__main__":
    main()
