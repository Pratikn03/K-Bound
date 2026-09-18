#!/usr/bin/env python3
"""Plot the Manski/Rosenbaum continuous sensitivity frontier beta*(Delta) = exp(2|Delta|).

Generates publication-quality PDF and PNG vector graphics in docs/research/kbound/figures/.
"""

from __future__ import annotations

import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
FIG_DIR = ROOT / "docs/research/kbound/figures"
PDF_OUT = FIG_DIR / "fig_sensitivity_frontier.pdf"
PNG_OUT = FIG_DIR / "fig_sensitivity_frontier.png"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)

    # -------------------------------------------------------------
    # Panel 1: Break-even odds ratio beta*(Delta) = exp(2*|Delta|)
    # -------------------------------------------------------------
    deltas = np.linspace(-0.22, 0.22, 500)
    beta_star = np.exp(2.0 * np.abs(deltas))

    # Plot frontier curve
    ax1.plot(deltas, beta_star, color="#1D4E89", linewidth=2.0, label=r"Frontier $\beta^*(\Delta) = \exp(2|\Delta|)$")
    ax1.axhline(1.0, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
    ax1.axvline(0.0, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)

    # Shade directional soundness zones
    pos_mask = deltas >= 0
    neg_mask = deltas <= 0
    ax1.fill_between(deltas[pos_mask], 1.0, beta_star[pos_mask], color="#2E7D32", alpha=0.15, label=r"Directionally Invariant $\mathbf{ADAPT}$")
    ax1.fill_between(deltas[neg_mask], 1.0, beta_star[neg_mask], color="#C62828", alpha=0.15, label=r"Directionally Invariant $\mathbf{FREEZE}$")

    # Annotate empirical benchmarks
    benchmarks = [
        ("CCT-20 (Adapt Gap)", 0.1819, np.exp(2 * 0.1819), "#2E7D32", (0.07, 1.35)),
        ("CIFAR-10-C Tent", 0.0062, np.exp(2 * 0.0062), "#1565C0", (0.015, 1.15)),
        ("CIFAR-10-C SAR", -0.0013, np.exp(2 * 0.0013), "#AD1457", (-0.16, 1.15)),
    ]

    for label, d_val, b_val, col, text_pos in benchmarks:
        ax1.scatter(d_val, b_val, color=col, s=45, zorder=6, edgecolors="#111111", linewidth=0.8)
        ax1.annotate(
            f"{label}\n($\\Delta={d_val:+.4f}$, $\\beta^*={b_val:.3f}$)",
            xy=(d_val, b_val),
            xytext=text_pos,
            fontsize=8,
            fontweight="medium",
            arrowprops=dict(arrowstyle="->", color=col, lw=0.9, shrinkA=3, shrinkB=3),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FAFAFA", edgecolor=col, alpha=0.9, lw=0.7),
        )

    ax1.set_xlabel(r"Observed Effect Contrast $\Delta = R_T(f_0) - R_T(f_a)$")
    ax1.set_ylabel(r"Break-Even Selection Odds Ratio $\beta^*(\Delta)$")
    ax1.set_title(r"(a) Continuous Manski/Rosenbaum Sensitivity Frontier")
    ax1.set_xlim(-0.22, 0.22)
    ax1.set_ylim(0.95, 1.60)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper center", framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 2: Identified Set Expansion Under Increasing Bias Bound beta
    # -------------------------------------------------------------
    betas = np.linspace(0.0, 0.12, 300)
    
    # Trace for a positive candidate (e.g. CCT-20 primary contrast Delta = +0.1819)
    delta_cct = 0.1819
    cct_lower = delta_cct - (np.exp(betas) - 1.0) / 2.0
    cct_upper = delta_cct + (1.0 - np.exp(-betas)) / 2.0

    # Trace for a subtle candidate (e.g. CIFAR-10-C Tent Delta = +0.0062)
    delta_tent = 0.0062
    tent_lower = delta_tent - (np.exp(betas) - 1.0) / 2.0
    tent_upper = delta_tent + (1.0 - np.exp(-betas)) / 2.0

    ax2.plot(betas, cct_lower, color="#2E7D32", linewidth=1.8, label=r"CCT-20 Lower Bound ($\Delta = +0.1819$)")
    ax2.plot(betas, cct_upper, color="#2E7D32", linestyle="--", linewidth=1.4, label=r"CCT-20 Upper Bound")

    ax2.plot(betas, tent_lower, color="#1565C0", linewidth=1.8, label=r"CIFAR Tent Lower Bound ($\Delta = +0.0062$)")
    ax2.plot(betas, tent_upper, color="#1565C0", linestyle="--", linewidth=1.4, label=r"CIFAR Tent Upper Bound")

    ax2.axhline(0.0, color="#111111", linestyle="-", linewidth=1.1, alpha=0.85)

    # Zero-crossing marker for Tent
    break_even_tent = np.log(1.0 + 2.0 * delta_tent)
    ax2.scatter(break_even_tent, 0.0, color="#D32F2F", s=50, zorder=6, marker="x", linewidth=2.0)
    ax2.annotate(
        f"Tent Zero Crossing\n$\\beta = {break_even_tent:.4f}$",
        xy=(break_even_tent, 0.0),
        xytext=(break_even_tent + 0.02, -0.06),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF9C4", edgecolor="#D32F2F", alpha=0.9, lw=0.7),
    )

    ax2.fill_between(betas, tent_lower, tent_upper, color="#1565C0", alpha=0.08)
    ax2.fill_between(betas, cct_lower, cct_upper, color="#2E7D32", alpha=0.08)

    ax2.set_xlabel(r"Unobserved Confounding Bound $\beta$")
    ax2.set_ylabel(r"Identified Set $[\Delta_{\min}(\beta), \Delta_{\max}(\beta)]$")
    ax2.set_title(r"(b) Identified Set Trajectory Under Unobserved Confounding")
    ax2.set_xlim(0.0, 0.12)
    ax2.set_ylim(-0.10, 0.25)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.savefig(PDF_OUT, format="pdf", bbox_inches="tight")
    plt.savefig(PNG_OUT, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated sensitivity figures:\n  {PDF_OUT}\n  {PNG_OUT}")


if __name__ == "__main__":
    main()
