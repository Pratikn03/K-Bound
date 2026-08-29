#!/usr/bin/env python3
"""Plot the KGA radius-scaling frontier from the canonical panel manifest."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json"
OUTPUT = ROOT / "docs/research/kbound/figures/fig_decision_value_frontier.png"


def main() -> None:
    data = json.loads(SOURCE.read_text())
    panels = data["panels"]
    tracks = [
        ("CIFAR-10-C", panels["cifar10c"]["panel"]["architecture_panel_aggregate"]),
        ("ImageNet-C", panels["imagenetc"]["panel"]["architecture_panel_aggregate"]),
        ("ImageNet-R", panels["imagenet_r"]["panel"]["architecture_panel_aggregate"]),
    ]

    plt.rcParams.update({"font.size": 9, "axes.titleweight": "semibold"})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), constrained_layout=True)
    colors = {"curve": "#246B8E", "adapt": "#B5413E", "freeze": "#4B7F52"}

    for axis, (name, panel) in zip(axes, tracks, strict=True):
        sweep = sorted(panel["kappa_sweep"], key=lambda row: row["yield"])
        axis.plot(
            [row["yield"] for row in sweep],
            [row["regret"] for row in sweep],
            color=colors["curve"],
            marker="o",
            markersize=3.2,
            linewidth=1.6,
            label="radius scale sweep",
        )
        operating = next(row for row in sweep if row["kappa"] == 1.0)
        axis.scatter(
            operating["yield"], operating["regret"], marker="*", s=120,
            color="#111111", zorder=5, label=r"KGA ($\kappa=1$)",
        )
        regret = panel["regret"]
        axis.scatter(1.0, regret["always_adapt"], marker="s", s=38,
                     color=colors["adapt"], label="always-adapt")
        axis.scatter(1.0, regret["always_freeze"], marker="^", s=46,
                     color=colors["freeze"], label="always-freeze")
        axis.scatter(0.0, regret["always_freeze"], marker="x", s=42,
                     color=colors["freeze"], label="always-abstain")
        axis.set_title(name)
        axis.set_xlabel("decision coverage")
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.set_xlim(-0.03, 1.04)

    axes[0].set_ylabel("regret to per-cell oracle")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
