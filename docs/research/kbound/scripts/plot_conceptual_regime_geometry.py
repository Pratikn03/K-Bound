#!/usr/bin/env python3
"""Generate the conceptual K-Bound regime geometry without measured coordinates."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "docs/research/kbound/figures/fig_phase_diagram.png"


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.titleweight": "semibold"})
    fig, axis = plt.subplots(figsize=(5.7, 4.15), constrained_layout=True)

    regions = [
        (0.0, 0.5, 0.5, 0.5, "#E7EEF3", "weak evidence /\nunsupported commitment", 0.25, 0.75),
        (0.5, 0.5, 0.5, 0.5, "#DDEDDD", "helpful-dominated", 0.75, 0.75),
        (0.0, 0.0, 0.5, 0.5, "#F2E2E0", "harmful-dominated", 0.25, 0.25),
        (0.5, 0.0, 0.5, 0.5, "#E7E3F1", "mixed and detectable", 0.75, 0.25),
    ]
    for x, y, width, height, color, label, tx, ty in regions:
        axis.add_patch(Rectangle((x, y), width, height, facecolor=color,
                                 edgecolor="white", linewidth=2.2))
        axis.text(tx, ty, label, ha="center", va="center", weight="semibold")

    axis.axhline(0.5, color="#555555", linewidth=0.8, alpha=0.55)
    axis.axvline(0.5, color="#555555", linewidth=0.8, alpha=0.55)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("Evidence distinguishability: low to high")
    axis.set_ylabel("Signed adaptation benefit: harmful to helpful")
    axis.set_title("K-Bound regime geometry (conceptual)")
    for spine in axis.spines.values():
        spine.set_visible(False)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
