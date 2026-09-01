#!/usr/bin/env python3
"""Render the maintained, illustrative KGA interval rule without experiment data.

Chart contract
--------------
Question: which strict interval signs produce ADAPT, FREEZE, or ABSTAIN?
Takeaway: the interval must exclude zero; its coverage remains a separate premise.
Surface: Figure 3 in the existing short/long PDFs and Word manuscript.
Family: horizontal dot-and-interval comparison of the three worked examples in
the shared manuscript. These are constructed examples, not benchmark estimates.
Grain: one decision example per row; the radius is 0.01 in all three examples.
Palette: two roots (blue/orange) plus neutral gray; distinct markers and direct
action labels keep the figure readable without color. This user-authored academic
manuscript is unbranded. QA: inspect the PNG and its final PDF/DOCX placements.

Only the output PNG is written. No model fitting, residual calibration, dataset
access, or changes to experimental records occur.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    ("ADAPT", 0.04, 0.01, "#1F5A85", "o"),
    ("FREEZE", -0.04, 0.01, "#C68B2B", "s"),
    ("ABSTAIN", 0.005, 0.01, "#71717A", "D"),
)


def render(output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, ax = plt.subplots(figsize=(7, 3.65), layout="constrained")
        ax.axvline(0, color="#454545", linewidth=0.9, linestyle="--", zorder=1)
        for row, (action, prediction, radius, color, marker) in enumerate(EXAMPLES):
            lower, upper = prediction - radius, prediction + radius
            expected_action = "ADAPT" if lower > 0 else "FREEZE" if upper < 0 else "ABSTAIN"
            if expected_action != action:
                raise ValueError(f"Example does not match the strict interval rule: {action}")
            ax.errorbar(
                prediction, row, xerr=radius, fmt=marker, color=color,
                markersize=6, markerfacecolor="white" if action == "ABSTAIN" else color,
                markeredgewidth=1.2, linewidth=2, capsize=5, zorder=3,
            )
            ax.text(
                prediction, row - 0.18, f"[{lower:.3f}, {upper:.3f}]",
                ha="center", va="bottom", fontsize=9, color="#333333",
            )
        ax.set_yticks(range(len(EXAMPLES)), [row[0] for row in EXAMPLES])
        ax.set_ylim(2.5, -0.65)
        ax.set_xlim(-0.065, 0.065)
        ax.set_xticks((-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06))
        ax.set_xlabel("Predicted measured-cell benefit (score difference)")
        ax.set_title(
            "Interval decisions for measured cell benefit\n"
            "Illustration only; fixed radius 0.01, not an experimental result",
            fontsize=11, pad=12, color="#222222",
        )
        ax.grid(axis="x", color="#e8e8e8", linewidth=0.6)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#888888")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, facecolor="white")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "fig_certificate.png")
    args = parser.parse_args()
    render(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
