"""Generate ELARA-U result figures (PNG) from VERIFIED data.

Data figures from experiments/elara_u/honest_benchmark.json (+ the verified ablation
files and D22 natural_shift_results.json):
  elara_u_rank.png, elara_u_regret.png, elara_u_calib.png, elara_u_family_heatmap.png,
  elara_u_ablation.png.
Conceptual figures: elara_u_architecture.png, elara_u_coverage.png, elara_u_lifecycle.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/elara_u"
H = json.loads((EXP / "honest_benchmark.json").read_text())
FIG = ROOT / "docs/research/figures"
FIG.mkdir(parents=True, exist_ok=True)

PT_AUC = {m: np.array(v, float) for m, v in H["per_task_auc"].items()}
PT_RANK = {m: np.array(v, float) for m, v in H["per_task_rank"].items()}
FAM = np.array(H["task_families"])
BEST = H["best_fixed"]
ORACLE = np.max(np.stack([PT_AUC[m] for m in PT_AUC if m.startswith("fixed/")]), axis=0)
METHODS = ["stack", "auto_select", "stack_rel", BEST, "cw_mean", "rank_mean"]
LBL = {"stack": "ELARA-U\nStack", "auto_select": "auto-\nselect", "stack_rel": "Stack+\nrel-gate",
       BEST: BEST.split("/")[-1] + "\n(best fix)", "cw_mean": "CW-\nmean", "rank_mean": "rank-\nmean"}


def _color(m):
    return "#d95f02" if m == "stack" else "#2c7fb8" if m == "auto_select" else "#999999"


def _bar(values, ylabel, title, fname, methods=METHODS):
    plt.figure(figsize=(6, 3))
    plt.bar([LBL[m] for m in methods], [values[m] for m in methods],
            color=[_color(m) for m in methods])
    plt.ylabel(ylabel); plt.title(title)
    plt.xticks(fontsize=8); plt.tight_layout()
    plt.savefig(FIG / fname, dpi=150); plt.close()


def _heatmap():
    fams = sorted(set(FAM.tolist()))
    rows = ["stack", "auto_select", BEST]
    M = np.array([[np.mean(PT_RANK[m][FAM == f]) for f in fams] for m in rows])
    plt.figure(figsize=(6, 3))
    im = plt.imshow(M, cmap="RdYlGn_r", aspect="auto")
    plt.colorbar(im, label="mean rank (lower=greener=better)")
    plt.xticks(range(len(fams)), fams, fontsize=8)
    plt.yticks(range(len(rows)), [LBL[m].replace("\n", " ") for m in rows], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            plt.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=8)
    plt.title("Per-family mean rank")
    plt.tight_layout(); plt.savefig(FIG / "elara_u_family_heatmap.png", dpi=150); plt.close()


def _ablation():
    """Verified reliability ablation: Δ AUROC of adding reliability, across regimes.
    All near zero or significantly negative -> reliability does not help."""
    def g(path, *keys):
        d = json.loads((EXP / path).read_text())
        for k in keys:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d
    bars = [
        ("stack\ngate", H["contrasts"]["stack_rel_vs_stack_ABLATION"]),
        ("learned\ni.i.d.", g("learned_router_ablation.json", "decisive_contrasts", "rel_vs_norel_ABLATION")),
        ("uniform\nshift", g("shift_stress_ablation.json", "severity_3.0", "contrasts", "rel_vs_norel_ABLATION")),
        ("missing-\nness", g("heterogeneous_degradation_ablation.json", "missing_0.7", "contrasts", "rel_vs_norel_ABLATION")),
    ]
    ns = json.loads((EXP / "natural_shift_results.json").read_text())
    bars.append(("natural\nshift D22", ns["drift_stack_vs_plain_stack"]))
    labels = [b[0] for b in bars]
    means = [b[1]["mean"] for b in bars]
    los = [b[1]["mean"] - b[1]["ci95"][0] for b in bars]
    his = [b[1]["ci95"][1] - b[1]["mean"] for b in bars]
    colors = ["#d7191c" if b[1]["ci95"][1] < 0 else "#999999" for b in bars]
    plt.figure(figsize=(6.5, 3))
    plt.bar(labels, means, yerr=[los, his], capsize=4, color=colors)
    plt.axhline(0, color="k", lw=0.8)
    plt.ylabel("Δ AUROC (reliability − none)")
    plt.title("Reliability ablation: Δ≈0 or <0 ⇒ reliability does not help")
    plt.xticks(fontsize=8); plt.tight_layout()
    plt.savefig(FIG / "elara_u_ablation.png", dpi=150); plt.close()


def _box(ax, xy, wh, text, face="#eef5fb", edge="#2c7fb8", fs=9):
    ax.add_patch(Rectangle(xy, wh[0], wh[1], facecolor=face, edgecolor=edge, linewidth=1.4))
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color="#444444"))


def _architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.5)); ax.axis("off")
    xs = [0.03, 0.22, 0.42, 0.62, 0.81]
    labels = ["Benchmark\nfamilies", "Detector\nzoo", "Score\narchive",
              "Rank-norm\nfeatures", "ELARA-U\nStack"]
    for x, label in zip(xs, labels):
        _box(ax, (x, 0.48), (0.15, 0.24), label)
    for x in xs[:-1]:
        _arrow(ax, (x + 0.15, 0.60), (x + 0.20, 0.60))
    _box(ax, (0.33, 0.12), (0.30, 0.20), "Validation labels only\nNo test-label routing", "#fff5eb", "#d95f02")
    _arrow(ax, (0.48, 0.32), (0.50, 0.48))
    _box(ax, (0.81, 0.12), (0.15, 0.20), "Routed\nscore", "#f0f9e8", "#41ab5d")
    _arrow(ax, (0.885, 0.48), (0.885, 0.32))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("ELARA-U score-archive and stacking architecture", fontsize=11)
    plt.tight_layout(); plt.savefig(FIG / "elara_u_architecture.png", dpi=150); plt.close(fig)


def _coverage():
    fams = sorted(set(FAM.tolist()))
    counts = {f: int((FAM == f).sum()) for f in fams}
    planned = {"time_series": 0, "industrial_v2": 0, "multimodal_3d": 0}
    labels = list(counts) + list(planned)
    vals = [counts[k] for k in counts] + [planned[k] for k in planned]
    colors = ["#2c7fb8"] * len(counts) + ["#cccccc"] * len(planned)
    plt.figure(figsize=(7, 3))
    plt.bar(labels, vals, color=colors)
    plt.ylabel("tasks in current archive")
    plt.title("Benchmark coverage: current (blue) + planned (grey)")
    plt.xticks(rotation=25, ha="right", fontsize=8)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.8, str(v), ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "elara_u_coverage.png", dpi=150); plt.close()


def _lifecycle():
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.axis("off")
    rows = [("1. Freeze protocol", "families, baselines"), ("2. Build archive", "train/val/test scores"),
            ("3. Fit stacker", "validation split only"), ("4. Score holdout", "no test-label tuning"),
            ("5. Audit report", "rank, regret, CI, failures")]
    x = 0.05
    for i, (head, body) in enumerate(rows):
        _box(ax, (x, 0.44), (0.16, 0.28), f"{head}\n{body}", "#f7f7f7", "#555555", fs=8)
        if i < len(rows) - 1:
            _arrow(ax, (x + 0.16, 0.58), (x + 0.19, 0.58))
        x += 0.19
    _box(ax, (0.20, 0.12), (0.60, 0.18),
         "Stacking beats selection (verified); reliability routing is negative (3 regimes + D22).\n"
         "Multimodal reliability test is pre-registered for the regime where it could help.",
         "#fff5eb", "#d95f02", fs=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("ELARA-U evidence lifecycle", fontsize=11)
    plt.tight_layout(); plt.savefig(FIG / "elara_u_lifecycle.png", dpi=150); plt.close(fig)


def main() -> int:
    _bar(H["average_rank"], "mean rank (lower better)", "Cross-domain mean rank (123 tasks)", "elara_u_rank.png")
    _bar({m: float((ORACLE - PT_AUC[m]).mean()) for m in METHODS}, "mean regret (lower better)",
         "Regret to best-single oracle", "elara_u_regret.png")
    calib_m = [m for m in METHODS if m in H["mean_ece"]]
    _bar(H["mean_ece"], "ECE (lower better)", "Calibration error", "elara_u_calib.png", methods=calib_m)
    _heatmap()
    _ablation()
    _architecture()
    _coverage()
    _lifecycle()
    print(f"wrote 8 figures to {FIG} from VERIFIED data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
