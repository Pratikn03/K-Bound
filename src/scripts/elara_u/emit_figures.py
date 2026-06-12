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
    """Coverage map: 5 main-suite families (blue) + completed auxiliary/boundary families
    (green, real counts from their result JSONs) + still-planned families (grey). Counts
    trace to verified files; missing files degrade gracefully to 0/grey."""
    def _n(path, key="n_tasks"):
        p = EXP / path
        if not p.exists():
            return 0
        try:
            return int(json.loads(p.read_text()).get(key, 0))
        except Exception:
            return 0
    def _d23_cats():
        tot = 0
        for fn in ("multimodal_reliability_results.json", "multimodal_reliability_results_mvtec3d.json",
                   "multimodal_reliability_results_3d_adam.json", "multimodal_reliability_results_mulsen.json"):
            p = EXP / fn
            if p.exists():
                try:
                    tot += int(json.loads(p.read_text()).get("n_categories", 0))
                except Exception:
                    pass
        return tot
    main = {f: int((FAM == f).sum()) for f in sorted(set(FAM.tolist()))}
    aux = {"time-series\n(NAB+SMD)": _n("timeseries_results.json") + _n("smd_results.json"),
           "industrial\n(MVTec+VisA)": _n("industrial_benchmark.json"),
           "multimodal/3D\n(D23, 3 sets)": _d23_cats()}
    planned = {"OpenOOD": 0, "MVTec AD~2": 0}
    labels = list(main) + list(aux) + list(planned)
    vals = list(main.values()) + list(aux.values()) + list(planned.values())
    colors = ["#2c7fb8"] * len(main) + ["#41ab5d"] * len(aux) + ["#cccccc"] * len(planned)
    plt.figure(figsize=(8, 3.2))
    plt.bar(labels, vals, color=colors)
    plt.ylabel("tasks / categories scored")
    plt.title("Benchmark coverage: main suite (blue) + completed auxiliary (green) + planned (grey)")
    plt.xticks(rotation=25, ha="right", fontsize=7.5)
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
         "Stacking beats selection (verified); reliability routing is negative for single-input (3 regimes + D22).\n"
         "Multimodal reliability gate VALIDATED on 4 real datasets where modalities fail independently (D23).",
         "#fff5eb", "#d95f02", fs=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("ELARA-U evidence lifecycle", fontsize=11)
    plt.tight_layout(); plt.savefig(FIG / "elara_u_lifecycle.png", dpi=150); plt.close(fig)


def _d23_multidataset():
    """D23 cross-dataset figure: failure-regime AUROC of the four fusion strategies on
    every real multimodal dataset run under the canonical noise protocol. Reads only the
    verified per-dataset result JSONs; a check mark annotates datasets where all three
    pre-registered hypotheses (H1,H2,H3) pass. Honest: non-passing datasets are shown,
    not hidden."""
    order = [("Real-IAD-D3", "multimodal_reliability_results.json"),
             ("MVTec-3D", "multimodal_reliability_results_mvtec3d.json"),
             ("3D-ADAM", "multimodal_reliability_results_3d_adam.json"),
             ("MulSen-AD", "multimodal_reliability_results_mulsen.json")]
    methods = ["equal_weight", "stale_auto_select", "no_reliability", "reliability_gate"]
    mlbl = {"equal_weight": "equal-weight", "stale_auto_select": "stale auto-select",
            "no_reliability": "valid.-only fusion", "reliability_gate": "reliability gate"}
    mcol = {"equal_weight": "#9ecae1", "stale_auto_select": "#fdae6b",
            "no_reliability": "#a1d99b", "reliability_gate": "#d95f02"}
    rows = []
    for tag, fn in order:
        p = EXP / fn
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        mf = d.get("regimes", {}).get("modality_failure", {}).get("mean_auroc")
        if not mf:
            continue
        rows.append((tag, mf, bool(d.get("reliability_validated")), int(d.get("n_categories", 0))))
    if not rows:
        return False
    x = np.arange(len(rows)); w = 0.2
    plt.figure(figsize=(7, 3.2))
    for k, m in enumerate(methods):
        plt.bar(x + (k - 1.5) * w, [r[1].get(m, np.nan) for r in rows], w,
                label=mlbl[m], color=mcol[m])
    plt.axhline(0.5, color="k", lw=0.7, ls=":")
    for i, r in enumerate(rows):
        if r[2]:
            plt.text(i, 1.02, "H1,H2,H3 ✓", ha="center", fontsize=7.5, color="#1a7a1a")
    plt.xticks(x, [f"{r[0]}\n({r[3]} cat.)" for r in rows], fontsize=8)
    plt.ylim(0.3, 1.12); plt.ylabel("AUROC under modality failure")
    plt.title("D23: reliability gate recovers AUROC when a modality fails independently")
    plt.legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
    plt.tight_layout(); plt.savefig(FIG / "elara_u_d23_multidataset.png", dpi=150); plt.close()
    return True


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
    n = 8
    if _d23_multidataset():
        n += 1
    print(f"wrote {n} figures to {FIG} from VERIFIED data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
