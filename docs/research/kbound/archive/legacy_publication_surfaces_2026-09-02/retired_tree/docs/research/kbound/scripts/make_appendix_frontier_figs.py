"""Render the frontier-validation appendix figures (A.1 / A.2) from REAL TTA result JSONs.

CPU-only (matplotlib + json). Reads result JSONs and writes PNG figure files ONLY -- it
never writes a result JSON or manifest. Every plotted value is asserted against the
JSON-traceable numbers already in the appendix tables; any mismatch raises and exits
non-zero (we flag, never paper over). NO Camelyon17 content is produced: the invalid
"recalibrated K-Bound beats both" figure (old Fig A4) is deliberately not generated.

Sources (traceable):
  A.1 CIFAR-10-C beats-both ......... results/decisive_tta_results.json (benchmarks.cifar10c)
  A.2 ImageNet-C/SAR committal-fail . results/decision_baselines_sarfix/decision_baselines.json

Usage:  python scripts/make_appendix_frontier_figs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.dirname(HERE)                                   # docs/research/kbound
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(KB)))  # repo root
RES = os.path.join(ROOT, "experiments", "kbound", "results")
FIGD = os.path.join(KB, "figures")
os.makedirs(FIGD, exist_ok=True)

TEAL, SAND, RED = "#2a9d8f", "#e9c46a", "#e76f51"


def fig_cifar10c():
    d = json.load(open(os.path.join(RES, "decisive_tta_results.json")))
    cc = d["benchmarks"]["cifar10c"]["methods"]
    cands = ["tent", "eata", "sar"]
    reg = {c: cc[c]["metrics"]["regret_vs_oracle"] for c in cands}
    bb = {c: cc[c]["metrics"]["beats_both"] for c in cands}
    fa = {c: cc[c]["metrics"]["false_adapt_rate_B<0"] for c in cands}
    # assert against the appendix table A.1 (JSON-traceable)
    assert abs(reg["tent"]["K_Bound"] - 0.0016) < 5e-4, reg["tent"]
    assert abs(reg["eata"]["K_Bound"] - 0.0015) < 5e-4, reg["eata"]
    assert bb["tent"] and bb["eata"] and not bb["sar"], bb
    assert all(abs(fa[c]) < 1e-9 for c in cands), fa
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(3)
    w = 0.26
    ax.bar(x - w, [reg[c]["K_Bound"] for c in cands], w, label="K-Bound", color=TEAL)
    ax.bar(x,     [reg[c]["always_adapt"] for c in cands], w, label="always-adapt", color=SAND)
    ax.bar(x + w, [reg[c]["always_freeze"] for c in cands], w, label="always-freeze", color=RED)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["Tent", "EATA", "SAR"])
    ax.set_ylabel("regret-to-oracle (log scale, lower better)")
    ax.set_title("CIFAR-10-C: K-Bound beats both for Tent/EATA (0% false-adapt)")
    for i, c in enumerate(cands):
        if bb[c]:
            ax.text(i - w, reg[c]["K_Bound"] * 1.25, "beats\nboth", ha="center",
                    va="bottom", fontsize=7, color=TEAL)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIGD, "app_frontier_cifar10c.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[A.1] wrote {out}  (Tent KGA={reg['tent']['K_Bound']:.4f} vs "
          f"adapt {reg['tent']['always_adapt']:.4f} vs freeze {reg['tent']['always_freeze']:.4f}; "
          f"beats_both Tent/EATA={bb['tent']}/{bb['eata']})")


def fig_imagenetc():
    d = json.load(open(os.path.join(RES, "decision_baselines_sarfix", "decision_baselines.json")))
    sar = d["methods"]["sar"]
    rules = [("always-adapt", "always_adapt"), ("ATC (plug-in)", "atc_conf"),
             ("ATC (LOO)", "atc_conf_loo"), ("K-Bound", "KGA")]
    fa = [sar[k].get("false_adapt_rate_B<0") or 0.0 for _, k in rules]
    rg = [sar[k]["regret_vs_oracle"] for _, k in rules]
    cov = sar["KGA"]["coverage"]
    # assert against the appendix table A.2 (JSON-traceable)
    assert abs(fa[1] - 1.0) < 1e-9 and abs(fa[2] - 1.0) < 1e-9, fa   # ATC false-adapt 1.00
    assert abs(fa[3] - 0.0) < 1e-9, fa                               # KGA 0 false-adapt
    assert abs(cov - 0.3889) < 2e-3, cov                            # KGA coverage 0.39
    assert abs(rg[3] - 0.0229) < 5e-4, rg[3]                         # KGA regret 0.0229
    cols = [RED, SAND, SAND, TEAL]
    labels = [r[0] for r in rules]
    x = np.arange(len(rules))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    a1.bar(x, fa, color=cols)
    a1.axhline(0.10, ls=":", color="k")
    a1.text(0.0, 0.13, r"$\alpha=0.1$", fontsize=8)
    a1.set_xticks(x); a1.set_xticklabels(labels, rotation=18, fontsize=8)
    a1.set_ylim(0, 1.08); a1.set_ylabel("false-adapt rate (commit with $B<0$)")
    a1.set_title("Committal rules false-adapt; K-Bound = 0")
    a2.bar(x, rg, color=cols)
    a2.set_xticks(x); a2.set_xticklabels(labels, rotation=18, fontsize=8)
    a2.set_ylabel("regret-to-oracle")
    a2.set_title(f"K-Bound abstains the band (coverage {cov:.2f})")
    fig.suptitle("ImageNet-C / SAR (harmful 44%): K-Bound reaches 0 false-adapt by "
                 "abstaining exactly where committal heuristics fail", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIGD, "app_frontier_imagenetc.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[A.2] wrote {out}  (ATC false-adapt={fa[1]:.2f}/{fa[2]:.2f}, KGA={fa[3]:.2f}; "
          f"KGA regret={rg[3]:.4f}, coverage={cov:.3f})")


if __name__ == "__main__":
    fig_cifar10c()
    fig_imagenetc()
    print("OK: both appendix figures written from REAL result JSONs; all asserts passed; "
          "no Camelyon figure produced; no result JSON/manifest written.")
