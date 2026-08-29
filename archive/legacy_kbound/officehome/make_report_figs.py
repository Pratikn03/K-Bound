"""Generate figures for the Office-Home K-Bound report from the result JSONs."""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "wilds"))
import oh_analyze as oha

# Results dir resolved from this file's location so the figure build is
# checkout-independent.  Was hard-coded to the author's external volume until
# 2026-07-26 (fix-queue item 30 / defect D8).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES = os.environ.get("KBOUND_RESULTS_ROOT", os.path.join(_REPO, "experiments", "kbound", "results"))
OUT = os.path.join(RES, "officehome_full_FINAL")
def newest(d):
    import glob
    return sorted(glob.glob(f"{RES}/{d}/result_*.json"), key=os.path.getmtime)[-1]
def load(p):
    _, r, c, n = oha.load(p); return oha.recompute(r, c), n
src, names = load(newest("officehome_full_source"))
tv, _ = load(newest("officehome_full_targetval"))
tt, _ = load(newest("officehome_full_targettest"))
gv = oha.grad_detectability(src, tv, names)
gt = oha.grad_detectability(src, tt, names)
DOMS = ["Art", "Clipart", "Product"]

# ---- Fig 1: detectability vs mixedness (goldilocks corner empty) ----
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.axhspan(0.75, 1.0, xmin=(0.15)/0.5*0  + 0, alpha=0)  # noop
# goldilocks corner: harm>=0.15 AND transfer-AUC>=0.75
ax.add_patch(plt.Rectangle((0.15, 0.75), 0.85, 0.25, color="#2E7D32", alpha=0.10))
ax.text(0.56, 0.95, "GOLDILOCKS corner\n(mixed AND detectable)\n— required for beats-both", ha="center", va="top", fontsize=9, color="#1B5E20")
for split, g, mk, col in [("val", gv, "o", "#1565C0"), ("test", gt, "s", "#C62828")]:
    for d in DOMS:
        v = g["per_domain"][d]
        x = v["base_rate_harmful"]; y = v["certificate_transfer_AUC"]
        if y is None: continue
        ax.scatter(x, y, marker=mk, s=120, color=col, edgecolor="k", zorder=5)
        ax.annotate(f"{d}", (x, y), textcoords="offset points", xytext=(7, 5), fontsize=8.5)
ax.axhline(0.5, ls="--", color="grey", lw=1); ax.text(0.012, 0.505, "chance (0.5)", fontsize=8, color="grey")
ax.axhline(0.75, ls=":", color="#2E7D32", lw=1); ax.text(0.012, 0.755, "detectable (0.75)", fontsize=8, color="#2E7D32")
ax.axvline(0.15, ls="--", color="grey", lw=1); ax.text(0.16, 0.30, "mixed >=15% harm ->", fontsize=8, color="grey", rotation=90, va="center")
ax.set_xlabel("mixedness  =  base-rate harmful (fraction of conditions with B<0)")
ax.set_ylabel("detectability  =  source->target certificate transfer-AUC")
ax.set_xlim(0, 0.55); ax.set_ylim(0.3, 1.02)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor="#1565C0",markeredgecolor="k",markersize=10,label="VAL"),
                   Line2D([0],[0],marker="s",color="w",markerfacecolor="#C62828",markeredgecolor="k",markersize=10,label="HELD-OUT TEST")], loc="lower right", fontsize=9)
ax.set_title("Gradient-TTA harm: detectability vs mixedness (per domain)\nGoldilocks corner is EMPTY -> no beats-both", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_detect_vs_mixed.png"), dpi=160); plt.close()

# ---- Fig 2: per-candidate mean B (val) by family ----
cands = sorted(set(r["candidate"] for r in tv))
def fam(c): return "label-shift" if c=="labelshift" else ("conservative" if c=="conservative" else "gradient TTA")
colmap = {"gradient TTA": "#1565C0", "label-shift": "#C62828", "conservative": "#F9A825"}
mb = {c: float(np.mean([r["B"] for r in tv if r["candidate"]==c])) for c in cands}
order = sorted(cands, key=lambda c: mb[c])
fig, ax = plt.subplots(figsize=(7.6, 5.4))
ax.barh([c for c in order], [mb[c] for c in order], color=[colmap[fam(c)] for c in order], edgecolor="k", lw=0.4)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("mean benefit  B = acc(adapted) - acc(frozen)   (target VAL, pooled over domains/conditions)")
ax.set_title("Per-candidate mean benefit (val): label-shift catastrophic,\nconservative mildly harmful, gradient-TTA mildly helpful", fontsize=11)
ax.legend(handles=[plt.Rectangle((0,0),1,1,color=colmap[k]) for k in colmap], labels=list(colmap), fontsize=9, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_candidate_benefit.png"), dpi=160); plt.close()

# ---- Fig 3: held-out test router regret ----
vt = json.load(open(os.path.join(OUT, "VERDICT_test.json")))
ra = vt["route_a_deployed"]["regret_vs_oracle"]; rb = vt["route_b_multicandidate"]["regret_vs_oracle"]
labels = ["always-\nadapt", "best-fixed-\nadapt", "K-Bound\n(route-b)", "always-\nfreeze"]
vals = [ra["always_adapt"], rb["best_fixed_always_adapt"], rb["router"], rb["always_freeze"]]
cols = ["#1565C0", "#6A1B9A", "#2E7D32", "#9E9E9E"]
fig, ax = plt.subplots(figsize=(6.6, 4.6))
b = ax.bar(labels, vals, color=cols, edgecolor="k", lw=0.5)
for r, v in zip(b, vals): ax.text(r.get_x()+r.get_width()/2, v+0.0006, f"{v:.4f}", ha="center", fontsize=9)
ax.set_ylabel("regret to per-condition oracle  (lower=better)")
ax.set_title("HELD-OUT TEST regret: K-Bound does NOT beat both\n(ties always-freeze; loses to always-adapt & best-fixed-adapt)", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_test_regret.png"), dpi=160); plt.close()
print("figs written to", OUT)
print("fig1 cert-transfer pts val:", {d: round(gv["per_domain"][d]["certificate_transfer_AUC"],3) for d in DOMS})
print("fig3 regret:", {l.replace(chr(10),""):round(v,4) for l,v in zip(labels,vals)})
