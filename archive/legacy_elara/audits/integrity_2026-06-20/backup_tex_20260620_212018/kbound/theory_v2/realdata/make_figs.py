import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/theory_v2/realdata"
p1 = json.load(open(os.path.join(HERE, "_p1_partial.json")))["P1"]
p2 = json.load(open(os.path.join(HERE, "_p2_partial.json")))["P2"]

# ---------- FIG 1: sign-recovery accuracy vs tau (P1) ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for col, mode, title in [(0, "per_task_median", "median thresholds"),
                         (1, "per_task_val_opt", "val-optimal thresholds")]:
    rows = [r for r in p1[mode] if "skipped" not in r and r["sign_ok"] is not None
            and r["tau"] is not None and r["tau_null_q95"] is not None]
    tau = np.array([r["tau"] for r in rows])
    q95 = np.array([r["tau_null_q95"] for r in rows])
    ok = np.array([1 if r["sign_ok"] else 0 for r in rows])
    Hrej = np.array([r["H_reject"] for r in rows])
    # x-axis: tau / q95 (>1 => H rejected). log scale.
    ratio = tau / np.maximum(q95, 1e-6)
    jit = (np.random.default_rng(1).random(len(ok)) - 0.5) * 0.12
    c = np.where(ok == 1, "#2ca02c", "#d62728")
    ax[col].scatter(ratio, ok + jit, c=c, s=45, edgecolor="k", linewidth=0.4, alpha=0.85)
    ax[col].axvline(1.0, color="gray", ls="--", lw=1.2, label="H-reject boundary (tau=q95)")
    ax[col].set_xscale("log")
    ax[col].set_yticks([0, 1]); ax[col].set_yticklabels(["sign WRONG", "sign CORRECT"])
    ax[col].set_xlabel("tau / null-q95  (>1 => H rejected)")
    sm = p1["summary_median" if "median" in mode else "summary_val_opt"]
    pa = sm["sign_acc_H_pass"][0]; ra = sm["sign_acc_H_reject"][0]
    npass = sm["n_H_pass"]; nrej = sm["n_H_reject"]
    ax[col].set_title(f"P1 sign-recovery vs tau ({title})\n"
                      f"H-pass acc={pa:.2f} (n={npass}) | H-reject acc={ra:.2f} (n={nrej})",
                      fontsize=10)
    ax[col].legend(fontsize=8, loc="center right"); ax[col].grid(alpha=0.25)
    ax[col].set_ylim(-0.4, 1.4)
fig.suptitle("P1: K-Bound Theory V2 on 123-task anomaly bank — relative-sign recovery vs H-falsification (tau)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(HERE, "fig_p1_sign_vs_tau.png"), dpi=130)
plt.close(fig)
print("wrote fig_p1_sign_vs_tau.png")

# ---------- FIG 2: CI forest plot (P2) ----------
comps = p2["comparisons"]
names = list(comps.keys())
# order: group freeze comparisons then adapt comparisons
names = sorted(names, key=lambda n: (("adapt" in n), n))
means = [comps[n]["mean_diff_KGA_minus_baseline"] for n in names]
los = [comps[n]["boot95_CI"][0] for n in names]
his = [comps[n]["boot95_CI"][1] for n in names]
surv = [comps[n]["holm"]["reject_holm"] for n in names]
y = np.arange(len(names))[::-1]
fig, ax = plt.subplots(figsize=(10.5, 5.2))
for i, n in enumerate(names):
    yy = y[i]
    col = "#2ca02c" if surv[i] else "#888888"
    ax.plot([los[i], his[i]], [yy, yy], color=col, lw=3, solid_capstyle="round")
    ax.plot(means[i], yy, "o", color=col, ms=9, markeredgecolor="k")
ax.axvline(0, color="red", ls="--", lw=1.3, label="no difference (Δregret=0)")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("Δ regret-to-oracle  (KGA − baseline);  negative = KGA better")
ax.set_title("P2: per-condition paired bootstrap on REAL 65-cell CIFAR-10-C grid (Holm-corrected)\n"
             "GREEN = survives Holm; GREY = does not.  "
             "KGA beats always-FREEZE (survives); ties always-ADAPT (does not)",
             fontsize=10)
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="x", alpha=0.3)
# annotate survivors
for i, n in enumerate(names):
    tag = "survives Holm" if surv[i] else "n.s."
    ax.text(his[i] + 0.004, y[i], tag, va="center", fontsize=8,
            color=("#2ca02c" if surv[i] else "#888888"))
ax.set_xlim(min(los) - 0.02, max(his) + 0.05)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_p2_ci_forest.png"), dpi=130)
plt.close(fig)
print("wrote fig_p2_ci_forest.png")
