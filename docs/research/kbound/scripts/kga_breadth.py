"""
kga_breadth.py — CPU-only KGA breadth sweep over precomputed score archives.

Job B: For each of the 123 score-archive tasks in experiments/elara_u/score_archive/
  - f0 = best-val-AUC single detector (frozen baseline)
  - f_a = mean ensemble of all 6 detectors (the "adapted" system)
  - Z  = label-free score statistics computed from the TEST batch
  - B  = AUC(f_a, test) - AUC(f0, test)  [true benefit, used only for evaluation]
  Datasets are partitioned into leave-one-task-out folds for KGA (matching the
  paper's LOO gradient-boosted benefit estimator).  Each task is one "condition".

Why this is a valid KGA instantiation:
  - The decision axis is: use the multi-detector ensemble (f_a) or stay with the
    best single detector selected on val (f0).
  - Z is purely label-free (score distribution statistics of the unlabeled test batch).
  - B uses true test labels only for EVALUATION (never for the KGA decision).
  - This mirrors the paper's anomaly-detection "harmful adaptation" scenario: some
    datasets have negative benefit (ensemble hurts vs best single detector), some positive.

Skipped archives (stated reason):
  - cv_* (61 image-OOD files): Stest shapes are tiny for many (min=26); skew/kurtosis
    unreliable at <30 samples, and the domain is image_ood anomaly — not the
    covariate-shift adaptation scenario.  Documented below.
  - Files with <15 test samples: too few for a meaningful AUC estimate.

Outputs:
  experiments/kbound/results/breadth_existing_datasets.json
  docs/research/kbound/results/breadth_table.md
"""
from __future__ import annotations
import os, sys, json, glob, warnings
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../.."))
SCORE_ARCHIVE = os.path.join(REPO, "experiments/elara_u/score_archive")
OUT_JSON = os.path.join(REPO, "experiments/kbound/results/breadth_existing_datasets.json")
OUT_MD   = os.path.join(REPO, "docs/research/kbound/results/breadth_table.md")

ALPHA       = 0.10
N_EST       = 200
MAX_DEPTH   = 2
LR          = 0.05
SEED        = 0
MIN_SAMPLES = 15   # skip tasks with fewer test samples
MIN_TASKS   = 10   # need at least this many valid tasks for KGA to be meaningful


# ---------------------------------------------------------------------------
# Z featuriser — label-free score statistics of the test batch
# ---------------------------------------------------------------------------
def compute_Z(S: np.ndarray) -> np.ndarray:
    """S shape (N, D). Returns 1-D feature vector (no labels used)."""
    feats = []
    for j in range(S.shape[1]):
        col = S[:, j]
        feats += [col.mean(), col.std(ddof=1) if len(col) > 1 else 0.0,
                  float(scipy_stats.skew(col)), float(scipy_stats.kurtosis(col))]
    # pairwise cross-detector correlations (upper triangle)
    for i in range(S.shape[1]):
        for jj in range(i + 1, S.shape[1]):
            c = np.corrcoef(S[:, i], S[:, jj])[0, 1]
            feats.append(0.0 if np.isnan(c) else c)
    return np.array(feats, dtype=float)


# ---------------------------------------------------------------------------
# AUC helper
# ---------------------------------------------------------------------------
def safe_auc(y, score):
    if len(np.unique(y)) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(roc_auc_score(y, score))


# ---------------------------------------------------------------------------
# KGA decision rule — identical to cifar_tent_mps_v2.decide_kga
# ---------------------------------------------------------------------------
def decide_kga(Z, B, alpha=ALPHA):
    Z = np.asarray(Z, float)
    B = np.asarray(B, float)
    N = len(B)
    Bhat = np.zeros(N)
    for i in range(N):
        tr = np.arange(N) != i
        m = GradientBoostingRegressor(
            n_estimators=N_EST, max_depth=MAX_DEPTH,
            learning_rate=LR, subsample=0.8, random_state=SEED)
        m.fit(Z[tr], B[tr])
        Bhat[i] = m.predict(Z[i:i+1])[0]
    eps = float(np.quantile(np.abs(Bhat - B), 1 - alpha))
    dec = np.where(Bhat - eps > 0, "ADAPT",
                   np.where(Bhat + eps < 0, "FREEZE", "ABSTAIN"))
    return Bhat, eps, dec


# ---------------------------------------------------------------------------
# Policy metrics — identical to cifar_tent_mps_v2.policy_metrics
# ---------------------------------------------------------------------------
def policy_metrics(dec, a0, aa, B):
    a0, aa, B = map(np.asarray, (a0, aa, B))
    adapt = dec == "ADAPT"
    kga = np.where(adapt, aa, a0)
    oracle = np.maximum(a0, aa)
    out = {
        "n": int(len(a0)),
        "decision_counts": {d: int((dec == d).sum()) for d in ["ADAPT", "FREEZE", "ABSTAIN"]},
        "coverage": float(np.mean(dec != "ABSTAIN")),
        "false_adapt_rate_B<0": float(np.mean(B[adapt] < 0)) if adapt.any() else None,
        "adapt_precision_B>0": float(np.mean(B[adapt] > 0)) if adapt.any() else None,
        "mean_auc": {
            "always_adapt":  float(aa.mean()),
            "always_freeze": float(a0.mean()),
            "K_Bound":       float(kga.mean()),
            "oracle":        float(oracle.mean()),
        },
        "regret_vs_oracle": {
            "always_adapt":  float((oracle - aa).mean()),
            "always_freeze": float((oracle - a0).mean()),
            "K_Bound":       float((oracle - kga).mean()),
        },
        "beats_both": bool(
            (oracle - kga).mean() < (oracle - aa).mean() - 1e-9 and
            (oracle - kga).mean() < (oracle - a0).mean() - 1e-9
        ),
        "ties_adapt": bool(
            abs((oracle - kga).mean() - (oracle - aa).mean()) < 1e-9 and
            (oracle - kga).mean() <= (oracle - a0).mean() + 1e-9
        ),
        "loses": bool(
            (oracle - kga).mean() > (oracle - aa).mean() + 1e-9 or
            (oracle - kga).mean() > (oracle - a0).mean() + 1e-9
        ),
    }
    return out


# ---------------------------------------------------------------------------
# Load all usable tasks from the score archive
# ---------------------------------------------------------------------------
def load_tasks(archive_dir: str):
    """Load tabular + special-domain score archives; skip cv_* and tiny tasks."""
    tasks = []
    skipped = []

    # Priority: adb_* (tabular ADBench) + domain-specific (creditcard, online, nlp, cyber)
    # Skip cv_* (image_ood): 61 files, many <30 samples, not the adaptation scenario
    files = sorted(glob.glob(os.path.join(archive_dir, "*.npz")))
    cv_files = [f for f in files if os.path.basename(f).startswith("cv_")]
    task_files = [f for f in files if not os.path.basename(f).startswith("cv_")]

    skipped.append({
        "group": "cv_* (image_ood, 61 files)",
        "reason": ("Image OOD anomaly detection — not the covariate-shift adaptation "
                   "scenario. Many files have <30 test samples (skew/kurtosis unreliable). "
                   "f0/fa distinction collapses when only one detector consistently dominates "
                   "across all class-vs-rest splits."),
        "n_files": len(cv_files),
    })

    for fpath in task_files:
        name = os.path.basename(fpath).replace(".npz", "")
        d = np.load(fpath, allow_pickle=True)
        Stest = d["Stest"]  # (N, 6)
        ytest = d["ytest"]
        val_auc = d["val_auc"]
        det_names = [str(x) for x in d["det_names"]]
        domain = str(d["domain"])

        N = Stest.shape[0]
        if N < MIN_SAMPLES:
            skipped.append({"name": name, "reason": f"only {N} test samples < {MIN_SAMPLES}"})
            continue
        if len(np.unique(ytest)) < 2:
            skipped.append({"name": name, "reason": "single class in test set (no AUC)"})
            continue

        # f0 = best val-AUC single detector
        best_idx = int(np.argmax(val_auc))
        a0 = safe_auc(ytest, Stest[:, best_idx])
        if np.isnan(a0):
            skipped.append({"name": name, "reason": "AUC undefined for f0"})
            continue

        # f_a = mean-score ensemble of all 6 detectors
        ens_score = Stest.mean(axis=1)
        aa = safe_auc(ytest, ens_score)
        if np.isnan(aa):
            skipped.append({"name": name, "reason": "AUC undefined for f_a ensemble"})
            continue

        # Z = label-free score statistics of test batch
        Z = compute_Z(Stest)

        tasks.append({
            "name": name,
            "domain": domain,
            "n_test": N,
            "f0_det": det_names[best_idx],
            "f0_val_auc": float(val_auc[best_idx]),
            "a0": float(a0),
            "aa": float(aa),
            "B": float(aa - a0),
            "Z": Z.tolist(),
        })

    return tasks, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== KGA Breadth Sweep: Existing Score Archives ===")
    tasks, skipped = load_tasks(SCORE_ARCHIVE)
    print(f"Loaded {len(tasks)} valid tasks, skipped {len(skipped)} entries")

    if len(tasks) < MIN_TASKS:
        print(f"ERROR: too few tasks ({len(tasks)} < {MIN_TASKS}) for meaningful KGA sweep")
        sys.exit(1)

    # Assemble arrays
    Z_mat = np.array([t["Z"] for t in tasks], dtype=float)
    a0_arr = np.array([t["a0"] for t in tasks], dtype=float)
    aa_arr = np.array([t["aa"] for t in tasks], dtype=float)
    B_arr  = aa_arr - a0_arr

    print(f"B distribution: mean={B_arr.mean():.4f}, std={B_arr.std():.4f}, "
          f"min={B_arr.min():.4f}, max={B_arr.max():.4f}")
    print(f"Harmful (B<0): {int((B_arr<0).sum())}/{len(B_arr)}")

    # Run KGA
    print("Running LOO KGA (this may take ~2-3 min)...")
    Bhat, eps, dec = decide_kga(Z_mat, B_arr, alpha=ALPHA)

    pm = policy_metrics(dec, a0_arr, aa_arr, B_arr)
    pm["eps_conformal"] = float(eps)
    pm["alpha"] = ALPHA
    pm["base_rate_harmful_B<0"] = float(np.mean(B_arr < 0))
    pm["mean_true_B"] = float(B_arr.mean())

    # Per-task detail
    per_task = []
    for i, t in enumerate(tasks):
        kga_acc = aa_arr[i] if dec[i] == "ADAPT" else a0_arr[i]
        per_task.append({
            "name": t["name"],
            "domain": t["domain"],
            "n_test": t["n_test"],
            "f0_det": t["f0_det"],
            "a0": round(t["a0"], 4),
            "aa": round(t["aa"], 4),
            "B": round(t["B"], 4),
            "Bhat": round(float(Bhat[i]), 4),
            "decision": dec[i],
            "kga_auc": round(float(kga_acc), 4),
            "oracle_auc": round(float(max(t["a0"], t["aa"])), 4),
            "regime": ("harmful" if t["B"] < -0.02 else
                       "helpful" if t["B"] > 0.02 else "marginal"),
            "correct_decision": bool(
                (dec[i] == "ADAPT" and t["B"] > 0) or
                (dec[i] == "FREEZE" and t["B"] < 0) or
                dec[i] == "ABSTAIN"
            ),
        })

    # Assemble output
    result = {
        "description": (
            "KGA breadth sweep on 123-task score archive. "
            "f0=best-val-AUC single detector, f_a=mean-ensemble of 6, "
            "Z=label-free score statistics, B=AUC(f_a)-AUC(f0) on test."
        ),
        "alpha": ALPHA,
        "n_tasks": len(tasks),
        "skipped": skipped,
        "aggregate_metrics": pm,
        "per_task": per_task,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {OUT_JSON}")

    # Build markdown table
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    build_markdown(result, per_task, skipped)
    print(f"Saved: {OUT_MD}")

    # Summary
    print("\n=== RESULTS SUMMARY ===")
    print(f"N tasks: {len(tasks)}")
    print(f"Harmful base rate (B<0): {pm['base_rate_harmful_B<0']:.3f}")
    print(f"Conformal eps: {eps:.4f}")
    print(f"Decisions: {pm['decision_counts']}")
    print(f"Coverage: {pm['coverage']:.3f}")
    if pm['false_adapt_rate_B<0'] is not None:
        print(f"False-adapt rate (B<0 | ADAPT): {pm['false_adapt_rate_B<0']:.3f}")
    print("Regret vs oracle:")
    for pol, val in pm["regret_vs_oracle"].items():
        print(f"  {pol}: {val:.4f}")
    print(f"Beats both: {pm['beats_both']}")
    print(f"Ties adapt: {pm['ties_adapt']}")
    print(f"Loses: {pm['loses']}")


def build_markdown(result, per_task, skipped):
    pm = result["aggregate_metrics"]
    lines = [
        "# KGA Breadth Table — Existing Score Archives",
        "",
        "**Framework**: f0 = best-val-AUC single detector (frozen); "
        "f_a = mean-score ensemble of all 6 detectors (adapted); "
        "Z = label-free score statistics; B = AUC(f_a) − AUC(f0) on held-out test set.  ",
        "KGA decision rule: LOO gradient-boosted Bhat ± split-conformal ε (α=0.10), identical to cifar_tent_mps_v2.py.",
        "",
        "## Aggregate Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| N tasks | {result['n_tasks']} |",
        f"| Harmful base rate (B<0) | {pm['base_rate_harmful_B<0']:.3f} |",
        f"| Mean true B | {pm['mean_true_B']:.4f} |",
        f"| Conformal ε | {pm['eps_conformal']:.4f} |",
        f"| α (miscoverage target) | {pm['alpha']} |",
        f"| Coverage | {pm['coverage']:.3f} |",
        f"| ADAPT decisions | {pm['decision_counts']['ADAPT']} |",
        f"| FREEZE decisions | {pm['decision_counts']['FREEZE']} |",
        f"| ABSTAIN decisions | {pm['decision_counts']['ABSTAIN']} |",
        f"| False-adapt rate (B<0 given ADAPT) | {pm['false_adapt_rate_B<0'] if pm['false_adapt_rate_B<0'] is not None else 'N/A'} |",
        f"| Adapt precision (B>0 given ADAPT) | {pm['adapt_precision_B>0'] if pm['adapt_precision_B>0'] is not None else 'N/A'} |",
        f"| Mean AUC — always-adapt | {pm['mean_auc']['always_adapt']:.4f} |",
        f"| Mean AUC — always-freeze | {pm['mean_auc']['always_freeze']:.4f} |",
        f"| Mean AUC — K-Bound | {pm['mean_auc']['K_Bound']:.4f} |",
        f"| Mean AUC — oracle | {pm['mean_auc']['oracle']:.4f} |",
        f"| Regret vs oracle — always-adapt | {pm['regret_vs_oracle']['always_adapt']:.4f} |",
        f"| Regret vs oracle — always-freeze | {pm['regret_vs_oracle']['always_freeze']:.4f} |",
        f"| Regret vs oracle — K-Bound | {pm['regret_vs_oracle']['K_Bound']:.4f} |",
        f"| Beats both baselines | {pm['beats_both']} |",
        f"| Ties best baseline | {pm['ties_adapt']} |",
        f"| Loses to a baseline | {pm['loses']} |",
        "",
        "## Per-Task Results",
        "",
        "| # | Dataset | Domain | N | f0 | a0 | aa | B | Decision | KGA AUC | Oracle AUC | Regime |",
        "|---|---------|--------|---|----|----|----|----|----------|---------|------------|--------|",
    ]
    for i, t in enumerate(per_task, 1):
        lines.append(
            f"| {i} | {t['name']} | {t['domain']} | {t['n_test']} | {t['f0_det']} "
            f"| {t['a0']:.3f} | {t['aa']:.3f} | {t['B']:+.3f} "
            f"| {t['decision']} | {t['kga_auc']:.3f} | {t['oracle_auc']:.3f} | {t['regime']} |"
        )
    lines += [
        "",
        "## Skipped Archives",
        "",
        "| Entry | Reason |",
        "|-------|--------|",
    ]
    for s in skipped:
        name = s.get("name", s.get("group", "?"))
        lines.append(f"| {name} | {s['reason']} |")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
