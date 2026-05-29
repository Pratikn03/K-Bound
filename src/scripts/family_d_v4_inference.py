"""Family-D v4 EXPLORATORY inference.

Computes both ROC-AUC delta AND Brier score delta per endpoint. Brier is
the matched primary metric for a reliability-weighted method because it
rewards both ranking and calibrated magnitude — unlike ROC-AUC which is
rank-invariant under the v3 hard-collapse operator.

Reads test-fold metadata.yaml `anomalous` flags for the v4 exploratory
fold. This reuses the same Eyecandies test split as v3 (no new label
exposure beyond what the v3 sign-off already authorised).

Decision rules per cell:
  V4_POSITIVE_DELTA_OBSERVED:
    mean_per_seed_delta > 0 AND (per-seed paired t-test p <= 0.05)
    AND bootstrap CI excludes 0
  V4_NULL_DELTA_REPRODUCED:
    not statistically significant in either direction
  V4_NEGATIVE_DELTA_OBSERVED:
    mean_per_seed_delta < 0 AND (per-seed paired t-test p <= 0.05)
    AND bootstrap CI excludes 0

Outputs:
  experiments/phase2/family_d/family_d_v4_primary_inference.csv
  experiments/phase2/family_d/family_d_v4_holm_k2.csv
"""

from __future__ import annotations

import csv
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ARCHIVE = Path(ROOT, "data", "raw", "eyecandies", "_archives")
PRED_DIR = Path(ROOT, "experiments", "phase2", "family_d", "predictions_v4")
OUT_DIR = Path(ROOT, "experiments", "phase2", "family_d")

PRIMARY_ENDPOINTS = ("D-EYE-1v4", "D-EYE-2v4")
SECONDARY_DESCRIPTIVE = ("D-EYE-3v4",)
BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 0
PRACTICAL_DELTA_AUC = 0.005   # v4 exploratory: looser than v3's 0.01


def _read_eyecandies_test_labels() -> dict[str, int]:
    labels: dict[str, int] = {}
    for tar_path in sorted(ARCHIVE.glob("*.tar")):
        if tar_path.name.startswith("._"):
            continue
        cat = tar_path.stem
        with tarfile.open(tar_path, "r") as tf:
            for m in tf:
                if not m.isfile() or "metadata.yaml" not in m.name:
                    continue
                parts = m.name.split("/")
                split = None
                for p in parts:
                    if p in ("test_public", "test_private"):
                        split = p
                        break
                if split is None:
                    continue
                base = parts[-1]
                try:
                    sample_id = base.split("_")[0]
                except IndexError:
                    continue
                if not sample_id.isdigit():
                    continue
                key = f"{cat}__{split}__{sample_id}"
                content = tf.extractfile(m).read().decode("utf-8")
                meta = yaml.safe_load(content)
                labels[key] = int(meta.get("anomalous", 0))
    return labels


def _load_per_seed_predictions(endpoint: str, method: str) -> dict[int, pd.DataFrame]:
    out = {}
    if not PRED_DIR.exists():
        return out
    for cell_dir in PRED_DIR.iterdir():
        if not cell_dir.is_dir() or cell_dir.name.startswith("._"):
            continue
        if not cell_dir.name.startswith(f"{endpoint}__"):
            continue
        for d in cell_dir.iterdir():
            if not d.is_dir() or d.name.startswith("._"):
                continue
            if d.name == method or d.name.startswith(f"{method}_"):
                test_dir = d / "test"
                if not test_dir.exists():
                    continue
                # parquet stem can be "seed_42" or "seed_42__rerun_1" if the
                # archive saw a duplicate write. Take the LAST rerun (most recent)
                # per seed, otherwise the bare "seed_N".
                by_seed: dict[int, Path] = {}
                rerun_by_seed: dict[int, tuple[int, Path]] = {}
                for p in sorted(test_dir.glob("seed_*.parquet")):
                    if p.name.startswith("._"):
                        continue
                    stem = p.stem
                    if "__rerun_" in stem:
                        base, _, rer = stem.partition("__rerun_")
                        try:
                            seed = int(base.replace("seed_", ""))
                            rer_n = int(rer)
                        except ValueError:
                            continue
                        prev = rerun_by_seed.get(seed)
                        if prev is None or rer_n > prev[0]:
                            rerun_by_seed[seed] = (rer_n, p)
                    else:
                        try:
                            seed = int(stem.replace("seed_", ""))
                        except ValueError:
                            continue
                        by_seed[seed] = p
                for seed, (_, path) in rerun_by_seed.items():
                    by_seed[seed] = path
                for seed, path in by_seed.items():
                    out[seed] = pd.read_parquet(path)
                break
    return out


def _stack_ensemble(per_seed, labels):
    seeds = sorted(per_seed.keys())
    s0 = per_seed[seeds[0]]
    sids_all = s0["sample_id"].to_numpy()
    keep_mask = np.array([sid in labels for sid in sids_all])
    sids = sids_all[keep_mask]
    lbl = np.array([labels[sid] for sid in sids], dtype=int)
    per_seed_scores = {}
    for s in seeds:
        df = per_seed[s]
        arr = df["raw_score"].to_numpy().astype(float)
        per_seed_scores[s] = arr[keep_mask]
    return sids, lbl, per_seed_scores


def _brier(scores, labels):
    s = np.clip(scores, 0.0, 1.0)
    return float(np.mean((s - labels.astype(float)) ** 2))


def _per_seed_aucs_and_briers(scores_by_seed, labels):
    seeds = sorted(scores_by_seed.keys())
    aucs, briers = [], []
    for s in seeds:
        sc = scores_by_seed[s]
        try:
            aucs.append(roc_auc_score(labels, sc))
        except ValueError:
            aucs.append(np.nan)
        briers.append(_brier(sc, labels))
    return np.array(aucs), np.array(briers)


def _paired_bootstrap_ci(values, n_iter=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED, ci=0.95):
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        means[i] = np.mean(values[idx])
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return lo, hi


def _decision_auc(mean_delta, p, ci_lo, ci_hi, thresh):
    if np.isnan(mean_delta):
        return "INVALID"
    significant = (p <= 0.05) and (ci_lo > 0 or ci_hi < 0)
    if not significant:
        return "V4_NULL_DELTA_REPRODUCED"
    if mean_delta > 0 and mean_delta >= thresh:
        return "V4_POSITIVE_DELTA_OBSERVED"
    if mean_delta < 0 and abs(mean_delta) >= thresh:
        return "V4_NEGATIVE_DELTA_OBSERVED"
    return "V4_NULL_DELTA_REPRODUCED"


def _decision_brier(mean_delta, p, ci_lo, ci_hi, thresh):
    """For Brier, NEGATIVE delta (RGA - static < 0) means RGA is BETTER calibrated."""
    if np.isnan(mean_delta):
        return "INVALID"
    significant = (p <= 0.05) and (ci_lo > 0 or ci_hi < 0)
    if not significant:
        return "V4_NULL_DELTA_REPRODUCED"
    if mean_delta < 0 and abs(mean_delta) >= thresh:
        return "V4_POSITIVE_DELTA_OBSERVED"   # better calibrated
    if mean_delta > 0 and mean_delta >= thresh:
        return "V4_NEGATIVE_DELTA_OBSERVED"   # worse calibrated
    return "V4_NULL_DELTA_REPRODUCED"


def holm_bonferroni(p_map, K):
    sorted_eps = sorted(p_map.items(), key=lambda kv: kv[1])
    adjusted = {}
    for i, (ep, p) in enumerate(sorted_eps):
        adj = min(1.0, p * (K - i))
        if i > 0:
            prev_ep = sorted_eps[i - 1][0]
            adj = max(adj, adjusted[prev_ep])
        adjusted[ep] = adj
    return adjusted


def main():
    print("Reading Eyecandies test labels (uses v3-authorised label fold)...", flush=True)
    labels = _read_eyecandies_test_labels()
    print(f"  loaded {len(labels)} labels ({sum(labels.values())} anomalous)")

    rows = []
    raw_p_auc = {}
    raw_p_brier = {}

    for endpoint in PRIMARY_ENDPOINTS:
        static = _load_per_seed_predictions(endpoint, "static_attention")
        rga = _load_per_seed_predictions(endpoint, "base_RGA")
        if not static or not rga:
            print(f"[{endpoint}] missing predictions; skipping")
            continue
        seeds = sorted(set(static) & set(rga))
        sids_s, lbl_s, scores_s = _stack_ensemble({s: static[s] for s in seeds}, labels)
        sids_r, lbl_r, scores_r = _stack_ensemble({s: rga[s] for s in seeds}, labels)
        if not np.array_equal(sids_s, sids_r) or not np.array_equal(lbl_s, lbl_r):
            raise SystemExit(f"{endpoint}: alignment mismatch")
        lbl = lbl_s

        aucs_s, briers_s = _per_seed_aucs_and_briers(scores_s, lbl)
        aucs_r, briers_r = _per_seed_aucs_and_briers(scores_r, lbl)
        delta_auc = aucs_r - aucs_s
        delta_brier = briers_r - briers_s

        # AUC inference
        mean_dauc = float(np.nanmean(delta_auc))
        t_auc = stats.ttest_rel(aucs_r, aucs_s, nan_policy="omit")
        ci_auc_lo, ci_auc_hi = _paired_bootstrap_ci(delta_auc[~np.isnan(delta_auc)])

        # Brier inference
        mean_dbrier = float(np.mean(delta_brier))
        t_brier = stats.ttest_rel(briers_r, briers_s)
        ci_brier_lo, ci_brier_hi = _paired_bootstrap_ci(delta_brier)

        rows.append({
            "endpoint": endpoint,
            "n_seeds": len(seeds),
            "n_test_samples": len(lbl),
            "mean_static_auc": float(np.nanmean(aucs_s)),
            "mean_rga_auc": float(np.nanmean(aucs_r)),
            "mean_delta_auc": mean_dauc,
            "ttest_p_auc": float(t_auc.pvalue),
            "ci_auc_lo": ci_auc_lo,
            "ci_auc_hi": ci_auc_hi,
            "mean_static_brier": float(np.mean(briers_s)),
            "mean_rga_brier": float(np.mean(briers_r)),
            "mean_delta_brier": mean_dbrier,
            "ttest_p_brier": float(t_brier.pvalue),
            "ci_brier_lo": ci_brier_lo,
            "ci_brier_hi": ci_brier_hi,
            "n_sign_pos_auc": int((delta_auc > 0).sum()),
            "n_sign_neg_auc": int((delta_auc < 0).sum()),
        })
        raw_p_auc[endpoint] = float(t_auc.pvalue)
        raw_p_brier[endpoint] = float(t_brier.pvalue)

    if not rows:
        print("no v4 primary predictions found; aborting")
        return 1

    holm_auc = holm_bonferroni(raw_p_auc, K=2)
    holm_brier = holm_bonferroni(raw_p_brier, K=2)
    for r in rows:
        r["holm_p_auc"] = holm_auc[r["endpoint"]]
        r["holm_p_brier"] = holm_brier[r["endpoint"]]
        r["decision_auc"] = _decision_auc(
            r["mean_delta_auc"], r["holm_p_auc"],
            r["ci_auc_lo"], r["ci_auc_hi"], PRACTICAL_DELTA_AUC)
        r["decision_brier"] = _decision_brier(
            r["mean_delta_brier"], r["holm_p_brier"],
            r["ci_brier_lo"], r["ci_brier_hi"], 0.0005)

    # Family decision
    positive_auc = sum(r["decision_auc"] == "V4_POSITIVE_DELTA_OBSERVED" for r in rows)
    positive_brier = sum(r["decision_brier"] == "V4_POSITIVE_DELTA_OBSERVED" for r in rows)
    negative_any = any(
        r["decision_auc"] == "V4_NEGATIVE_DELTA_OBSERVED" or
        r["decision_brier"] == "V4_NEGATIVE_DELTA_OBSERVED"
        for r in rows
    )
    if positive_auc + positive_brier >= 2 and not negative_any:
        family = "FAMILY_D_V4_EXPLORATORY_POSITIVE_SIGNAL_OBSERVED"
    elif positive_brier >= 1 and not negative_any:
        family = "FAMILY_D_V4_EXPLORATORY_PARTIAL_BRIER_SIGNAL"
    elif negative_any:
        family = "FAMILY_D_V4_EXPLORATORY_NEGATIVE_SIGNAL_OBSERVED"
    else:
        family = "FAMILY_D_V4_EXPLORATORY_NULL_REPRODUCED"

    primary_csv = OUT_DIR / "family_d_v4_primary_inference.csv"
    fields = list(rows[0].keys())
    with primary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {primary_csv}")

    holm_csv = OUT_DIR / "family_d_v4_holm_k2.csv"
    with holm_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "endpoint", "mean_delta_auc", "ci_auc_lo", "ci_auc_hi", "ttest_p_auc",
            "holm_p_auc", "decision_auc", "mean_delta_brier", "ci_brier_lo",
            "ci_brier_hi", "ttest_p_brier", "holm_p_brier", "decision_brier",
        ])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"wrote {holm_csv}")

    print()
    print("=== Family-D v4 EXPLORATORY primary results ===")
    for r in rows:
        print(
            f"{r['endpoint']}: AUC static={r['mean_static_auc']:.4f} "
            f"rga={r['mean_rga_auc']:.4f}  Δ={r['mean_delta_auc']:+.4f}  "
            f"CI=[{r['ci_auc_lo']:+.4f},{r['ci_auc_hi']:+.4f}]  "
            f"holm_p={r['holm_p_auc']:.3g}  → {r['decision_auc']}"
        )
        print(
            f"             Brier static={r['mean_static_brier']:.4f} "
            f"rga={r['mean_rga_brier']:.4f}  Δ={r['mean_delta_brier']:+.5f}  "
            f"CI=[{r['ci_brier_lo']:+.5f},{r['ci_brier_hi']:+.5f}]  "
            f"holm_p={r['holm_p_brier']:.3g}  → {r['decision_brier']}"
        )
    print(f"\nFamily v4 decision: {family}")

    with (OUT_DIR / "family_d_v4_family_decision.txt").open("w") as f:
        f.write(family + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
