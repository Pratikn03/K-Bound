"""Run the canonical category-averaged one-class degradation sweep under multiple modes.

Implements the standard one-class evaluation protocol:
1. Category-averaged test ROC-AUC.
2. Parameter-free confidence-weighted mean (CW) and RGA-gated-CW fusions.
3. Degradation sweeps on the depth channel under 3 modes:
   - Uniform Noise (general hardware noise)
   - Sensor Dropout (active sensor blockages / shadows)
   - Calibration Shift (temperature drift / bias)
4. Category-level KS-drift check using validation references.
5. Paired bootstrap statistics over category-averaged values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
MARGIN = 0.05
BOOT = 10000


def _pivot(df, split):
    """Pivot the long-format fusion CSV for one split into per-sample arrays:
    rgb/depth scores, rgb/depth confidences, and labels (aligned by sample_id)."""
    s = df[df["split"] == split]
    r = s[s.domain == "rgb"].set_index("sample_id")
    
    # Auto-detect second modality (non-RGB domain)
    other_domains = [dom for dom in s.domain.unique() if dom != "rgb"]
    second_domain = other_domains[0] if other_domains else "depth_or_xyz"
    d = s[s.domain == second_domain].set_index("sample_id")
    
    ids = r.index.intersection(d.index)
    one = np.ones(len(ids))
    return dict(
        rgb=r.loc[ids, "score"].to_numpy(), depth=d.loc[ids, "score"].to_numpy(),
        rc=r.loc[ids, "confidence"].to_numpy() if "confidence" in r else one,
        dc=d.loc[ids, "confidence"].to_numpy() if "confidence" in d else one,
        y=r.loc[ids, "label"].to_numpy().astype(int),
    )


def _ksr(t, ref):
    """KS-drift reliability of a modality: 1 - KS(test, validation-reference)."""
    return float(np.clip(1 - ks_2samp(t, ref).statistic, 0, 1)) if t.size >= 5 else 1.0


def _fast_auc(y_true, y_score):
    n_pos = np.sum(y_true)
    n_neg = y_true.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    r = np.argsort(np.argsort(y_score)) + 1
    u = np.sum(r[y_true == 1]) - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def _boot_delta_one_class(cats_data, alpha, mode="uniform", seed=0):
    """Paired bootstrap of category-averaged ROC-AUC delta: Gated-CW minus CW."""
    rng = np.random.default_rng(seed)
    
    # Pre-compute degraded depth, cw, and gcw scores for the full test set of each category
    precomputed = {}
    for c, (val, test) in cats_data.items():
        n = test["depth"].size
        # Apply degradation on the full test set
        if mode == "uniform":
            noise = rng.uniform(0, 1, size=n)
            depth_deg = (1 - alpha) * test["depth"] + alpha * noise
        elif mode == "dropout":
            mask = rng.random(size=n) < alpha
            depth_deg = np.where(mask, 1.0, test["depth"])
        elif mode == "shift":
            depth_deg = np.clip(test["depth"] + alpha * 0.5, 0.0, 1.0)
        else:
            depth_deg = test["depth"]
            
        cw = (test["rc"] * test["rgb"] + test["dc"] * depth_deg) / (test["rc"] + test["dc"] + 1e-9)
        
        # Reliability gates are fixed parameters of the fusion model for this test session
        rr = _ksr(test["rgb"], val["rgb"])
        rd = _ksr(depth_deg, val["depth"])
        rgawt = (rr * test["rgb"] + rd * depth_deg) / (rr + rd + 1e-9)
        
        rr0 = _ksr(test["rgb"], val["rgb"])
        rd0 = _ksr(test["depth"], val["depth"])
        tau = min(rr0, rd0) - MARGIN
        fires = bool(min(rr, rd) < tau)
        gcw = rgawt if fires else cw
        
        precomputed[c] = {
            "cw": cw,
            "gcw": gcw,
            "y": test["y"]
        }
        
    deltas = []
    for _ in range(BOOT):
        boot_cw = []
        boot_gated_cw = []
        
        for c, data in precomputed.items():
            y = data["y"]
            n = y.size
            idx = rng.integers(0, n, n)
            yb = y[idx]
            if np.sum(yb) == 0 or np.sum(yb) == n:
                continue
                
            cw_b = data["cw"][idx]
            gcw_b = data["gcw"][idx]
            
            boot_cw.append(_fast_auc(yb, cw_b))
            boot_gated_cw.append(_fast_auc(yb, gcw_b))
            
        if boot_cw:
            deltas.append(np.mean(boot_gated_cw) - np.mean(boot_cw))
            
    deltas = np.array(deltas)
    if deltas.size == 0:
        return 0.0, 0.0
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def run_one_class_sweep(csv_path: Path, mode: str = "uniform") -> dict:
    """Run category-averaged one-class sweep across noise levels under specified mode."""
    df = pd.read_csv(csv_path)
    categories = sorted(df["category"].unique())
    
    # Group category data to avoid re-filtering in loops
    cats_data = {}
    
    global_val = _pivot(df, "validation")
    
    for c in categories:
        df_cat = df[df["category"] == c]
        test_df = df_cat[df_cat["split"] == "test"]
        val_df = df_cat[df_cat["split"] == "validation"]
        
        if test_df.empty:
            continue
            
        test = _pivot(df_cat, "test")
        if not val_df.empty:
            val = _pivot(df_cat, "validation")
        else:
            val = global_val
            
        cats_data[c] = (val, test)
        
    rng = np.random.default_rng(0)
    rows = []
    
    for alpha in ALPHAS:
        cat_results = []
        for c, (val, test) in cats_data.items():
            # Apply degradation according to mode
            if mode == "uniform":
                noise = rng.uniform(0, 1, size=test["depth"].size)
                dep_deg = (1 - alpha) * test["depth"] + alpha * noise
            elif mode == "dropout":
                mask = rng.random(size=test["depth"].size) < alpha
                dep_deg = np.where(mask, 1.0, test["depth"])
            elif mode == "shift":
                dep_deg = np.clip(test["depth"] + alpha * 0.5, 0.0, 1.0)
            else:
                dep_deg = test["depth"]
            
            # Static Mean
            static = 0.5 * (test["rgb"] + dep_deg)
            # Confidence-Weighted Mean
            cw = (test["rc"] * test["rgb"] + test["dc"] * dep_deg) / (test["rc"] + test["dc"] + 1e-9)
            
            # RGA Gated-CW
            rr = _ksr(test["rgb"], val["rgb"])
            rd = _ksr(dep_deg, val["depth"])
            rgawt = (rr * test["rgb"] + rd * dep_deg) / (rr + rd + 1e-9)
            
            rr0 = _ksr(test["rgb"], val["rgb"])
            rd0 = _ksr(test["depth"], val["depth"])
            tau = min(rr0, rd0) - MARGIN
            fires = bool(min(rr, rd) < tau)
            gcw = rgawt if fires else cw
            
            y = test["y"]
            cat_results.append({
                "category": c,
                "rgb_auroc": float(roc_auc_score(y, test["rgb"])),
                "depth_auroc": float(roc_auc_score(y, dep_deg)),
                "static_auroc": float(roc_auc_score(y, static)),
                "cw_auroc": float(roc_auc_score(y, cw)),
                "rga_auroc": float(roc_auc_score(y, rgawt)),
                "gated_cw_auroc": float(roc_auc_score(y, gcw)),
            })
            
        # Arithmetic mean across categories
        mean_rgb = float(np.mean([res["rgb_auroc"] for res in cat_results]))
        mean_depth = float(np.mean([res["depth_auroc"] for res in cat_results]))
        mean_static = float(np.mean([res["static_auroc"] for res in cat_results]))
        mean_cw = float(np.mean([res["cw_auroc"] for res in cat_results]))
        mean_rga = float(np.mean([res["rga_auroc"] for res in cat_results]))
        mean_gcw = float(np.mean([res["gated_cw_auroc"] for res in cat_results]))
        
        # Paired bootstrap CI on the mean difference (Gated-CW - CW)
        lo, hi = _boot_delta_one_class(cats_data, alpha, mode, seed=0)
        delta = mean_gcw - mean_cw
        significant = bool(lo > 0 or hi < 0)
        
        rows.append({
            "alpha": alpha,
            "mean_rgb_auroc": mean_rgb,
            "mean_depth_auroc": mean_depth,
            "mean_static_auroc": mean_static,
            "mean_cw_auroc": mean_cw,
            "mean_rga_auroc": mean_rga,
            "mean_gated_cw_auroc": mean_gcw,
            "delta_gatedcw_minus_cw": delta,
            "ci95": [lo, hi],
            "significant": significant,
            "category_level_runs": cat_results,
        })
        
    return {"rows": rows}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run the canonical category-averaged one-class degradation sweep.")
    parser.add_argument("--csv-path", type=Path, default=ROOT / "experiments/fusion/mvtec3d_patchcore_v3_inputs.csv")
    parser.add_argument("--out-json", type=Path, default=ROOT / "experiments/fusion/mvtec3d_one_class_degradation_results.json")
    args = parser.parse_args()
    
    csv_path = args.csv_path
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return 1
        
    modes = ["uniform", "dropout", "shift"]
    all_results = {}
    
    for mode in modes:
        print(f"\nRunning canonical one-class sweep on {csv_path.name} (mode={mode})...")
        res = run_one_class_sweep(csv_path, mode)
        all_results[mode] = res
        
        # Print the summary table
        print(f"\nCanonical One-Class Summary (mode={mode}):")
        print("-" * 88)
        print(f"{'alpha':<6} | {'RGB':<6} | {'Depth':<6} | {'Static':<6} | {'CW':<6} | {'Gated-CW':<8} | {'Delta':<8} | {'CI-95%':<18} | {'Sig'}")
        print("-" * 88)
        for r in res["rows"]:
            star = "*" if r["significant"] else " "
            ci_str = f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
            print(f"{r['alpha']:<6.2f} | {r['mean_rgb_auroc']:<6.4f} | {r['mean_depth_auroc']:<6.4f} | {r['mean_static_auroc']:<6.4f} | {r['mean_cw_auroc']:<6.4f} | {r['mean_gated_cw_auroc']:<8.4f} | {r['delta_gatedcw_minus_cw']:+8.4f}{star} | {ci_str:<18} | {r['significant']}")
        print("-" * 88)
        
    out_json = args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Saved results -> {out_json}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
