import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path("/Volumes/T9/uav/AutoML_Flagship_V8")
ARCHIVE_DIR = ROOT / "experiments" / "phase2" / "family_d" / "archives"

def diagnose_cell(cell_id):
    cell_dir = ARCHIVE_DIR / f"family_d_{cell_id.replace('-','_').lower()}"
    if not cell_dir.exists():
        print(f"Directory {cell_dir} does not exist.")
        return

    # Target D-EYE-v3 folder only
    parquet_files = sorted([f for f in cell_dir.rglob("*.parquet") if "D-EYE-v3__" in str(f) and not f.name.startswith(".")])
    if not parquet_files:
        print(f"No parquet files found in {cell_dir}")
        return

    print(f"\n=================== DIAGNOSIS FOR {cell_id} ===================")
    
    # Load first file to check columns
    df_sample = pd.read_parquet(parquet_files[0])
    print(f"Loaded sample dataframe from {parquet_files[0].name}")
    print(f"Columns: {df_sample.columns.tolist()}")
    print(f"Unique methods: {df_sample['method'].unique().tolist()}")
    print(f"Unique method variants: {df_sample['method_variant'].unique().tolist()}")
    
    # We will accumulate metrics over all seeds
    results = []
    
    # Eyecandies inputs for correlation
    inputs_df = pd.read_csv(ROOT / "experiments" / "fusion" / "eyecandies_inputs.csv")
    test_inputs = inputs_df[(inputs_df['fusion_split'] == 'test') & (inputs_df['label'] != -1)]
    test_pivoted = test_inputs.pivot(index='sample_id', columns='domain', values='score')
    
    print(f"Found {len(parquet_files)} parquet files.")
    for pf in parquet_files[:5]:
        print(f"  {pf.relative_to(ROOT)}")
        
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        # Groups by seed
        for (method, seed, variant), grp in df.groupby(["method", "seed", "method_variant"]):
            # print(f"File {pf.name}: method={method}, seed={seed}, variant={variant}")
            # We want the degraded variant (e.g. D-EYE-1 or D-EYE-2)
            if variant == "clean":
                continue
            
            # Extract scores and labels
            labels = grp["label"].values
            scores = grp["raw_score"].values
            sample_ids = grp["sample_id"].values
            
            # Skip if labels are all placeholder or invalid
            if len(np.unique(labels)) < 2:
                continue
                
            results.append({
                "seed": seed,
                "method": method,
                "variant": variant,
                "scores": scores,
                "labels": labels,
                "sample_ids": sample_ids
            })
            
    print(f"Loaded {len(results)} degraded results from the files.")
    if results:
        print(f"Unique variants in results: {sorted(list({r['variant'] for r in results}))}")
        print(f"Unique methods in results: {sorted(list({r['method'] for r in results}))}")
            
    # Group results by seed
    seeds = sorted(list({r["seed"] for r in results}))
    seed_metrics = []
    for seed in seeds:
        seed_rga_list = [r for r in results if r["seed"] == seed and "rga" in r["method"]]
        seed_static_list = [r for r in results if r["seed"] == seed and "static" in r["method"]]
        
        if not seed_rga_list or not seed_static_list:
            continue
            
        seed_rga = seed_rga_list[0]
        seed_static = seed_static_list[0]
        
        # Calculate AUCs
        auc_rga = roc_auc_score(seed_rga["labels"], seed_rga["scores"])
        auc_static = roc_auc_score(seed_static["labels"], seed_static["scores"])
        delta = auc_rga - auc_static
        
        # Check prediction ranges and variations
        std_rga = np.std(seed_rga["scores"])
        std_static = np.std(seed_static["scores"])
        corr = np.corrcoef(seed_rga["scores"], seed_static["scores"])[0, 1]
        
        seed_metrics.append({
            "seed": seed,
            "auc_static": auc_static,
            "auc_rga": auc_rga,
            "delta": delta,
            "std_static": std_static,
            "std_rga": std_rga,
            "corr": corr
        })
        
    metrics_df = pd.DataFrame(seed_metrics)
    print("\nSummary statistics across seeds:")
    if not metrics_df.empty:
        print(metrics_df[["auc_static", "auc_rga", "delta", "std_static", "std_rga", "corr"]].mean().to_string())
    else:
        print("No seed metrics calculated.")
    print("\nSample predictions check (Seed 42):")
    seed42_rga_list = [r for r in results if r["seed"] == 42 and "rga" in r["method"]]
    seed42_static_list = [r for r in results if r["seed"] == 42 and "static" in r["method"]]
    if seed42_rga_list and seed42_static_list:
        seed42_rga = seed42_rga_list[0]
        seed42_static = seed42_static_list[0]
        print(f"RGA scores (first 5): {seed42_rga['scores'][:5]}")
        print(f"Static scores (first 5): {seed42_static['scores'][:5]}")
        print(f"Unique RGA scores count: {len(np.unique(seed42_rga['scores']))}")
        print(f"Unique Static scores count: {len(np.unique(seed42_static['scores']))}")

if __name__ == "__main__":
    diagnose_cell("D-EYE-1")
    diagnose_cell("D-EYE-2")
