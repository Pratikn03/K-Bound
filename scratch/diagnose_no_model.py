import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path("/Volumes/T9/uav/AutoML_Flagship_V8")

def diagnose_no_model():
    print("\n=================== DIAGNOSIS WITH DIRECT ONE-CLASS SCORE AVERAGE (NO MODEL) ===================")
    
    # Load test labels
    from scripts.family_d_v2_inference import _read_eyecandies_test_labels
    labels_dict = _read_eyecandies_test_labels()
    
    # Load Eyecandies CSV
    df = pd.read_csv(ROOT / "experiments" / "fusion" / "eyecandies_inputs.csv")
    sub = df[df["fusion_split"] == "test"].copy()
    sub = sub.sort_values(["sample_id", "domain"]).reset_index(drop=True)
    
    samples = sorted(sub["sample_id"].unique())
    samples = [sid for sid in samples if sid in labels_dict]
    labels = np.array([labels_dict[sid] for sid in samples], dtype=int)
    
    rgb_scores = []
    depth_scores = []
    for sid in samples:
        rows = sub[sub["sample_id"] == sid]
        rgb_r = rows[rows["domain"] == "rgb"]
        dep_r = rows[rows["domain"] == "depth"]
        rgb_scores.append(float(rgb_r["score"].iloc[0]) if len(rgb_r) > 0 else 0.5)
        depth_scores.append(float(dep_r["score"].iloc[0]) if len(dep_r) > 0 else 0.5)
        
    rgb_scores = np.array(rgb_scores)
    depth_scores = np.array(depth_scores)
    
    print(f"Total labeled samples in test set: {len(samples)}")
    print(f"  Anomalous: {sum(labels)}")
    print(f"  Normal:    {len(labels) - sum(labels)}")
    print()
    print(f"Individual modality AUCs:")
    print(f"  RGB alone:   {roc_auc_score(labels, rgb_scores):.4f}")
    print(f"  Depth alone: {roc_auc_score(labels, depth_scores):.4f}")
    print(f"  Static average of both clean modalities: {roc_auc_score(labels, (rgb_scores + depth_scores)/2.0):.4f}")
    print()
    
    for cell_id, target in [("D-EYE-1", "depth"), ("D-EYE-2", "rgb")]:
        print(f"--- Degradation scenario: {cell_id} ({target} collapse) ---")
        
        # Apply degradation (setting target domain score to 0.0)
        deg_rgb = rgb_scores.copy()
        deg_depth = depth_scores.copy()
        if target == "depth":
            deg_depth[:] = 0.0
        else:
            deg_rgb[:] = 0.0
            
        # 1. Static prediction: equal-weighted average under degradation
        static_pred = (deg_rgb + deg_depth) / 2.0
        auc_static = roc_auc_score(labels, static_pred)
        
        # 2. RGA prediction:
        # Under degradation, target modality reliability goes to 0.0, so RGA routes to reliability-weighted average
        # where the clean modality gets weight 1.0 and degraded gets weight 0.0.
        # So rga_pred is exactly the clean modality's score.
        if target == "depth":
            rga_pred = rgb_scores # clean modality is RGB
        else:
            rga_pred = depth_scores # clean modality is Depth
            
        auc_rga = roc_auc_score(labels, rga_pred)
        delta = auc_rga - auc_static
        
        print(f"  Static AUC (degraded): {auc_static:.4f}")
        print(f"  RGA AUC (degraded):    {auc_rga:.4f}")
        print(f"  Delta AUC (RGA-Static): {delta:+.4f}")
        print()

if __name__ == "__main__":
    diagnose_no_model()
