import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from pathlib import Path
import sys

ROOT = Path("/Volumes/T9/uav/AutoML_Flagship_V8")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.run_breakthrough_experiment import (
    _build_model,
    _load_data,
    _make_loaders,
    _make_reliability_estimator,
    _predict_craf_with_stats,
    _predict_static,
    _split,
    _train_model,
    set_seed,
)

# Load test labels
from scripts.family_d_v2_inference import _read_eyecandies_test_labels
labels_dict = _read_eyecandies_test_labels()

def augment_with_synthetic_anomalies_independent(features, masks, labels, train_idx, score_idx, random_seed=42):
    rng = np.random.default_rng(random_seed)
    
    n_train = len(train_idx)
    n_synth = int(0.6 * n_train)  # 60% synthetic anomalies
    
    perturb_sub_idx = rng.choice(n_train, size=n_synth, replace=False)
    perturb_global_idx = train_idx[perturb_sub_idx]
    
    augmented_features = features.copy()
    augmented_labels = labels.copy()
    
    for idx in perturb_global_idx:
        # Choose domains randomly to perturb: RGB only, Depth only, or both
        mode = rng.choice(["rgb", "depth", "both"])
        
        # score_idx is 0
        if mode == "rgb":
            augmented_features[idx, 0, score_idx] = rng.uniform(0.7, 1.0)
        elif mode == "depth":
            augmented_features[idx, 1, score_idx] = rng.uniform(0.7, 1.0)
        else:
            augmented_features[idx, 0, score_idx] = rng.uniform(0.7, 1.0)
            augmented_features[idx, 1, score_idx] = rng.uniform(0.7, 1.0)
            
        augmented_labels[idx] = 1
        
    return augmented_features, augmented_labels

def evaluate_synthetic_training():
    set_seed(42)
    device = torch.device("cpu")
    
    import yaml
    with open(ROOT / "configs/phase2/family_d_v3_scoring_pipeline.yaml") as f:
        pipeline_spec = yaml.safe_load(f)
    rga_cfg = pipeline_spec.get("eyecandies_rga_config", {})
    rga_cfg["data"]["path"] = str(ROOT / rga_cfg["data"]["path"])
    
    features, masks, labels, sample_ids, domain_order, _, conf_idx, score_idx, sample_splits, _ = _load_data(rga_cfg)
    
    test_idx_raw = np.flatnonzero(sample_splits == 'test')
    for idx in test_idx_raw:
        sid = sample_ids[idx]
        if sid in labels_dict:
            labels[idx] = labels_dict[sid]
            
    train_idx, val_idx, test_idx = _split(labels, rga_cfg["training"], split_values=sample_splits)
    
    valid_test = labels[test_idx] != -1
    test_idx = test_idx[valid_test]
    
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")
    
    # Augment training features with independent synthetic anomalies
    aug_features, aug_labels = augment_with_synthetic_anomalies_independent(features, masks, labels, train_idx, score_idx)
    
    # Train loaders
    train_loader = DataLoader(
        list(zip(aug_features[train_idx], masks[train_idx], aug_labels[train_idx])),
        batch_size=64,
        shuffle=True
    )
    
    # Build and train model
    model = _build_model(rga_cfg, features.shape[1], features.shape[2], conf_idx, device)
    
    # Train model using standard BCE loss on augmented labels
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    model.train()
    for epoch in range(15):
        epoch_loss = []
        for batch in train_loader:
            feats, msks, lbls = [x.to(device) for x in batch]
            optimizer.zero_grad()
            logits, _, _ = model(feats, key_padding_mask=msks)
            loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), lbls.float())
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        print(f"Epoch {epoch+1}: loss = {np.mean(epoch_loss):.4f}")
        
    model.eval()
    
    # Fit Reliability Estimator
    estimator = _make_reliability_estimator(rga_cfg.get("reliability", {}), list(domain_order), score_idx)
    estimator.fit(features[train_idx], masks[train_idx], labels[train_idx])
    estimator.re_fit_ks_reference(features[val_idx], masks[val_idx])
    
    # Test set evaluation under degradation
    test_feat = features[test_idx]
    test_mask = masks[test_idx]
    test_labels = labels[test_idx]
    
    for cell_id, target in [("D-EYE-1", "depth"), ("D-EYE-2", "rgb")]:
        print(f"\nEvaluating {cell_id} ({target} collapse)...")
        # Apply degradation operator
        deg_feat = test_feat.copy()
        deg_mask = test_mask.copy()
        d_idx = list(domain_order).index(target)
        deg_feat[:, d_idx, score_idx] = 0.0
        
        # Static prediction
        static_probs = _predict_static(model, deg_feat, deg_mask, device)
        
        # RGA prediction
        craf_probs, _ = _predict_craf_with_stats(
            model,
            estimator,
            deg_feat,
            deg_mask,
            device,
            clean_gate_threshold=0.52 if target == "depth" else 0.51,
            per_sample_gating=True
        )
        
        auc_static = roc_auc_score(test_labels, static_probs)
        auc_rga = roc_auc_score(test_labels, craf_probs)
        print(f"  Static AUC: {auc_static:.4f}")
        print(f"  RGA AUC:    {auc_rga:.4f}")
        print(f"  Delta AUC:  {auc_rga - auc_static:+.4f}")

if __name__ == "__main__":
    evaluate_synthetic_training()
