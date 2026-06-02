#!/usr/bin/env python3
import yaml
from pathlib import Path

# Target configuration files and their target batch sizes
TARGETS = {
    "configs/attention_real_fusion.yaml": 256,
    "configs/attention_mvtec3d_patchcore_supervised_paired.yaml": 128,
    "configs/attention_visa_supervised_paired.yaml": 128,
    "configs/attention_mvtec_loco_patchcore_supervised_paired.yaml": 128,
    "configs/attention_unsw_paired.yaml": 512,
    "configs/attention_real3d_fusion.yaml": 128,
    "configs/attention_m2_external_3d_adam_sealed.yaml": 128,
    "configs/attention_m2_confirmatory_sealed.yaml": 128,
    "configs/attention_m2_external_mulsen_sealed.yaml": 128,
    "configs/attention_m3_healthcare_confirmatory.yaml": 512,
    "configs/attention_m3_bidmc_confirmatory.yaml": 256,
    "configs/attention_eyecandies_transfer_dev_v1.yaml": 128,
    "configs/attention_m2_external_3d_adam_transfer_v1.yaml": 128,
}

SEEDS = list(range(42, 72))  # 30 seeds: 42 to 71

def patch_configs():
    root = Path(__file__).resolve().parents[1]
    for rel_path, batch_size in TARGETS.items():
        yaml_path = root / rel_path
        if not yaml_path.is_file():
            print(f"Warning: {rel_path} not found")
            continue
            
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            
        if "training" in content:
            content["training"]["epochs"] = 50
            content["training"]["lr"] = 0.003
            content["training"]["batch_size"] = batch_size
            
        if "evaluation" in content:
            content["evaluation"]["seeds"] = SEEDS
            
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False)
            
        print(f"Patched {rel_path}: epochs=50, lr=0.003, batch_size={batch_size}, seeds=30")

if __name__ == "__main__":
    patch_configs()
