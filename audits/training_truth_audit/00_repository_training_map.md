# ELARA / RGA / RGA+ Training Truth Audit
## Part 1: Repository Training Map
**Audit Date:** 2026-05-23  
**Auditor Role:** Automated — read-only evidence-backed sweep  
**Source of truth:** Executable code, configs, saved artifacts  
**Operating rule:** Claims treated as hypotheses; marked CONFIRMED / DISCREPANCY / UNKNOWN based on code evidence

---

## 1.1 Repository Root
```
/Volumes/T9/uav/AutoML_Flagship_V8/
```

## 1.2 Top-Level Directory Inventory

| Directory / File | Purpose |
|---|---|
| `src/uais/fusion/attention/` | Core RGA / RGA+ implementation (Python) |
| `src/scripts/` | Benchmark preparation scripts |
| `scripts/` | Shell runners (train, rebuild paper) |
| `configs/` | YAML configs for each benchmark × protocol pair |
| `experiments/fusion/` | Result JSON/CSV artifacts and metadata |
| `models/fusion/` | Trained checkpoint storage (.pt files) |
| `docs/research/` | Paper draft (.tex), claim matrix, metrics manifest |
| `src/models/` | Domain-expert model artifacts (.pkl) |

## 1.3 Core Source Files

| File | Role | Lines |
|---|---|---|
| `train_attention_fusion.py` | Main training entry point for AttentionFusionModel | 304 |
| `cross_modal_attention.py` | AttentionFusionModel + CrossModalAttentionFusion architecture | ~350 |
| `reliability_estimator.py` | ReliabilityEstimator (RGA core): ECE + KS + sharpness | 515 |
| `reliability_boosted_fusion.py` | RGA+ supervised boost head (HGB + LogReg grid) | 309 |
| `meta_router.py` | RGA+ validation-only router (selects best prediction) | 260 |
| `learned_gate.py` | LearnedReliabilityGate (logistic regression gate alternative) | ~200 |
| `baselines.py` | 8 baselines: EarlyFusionMLP, LateFusionEnsemble, RF, ConfWeightedMean, Tent, TTT, EATA, SAR | 777 |
| `evaluate_attention_harness.py` | Multi-seed evaluation loop | ~700 |
| `leakage_guard.py` | PipelineGuard: hash-based train/test contamination detection | ~300 |
| `attention_utils.py` | FusionDataset, data loading, schema validation | ~400 |

## 1.4 Benchmark Families Identified in Code + Configs

| Benchmark | Config File(s) | Data Path | Protocols |
|---|---|---|---|
| MVTec 3D-AD (image_statistics) | `attention_mvtec3d_fusion.yaml` | `experiments/fusion/mvtec3d_fusion_inputs.csv` | Canonical (one-class) |
| MVTec 3D-AD (PatchCore) | `attention_mvtec3d_patchcore.yaml` | `experiments/fusion/mvtec3d_patchcore_inputs.csv` | Canonical (one-class) |
| MVTec 3D-AD (PatchCore SP) | `attention_mvtec3d_patchcore_supervised_paired.yaml` | `experiments/fusion/mvtec3d_patchcore_supervised_paired_inputs.csv` | Supervised paired |
| MVTec 3D-AD (PatchCore HO) | `attention_mvtec3d_patchcore_heldout.yaml` | `experiments/fusion/mvtec3d_patchcore_heldout_inputs.csv` | Held-out category |
| MVTec LOCO-AD | `attention_mvtec_loco_patchcore.yaml` | `experiments/fusion/mvtec_loco_patchcore_inputs.csv` | Canonical + supervised |
| VisA | `attention_visa_fusion.yaml` | `experiments/fusion/visa_fusion_inputs.csv` | Canonical (one-class) |
| VisA (supervised paired) | `attention_visa_supervised_paired.yaml` | `experiments/fusion/visa_supervised_paired_inputs.csv` | Supervised paired |
| Real3D-AD | `attention_real3d_fusion.yaml` | `experiments/fusion/real3d_fusion_inputs.csv` | Canonical |
| Real3D-AD (SP) | `attention_real3d_supervised_paired.yaml` | `experiments/fusion/real3d_supervised_paired_inputs.csv` | Supervised paired |
| UNSW-NB15 | `attention_unsw_paired.yaml` | `experiments/fusion/unsw_paired_inputs.csv` | Naturally paired, 3 domains |
| UNSW-NB15 (held-out attack) | `attention_unsw_heldout_attack.yaml` | `experiments/fusion/unsw_heldout_attack_inputs.csv` | Held-out attack category |
| ELARA-Bench-LA | (secondary benchmark) | 4-domain label-aligned (fraud/cyber/behavior/NLP) | Label-aligned (synthetically paired) |

**Note:** ELARA-Bench-LA is referenced in the paper as the secondary benchmark (8,000 composite samples, 4 domains) but its preparation config was not directly inspected during this audit sweep. Evidence sourced from paper and metrics_manifest.

## 1.5 Model Checkpoints Found

**PyTorch `.pt` checkpoints:** NONE found in `models/fusion/attention_*/`  
Evidence: `find /Volumes/T9/uav/AutoML_Flagship_V8 -name "*.pt"` returned no results.  
**IMPLICATION:** The attention fusion models have not been saved to disk in this workspace, or checkpoints were removed. Results in `experiments/fusion/*_results.json` are the primary evidence of what was run.

**Scikit-learn `.pkl` models found:**
- `src/models/fusion/fusion_meta_model.pkl` — legacy meta-model
- `src/models/fraud/supervised/fraud_model.pkl` — domain expert
- `src/models/cyber/supervised/cyber_model.pkl` — domain expert
- `src/models/behavior/behavior_autoencoder.pkl` — domain expert
- `src/models/behavior/behavior_lof.pkl` — domain expert

## 1.6 Key Result Artifacts Found

| File | Content |
|---|---|
| `experiments/fusion/mvtec3d_patchcore_metadata.json` | 3226 samples, 6452 rows, 8 categories |
| `experiments/fusion/mvtec3d_patchcore_supervised_paired_metadata.json` | SP split: seed=42, val_fraction=0.15, test_fraction=0.30 |
| `experiments/fusion/mvtec3d_patchcore_heldout_metadata.json` | Train cats: bagel,cable_gland,cookie,dowel; Test: foam,peach,rope,tire |
| `experiments/fusion/unsw_paired_metadata.json` | 60k samples, 180k rows, 10 categories, 3 domains |
| `experiments/fusion/visa_supervised_paired_metadata.json` | 10821 samples, train:8548, val:1625, test:648 |
| `experiments/fusion/real3d_supervised_paired_metadata.json` | 1254 samples, 2508 rows, 12 categories |
| `experiments/fusion/meta_router_pac_audit.json` | PAC slack values for 5 fold × n combinations |
| `experiments/fusion/switching_certificate_t5_audit.json` | Switching certificate for 10 benchmark × protocol pairs |

---
*Evidence basis: direct file reads, `find` commands, config file analysis*
