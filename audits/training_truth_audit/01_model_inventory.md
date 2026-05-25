# ELARA / RGA / RGA+ Training Truth Audit
## Part 2: Model Inventory
**Audit Date:** 2026-05-23  
**Evidence sources:** `cross_modal_attention.py`, `reliability_estimator.py`, `reliability_boosted_fusion.py`, `meta_router.py`, `learned_gate.py`, `baselines.py`, all configs

---

## 2.1 Primary Models

### MODEL-01: AttentionFusionModel (Static Attention / RGA base)
| Field | Value | Evidence |
|---|---|---|
| Class name | `AttentionFusionModel` | `cross_modal_attention.py` |
| Architecture | DomainEncoder (per domain) → CrossModalAttentionFusion (multi-head self-attention) → sigmoid output | `cross_modal_attention.py` |
| Training framework | PyTorch | `train_attention_fusion.py:line 8` |
| Loss function | BCE + entropy regularization (−λ×entropy) + confidence regularization (λ×mean((conf−1)²)) | `train_attention_fusion.py:lines 41-62` |
| Optimizer | AdamW | `train_attention_fusion.py:line 201` |
| Early stopping metric | PR-AUC on validation set | `train_attention_fusion.py:lines 207-258` |
| Checkpoint saving criterion | Best val PR-AUC | `train_attention_fusion.py:line 239` |
| Seed handling | `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all` | `train_attention_fusion.py:lines 35-38` |
| Missing domain handling | Key-padding mask; missing-domain embedding | `cross_modal_attention.py` |
| Saved checkpoints present | **NO** — no `.pt` files found in workspace | `find` command result |

**Hyperparameter variants by benchmark:**

| Benchmark | embed_dim | num_heads | num_layers | dropout | batch_size | epochs | lr | weight_decay | early_stop | lambda_reg |
|---|---|---|---|---|---|---|---|---|---|---|
| MVTec3D (img_stats) | 32 | 4 | 1 | 0.1 | 64 | 20 | 0.001 | 0.01 | 5 | 0.01 |
| MVTec3D PatchCore | 48 | 4 | 1 | 0.1 | 64 | 20 | 0.001 | 0.01 | 5 | 0.01 |
| MVTec3D PC supervised | 48 | 4 | 1 | 0.1 | 64 | 20 | 0.001 | 0.01 | 5 | 0.01 |
| MVTec3D PC held-out | 48 | 4 | 1 | 0.1 | 64 | 20 | 0.001 | 0.01 | 5 | 0.01 |
| UNSW-NB15 | 32 | 4 | 1 | 0.1 | 256 | 15 | 0.001 | 0.01 | 4 | 0.01 |
| VisA | 48 | 4 | 1 | 0.1 | 64 | 20 | 0.001 | 0.01 | 5 | 0.01 |
| Real3D-AD | 48 | 4 | 1 | 0.1 | 64 | 20 | 0.001 | 0.01 | 5 | 0.01 |
| Default config | 64 | 8 | 1 | 0.1 | 128 | 20 | 0.001 | 0.01 | 5 | 0.01 |

**Architecture toggles (all benchmarks use same settings):**
- `use_attention: true`, `use_confidence: true`, `use_input_confidence: true`
- `use_domain_embeddings: true`, `use_positional_embeddings: true`, `use_missing_embedding: true`

---

### MODEL-02: ReliabilityEstimator (RGA mechanism)
| Field | Value | Evidence |
|---|---|---|
| Class name | `ReliabilityEstimator` | `reliability_estimator.py:line 29` |
| Also aliased as | `RGAReliabilityEstimator` | `reliability_estimator.py:line 507` |
| Training method | Post-hoc fit on **validation split** after AttentionFusionModel is trained | `reliability_estimator.py:lines 74-115` |
| Components fitted | Isotonic calibrators (one per domain, sklearn), reference score distributions | `reliability_estimator.py:lines 104-112` |
| Reliability formula | r_d = 0.45×(1-ECE_d) + 0.35×KS_p + 0.20×sharpness | `reliability_estimator.py:lines 164-168` |
| Gate threshold τ | 0.66 (all benchmarks) | All config files |
| Gate modes | mean (default), minimum, hybrid | `reliability_estimator.py:lines 176-212` |
| Subclass: CategoryAwareReliabilityEstimator | Fits per-category KS reference distributions | `reliability_estimator.py:lines 283-420` |
| Subclass: PerSampleReliabilityEstimator | Per-sample rank-based KS (streaming-grade variant) | `reliability_estimator.py:lines 423-498` |
| Serialization | joblib | `reliability_estimator.py:lines 239-280` |
| Saved estimators present | UNKNOWN — no `.joblib` found in `experiments/fusion/` in scan | UNKNOWN |

---

### MODEL-03: ReliabilityBoostedFusion (RGA+ supervised head)
| Field | Value | Evidence |
|---|---|---|
| Class name | `ReliabilityBoostedFusion` | `reliability_boosted_fusion.py:line 117` |
| Training split | Same train/val split as attention model harness | `reliability_boosted_fusion.py:lines 251-298` |
| Candidate grid | 7× HistGradientBoostingClassifier (varying lr/depth/leaf) + 4× LogisticRegression (varying C) = 11 candidates | `reliability_boosted_fusion.py:lines 144-186` |
| Model selection metric | validation roc_auc OR roc_pr_f1 (configurable, benchmark SP uses roc_pr_f1) | configs `rga_plus.selection_metric` |
| Feature construction | Flattened domain features + masks + score stats + confidence stats + reliability features (if estimator provided) | `reliability_boosted_fusion.py:lines 188-249` |
| Test labels used for selection | **NEVER** — strictly validation-only | `reliability_boosted_fusion.py:line 284-297` |
| Serialization | No checkpoint saved; re-fit per harness run | Inferred from code |

---

### MODEL-04: RGAMetaRouter (RGA+ router)
| Field | Value | Evidence |
|---|---|---|
| Class name | `RGAMetaRouter` | `meta_router.py:line 99` |
| Function | `fit_rga_meta_router` | `meta_router.py:line 161` |
| Candidate pool | base:each_method + logistic_stack + mean:top-k | `meta_router.py:lines 174-236` |
| Logistic stack training | On 60% of val fold (train_idx); selected on remaining 40% (select_idx) | `meta_router.py:lines 192-194` |
| Top-k mean ensembles | k=2, k=3; ranked by train-split AUC | `meta_router.py:lines 213-230` |
| Final selection | max validation score on select fold | `meta_router.py:line 232` |
| When logistic selected | Refitted on entire val fold before production use | `meta_router.py:lines 238-247` |
| Test oracle | **NEVER used** | confirmed |

---

### MODEL-05: Baselines (8 methods)
| Baseline | Class | Key parameters |
|---|---|---|
| EarlyFusionMLP | `EarlyFusionMLP` | hidden=(128,64), dropout=0.3, epochs=50, patience=8 |
| LateFusionEnsemble | `LateFusionEnsemble` | per-domain LogReg → meta LogReg |
| RandomForestFusion | `RandomForestFusion` | n_est=200, balanced weights, min_samples_leaf=5 |
| ConfidenceWeightedMean | `ConfidenceWeightedMean` | sharpness weight = 2×|score-0.5|, parameter-free |
| TentScoreAdapter | `TentScoreAdapter` | entropy minimization, 25 steps, lr=0.03 |
| TTTPseudoLabelAdapter | `TTTPseudoLabelAdapter` | pseudo-label TTA, conf_threshold=0.85 |
| EATAScoreAdapter | `EATAScoreAdapter` | EATA-style, entropy_margin=0.4, cosine_threshold=0.05 |
| SARScoreAdapter | `SARScoreAdapter` | SAR-style, entropy_margin=0.5, stability guard |

**Evidence:** `baselines.py:lines 59-777`

---

### MODEL-06: LearnedReliabilityGate (ablation only)
| Field | Value | Evidence |
|---|---|---|
| Class | `LearnedReliabilityGate` | `learned_gate.py` |
| Implementation | Logistic regression on (val_reliability, val_label) pairs | `learned_gate.py` |
| Purpose | Alternative to heuristic gate — trained to predict when reliability path beats static | `learned_gate.py` |
| Paper status | Ablation candidate; paper states it does **not** match heuristic gate's gain | `PAPER_DRAFT_v1.tex:line 53` |

---

## 2.2 Domain Expert Models (pre-fusion)
| Model | File | Type | Note |
|---|---|---|---|
| Fraud scorer | `src/models/fraud/supervised/fraud_model.pkl` | sklearn | ELARA-Bench-LA domain expert |
| Cyber scorer | `src/models/cyber/supervised/cyber_model.pkl` | sklearn | ELARA-Bench-LA domain expert |
| Behavior autoencoder | `src/models/behavior/behavior_autoencoder.pkl` | sklearn | ELARA-Bench-LA domain expert |
| Behavior LOF | `src/models/behavior/behavior_lof.pkl` | sklearn LOF | ELARA-Bench-LA domain expert |
| Fusion meta-model | `src/models/fusion/fusion_meta_model.pkl` | sklearn | Legacy pre-attention meta-model |

**Note:** These are fitted scikit-learn objects. No PyTorch checkpoint (.pt) models exist on disk. The attention fusion results in `experiments/fusion/` appear to have been generated by the harness script and saved as metric JSON without preserving the underlying model state.
