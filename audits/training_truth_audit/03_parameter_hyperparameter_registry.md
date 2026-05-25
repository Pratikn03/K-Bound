# ELARA / RGA / RGA+ Training Truth Audit
## Part 4: Parameter and Hyperparameter Registry
**Audit Date:** 2026-05-23  
**Evidence sources:** All config YAML files, `train_attention_fusion.py`, `reliability_estimator.py`, `reliability_boosted_fusion.py`, `meta_router.py`, `baselines.py`

---

## 4.1 AttentionFusionModel Hyperparameters (by benchmark)

### Architecture Parameters

| Parameter | MVTec3D img_stats | MVTec3D PC / SP / HO | UNSW-NB15 | VisA | Real3D-AD | Default config |
|---|---|---|---|---|---|---|
| `embed_dim` | 32 | 48 | 32 | 48 | 48 | 64 |
| `num_heads` | 4 | 4 | 4 | 4 | 4 | 8 |
| `num_layers` | 1 | 1 | 1 | 1 | 1 | 1 |
| `dropout` | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| `num_domains` | 2 | 2 | 3 | 2 | 2 | 3 |
| `input_dim` | 10 (8 emb + score + conf) | 18 (16 emb + score + conf) | 10 | 18 | 18 | varies |
| `use_attention` | true | true | true | true | true | true |
| `use_confidence` | true | true | true | true | true | true |
| `use_input_confidence` | true | true | true | true | true | true |
| `use_domain_embeddings` | true | true | true | true | true | true |
| `use_positional_embeddings` | true | true | true | true | true | true |
| `use_missing_embedding` | true | true | true | true | true | true |

### Training Parameters

| Parameter | MVTec3D img_stats | MVTec3D PC / SP / HO | UNSW-NB15 | VisA | Real3D-AD | Default config |
|---|---|---|---|---|---|---|
| `seed` | 42 | 42 | 42 | 42 | 42 | 42 |
| `batch_size` | 64 | 64 | **256** | 64 | 64 | 128 |
| `epochs` | 20 | 20 | **15** | 20 | 20 | 20 |
| `lr` | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| `weight_decay` | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| `domain_dropout` | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| `early_stopping` | 5 | 5 | **4** | 5 | 5 | 5 |
| `lambda_reg` | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| `split_column` | split | split | fusion_split | split | split | (random) |
| `test_size` | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| `val_size` | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |

**Notes:**
- UNSW-NB15 uses `batch_size=256` (vs 64 elsewhere) due to large sample count (60k samples)
- UNSW-NB15 uses `early_stopping=4` (vs 5) and `epochs=15` (vs 20)
- No hyperparameter search was performed across benchmarks; all use Adam lr=0.001

### Early Stopping Criterion
- Metric: **validation PR-AUC** (confirmed in `train_attention_fusion.py:line 219-239`)
- Patience: 5 epochs (4 for UNSW)
- Best checkpoint restored: yes (model saved on best val PR-AUC improvement)

---

## 4.2 Reliability Estimator Hyperparameters

| Parameter | All benchmarks (except UNSW) | UNSW |
|---|---|---|
| `ece_weight` (α) | 0.45 | 0.45 |
| `ks_weight` (β) | 0.35 | 0.35 |
| `sharpness_weight` (γ) | 0.20 | 0.20 |
| `clean_gate_threshold` (τ) | **0.66** | **0.66** |
| `n_calibration_bins` | 10 | 10 |
| `min_samples_for_ks` | 20 | **30** |
| `gate_mode` | mean | mean |
| `min_gate_threshold` | 0.34 (=1−τ) | 0.34 |
| `category_aware` | true | true |
| `unknown_category_policy` | global | global |

**Source:** All benchmark configs under `reliability:` section + `attention_config.yaml`  
**Formula confirmed:** `r_d = 0.45×(1−ECE_d) + 0.35×KS_p + 0.20×sharpness` — matches paper Eq. 1

---

## 4.3 RGA+ Hyperparameters

### ReliabilityBoostedFusion Candidate Grid (confirmed from code)

| Candidate | Algorithm | lr | max_depth | min_samples_leaf | Notes |
|---|---|---|---|---|---|
| hgb_lr0.1_depthNone_leaf20 | HGB | 0.10 | None | 20 | Unlimited depth |
| hgb_lr0.1_depth4_leaf30 | HGB | 0.10 | 4 | 30 | |
| hgb_lr0.1_depth3_leaf20 | HGB | 0.10 | 3 | 20 | |
| hgb_lr0.1_depth3_leaf10 | HGB | 0.10 | 3 | 10 | |
| hgb_lr0.1_depth2_leaf20 | HGB | 0.10 | 2 | 20 | |
| hgb_lr0.05_depth3_leaf20 | HGB | 0.05 | 3 | 20 | |
| hgb_lr0.05_depth3_leaf10 | HGB | 0.05 | 3 | 10 | |
| logistic_c0.1 | LogReg + StandardScaler | — | — | — | C=0.1 |
| logistic_c1.0 | LogReg + StandardScaler | — | — | — | C=1.0 |
| logistic_c10.0 | LogReg + StandardScaler | — | — | — | C=10.0 |
| logistic_c100.0 | LogReg + StandardScaler | — | — | — | C=100.0 |

**All HGB models:** `max_iter=300`, `l2_regularization=0.01`, `random_state=seed`  
**All LogReg models:** `class_weight="balanced"`, `max_iter=1000`, `random_state=seed`

**Selection metric:** `roc_pr_f1` (composite of ROC-AUC + PR-AUC + F1) for supervised-paired benchmarks  
**Confirmed from results:** MVTec3D PC SP seed=42 selected `hgb_lr0.1_depthNone_leaf20` (val ROC-AUC=0.7266)

---

## 4.4 RGAMetaRouter Hyperparameters

| Parameter | Value | Evidence |
|---|---|---|
| Val split ratio (train/select) | 60/40 via StratifiedShuffleSplit | `meta_router.py:lines 192-194` |
| Logistic stacking C | 1.0, balanced weights | `meta_router.py` |
| Top-k mean ensembles | k=2, k=3 | `meta_router.py:lines 213-230` |
| Tie-break rule | alphabetical (boost preferred over router per paper) | `meta_router.py:line 232` |

---

## 4.5 Baseline Hyperparameters

| Baseline | Key Parameters | Evidence |
|---|---|---|
| RandomForestFusion | n_estimators=200, class_weight=balanced, min_samples_leaf=5 | `baselines.py` |
| EarlyFusionMLP | hidden=(128,64), dropout=0.3, epochs=50, patience=8 | `baselines.py` |
| LateFusionEnsemble | per-domain LogReg → meta LogReg (class_weight=balanced) | `baselines.py` |
| ConfidenceWeightedMean | weight = 2×|score−0.5|, no learnable params | `baselines.py` |
| TentScoreAdapter | n_steps=25, lr=0.03, entropy minimization | `baselines.py` |
| TTTPseudoLabelAdapter | conf_threshold=0.85, pseudo-label TTA | `baselines.py` |
| EATAScoreAdapter | entropy_margin=0.4, cosine_threshold=0.05 | `baselines.py` |
| SARScoreAdapter | entropy_margin=0.5, stability guard | `baselines.py` |

---

## 4.6 Evaluation Harness Parameters

| Parameter | 5-seed configs | 30-seed configs | Evidence |
|---|---|---|---|
| Seeds | [42,43,44,45,46] | [42..71] | All configs |
| n_bootstrap | 200 | 200 | All configs |
| bootstrap_alpha | 0.05 | 0.05 | All configs |
| decision_threshold | val_f1 | val_f1 | All configs |
| domain_dropout_probs | [0.0,0.1,0.3,0.5] | [0.0,0.1,0.3,0.5] | All configs |
| cda_samples | 80 | 80 | All configs |
| performance_iters | 10 | 10 | All configs |

**30-seed benchmarks:** MVTec3D PC SP, MVTec3D PC HO, VisA SP (seeds 42–71)  
**5-seed benchmarks:** MVTec3D canonical, UNSW, Real3D-AD, VisA canonical

---

## 4.7 CRAF / Mechanism Isolation Parameters

| Parameter | Value | Evidence |
|---|---|---|
| drift_noise_levels | [0.0, 0.05, 0.1, 0.2, 0.3] | All configs |
| adversarial_attacks | [zero_attack, max_attack, gaussian_noise] | All configs |
| adversarial_sigma | 0.1 | All configs |
| tau_sweep_thresholds | [0.4, 0.5, 0.6, 0.66, 0.7, 0.8, 0.9] | All configs |
| component_ablation_variants | [full, no_ece, no_ks, no_sharpness, no_gate] | All configs |
| learned_gate | true | All configs |

---

## 4.8 Hyperparameter Selection Procedure

**AttentionFusionModel:** No grid search. Parameters set once per benchmark class (image/cyber/tabular) based on domain size. No test-set oracle for architecture selection.

**ReliabilityEstimator weights:** Fixed at (0.45, 0.35, 0.20) across all benchmarks. No search performed.

**Gate threshold τ=0.66:** Fixed across all benchmarks. Paper describes this as a heuristic conservative threshold. No grid search on test set.

**RGA+ selection:** Grid search of 11 candidates evaluated strictly on validation fold. Test labels never used.

**PRIMARY CONCERN:** The reliability weight triplet (0.45, 0.35, 0.20) and the gate threshold τ=0.66 appear to be **fixed constants**, not searched. No evidence of a validation-set search for these values was found. This is methodologically sound (avoids tuning on test data), but means the hyperparameters are not empirically optimized — they are engineering choices. This should be stated clearly in the paper.
