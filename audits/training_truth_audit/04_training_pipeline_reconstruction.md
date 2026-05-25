# ELARA / RGA / RGA+ Training Truth Audit
## Part 5: Training Pipeline Reconstruction
**Audit Date:** 2026-05-23  
**Evidence sources:** `train_attention_fusion.py`, `run_attention_harness.py` (referenced), `scripts/run_train_vision.sh`, all configs

---

## 5.1 Full Pipeline Reconstruction

The complete training pipeline executes in this order:

```
Step 1: Data preparation
  └── prepare_*_fusion_benchmark.py
      ├── Reads raw dataset
      ├── Fits scorer on train split ONLY
      ├── Emits long-format CSV (*_inputs.csv)
      └── Writes metadata JSON

Step 2: Per-seed training loop (in evaluation harness)
  For each seed in [42..71] (or [42..46] for 5-seed configs):
    Step 2a: train_attention_fusion()
      ├── Set seed (numpy + torch)
      ├── Load *_inputs.csv
      ├── Build fusion tensors (features [N,D,F], masks [N,D], labels [N])
      ├── Split: train/val/test (from pre-assigned split column OR random stratified)
      ├── Construct FusionDataset + DataLoader
      ├── Initialize AttentionFusionModel
      ├── Training loop (AdamW, BCE + entropy + conf_reg)
      │   ├── Apply domain dropout per batch
      │   ├── Evaluate on val after each epoch
      │   ├── Early stop on val PR-AUC (patience=5)
      │   └── Save checkpoint on best val PR-AUC
      └── Evaluate on test; write metrics JSON

    Step 2b: ReliabilityEstimator fitting
      ├── Load best checkpoint
      ├── Fit on VALIDATION split (calibrators + reference distributions)
      └── Compute per-domain ECE, KS references, sharpness

    Step 2c: Baseline fitting
      ├── EarlyFusionMLP: train fold only
      ├── RandomForestFusion: train fold only
      ├── LateFusionEnsemble: train fold only
      ├── TentScoreAdapter / TTT / EATA / SAR: test-time adaptation (no train fit)
      └── ConfidenceWeightedMean: parameter-free

    Step 2d: RGA+ fitting (supervised-paired benchmarks)
      ├── ReliabilityBoostedFusion.fit(train_fold, val_fold)
      │   ├── Build feature matrix (flattened features + masks + stats + reliability)
      │   ├── Train all 11 candidates on train fold
      │   ├── Select best by val metric (roc_pr_f1)
      │   └── Store selected candidate name + val scores
      └── RGAMetaRouter
          ├── Split val fold 60/40 (StratifiedShuffleSplit)
          ├── Train logistic stacker on 60% of val
          ├── Evaluate all candidates on 40% of val
          ├── Select best candidate by val AUC
          └── Refit logistic stacker on full val if selected

    Step 2e: Evaluation on test fold
      ├── Static attention: direct model output
      ├── CRAF/RGA: gate decisions from reliability estimator
      ├── RGA+: predict_proba from selected candidate
      ├── All baselines: predict_proba
      └── Bootstrap CI (n=200, alpha=0.05) for all metrics

Step 3: Aggregate across seeds
  ├── Mean ± std for each metric
  └── DeLong p-values for paired ROC-AUC comparisons
```

---

## 5.2 Critical Implementation Details

### 5.2.1 How the split column is used

The configs specify `split_column: split` (or `fusion_split` for UNSW). When present, the harness reads the pre-assigned split from the CSV rather than performing a fresh random stratified split. This means:
- **The split used during multi-seed evaluation is the SAME fixed split** for all seeds
- Seed variation affects model weight initialization only — not the train/val/test assignment
- This is confirmed by `train_attention_fusion.py` code: `split_column` triggers a lookup rather than `train_test_split`

**CRITICAL IMPLICATION:** For 30-seed configurations (SP benchmarks), 30 models are trained on the identical train fold with different random initializations. This estimates initialization variance, not data-split variance.

### 5.2.2 Domain dropout during training

Domain dropout is applied per batch during training:
```python
masks_batch = apply_domain_dropout(masks_batch, domain_dropout=0.10)
```
This randomly masks 10% of domain observations per batch, teaching the model to handle missing domains. This is **not applied at test time** for primary metrics (domain_dropout_prob=0.0 for clean test).

### 5.2.3 Decision threshold selection

`decision_threshold: val_f1` — the threshold used for binary classification metrics (F1, precision, recall) is chosen to maximize F1 on the **validation fold**. This is a legitimate practice, but means:
- F1, precision, recall are threshold-dependent metrics selected on validation
- ROC-AUC and PR-AUC are threshold-independent and are the primary metrics

### 5.2.4 Loss function

```python
total_loss = bce_loss - lambda_reg * entropy + lambda_reg * conf_reg
```
- `bce_loss`: standard binary cross-entropy
- `entropy` term: encourages uniform attention distribution (regularizer)
- `conf_reg`: penalizes confidences far from 1.0 (encourages sharp confidence estimates)
- `lambda_reg = 0.01` (same for both regularizers)

### 5.2.5 Checkpoint behavior

Training saves checkpoints to `models/fusion/attention_*/` on each PR-AUC improvement. However, **no `.pt` checkpoint files were found** in the workspace. This means either:
1. The evaluation harness loads the checkpoint within each seed run and then the file is overwritten for the next seed, OR
2. Checkpoints were cleaned up after the results JSON was written, OR
3. The harness runs training in-memory without persistent checkpoints between the training and evaluation phases

The result artifacts (`*_results.json`) are the surviving evidence of training outcomes.

---

## 5.3 Shell Runners

### `scripts/run_train_vision.sh`
- Triggers the full training pipeline for vision benchmarks
- Calls `prepare_mvtec3d_fusion_benchmark.py` with `--feature-mode patchcore`
- Then calls the harness for each config variant

### `scripts/rebuild_paper.sh`
- Regenerates all paper tables/figures from existing result JSON artifacts
- Does **not** retrain models — reads pre-computed results
- Calls LaTeX compilation at the end

---

## 5.4 Pipeline Reproducibility Assessment

| Aspect | Status | Evidence |
|---|---|---|
| Seed is set before every run | ✅ CONFIRMED | `set_seed()` in `train_attention_fusion.py:lines 35-38` |
| Split column preserved across seeds | ✅ CONFIRMED | Config `split_column` key used |
| Scorer fit uses train-only data | ✅ CONFIRMED | All `prepare_*.py` scripts |
| Test labels not used for model/threshold selection | ✅ CONFIRMED | `val_f1` threshold, val-only router/boost selection |
| Checkpoint saved on val metric | ✅ CONFIRMED | `train_attention_fusion.py:line 239` |
| Harness reproducibility across machines | ⚠️ PARTIAL | Floating-point non-determinism with GPU; seed set but `torch.backends.cudnn.deterministic` not confirmed |
| MLflow logging | ⚠️ Optional | `mlflow.enabled: false` in all benchmark configs |
