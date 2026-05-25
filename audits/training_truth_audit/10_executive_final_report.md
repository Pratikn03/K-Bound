# ELARA / RGA / RGA+ Training Truth Audit
## Part 11: Executive Final Report
**Audit Date:** 2026-05-23  
**Auditor:** Automated read-only sweep — evidence-backed  
**Repository:** `/Volumes/T9/uav/AutoML_Flagship_V8/`  
**System:** ELARA (Evidence-Layered Anomaly Reliability Architecture)  
**Method:** RGA (Reliability-Gated Attention), RGA+ (supervised router/boost head)

---

## Executive Summary

This audit examined the ELARA/RGA/RGA+ research repository for publication-grade training transparency. The audit covered: model inventory (6 model types), data lineage (11 benchmark configurations across 5 dataset families), hyperparameter registry, training pipeline reconstruction, model selection discipline, overfitting and leakage risks, and paper-to-code consistency (14 claim corrections).

**The system is methodologically sound after Phase 1 corrections.** No undocumented data leakage was found. All model selection decisions use validation data only. The paper explicitly acknowledges its own historical errors through a documented claim matrix. Three blocking issues must be addressed before submission.

---

## What Is Actually Being Trained

### The fusion model (AttentionFusionModel)
A single-layer, 1–3 domain cross-modal attention transformer trained per benchmark × seed. Loss = BCE + attention entropy regularization + confidence regularization (λ=0.01). Trained by AdamW (lr=0.001, weight_decay=0.01) with early stopping on validation PR-AUC (patience=4–5 epochs, max 15–20 epochs).

### The reliability estimator (RGA mechanism)
Post-hoc isotonic calibrators fit on the validation split after training. Reliability = 0.45×(1−ECE) + 0.35×KS_p + 0.20×sharpness, with gate threshold τ=0.66. Not a trained neural network — a fixed-weight heuristic with learned calibration.

### The RGA+ boosted head
A grid of 11 sklearn candidates (7 HGB + 4 LogReg) trained on the training fold and selected by validation roc_pr_f1. The selected candidate is frozen before test evaluation.

### The RGA+ meta router
A validation-only stacking procedure: 60/40 split of validation fold used to train a logistic stacker and select the best prediction from all available methods.

---

## What Data Is Used

| Benchmark | Type | N samples | Protocol | Test discipline |
|---|---|---|---|---|
| MVTec 3D-AD PatchCore | Naturally paired visual | 3,226 | Canonical (one-class) | Official MVTec split |
| MVTec 3D-AD PC Supervised Paired | Naturally paired visual | 3,226 | Supervised fusion | Redistributed MVTec test rows |
| MVTec 3D-AD PC Held-Out | Naturally paired visual | 3,226 | Held-out category | 4 train / 4 test categories |
| MVTec LOCO-AD | Logical/structural visual | ~4,000 | Canonical + supervised | Official split |
| VisA RGB+edge | Paired visual | 10,821 | Canonical + supervised | Official 1cls split |
| Real3D-AD | Point cloud | 1,254 | Supervised paired | Official split |
| UNSW-NB15 | Naturally paired cyber | 60,000 | 3-domain, fresh stratified | Fresh split + held-out attack |
| ELARA-Bench-LA | Label-aligned 4-domain | 8,000 | Secondary benchmark | Out-of-fold | 

---

## Key Numerical Results (from artifacts)

### MVTec 3D-AD PatchCore Canonical (one-class) — 5 seeds
All methods near chance (0.49–0.60 ROC-AUC). This is the protocol-diagnostic finding: supervised fusion cannot generalize from normal-only training. The paper correctly reports this as a protocol limitation, not a metric failure.

### MVTec 3D-AD PatchCore Supervised Paired — 30 seeds  
RGA+ (router) test ROC-AUC = **0.739** vs. val-frozen comparator (SAR) = 0.735.  
Switching certificate: certified (LCB=+0.102). DeLong p=0.919 (n.s.) for the specific comparison in claim M001.  
**Interpretation:** Switching certificate shows RGA+ consistently beats static attention (Δ=+0.105); DeLong test on a specific single-seed comparator pair is not significant.

### UNSW-NB15 — 5 seeds, naturally paired
RGA meta router ROC-AUC = 0.989 vs. static = 0.979.  
Switching cert: certified (LCB=+0.010). DeLong p_Holm=6.7e-6 vs. val-frozen RF comparator (Δ=+0.0003).  
**Interpretation:** Statistically significant at n=55,491 events but practically negligible effect size.

### ELARA-Bench-LA (secondary, label-aligned)
KS-drift signal delivers: zero_attack Δ=+0.0506, max_attack Δ=+0.0319 robustness gain.  
Component ablation shows removing KS eliminates the gain (reclassified as Family B descriptive).

---

## Critical Findings

### ✅ What Works
1. **No test leakage** — scorer fitting, embedding normalization, model selection, threshold selection all use train/validation only.
2. **Historical leakage corrected** — 14 documented claim corrections, all pre-submission. The most important: test-oracle RGA+ selection replaced by validation-frozen selection (M001, M004).
3. **Correct one-class protocol** — the protocol erratum is acknowledged and corrected. Canonical cells are labeled as protocol diagnostics, not discriminability claims.
4. **Honest statistical reporting** — 3 of 5 Family A primary cells are not significant. Only UNSW achieves Holm-corrected significance (with trivial practical effect). The paper does not hide these non-findings.
5. **Mechanism analysis is correctly labeled descriptive** — KS ablation, τ sweep, k-of-D sweep are all Family B/C (descriptive), not Family A (inferential).

### ⚠️ Concerns

1. **No model checkpoints exist** — models cannot be inspected or reproduced without rerunning full training. Result JSON files are the only artifacts.
2. **ELARA-Bench-LA preparation not audited** — the secondary benchmark's preparation pipeline was not directly read. KS ablation numbers in the abstract cite this benchmark.
3. **Val fold triple use** — same validation fold used for early stopping, reliability fitting, and RGA+ candidate selection. Standard practice but should be stated.
4. **Reliability weights and τ are engineering heuristics** — not validated empirically. Paper should state this explicitly.
5. **UNSW-NB15 fresh split** — the 0.989 ROC-AUC reflects a fresh stratified split that keeps attack types in train, not the original UNSW generalization setup. The held-out attack protocol (0.994 ROC-AUC) gives a more honest picture.
6. **Real3D-AD n=48 normal references** — very small for PatchCore memory bank; confidence intervals are wide.

---

## Blocking Issues for Submission

| # | Issue | Action Required |
|---|---|---|
| **B1** | No saved model checkpoints (.pt) | Document that results require retraining OR commit checkpoints to supplementary |
| **B2** | ELARA-Bench-LA preparation not audited | Run audit on ELARA-Bench-LA preparation scripts; verify KS ablation numbers |
| **B3** | KS ablation numbers (+0.0506, +0.0319) not verified from code | Read the ELARA-Bench-LA results JSON directly and confirm values |

---

## Final Verdict

| Dimension | Verdict |
|---|---|
| Training transparency | ✅ ADEQUATE — pipeline fully reconstructible from code and configs |
| Data discipline | ✅ ADEQUATE — train-only scoring; no test leakage found |
| Model selection | ✅ ADEQUATE — validation-only; corrections documented |
| Statistical rigor | ✅ ADEQUATE — Holm correction; DeLong; non-findings reported honestly |
| Reproducibility | ⚠️ PARTIAL — scripts present; checkpoints absent |
| Claim accuracy | ✅ ADEQUATE after Phase 1 corrections — 14 documented corrections applied |
| Overall | **CONDITIONALLY READY — address B1/B2/B3 before submission** |
