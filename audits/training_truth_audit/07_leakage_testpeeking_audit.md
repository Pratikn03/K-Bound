# ELARA / RGA / RGA+ Training Truth Audit
## Part 8: Leakage and Test-Peeking Audit
**Audit Date:** 2026-05-23  
**Evidence sources:** `train_attention_fusion.py`, all `prepare_*.py` scripts, `reliability_boosted_fusion.py`, `meta_router.py`, `leakage_guard.py` (referenced), `claim_matrix.csv`

---

## 8.1 Test-Peeking Risk Matrix

| Risk Type | Code Path | Test Labels Seen? | Verdict |
|---|---|---|---|
| AttentionFusionModel training | `train_attention_fusion.py` | ❌ NO — training only uses train/val splits | ✅ CLEAN |
| Early stopping criterion | val PR-AUC | ❌ NO | ✅ CLEAN |
| ReliabilityEstimator fitting | `reliability_estimator.py:fit()` | ❌ NO — fitted on val split | ✅ CLEAN |
| RGA+ candidate selection | `reliability_boosted_fusion.py:fit()` | ❌ NO — val labels only | ✅ CLEAN |
| RGA+ meta router selection | `meta_router.py:fit_rga_meta_router()` | ❌ NO — val labels only | ✅ CLEAN |
| Decision threshold (val_f1) | harness | ❌ NO — val labels only | ✅ CLEAN |
| Switching certificate | `switching_certificate_t5_audit.json` | Test AUC used post-hoc (descriptive only) | ✅ CLEAN (post-hoc analysis) |
| τ sweep, component ablation | configs | Test AUC computed but NOT used for τ selection | ✅ CLEAN (descriptive) |
| Score normalization | `_minmax`, `_normal_reference_scores` | ❌ NO — train split only | ✅ CLEAN |
| Embedding PCA | `fit_pca_projection` | ❌ NO — train mask only | ✅ CLEAN |
| KS reference distribution | `reliability_estimator.py` | ❌ NO — val split only | ✅ CLEAN |
| Paper claim corrections (M001–M014) | `claim_matrix.csv` | Previous claims used test-optimal selection | ⚠️ CORRECTED |

---

## 8.2 Historical Leakage (Corrected Pre-Submission)

### Issue 1: Test-Oracle RGA+ Selection (M001, M004 — CORRECTED)
**Original behavior:** The headline RGA+ metric used `max(router_test_auc, boost_test_auc)` — i.e., test performance was used to pick between router and boost.  
**Corrected behavior:** Selection is made by validation ROC-AUC only. Router chosen if val_router_auc > val_boost_auc, otherwise boost. Tie-break: boost.  
**Impact:** MVTec3D PC SP: 0.738 → 0.739 (minimal impact in this case). MVTec LOCO: sign flipped (−0.008 vs +0.008). Real3D: 0.566 → 0.534.  
**Evidence:** `claim_matrix.csv` rows M001, M004, M005, M007

### Issue 2: Non-Audited Comparator Selection (M001, M004 — CORRECTED)
**Original behavior:** "Best non-router" baseline was computed as max over baselines on test split.  
**Corrected behavior:** Primary comparator is the baseline with the highest **seed-mean validation ROC-AUC**, frozen before test evaluation.  
**Impact:** Changes the comparator from Tent to SAR in MVTec3D PC SP.  
**Evidence:** `claim_matrix.csv` M001, M004

### Issue 3: Non-Pre-Registered Protocol (M003 — CORRECTED)
**Original behavior:** Holm correction applied across all 9 evaluated cells (K=9).  
**Corrected behavior:** K=5 (Family A primary cells only). Protocol-diagnostic and exploratory cells excluded from inferential count.  
**Impact:** MVTec LOCO SP and VisA SP lose Holm-corrected significance (were significant under K=9, now n.s. under K=5).  
**Evidence:** `claim_matrix.csv` M003

### Issue 4: MVTec Protocol Erratum (acknowledged in paper)
**Original behavior:** Early drafts used random train/test split for MVTec 3D-AD.  
**Corrected behavior:** Official MVTec one-class protocol (normal-only train).  
**Impact:** All supervised methods collapse to ≈chance AUC under correct protocol.  
**Evidence:** `PAPER_DRAFT_v1.tex` lines 706–719, "Protocol erratum" section

---

## 8.3 Data Contamination Guards in Code

### PipelineGuard / leakage_guard.py
- Referenced in `train_attention_fusion.py` via `hash_file(data_path)` in the metrics output
- The hash is stored in the metrics JSON to allow traceability of which CSV generated which result
- `leakage_guard.py` described as implementing hash-based train/test contamination detection
- **CANNOT FULLY VERIFY:** `leakage_guard.py` not directly read in this audit; hash output observed in metrics

### Split column discipline
- When `split_column` is specified in config, the harness reads pre-assigned splits from CSV
- This means the split assignment happens **once** in `prepare_*.py` and is frozen
- All seeds train on the same fold — preventing split-peeking across seeds

---

## 8.4 Score Normalization Leakage Check

**MVTec 3D-AD (PatchCore mode):**
```python
# In prepare_mvtec3d_fusion_benchmark.py:
normal_reference_mask = train_mask & (defect_types == "good")
rgb_scores = patchcore_knn_score(rgb_resnet, normal_reference_mask, k=patchcore_k)
```
- Memory bank built from train/good only ✅
- KNN distance computed against this fixed memory bank ✅
- Score minmax normalization fitted on train_mask only ✅
- **No test information used** ✅

**UNSW-NB15:**
```python
# In prepare_unsw_paired_fusion_benchmark.py:
train_mask = (combined["fusion_split"].to_numpy() == "train")
domain_scores[domain] = _domain_score(block, train_mask)
# _minmax_clip(block[:, j], train_mask)
```
- Centroid and std computed from train split only ✅
- Score normalization (p95) from train only ✅
- **No test information used** ✅

---

## 8.5 Potential Subtle Leakage: UNSW-NB15 Fresh Split

**Concern:** UNSW-NB15 canonical run uses a fresh stratified split (not the original UNSW train/test split). The original UNSW split has known attack-category overlap; the fresh split avoids this but means attack types in training are the same as in test (just different individual events).

**Assessment:** This is **not leakage** — it's a design choice. The scorer is fitted on the train portion of the fresh split. No test labels are used anywhere in the pipeline. However, the 0.989 ROC-AUC should be contextualized: the high AUC reflects that UNSW-NB15 is a linearly separable classification problem for these features, not that RGA+ has discovered something novel.

---

## 8.6 Polarity Calibration Guard (M014 — CORRECTED)

**Original behavior:** A "polarity calibration" step could flip domain score orientation based on val/test correlation.  
**Corrected behavior:** Phase 1.F lock: polarity flip is a **validation-only diagnostic** and does NOT alter primary metrics. Per-seed flip decisions are logged for auditability but the primary path uses raw model predictions.  
**Evidence:** `claim_matrix.csv` M014

---

## 8.7 Summary: Leakage Assessment

| Category | Status |
|---|---|
| Test labels used in model training | ❌ NOT PRESENT |
| Test labels used in threshold selection | ❌ NOT PRESENT (val_f1 used) |
| Test labels used in candidate selection | ❌ NOT PRESENT (val only) |
| Test performance used to pick headline metric | ⚠️ WAS PRESENT — CORRECTED via Phase 1.B (M001, M004) |
| Scorer fitted on test data | ❌ NOT PRESENT |
| Embedding normalization on test data | ❌ NOT PRESENT |
| UNSW attack-category overlap in fresh split | ⚠️ By design — acknowledged; held-out protocol exists |
| Real3D small normal reference (n=48) | ⚠️ Data quality constraint — acknowledged in metadata |
| Val fold triple use | ⚠️ Present but standard; low severity |

**Overall leakage verdict: CLEAN after Phase 1 corrections.** The original test-oracle RGA+ selection has been corrected and documented in the claim matrix. No undocumented data leakage paths were found.
