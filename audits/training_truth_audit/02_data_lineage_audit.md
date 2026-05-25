# ELARA / RGA / RGA+ Training Truth Audit
## Part 3: Data Lineage Audit
**Audit Date:** 2026-05-23  
**Evidence sources:** `prepare_*.py` scripts, `*_metadata.json` files, configs

---

## 3.1 Benchmark Data Flow Overview

```
Raw public datasets  →  prepare_*.py  →  *_inputs.csv  →  harness  →  *_results.json
```

Each preparation script is responsible for:
1. Reading raw data
2. Fitting scorers/embeddings **strictly on train split**
3. Assigning train/val/test splits
4. Emitting the long-format fusion CSV

---

## 3.2 MVTec 3D-AD (All Variants)

### Raw Data Source
- **Location expected:** `data/raw/mvtec3d/<category>/<split>/<defect_type>/rgb/*.png` + `depth_or_xyz/*.tiff`
- **Public dataset:** MVTec 3D-AD (Bergmann et al. 2022), publicly available
- **Samples:** 3,226 paired samples, 8 categories, 22.4% positive

### Preparation Script: `src/scripts/prepare_mvtec3d_fusion_benchmark.py`

**Score fitting discipline:**
- `normal_reference_mask = train_mask & (defect_types == "good")`
- Scorer fitted on **train/good only** — correct one-class assumption
- For image_statistics mode: Mahalanobis-style distance from normal centroid
- For patchcore mode: kNN distance to ResNet-50 memory bank of train/good samples
- Score normalization fitted on train fold only (`_minmax` with `fit_mask=train_mask`)
- Embeddings: min-max normalized using train statistics

**Split assignment:**

| Protocol | How splits assigned |
|---|---|
| Canonical (one-class) | MVTec official splits preserved verbatim via `split` column |
| Supervised Paired (SP) | MVTec test rows redistributed: stratified by (category, label), seed=42, val_fraction=0.15, test_fraction=0.30 |
| Held-Out Category | bagel/cable_gland/cookie/dowel → train; foam/peach/rope/tire → test; 20% of in-category test rows → validation |

**Leakage assessment:**
- ✅ Scorer fit on train/good only — **no test label leakage into scoring**
- ✅ Embeddings normalized on train fold only
- ✅ Supervised paired redistribution is stratified by (category, label) without using test labels for scoring
- ⚠️ IMPORTANT CAVEAT: The ResNet-50 PCA projection is fitted on `train_mask` rows, which for the **canonical protocol** are normal-only. PCA components thus reflect normal-class structure only — correct. For **supervised-paired**, the scorer remains fit on original train/good; only the fusion labels are redistributed.

---

## 3.3 UNSW-NB15 (Naturally Paired)

### Raw Data Source
- **Files expected:** `data/raw/cyber/UNSW_NB15_training-set.csv`, `data/raw/cyber/UNSW_NB15_testing-set.csv`
- **Public dataset:** UNSW-NB15 (Moustafa & Slay 2015), publicly available
- **Samples:** 60,000 (subsampled, seed=42), 3 domains (flow/connection/context)
- **Positive fraction:** 63.9%

### Preparation Script: `src/scripts/prepare_unsw_paired_fusion_benchmark.py`

**Domain feature assignment:**
- `flow`: 13 timing/throughput features
- `connection`: 14 protocol/session features
- `context`: 12 aggregated window counters

**Score fitting discipline:**
- Score = Mahalanobis-style distance from **train split** normal centroid
- `_domain_score(block, train_mask)` — strictly uses `train_mask` rows for fitting
- Embedding normalization: `_minmax_clip(block[:, j], train_mask)` — train-only

**Split assignment:**
- Default: `_patient_style_stratified_split` — stratified by (attack_cat, label), val_fraction=0.15, test_fraction=0.30, seed=42
- Held-out attack: `Backdoor, Shellcode, Worms` → test only; remaining stratified

**Known UNSW-NB15 leakage risk (acknowledged in codebase):**
- ⚠️ The standard UNSW-NB15 train/test split has the same attack categories in both sets. The `unsw_paired` canonical run therefore uses a **fresh stratified split** (not the original UNSW split), meaning all attack categories appear in train. This is a **design choice** that avoids held-out generalization.
- ✅ The `unsw_heldout_attack` variant addresses this by routing Backdoor/Shellcode/Worms entirely to test — this is the leakage-defensive protocol.
- Evidence: `configs/attention_unsw_heldout_attack.yaml` and metadata comment: "Defends against UNSW-NB15's known train/test attack-category overlap"

---

## 3.4 VisA

### Raw Data Source
- **Location expected:** `data/raw/visa/` with official `split_csv/1cls.csv`
- **Public dataset:** VisA (Zou et al. ECCV 2022)
- **Domain construction:** RGB + Sobel-edge-proxy (both derived from same image)

### Preparation Script: `src/scripts/prepare_visa_fusion_benchmark.py`

**Score fitting discipline:**
- PatchCore kNN score fitted on `normal_reference_mask = train_mask & (labels == 0)`
- `patchcore_knn_score(rgb_features, normal_reference_mask, k=3)`
- PCA projection fitted on `train_mask`
- Validation carved from official train: 15% of train indices, seed=42

**Split assignment:**
- Canonical: official VisA 1-class split + 15% of train → validation
- Supervised paired: test rows redistributed (stratified by category+label, seed=42, test_fraction=0.30)

**Leakage assessment:**
- ✅ Score fitting train-only, normal-only
- ✅ PCA fitting train-only
- ✅ Supervised paired redistribution is from test rows only (original train/val untouched)

---

## 3.5 Real3D-AD

### Raw Data Source
- **Public dataset:** Real3D-AD (Liu et al. 2023), point cloud dataset
- **Domains:** pointcloud (FPFH descriptors) + depth_or_xyz (ResNet-50 on depth projection)
- **Samples:** 1,254 samples, 12 categories, 48.1% positive

### Preparation Script: `src/scripts/prepare_real3d_fusion_benchmark.py`

**Score fitting discipline:**
- `normal_reference_samples: 48` (train/good only — very small)
- PatchCore kNN score on train-good memory bank
- ⚠️ **RISK:** Only 48 training normal reference samples is very small for PCA fitting and memory bank quality. This is acknowledged in metadata but represents a data quality constraint.

---

## 3.6 ELARA-Bench-LA (Secondary / Label-Aligned)

### Raw Data Source
- 4 public/local datasets: Credit Card Fraud, UNSW-NB15, Online Shoppers Intention, labeled news text
- **Generated:** 8,000 composite samples, 4 domains, 0.307 positive rate
- Preparation involves: domain-specific out-of-fold scorers, then common fusion schema

**Evidence from paper (§Benchmark Construction):**
- "domain-specific out-of-fold scorers are trained first"
- "their scores, confidences, and compact embeddings are then converted into the common fusion schema"
- **Out-of-fold discipline:** scorers evaluated on held-out folds to prevent within-domain leakage

**Leakage assessment:**
- ✅ Out-of-fold scoring described as the method — prevents within-domain train-test contamination
- ⚠️ The preparation script for ELARA-Bench-LA was **not directly read** during this audit. Evidence is from paper text and domain scorer pkl files found in `src/models/`
- **STATUS: PARTIALLY CONFIRMED** — domain scorer pkl files exist; full data preparation code not inspected

---

## 3.7 Data Lineage Summary Table

| Benchmark | Raw Source | Score Fit Split | Embedding Fit Split | Leakage Risk | Status |
|---|---|---|---|---|---|
| MVTec 3D-AD (image_stats) | MVTec 3D official | train/good only | train only | Low | ✅ CONFIRMED |
| MVTec 3D-AD (PatchCore) | MVTec 3D official | train/good only | train only | Low | ✅ CONFIRMED |
| MVTec 3D-AD SP | MVTec 3D official | train/good only | train only | Low (redistribution only) | ✅ CONFIRMED |
| MVTec 3D-AD Held-Out | MVTec 3D official | train/good in-category only | train only | Low | ✅ CONFIRMED |
| UNSW-NB15 (paired) | UNSW official | train split | train split | ⚠️ UNSW train/test overlap (fresh split used) | PARTIAL |
| UNSW-NB15 (held-out attack) | UNSW official | train split | train split | Low (attack held out) | ✅ CONFIRMED |
| VisA | VisA 1cls official | train/good only | train only | Low | ✅ CONFIRMED |
| Real3D-AD | Real3D official | train/good only (n=48) | train only | ⚠️ Small reference set | PARTIAL |
| ELARA-Bench-LA | 4 public/local | out-of-fold | UNKNOWN | ⚠️ Not directly inspected | UNKNOWN |
