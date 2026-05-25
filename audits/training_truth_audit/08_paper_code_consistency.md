# ELARA / RGA / RGA+ Training Truth Audit
## Part 9: Paper-to-Code Consistency Check
**Audit Date:** 2026-05-23  
**Evidence sources:** `PAPER_DRAFT_v1.tex`, `claim_matrix.csv`, `switching_certificate_t5_audit.json`, `reliability_estimator.py`, `reliability_boosted_fusion.py`, configs

---

## 9.1 Method Description vs. Code

### 9.1.1 Reliability Formula (Paper Eq. 1 vs. Code)

**Paper states (§Method, Eq. 2):**
```
r_d = α(1 − ECE_d) + β·p_KS,d + γ·S_d
with (α, β, γ) = (0.45, 0.35, 0.20)
```

**Code (`reliability_estimator.py:lines 164-168`):**
```python
rel_d = (
    self.ece_weight * ece_reliability       # 0.45 × (1 - ECE_d)
    + self.ks_weight * ks_reliability       # 0.35 × KS_p-value
    + self.sharpness_weight * sharpness     # 0.20 × sharpness
)
```

**Verdict: ✅ EXACT MATCH**  
Weights confirmed in all configs as `ece_weight: 0.45, ks_weight: 0.35, sharpness_weight: 0.20`

---

### 9.1.2 Gate Logic (Paper vs. Code)

**Paper states (§Method):**
> "If r̄ ≥ τ, ELARA uses the static attention path. If r̄ < τ, reliability weights are injected."
> "τ = 0.66 (default)"

**Code (`reliability_estimator.py:lines 176-212`):**
```python
mean_fire = mean_r < threshold   # fire = True → use reliability path
# Returns True (activate reliability) when mean_r < τ
```

**Paper also notes:** "If the same rule is written with unreliability u=1−r̄, the correct threshold is u>1−τ=0.34, not u>0.66." This is an explicit mathematical clarification in the paper matching the code.

**Verdict: ✅ EXACT MATCH**

---

### 9.1.3 RGA+ Candidate Grid (Paper vs. Code)

**Paper states (§Method, RGA+ subsection):**
> "a reliability-boosted fusion head trains candidate classifiers... The current candidate set contains histogram gradient-boosted heads and calibrated logistic heads. The selected head is the one with the best validation ROC-AUC."

**Code:** 7 HGB + 4 LogReg = 11 candidates (confirmed from `reliability_boosted_fusion.py`)

**Verdict: ✅ CONSISTENT** (paper does not enumerate all 11 candidates, which is acceptable)

---

### 9.1.4 Router Logic (Paper vs. Code)

**Paper states:**
> "a validation-only router can choose among static attention, base RGA, RGA+ boosted fusion, and the strong score-level baselines"

**Code:** Meta router candidate pool = all 11 method outputs + logistic stack + mean ensembles

**Verdict: ✅ CONSISTENT** (paper summary accurate; code more detailed than described)

---

### 9.1.5 Algorithm Pseudocode (Paper Algorithm 1 vs. Code)

**Paper Algorithm 1, Step 2:**
```
r_{i,d} ← α(1 − ECE_d) + β·p_KS,d(x_{i,d}) + γ·S_{i,d}
```

**Code comment in `reliability_estimator.py:line 505`:**
```
# The paper (ELARA, 2026) calls this component "RGA" (Reliability-Gated Attention).
# The code uses the internal project name "CRAF" (Calibration-aware Reliability-
# Adaptive Fusion). Both names refer to the same class.
```

**Verdict: ✅ MATCH** — the paper's Algorithm 1 faithfully describes the `compute_reliability_weights()` logic. The paper uses per-sample notation (r_{i,d}) but the default implementation is batch-level (all samples in batch get the same scalar). This is disclosed in the paper: "The reported runs use the batch-level reliability vector described above."

---

### 9.1.6 One-Class Protocol Collapse (Paper Claim vs. Results)

**Paper claims (Abstract):**
> "on the naturally paired MVTec 3D-AD benchmark under its canonical one-class anomaly-detection protocol... every supervised fusion baseline... remains near chance ROC-AUC"

**Results from `mvtec3d_patchcore_results.json`:**
```
static_attention:     mean ROC-AUC = 0.4909
craf_attention:       mean ROC-AUC = 0.5143
rga_boosted_fusion:   mean ROC-AUC = 0.5000
random_forest:        mean ROC-AUC = 0.5000
early_fusion_mlp:     mean ROC-AUC = 0.5468
```

**Verdict: ✅ CONFIRMED** — paper claim is accurate. All supervised methods are near chance.

---

### 9.1.7 KS Component Ablation Claim (Paper vs. Results)

**Paper claims (Abstract):**
> "the Kolmogorov–Smirnov drift signal carries the entire all-domain coherent score-collapse robustness gain (+0.0506 zero_attack, +0.0319 max_attack) delivered by the reliability gate; removing the KS-drift component eliminates the gain."

**Results from `mvtec3d_patchcore_results.json` component_ablation (target=all, best mechanism evidence):**
- full + zero_attack: static=0.4098, craf=0.4866, Δ=+0.0768
- full + max_attack: static=0.4142, craf=0.4885, Δ=+0.0743
- no_ks + various: reported as worse performance in ablation

**NOTE:** The abstract claims refer to ELARA-Bench-LA (the secondary benchmark), not MVTec. The MVTec ablation numbers above are protocol-diagnostic. The ELARA-Bench-LA ablation results (B1/B2 in the Family B descriptive surface) were not directly read in this audit.

**Verdict: ⚠️ PARTIALLY VERIFIED** — The KS ablation numbers for ELARA-Bench-LA (+0.0506, +0.0319) are cited in the abstract but the corresponding ELARA-Bench-LA result files were not directly inspected. The MVTec ablation confirms ablation infrastructure works.

---

### 9.1.8 UNSW-NB15 p-value Claim (Paper vs. Audit)

**Paper (audited claim M006):**
> "single-representative-seed DeLong p=1.3×10⁻⁶, p_Holm=6.7×10⁻⁶ within Family A K=5; point-estimate delta = +0.0003 ROC-AUC"

**Switching certificate:** UNSW-NB15 certified (LCB=+0.0099), RGA ROC-AUC=0.9893, static=0.9790, Δ=+0.0103  
**NOTE:** The paper's Δ=+0.0003 refers to a *DeLong paired comparison against the validation-frozen comparator (random forest)*, which may be different from the static→RGA gap. The switching certificate compares against static attention.

**Verdict: ⚠️ CONSISTENT BUT REQUIRES VERIFICATION** — Two different comparisons (static attention vs. val-frozen RF). The DeLong p-value is plausible given n=55,491 events.

---

### 9.1.9 Baseline Roster (Paper vs. Code)

**Paper (post-M012 correction):**
> "Random forest, early-fusion MLP, late-fusion ensemble, confidence-weighted mean, Tent, TTT, EATA, and SAR score adapters, plus static attention"

**Code (`baselines.py`):**
```python
EarlyFusionMLP, LateFusionEnsemble, RandomForestFusion, ConfidenceWeightedMean,
TentScoreAdapter, TTTPseudoLabelAdapter, EATAScoreAdapter, SARScoreAdapter
```

**Verdict: ✅ EXACT MATCH** (after M012 correction added EATA and SAR)

---

### 9.1.10 Domain Counts and Benchmark Sizes

| Benchmark | Paper claim | Code/metadata confirmed |
|---|---|---|
| MVTec 3D-AD samples | "3,226 paired samples, 22.4% positive" | metadata: 3226 samples, 0.2244 positive ✅ |
| MVTec 3D-AD categories | "eight categories" | 8 categories ✅ |
| ELARA-Bench-LA | "8,000 composite samples, 4 domains, 0.307 positive" | From paper text — not directly verified from code |
| UNSW-NB15 | "naturally paired, three modalities" | 3 domains (flow/connection/context) ✅ |
| VisA SP | "supervised-paired" | n_samples=10,821 ✅ |
| Real3D-AD | "12 categories" | 12 categories ✅ |

---

## 9.2 Nomenclature Cross-Reference

| Paper name | Code name | Notes |
|---|---|---|
| RGA | CRAF / ReliabilityEstimator | Aliased via `RGAReliabilityEstimator = ReliabilityEstimator` |
| ELARA | System boundary (whole pipeline) | — |
| craf_attention | CRAF attention (= RGA) | Output key in results JSON |
| rga_boosted_fusion | RGA+ boost head | Output key in results JSON |
| rga_meta_router | RGA+ router | Output key in results JSON |
| static_attention | Static path (no reliability weighting) | Output key in results JSON |
| TTRA | Test-Time Reliability Adaptation | Module docstring term |

---

## 9.3 Consistency Summary

| Claim | Status |
|---|---|
| Reliability formula (α, β, γ weights) | ✅ EXACT MATCH |
| Gate threshold τ=0.66 | ✅ EXACT MATCH |
| One-class protocol collapse | ✅ CONFIRMED |
| Baseline roster (post-correction) | ✅ EXACT MATCH |
| Val-frozen selection rule | ✅ CONFIRMED |
| MVTec size/categories | ✅ CONFIRMED |
| KS ablation numbers (ELARA-Bench-LA) | ⚠️ NOT DIRECTLY VERIFIED (files not read) |
| UNSW p-value chain | ⚠️ CONSISTENT BUT TWO-COMPARATOR AMBIGUITY |
| Causal language removal | ✅ CORRECTED (M008) |
| PR/ECE/Brier omission from canonical | ✅ CORRECTED (M009) |
