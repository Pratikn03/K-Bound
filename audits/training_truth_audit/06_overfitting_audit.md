# ELARA / RGA / RGA+ Training Truth Audit
## Part 7: Overfitting Audit
**Audit Date:** 2026-05-23  
**Evidence sources:** `mvtec3d_patchcore_results.json`, `mvtec3d_patchcore_supervised_paired_results.json`, `switching_certificate_t5_audit.json`, all configs

---

## 7.1 Overfitting Risk Factors

| Risk Factor | Present? | Severity | Evidence |
|---|---|---|---|
| Single fixed seed for all decisions | ⚠️ Partial | Medium | Primary seed=42, but multi-seed (5 or 30) evaluation used |
| No test-held-out model selection | ✅ Clean | — | All selection on validation |
| Same val fold used for 3 purposes (early stop + reliability fit + RGA+ selection) | ⚠️ Present | Low-Medium | By design; standard practice |
| Small test fold (especially Real3D n=1254) | ⚠️ Present | Medium | Real3D n_test ~375; CI widths large |
| High variance across seeds | ⚠️ Present in canonical protocols | High | static_attention std=0.0616 on canonical MVTec |
| 30 seeds evaluated on same fixed test fold | ⚠️ Present | Medium | 30-seed configs use same CSV split |
| Architecture tuned per benchmark class | ✅ Minimal | — | Only embed_dim (32 vs 48) differs |

---

## 7.2 Seed Variance Analysis

### MVTec 3D-AD PatchCore Canonical (5 seeds)
From `mvtec3d_patchcore_results.json` clean_metric_summary:

| Method | ROC-AUC mean | std | CI low | CI high |
|---|---|---|---|---|
| static_attention | 0.4909 | **0.0616** | 0.4042 | 0.5594 |
| craf_attention | 0.5143 | 0.0139 | 0.500 | 0.5345 |
| rga_meta_router | 0.5143 | 0.0139 | 0.500 | 0.5345 |
| rga_boosted_fusion | 0.5000 | 0.0000 | 0.500 | 0.500 |
| early_fusion_mlp | 0.5468 | 0.0108 | 0.5315 | 0.5618 |
| confidence_weighted_mean | 0.5998 | 0.0000 | 0.5998 | 0.5998 |

**Interpretation:** Under one-class canonical protocol, `static_attention` has σ=0.062 — very high variance for a near-chance model. All supervised models collapse to ≈0.5. The large std for static_attention is expected under canonical one-class (model learns from normal-only training data, producing near-random test discriminability with initialization-dependent direction).

### MVTec 3D-AD PatchCore Supervised Paired (30 seeds)
From `switching_certificate_t5_audit.json`:
```json
"rga_auroc_mean": 0.7390,
"static_auroc_mean": 0.6337,
"paired_benefit_mean": 0.1052,
"lcb": 0.1022,
"certified": true
```
With 30 seeds the LCB on the paired benefit is +0.102 — much tighter bounds than the 5-seed canonical case. The certification is robust.

---

## 7.3 Val-Fold Reuse (Triple Use of Validation Data)

The validation fold is used for:
1. **Early stopping** of AttentionFusionModel (saves best val PR-AUC checkpoint)
2. **ReliabilityEstimator fitting** (calibrators, KS references, ECE computation)
3. **RGA+ candidate selection** (among 11 boosted head candidates)

**Risk assessment:**
- Risks 1 and 2 are **standard practice** (post-hoc calibration always uses a held-out validation set)
- Risk 3 (candidate selection on same val fold) introduces mild optimism: the selected candidate has seen the val fold labels implicitly via the selection criterion
- The triple reuse means the val fold serves as both a calibration reference and a selection oracle — this is the standard stacking procedure, but it slightly inflates val-AUC estimates for the selected candidate
- **Impact on test metrics:** Because test evaluation uses a completely disjoint fold, the test metrics are not directly biased. The potential inflation is in the **selection decision** (we may pick a slightly suboptimal candidate), not in the reported test numbers
- **Severity: LOW** — standard for practical ML pipelines; acknowledged in the codebase

---

## 7.4 Overfitting to the Benchmark: Per-Benchmark Risk

### MVTec 3D-AD Canonical (One-Class)
- **Risk: LOW** — all methods at chance; no meaningful overfitting possible
- Static attention σ=0.062 is initialization variance, not overfitting

### MVTec 3D-AD Supervised Paired
- **Risk: MEDIUM** — 30-seed runs reduce selection noise, but the fixed split means all seeds train on the same fold
- The switching certificate LCB=+0.102 is a strong positive signal

### UNSW-NB15 (Naturally Paired)
- **Risk: LOW for test performance; MEDIUM for generalizability claim**
- 0.989 ROC-AUC for the canonical split reflects UNSW's difficulty: fresh stratified split contains same attack types in train/test
- Held-out attack protocol exists but attack categories are known before split design — this is a design limitation, not a data leak

### VisA Supervised Paired  
- **Risk: LOW** — 30-seed evaluation, large dataset (10,821 samples)
- Switching cert LCB=+0.043 is positive and tight

### Real3D-AD Supervised Paired
- **Risk: HIGH** — only 1,254 samples, 48 normal references for scoring
- 5-seed evaluation; individual seed variance not reported in the switched certificate
- Switching cert LCB=+0.037 but n=1,254 is very small for confident inference

### ELARA-Bench-LA
- **Risk: MEDIUM** — out-of-fold scoring reduces within-domain overfitting
- Label-aligned (synthetic pairing) may introduce distributional mismatch from the label-alignment procedure that benefits fusion models in ways not representative of natural co-occurrence

---

## 7.5 Multiple Testing Exposure

The paper evaluates multiple benchmarks (5 in Family A) and applies Holm correction within Family A (K=5). However:

- 30 seeds × 12 methods = 360 individual AUC values computed per benchmark
- The paper correctly restricts multiple testing corrections to the **5 Family A primary inferential cells**
- Family B (mechanism analysis), C (exploratory), and D (protocol diagnostics) are **descriptive only** — no Holm correction claimed
- This is the correct scientific posture and matches the Phase 0.6 policy lock

**Most critical result:** Only UNSW-NB15 (A8) achieves Holm-corrected significance (p_Holm=6.7e-6), but the practical effect is Δ=+0.0003 ROC-AUC. Statistical significance at n=55,491 paired events does not imply practical significance.

---

## 7.6 Switching Certificate Robustness Assessment

From `switching_certificate_t5_audit.json`:

| Benchmark | Protocol | n_seeds | paired_benefit | LCB | Certified |
|---|---|---|---|---|---|
| MVTec 3D-AD | PatchCore canonical | 5 | +0.0234 | **−0.0286** | ❌ NOT certified |
| MVTec 3D-AD | PatchCore supervised | 30 | +0.1052 | **+0.1022** | ✅ CERTIFIED |
| MVTec 3D-AD | PatchCore held-out | 30 | +0.0448 | **+0.0410** | ✅ CERTIFIED |
| MVTec LOCO-AD | PatchCore canonical | 5 | −0.0149 | −0.0399 | ❌ NOT certified |
| MVTec LOCO-AD | PatchCore supervised | 30 | +0.1027 | **+0.0987** | ✅ CERTIFIED |
| Real3D-AD | PCA shape+depth SP | 5 | +0.0401 | **+0.0374** | ✅ CERTIFIED |
| VisA | RGB+edge canonical | 5 | +0.0160 | −0.0357 | ❌ NOT certified |
| VisA | RGB+edge supervised | 30 | +0.0443 | **+0.0429** | ✅ CERTIFIED |
| UNSW-NB15 | flow/conn/context | 5 | +0.0103 | **+0.0099** | ✅ CERTIFIED |
| UNSW-NB15 | held-out attack | 5 | +0.0057 | **+0.0052** | ✅ CERTIFIED |

**Key findings:**
- All **canonical (one-class)** protocols are uncertified (LCB ≤ 0)
- All **supervised-paired** protocols are certified (LCB > 0) with 30 seeds
- Real3D-AD is certified despite only 5 seeds (tight benefit estimate at small effect)
- UNSW-NB15 is certified but Δ is tiny (+0.0103)
