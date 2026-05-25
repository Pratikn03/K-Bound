# ELARA / RGA / RGA+ Training Truth Audit
## Part 10: Publication Readiness Checklist
**Audit Date:** 2026-05-23

---

## 10.1 Methodology Checklist

| Item | Status | Notes |
|---|---|---|
| Training/validation/test split discipline | ✅ PASS | Pre-assigned splits, train-only scorer fitting |
| Test labels not used for model selection | ✅ PASS | All selection on validation |
| Multiple seeds reported | ✅ PASS | 5 or 30 seeds depending on benchmark |
| Baseline comparison includes strong methods | ✅ PASS | 8 baselines including RF, MLP, Tent, TTT, EATA, SAR |
| Holm-Bonferroni multiple testing correction | ✅ PASS | K=5 within Family A |
| Statistical significance reported with effect size | ✅ PASS | DeLong p-values + Δ ROC-AUC |
| Non-significant results disclosed | ✅ PASS | 3 of 5 Family A cells n.s.; canonical protocol cells reported as diagnostic |
| Practical effect size discussed | ✅ PASS | UNSW Δ=+0.0003 explicitly called "negligible" |
| Causality claims removed | ✅ PASS | M008 corrected to "model-response sensitivity" |
| Protocol diagnostic cells separated from confirmatory | ✅ PASS | Family A/B/C/D four-family split |
| One-class protocol collapse acknowledged | ✅ PASS | Paper §Benchmark Construction "Protocol erratum" |
| Historical corrections documented | ✅ PASS | claim_matrix.csv, 14 items |

---

## 10.2 Reproducibility Checklist

| Item | Status | Notes |
|---|---|---|
| Config files for all benchmarks | ✅ PRESENT | 11 config YAMLs |
| Preparation scripts for all benchmarks | ✅ PRESENT | 5 prepare_*.py scripts |
| Metadata JSONs for all benchmarks | ✅ PRESENT | 8 metadata files |
| Result artifacts (JSON) | ✅ PRESENT | Results in experiments/fusion/ |
| Model checkpoints (.pt) | ❌ ABSENT | No .pt files found in workspace |
| Domain expert models (.pkl) | ✅ PARTIAL | fraud/cyber/behavior present; text model not found |
| Rebuild paper script | ✅ PRESENT | scripts/rebuild_paper.sh |
| Training runner script | ✅ PRESENT | scripts/run_train_vision.sh |
| Seed set in all training runs | ✅ CONFIRMED | set_seed() called before training |
| Data hash in metrics output | ✅ PRESENT | hash_file() in train_attention_fusion.py |
| **Missing:** .pt checkpoints | ❌ CRITICAL GAP | Models cannot be reloaded without retraining |
| **Missing:** ELARA-Bench-LA preparation script inspection | ⚠️ GAP | Secondary benchmark prep not audited |

---

## 10.3 Transparency Checklist

| Item | Status | Notes |
|---|---|---|
| Limitations section in paper | ✅ PRESENT | §Limitations in paper |
| Construct validity boundary stated | ✅ PRESENT | "fusion and verification layer" scope stated |
| UNSW-NB15 attack-category overlap disclosed | ✅ PRESENT | Held-out protocol exists and is described |
| Real3D-AD small reference set disclosed | ✅ PRESENT | metadata important_limitation field |
| ELARA-Bench-LA label-alignment limitation disclosed | ✅ PRESENT | "synthetically paired" terminology used |
| Val-frozen comparator selection disclosed | ✅ PRESENT | Abstract and §Master comparison table |
| Switching certificate documented | ✅ PRESENT | switching_certificate_t5_audit.json |
| PAC slack documented | ✅ PRESENT | meta_router_pac_audit.json |
| Code vs. paper name mapping | ✅ PRESENT | reliability_estimator.py alias comment |
| Phase 0.6 policy lock documented | ✅ PRESENT | Referenced in paper abstract |

---

## 10.4 Open Issues for Publication

### BLOCKING (must fix before submission)

| # | Issue | Location | Fix Required |
|---|---|---|---|
| B1 | No saved model checkpoints (.pt) | `models/fusion/attention_*/` empty | Either commit checkpoints or explicitly document that models must be retrained from scripts |
| B2 | ELARA-Bench-LA data preparation not fully audited | `prepare_elara_bench_la.py` (not inspected) | Read and audit the full ELARA-Bench-LA preparation pipeline |
| B3 | KS ablation numbers (+0.0506, +0.0319) for ELARA-Bench-LA not verified from code | Cited in Abstract | Directly read the ELARA-Bench-LA results JSON to confirm these numbers |

### IMPORTANT (should address before submission)

| # | Issue | Location | Note |
|---|---|---|---|
| I1 | Reliability weight triplet (0.45, 0.35, 0.20) presented without empirical justification | Paper §Method | State explicitly that these are engineering priors, not searched. Add a sentence. |
| I2 | Gate threshold τ=0.66 presented without empirical justification | Paper §Method | Same as I1 — explicitly label as heuristic |
| I3 | Val-fold triple use (early stop + reliability fit + RGA+ selection) | Architecture | Acknowledge this limitation explicitly; it's standard but should be stated |
| I4 | 30-seed runs use fixed split — seed variation is initialization only | Paper §Evaluation | Clarify that multi-seed reflects initialization variance, not split variance |
| I5 | UNSW Δ=+0.0003 ROC-AUC as the only Holm-significant result | Abstract/Conclusions | This is already disclosed; ensure conclusions don't overstate |
| I6 | Real3D-AD n=48 normal references is very small | metadata | Disclose in paper as a data quality note |
| I7 | `torch.backends.cudnn.deterministic` not confirmed set | `train_attention_fusion.py` | Add for strict reproducibility; currently only np/torch seeds are set |

### MINOR (quality improvements)

| # | Issue | Fix |
|---|---|---|
| m1 | Paper Eq. 1 uses per-sample r_{i,d} notation but default is batch-level | Already disclosed in §Method; add explicit footnote |
| m2 | `attention_config.yaml` default model uses embed_dim=64/num_heads=8 but benchmarks use 32/48 with 4 heads | Document that default config is not used for any reported benchmark |
| m3 | `rga_meta_router` vs `craf_attention` naming in result JSONs vs paper | Add key→paper name table in supplementary |

---

## 10.5 Overall Readiness Score

| Dimension | Score | Notes |
|---|---|---|
| **Methodology** | 9/10 | All selection validation-only; Holm correction applied; historical leakage corrected |
| **Reproducibility** | 6/10 | Scripts present but no saved checkpoints; ELARA-Bench-LA not fully documented |
| **Transparency** | 9/10 | Corrections documented; limitations explicit; Phase 0.6 policy lock present |
| **Statistical Rigor** | 8/10 | Multi-seed, DeLong p-values, Holm K=5; UNSW significance with tiny effect |
| **Claim-Code Match** | 8/10 | 13/14 claims verified; KS ablation numbers need direct verification |

**Overall: CONDITIONALLY READY for submission** — blocking issues B1–B3 must be resolved first.
