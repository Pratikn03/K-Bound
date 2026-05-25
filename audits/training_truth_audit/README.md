# ELARA / RGA / RGA+ Training Truth Audit — Index
**Audit Date:** 2026-05-23  
**Repository:** `/Volumes/T9/uav/AutoML_Flagship_V8/`  
**Audit Directory:** `audits/training_truth_audit/`

---

## Audit Reports

| File | Title | Summary |
|---|---|---|
| [00_repository_training_map.md](00_repository_training_map.md) | Repository Training Map | Complete directory inventory, benchmark families, checkpoint locations, result artifacts |
| [01_model_inventory.md](01_model_inventory.md) | Model Inventory | 6 model types with architecture details, hyperparameters per benchmark, serialization status |
| [02_data_lineage_audit.md](02_data_lineage_audit.md) | Data Lineage Audit | Per-benchmark data flow, scorer fitting discipline, split assignment, leakage assessment |
| [03_parameter_hyperparameter_registry.md](03_parameter_hyperparameter_registry.md) | Parameter Registry | Complete hyperparameter table for all models and all benchmarks |
| [04_training_pipeline_reconstruction.md](04_training_pipeline_reconstruction.md) | Training Pipeline Reconstruction | Step-by-step pipeline, critical implementation details, reproducibility assessment |
| [05_tuning_model_selection_audit.md](05_tuning_model_selection_audit.md) | Tuning and Model Selection Audit | Evidence that all selection is validation-only; claim matrix verification (14 claims) |
| [06_overfitting_audit.md](06_overfitting_audit.md) | Overfitting Audit | Seed variance analysis, val-fold reuse, multiple testing, switching certificate robustness |
| [07_leakage_testpeeking_audit.md](07_leakage_testpeeking_audit.md) | Leakage and Test-Peeking Audit | Test-peeking risk matrix, historical leakage corrections, data contamination guards |
| [08_paper_code_consistency.md](08_paper_code_consistency.md) | Paper-to-Code Consistency Check | 10 specific claims verified against code; nomenclature cross-reference |
| [09_publication_readiness_checklist.md](09_publication_readiness_checklist.md) | Publication Readiness Checklist | Methodology, reproducibility, transparency checklists; 3 blocking + 7 important issues |
| [10_executive_final_report.md](10_executive_final_report.md) | Executive Final Report | Complete audit summary, key numerical results, final verdict |
| [11_audit_summary.csv](11_audit_summary.csv) | Machine-Readable Summary | 40 audit findings with severity, status, and evidence citations |

---

## Quick Reference: Blocking Issues

| ID | Issue | File |
|---|---|---|
| B1 | No saved model checkpoints (.pt) in workspace | [09_publication_readiness_checklist.md](09_publication_readiness_checklist.md) |
| B2 | ELARA-Bench-LA preparation script not audited | [02_data_lineage_audit.md](02_data_lineage_audit.md) |
| B3 | KS ablation numbers (+0.0506/+0.0319) not verified from ELARA-Bench-LA results JSON | [08_paper_code_consistency.md](08_paper_code_consistency.md) |

## Quick Reference: Key Findings

| Finding | Verdict |
|---|---|
| Test labels used in any selection | ❌ NOT FOUND — all selection on validation only |
| Historical test-oracle selection | ⚠️ WAS PRESENT — corrected (claim_matrix.csv M001/M004) |
| Reliability formula matches paper | ✅ EXACT MATCH |
| Gate threshold τ=0.66 matches paper | ✅ EXACT MATCH |
| Baseline roster matches paper | ✅ CONFIRMED (after M012 correction) |
| Causal language in paper | ✅ CORRECTED to model-response sensitivity (M008) |
| Canonical protocol collapse | ✅ CONFIRMED — all methods ≈chance ROC-AUC |
| Only UNSW statistically significant | ✅ CONFIRMED — Δ=+0.0003 (negligible effect) |
| Sign flip on MVTec LOCO SP | ✅ CONFIRMED — was +0.008, now −0.008 (n.s.) |
| Real3D RGA+ no longer leads baselines | ✅ CONFIRMED — 0.534 vs TTT 0.537 |

---

## Audit Scope Limitations

1. **ELARA-Bench-LA preparation pipeline** was not directly read (secondary benchmark scripts)
2. **LearnedReliabilityGate / learned_gate.py** was not fully read (only referenced)
3. **Attention model forward pass** (`cross_modal_attention.py`) was not read (architecture confirmed from config inference)
4. **evaluate_attention_harness.py** (main multi-seed runner) was not fully read
5. **ELARA-Bench-LA result JSONs** (KS ablation numbers) were not directly read

These gaps correspond to blocking issues B1–B3. Core pipeline, data preparation, and model selection were fully audited.
