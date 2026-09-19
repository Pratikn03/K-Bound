# K-Bound and KGA: Reproducibility and Artifact Provenance Guide

This document provides a consolidated mapping between manuscript tables, figures, theoretical declarations, empirical data artifacts, and executable reproduction scripts.

---

## 1. Environment and Toolchain Requirements

- **Python Version**: Python 3.11 or Python 3.12 (tested on Python 3.11.14 and 3.12.13)
- **Core Dependencies**: `torch >= 2.0.0`, `numpy >= 1.24.0`, `scipy >= 1.10.0`, `scikit-learn >= 1.3.0`
- **Formal Verification**: Lean 4 (`v4.8.0` / Mathlib4) for kernel-checked mathematical declarations
- **LaTeX Engine**: TeX Live 2024/2025 (`latexmk`, `pdflatex`, `biber`/`bibtex`)

---

## 2. Table-to-Artifact Mapping

| Table ID in Manuscript | Title / Topic | Generating Script / Source | Output Artifacts & Manifests |
|:---|:---|:---|:---|
| **Table 1** (Body) | Theory--Algorithm Bridge | Static narrative specification | `kbound_submission_body.tex` |
| **Table 2** (Body) | Primary CIFAR-10-C (Protocol B) | `scripts/reconcile_result_panels.py` | `paper/generated/kbound_primary_accuracy_table.tex`, `experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json` |
| **Table 3** (Body) | Primary Regret Differences | `scripts/reconcile_result_panels.py` | `paper/generated/cifar10c_primary_regret_diffs.tex` |
| **Table 4** (Body) | Synchronized Baselines (2,160 Cells) | `scripts/replay_cifar_current_arithmetic.py` | `paper/generated/current_cifar_baselines_20260912/current_arithmetic_panel.tex` |
| **Table 5** (Body) | CIFAR-10-C Decision Baselines | `scripts/replay_cifar_current_arithmetic.py` | `paper/generated/tab_cifar10c_decision_rules_errors.tex` |
| **Table 6** (Body) | CCT-20 Safe Utility Evaluation | `scripts/make_tables.py` | `paper/generated/cct20_primary_table_display.tex`, `paper/generated/cct20_release_manifest.json` |
| **Table 7** (Body) | CCT-20 Location Effects | `scripts/make_tables.py` | `paper/generated/cct20_location_effects_display.tex` |
| **Table 8** (Body) | Interval Diagnostics (Protocol B) | `scripts/build_current_policy_interval_diagnostics.py` | `paper/generated/current_policy_interval_diagnostics.tex` |
| **Table 9** (Body) | ImageNet-C 108-Cell Population Replay | `scripts/replay_task2_population_panel.py` | `paper/generated/tab_imagenetc_108_population_replay.tex`, `experiments/kbound/results/task2_imagenetc_population_replay_20260911/population_panel.json` |
| **Table 10/11** (Supp) | Auxiliary Reconciled Panel Metrics | `scripts/reconcile_result_panels.py` | `paper/generated/kbound_auxiliary_accuracy_table.tex`, `paper/generated/kbound_auxiliary_balanced_accuracy_table.tex` |
| **Table 12** (Supp) | Synthetic Population Diagnostic | `experiments/kbound/synthetic_population_diagnostic.py` | `experiments/kbound/results/synthetic_diagnostic/diagnostic_summary.json` |
| **Table 13** (Supp) | Non-Oracle Population Diagnostic | `experiments/kbound/synthetic_population_diagnostic.py` | `experiments/kbound/results/synthetic_diagnostic/nonoracle_diagnostic_summary.json` |
| **Table 14** (Supp) | Matched SAR Control Arms | `docs/research/kbound/scripts/run_task4_matched_sar_panel.py` | `paper/generated/tab_task4_matched_sar.tex`, `experiments/kbound/results/task4_matched_sar_v1/` |
| **Table 15** (Supp) | Corrected-BN SAR Five-Seed Study | `scripts/replay_sar_release_gate.py` | `paper/generated/tab_sar_corrected_fiveseed.tex`, `experiments/kbound/results/sar_release_gate_20260913/` |
| **Table 16** (Supp) | Independent Model Checkpoint Replication | `scripts/run_independent_source_model_replication.py` | `paper/generated/tab_task1_independent_models.tex`, `experiments/kbound/results/task1_independent_models_replication/` |
| **Table 17** (Supp) | Camelyon17 Pathology OOD Diagnostic | `scripts/run_camelyon_diagnostics_resolved.py` | `paper/generated/tab_camelyon17_adapt_diagnostic.tex`, `experiments/kbound/results/camelyon17_adapt_diagnostic/` |
| **Table 18** (Supp) | ImageNet-R Multi-Architecture Panel | `scripts/generate_statistical_synthesis_tables.py` | `paper/generated/tab_architecture_generalization.tex`, `experiments/kbound/results/task_vit_architecture/` |
| **Table 19** (Supp) | Native-Component 45-Cell ImageNet-C | `scripts/run_task3_native_benchmark_panel.py` | `paper/generated/tab_native_component_45.tex`, `experiments/kbound/results/task3_native_common_45_20260914_score/` |
| **Table 20** (Supp) | Master Statistical Synthesis Matrix | `scripts/generate_statistical_synthesis_tables.py` | `paper/generated/tab_master_statistical_synthesis.tex` |

---

## 3. Figure Generation Mapping

| Figure | Description | Generating Script | Output Artifact |
|:---|:---|:---|:---|
| **Figure 1** | Partial-Identification Frontier Schematic | `scripts/make_submission_figures.py --frontier-only` | `figures/fig_frontier_schematic.pdf` |
| **Figure 2** | Certificate Geometry and Decision Thresholds | `scripts/plot_kga_interval_rule.py` | `figures/fig_certificate.pdf` |
| **Figure 3** | Decisive Pareto Frontiers on CIFAR-10-C | `scripts/make_submission_figures.py` | `figures/fig_decisive_pareto_cifar10c.pdf` |
| **Figure 4** | Non-Stationary Online Sequential Tracking | `scripts/make_submission_figures.py` | `figures/fig_cifar_tent_online.pdf` |

---

## 4. Execution Commands

### A. Run Verification Tests
```bash
pytest tests/test_kbound_narrative_revision.py \
       tests/test_kbound_estimand_inference_wording.py \
       tests/test_manuscript_claim_consistency.py
```

### B. Validate Manuscript Claim Integrity
```bash
python src/scripts/validate_manuscript_claims.py
```

### C. Build Main Submission and Split Supplemental PDFs
```bash
bash docs/research/kbound/scripts/build_pdfs.sh
```

Compilation produces:
- `docs/research/kbound/kbound_submission.pdf` (Combined manuscript + supplementary)
- `docs/research/kbound/kbound_short_main.pdf` (Standalone main body, 26 pages)
- `docs/research/kbound/kbound_short_supplement.pdf` (Standalone supplementary appendices)
- `docs/research/kbound/kbound_tmlr.pdf` (TMLR format)
