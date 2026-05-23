# Phase 1 Changelog

**Branch:** `fix/elara-phase1-empirical-validity`
**Starting commit:** `b88ad3b` (Phase-D closure)
**Phase-0.6 lock commit:** policy artifacts only, on the same branch.

This file records every file modified or created during Phase 1, in order.
Subsections track each Phase 1 sub-stage (A / B / C / D / E / F / G / H / I).

---

## Phase 1.A — Canonical label / metric semantics audit

(Entries appended as work proceeds.)


## Phase 1.A — Canonical label / metric semantics audit
- **Created:** src/scripts/audit_canonical_label_semantics.py, tests/test_canonical_label_semantics.py, experiments/audit/canonical_label_semantics.json, experiments/audit/polarity_diagnostic_log.csv, docs/research/audit/CANONICAL_LABEL_METRIC_AUDIT.md
- **Verdict:** METRICS_VALID_BUT_MISINTERPRETED. No code/data bug; canonical 0.7835 values are the canonical test-fold prevalence reflected through degenerate constant predictors. No re-runs.

## Phase 1.B — Remove RGA+ test-set oracle selection
- **Created:** src/scripts/emit_rga_plus_validation_frozen_selection.py, experiments/audit/rga_plus_validation_frozen_selection.csv, tests/test_no_test_selected_rga_plus.py, tests/test_validation_frozen_rga_plus.py
- **Modified:** src/scripts/emit_milestone2_cross_benchmark.py (rewritten: reads from val-frozen artifacts, no test-max)
- **Headline numerical changes (val-frozen vs prior test-max):** MVTec 3D SP 0.739→0.739 (router still wins); MVTec LOCO SP 0.734 (boost test-max) → 0.718 (router val-frozen); Real3D 0.566 → 0.534 (router val-frozen); VisA SP 0.866→0.866; UNSW 0.989→0.989.

## Phase 1.C — Repair comparator policy
- **Created:** src/scripts/select_audited_validation_frozen_comparator.py, experiments/audit/audited_comparator_selection.csv, tests/test_no_test_selected_comparator.py, tests/test_comparator_selection_from_validation_only.py
- **No pre-declaration for existing cells** (Phase 0.6 AR-11). Pre-declared comparator registry reserved for Family D only.

## Phase 1.D — Repair statistical inference + multiplicity
- **Created:** src/scripts/emit_locked_audited_statistics.py, experiments/audit/statistical_family_registry.csv, experiments/audit/audited_ensemble_inference_results.csv, experiments/audit/descriptive_seed_variability.csv, docs/research/audit/AUDITED_INFERENCE_REPORT.md, tests/test_no_fisher_seed_combination.py, tests/test_analysis_family_partition.py, tests/test_holm_family_size_matches_registry.py, tests/test_ensemble_inference_label.py, tests/test_no_retroactive_confirmatory_language.py
- **Fisher combination removed.** Single-representative-seed DeLong used per Family A audited-primary cell; Family A Holm K=5.
- **Audited reanalysis headline finding:** under Phase 1 corrections, only UNSW SP retains Holm-corrected significance in Family A (Δ=+0.0003, p_Holm=6.7e-6). MVTec LOCO SP and Real3D flip sign. Per Rule 2 these lower numbers stand.

## Phase 1.E — Metrics manifest + claim validator
- **Created:** docs/research/metrics_manifest.json (11 claims, 77 macros), docs/research/claim_matrix.csv, docs/research/generated/elara_verified_metrics_macros.tex, src/scripts/build_metrics_manifest.py, src/scripts/validate_manuscript_claims.py, tests/test_metrics_manifest_integrity.py, tests/test_manuscript_claim_consistency.py

## Phase 1.F — Polarity policy (LOCKED: no flip in primary path)
- **Modified:** src/scripts/run_breakthrough_experiment.py (removed flip; added Phase 1.F lock comment)
- **Created:** docs/research/audit/POLARITY_DIAGNOSTIC_REPORT.md, tests/test_primary_metrics_do_not_apply_polarity_flip.py, tests/test_polarity_probe_diagnostic_only.py

## Phase 1.G — Manuscript and table repair
- **Modified:** docs/research/PAPER_DRAFT_v1.tex (abstract; §sec:cross-benchmark-master rewritten; §sec:causal-attribution renamed to Model-Response Sensitivity; SOTA-demarcation section renamed; 5 orphan bibitems removed)
- **Modified:** docs/research/THESIS_CHAPTER_v1.tex (sota-demarcation section + table caption renamed; adapt rate prose updated; calibration monitor wording)
- **Modified:** src/scripts/emit_gradient_adversarial_table.py (FGSM/PGD multicolumn header)
- **Modified:** src/scripts/emit_rga_plus_ablation.py, src/scripts/audit_switching_certificate_t5.py (Real3D label PCA shape + depth)
- **Regenerated:** docs/research/tables/milestone2_cross_benchmark.tex, docs/research/tables/gradient_adversarial.tex, docs/research/tables/rga_plus_ablation.tex, docs/research/tables/switching_certificate_t5.tex

## Phase 1.H — Regenerate PDFs + forbidden-string scan
- **Rebuilt:** output/pdf/PAPER_DRAFT_v1.pdf (35 pp), output/pdf/THESIS_CHAPTER_v1.pdf (39 pp)
- **0 LaTeX errors. Bibliography 187/187 paper, 21/21 thesis. 0 forbidden-string hits.**
- **Test suite: 363 passed, 2 skipped.**
