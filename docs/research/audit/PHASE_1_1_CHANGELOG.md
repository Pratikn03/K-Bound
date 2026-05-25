# Phase 1.1 Changelog

**Branch:** `fix/elara-phase1-1-pdf-source-consistency`
**Pre-Phase-1.1 commit:** `5d9cf46`

## Source code created
- `src/scripts/phase1_1_canonical_cleanup.py` — strips canonical PR-AUC / ECE / Brier from promoted tables and figures.
- `src/scripts/validate_phase1_1_pdf_claims.py` — Phase-1.1 deterministic source-PDF consistency validator.

## Source code modified
- `src/scripts/emit_rga_plus_ablation.py` — replace "Best non-router / Best ROC / Router-Best / Boost-Best" column headers with validation-frozen comparator framing; consume `experiments/audit/audited_comparator_selection.csv`.
- `src/scripts/emit_milestone1_comparison.py` — same: validation-frozen comparator + audited delta.
- `scripts/rebuild_paper.sh` — adds Phase 1.1 canonical-cleanup + manifest rebuild steps before LaTeX compile.

## LaTeX source modified
- `docs/research/PAPER_DRAFT_v1.tex` — UNSW section rewritten; Real3D paragraph rewritten; RGA+ component ablation paragraph rewritten; SOTA-demarcation section renamed to "Demarcation from Published…"; orphan SCM-related bibitems removed; "$0.7835$" / "max(router, boost)" / "best non-router" / "strongest non-router" / "Causal Reliability Attribution" / "interventional ATE" / "Structural Causal Model" / "Fisher-combined" / "nine evaluated cells" / "Family A confirmatory" / "deployment-grade" / "SOTA" / "universally superior" / "production-ready" / stale "FPFH+depth" labels all removed.
- `docs/research/THESIS_CHAPTER_v1.tex` — abstract rewritten to reflect mixed audited outcomes + primary B1/B2 deltas; new §sec:thesis-audited-policy subsection added; UNSW section rewritten; EATA+SAR paragraph + bibitems added; section renames; canonical metric reframing.

## Tables / figures regenerated
- `docs/research/tables/milestone2_cross_benchmark.tex` (already corrected by Phase 1)
- `docs/research/tables/rga_plus_ablation.tex` — validation-frozen comparator headers
- `docs/research/tables/mvtec3d_milestone1_comparison.tex` — validation-frozen comparator headers
- `docs/research/tables/mvtec3d_patchcore_clean_ci_results.tex` — canonical ROC-AUC-only
- `docs/research/tables/mvtec3d_patchcore_calibration_cda.tex` — canonical diagnostic-only
- `docs/research/tables/mvtec3d_clean_ci_results.tex` — canonical ROC-AUC-only
- `docs/research/tables/mvtec3d_clean_results.tex` — canonical ROC-AUC-only
- `docs/research/tables/mvtec3d_calibration_cda.tex` — canonical diagnostic-only
- `docs/research/figures/mvtec3d_patchcore_clean_benchmark.png` — canonical ROC-AUC-only bar chart
- `docs/research/figures/mvtec3d_clean_benchmark.png` — canonical ROC-AUC-only bar chart

## Tests created (Phase 1.1)
- `tests/test_phase1_1_pdf_claim_validation.py`
- `tests/test_phase1_1_primary_story_consistency.py`
- `tests/test_one_primary_elara_bench_la_story.py`
- `tests/test_phase1_1_bundle.py` (covers Real3D, canonical, UNSW, master table, Family-A naming, causal language, polarity, audited-policy subsection, thesis abstract, baselines, stale tables)

## Audit reports created (Phase 1.1)
- `docs/research/audit/PHASE_1_1_BASELINE_SNAPSHOT.md`
- `docs/research/audit/PHASE_1_1_CONTRADICTION_LEDGER.csv`
- `docs/research/audit/PHASE_1_1_PRIMARY_RUN_RESOLUTION.md`
- `docs/research/audit/PHASE_1_1_REAL3D_RESOLUTION.md`
- `docs/research/audit/PHASE_1_1_BUILD_AND_HASH_REPORT.md`
- `docs/research/audit/PHASE_1_1_PDF_TEXT_SCAN_REPORT.md`
- `docs/research/audit/PHASE_1_1_VISUAL_PDF_AUDIT.md`
- `docs/research/audit/PHASE_1_1_HOSTILE_REVIEW_REPORT.md`
- `docs/research/audit/PHASE_1_1_REMAINING_OPEN_GAPS.md`
- `docs/research/audit/PHASE_1_1_REPRODUCTION_COMMANDS.md`
- `docs/research/audit/PHASE_1_1_CHANGELOG.md` (this file)

## Final PDFs produced
- `output/pdf/PAPER_DRAFT_v1.pdf` (35 pp, SHA256 `4394833e14d10d70a445371181b4d78a035e4533c5582665f539dfa0e630e3ef`)
- `output/pdf/THESIS_CHAPTER_v1.pdf` (40 pp, SHA256 `77de50a99db57790e9cc13bf64d27cf570306fb63ae0e4e60b1dc2d0e917d585`)
- `output/pdf/PAPER_DRAFT_PHASE1_1_VERIFIED.pdf` (35 pp, matches standard hash)
- `output/pdf/THESIS_CHAPTER_PHASE1_1_VERIFIED.pdf` (40 pp, matches standard hash)
