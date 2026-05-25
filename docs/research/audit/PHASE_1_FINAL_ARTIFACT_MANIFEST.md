# Phase 1 Final Artifact Manifest

**Branch:** `fix/elara-phase1-empirical-validity`
**Phase 1 complete:** all 11 substages (prerequisite gate + 1.A–1.I).

---

## Final PDFs

| Path | Pages | Build log |
|---|---|---|
| `output/pdf/PAPER_DRAFT_v1.pdf` | 35 | `.tex_build/PAPER_DRAFT_v1.log` (0 errors, 0 undefined refs) |
| `output/pdf/THESIS_CHAPTER_v1.pdf` | 39 | `.tex_build_thesis/THESIS_CHAPTER_v1.log` (0 errors, 0 undefined refs) |

## LaTeX sources

- `docs/research/PAPER_DRAFT_v1.tex` — conference paper source. 187/187 bibliography hygiene clean.
- `docs/research/THESIS_CHAPTER_v1.tex` — thesis chapter source. 21/21 bibliography hygiene clean.

## Generated tables (single source of truth = the audit CSVs)

- `docs/research/tables/milestone2_cross_benchmark.tex` — cross-benchmark master comparison (val-frozen RGA+ + val-frozen comparator + Family-A K=5 Holm; no test-set oracle).
- `docs/research/tables/gradient_adversarial.tex` — FGSM/PGD with explicit multicolumn header.
- `docs/research/tables/rga_plus_ablation.tex` — RGA+ component ablation (Real3D label corrected).
- `docs/research/tables/switching_certificate_t5.tex` — T5 audit (Real3D label corrected).

## Generated figures

All existing figures retained; no figure regeneration was required (no figure-bound data changed).

## Audit / policy / evidence artifacts (Phase 0/0.5/0.6/1)

- `docs/research/audit/PHASE_0_6_FINAL_POLICY_LOCK.md` — policy lock.
- `docs/research/audit/PHASE_1_CHANGELOG.md` — every file modified or created in Phase 1.
- `docs/research/audit/PHASE_1_REPRODUCTION_COMMANDS.md` — every command run during Phase 1.
- `docs/research/audit/PHASE_1_FINAL_ARTIFACT_MANIFEST.md` — this file.
- `docs/research/audit/PHASE_1_REMAINING_OPEN_GAPS.md` — work explicitly deferred.
- `docs/research/audit/PHASE_1_HOSTILE_REVIEW_REPORT.md` — read-only hostile review.
- `docs/research/audit/CANONICAL_LABEL_METRIC_AUDIT.md` — Phase 1.A findings.
- `docs/research/audit/AUDITED_INFERENCE_REPORT.md` — Phase 1.D findings.
- `docs/research/audit/POLARITY_DIAGNOSTIC_REPORT.md` — Phase 1.F findings.

## Phase 1 evidence CSVs / JSONs

- `experiments/audit/canonical_label_semantics.json` — Phase 1.A per-cell evidence + verdict.
- `experiments/audit/polarity_diagnostic_log.csv` — per-seed polarity probe log (primary_metrics_use_flip = False everywhere).
- `experiments/audit/rga_plus_validation_frozen_selection.csv` — RGA+ headline selection per cell (166 rows: 11 ensemble + 155 per-seed).
- `experiments/audit/audited_comparator_selection.csv` — primary comparator selection per cell (11 rows).
- `experiments/audit/statistical_family_registry.csv` — family A/B/C partition and Holm K-counts (11 rows).
- `experiments/audit/audited_ensemble_inference_results.csv` — audited inference per cell (11 rows).
- `experiments/audit/descriptive_seed_variability.csv` — per-method seed variability (132 rows).

## Claim locking artifacts

- `docs/research/metrics_manifest.json` — 11 claims, 77 macros, single source of truth.
- `docs/research/claim_matrix.csv` — 14 manuscript edits with before/after and source-artifact pointer.
- `docs/research/generated/elara_verified_metrics_macros.tex` — 77 LaTeX macros (one-time consumption pattern).

## Validator + tests

- `src/scripts/validate_manuscript_claims.py` — forbidden-token scanner; returns 0 violations.
- `src/scripts/build_metrics_manifest.py` — builds the manifest from audit CSVs.

Phase 1 added 14 new test files; the full suite is **363 passed, 2 skipped**.

## Reproduction commands (one-shot)

```bash
# Phase 1.A — canonical semantics audit
PYTHONPATH=src .venv/bin/python src/scripts/audit_canonical_label_semantics.py

# Phase 1.B / 1.C selection artifacts
PYTHONPATH=src .venv/bin/python src/scripts/emit_rga_plus_validation_frozen_selection.py
PYTHONPATH=src .venv/bin/python src/scripts/select_audited_validation_frozen_comparator.py

# Phase 1.D audited statistics
PYTHONPATH=src .venv/bin/python src/scripts/emit_locked_audited_statistics.py

# Phase 1.E manifest + macros
PYTHONPATH=src .venv/bin/python src/scripts/build_metrics_manifest.py

# Phase 1.G regenerate master comparison + adversarial + ablation tables
PYTHONPATH=src .venv/bin/python src/scripts/emit_milestone2_cross_benchmark.py
PYTHONPATH=src .venv/bin/python src/scripts/emit_gradient_adversarial_table.py
PYTHONPATH=src .venv/bin/python src/scripts/emit_rga_plus_ablation.py
PYTHONPATH=src .venv/bin/python src/scripts/audit_switching_certificate_t5.py
PYTHONPATH=src .venv/bin/python src/scripts/emit_switching_certificate_t5_table.py

# Phase 1.H rebuild PDFs
./scripts/rebuild_paper.sh

# Validators
PYTHONPATH=src .venv/bin/python src/scripts/validate_manuscript_claims.py
PYTHONPATH=src:. .venv/bin/python -m pytest tests/ -q
```

## Git commit hash (at Phase 1 completion)

To be recorded by the final completion commit (see PHASE_1_REPRODUCTION_COMMANDS.md).
