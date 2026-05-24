# Phase 2.2A — Changelog

## New code

- `src/scripts/run_phase2_family_a_cell.py` — registry-driven Family-A cell driver. Reads benchmark / protocol / pairing-strength / config from `PHASE_2_EXPERIMENT_REGISTRY_v2.csv`. Refuses non-A-POWERED-* IDs, refuses overrides of the locked comparator, refuses to write to historical A-POWERED-1 output paths.
- `src/scripts/run_phase2_family_a_analysis.py` — produces the `PRIMARY_FAMILY_A_CELL_LEVEL_STATIC_REFERENCE_AUDIT` surface. Compares RGA+ (validation-frozen head) only against `static_attention`. Applies K = 5 Holm only when all 5 cells are present; otherwise emits per-cell rows with `holm_p = pending_full_family`.

## New tests (8 files)

| File | Count | Purpose |
|---|---|---|
| `tests/test_phase2_family_a_driver_registry.py` | 7 | rejects non-Family-A IDs, accepts every locked A-POWERED-N |
| `tests/test_phase2_family_a_static_reference_policy.py` | 4 | the analysis driver compares only against `static_attention`; K = 5 Holm; v2 output paths separate from historical |
| `tests/test_phase2_family_a_output_separation.py` | 3 | historical vs v2 path separation enforced |
| `tests/test_phase2_family_a_k5_primary_surface.py` | 4 | primary CSV row label correct; K = 5 vs PARTIAL handled cleanly |
| `tests/test_phase2_family_a_no_competitive_superiority_claim.py` | 4 | the v2 report contains no competitive-superiority phrasing |
| `tests/test_phase2_family_a_prediction_archive_complete.py` | 9 (parametric, mostly skip-while-pending) | archives have required methods + sample-ID alignment + no test-leakage |
| `tests/test_phase2_family_a_historical_pilot_unchanged.py` | 3 | historical pilot CSVs and archive directory remain intact |
| `tests/test_phase2_family_d_untouched_during_family_a.py` | 5 | no Family-D file modified or created during Phase 2.2A |

## New documentation

- `docs/research/phase2/FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md`
- `docs/research/phase2/PHASE_2_2A_CHANGELOG.md` (this file)
- `docs/research/phase2/PHASE_2_2A_REPRODUCTION_COMMANDS.md`
- `docs/research/phase2/PHASE_2_2A_ARTIFACT_MANIFEST.md`
- `docs/research/phase2/PHASE_2_2A_HOSTILE_REVIEW_REPORT.md`
- `docs/research/phase2/PHASE_2_2A_REMAINING_GAPS.md`

## New data outputs

- `experiments/phase2/statistics/family_a_v2_primary_cell_level_raw.csv` — one row per completed cell with full descriptive + inferential statistics
- `experiments/phase2/statistics/family_a_v2_primary_cell_level_holm_k5.csv` — Holm correction across K = 5 cells (`K5_FULL_FAMILY` once complete; `PARTIAL_FAMILY` with `pending_full_family` placeholder until then)
- `experiments/phase2/statistics/family_a_v2_A-POWERED-N_seed_metrics.csv` — per-cell per-seed metrics
- `experiments/phase2/statistics/family_a_v2_A-POWERED-N_selection_log.csv` — per-cell selection log with `selection_used_test_metrics=False`

## Things explicitly NOT touched

- The historical K = 10 secondary pilot files (`family_a_powered_ensemble_inference.csv`, `family_a_powered_holm_results.csv`, `family_a_powered_seed_metrics.csv`, `family_a_selection_log.csv`) remain byte-identical.
- The original A-POWERED-1 prediction archive at `experiments/phase2/predictions/A-POWERED-1__MVTec_3D-AD__PatchCore_supervised-paired/` is unchanged.
- All Family-D v1 files are unchanged.
- No Family-D v2 artefact was created.
- No paper / thesis edits.
- No Family-B / RGA-v2 / KS / certificate / Family-D / Phase-3 / ELARA-Universal / ORIUS work.
