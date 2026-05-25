# Phase 2.2A — Artifact Manifest

## Code added

| Path | Lines | Purpose |
|---|---:|---|
| `src/scripts/run_phase2_family_a_cell.py` | ~210 | registry-driven Family-A cell driver |
| `src/scripts/run_phase2_family_a_analysis.py` | ~295 | static-reference primary surface + K = 5 Holm |

Both files import the existing pilot's `run_one_seed()` to keep the training path identical to the validated A-POWERED-1 path.

## Tests added (8 files)

| Path | Lines | What it guards |
|---|---:|---|
| `tests/test_phase2_family_a_driver_registry.py` | ~50 | non-A-POWERED-* IDs are rejected; every locked A-POWERED-N is accepted |
| `tests/test_phase2_family_a_static_reference_policy.py` | ~60 | the analysis driver compares only against `static_attention`; K = 5 Holm; v2 paths separate from historical |
| `tests/test_phase2_family_a_output_separation.py` | ~40 | historical vs v2 file separation |
| `tests/test_phase2_family_a_k5_primary_surface.py` | ~55 | primary CSV row schema; K = 5 vs PARTIAL handled cleanly |
| `tests/test_phase2_family_a_no_competitive_superiority_claim.py` | ~60 | v2 report contains no competitive-superiority phrasing |
| `tests/test_phase2_family_a_prediction_archive_complete.py` | ~80 | required methods + sample-ID alignment + no test-leakage per cell |
| `tests/test_phase2_family_a_historical_pilot_unchanged.py` | ~55 | historical pilot CSVs and archive directory remain intact |
| `tests/test_phase2_family_d_untouched_during_family_a.py` | ~70 | no Family-D file modified or created during Phase 2.2A |

## Documentation added

- `docs/research/phase2/FAMILY_A_V2_STATIC_REFERENCE_AUDIT_REPORT.md`
- `docs/research/phase2/PHASE_2_2A_CHANGELOG.md`
- `docs/research/phase2/PHASE_2_2A_REPRODUCTION_COMMANDS.md`
- `docs/research/phase2/PHASE_2_2A_ARTIFACT_MANIFEST.md` (this file)
- `docs/research/phase2/PHASE_2_2A_HOSTILE_REVIEW_REPORT.md`
- `docs/research/phase2/PHASE_2_2A_REMAINING_GAPS.md`

## Data produced

### v2 primary-surface CSVs (new)

- `experiments/phase2/statistics/family_a_v2_primary_cell_level_raw.csv` — one row per completed cell
- `experiments/phase2/statistics/family_a_v2_primary_cell_level_holm_k5.csv` — Holm correction (K = 5 once all cells complete)

### v2 per-cell metrics + selection logs (new)

- `experiments/phase2/statistics/family_a_v2_A-POWERED-N_seed_metrics.csv` per cell
- `experiments/phase2/statistics/family_a_v2_A-POWERED-N_selection_log.csv` per cell

### v2 per-cell prediction archives (new)

- `experiments/phase2/predictions/A-POWERED-N__<bench>__<protocol>/<method>/<split>/seed_NN.parquet`
- The shared archive index `experiments/phase2/predictions/PREDICTION_ARCHIVE_INDEX.csv` grows with one row per (cell × method × split × seed).

## Data preserved (UNCHANGED)

- `experiments/phase2/predictions/A-POWERED-1__MVTec_3D-AD__PatchCore_supervised-paired/` — entire prediction archive
- `experiments/phase2/statistics/family_a_powered_seed_metrics.csv` — historical
- `experiments/phase2/statistics/family_a_selection_log.csv` — historical
- `experiments/phase2/statistics/family_a_powered_ensemble_inference.csv` — historical K = 10 secondary pilot
- `experiments/phase2/statistics/family_a_powered_holm_results.csv` — historical K = 10 secondary pilot
- All `docs/research/phase2/FAMILY_D_*.md` and `FAMILY_D_*.json` and `FAMILY_D_*.csv` (v1, preserved)

## Stop boundary respected

- No Family-B, RGA-v2, KS, certificate, Family-D, Phase-3, ELARA-Universal, ORIUS work.
- Paper / thesis claims not modified based on Phase 2.2A results.
- Phase 2.1 contract invariants intact: `PHASE_2_RESEARCH_CONTRACT_v2.md`, `PHASE_2_STATISTICAL_POLICY_v2.md`, `FAMILY_D_V1_INVALIDATION_NOTICE.md`, `FAMILY_D_V2_DESIGN_STATUS.md` all preserved.
