# Phase 2.1 — Changelog

## New files

### Reports / contracts
- `docs/research/phase2/PHASE_2_1_PILOT_PRESERVATION_REPORT.md`
- `docs/research/phase2/PHASE_2_1_FAMILY_A_POLICY_RECONCILIATION.md`
- `docs/research/phase2/PHASE_2_1_REGISTRY_RECONCILIATION.md`
- `docs/research/phase2/PHASE_2_RESEARCH_CONTRACT_v2.md`
- `docs/research/phase2/PHASE_2_STATISTICAL_POLICY_v2.md`
- `docs/research/phase2/FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT_v2.md`
- `docs/research/phase2/PHASE_2_INTERIM_REPORT_v2.md`
- `docs/research/phase2/FAMILY_D_V1_INVALIDATION_NOTICE.md`
- `docs/research/phase2/FAMILY_D_V2_DATASET_ELIGIBILITY_REVIEW.md`
- `docs/research/phase2/FAMILY_D_V2_DESIGN_STATUS.md`
- `docs/research/phase2/PHASE_2_1_HOSTILE_REVIEW_REPORT.md`
- `docs/research/phase2/PHASE_2_1_CHANGELOG.md` (this file)
- `docs/research/phase2/PHASE_2_1_REPRODUCTION_COMMANDS.md`

### Registries (v2, machine-written via csv module)
- `docs/research/phase2/PHASE_2_EXPERIMENT_REGISTRY_v2.csv` (16 rows, 21 columns)
- `docs/research/phase2/PHASE_2_CLAIM_MATRIX_v2.csv` (17 rows, 12 columns)

### Code
- `src/scripts/emit_phase2_registries_v2.py` — deterministic writer that round-trip-validates every row's field count

### Tests (8 new files)
- `tests/test_phase2_registry_csv_schema.py` (4 tests)
- `tests/test_phase2_registry_family_alignment.py` (4 tests)
- `tests/test_phase2_report_registry_consistency.py` (4 tests)
- `tests/test_family_d_v1_never_executable.py` (4 tests)
- `tests/test_family_d_v2_no_placeholders_before_freeze.py` (6 tests; 5 skip because v2 is design-pending)
- `tests/test_family_d_no_previously_touched_dataset.py` (3 tests)
- `tests/test_family_d_claim_boundary.py` (2 tests)
- `tests/test_phase2_no_forbidden_claims_in_manuscripts.py` (2 tests)

## Files preserved unchanged (frozen historical record)

- `docs/research/phase2/PHASE_2_RESEARCH_CONTRACT.md` (v1)
- `docs/research/phase2/PHASE_2_STATISTICAL_POLICY.md` (v1)
- `docs/research/phase2/PHASE_2_EXPERIMENT_REGISTRY.csv` (v1, malformed but preserved)
- `docs/research/phase2/PHASE_2_CLAIM_MATRIX.csv` (v1)
- `docs/research/phase2/FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT.md` (v1)
- `docs/research/phase2/PHASE_2_INTERIM_REPORT.md` (v1)
- `docs/research/phase2/FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md` (v1, INVALID_FOR_EXECUTION)
- `docs/research/phase2/FAMILY_D_DATASET_INVENTORY.md` (v1)
- `docs/research/phase2/FAMILY_D_HYPOTHESES.csv` (v1)
- `docs/research/phase2/FAMILY_D_PARTITION_MANIFEST.json` (v1, placeholders preserved)
- `docs/research/phase2/FAMILY_D_SELECTION_AND_STATISTICAL_POLICY.md` (v1)
- `docs/research/phase2/FAMILY_D_EXECUTION_COMMANDS_NOT_RUN.md` (v1)

## Data unchanged

- `experiments/phase2/predictions/*` (entire prediction archive)
- `experiments/phase2/statistics/family_a_powered_*.csv`
- `experiments/phase2/predictions/PREDICTION_ARCHIVE_INDEX.csv`

## Repairs by drift class

| Drift | Repair | Test that prevents regression |
|---|---|---|
| Within-cell K = 10 Holm reported as if it were Family-A K = 5 | `PHASE_2_STATISTICAL_POLICY_v2.md` §3 splits primary / secondary surfaces; report v2 labels K = 10 output as `SECONDARY_ALL_COMPARATOR_PILOT_AUDIT` | `test_v2_report_labels_existing_output_as_secondary_audit` |
| Report described A-POWERED-2..5 as Real3D / EfficientAD | `FAMILY_A_POWERED_AUDITED_REPRODUCTION_REPORT_v2.md` §1 restores registry-locked cells; EfficientAD / Real3D move to Family C as C-EXP-* | `test_family_a_cells_match_locked_identities`, `test_real3d_and_efficientad_not_in_family_a` |
| Family-A primary comparator drifted per cell | v2 policy locks `static_attention` as Family-A primary comparator across all 5 cells | `test_family_a_primary_comparator_is_static_attention_everywhere` |
| Experiment registry CSV malformed (4 rows off-by-N) | Rewritten via `csv.writer` with QUOTE_MINIMAL; round-trip-validated at write time | `test_csv_every_row_has_header_field_count` (parametric across registry + claims) |
| Family-D v1 manifest frozen with placeholders | v1 marked `INVALID_FOR_EXECUTION`; v2 is `V2_DESIGN_PENDING`; future v2 manifest can't ship with placeholders | `test_no_placeholders_in_any_frozen_v2_file`, `test_v2_design_status_file_exists` |
| Family-D v1 listed VisA as untouched while VisA is in Family A | v1 invalidation notice §1.3 explicitly cites this; v2 eligibility review marks VisA `INELIGIBLE_FOR_FAMILY_D` | `test_eligibility_review_explicitly_excludes_visa`, `test_v2_hypotheses_do_not_reference_visa_anywhere` |
| Family-D v1 claimed success removes Family-A audited-reanalysis status | v2 design-status §3 limits the v2 claim boundary explicitly | `test_no_doc_states_family_d_success_removes_family_a_audited_status` |
| Forbidden Phase-2 claims could leak into LaTeX | grep test added | `test_manuscript_contains_no_forbidden_phase2_claim` |

## Test suite delta

- Before Phase 2.1: 406 passed / 2 skipped.
- After Phase 2.1: 431 passed / 7 skipped (5 of the 5 new skips are the placeholder-guard tests against v2 files that are correctly absent while v2 is design-pending).
