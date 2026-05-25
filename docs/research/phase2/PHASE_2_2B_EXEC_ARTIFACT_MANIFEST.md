# Phase 2.2B.exec — Artifact Manifest

## Pre-execution commit
`2719d8111405a4fcc75e288678cd5a18d37134c5` (Phase 2.2B infrastructure)

## Protocol-lock commit
`204775b` (MIXTURE_SHIFT_PROTOCOL + PHASE_2_2B_EXECUTION_PRECHECK)

## Code

| Path | Action |
|---|---|
| `src/scripts/run_phase2_mechanism_replication.py` | unchanged (drove B-MECH-1) |
| `src/scripts/run_phase2_b_mech_1_inference.py` | NEW |
| `src/scripts/run_phase2_certificate_audit.py` | MODIFIED (real pairing logic) |
| `src/scripts/run_phase2_rga_v2_gate_sweep.py` | unchanged (scaffold) |
| `src/scripts/run_phase2_mixture_shift.py` | unchanged (scaffold) |
| `src/scripts/run_phase2_ks_power_sweep.py` | unchanged (scaffold) |

## Data produced

| Path | Rows | Description |
|---|---:|---|
| `experiments/phase2/mechanism/b_mech_1_prediction_archives/B-MECH-1__.../**/*.parquet` | 120 files | 30 seeds × 2 scenarios × 2 methods |
| `experiments/phase2/mechanism/family_b_primary_replication_seed_metrics.csv` | 60 | per-seed per-scenario descriptive |
| `experiments/phase2/mechanism/family_b_primary_replication_inference.csv` | 2 | seed-ensemble inference + Holm K=2 + replication decision |
| `experiments/phase2/mechanism/family_b_primary_replication_holm_k2.csv` | 2 | Holm summary |
| `experiments/phase2/certification/risk_dominance_terms.csv` | 2 | inadmissibility note per scenario |
| `experiments/phase2/certification/switching_certificates.csv` | 2 | 1 CERTIFIED + 1 NOT_CERTIFIED |

## Documents produced (11)

- `PHASE_2_2B_EXECUTION_PRECHECK.md`
- `MIXTURE_SHIFT_PROTOCOL.md`
- `FAMILY_B_PRIMARY_MECHANISM_REPLICATION_REPORT_v2.md`
- `RGA_V2_PARTIAL_FAILURE_REPORT_v2.md`
- `DOMAIN_COMPOSITION_SHIFT_AUDIT_REPORT.md`
- `KS_REFERENCE_AND_POWER_REPORT_v2.md`
- `RISK_DOMINANCE_AND_CERTIFICATE_REPORT_v2.md`
- `PHASE_2_2B_EXEC_HOSTILE_REVIEW_REPORT.md`
- `PHASE_2_2B_EXEC_FINAL_DECISION.md`
- `PHASE_2_2B_EXEC_REMAINING_GAPS.md`
- `PHASE_2_2B_EXEC_CHANGELOG.md`
- `PHASE_2_2B_EXEC_REPRODUCTION_COMMANDS.md`
- `PHASE_2_2B_EXEC_ARTIFACT_MANIFEST.md` (this file)

## Files preserved unchanged

- All Phase 2.2A Family-A CSVs and prediction archives.
- All Family-D v1 contract files.
- All Family-D v2 design-pending files.
- The paper / thesis LaTeX sources.
- All Phase 2.1 contract files.

## Test suite

- 535 passed / 11 skipped before and after (no regressions; the prediction-archive-complete test should flip from SKIP to PASS automatically once B-MECH-1 archives exist).
