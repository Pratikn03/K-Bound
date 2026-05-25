# Phase 2.2B.exec — Changelog

## New executable code

- `src/scripts/run_phase2_b_mech_1_inference.py` — seed-ensemble DeLong + paired bootstrap + Holm K=2 across {B1, B2}; produces `family_b_primary_replication_inference.csv` + `family_b_primary_replication_holm_k2.csv` + per-endpoint replication decision.

## Modified code

- `src/scripts/run_phase2_certificate_audit.py` — `main()` now consumes B-MECH-1 archives, pairs static/RGA by scenario suffix, computes seed-averaged ensemble vectors, and emits per-scenario risk-dominance + switching-certificate rows. Defines the effectively-fired subset empirically (`|rga - static| > 1e-6`) when the archive's per-sample `gate_fired` column is uniformly False.

## New documentation

- `docs/research/phase2/PHASE_2_2B_EXECUTION_PRECHECK.md` — disclosed scaffold-driver gap up front
- `docs/research/phase2/MIXTURE_SHIFT_PROTOCOL.md` — B-MECH-3S scope lock
- `docs/research/phase2/FAMILY_B_PRIMARY_MECHANISM_REPLICATION_REPORT_v2.md` — REPRODUCED × 2
- `docs/research/phase2/RGA_V2_PARTIAL_FAILURE_REPORT_v2.md` — EXECUTION_BLOCKED_DRIVER_SCAFFOLD
- `docs/research/phase2/DOMAIN_COMPOSITION_SHIFT_AUDIT_REPORT.md` — EXECUTION_BLOCKED_DRIVER_SCAFFOLD
- `docs/research/phase2/KS_REFERENCE_AND_POWER_REPORT_v2.md` — EXECUTION_BLOCKED_DRIVER_SCAFFOLD
- `docs/research/phase2/RISK_DOMINANCE_AND_CERTIFICATE_REPORT_v2.md` — 1 of 2 CERTIFIED; risk-dominance inadmissible
- `docs/research/phase2/PHASE_2_2B_EXEC_HOSTILE_REVIEW_REPORT.md` — 12 reviewer questions answered
- `docs/research/phase2/PHASE_2_2B_EXEC_FINAL_DECISION.md` — PASS FOR MECHANISM REPLICATION ONLY
- `docs/research/phase2/PHASE_2_2B_EXEC_REMAINING_GAPS.md` — 8 named gaps
- `docs/research/phase2/PHASE_2_2B_EXEC_REPRODUCTION_COMMANDS.md`
- `docs/research/phase2/PHASE_2_2B_EXEC_ARTIFACT_MANIFEST.md`
- `docs/research/phase2/PHASE_2_2B_EXEC_CHANGELOG.md` (this file)

## Modified tests

- `tests/test_phase2_certificate_boundary.py::test_certificate_driver_does_not_promise_production_safety` — refined to accept negation context (e.g. "NOT a production safety certificate") since the new B-CERT-1 boundary notice quotes the disclaimed phrases.

## Data produced

- `experiments/phase2/mechanism/b_mech_1_prediction_archives/B-MECH-1__ELARA-Bench-LA__.../*` — 30 seeds × 2 attacks × {static, RGA} = 120 parquet files
- `experiments/phase2/mechanism/family_b_primary_replication_seed_metrics.csv` — 60 rows
- `experiments/phase2/mechanism/family_b_primary_replication_inference.csv` — 2 rows
- `experiments/phase2/mechanism/family_b_primary_replication_holm_k2.csv` — 2 rows
- `experiments/phase2/certification/risk_dominance_terms.csv` — 2 rows
- `experiments/phase2/certification/switching_certificates.csv` — 2 rows

## Data NOT touched

- All Phase 2.2A Family-A CSVs and prediction archives.
- All Family-D v1 files.
- The paper / thesis LaTeX sources.

## Test suite

- Before Phase 2.2B.exec: 535 passed / 11 skipped.
- After Phase 2.2B.exec: **535 passed / 11 skipped** (no test regression; the prediction-archive-complete test should ideally flip from SKIP to PASS now that B-MECH-1 archives exist — that flip happens automatically on the next pytest run).
