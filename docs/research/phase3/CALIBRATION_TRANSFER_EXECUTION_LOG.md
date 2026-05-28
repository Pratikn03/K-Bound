# Calibration-Transfer Execution Log

Status: executed (orchestrator + inference refresh)

## Run metadata

- Protocol ID: PHASE3_CAL_TRANSFER_001
- Runner: src/scripts/run_phase3_calibration_transfer_closure.py
- Manifest: docs/research/phase3/CALIBRATION_TRANSFER_CLOSURE_MANIFEST.json
- Run mode: full_run
- Timestamp (UTC): 2026-05-28T23:03:20.378748+00:00

## Preflight checks

- Calibration provenance constraints loaded: passed
- Test-information exclusion checks passed: passed
- Frozen config hashes recorded: passed

## Cell execution

- D-EYE-1: skipped (already executed in one-time Phase-2 v3 manifest)
- D-EYE-2: skipped (already executed in one-time Phase-2 v3 manifest)
- D-EYE-3: dry-run training/calibration executed (seed=142, validation-only, no test access)

## Inference

- Inference runner executed: yes
- Output report path: docs/research/phase2/FAMILY_D_V3_INFERENCE_REPORT.md

## Notes

Notes:
- This invocation respected one-time execution guards and did not re-run held-out cells.
- Inference outputs were regenerated after fixing DeLong paired-variance implementation.