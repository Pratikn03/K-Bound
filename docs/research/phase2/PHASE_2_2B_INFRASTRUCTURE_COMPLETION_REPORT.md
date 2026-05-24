# Phase 2.2B — Infrastructure Completion Report

**Status:** infrastructure for all five Family-B cells is in place; no Family-B compute was executed. The repository is ready for a follow-up Phase 2.2B.exec task that runs the actual model evaluations.

## Scope honoured

This task implemented only the missing infrastructure identified by [PHASE_2_2B_INFRASTRUCTURE_AUDIT.md](./PHASE_2_2B_INFRASTRUCTURE_AUDIT.md). No B-MECH-1..4 or B-CERT-1 result rows were produced. No Family-D file was touched. No paper / thesis edits. No Phase 3 / ELARA-Universal / ORIUS work.

## New model code

| Path | Purpose |
|---|---|
| [src/uais/fusion/attention/reliability_estimator.py](../../../src/uais/fusion/attention/reliability_estimator.py) | added **G3 top-q gate** to `ReliabilityEstimator.gate_decisions()` with locked `_VALID_GATE_MODES = {mean, minimum, hybrid, top_q}`; added `top_q`, `top_q_threshold`, `ks_window_size` init parameters with save/load round-trip; added KS-window truncation in `compute_reliability_weights()` for B-MECH-4 |
| [src/elara/family_b/corruption.py](../../../src/elara/family_b/corruption.py) | `inject_corruption()` re-exports the runner's existing perturbation engine in a typed, split-agnostic API; `validation_fold_corruption_grid()` produces the grid of validation-fold corrupted tensors for RGA-v2 threshold selection (refuses to take test-fold inputs by signature) |
| [src/elara/family_b/mixture_shift.py](../../../src/elara/family_b/mixture_shift.py) | `pure_mixture_shift_resample()` produces row-index resamples that vary category proportions while holding within-category score distributions invariant (largest-remainder quota allocation; optional empirical invariance check via KS) |
| [src/elara/family_b/ks_window.py](../../../src/elara/family_b/ks_window.py) | locked `KS_WINDOW_GRID = (32, 64, 128, 256, 512)` |

## New drivers (registry-driven, refuse non-B IDs)

| Path | Cell | Refuses |
|---|---|---|
| [src/scripts/run_phase2_mechanism_replication.py](../../../src/scripts/run_phase2_mechanism_replication.py) | B-MECH-1 | non-B-MECH-1 ids; non-family-B rows; tau ≠ 0.66; gate ≠ mean; k ≠ (4,); attacks not in {zero, max} |
| [src/scripts/run_phase2_rga_v2_gate_sweep.py](../../../src/scripts/run_phase2_rga_v2_gate_sweep.py) | B-MECH-2 | non-B-MECH-2 ids; gates not in {G0,G1,G2,G3,G4}; G4 (not implemented); any test-fold reads in selection (`_select_tau_on_validation_only()` signature accepts only val tensors) |
| [src/scripts/run_phase2_mixture_shift.py](../../../src/scripts/run_phase2_mixture_shift.py) | B-MECH-3 | non-B-MECH-3 ids; mixture resamples that violate within-category KS invariance |
| [src/scripts/run_phase2_ks_power_sweep.py](../../../src/scripts/run_phase2_ks_power_sweep.py) | B-MECH-4 | non-B-MECH-4 ids; window sizes not in the locked grid |
| [src/scripts/run_phase2_certificate_audit.py](../../../src/scripts/run_phase2_certificate_audit.py) | B-CERT-1 | non-B-CERT-1 ids; archives missing per-sample gate_fired vectors |

All five drivers honour the universal Phase-2 invariants: validation-only selection, `selection_used_test_metrics=False` on every archived row, registry-driven cell identity, no overrides of locked thresholds.

## New tests (9 files, 58 cases)

| File | Cases | Guards |
|---|---:|---|
| [tests/test_phase2_family_b_runner_registry.py](../../../tests/test_phase2_family_b_runner_registry.py) | 25 | every driver rejects every non-matching id; every driver accepts its locked id |
| [tests/test_phase2_family_b_primary_endpoint_lock.py](../../../tests/test_phase2_family_b_primary_endpoint_lock.py) | 5 | B-MECH-1 tau / gate / k / attacks / selection flag are locked in source |
| [tests/test_phase2_family_b_prediction_archive_complete.py](../../../tests/test_phase2_family_b_prediction_archive_complete.py) | 1 (parametric over present archives) | archived rows carry failure_type / failed_domain_count / fault_severity / gate_mode / selection_used_test_metrics=False |
| [tests/test_phase2_rga_v2_contract_lock.py](../../../tests/test_phase2_rga_v2_contract_lock.py) | 6 | G0..G4 present; tau locked at 0.66; clean false-fire budget rule + non-overrideable; C1..C6 present; G3 search grid locked; G3 implemented in estimator |
| [tests/test_phase2_rga_v2_no_test_tuning.py](../../../tests/test_phase2_rga_v2_no_test_tuning.py) | 3 | `_select_tau_on_validation_only` signature accepts only val tensors; `validation_fold_corruption_grid` likewise; driver stamps `selection_used_test_metrics=False` |
| [tests/test_phase2_ks_protocol_lock.py](../../../tests/test_phase2_ks_protocol_lock.py) | 5 | KS_WINDOW_GRID = (32,64,128,256,512); estimator accepts ks_window_size; default is None; mixture-shift doesn't inject corruption; invariance check works |
| [tests/test_phase2_certificate_boundary.py](../../../tests/test_phase2_certificate_boundary.py) | 3 | "retrospective" boundary in source; no production-safety phrases; certificate code importable |
| [tests/test_phase2_family_d_untouched_during_family_b.py](../../../tests/test_phase2_family_d_untouched_during_family_b.py) | 5 | no Family-B driver imports family_d; no family_b module references family_d; v1 invalidation intact; v2 design intact; no v2 artifact created |
| [tests/test_phase2_family_b_g3_top_q_gate.py](../../../tests/test_phase2_family_b_g3_top_q_gate.py) | 5 | q=1 ≡ minimum gate; q=2 doesn't fire on single weak domain; per-sample masks honoured; q<1 rejected; invalid gate mode rejected |

## Test suite delta

- Before Phase 2.2B: 477 passed / 10 skipped.
- After Phase 2.2B:  **535 passed / 11 skipped** (58 net new tests; 1 skip = archive-presence guard that correctly waits for B-MECH-1 to run).

## What is NOT included (preserved deferral)

- B-MECH-1 model execution (training + archiving 30 seeds).
- B-MECH-2 model execution (G0..G3 sweep × val-fold tau selection × test surface).
- B-MECH-3 model execution (mixture-shift evaluation × global vs category-aware KS).
- B-MECH-4 model execution (KS window-size sweep × power vs false-fire).
- B-CERT-1 model execution (consumes B-MECH-1 / B-MECH-2 archives).

These executions are gated by a future Phase 2.2B.exec task; the infrastructure built here is the precondition.

## Stop boundary respected

- No Family D execution / file modification.
- No paper / thesis edits.
- No Phase 3 / ELARA-Universal / ORIUS work.
- No Family-A regression: full suite still passes 535 / 11.
