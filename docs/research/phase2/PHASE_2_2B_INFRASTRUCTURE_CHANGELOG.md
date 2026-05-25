# Phase 2.2B — Infrastructure Changelog

## Code added

### `src/elara/family_b/` (new package)
- `__init__.py` — package exports
- `corruption.py` — `inject_corruption`, `validation_fold_corruption_grid`, `KOfDCorruptionResult`
- `mixture_shift.py` — `pure_mixture_shift_resample`, `MixtureShiftResample`
- `ks_window.py` — `KS_WINDOW_GRID = (32, 64, 128, 256, 512)`

### `src/uais/fusion/attention/reliability_estimator.py` (modified)
- Added `_VALID_GATE_MODES = {"mean", "minimum", "hybrid", "top_q"}` class constant.
- `ReliabilityEstimator.__init__` accepts new params: `top_q`, `top_q_threshold`, `ks_window_size`.
- `ReliabilityEstimator.gate_decisions` accepts new params: `top_q`, `top_q_threshold`. New code path implements G3 top-q (`fire iff the q-th smallest reliability across present domains < tau_q`). Conservatively does not fire when fewer than q domains are present.
- `ReliabilityEstimator.compute_reliability_weights` truncates both reference and current scores to `ks_window_size` (when set) before running KS — supports B-MECH-4 window-size sweep.
- `save` / `load` round-trip the three new params.

### `src/scripts/` (5 new drivers)
- `run_phase2_mechanism_replication.py` — B-MECH-1
- `run_phase2_rga_v2_gate_sweep.py` — B-MECH-2
- `run_phase2_mixture_shift.py` — B-MECH-3
- `run_phase2_ks_power_sweep.py` — B-MECH-4
- `run_phase2_certificate_audit.py` — B-CERT-1

## Tests added (9 files)

- `tests/test_phase2_family_b_runner_registry.py`
- `tests/test_phase2_family_b_primary_endpoint_lock.py`
- `tests/test_phase2_family_b_prediction_archive_complete.py`
- `tests/test_phase2_rga_v2_contract_lock.py`
- `tests/test_phase2_rga_v2_no_test_tuning.py`
- `tests/test_phase2_ks_protocol_lock.py`
- `tests/test_phase2_certificate_boundary.py`
- `tests/test_phase2_family_d_untouched_during_family_b.py`
- `tests/test_phase2_family_b_g3_top_q_gate.py`

## Documentation added

- `docs/research/phase2/PHASE_2_2B_INFRASTRUCTURE_AUDIT.md` (pre-execution audit)
- `docs/research/phase2/PHASE_2_2B_INFRASTRUCTURE_COMPLETION_REPORT.md`
- `docs/research/phase2/PHASE_2_2B_INFRASTRUCTURE_CHANGELOG.md` (this file)
- `docs/research/phase2/PHASE_2_2B_INFRASTRUCTURE_TEST_REPORT.md`
- `docs/research/phase2/PHASE_2_2B_READY_TO_COMPUTE_CHECKLIST.md`

## Files preserved (no edits)

- All Phase 2.1 contract files.
- All Phase 2.2A Family-A reports + data outputs.
- All Family-D v1 files.
- All Family-D v2 design-pending files.
- The paper / thesis LaTeX sources.
- The historical A-POWERED-1 K=10 secondary-pilot-audit CSVs.

## Test suite delta

- Before: 477 passed / 10 skipped.
- After:  **535 passed / 11 skipped** (+58 cases, +1 correctly-skipped archive-presence guard).
