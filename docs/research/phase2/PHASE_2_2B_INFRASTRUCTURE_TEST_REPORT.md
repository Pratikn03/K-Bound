# Phase 2.2B — Infrastructure Test Report

**Full test suite:** `pytest tests/ --no-header --tb=no -p no:warnings`

```
535 passed, 11 skipped in 66.48s
```

## Family-B-specific test breakdown

| Test file | Cases | Result |
|---|---:|:---:|
| `tests/test_phase2_family_b_runner_registry.py` | 25 | PASS |
| `tests/test_phase2_family_b_primary_endpoint_lock.py` | 5 | PASS |
| `tests/test_phase2_family_b_prediction_archive_complete.py` | 1 (parametric; 1 skip: archive not yet present) | PASS / SKIP |
| `tests/test_phase2_rga_v2_contract_lock.py` | 6 | PASS |
| `tests/test_phase2_rga_v2_no_test_tuning.py` | 3 | PASS |
| `tests/test_phase2_ks_protocol_lock.py` | 5 | PASS |
| `tests/test_phase2_certificate_boundary.py` | 3 | PASS |
| `tests/test_phase2_family_d_untouched_during_family_b.py` | 5 | PASS |
| `tests/test_phase2_family_b_g3_top_q_gate.py` | 5 | PASS |
| **Total Phase 2.2B new** | **58** | **PASS (1 skip)** |

The single skip is correct behaviour: `test_b_mech_1_archive_directory_either_absent_or_well_formed` skips when the archive directory does not yet exist (it has not been executed in this infrastructure-only phase). It will switch to PASS automatically once a future Phase 2.2B.exec task runs B-MECH-1.

## Hostile-review-style sweeps (manual)

| Check | Result |
|---|:---:|
| Each Family-B driver rejects every non-matching `--experiment-id` (A-POWERED-*, B-MECH-* off-by-one, C-EXP-*, D-CONTRACT-V2) | PASS |
| Every driver accepts its own locked `--experiment-id` with `--seeds 0` / `--dry-run` and exits 0 | PASS |
| B-MECH-2 driver refuses `--gates G4` because G4 (learned gate) is not implemented | PASS |
| B-MECH-4 driver refuses any window size outside {32,64,128,256,512} | PASS |
| G3 top-q gate matches the G1 minimum gate when q=1 | PASS |
| G3 top-q gate does NOT fire on a single weak domain when q=2 | PASS |
| Mixture-shift sampler raises on a `target_proportions` referring to an unknown category | PASS |
| Mixture-shift sampler honours within-category KS invariance (no score corruption) | PASS |
| Validation-fold corruption-grid function signature accepts only val_features/val_masks; rejects test_* params | PASS |
| `_select_tau_on_validation_only` source body contains no `test_features`/`test_masks`/`test_labels` strings | PASS |
| ReliabilityEstimator round-trips `top_q`, `top_q_threshold`, `ks_window_size` through save/load | PASS (by construction; tests cover existence) |
| Family-D v1 invalidation notice still says `INVALID_FOR_EXECUTION` | PASS |
| Family-D v2 design status still says `V2_DESIGN_PENDING` | PASS |
| No v2 Family-D artifact was created during Phase 2.2B | PASS |
| Historical A-POWERED-1 K=10 secondary pilot CSVs unchanged | PASS (verified by `test_historical_csv_still_has_legacy_schema`) |
| `family_a_v2_primary_cell_level_*` files unchanged | PASS (no driver writes to those paths) |

## Smoke command transcript

```
$ PYTHONPATH=src python src/scripts/run_phase2_mechanism_replication.py --experiment-id B-MECH-1 --seeds 0
[b-mech-1 B-MECH-1] validation-only invocation; exiting OK

$ PYTHONPATH=src python src/scripts/run_phase2_rga_v2_gate_sweep.py --experiment-id B-MECH-2 --seeds 0
[b-mech-2 B-MECH-2] validation-only invocation; exiting OK

$ PYTHONPATH=src python src/scripts/run_phase2_mixture_shift.py --experiment-id B-MECH-3 --seeds 0
[b-mech-3 B-MECH-3] validation-only invocation; exiting OK

$ PYTHONPATH=src python src/scripts/run_phase2_ks_power_sweep.py --experiment-id B-MECH-4 --seeds 0
[b-mech-4 B-MECH-4] validation-only invocation; exiting OK

$ PYTHONPATH=src python src/scripts/run_phase2_certificate_audit.py --experiment-id B-CERT-1 --dry-run
[b-cert-1] archive scan: 0 (cell, method) groups found at .../b_mech_1_prediction_archives
[b-cert-1] no archived inputs to process; exiting OK (run B-MECH-1 first)

$ PYTHONPATH=src python src/scripts/run_phase2_mechanism_replication.py --experiment-id A-POWERED-1 --seeds 0
this driver runs B-MECH-1 only; got 'A-POWERED-1'

$ PYTHONPATH=src python src/scripts/run_phase2_rga_v2_gate_sweep.py --experiment-id B-MECH-2 --seeds 1 --gates G4
G4 (learned low-capacity gate) is marked optional in the contract and is NOT implemented in this code base. Re-run without G4 in --gates.
```
