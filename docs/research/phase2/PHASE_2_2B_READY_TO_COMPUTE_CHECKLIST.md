# Phase 2.2B — Ready-to-Compute Checklist

This checklist is for a future Phase 2.2B.exec task. Every box must be checked before launching the actual Family-B compute runs.

## Hard prerequisites (verified at end of Phase 2.2B-infrastructure task)

- [x] G3 top-q gate is implemented in `ReliabilityEstimator` and unit-tested.
- [x] G4 learned gate is intentionally **not** implemented; B-MECH-2 driver explicitly rejects `--gates G4`.
- [x] Validation-fold corruption-grid helper exists with signature that prevents test-fold leakage.
- [x] Pure mixture-shift sampler exists with within-category KS invariance check.
- [x] KS window-size parameter is wired into `ReliabilityEstimator`; locked grid is (32, 64, 128, 256, 512).
- [x] All five Family-B drivers exist; each rejects non-matching experiment_ids.
- [x] Prediction-archive schema (Phase-2.B 28-column) supports all required Family-B columns (`failure_type_if_applicable`, `failed_domain_count_if_applicable`, `fault_severity_if_applicable`, `gate_mode`, `gate_fired_if_applicable`, `mean_reliability_if_applicable`).
- [x] 58 new Family-B tests pass; full suite is 535 / 11.
- [x] No Family-D file modified.
- [x] No paper / thesis edit.
- [x] No Family-A regression.

## Pre-execution gate (must be re-verified at the top of Phase 2.2B.exec)

- [ ] Re-run `PYTHONPATH=src .venv/bin/python -m pytest tests/ --no-header --tb=no -p no:warnings` and confirm ≥ 535 / 11.
- [ ] Re-confirm ELARA-Bench-LA data still at `experiments/fusion/real_domain_fusion_inputs.csv` with 28 110 rows and 4 domains.
- [ ] Re-confirm `configs/phase2/rga_v2_gate_contract.yaml` SHA256 matches the freeze-time hash recorded in `PHASE_2_ARTIFACT_MANIFEST.md`.
- [ ] Re-confirm `docs/research/phase2/FAMILY_D_V1_INVALIDATION_NOTICE.md` still says `INVALID_FOR_EXECUTION`.
- [ ] Re-confirm `docs/research/phase2/FAMILY_D_V2_DESIGN_STATUS.md` still says `V2_DESIGN_PENDING`.

## B-MECH-1 readiness

- [ ] `experiments/phase2/mechanism/b_mech_1_prediction_archives/` does not yet exist (no overwrite risk), OR the existing archive will be intentionally appended via `rerun_N` suffix.
- [ ] Wall-clock budget allocated: ~30–60 min for 30 seeds.
- [ ] Run: `PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_mechanism_replication.py --experiment-id B-MECH-1 --seeds 30 --seed-start 42`.
- [ ] Post-run: `test_phase2_family_b_prediction_archive_complete::test_b_mech_1_archive_directory_either_absent_or_well_formed` should switch from SKIP to PASS.

## B-MECH-2 readiness

- [ ] Wall-clock budget allocated: ≥ 4× B-MECH-1 (3 gates × val-fold tau selection × full fault surface).
- [ ] Confirm `--gates` does NOT include G4 (which is intentionally not implemented).
- [ ] Run: `PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_rga_v2_gate_sweep.py --experiment-id B-MECH-2 --seeds 30 --seed-start 42 --gates G0,G1,G2,G3`.

## B-MECH-3 readiness

- [ ] Decide on the category column for ELARA-Bench-LA. The benchmark has a natural `domain` column (4 domains) but no per-sample "category" beyond that. Either:
  - use the domain identity as the mixture-shift category (4-way mixture), or
  - construct a derived category from the existing dataset (e.g., class-label parity bands).
- [ ] Document the chosen category in a `MIXTURE_SHIFT_PROTOCOL.md` before execution.
- [ ] Run: `PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_mixture_shift.py --experiment-id B-MECH-3 --seeds 5 --seed-start 42 --mixture-shifts 10`.

## B-MECH-4 readiness

- [ ] Wall-clock budget allocated: 5 windows × 5 seeds × ELARA-Bench-LA training ≈ 1–2 h.
- [ ] Run: `PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_ks_power_sweep.py --experiment-id B-MECH-4 --seeds 5 --seed-start 42`.

## B-CERT-1 readiness

- [ ] B-MECH-1 archives present at `experiments/phase2/mechanism/b_mech_1_prediction_archives/`.
- [ ] Optionally: B-MECH-2 archives also present, to extend the certificate audit to partial-failure scenarios.
- [ ] Run: `PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_certificate_audit.py --experiment-id B-CERT-1`.

## Out-of-scope reminders for Phase 2.2B.exec

- Do NOT execute Family-D.
- Do NOT modify Family-D v1 or v2 files.
- Do NOT edit paper / thesis based on partial Family-B outputs.
- Do NOT modify Phase 2.2A Family-A reports or CSVs.
- Do NOT redefine B1/B2 endpoints after seeing outcomes.
- Do NOT promote RGA-v2 unless every C1..C6 promotion criterion passes.
- Do NOT begin Phase 3 / ELARA-Universal / ORIUS.
