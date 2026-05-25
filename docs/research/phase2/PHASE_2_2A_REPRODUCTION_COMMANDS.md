# Phase 2.2A — Reproduction Commands

All commands assume the project venv is active and the working directory is the repo root.

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
source .venv/bin/activate
```

## A. Verify Phase 2.1 prerequisites

```bash
PYTHONPATH=src .venv/bin/python src/scripts/emit_phase2_registries_v2.py
PYTHONPATH=src .venv/bin/python -m pytest tests/ --no-header --tb=no -p no:warnings | tail -3
#   expected: ≥ 431 passed (Phase 2.1 baseline) ; 7+ skipped is fine
```

## B. Recompute the A-POWERED-1 primary surface from the existing archive

```bash
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_family_a_analysis.py
#   writes experiments/phase2/statistics/family_a_v2_primary_cell_level_raw.csv
#   writes experiments/phase2/statistics/family_a_v2_primary_cell_level_holm_k5.csv
#     - PARTIAL until all 5 cells exist
```

This is a CPU recompute against the existing prediction archive; it does **not** retrain any model.

## C. Run each new Family-A cell — 30 seeds, registry-driven

Each command takes ~6–60 min depending on benchmark size. The driver reads benchmark / protocol / pairing-strength / config from `PHASE_2_EXPERIMENT_REGISTRY_v2.csv`. Output paths are separate from the historical A-POWERED-1 K = 10 files.

```bash
# A-POWERED-2 — MVTec 3D-AD PatchCore held-out category
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_family_a_cell.py \
    --experiment-id A-POWERED-2 --seeds 30 --seed-start 42

# A-POWERED-3 — MVTec LOCO-AD PatchCore supervised-paired
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_family_a_cell.py \
    --experiment-id A-POWERED-3 --seeds 30 --seed-start 42

# A-POWERED-4 — VisA RGB+edge supervised-paired (derived_view_proxy)
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_family_a_cell.py \
    --experiment-id A-POWERED-4 --seeds 30 --seed-start 42

# A-POWERED-5 — UNSW-NB15 flow/conn/context (naturally_structured_views; large file)
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_family_a_cell.py \
    --experiment-id A-POWERED-5 --seeds 30 --seed-start 42
```

## D. After all 5 cells exist, compute the K = 5 Holm correction

```bash
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_family_a_analysis.py
#   automatically applies K = 5 Holm once all 5 cells are present
```

The `family_a_v2_primary_cell_level_holm_k5.csv` row's `holm_status`
will flip from `PARTIAL_FAMILY` to `K5_FULL_FAMILY` and the
`delong_p_holm_k5` cells will fill with numeric values.

## E. Run the full Phase 2.2A test suite

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
    tests/test_phase2_family_a_driver_registry.py \
    tests/test_phase2_family_a_static_reference_policy.py \
    tests/test_phase2_family_a_output_separation.py \
    tests/test_phase2_family_a_k5_primary_surface.py \
    tests/test_phase2_family_a_no_competitive_superiority_claim.py \
    tests/test_phase2_family_a_prediction_archive_complete.py \
    tests/test_phase2_family_a_historical_pilot_unchanged.py \
    tests/test_phase2_family_d_untouched_during_family_a.py \
    --no-header --tb=short -q
```

## F. Full suite

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/ --no-header --tb=no -p no:warnings | tail -3
```

## G. Status snapshot expected after this task

If A-POWERED-2..5 have completed and the K = 5 Holm has been applied:

- 5 cells in `family_a_v2_primary_cell_level_raw.csv` (one per A-POWERED-N).
- 5 cells in `family_a_v2_primary_cell_level_holm_k5.csv` with `holm_status = K5_FULL_FAMILY` and numeric `delong_p_holm_k5`.
- A-POWERED-1 historical K = 10 outputs remain byte-identical.
- All v1 Family-D files remain unchanged; no v2 Family-D artefact created.
- Full pytest suite: ≥ 439 passed, > 7 skipped (the new QC tests start passing once their target CSVs land).

## H. Things this command set deliberately does NOT do

- Run any Family-B cell (B-MECH-1..4, B-CERT-1).
- Run RGA-v2 gate sweep.
- Run KS power / mixture-shift sweep.
- Run any Family-D experiment.
- Edit paper / thesis claims.
- Begin Phase 3, ELARA-Universal, ORIUS.
