# Phase 2 — Reproduction Commands

Every command below MUST run from the repo root with the project venv active.

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
source .venv/bin/activate
```

## A. Re-run the 30-seed pilot (A-POWERED-1)

```bash
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_powered_audited_pilot.py \
    --seeds 30 --seed-start 42 2>&1 | tee /tmp/phase2_pilot.log
```

Expected: 30 seeds × 12 methods × 2 splits ≈ 8 640 prediction-archive entries written under [experiments/phase2/predictions/A-POWERED-1__MVTec_3D-AD__PatchCore_supervised-paired/](../../../experiments/phase2/predictions/), plus an updated [experiments/phase2/predictions/PREDICTION_ARCHIVE_INDEX.csv](../../../experiments/phase2/predictions/PREDICTION_ARCHIVE_INDEX.csv).

## B. Re-run the audited inference

```bash
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_powered_audited_analysis.py
```

Expected: refreshes [experiments/phase2/statistics/family_a_powered_ensemble_inference.csv](../../../experiments/phase2/statistics/family_a_powered_ensemble_inference.csv) and [experiments/phase2/statistics/family_a_powered_holm_results.csv](../../../experiments/phase2/statistics/family_a_powered_holm_results.csv). Headline values: ensemble RGA+ AUC = 0.7420; 5 of 10 comparators Holm-significant at α = 0.05.

## C. Validate prediction-archive integrity

```bash
PYTHONPATH=src .venv/bin/python src/scripts/validate_phase2_prediction_archives.py
```

Expected: no SHA256 mismatches, no schema mismatches, no `selection_used_test_metrics=True` rows on test splits.

## D. Run the Phase-2 test suite

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
    tests/test_phase2_prediction_archive_schema.py \
    tests/test_phase2_prediction_archive_no_leakage.py \
    tests/test_phase2_validation_only_selection.py \
    tests/test_phase2_certification.py \
    --no-header --tb=short -q
```

Expected: 15 tests passing (4 + 2 + 5 + 4).

## E. Family D (pre-registered, not run)

See [FAMILY_D_EXECUTION_COMMANDS_NOT_RUN.md](./FAMILY_D_EXECUTION_COMMANDS_NOT_RUN.md) for the verbatim commands and the freeze-integrity contract.

## F. Hashes recorded at session end

The freeze SHA256 of every contract file is intended to be captured at commit time via:

```bash
shasum -a 256 \
    docs/research/phase2/PHASE_2_RESEARCH_CONTRACT.md \
    docs/research/phase2/PHASE_2_STATISTICAL_POLICY.md \
    docs/research/phase2/FAMILY_D_CONFIRMATORY_REPLICATION_CONTRACT.md \
    docs/research/phase2/FAMILY_D_DATASET_INVENTORY.md \
    docs/research/phase2/FAMILY_D_HYPOTHESES.csv \
    docs/research/phase2/FAMILY_D_PARTITION_MANIFEST.json \
    docs/research/phase2/FAMILY_D_SELECTION_AND_STATISTICAL_POLICY.md \
    docs/research/phase2/FAMILY_D_EXECUTION_COMMANDS_NOT_RUN.md \
    configs/phase2/rga_v2_gate_contract.yaml
```

The output of this command, recorded in [PHASE_2_ARTIFACT_MANIFEST.md](./PHASE_2_ARTIFACT_MANIFEST.md), is the integrity anchor for any future Family-D execution.
