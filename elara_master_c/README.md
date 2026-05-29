# ELARA Master Scenario C — Training Workspace

Operational home for the **T0–T12** training program. Governance contracts live in
`research_lock/` (immutable `_v1` files + superseding `_v2` where ratified).

## Quick start

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8

# Run all automatable checklist steps (infra + training)
PYTHONPATH=src .venv/bin/python src/scripts/scenario_c/complete_master_c_checklist.py

# Progress report
PYTHONPATH=src .venv/bin/python src/scripts/scenario_c/audit_checklist_progress.py

# T0 — verify governance files and training fixes
PYTHONPATH=src .venv/bin/python src/scripts/scenario_c/validate_master_c_governance.py

# Show what a stage would run (no training)
PYTHONPATH=src .venv/bin/python src/scripts/scenario_c/run_training_stage.py --stage T4 --dry-run

# Run mechanism benchmark prep + fusion (development only)
PYTHONPATH=src .venv/bin/python src/scripts/scenario_c/run_training_stage.py --stage T1 --only elara_bench_la
```

## Directory map

| Path | Purpose |
|------|---------|
| `../research_lock/` | Frozen registries, claim contract, decisions |
| `configs/training_stage_registry.yaml` | Maps T0–T12 → scripts |
| `configs/hyperparameter_search_space_v1.yaml` | Validation-only search grids |
| `../src/scripts/scenario_c/` | Governance validator + stage runner |
| `../docs/research/scenario_c/` | Human checklist + status |
| `data/` | Placeholder for processed splits (populate via prepare scripts) |
| `audits/` | Run manifests and gate reports from stage runner |

## Non-negotiable rule

Do **not** train one giant end-to-end model for all claims. Follow stages in
`configs/training_stage_registry.yaml`.

## Eyecandies (D1)

Policy **B** is ratified: Eyecandies is **development-only**. Final transfer (P4)
requires a **new untouched** RGB+depth dataset (`m2_new_untouched_transfer`).

## Transfer pipeline v1 (gate decision + frozen calibrators)

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
PYTHONPATH=src .venv/bin/python src/scripts/scenario_c/run_transfer_development_pipeline.py --dry-run
PYTHONPATH=src .venv/bin/python src/scripts/scenario_c/run_transfer_development_pipeline.py
```

Configs: `configs/attention_eyecandies_transfer_dev_v1.yaml`,
`configs/attention_m2_external_3d_adam_transfer_v1.yaml`.
