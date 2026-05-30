# WIN vs SAR validation (ELARA deploy v1)

Development path to improve Gate D/E vs frozen SAR without burning confirmatory test runs.

## Policy

Locked in `research_lock/ELARA_DEPLOY_v1.yaml`:

- **When** gate decision rule allows switching (coherent batch + switching certificate): per-sample fires use the CRAF reliability path; others use SAR.
- **When** switching is blocked: entire batch uses SAR (not static attention).

## Commands

From repo root (`AutoML_Flagship_V8`):

```bash
# M1 validation sweep (3 seeds by default)
PYTHONPATH=.:src .venv/bin/python src/scripts/scenario_c/run_win_vs_sar_validation.py --preset m1

# M2 external (3D-ADAM sealed CSV) validation sweep
PYTHONPATH=.:src .venv/bin/python src/scripts/scenario_c/run_win_vs_sar_validation.py --preset m2

# Phase-1 diagnosis (per-category, GDR suppress rate, SAR TTA ablation)
PYTHONPATH=.:src .venv/bin/python src/scripts/scenario_c/run_win_vs_sar_diagnosis.py --preset m2 --seeds 42

# Optional: read-only diagnosis from confirmation archive (test split)
PYTHONPATH=.:src .venv/bin/python src/scripts/scenario_c/run_win_vs_sar_diagnosis.py \
  --archive-index elara_master_c/predictions/confirmation/PREDICTION_ARCHIVE_INDEX.csv \
  --archive-experiment-id M2-EXTERNAL-3D-ADAM --archive-split test --seeds 42 43
```

Outputs:

- `elara_master_c/audits/win_vs_sar_validation_<preset>.json`
- `elara_master_c/audits/WIN_VS_SAR_DIAGNOSIS_<preset>.json`

Exit code **0** only when mean validation Δ ROC-AUC (`elara_deploy_v1` − SAR) is **> 0** (stop rule passed).

## Confirmatory one-shot (after validation WIN)

```bash
PYTHONPATH=.:src .venv/bin/python src/scripts/scenario_c/run_win_vs_sar_validation.py \
  --preset m2 --eval-split test --allow-test --seeds 42 43 44 45 46
```

Do not run test scoring until validation stop rule passes.

## Phase 2 — policy sweep (validation only)

Grid over GDR `coherence_min`, `tau`, and routing mode (`gdr_sar_fallback` vs `gdr_val_router_fallback`). The val-router mode picks SAR or RGA+ on the validation fold, then uses that as the GDR-blocked fallback (no test labels).

```bash
PYTHONPATH=.:src .venv/bin/python src/scripts/scenario_c/run_win_vs_sar_policy_sweep.py --preset m2 --seeds 42 43 44
```

Output: `elara_master_c/audits/win_vs_sar_policy_sweep_<preset>.json` with a **leaderboard** and **winner** block.

**M2 winner (locked):** `research_lock/ELARA_DEPLOY_v2.yaml` — `gdr_val_router_fallback`, `coherence_min=0.35`, `tau=0.55`, validation Δ vs SAR ≈ **+0.0033** (3 seeds on val; GDR never fired → RGA+ fallback).

## Flagship sprint (method modes)

```bash
PYTHONPATH=.:src .venv/bin/python src/scripts/scenario_c/run_flagship_val_sweep.py --preset m2 --seeds 42 43 44
```

Variants: `rga_plus_baseline`, `rga_plus_tta`, `rga_plus_category`, `rga_plus_tta_category`, `elara_deploy_v3` (SAR on unknown categories).

Output: `elara_master_c/audits/flagship_val_sweep_m2_validation.json`  
Stop rule: mean val Δ vs SAR ≥ **0.01** before confirmatory test.

## Phase 3 — confirmatory one-shot (M2 test)

Only after M2 validation stop rule passes with v2:

```bash
PYTHONPATH=.:src .venv/bin/python src/scripts/scenario_c/run_win_vs_sar_validation.py \
  --preset m2 --deploy-lock research_lock/ELARA_DEPLOY_v2.yaml \
  --eval-split test --allow-test --seeds 42 43 44 45 46
```

## Code map

| Module | Role |
|--------|------|
| `src/uais/fusion/attention/elara_deploy_policy.py` | GDR + SAR routing |
| `src/scripts/scenario_c/win_vs_sar_harness.py` | Train/eval one seed |
| `src/scripts/scenario_c/run_win_vs_sar_validation.py` | Multi-seed validation runner |
| `src/scripts/scenario_c/run_win_vs_sar_diagnosis.py` | Per-category + TTA ablation |
| `src/scripts/scenario_c/run_win_vs_sar_policy_sweep.py` | Phase 2 grid search on validation |
