# Training Critical Fixes (2026-05-28)

## Issues found

| ID | Severity | Issue | Impact |
|----|----------|-------|--------|
| **T-1** | **Critical** | `evaluate_attention_harness._train_model` tracked best val PR-AUC but **never restored** weights before test | Test metrics used **post-patience degraded** weights |
| **T-2** | **Critical** | `train_attention_fusion.py` saved checkpoint on best epoch but **evaluated test without reloading** in-memory state | Same bug as T-1 for standalone training script |
| **T-3** | **High** | `run_breakthrough_experiment._train_model` early-stopped on **val loss** while audit/docs specify **val PR-AUC** | Inconsistent selection criterion across entry points |
| **T-4** | **Medium** | Harness training lacked **gradient clipping** present in breakthrough path | Minor stability gap |
| **T-5** | **Process** | No single training module — three divergent loops | Drift risk on every change |

## Fixes applied

1. **New module** `src/uais/fusion/attention/training_loop.py`
   - `train_attention_model()` — single training loop
   - Default `early_stopping_metric: pr_auc`
   - Default `restore_best_weights: true`
   - Optional `val_loss` metric for legacy reproduction
   - Gradient clip (`grad_clip_norm: 1.0`), LR scheduler on val loss

2. **Wired all entry points** to `train_attention_model`:
   - `run_breakthrough_experiment.py`
   - `train_attention_fusion.py`
   - `evaluate_attention_harness.py`

3. **Config keys** added to `attention_config.yaml`, `attention_real_fusion.yaml`, `attention_mvtec3d_patchcore_supervised_paired.yaml`

4. **Tests** `tests/test_attention_training_loop.py` + existing `test_train_restore_best_weights.py`

## Locked artifact reproduction

Pre-fix breakthrough runs used **val loss** early stopping (implicit default). To reproduce old numbers when re-training:

```yaml
training:
  early_stopping_metric: val_loss
  restore_best_weights: false   # only if matching pre-restore-fix artifacts
```

New runs should use **`pr_auc` + `restore_best_weights: true`** (now the default).

## Not changed (requires new experiments)

- Reliability weights (α, β, γ) and τ=0.66 remain **fixed engineering constants** (no grid search).
- Val fold still used for early stop + reliability fit + RGA+ selection (standard stacking).
- Multi-seed runs still share one fixed split column.

## Re-train checklist

After pulling these fixes, re-run harness/breakthrough for any benchmark whose JSONs must reflect corrected training:

```bash
cd AutoML_Flagship_V8
PYTHONPATH=src .venv/bin/python -m pytest tests/test_attention_training_loop.py tests/test_train_restore_best_weights.py -q
# Then re-run the relevant config harness (example):
# PYTHONPATH=src python src/scripts/run_breakthrough_experiment.py --config configs/attention_real_fusion.yaml
```

`scripts/rebuild_paper.sh` still **does not retrain** — it only regenerates tables from existing JSON.
