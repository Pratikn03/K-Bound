# Current paper baseline (frozen reference)

**Purpose:** T0 archive pointer for Master Scenario C. Do not delete prior results.

## Manuscript state

- Draft: `docs/research/PAPER_DRAFT_v1.tex`
- Rebuild tables: `bash scripts/rebuild_paper.sh`
- Experiment registry: `docs/research/audit/EXPERIMENT_REGISTRY.csv`

## Admissible claim (until P1–P6 confirmatory pass)

From `research_lock/BASELINE_STATE_v1.md`:

> Bounded reliability-stress gains under controlled collapse; held-out transfer **not confirmed** on Eyecandies (historical attempt).

## Family D (transfer)

- Record: `research_lock/family_d_failure_record.md`
- Eyecandies: reclassified to **development** per D1 (Policy B); original failure preserved.

## Static attention baseline (locked hyperparameters)

| Parameter | Value |
|-----------|------:|
| embed_dim | 48 |
| num_heads | 4 |
| num_layers | 1 |
| epochs | 25 |
| domain_dropout | 0.15 |
| seeds | 42–46 |

Training loop (2026-05 fix): `early_stopping_metric: pr_auc`, `restore_best_weights: true`.

## Locked JSON artifacts

Phase-2 locked result JSONs were produced with **legacy** early stopping (`val_loss`, no restore).
Re-run with new defaults before updating confirmatory claims; mark runs `NEW CONFIRMATORY` in registry.
