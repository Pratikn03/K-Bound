# Family D v4 — Exploratory Remediation

**Status:** exploratory (does not supersede frozen `FAMILY_D_V3_NOT_CONFIRMED` without a new preregistered confirmatory package).

## Problem (v3)

Held-out Eyecandies transfer showed near-chance fusion (~0.50 AUC) and negative or null Δ(RGA − static) because:

1. Train/val labels are all-normal → supervised BCE collapses to a constant normal predictor.
2. `category_aware: true` in config was not wired in `_make_reliability_estimator`.
3. When the gate fired, RGA still routed through the same under-trained attention head instead of down-weighting collapsed modality scores.

## v4 fixes (code + `configs/phase2/family_d_v4_scoring_pipeline.yaml`)

| Fix | Effect |
|-----|--------|
| `one_class_score_supervision` | Train fusion against max(domain score) weak targets |
| `CategoryAwareReliabilityEstimator` + per-category KS re-fit on validation | Correct drift detection across 10 categories |
| `score_blend_on_gate` | Gated path uses reliability-weighted domain scores (static path unchanged) |

## Run (exploratory)

```bash
cd AutoML_Flagship_V8

# Primary cells (30 seeds)
PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_family_d_v2_cell.py \
  --cell D-EYE-1 --seeds 30 --seed-start 42 \
  --pipeline-spec configs/phase2/family_d_v4_scoring_pipeline.yaml \
  --allow-rerun --output-suffix v4_full --experiment-id D-EYE-v4

PYTHONPATH=src .venv/bin/python src/scripts/run_phase2_family_d_v2_cell.py \
  --cell D-EYE-2 --seeds 30 --seed-start 42 \
  --pipeline-spec configs/phase2/family_d_v4_scoring_pipeline.yaml \
  --allow-rerun --output-suffix v4_full --experiment-id D-EYE-v4

# Aggregate
PYTHONPATH=src .venv/bin/python src/scripts/summarize_family_d_eye3.py \
  --input experiments/phase2/family_d/family_d_d_eye_1_full_test_evaluation_v4_full_per_seed.csv \
  --out-prefix experiments/phase2/family_d/family_d_d_eye_1_v4_full_aggregate_summary
```

## Smoke-test result (3 seeds, D-EYE-1)

Mean Δ AUC = **+0.036** (95% CI [+0.018, +0.057]); static ≈ 0.568, RGA ≈ 0.604.

## Claim ceiling

v4 is **exploratory remediation** until a new frozen manifest + independent review. Do not replace v3 `NOT_CONFIRMED` in the paper without that process.
