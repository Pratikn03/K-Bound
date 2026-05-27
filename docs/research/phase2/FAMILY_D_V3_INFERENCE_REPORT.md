# Family-D v3 — Inference Report

**Phase:** 2.2E / Stage 4
**Family decision:** `FAMILY_D_V3_INVALID_CLEAN_FALSE_FIRE_POLICY_FAILURE`

## 1. Per-cell primary inference

| Cell | n_seeds | Δ(AUC) ensemble | DeLong p (corrected) | Bootstrap CI (95%) | Pre-Holm outcome | Holm-adjusted |
|---|---|---|---|---|---|---|
| D-EYE-1 | 30 | -0.0072 | 0.3323 | [-0.0222, +0.0074] | NOT_CONFIRMED | NOT_CONFIRMED |
| D-EYE-2 | 30 | -0.0053 | 0.3127 | [-0.0158, +0.0048] | NOT_CONFIRMED | NOT_CONFIRMED |

## 2. Holm–Bonferroni correction (K=2, corrected p-values)

| Step | Cell | p-value | Threshold (α/k) | Reject H0 |
|---|---|---|---|---|
| 1 | D-EYE-2 | 0.3127 | 0.0250 | False |
| 2 | D-EYE-1 | 0.3323 | 0.0500 | False |

## 3. Family decision

> **`FAMILY_D_V3_INVALID_CLEAN_FALSE_FIRE_POLICY_FAILURE`**

### Claim ceiling (per frozen selection policy §7)

> A planned Eyecandies held-out evaluation was executed but excluded from primary evidence because protocol validity requirements were not satisfied.

### Validity notes

- Frozen clean false-fire budget (`<= 0.010`) was violated in validation (observed `1.0`).
- The originally reported DeLong `0.0000` values were invalid due to a double-division variance bug; corrected p-values are shown above.

### Forbidden claims (regardless of outcome)

- ELARA is universal
- ELARA is SOTA
- ELARA is deployment-safe
- Family A becomes confirmatory
- RGA+ beats strongest baselines
- Physical-AI safety validation
- Raw-sensor corruption robustness

## 4. Reproducibility

- Bootstrap: 10,000 iterations, seed=0
- CI: 95% two-sided
- Multiplicity correction: Holm–Bonferroni K=2
- Minimum practical delta: 0.01
- Minimum seeds for inference: 15
- Clean false-fire budget: <= 0.010 (policy requirement)

## 5. selection_used_test_metrics audit

All per-seed rows must have `selection_used_test_metrics = False`. Any row with True invalidates the family.
