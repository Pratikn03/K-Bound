# Family-D v3 — Inference Report

**Phase:** 2.2E / Stage 4
**Family decision:** `FAMILY_D_V3_NOT_CONFIRMED`

## 1. Per-cell primary inference

| Cell | n_seeds | Δ(AUC) ensemble | DeLong p | Bootstrap CI (95%) | Pre-Holm outcome | Holm-adjusted |
|---|---|---|---|---|---|---|
| D-EYE-1 | 30 | -0.0010 | 0.8470 | [-0.0114, +0.0092] | NOT_CONFIRMED | NOT_CONFIRMED |
| D-EYE-2 | 30 | -0.0109 | 0.1391 | [-0.0254, +0.0034] | NOT_CONFIRMED | NOT_CONFIRMED |

## 2. Holm–Bonferroni correction (K=2)

| Step | Cell | p-value | Threshold (α/k) | Reject H0 |
|---|---|---|---|---|
| 1 | D-EYE-2 | 0.1391 | 0.0250 | False |
| 2 | D-EYE-1 | 0.8470 | 0.0500 | False |

## 3. Family decision

> **`FAMILY_D_V3_NOT_CONFIRMED`**

### Claim ceiling (per frozen selection policy §7)

> Held-out confirmation was not obtained for the evaluated endpoint(s); negative results are retained.

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

## 5. selection_used_test_metrics audit

All per-seed rows must have `selection_used_test_metrics = False`. Any row with True invalidates the family.
