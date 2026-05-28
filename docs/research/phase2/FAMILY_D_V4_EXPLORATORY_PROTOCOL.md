# Family-D v4 — Exploratory Protocol (Post-v3, Non-Confirmatory)

**Status:** `FAMILY_D_V4_EXPLORATORY_NON_CONFIRMATORY`
**Date (UTC):** 2026-05-25
**Relationship to v3:** v3 NOT_CONFIRMED result is **preserved as primary held-out evidence**. v4 is a follow-on diagnostic, run in exploratory mode, to test whether the v3 null result is an artefact of three frozen protocol choices that mathematically suppress any RGA / static delta.

## 1. Rationale: why v3 produced a near-zero delta

Three properties of the v3 frozen contract jointly forced Δ-AUC → 0 independent of any gating effect:

1. **Hard score-collapse operator** (`score → 0`). Under one-modality collapse, both Static = `(rgb + 0)/2` and RGA = `(w_rgb·rgb + 0)/(w_rgb+w_depth)` reduce to monotone transforms of a single modality score. AUC is rank-invariant under monotone transforms → AUC(static) = AUC(RGA) by construction. No gating mechanism can produce a non-zero delta in this regime.
2. **Rank-based primary metric** (ROC-AUC). Reliability weighting changes score *magnitudes* but only weakly perturbs *ranks*. A magnitude-sensitive metric (Brier, ECE) is the matched primary endpoint for a reliability-weighted method.
3. **Seed variance ≈ effect size.** Per-seed SD ≈ 0.011 vs mean shift ±0.001 → 95% CI necessarily straddles zero at 30 seeds.

These are mathematical consequences of the contract, not bugs in the v3 implementation.

## 2. Three protocol changes (v4)

| # | Change | v3 | v4 | Why |
|---|---|---|---|---|
| 1 | **Degradation operator** | Hard collapse: `score → 0` for every test sample | Soft corruption: `score' = α·U(0,1) + (1−α)·score` with α = 0.5 | Preserves partial signal in the corrupted modality → both modalities carry information → RGA's reliability weighting can produce non-monotone rank shifts |
| 2 | **Primary metric set** | ROC-AUC only | ROC-AUC **and** Brier score (lower is better) | Brier is non-rank-invariant: it rewards both ranking AND calibrated magnitude. Reliability weighting acts on magnitudes. |
| 3 | **Seed count** | 30 | 60 | Halves the standard error of the mean delta (~0.011/√60 ≈ 0.0014), enabling detection of effects in the 0.005–0.010 range |

All other v3 frozen artefacts (categories, splits, base feature extractor, train/val protocol, sample-id schema, gate threshold τ = 0.66, coreset fraction 10%, validation-only selection rule) **remain unchanged**.

## 3. Endpoints

| ID | Operator | Modality target | α |
|---|---|---|---|
| D-EYE-1v4 | Soft-corruption | depth | 0.5 |
| D-EYE-2v4 | Soft-corruption | rgb | 0.5 |
| D-EYE-3v4 | Single-modality dropout (alternating) | both (alternates) | n/a |

Holm K=2 across D-EYE-1v4 / D-EYE-2v4 primary cells. D-EYE-3v4 reported descriptive.

## 4. Decision rules

Each cell is reported in the v4 final decision file with:
- `mean_per_seed_delta_auc` and 95% bootstrap CI
- `mean_per_seed_delta_brier` and 95% bootstrap CI (positive = RGA worse calibrated; negative = RGA better calibrated)
- Holm K=2 p-value across both endpoints
- Status: `V4_POSITIVE_DELTA_OBSERVED` / `V4_NULL_DELTA_REPRODUCED` / `V4_NEGATIVE_DELTA_OBSERVED`

**v4 cannot be used to revise the v3 primary held-out claim.** v3 stays NOT_CONFIRMED. v4 contextualizes v3.

## 5. Reporting

- v4 results appear in `experiments/phase2/family_d/family_d_v4_*.csv`.
- A short `FAMILY_D_V4_EXPLORATORY_DECISION.md` documents the outcome.
- The manuscript is **not modified** by v4 results in this session; any incorporation requires a separate editorial decision.

## 6. Forbidden claims (carried over from v3)

- ELARA is universal
- ELARA is SOTA
- ELARA is deployment-safe
- Family-A becomes confirmatory because of v4
- RGA+ beats strongest baselines
- v4 confirms what v3 did not (forbidden — v4 is exploratory, not confirmatory)
