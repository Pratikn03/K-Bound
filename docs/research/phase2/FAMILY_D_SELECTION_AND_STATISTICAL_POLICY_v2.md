# Family-D v2 — Selection and Statistical Policy

**Phase:** 2.2C / Step 4
**Status:** FROZEN. Any deviation invalidates the held-out invariant.

## 1. Selection rules

For every Family-D v2 cell:

1. **Validation-only selection.** Threshold/calibration uses normal-only validation data and the pre-specified evidence-degradation injections only (D-EYE-1, D-EYE-2, and D-EYE-3 operators per [FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md](./FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md)).
2. **Frozen primary method = base RGA.** No supervised RGA+ head selection (no anomalous validation labels exist).
3. **Frozen comparator = `static_attention`.** No per-cell comparator selection.
4. **Frozen clean false-fire budget** = 0.010 (≤ 1.0% gate activation on clean validation data). Verified before test execution.
5. **Frozen τ** = derived from validation evidence-degradation injections under the budget constraint. Recorded in the per-seed selection log before any test-fold read.

## 2. Forbidden inputs to selection

- Official anomalous test labels.
- Test-fold metrics.
- Anomaly masks (under any split).
- Any property of the test set inspected directly or indirectly.

`selection_used_test_metrics = False` is asserted in every per-seed selection log row.

## 3. Inference rule

Per primary endpoint (D-EYE-1, D-EYE-2):

1. Stack per-seed test prediction vectors per method (base RGA + `static_attention`).
2. Compute seed-averaged ensemble prediction vector per method.
3. Run DeLong paired test on the ensemble vectors.
4. Compute paired test-sample bootstrap CI on `AUROC(RGA_ensemble) − AUROC(static_ensemble)`, 10 000 iterations, fixed seed 0.
5. Apply Holm–Bonferroni across `K = 2` cell-level p-values (`D-EYE-PRIMARY-K2`).
6. Report practical-effect band per Phase-2 statistical policy §5.

The secondary endpoint D-EYE-3 is **descriptive only**; it is NOT included in the Holm correction and may NOT be cited as confirmatory evidence.

## 4. Seed plan

- Target seed count: **30**.
- Minimum for inference: **15** (per the Phase-2 statistical policy default).
- Seed range: 42 through 71 (or 56 if 15 seeds).
- Per-seed reliability calibration uses the same global seed; degradation operator PRNG seeds are offset deterministically per [FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md](./FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md) §2.

## 5. Decision rules per cell

| Outcome | Required evidence |
|---|---|
| **CONFIRMED** | Holm K=2 p ≤ 0.05 AND paired bootstrap CI strictly excludes 0 AND \|Δ\| ≥ 0.010 (minimum practical) AND direction = positive |
| **DIRECTIONALLY_SUPPORTED** | Holm p ≤ 0.05 but CI includes 0 OR \|Δ\| < 0.010 |
| **NOT_CONFIRMED** | Holm p > 0.05 OR direction non-positive |
| **INVALID** | any `selection_used_test_metrics = True` row found, or operator parameters changed after test read |

## 6. Family decision

The Family-D v2 family decision is:

- `FAMILY_D_V2_CONFIRMED_BOTH_ENDPOINTS` if D-EYE-1 and D-EYE-2 both = CONFIRMED.
- `FAMILY_D_V2_PARTIAL_CONFIRMATION` if exactly one of {D-EYE-1, D-EYE-2} = CONFIRMED.
- `FAMILY_D_V2_NOT_CONFIRMED` if neither endpoint = CONFIRMED.
- `FAMILY_D_V2_INVALID` if any selection or operator integrity check fails.

## 7. Claim ceiling (per outcome)

- `CONFIRMED` outcome ⇒ may state: "Held-out confirmatory evidence under the frozen Eyecandies RGB+depth one-class degradation-stress protocol for the evaluated endpoint(s)."
- Any outcome ⇒ MAY NOT state any of:
  - ELARA is universal
  - ELARA is SOTA
  - ELARA is deployment-safe / clinically validated
  - Family A becomes confirmatory
  - RGA+ beats strongest baselines
  - Physical-AI safety validation
  - Raw-sensor corruption robustness

## 8. Pre-registration integrity contract

- The SHA256 of [FAMILY_D_HYPOTHESES_v2.csv](./FAMILY_D_HYPOTHESES_v2.csv), [family_d_v2_eyecandies_protocol.yaml](../../../configs/phase2/family_d_v2_eyecandies_protocol.yaml), [FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md](./FAMILY_D_V2_DEGRADATION_OPERATOR_SPEC.md), and this file MUST equal the values recorded in the partition manifest at freeze time.
- Any post-freeze modification to any of the four files invalidates the freeze; the run becomes `FAMILY_D_V2_INVALID`.

## 9. Independent review

The future execution task may proceed ONLY after independent external review of the freeze artifacts has produced explicit sign-off. The independent reviewer is not myself.
