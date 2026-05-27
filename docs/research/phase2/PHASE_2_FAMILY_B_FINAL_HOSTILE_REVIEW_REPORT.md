# Phase 2 Family-B — Final Hostile Review Report

**Phase:** 2.2B.2 / Step 7
**Reviewer posture:** senior trustworthy-ML reviewer + reproducibility engineer + confirmatory-evaluation auditor.

## Q1. Did B1 reproduce?

**YES.** Phase-2 Δ AUC = +0.0507 (95% CI [+0.0364, +0.0650]; Holm K=2 p = 4.31 × 10⁻¹²). Phase-1 target = +0.0506. 29/30 seeds sign-positive. Status `VERIFIED_REPRODUCED`. Source: `family_b_primary_replication_holm_k2.csv`.

## Q2. How must B2 be described under the estimator change?

Per [PHASE_2_B1_B2_INTEGRATION_POLICY.md](./PHASE_2_B1_B2_INTEGRATION_POLICY.md) §2: status `COMPARABLE_BUT_ESTIMATOR_CHANGED_POSITIVE_RESULT`. Manuscript must use dual-number form citing both Phase-1 (+0.0319, per-seed-mean over ≈ 5 seeds) and Phase-2 (+0.0939, 30-seed ensemble-pooled) without replacement.

## Q3. Did RGA-v2 execute under a verified selection policy?

**YES.** Per [RGA_V2_SELECTION_PROVENANCE_RECONCILIATION.md](./RGA_V2_SELECTION_PROVENANCE_RECONCILIATION.md): 15 selection seeds × 4 gates → 60 threshold-selection rows; 15 evaluation seeds × 4 gates × 3 attacks × 5 k values → ~2 880 failure-surface rows. Every row carries `selection_used_test_metrics = False`. Selection signature accepts only val-fold tensors (test-fold tensors rejected by signature).

## Q4. Did any RGA-v2 candidate pass C1..C6?

**NO.** Source: `rga_v2_failure_surface_inference.csv`.

| Gate | C1 budget | C2 partial improve | C3 k=4 not worsened | C5 val-only | C6 single policy | Decision |
|---|---|---|---|---|---|---|
| G0 | PASS | n/a | PASS | PASS | PASS | BASELINE_REFERENCE |
| G1 | **FAIL** (1.000 vs 0.010 budget) | FAIL (0/2+) | PASS | PASS | PASS | NOT_IMPROVED |
| G2 | **FAIL** (1.000) | FAIL (0/2+) | PASS | PASS | PASS | NOT_IMPROVED |
| G3 | **FAIL** (1.000) | FAIL (0/2+) | PASS | PASS | PASS | NOT_IMPROVED |

C4 (positive certificate) is `False (no extension admissible)` per [RGA_V2_CERTIFICATE_EXTENSION_DECISION.md](./RGA_V2_CERTIFICATE_EXTENSION_DECISION.md) because no candidate satisfies the prerequisite C1.

## Q5. What does the clean false-fire failure imply scientifically?

The current G1/G2/G3 implementation uses **batch-level min pooling** on reliability weights. Under ELARA-Bench-LA's 4-domain feature tensor, batch-level minimum reliability falls below the candidate τ_min on virtually every batch — clean or corrupted — driving clean false-fire to ≈ 1.000. This is a structural failure mode of batch-level minimum-pooling, not a tuning problem. A future RGA-v2 redesign would require a per-sample firing decision plus a calibration that admits clean reliability much closer to 1.0.

## Q6. Did B-MECH-3S reduce false firing?

**NO.** Per `domain_composition_shift_metrics.csv` (50 rows = 5 seeds × 10 mixtures): mean reduction_delta = 0; both global_ks_fire_rate and domain_aware_fire_rate = 1.0 across all 50 rows. Decision: `DOMAIN_COMPOSITION_FALSE_FIRE_NOT_REDUCED`.

## Q7. Does B-MECH-3S close the general category/cohort theorem?

**NO.** Per [MIXTURE_SHIFT_PROTOCOL.md](./MIXTURE_SHIFT_PROTOCOL.md), the original general theorem remains `DEFERRED_PENDING_NATURAL_CATEGORY_METADATA`. B-MECH-3S is **exploratory domain-composition shift only** and may not be cited as theorem closure.

## Q8. What does B-MECH-4 support?

Decision `TRADEOFF_IMPROVED` on the **locked window grid {32, 64, 128, 256, 512}** under the **three evaluated degradation types** (score collapse, score noise, missingness). Window 512 delivers detection power 62.4% with clean false-activation ≤ 0.06%. Source: `ks_window_size_power.csv`, `ks_true_degradation_power.csv`.

**Bounded claim:** "On the locked window grid and evaluated degradation types, larger KS windows raise detection power without raising false fire." Forbidden generalization beyond the evaluated grid / degradations.

## Q9. Are risk-dominance terms now admissible?

**YES** — after the Phase 2.2B.2 clean-arm run (30 seeds × static + RGA on uncorrupted ELARA-Bench-LA test fold) and the B-CERT-1 v2 re-run. Output: `risk_dominance_terms_v2.csv` populates (q₀, q₁, Δ₀, Δ₁, π*) per scenario. Source: [RISK_DOMINANCE_AND_CERTIFICATE_REPORT_FINAL.md](./RISK_DOMINANCE_AND_CERTIFICATE_REPORT_FINAL.md).

## Q10. Which certificates are positive or negative?

- B2 max_attack k=4: `CERTIFIED` (LCB ≈ +0.0085).
- B1 zero_attack k=4: `NOT_CERTIFIED` (LCB ≈ −0.005).

This is a genuine split: AUC delta positive on both scenarios; per-sample paired loss benefit positive only on max_attack. Reported honestly without averaging.

## Q11. Are any remaining Family-B items unresolved?

- Optional: 30-seed extension of B-MECH-2 (currently 15; minimum_for_inference = 15 per contract; not required).
- Optional: per-sample gate-fired flag in B-MECH-1/B-MECH-2 archives currently uses batch-level `adapted`; B-CERT-1 worked around this with an empirical-firing definition. A future driver patch could write per-sample firing directly.
- Out of scope for Family-B: general category/cohort theorem (deferred); G4 learned gate (intentionally not implemented).

No unresolved validity issue.

## Q12. Were negative results fully preserved?

**YES.** All negative findings are reported, not suppressed:

- G1/G2/G3 `NOT_IMPROVED`.
- B-MECH-3S `DOMAIN_COMPOSITION_FALSE_FIRE_NOT_REDUCED`.
- B1 zero_attack k=4 `NOT_CERTIFIED`.
- B2 magnitude change flagged as `COMPARABLE_BUT_ESTIMATOR_CHANGED` rather than silently adopted.

## Q13. Was Family-D untouched?

**YES.** v1 still `INVALID_FOR_EXECUTION`; v2 still `V2_DESIGN_PENDING`. No Family-D file modified by Phase 2.2B.2 prior to the Step 10/11 stage. The Phase 2.2B.2 task explicitly forbids Family-D execution even during the v2 freeze step.

## Q14. What claims may a future manuscript-update phase add?

1. B1 reproduction at +0.0507 [+0.0364, +0.0650] (dual-citation with Phase-1 +0.0506).
2. B2 positive at +0.0939 [+0.0741, +0.1149] under the explicit estimator-change caveat.
3. RGA-v2 negative result: G1/G2/G3 fail C1 clean false-fire budget; no promotion.
4. KS window tradeoff on the locked grid: larger windows raise detection power without raising false fire.
5. Domain-composition shift exploratory audit: false-fire not reduced.
6. B-CERT-1 split: max_attack CERTIFIED, zero_attack NOT_CERTIFIED.
7. Full risk-dominance table (q₀, q₁, Δ₀, Δ₁, π*) per B1/B2 scenario.

All claims must carry the locked boundary text "These are retrospective evaluation certificates under defined stress protocols; they are not production safety certificates or real-world deployment guarantees" wherever certificates are cited.

## Q15. What claims remain forbidden?

- RGA-v2 solves partial failures.
- ELARA handles partial failures k=1, k=2 or k=3 better.
- Category-aware KS reduces false firing.
- KS window-size tradeoff is validated beyond the evaluated grid / degradations.
- Theory closure is complete (general category/cohort theorem deferred).
- Family-D has confirmed ELARA.
- ELARA is universal / SOTA / production-ready / clinically validated.
- Retrospective certificates equal deployment safety.
- Phase 3 may begin.
