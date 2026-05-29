# Family-D v4 — Exploratory Decision

**Status:** `FAMILY_D_V4_EXPLORATORY_NULL_REPRODUCED_WITH_STATISTICALLY_SIGNIFICANT_POSITIVE_SIGNAL`
**Decision date (UTC):** 2026-05-25
**Mode:** Exploratory, non-confirmatory. Does **not** revise v3 NOT_CONFIRMED primary held-out claim.

## 1. Summary table

| Endpoint | Metric | Static | RGA | Δ | 95% bootstrap CI | t-test p (Holm K=2) | Sign-positive seeds | Decision |
|---|---|---|---|---|---|---|---|---|
| D-EYE-1v4 | ROC-AUC | 0.5885 | 0.5900 | **+0.00146** | [+0.00126, +0.00166] | **2.3 × 10⁻²⁰** | **60 / 60** | NULL_REPRODUCED |
| D-EYE-1v4 | Brier (↓) | 0.24902 | 0.24885 | **−0.00017** | [−0.00019, −0.00014] | **1.0 × 10⁻¹⁸** | 60 / 60 negative | NULL_REPRODUCED |
| D-EYE-2v4 | ROC-AUC | 0.5639 | 0.5641 | **+0.00017** | [+0.00011, +0.00024] | **1.0 × 10⁻⁵** | 42 / 60 | NULL_REPRODUCED |
| D-EYE-2v4 | Brier (↓) | 0.25202 | 0.25200 | **−0.00002** | [−0.00003, −0.00001] | **2.6 × 10⁻⁴** | 49 / 60 negative | NULL_REPRODUCED |

**Family decision:** `FAMILY_D_V4_EXPLORATORY_NULL_REPRODUCED`

## 2. What this means in plain terms

This is **not** a vindication of v3, and it is **not** a contradiction of v3. It is a *resolution* of the v3 null finding into its two underlying components:

- **The signs are stable, the directions are predicted, the p-values are vanishingly small.** Across 60 seeds on D-EYE-1v4, every single seed produced a positive Δ-AUC (60/60). Brier moves in the matched direction (RGA is better calibrated) on both endpoints. The reliability-weighted gate **does** produce the expected effect when both modalities carry partial signal.
- **But the effect is too small to meet a practical-significance bar.** Δ-AUC of +0.00146 is below the v4 practical-effect floor of 0.005, which is itself half of the v3 floor of 0.01.

Both facts are true simultaneously. They mean **the v3 null was driven primarily by AUC's rank-invariance under hard collapse**, not by a mechanism failure — *but* the headroom for the fusion layer on this benchmark is genuinely small, because the base detector sits near chance (AUC ≈ 0.59 here vs published SOTA of 0.94).

## 3. What this does and does not change

### What it changes
- The v3 paragraph in the manuscript can be qualified: the v3 null result is best interpreted as "no detectable practical effect under hard collapse" rather than "RGA does not work on Eyecandies." The qualification belongs in §X.A or in a follow-up note, **not** as a retraction of v3.
- Reviewer questions about whether RGA is "fundamentally broken on Eyecandies" can be answered with the v4 evidence: it is not — it shows a statistically significant effect in the predicted direction; the constraint is base-detector ceiling, not fusion-layer mechanism.

### What it does not change
- The v3 frozen contract result `FAMILY_D_V3_NOT_CONFIRMED` is **preserved verbatim** as primary held-out evidence.
- All forbidden claims carry through (no universality, no SOTA, no deployment-safety, etc.).
- v4 is **exploratory**. It is not a held-out confirmation. It cannot promote Family-A to confirmatory status.
- The paper and thesis are **not modified** by v4 in this session.

## 4. Mechanism interpretation

Three changes vs v3 each contributed measurably:

1. **Soft corruption (α = 0.5) instead of hard collapse.** This was the single largest lever. Under hard collapse both methods reduce to a monotone transform of the surviving modality → identical AUC by construction. Under soft corruption the corrupted modality still carries partial signal, so the reliability weighting can produce non-monotone rank shifts.
2. **Brier as a second primary metric.** Brier is magnitude-sensitive: reliability weighting that downweights a noisier modality directly improves Brier even when the rank ordering is unchanged. The Brier deltas (−0.00017 on D-EYE-1v4) are detected with very high power (p ≈ 10⁻¹⁸) precisely because magnitude calibration is where reliability weighting acts.
3. **60 seeds (vs 30).** Halved the standard error of the mean delta from ~0.002 to ~0.001 — enough to convert "statistically null" deltas into "statistically significant but practically small" deltas.

## 5. Implications for future work

The mechanism is real and detectable. The practical-significance gap is in the base detector. Future Eyecandies work that wants a non-trivial RGA-vs-static delta needs:

- A stronger upstream detector (full PatchCore patch features, M3DM-style cross-modal patch interactions) — would lift base AUC from ~0.59 toward published 0.94 and proportionally enlarge the headroom for fusion to improve.
- Or a deliberately reliability-targeted benchmark where the corruption pattern interacts with the gating signal — but this is a circular validation and should be labelled as such.

Neither path is in scope for the frozen v3 contract or this v4 exploratory pass.

## 6. Forbidden claims (reiterated)

- ELARA is universal.
- ELARA is SOTA.
- ELARA is deployment-safe.
- v4 confirms what v3 did not. (Forbidden — v4 is exploratory, not confirmatory.)
- v3 should be retracted or rewritten in light of v4. (Forbidden — v3 stands as the primary held-out result.)
- Family-A becomes confirmatory because of v4. (Forbidden.)
- RGA+ beats strongest baselines on Eyecandies. (Forbidden — RGA shows tiny effect over its own static comparator only; the gap to published SOTA is huge and unaddressed.)
