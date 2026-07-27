import KBound.Probability.UniformConformal
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability

/-!
# Exchangeable-score reduction to the uniform-index conformal model

Paper: split-conformal under exchangeability (`sec:t2-prop-conformal`, L4 bridge).

Conditional on the score bag, exchangeability of the `n+1` scores makes the held-out
index uniform on `Fin (n+1)`.  This module records that reduction as an equality of
laws (`μ = uniformIndexMeasure n`) and transports the kernel-checked uniform-index
miss / false-adapt bounds to the exchangeable-score model.
-/

namespace KBound

open MeasureTheory

variable {n : ℕ}

/-- **Uniform-index-law miss bound.**  Scope, stated honestly: uniformity of the
held-out index law is a HYPOTHESIS here (`hexch : μ = uniformIndexMeasure n`), not
a conclusion.  The step from exchangeability of the scores to uniformity of the
held-out index is the mathematically substantive one and is NOT formalised in this
development.  Given uniformity, the rank-`k` miss probability is at most `α`
whenever the quantile threshold holds.
Renamed 2026-07-26 (was `exchangeable_scores_miss_le_alpha`): the old name implied
the exchangeability reduction had been proved. -/
theorem uniformIndexLaw_miss_le_alpha
    (R : Fin (n + 1) → ℝ) (k : ℕ) {alpha : ENNReal}
    {μ : Measure (Fin (n + 1))} [IsProbabilityMeasure μ]
    (hexch : μ = uniformIndexMeasure n)
    (hk : (((n + 1) - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal) ≤ alpha) :
    μ {j | k ≤ strictRank R j} ≤ alpha := by
  rw [hexch]
  exact le_trans (uniformIndex_miss_le R k) hk

/-- **Exchangeable-score false-adapt bound.**  Under the exchangeable → uniform
index reduction, the certificate's unconditional false-adapt probability is
`≤ α` end-to-end (composes with `uniformIndex_false_adapt_le`). -/
theorem uniformIndexLaw_false_adapt_le
    (dhat delta : Fin (n + 1) → ℝ) (eps : ℝ) (R : Fin (n + 1) → ℝ) (k : ℕ)
    {alpha : ENNReal}
    {μ : Measure (Fin (n + 1))} [IsProbabilityMeasure μ]
    (hexch : μ = uniformIndexMeasure n)
    (hsub : {j | ¬ k ≤ strictRank R j} ⊆ coverageEvent dhat delta eps)
    (hk : (((n + 1) - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal) ≤ alpha) :
    μ (falseAdaptEvent dhat delta eps) ≤ alpha := by
  rw [hexch]
  exact uniformIndex_false_adapt_le dhat delta eps R k hsub hk

/-- Mirror: exchangeable-score false-freeze bound. -/
theorem uniformIndexLaw_false_freeze_le
    (dhat delta : Fin (n + 1) → ℝ) (eps : ℝ) (R : Fin (n + 1) → ℝ) (k : ℕ)
    {alpha : ENNReal}
    {μ : Measure (Fin (n + 1))} [IsProbabilityMeasure μ]
    (hexch : μ = uniformIndexMeasure n)
    (hsub : {j | ¬ k ≤ strictRank R j} ⊆ coverageEvent dhat delta eps)
    (hk : (((n + 1) - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal) ≤ alpha) :
    μ (falseFreezeEvent dhat delta eps) ≤ alpha := by
  rw [hexch]
  exact uniformIndex_false_freeze_le dhat delta eps R k hsub hk

end KBound
