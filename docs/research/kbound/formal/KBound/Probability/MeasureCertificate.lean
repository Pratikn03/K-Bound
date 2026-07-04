import KBound.Certificate
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.MeasureTheory.Constructions.BorelSpace.Order

/-!
# Measure-theoretic certificate bound (`thm:certificate`, probability layer)

Paper: main text Theorem `thm:certificate` (short paper) / `thm:cert` (long paper).

This is the genuine probabilistic form of the false-adapt/false-freeze guarantee:
for an arbitrary probability measure `μ` on a measurable space `Ω`, if the benefit
estimate covers the true benefit with probability at least `1 − α`, then the
certificate's unconditional false-adapt and false-freeze probabilities are each at
most `α`.  The event containment is inherited pointwise from
`KBound.Certificate` (`cert_false_adapt_implies_coverage_failure`), so this file
adds exactly the measure layer: monotonicity plus complement arithmetic in `ℝ≥0∞`.

Only the *coverage* event needs to be measurable (`measure_mono` requires no
measurability of the smaller set); a helper derives that measurability from
measurability of the estimate and benefit maps.
-/

namespace KBound

open Decision MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]

/-- The coverage event: the benefit estimate is within `eps` of the true benefit. -/
def coverageEvent (dhat delta : Ω → ℝ) (eps : ℝ) : Set Ω :=
  {ω | |dhat ω - delta ω| ≤ eps}

/-- The false-adapt event: the certificate commits `adapt` while the true benefit
is nonpositive. -/
def falseAdaptEvent (dhat delta : Ω → ℝ) (eps : ℝ) : Set Ω :=
  {ω | certificate (dhat ω) eps = adapt ∧ delta ω ≤ 0}

/-- The false-freeze event: the certificate commits `freeze` while the true benefit
is nonnegative. -/
def falseFreezeEvent (dhat delta : Ω → ℝ) (eps : ℝ) : Set Ω :=
  {ω | certificate (dhat ω) eps = freeze ∧ 0 ≤ delta ω}

/-- Measurability of the coverage event from measurability of the two maps. -/
lemma measurableSet_coverageEvent {dhat delta : Ω → ℝ}
    (hdhat : Measurable dhat) (hdelta : Measurable delta) (eps : ℝ) :
    MeasurableSet (coverageEvent dhat delta eps) :=
  measurableSet_le ((hdhat.sub hdelta).abs) measurable_const

/-- Pointwise: a false adapt lies outside the coverage event. -/
lemma falseAdaptEvent_subset_compl_coverage (dhat delta : Ω → ℝ) (eps : ℝ) :
    falseAdaptEvent dhat delta eps ⊆ (coverageEvent dhat delta eps)ᶜ := by
  rintro ω ⟨hcert, hnonpos⟩ hcov
  exact cert_false_adapt_implies_coverage_failure hcert (not_lt.mpr hnonpos) hcov

/-- Pointwise: a false freeze lies outside the coverage event. -/
lemma falseFreezeEvent_subset_compl_coverage (dhat delta : Ω → ℝ) (eps : ℝ) :
    falseFreezeEvent dhat delta eps ⊆ (coverageEvent dhat delta eps)ᶜ := by
  rintro ω ⟨hcert, hnonneg⟩ hcov
  exact cert_false_freeze_implies_coverage_failure hcert (not_lt.mpr hnonneg) hcov

/-- `1 - (1 - α) ≤ α` in `ℝ≥0∞` (truncated subtraction), unconditionally. -/
lemma one_tsub_one_tsub_le (alpha : ENNReal) : 1 - (1 - alpha) ≤ alpha := by
  by_cases h : alpha ≤ 1
  · rw [tsub_tsub_cancel_of_le h]
  · calc (1 : ENNReal) - (1 - alpha) ≤ 1 := tsub_le_self
      _ ≤ alpha := le_of_not_ge h

/-- Any event contained in the complement of a `≥ 1 − α` event has probability `≤ α`. -/
theorem measure_le_alpha_of_subset_compl {μ : Measure Ω} [IsProbabilityMeasure μ]
    {s t : Set Ω} {alpha : ENNReal}
    (hs : MeasurableSet s) (hsub : t ⊆ sᶜ) (hcov : 1 - alpha ≤ μ s) :
    μ t ≤ alpha := by
  calc μ t ≤ μ sᶜ := measure_mono hsub
    _ = 1 - μ s := prob_compl_eq_one_sub hs
    _ ≤ 1 - (1 - alpha) := tsub_le_tsub_left hcov 1
    _ ≤ alpha := one_tsub_one_tsub_le alpha

/-- **Paper `thm:certificate`, false-adapt direction, measure form.**
If the coverage event has probability at least `1 − α`, the unconditional
false-adapt probability is at most `α`. -/
theorem measure_false_adapt_le_alpha {μ : Measure Ω} [IsProbabilityMeasure μ]
    {dhat delta : Ω → ℝ} {eps : ℝ} {alpha : ENNReal}
    (hs : MeasurableSet (coverageEvent dhat delta eps))
    (hcov : 1 - alpha ≤ μ (coverageEvent dhat delta eps)) :
    μ (falseAdaptEvent dhat delta eps) ≤ alpha :=
  measure_le_alpha_of_subset_compl hs
    (falseAdaptEvent_subset_compl_coverage dhat delta eps) hcov

/-- **Paper `thm:certificate`, false-freeze direction, measure form.** -/
theorem measure_false_freeze_le_alpha {μ : Measure Ω} [IsProbabilityMeasure μ]
    {dhat delta : Ω → ℝ} {eps : ℝ} {alpha : ENNReal}
    (hs : MeasurableSet (coverageEvent dhat delta eps))
    (hcov : 1 - alpha ≤ μ (coverageEvent dhat delta eps)) :
    μ (falseFreezeEvent dhat delta eps) ≤ alpha :=
  measure_le_alpha_of_subset_compl hs
    (falseFreezeEvent_subset_compl_coverage dhat delta eps) hcov

/-- Convenience form with measurable maps instead of a measurable-set hypothesis. -/
theorem measure_false_adapt_le_alpha_of_measurable
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    {dhat delta : Ω → ℝ} {eps : ℝ} {alpha : ENNReal}
    (hdhat : Measurable dhat) (hdelta : Measurable delta)
    (hcov : 1 - alpha ≤ μ (coverageEvent dhat delta eps)) :
    μ (falseAdaptEvent dhat delta eps) ≤ alpha :=
  measure_false_adapt_le_alpha (measurableSet_coverageEvent hdhat hdelta eps) hcov

end KBound
