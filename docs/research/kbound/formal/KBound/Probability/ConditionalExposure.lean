import KBound.Basics
import Mathlib.MeasureTheory.Measure.Real
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.Probability.ConditionalProbability

namespace KBound.ConditionalExposure

open MeasureTheory Set
open scoped ENNReal ProbabilityTheory

/-- A marginal event bound gives a conditional ratio bound only at positive
exposure. Both masses are evaluated under the same finite measure. -/
theorem event_ratio_le_min {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsFiniteMeasure μ] (A B : Set Ω) (alpha : ℝ)
    (hexposure : 0 < μ.real A) (hmarginal : μ.real (A ∩ B) ≤ alpha) :
    μ.real (A ∩ B) / μ.real A ≤ min 1 (alpha / μ.real A) := by
  apply le_min
  · exact (div_le_one hexposure).mpr (measureReal_mono inter_subset_left)
  · exact div_le_div_of_nonneg_right hmarginal hexposure.le

/-- False ADAPT includes zero benefit. No calibration or coverage premise is
inferred: the actual marginal error bound is an explicit hypothesis. -/
theorem false_adapt_ratio_le_min {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsFiniteMeasure μ] (action : Ω → Decision) (truth : Ω → ℝ)
    (alpha : ℝ) (hexposure : 0 < μ.real {ω | action ω = Decision.adapt})
    (hmarginal : μ.real {ω | action ω = Decision.adapt ∧ truth ω ≤ 0} ≤ alpha) :
    μ.real {ω | action ω = Decision.adapt ∧ truth ω ≤ 0} /
      μ.real {ω | action ω = Decision.adapt} ≤
        min 1 (alpha / μ.real {ω | action ω = Decision.adapt}) := by
  exact event_ratio_le_min μ {ω | action ω = Decision.adapt} {ω | truth ω ≤ 0}
    alpha hexposure hmarginal

/-- The ratio is the real mass of the bad-benefit event under mathlib's actual
conditional measure. Positive exposure excludes the undefined conditioning case
in the report; normalization applies to any finite base measure. -/
theorem conditional_false_adapt_le_min {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsFiniteMeasure μ] (action : Ω → Decision) (truth : Ω → ℝ)
    (alpha : ℝ) (hA : MeasurableSet {ω | action ω = Decision.adapt})
    (hexposure : 0 < μ.real {ω | action ω = Decision.adapt})
    (hmarginal : μ.real {ω | action ω = Decision.adapt ∧ truth ω ≤ 0} ≤ alpha) :
    (μ[|{ω | action ω = Decision.adapt}]).real {ω | truth ω ≤ 0} ≤
      min 1 (alpha / μ.real {ω | action ω = Decision.adapt}) := by
  have h := false_adapt_ratio_le_min μ action truth alpha hexposure hmarginal
  simpa only [Measure.real, ProbabilityTheory.cond_apply hA, ENNReal.toReal_mul,
    ENNReal.toReal_inv, div_eq_mul_inv, mul_comm] using h

end KBound.ConditionalExposure

#print axioms KBound.ConditionalExposure.event_ratio_le_min
#print axioms KBound.ConditionalExposure.false_adapt_ratio_le_min
#print axioms KBound.ConditionalExposure.conditional_false_adapt_le_min
