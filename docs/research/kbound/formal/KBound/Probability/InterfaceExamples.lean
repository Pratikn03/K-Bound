import KBound.Probability.RandomizedActionLaw
import KBound.Probability.ExtendedRadiusCertificate

open KBound KBound.Decision MeasureTheory
open scoped ENNReal NNReal

namespace KBound.ProbabilityInterfaceExamples

section Randomized

variable [MeasurableSpace Decision] [MeasurableSingletonClass Decision]

/-- The full two-world rule theorem specializes to an 80 percent abstention
bound when both worldwise directional error budgets are 10 percent. -/
theorem tenth_error_bound
    {Ωneg Ωpos : Type*} [MeasurableSpace Ωneg] [MeasurableSpace Ωpos]
    {μneg : Measure Ωneg} {μpos : Measure Ωpos}
    [IsProbabilityMeasure μneg] [IsProbabilityMeasure μpos]
    {ruleNeg : Ωneg → Decision} {rulePos : Ωpos → Decision}
    (hmneg : Measurable ruleNeg) (hmpos : Measurable rulePos)
    (hcommon : μneg.map ruleNeg = μpos.map rulePos)
    (ha : μneg {ω | ruleNeg ω = adapt} ≤ ((1 / 10 : ℝ≥0) : ℝ≥0∞))
    (hf : μpos {ω | rulePos ω = freeze} ≤ ((1 / 10 : ℝ≥0) : ℝ≥0∞)) :
    ((4 / 5 : ℝ≥0) : ℝ≥0∞) ≤ μneg {ω | ruleNeg ω = abstain} ∧
      ((4 / 5 : ℝ≥0) : ℝ≥0∞) ≤ μpos {ω | rulePos ω = abstain} := by
  have h := RandomizedActionLaw.randomized_rules_force_abstention
    (deltaNeg := -1) (deltaPos := 1) (alpha := ((1 / 10 : ℝ≥0) : ℝ≥0∞))
    hmneg hmpos (by norm_num) (by norm_num) hcommon
    (by simpa using ha) (by simpa using hf)
  have hnum : ((4 / 5 : ℝ≥0) : ℝ≥0∞) ≤
      1 - 2 * ((1 / 10 : ℝ≥0) : ℝ≥0∞) := by
    rw [← ENNReal.coe_two, ← ENNReal.coe_mul, ← ENNReal.coe_one,
      ← ENNReal.coe_sub, ENNReal.coe_le_coe]
    apply le_tsub_of_add_le_right
    norm_num
  exact ⟨hnum.trans h.1, hnum.trans h.2⟩

/-- Zero-benefit boundary world: both strict actions count as errors, so the
same probability bound follows even without opposite nonzero worlds. -/
theorem zero_benefit_boundary {ρ : Measure Decision} [IsProbabilityMeasure ρ]
    {alpha : ℝ≥0∞}
    (ha : ρ {a | a = adapt ∧ (0 : ℝ) ≤ 0} ≤ alpha)
    (hf : ρ {a | a = freeze ∧ (0 : ℝ) ≤ 0} ≤ alpha) :
    1 - 2 * alpha ≤ ρ {abstain} := by
  exact RandomizedActionLaw.abstention_lower_bound
    (by simpa using ha) (by simpa using hf)

end Randomized

section MixedRadius

open KBound.ExtendedRadiusCertificate

local instance : MeasurableSpace Bool := ⊤

def radius (b : Bool) : ℝ≥0∞ := if b then ⊤ else 0
def estimate (b : Bool) : ℝ := if b then 10 else 1
def truth (b : Bool) : ℝ := if b then -10 else 1

/-- A single measurable radius function has an infinite branch and a finite
branch. Coverage holds everywhere, and the probability theorem gives zero
directional error for every probability law on these two branches. -/
theorem mixed_radius_zero_error (μ : Measure Bool) [IsProbabilityMeasure μ] :
    μ (falseDirection estimate truth radius) = 0 := by
  have hcoverage : coverage estimate truth radius = Set.univ := by
    ext b
    cases b <;> norm_num [coverage, estimate, truth, radius]
  have h := directional_error_probability μ
    (measurable_of_countable estimate) (measurable_of_countable truth)
    (measurable_of_countable radius) (alpha := 0)
    (by rw [hcoverage, measure_univ]; simp only [tsub_zero, le_refl])
  exact le_antisymm h (zero_le _)

end MixedRadius

end KBound.ProbabilityInterfaceExamples

#print axioms KBound.ProbabilityInterfaceExamples.tenth_error_bound
#print axioms KBound.ProbabilityInterfaceExamples.zero_benefit_boundary
#print axioms KBound.ProbabilityInterfaceExamples.mixed_radius_zero_error

namespace KBound.ProbabilityInterfaceExamples.Pointwise
open KBound.ExtendedRadiusCertificate

def mixedRadius (b : Bool) : ℝ≥0∞ := if b then ⊤ else 0

example : action 1 (mixedRadius true) = abstain := by
  simp [mixedRadius]

example : action 1 (mixedRadius false) = adapt := by
  norm_num [mixedRadius, action, certificate]

example : action (-1) (mixedRadius false) = freeze := by
  norm_num [mixedRadius, action, certificate]

example : action 0 (mixedRadius false) = abstain := by
  norm_num [mixedRadius, action, certificate]

example : ENNReal.ofReal |(10 : ℝ) - (-10)| ≤ mixedRadius true := by
  simp [mixedRadius]

end KBound.ProbabilityInterfaceExamples.Pointwise
