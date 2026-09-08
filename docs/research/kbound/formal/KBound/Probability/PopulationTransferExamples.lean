import KBound.Probability.PopulationTransfer
import Mathlib.MeasureTheory.Measure.Dirac
import Mathlib.Tactic.NormNum

open MeasureTheory Set KBound.PopulationTransfer
open scoped ENNReal

namespace KBound.PopulationTransferExamples

theorem nonzero_composition : |(5 : ℝ) - 3| ≤ 1 + 1 :=
  pointwise (cell := 4) (by norm_num) (by norm_num)

theorem nonzero_positive : (0 : ℝ) < 3 :=
  positive (dhat := 5) (cell := 4) (eps := 1) (b := 1)
    (by norm_num) (by norm_num) (by norm_num)

theorem nonzero_negative : (-3 : ℝ) < 0 :=
  negative (dhat := -5) (cell := -4) (eps := 1) (b := 1)
    (by norm_num) (by norm_num) (by norm_num)

/-- Lower endpoint zero with both premise errors nonzero: no strict sign. -/
theorem lower_zero_boundary : |(2 : ℝ) - 1| ≤ 1 ∧ |(1 : ℝ) - 0| ≤ 1 ∧
    2 - (1 + 1 : ℝ) = 0 ∧ ¬ (0 < (0 : ℝ)) := by norm_num

/-- Upper endpoint zero, the symmetric boundary counterexample. -/
theorem upper_zero_boundary : |(-2 : ℝ) - (-1)| ≤ 1 ∧ |(-1 : ℝ) - 0| ≤ 1 ∧
    -2 + (1 + 1 : ℝ) = 0 ∧ ¬ ((0 : ℝ) < 0) := by norm_num

theorem exhausted_budget : (1 : ℝ≥0∞) - (1 + 1) = 0 := by
  apply tsub_eq_zero_of_le
  exact le_add_right le_rfl

/- A finite, perfectly correlated pair: A = B = {true} under a fair Boolean
measure. P(A ∩ B) = 1/2, whereas P(A)P(B) = 1/4. -/
local instance : MeasurableSpace Bool := ⊤

noncomputable def fairBool : Measure Bool :=
  (1 / 2 : ℝ≥0∞) • Measure.dirac true + (1 / 2 : ℝ≥0∞) • Measure.dirac false

instance : IsProbabilityMeasure fairBool := by
  constructor
  norm_num [fairBool, Measure.add_apply, Measure.smul_apply, ENNReal.inv_two_add_inv_two]

theorem fair_true : fairBool {true} = 1 / 2 := by
  norm_num [fairBool, Measure.add_apply, Measure.smul_apply, Measure.dirac_apply]

theorem correlated_not_product :
    fairBool ({true} ∩ {true}) ≠ fairBool {true} * fairBool {true} := by
  rw [inter_self, fair_true]
  intro h
  have hr := congrArg ENNReal.toReal h
  norm_num [ENNReal.toReal_mul] at hr

/-- The same general bridge applies despite the explicit nonfactorization. -/
theorem correlated_bridge :
    (1 : ℝ≥0∞) - (1 / 2 + 1 / 2) ≤ fairBool ({true} ∩ {true}) := by
  apply event_transfer fairBool (A := {true}) (B := {true})
      MeasurableSet.of_discrete MeasurableSet.of_discrete
  · rw [fair_true]; norm_num [ENNReal.one_sub_inv_two]
  · rw [fair_true]; norm_num [ENNReal.one_sub_inv_two]
  · exact Subset.rfl

end KBound.PopulationTransferExamples

#print axioms KBound.PopulationTransferExamples.correlated_not_product
#print axioms KBound.PopulationTransferExamples.correlated_bridge
