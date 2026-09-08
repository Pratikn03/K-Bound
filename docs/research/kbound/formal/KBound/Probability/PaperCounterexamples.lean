import KBound.Probability.JointTargetReduction
import KBound.Probability.ExtendedRadiusCertificate

/-!
# Exact finite probability witnesses printed in the maintained appendix

`true` denotes input a and label1, `false` denotes input b and label0.
Each source/target law below is an actual joint probability measure. All benefits
are integrals of the existing zero-one loss difference, not separate scalars.
These are mathematical witnesses, not fitted models or experimental evidence.
-/

namespace KBound.PaperCounterexamples

open MeasureTheory JointTargetReduction
open scoped ENNReal

noncomputable def calibrationLaw : Measure (Bool × Bool) :=
  (1 / 2 : ENNReal) • Measure.dirac (true, true) +
  (1 / 2 : ENNReal) • Measure.dirac (false, false)

instance : IsProbabilityMeasure calibrationLaw := by
  constructor
  norm_num [calibrationLaw]
  exact ENNReal.inv_two_add_inv_two

/-- The source and target are the same joint law. At the only score value1/2
and the only predicted class1, actual candidate correctness is1/2. -/
theorem calibration_correct_probability :
    calibrationLaw {xy | xy.2 = true} = (1 / 2 : ENNReal) := by
  norm_num [calibrationLaw]

noncomputable def calibrationScore (_xy : Bool × Bool) : ℝ := 1 / 2

/-- The full ordinary score-calibration event identity, including score values
outside the support (where both sides are zero). -/
theorem calibration_score_calibrated (t : ℝ) :
    calibrationLaw {xy | xy.2 = true ∧ calibrationScore xy = t} =
      ENNReal.ofReal t * calibrationLaw {xy | calibrationScore xy = t} := by
  simp only [calibrationScore]
  by_cases ht : (1 / 2 : ℝ) = t
  · subst t
    simp only [and_true, Set.setOf_true, measure_univ, mul_one,
      calibration_correct_probability]
    norm_num [ENNReal.ofReal_div_of_pos]
  · simp only [ht, and_false, Set.setOf_false, measure_empty, mul_zero]

/-- Because the predicted class is always1, conditioning additionally on that
class preserves calibration; the other class has zero mass. -/
theorem calibration_top_label_calibrated (t : ℝ) (label : Bool) :
    calibrationLaw {xy | xy.2 = true ∧ true = label ∧ calibrationScore xy = t} =
      ENNReal.ofReal t * calibrationLaw {xy | true = label ∧ calibrationScore xy = t} := by
  cases label
  · simp
  · simpa using calibration_score_calibrated t

/-- Disagreement has mass1/2 and candidate-correct disagreement has the same
mass, so the disagreement-conditional candidate accuracy is exactly1. -/
theorem calibration_disagreement_accuracy :
    calibrationLaw.real (disagreementEvent (fun x => !x) (fun _ => true)) = 1 / 2 ∧
    calibrationLaw.real (correctOnDisagreement (fun x => !x) (fun _ => true)
      (fun _ => true)) /
      calibrationLaw.real (disagreementEvent (fun x => !x) (fun _ => true)) = 1 := by
  norm_num [calibrationLaw, Measure.real, disagreementEvent, correctOnDisagreement]

theorem calibration_actual_score_margin :
    (∫ xy in disagreementEvent (fun x => !x) (fun _ => true), calibrationScore xy ∂calibrationLaw) /
      calibrationLaw.real (disagreementEvent (fun x => !x) (fun _ => true)) - 1 / 2 = 0 := by
  simp only [calibrationScore, setIntegral_const, smul_eq_mul,
    calibration_disagreement_accuracy.1]
  norm_num

/-- The printed M=0, gamma=1/2 values use the constant score1/2 and the actual
conditional correctness event ratio from the same source/target probability law. -/
theorem calibration_margin_residual :
    ((1 / 2 : ℝ) - 1 / 2) = 0 ∧
    calibrationLaw.real (correctOnDisagreement (fun x => !x) (fun _ => true)
      (fun _ => true)) /
      calibrationLaw.real (disagreementEvent (fun x => !x) (fun _ => true)) - 1 / 2 = 1 / 2 := by
  rw [calibration_disagreement_accuracy.2]
  norm_num

theorem calibration_population_benefit :
    populationBenefit (fun x => !x) (fun _ => true) calibrationLaw = 1 / 2 := by
  rw [population_benefit_event_difference calibrationLaw _ _
    (measurable_of_countable _) measurable_const]
  norm_num [calibrationLaw, Measure.real, correctOnDisagreement]

theorem calibration_witness_exists : ∃ P : Measure (Bool × Bool),
    IsProbabilityMeasure P ∧ populationBenefit (fun x => !x) (fun _ => true) P = 1 / 2 :=
  ⟨calibrationLaw, inferInstance, calibration_population_benefit⟩

noncomputable def cellLaw : Measure (Bool × Bool) :=
  (9 / 20 : ENNReal) • Measure.dirac (true, true) +
  (11 / 20 : ENNReal) • Measure.dirac (false, false)

instance : IsProbabilityMeasure cellLaw := by
  constructor
  norm_num [cellLaw]
  rw [ENNReal.div_add_div_same]
  norm_num
  exact ENNReal.div_self (by norm_num) (by simp)

def cellEstimate (xy : Bool × Bool) : ℝ := if xy.1 then 1 else -1

/-- Only the input coordinate is read by this fixed predictor. -/
theorem cellEstimate_label_free (x y y' : Bool) :
    cellEstimate (x, y) = cellEstimate (x, y') := rfl

theorem cell_population_benefit :
    populationBenefit (fun _ => false) (fun _ => true) cellLaw = -(1 / 10) := by
  rw [population_benefit_event_difference cellLaw _ _ measurable_const measurable_const]
  norm_num [cellLaw, Measure.real, correctOnDisagreement]

/-- The target in the perfect coverage event is the *actual one-item score
difference*, not the population expectation. -/
theorem cell_coverage_probability :
    cellLaw {xy | |cellEstimate xy - zeroOneBenefit (fun _ => false) (fun _ => true) xy| ≤ 0} = 1 := by
  norm_num [cellLaw, cellEstimate, zeroOneBenefit]
  rw [ENNReal.div_add_div_same]
  norm_num
  exact ENNReal.div_self (by norm_num) (by simp)

theorem cell_adapt_probability :
    cellLaw {xy | certificate (cellEstimate xy) 0 = Decision.adapt} = (9 / 20 : ENNReal) := by
  norm_num [cellLaw, cellEstimate, certificate, Set.indicator_apply]
  simp

theorem cell_false_adapt_probability :
    cellLaw {xy | certificate (cellEstimate xy) 0 = Decision.adapt ∧
      populationBenefit (fun _ => false) (fun _ => true) cellLaw ≤ 0} = (9 / 20 : ENNReal) := by
  rw [cell_population_benefit]
  norm_num [cellLaw, cellEstimate, certificate, Set.indicator_apply]
  simp

/-- The same witness separates a marginal error probability of0.45 from
conditional error1 on ADAPT; both ratios use the same actual joint law. -/
theorem cell_conditional_false_adapt_one :
    cellLaw.real {xy | certificate (cellEstimate xy) 0 = Decision.adapt ∧
      populationBenefit (fun _ => false) (fun _ => true) cellLaw ≤ 0} /
    cellLaw.real {xy | certificate (cellEstimate xy) 0 = Decision.adapt} = 1 := by
  simp only [Measure.real, cell_false_adapt_probability, cell_adapt_probability]
  norm_num

theorem no_adaptation_no_false_adaptation {Ω : Type*} [MeasurableSpace Ω]
    (mu : Measure Ω) (action : Ω → Decision) (truth : Ω → ℝ)
    (h : ∀ omega, action omega ≠ Decision.adapt) :
    mu {omega | action omega = Decision.adapt ∧ truth omega ≤ 0} = 0 := by
  simp [h]

theorem cell_population_witness_exists : ∃ P : Measure (Bool × Bool),
    IsProbabilityMeasure P ∧
    populationBenefit (fun _ => false) (fun _ => true) P = -(1 / 10) ∧
    P {xy | certificate (if xy.1 then 1 else -1) 0 = Decision.adapt} = (9 / 20 : ENNReal) :=
  ⟨cellLaw, inferInstance, cell_population_benefit, cell_adapt_probability⟩

/-- With three labels, a disagreement point can make both models wrong; the
binary complementarity assertion cannot be extended to arbitrary multiclass Y. -/
theorem multiclass_neither_correct :
    let P : Measure (Unit × Fin 3) := Measure.dirac ((), 2)
    populationBenefit (fun _ => 0) (fun _ => 1) P = 0 ∧
    P {xy | xy.2 = (1 : Fin 3)} + P {xy | xy.2 = (0 : Fin 3)} = 0 := by
  have h20 : (2 : Fin 3) ≠ 0 := by decide
  have h21 : (2 : Fin 3) ≠ 1 := by decide
  simp [populationBenefit, zeroOneBenefit, h20, h21]

end KBound.PaperCounterexamples

#print axioms KBound.PaperCounterexamples.calibration_correct_probability
#print axioms KBound.PaperCounterexamples.calibration_score_calibrated
#print axioms KBound.PaperCounterexamples.calibration_top_label_calibrated
#print axioms KBound.PaperCounterexamples.calibration_actual_score_margin
#print axioms KBound.PaperCounterexamples.calibration_disagreement_accuracy
#print axioms KBound.PaperCounterexamples.calibration_margin_residual
#print axioms KBound.PaperCounterexamples.calibration_population_benefit
#print axioms KBound.PaperCounterexamples.cell_population_benefit
#print axioms KBound.PaperCounterexamples.cell_coverage_probability
#print axioms KBound.PaperCounterexamples.cell_adapt_probability
#print axioms KBound.PaperCounterexamples.cell_false_adapt_probability
#print axioms KBound.PaperCounterexamples.cell_conditional_false_adapt_one
#print axioms KBound.PaperCounterexamples.no_adaptation_no_false_adaptation
#print axioms KBound.PaperCounterexamples.multiclass_neither_correct
