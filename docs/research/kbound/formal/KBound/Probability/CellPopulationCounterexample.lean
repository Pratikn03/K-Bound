import KBound.Probability.RandomRadiusCertificate
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic

/-! An exact probability-measure realization of `prop:cell-fail`.
The predictor reads only the input atom; the deterministic label is a separate
function. This is an existence counterexample, not a claim that this perfect
predictor can be learned from arbitrary label-free observations. -/

namespace KBound.CellPopulationCounterexample
open MeasureTheory KBound Decision
open scoped ENNReal

inductive Atom | a | b deriving DecidableEq, Fintype
instance : MeasurableSpace Atom := ⊤
instance : MeasurableSingletonClass Atom := ⟨fun _ => trivial⟩

noncomputable def law : Measure Atom :=
  (9 / 20 : ℝ≥0∞) • Measure.dirac Atom.a +
  (11 / 20 : ℝ≥0∞) • Measure.dirac Atom.b

instance law_probability : IsProbabilityMeasure law := by
  constructor
  norm_num [law, Measure.add_apply, Measure.smul_apply, Measure.dirac_apply']
  rw [← ENNReal.add_div]
  norm_num
  exact ENNReal.div_self (by norm_num) (by norm_num)

def label : Atom → Bool | .a => true | .b => false
def frozen (_ : Atom) : Bool := false
def adapted (_ : Atom) : Bool := true
noncomputable def cellBenefit (x : Atom) : ℝ :=
  (if adapted x = label x then 1 else 0) -
    (if frozen x = label x then 1 else 0)
def inputPredictor : Atom → ℝ | .a => 1 | .b => -1

@[simp] theorem prediction_equals_cell (x : Atom) :
    inputPredictor x = cellBenefit x := by cases x <;> norm_num [inputPredictor,cellBenefit,adapted,frozen,label]

theorem perfect_cell_coverage :
    law {x | |inputPredictor x - cellBenefit x| ≤ 0} = 1 := by simp

theorem frozen_risk : law {x | frozen x ≠ label x} = (9 / 20 : ℝ≥0∞) := by
  norm_num [law, Measure.add_apply, Measure.smul_apply, Measure.dirac_apply', frozen,label]

theorem adapted_risk : law {x | adapted x ≠ label x} = (11 / 20 : ℝ≥0∞) := by
  norm_num [law, Measure.add_apply, Measure.smul_apply, Measure.dirac_apply', adapted,label]

theorem population_benefit :
    law.real {x | frozen x ≠ label x} - law.real {x | adapted x ≠ label x} = -(1 / 10 : ℝ) := by
  simp only [measureReal_def, frozen_risk, adapted_risk]
  norm_num

theorem false_adapt_mass :
    law {x | extendedCertificate (inputPredictor x) 0 = adapt ∧
      law.real {x | frozen x ≠ label x} - law.real {x | adapted x ≠ label x} ≤ 0} =
      (9 / 20 : ℝ≥0∞) := by
  rw [population_benefit]
  have hevent : {x | extendedCertificate (inputPredictor x) 0 = adapt ∧
      -(1 / 10 : ℝ) ≤ 0} = {Atom.a} := by
    ext x
    cases x <;> norm_num [extendedCertificate, certificate, inputPredictor]
    decide
  rw [hevent]
  norm_num [law, Measure.add_apply, Measure.smul_apply, Measure.dirac_apply', Pi.single_apply]
  simp [show Atom.b ≠ Atom.a by decide]

end KBound.CellPopulationCounterexample
