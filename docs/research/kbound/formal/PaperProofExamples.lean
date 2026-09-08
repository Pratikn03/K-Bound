import KBound


open MeasureTheory

namespace KBound.PaperProofExamples

-- The original constructed-kernel theorem cannot discharge this arbitrary-law
-- statement. Removing or restricting the new theorem must break this regression.
example {X : Type*} [MeasurableSpace X]
    (P : Measure (X × Bool)) [IsProbabilityMeasure P]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (hD : 0 < P.real {xy | f0 xy.1 ≠ fa xy.1}) :
    populationBenefit f0 fa P =
      2 * P.real {xy | f0 xy.1 ≠ fa xy.1} *
        (P.real {xy | f0 xy.1 ≠ fa xy.1 ∧ xy.2 = fa xy.1} /
          P.real {xy | f0 xy.1 ≠ fa xy.1} - 1 / 2) := by
  exact JointTargetReduction.binary_population_reduction P f0 fa h0 ha hD

end KBound.PaperProofExamples


open MeasureTheory

-- A real probability law, not an independently supplied scalar benefit, must
-- realize the appendix's claimed half-unit population benefit.
example : ∃ P : Measure (Bool × Bool), IsProbabilityMeasure P ∧
    KBound.populationBenefit (fun x => !x) (fun _ => true) P = 1 / 2 := by
  exact KBound.PaperCounterexamples.calibration_witness_exists

-- Exact cell coverage must coexist with negative actual population benefit and
-- a nonzero false-adapt event under that same population law.
example : ∃ P : Measure (Bool × Bool), IsProbabilityMeasure P ∧
    KBound.populationBenefit (fun _ => false) (fun _ => true) P = -(1 / 10) ∧
    P {xy | KBound.certificate (if xy.1 then 1 else -1) 0 = KBound.Decision.adapt} =
      (9 / 20 : ENNReal) := by
  exact KBound.PaperCounterexamples.cell_population_witness_exists


open MeasureTheory

example : ∃ P : Measure (Fin 9 → ℝ), IsProbabilityMeasure P ∧
    KBound.ScoreLawExchangeable P ∧
    P {v | v 0 = 1} = (1 / 2 : ENNReal) := by
  exact KBound.DependentSignFlip.exchangeable_fair_witness

