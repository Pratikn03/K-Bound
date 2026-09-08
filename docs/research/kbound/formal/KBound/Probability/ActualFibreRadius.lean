import KBound.Probability.ActualWorldEvidence
import KBound.Probability.AuditFloorCorrectness

/-! # Radius of the actual full joint-law class and its augmented-law fibre

The supremum below ranges over actual joint ProbabilityMeasures and their actual
scoreResidual, not over correctness fields or an independently supplied margin.
Fields only construct an extremal actual witness. The printed small-budget
restriction0<=beta<=1/2, positive disagreement and fixed observation experiment
remain explicit. No arbitrary restricted-class or empirical-radius claim follows.
-/

namespace KBound.ActualFibreRadius

open MeasureTheory ProbabilityTheory Set JointKernelScore ActualWorldFrontier ActualWorldEvidence
open scoped ENNReal ProbabilityTheory

variable {X : Type*} [MeasurableSpace X]

theorem actual_abs_residual_attained (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ)
    (hb0 : 0 ≤ beta) (hb1 : beta ≤ 1 / 2) :
    ∃ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P ∧
      |scoreResidual (P : Measure (X × Bool)) f0 fa ha score| = beta := by
  have hM := scoreMargin_bounds μ {x | f0 x ≠ fa x} score hD
  obtain ⟨η, hη, he⟩ := correctness_fibre_abs_residual_attained μ _ hD
    (scoreMargin μ {x | f0 x ≠ fa x} score) beta hM.1 hM.2 hb0 hb1
  refine ⟨fieldWorld μ f0 fa h0 ha κ0 η, ⟨fieldWorld_fst μ f0 fa h0 ha κ0 η, ?_⟩, ?_⟩
  · rw [fieldWorld_residual μ f0 fa h0 ha κ0 score η hD]
    exact hη
  · rw [fieldWorld_residual μ f0 fa h0 ha κ0 score η hD]
    exact he

theorem actual_class_radius (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ)
    (hb0 : 0 ≤ beta) (hb1 : beta ≤ 1 / 2) :
    fibreRadius {P | admissible μ f0 fa ha score beta P}
      (fun P => scoreResidual (P : Measure (X × Bool)) f0 fa ha score) = beta := by
  obtain ⟨P, hP, he⟩ := actual_abs_residual_attained μ f0 fa h0 ha κ0 score hD beta hb0 hb1
  exact fibreRadius_eq_of_bound_and_witness _ _ beta (fun Q => Q.property.2)
    ⟨⟨P, hP⟩, he⟩

theorem actual_augmented_fibre_eq_class {Z : Type*} [MeasurableSpace Z]
    (μ : Measure X) (f0 fa : X → Bool) (ha : Measurable fa)
    (score : CorrectnessField X) (beta : ℝ)
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (Q : ProbabilityMeasure (X × Bool)) (hQ : (Q : Measure (X × Bool)).fst = μ) :
    {P | admissible μ f0 fa ha score beta P ∧
        augmentedLaw observation f0 fa score P = augmentedLaw observation f0 fa score Q} =
      {P | admissible μ f0 fa ha score beta P} := by
  ext P
  constructor
  · exact fun h => h.1
  · intro hP
    exact ⟨hP, augmentedLaw_eq observation f0 fa score (hP.1.trans hQ.symm)⟩

theorem actual_augmented_fibre_radius {Z : Type*} [MeasurableSpace Z]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (Q : ProbabilityMeasure (X × Bool)) (hQ : (Q : Measure (X × Bool)).fst = μ)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ)
    (hb0 : 0 ≤ beta) (hb1 : beta ≤ 1 / 2) :
    fibreRadius {P | admissible μ f0 fa ha score beta P ∧
        augmentedLaw observation f0 fa score P = augmentedLaw observation f0 fa score Q}
      (fun P => scoreResidual (P : Measure (X × Bool)) f0 fa ha score) = beta := by
  rw [actual_augmented_fibre_eq_class μ f0 fa ha score beta observation Q hQ]
  exact actual_class_radius μ f0 fa h0 ha κ0 score hD beta hb0 hb1

end KBound.ActualFibreRadius

#print axioms KBound.ActualFibreRadius.actual_abs_residual_attained
#print axioms KBound.ActualFibreRadius.actual_class_radius
#print axioms KBound.ActualFibreRadius.actual_augmented_fibre_eq_class
#print axioms KBound.ActualFibreRadius.actual_augmented_fibre_radius
