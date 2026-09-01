import MulticlassVectorCapacity.Examples

namespace MulticlassVectorCapacity.EdgeCases

open scoped BigOperators

def zeroCostModel : Model (Fin 1) (Fin 3) (Fin 1) where
  q := fun _ => 1
  q_nonneg := by intro _; norm_num
  q_sum := by norm_num [Fin.sum_univ_succ]
  cost := fun _ _ _ => 0
  cost_nonneg := by intros; norm_num
  cost_le_one := by intros; norm_num

theorem zero_cost_benefit (η : ConditionalLabels (Fin 1) (Fin 3)) :
    benefit zeroCostModel η 0 = 0 := by
  simp [benefit, costContrast, zeroCostModel]

theorem zero_cost_identified_set :
    identifiedBenefits zeroCostModel ThreeClass.observable ThreeClass.observedValue 0 = {0} := by
  ext z
  constructor
  · rintro ⟨η, _, rfl⟩
    simp [zero_cost_benefit]
  · intro hz
    have hz0 : z = 0 := hz
    subst z
    obtain ⟨η, hη⟩ := ThreeClass.fiber_nonempty
    exact ⟨η, hη, zero_cost_benefit η⟩

theorem zero_cost_boundary_abstains :
    fiberDecision zeroCostModel ThreeClass.observable ThreeClass.observedValue 0 = some .abstain := by
  apply (abstain_decision_iff zeroCostModel ThreeClass.observable ThreeClass.observedValue 0).mpr
  refine ⟨ThreeClass.fiber_nonempty, ?_, ?_⟩
  · simp [lowerBenefit, zero_cost_identified_set]
  · simp [upperBenefit, zero_cost_identified_set]

def zeroObservable : ObservableOperator (Fin 1) (Fin 3) (Fin 1) where
  coeff := fun _ _ _ => 0

def impossibleValue : Fin 1 → ℝ := fun _ => 1

theorem inconsistent_fiber_empty : ¬ (fiber zeroObservable impossibleValue).Nonempty := by
  rintro ⟨η, hη⟩
  have h0 := congrFun hη 0
  norm_num [observe, zeroObservable, impossibleValue] at h0

theorem inconsistent_fiber_rejected :
    fiberDecision ThreeClass.model zeroObservable impossibleValue 0 = none := by
  exact empty_fiber_no_certificate _ _ _ _ inconsistent_fiber_empty

theorem negative_conditional_coordinate_rejected :
    ¬ ∃ η : ConditionalLabels (Fin 1) (Fin 3), η.prob 0 1 = -1 / 5 := by
  rintro ⟨η, hη⟩
  have h := η.nonneg 0 1
  rw [hη] at h
  norm_num at h

theorem unnormalized_simplex_rejected :
    ¬ ∃ η : ConditionalLabels (Fin 1) (Fin 3), ∀ y, η.prob 0 y = 1 := by
  rintro ⟨η, hη⟩
  have h := η.sum_one 0
  norm_num [hη, Fin.sum_univ_succ] at h

end MulticlassVectorCapacity.EdgeCases
