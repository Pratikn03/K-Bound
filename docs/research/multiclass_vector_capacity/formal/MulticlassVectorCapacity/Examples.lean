import MulticlassVectorCapacity.SignCapacity
import Mathlib.Data.Fin.VecNotation

/-!
# Exact three-class counterexample

One stratum, one candidate, and the declared structural moment `η₀ = 1/5`.
The moment is hypothetical model knowledge, not something inferred from labels
by an unlabeled algorithm. The candidate is worse on class 0 and better on the
other classes. Its benefit ranges over `[1/5, 3/5]`: not point identified, but
uniformly positive. A surviving observable-null contrast need not cross zero.
-/

namespace MulticlassVectorCapacity.ThreeClass

open scoped BigOperators

noncomputable def model : Model (Fin 1) (Fin 3) (Fin 1) where
  q := fun _ => 1
  q_nonneg := by intro _; norm_num
  q_sum := by norm_num [Fin.sum_univ_succ]
  cost := fun _ y a => match a with
    | none => ![0, 1 / 2, 1] y
    | some _ => ![1, 0, 0] y
  cost_nonneg := by intro _ y a; cases a <;> fin_cases y <;> norm_num
  cost_le_one := by intro _ y a; cases a <;> fin_cases y <;> norm_num

def observable : ObservableOperator (Fin 1) (Fin 3) (Fin 1) where
  coeff := fun _ _ y => ![1, 0, 0] y

noncomputable def observedValue : Fin 1 → ℝ := fun _ => 1 / 5

noncomputable def labels (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 4 / 5) :
    ConditionalLabels (Fin 1) (Fin 3) where
  prob := fun _ => ![1 / 5, t, 4 / 5 - t]
  nonneg := by intro _ y; fin_cases y <;> dsimp <;> linarith
  sum_one := by intro _; simp [Fin.sum_univ_succ]; ring

theorem mem_fiber_iff (η : ConditionalLabels (Fin 1) (Fin 3)) :
    η ∈ fiber observable observedValue ↔ η.prob 0 0 = 1 / 5 := by
  constructor
  · intro h
    have h0 := congrFun h 0
    simpa [observe, observable, observedValue, Fin.sum_univ_succ] using h0
  · intro h
    funext r
    fin_cases r
    simpa [observe, observable, observedValue, Fin.sum_univ_succ] using h

theorem labels_mem_fiber (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 4 / 5) :
    labels t ht0 ht1 ∈ fiber observable observedValue := by
  apply (mem_fiber_iff _).mpr
  rfl

theorem benefit_formula (η : ConditionalLabels (Fin 1) (Fin 3)) :
    benefit model η 0 = -η.prob 0 0 + (1 / 2) * η.prob 0 1 + η.prob 0 2 := by
  simp [benefit, costContrast, model, Fin.sum_univ_succ]
  ring

theorem labels_benefit (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 4 / 5) :
    benefit model (labels t ht0 ht1) 0 = 3 / 5 - t / 2 := by
  rw [benefit_formula]
  simp [labels]
  ring

theorem fiber_benefit_bounds (η : ConditionalLabels (Fin 1) (Fin 3))
    (hη : η ∈ fiber observable observedValue) :
    benefit model η 0 ∈ Set.Icc (1 / 5) (3 / 5) := by
  have h0 := (mem_fiber_iff η).mp hη
  have h1 := η.nonneg 0 1
  have h2 := η.nonneg 0 2
  have hs : η.prob 0 0 + η.prob 0 1 + η.prob 0 2 = 1 := by
    simpa [Fin.sum_univ_succ, add_assoc] using η.sum_one 0
  rw [benefit_formula]
  constructor <;> linarith

/-- Exact attainable interval, not merely an outer numerical bound. -/
theorem identified_interval :
    identifiedBenefits model observable observedValue 0 = Set.Icc (1 / 5) (3 / 5) := by
  ext z
  constructor
  · rintro ⟨η, hη, rfl⟩
    exact fiber_benefit_bounds η hη
  · intro hz
    have ht0 : 0 ≤ (6 / 5 : ℝ) - 2 * z := by linarith [hz.2]
    have ht1 : (6 / 5 : ℝ) - 2 * z ≤ 4 / 5 := by linarith [hz.1]
    refine ⟨labels (6 / 5 - 2 * z) ht0 ht1,
      labels_mem_fiber (6 / 5 - 2 * z) ht0 ht1, ?_⟩
    change benefit model (labels (6 / 5 - 2 * z) ht0 ht1) 0 = z
    rw [labels_benefit]
    ring

theorem fiber_nonempty : (fiber observable observedValue).Nonempty := by
  exact ⟨labels 0 (by norm_num) (by norm_num), labels_mem_fiber 0 (by norm_num) (by norm_num)⟩

theorem lowerBenefit_eq : lowerBenefit model observable observedValue 0 = 1 / 5 := by
  unfold lowerBenefit
  rw [identified_interval]
  exact csInf_Icc (by norm_num)

theorem upperBenefit_eq : upperBenefit model observable observedValue 0 = 3 / 5 := by
  unfold upperBenefit
  rw [identified_interval]
  exact csSup_Icc (by norm_num)

theorem not_point_identified : ¬ pointIdentified model observable observedValue 0 := by
  rintro ⟨_, h⟩
  have heq := h (labels 0 (by norm_num) (by norm_num))
    (labels_mem_fiber 0 (by norm_num) (by norm_num))
    (labels (4 / 5) (by norm_num) (by norm_num))
    (labels_mem_fiber (4 / 5) (by norm_num) (by norm_num))
  rw [labels_benefit, labels_benefit] at heq
  norm_num at heq

theorem strict_adapt : fiberDecision model observable observedValue 0 = some .adapt := by
  apply (adapt_decision_iff model observable observedValue 0).mpr
  exact ⟨fiber_nonempty, by rw [lowerBenefit_eq]; norm_num⟩

/-- The candidate is not pointwise dominant: its class-0 cost is strictly higher. -/
theorem candidate_not_pointwise_dominant : model.cost 0 0 none < model.cost 0 0 (some 0) := by
  norm_num [model]

def nullVariation : Fin 1 → Fin 3 → ℝ := fun _ => ![0, 1, -1]

theorem nullVariation_simplex_tangent (s : Fin 1) : ∑ y, nullVariation s y = 0 := by
  norm_num [nullVariation, Fin.sum_univ_succ]

theorem nullVariation_observable_zero (r : Fin 1) :
    (∑ s, ∑ y, observable.coeff r s y * nullVariation s y) = 0 := by
  norm_num [observable, nullVariation, Fin.sum_univ_succ]

theorem nullVariation_changes_benefit :
    (∑ s, model.q s * ∑ y, costContrast model 0 s y * nullVariation s y) = -1 / 2 := by
  norm_num [model, costContrast, nullVariation, Fin.sum_univ_succ]

theorem no_negative_world :
    ¬ ∃ η ∈ fiber observable observedValue, benefit model η 0 < 0 := by
  rintro ⟨η, hη, hneg⟩
  have hlo := (fiber_benefit_bounds η hη).1
  linarith

/-- Concrete falsification of “surviving null contrast implies opposite strict signs.” -/
theorem surviving_null_contrast_without_sign_ambiguity :
    (∑ s, model.q s * ∑ y, costContrast model 0 s y * nullVariation s y) ≠ 0 ∧
    (∀ r, (∑ s, ∑ y, observable.coeff r s y * nullVariation s y) = 0) ∧
    (∀ s, ∑ y, nullVariation s y = 0) ∧
    (fiber observable observedValue).Nonempty ∧
    (∀ η ∈ fiber observable observedValue, 0 < benefit model η 0) := by
  refine ⟨?_, nullVariation_observable_zero, nullVariation_simplex_tangent,
    fiber_nonempty, ?_⟩
  · rw [nullVariation_changes_benefit]
    norm_num
  · intro η hη
    have hlo := (fiber_benefit_bounds η hη).1
    linarith

end MulticlassVectorCapacity.ThreeClass
