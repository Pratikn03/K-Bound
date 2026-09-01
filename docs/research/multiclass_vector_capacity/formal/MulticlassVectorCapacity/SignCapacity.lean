import MulticlassVectorCapacity.ObservableFiber
import Mathlib.Data.Real.Archimedean

/-!
# Realized-fiber strict decisions

These are fixed-candidate, finite-model soundness results, not uniform rank
identification or statistical false-commit theorems. Empty fibers return `none`.
For nonempty fibers the controller has the three actions ADAPT/FREEZE/ABSTAIN.
-/

namespace MulticlassVectorCapacity

variable {S Y J R : Type*} [Fintype S] [Fintype Y]

noncomputable def lowerBenefit (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) : ℝ := sInf (identifiedBenefits M A b j)

noncomputable def upperBenefit (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) : ℝ := sSup (identifiedBenefits M A b j)

inductive StrictAction where
  | adapt
  | freeze
  | abstain
  deriving DecidableEq, Repr

/-- `none` is an inconsistent-observable protocol failure, not a fourth commitment. -/
noncomputable def fiberDecision (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) : Option StrictAction := by
  classical
  exact if (fiber A b).Nonempty then
    if 0 < lowerBenefit M A b j then some .adapt
    else if upperBenefit M A b j < 0 then some .freeze
    else some .abstain
  else none

theorem lowerBenefit_le_benefit (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) {η : ConditionalLabels S Y} (hη : η ∈ fiber A b) :
    lowerBenefit M A b j ≤ benefit M η j := by
  exact csInf_le (identifiedBenefits_bddBelow M A b j) ⟨η, hη, rfl⟩

theorem benefit_le_upperBenefit (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) {η : ConditionalLabels S Y} (hη : η ∈ fiber A b) :
    benefit M η j ≤ upperBenefit M A b j := by
  exact le_csSup (identifiedBenefits_bddAbove M A b j) ⟨η, hη, rfl⟩

theorem lowerBenefit_le_upperBenefit (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) (h : (fiber A b).Nonempty) :
    lowerBenefit M A b j ≤ upperBenefit M A b j := by
  obtain ⟨η, hη⟩ := h
  exact (lowerBenefit_le_benefit M A b j hη).trans (benefit_le_upperBenefit M A b j hη)

/-- A strictly positive infimum is equivalent to a positive uniform margin on this fiber. -/
theorem lowerBenefit_pos_iff_uniform_margin (M : Model S Y J)
    (A : ObservableOperator S Y R) (b : R → ℝ) (j : J)
    (h : (fiber A b).Nonempty) :
    0 < lowerBenefit M A b j ↔
      ∃ ε : ℝ, 0 < ε ∧ ∀ η ∈ fiber A b, ε ≤ benefit M η j := by
  constructor
  · intro hl
    exact ⟨lowerBenefit M A b j, hl, fun _ hη => lowerBenefit_le_benefit M A b j hη⟩
  · rintro ⟨ε, hε, hb⟩
    apply lt_of_lt_of_le hε
    apply le_csInf (identifiedBenefits_nonempty M A b j h)
    rintro _ ⟨η, hη, rfl⟩
    exact hb η hη

theorem upperBenefit_neg_iff_uniform_margin (M : Model S Y J)
    (A : ObservableOperator S Y R) (b : R → ℝ) (j : J)
    (h : (fiber A b).Nonempty) :
    upperBenefit M A b j < 0 ↔
      ∃ ε : ℝ, 0 < ε ∧ ∀ η ∈ fiber A b, benefit M η j ≤ -ε := by
  constructor
  · intro hu
    refine ⟨-upperBenefit M A b j, neg_pos.mpr hu, ?_⟩
    simpa only [neg_neg] using fun η hη => benefit_le_upperBenefit M A b j (η := η) hη
  · rintro ⟨ε, hε, hb⟩
    have hu : upperBenefit M A b j ≤ -ε := by
      apply csSup_le (identifiedBenefits_nonempty M A b j h)
      rintro _ ⟨η, hη, rfl⟩
      exact hb η hη
    linarith

theorem empty_fiber_no_certificate (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) (h : ¬ (fiber A b).Nonempty) :
    fiberDecision M A b j = none := by
  simp [fiberDecision, h]

theorem adapt_decision_iff (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) :
    fiberDecision M A b j = some .adapt ↔
      (fiber A b).Nonempty ∧ 0 < lowerBenefit M A b j := by
  classical
  by_cases hf : (fiber A b).Nonempty
  · by_cases hl : 0 < lowerBenefit M A b j
    · simp [fiberDecision, hf, hl]
    · by_cases hu : upperBenefit M A b j < 0 <;> simp [fiberDecision, hf, hl, hu]
  · simp [fiberDecision, hf]

theorem freeze_decision_iff (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) :
    fiberDecision M A b j = some .freeze ↔
      (fiber A b).Nonempty ∧ upperBenefit M A b j < 0 := by
  classical
  by_cases hf : (fiber A b).Nonempty
  · have hlu := lowerBenefit_le_upperBenefit M A b j hf
    by_cases hu : upperBenefit M A b j < 0
    · have hl : ¬ 0 < lowerBenefit M A b j := by linarith
      simp [fiberDecision, hf, hl, hu]
    · by_cases hl : 0 < lowerBenefit M A b j <;> simp [fiberDecision, hf, hl, hu]
  · simp [fiberDecision, hf]

/-- T4 soundness: a strict ADAPT certificate lowers cost in every world of the realized fiber. -/
theorem adapt_decision_sound (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) (h : fiberDecision M A b j = some .adapt) :
    (fiber A b).Nonempty ∧ ∀ η ∈ fiber A b,
      expectedCost M η (some j) < expectedCost M η none := by
  obtain ⟨hne, hl⟩ := (adapt_decision_iff M A b j).mp h
  refine ⟨hne, fun η hη => ?_⟩
  apply (positive_benefit_iff_lower_cost M η j).mp
  exact hl.trans_le (lowerBenefit_le_benefit M A b j hη)

/-- T4 soundness: a strict FREEZE certificate lowers cost in every world of the realized fiber. -/
theorem freeze_decision_sound (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) (h : fiberDecision M A b j = some .freeze) :
    (fiber A b).Nonempty ∧ ∀ η ∈ fiber A b,
      expectedCost M η none < expectedCost M η (some j) := by
  obtain ⟨hne, hu⟩ := (freeze_decision_iff M A b j).mp h
  refine ⟨hne, fun η hη => ?_⟩
  apply (negative_benefit_iff_higher_cost M η j).mp
  exact (benefit_le_upperBenefit M A b j hη).trans_lt hu

theorem abstain_decision_iff (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) :
    fiberDecision M A b j = some .abstain ↔
      (fiber A b).Nonempty ∧ lowerBenefit M A b j ≤ 0 ∧ 0 ≤ upperBenefit M A b j := by
  classical
  by_cases hf : (fiber A b).Nonempty
  · by_cases hl : 0 < lowerBenefit M A b j
    · simp [fiberDecision, hf, hl, not_le.mpr hl]
    · by_cases hu : upperBenefit M A b j < 0
      · simp [fiberDecision, hf, hl, hu, not_le.mpr hu]
      · simp [fiberDecision, hf, hl, hu, le_of_not_gt hl, le_of_not_gt hu]
  · simp [fiberDecision, hf]

end MulticlassVectorCapacity
