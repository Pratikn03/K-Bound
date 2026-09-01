import MulticlassVectorCapacity.Benefit

namespace MulticlassVectorCapacity

open scoped BigOperators

variable {S Y J R : Type*} [Fintype S] [Fintype Y]

/-- Linear moments over finite strata/classes, indexed by `R`; the finite-row setting is a
specialization. Scientific availability of rows is a separate protocol assumption. -/
structure ObservableOperator (S Y R : Type*) where
  coeff : R → S → Y → ℝ

def observe (A : ObservableOperator S Y R) (η : ConditionalLabels S Y) : R → ℝ :=
  fun r => ∑ s, ∑ y, A.coeff r s y * η.prob s y

/-- Simplex normalization is intrinsic to the domain, not an omitted linear restriction. -/
def fiber (A : ObservableOperator S Y R) (b : R → ℝ) : Set (ConditionalLabels S Y) :=
  {η | observe A η = b}

def observablyEquivalent (A : ObservableOperator S Y R)
    (η ξ : ConditionalLabels S Y) : Prop := observe A η = observe A ξ

def identifiedBenefits (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) : Set ℝ := (fun η => benefit M η j) '' fiber A b

/-- Point identification concerns all feasible worlds at this realized observable value. -/
def pointIdentified (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) : Prop :=
  (fiber A b).Nonempty ∧
    ∀ η ∈ fiber A b, ∀ ξ ∈ fiber A b, benefit M η j = benefit M ξ j

theorem same_fiber_observably_equivalent (A : ObservableOperator S Y R) (b : R → ℝ)
    {η ξ : ConditionalLabels S Y} (hη : η ∈ fiber A b) (hξ : ξ ∈ fiber A b) :
    observablyEquivalent A η ξ := by
  exact hη.trans hξ.symm

theorem identifiedBenefits_nonempty (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) (h : (fiber A b).Nonempty) :
    (identifiedBenefits M A b j).Nonempty := by
  obtain ⟨η, hη⟩ := h
  exact ⟨benefit M η j, η, hη, rfl⟩

theorem identifiedBenefits_bddBelow (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) : BddBelow (identifiedBenefits M A b j) := by
  refine ⟨-1, ?_⟩
  rintro _ ⟨η, _, rfl⟩
  exact (benefit_mem_Icc M η j).1

theorem identifiedBenefits_bddAbove (M : Model S Y J) (A : ObservableOperator S Y R)
    (b : R → ℝ) (j : J) : BddAbove (identifiedBenefits M A b j) := by
  refine ⟨1, ?_⟩
  rintro _ ⟨η, _, rfl⟩
  exact (benefit_mem_Icc M η j).2

end MulticlassVectorCapacity
