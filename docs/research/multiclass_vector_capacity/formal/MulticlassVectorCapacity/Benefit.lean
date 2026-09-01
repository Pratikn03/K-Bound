import MulticlassVectorCapacity.Basic

namespace MulticlassVectorCapacity

open scoped BigOperators

variable {S Y J : Type*} [Fintype S] [Fintype Y]

/-- Population expected safety cost under a finite conditional label law. -/
def expectedCost (M : Model S Y J) (η : ConditionalLabels S Y) (a : Option J) : ℝ :=
  ∑ s, M.q s * ∑ y, M.cost s y a * η.prob s y

/-- Known per-label frozen-minus-candidate cost contrast. -/
def costContrast (M : Model S Y J) (j : J) (s : S) (y : Y) : ℝ :=
  M.cost s y none - M.cost s y (some j)

/-- Positive benefit means lower expected safety cost for the candidate. -/
def benefit (M : Model S Y J) (η : ConditionalLabels S Y) (j : J) : ℝ :=
  ∑ s, M.q s * ∑ y, costContrast M j s y * η.prob s y

/-- T1: exact finite cost-benefit identity, for every candidate simultaneously. -/
theorem cost_benefit_identity (M : Model S Y J) (η : ConditionalLabels S Y) (j : J) :
    expectedCost M η none - expectedCost M η (some j) = benefit M η j := by
  simp only [expectedCost, benefit, costContrast, sub_mul,
    Finset.sum_sub_distrib, mul_sub]

/-- Declared bounded costs and simplex masses imply actual expected-cost bounds. -/
theorem expectedCost_mem_unitInterval (M : Model S Y J) (η : ConditionalLabels S Y)
    (a : Option J) : 0 ≤ expectedCost M η a ∧ expectedCost M η a ≤ 1 := by
  constructor
  · exact Finset.sum_nonneg fun s _ => mul_nonneg (M.q_nonneg s)
      (Finset.sum_nonneg fun y _ => mul_nonneg (M.cost_nonneg s y a) (η.nonneg s y))
  · calc
      expectedCost M η a ≤ ∑ s, M.q s * ∑ y, η.prob s y := by
        apply Finset.sum_le_sum
        intro s _
        apply mul_le_mul_of_nonneg_left _ (M.q_nonneg s)
        apply Finset.sum_le_sum
        intro y _
        exact mul_le_of_le_one_left (η.nonneg s y) (M.cost_le_one s y a)
      _ = 1 := by simp [η.sum_one, M.q_sum]

theorem benefit_mem_Icc (M : Model S Y J) (η : ConditionalLabels S Y) (j : J) :
    benefit M η j ∈ Set.Icc (-1) 1 := by
  rw [← cost_benefit_identity]
  obtain ⟨hf0, hf1⟩ := expectedCost_mem_unitInterval M η none
  obtain ⟨ha0, ha1⟩ := expectedCost_mem_unitInterval M η (some j)
  constructor <;> linarith

theorem positive_benefit_iff_lower_cost (M : Model S Y J) (η : ConditionalLabels S Y)
    (j : J) : 0 < benefit M η j ↔ expectedCost M η (some j) < expectedCost M η none := by
  rw [← cost_benefit_identity]
  exact sub_pos

theorem negative_benefit_iff_higher_cost (M : Model S Y J) (η : ConditionalLabels S Y)
    (j : J) : benefit M η j < 0 ↔ expectedCost M η none < expectedCost M η (some j) := by
  rw [← cost_benefit_identity]
  exact sub_neg

end MulticlassVectorCapacity
