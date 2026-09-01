import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Finite multiclass deployment primitives

`none` denotes the frozen policy and `some j` denotes candidate `j`.
The identities below are valid for every finite label type, hence also for
the research program's intended `3 ≤ Fintype.card Y` setting. No label-mass
lower bound is imposed: zero-mass strata are permitted.
-/

namespace MulticlassVectorCapacity

open scoped BigOperators

/-- Known target stratum masses and bounded, stratum-dependent policy costs. -/
structure Model (S Y J : Type*) [Fintype S] where
  q : S → ℝ
  q_nonneg : ∀ s, 0 ≤ q s
  q_sum : ∑ s, q s = 1
  cost : S → Y → Option J → ℝ
  cost_nonneg : ∀ s y a, 0 ≤ cost s y a
  cost_le_one : ∀ s y a, cost s y a ≤ 1

/-- Unknown target conditional label distributions, one simplex per stratum. -/
structure ConditionalLabels (S Y : Type*) [Fintype Y] where
  prob : S → Y → ℝ
  nonneg : ∀ s y, 0 ≤ prob s y
  sum_one : ∀ s, ∑ y, prob s y = 1

variable {S Y J : Type*} [Fintype S] [Fintype Y]

omit [Fintype Y] in
theorem stratum_mass_le_one (M : Model S Y J) (s : S) : M.q s ≤ 1 := by
  rw [← M.q_sum]
  exact Finset.single_le_sum (fun t _ => M.q_nonneg t) (Finset.mem_univ s)

omit [Fintype S] in
theorem conditional_probability_le_one (η : ConditionalLabels S Y) (s : S) (y : Y) :
    η.prob s y ≤ 1 := by
  rw [← η.sum_one s]
  exact Finset.single_le_sum (fun z _ => η.nonneg s z) (Finset.mem_univ y)

end MulticlassVectorCapacity
