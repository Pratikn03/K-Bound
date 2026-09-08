import KBound.PaperDecisionAlgebra
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-! # Finite weighted composition of helpful-only routing

The same nonnegative cell weights are used for the routed and always-adapt
comparisons. A normalized mean explicitly has positive total weight. These are
deterministic consequences of per-cell helpfulness, not claims that any measured
panel satisfies that premise. FREEZE and ABSTAIN both serve the frozen score.
-/

namespace KBound.WeightedHelpful

open PaperDecisionAlgebra
open scoped BigOperators

theorem helpful_weighted_sum {ι : Type*} (s : Finset ι)
    (w frozen adapted : ι → ℝ) (a : ι → Decision)
    (hw : ∀ i ∈ s, 0 ≤ w i) (hh : ∀ i ∈ s, frozen i ≤ adapted i) :
    ∑ i ∈ s, w i * servedScore (frozen i) (adapted i) (a i) ≤
      ∑ i ∈ s, w i * adapted i := by
  apply Finset.sum_le_sum
  intro i hi
  exact mul_le_mul_of_nonneg_left (helpful_only_no_score_improvement (hh i hi) (a i)) (hw i hi)

theorem helpful_weighted_mean {ι : Type*} (s : Finset ι)
    (w frozen adapted : ι → ℝ) (a : ι → Decision)
    (hw : ∀ i ∈ s, 0 ≤ w i) (hh : ∀ i ∈ s, frozen i ≤ adapted i)
    (htotal : 0 < ∑ i ∈ s, w i) :
    (∑ i ∈ s, w i * servedScore (frozen i) (adapted i) (a i)) / (∑ i ∈ s, w i) ≤
      (∑ i ∈ s, w i * adapted i) / (∑ i ∈ s, w i) := by
  exact div_le_div_of_nonneg_right (helpful_weighted_sum s w frozen adapted a hw hh) htotal.le

theorem helpful_weighted_regret_mean {ι : Type*} (s : Finset ι)
    (w frozen adapted : ι → ℝ) (a : ι → Decision)
    (hw : ∀ i ∈ s, 0 ≤ w i) (hh : ∀ i ∈ s, frozen i ≤ adapted i)
    (htotal : 0 < ∑ i ∈ s, w i) :
    (∑ i ∈ s, w i * oracleRegret (frozen i) (adapted i) Decision.adapt) /
        (∑ i ∈ s, w i) ≤
      (∑ i ∈ s, w i * oracleRegret (frozen i) (adapted i) (a i)) / (∑ i ∈ s, w i) := by
  apply div_le_div_of_nonneg_right ?_ htotal.le
  apply Finset.sum_le_sum
  intro i hi
  apply mul_le_mul_of_nonneg_left ?_ (hw i hi)
  have hs := helpful_only_no_score_improvement (hh i hi) (a i)
  simpa only [oracleRegret, servedScore, if_true] using
    sub_le_sub_left hs (max (frozen i) (adapted i))

end KBound.WeightedHelpful

#print axioms KBound.WeightedHelpful.helpful_weighted_sum
#print axioms KBound.WeightedHelpful.helpful_weighted_mean
#print axioms KBound.WeightedHelpful.helpful_weighted_regret_mean
