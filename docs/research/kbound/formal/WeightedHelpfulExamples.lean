import KBound

open KBound KBound.PaperDecisionAlgebra
open scoped BigOperators

-- Missing weighted composition must not be replaced by a pointwise assertion.
example {ι : Type*} (s : Finset ι) (w frozen adapted : ι → ℝ) (a : ι → Decision)
    (hw : ∀ i ∈ s, 0 ≤ w i) (hh : ∀ i ∈ s, frozen i ≤ adapted i) :
    ∑ i ∈ s, w i * servedScore (frozen i) (adapted i) (a i) ≤
      ∑ i ∈ s, w i * adapted i :=
  KBound.WeightedHelpful.helpful_weighted_sum s w frozen adapted a hw hh

example {ι : Type*} (s : Finset ι) (w frozen adapted : ι → ℝ) (a : ι → Decision)
    (hw : ∀ i ∈ s, 0 ≤ w i) (hh : ∀ i ∈ s, frozen i ≤ adapted i)
    (htotal : 0 < ∑ i ∈ s, w i) :
    (∑ i ∈ s, w i * servedScore (frozen i) (adapted i) (a i)) / (∑ i ∈ s, w i) ≤
      (∑ i ∈ s, w i * adapted i) / (∑ i ∈ s, w i) :=
  KBound.WeightedHelpful.helpful_weighted_mean s w frozen adapted a hw hh htotal

example {ι : Type*} (s : Finset ι) (w frozen adapted : ι → ℝ) (a : ι → Decision)
    (hw : ∀ i ∈ s, 0 ≤ w i) (hh : ∀ i ∈ s, frozen i ≤ adapted i)
    (htotal : 0 < ∑ i ∈ s, w i) :
    (∑ i ∈ s, w i * oracleRegret (frozen i) (adapted i) Decision.adapt) /
        (∑ i ∈ s, w i) ≤
      (∑ i ∈ s, w i * oracleRegret (frozen i) (adapted i) (a i)) / (∑ i ∈ s, w i) :=
  KBound.WeightedHelpful.helpful_weighted_regret_mean s w frozen adapted a hw hh htotal

-- Hand-computed mixed actions use all three choices and a zero-weight cell.
example :
    ((2 : ℝ) * servedScore 0 1 Decision.freeze +
        0 * servedScore 1 2 Decision.adapt + 1 * servedScore 2 3 Decision.abstain) / 3 = 2 / 3 ∧
      ((2 : ℝ) * 1 + 0 * 2 + 1 * 3) / 3 = 5 / 3 := by
  simp only [servedScore, reduceCtorEq, if_false, if_true]
  norm_num

-- Nonnegative weights are essential even when the total weight is positive.
example : (0 : ℝ) ≤ 1 ∧ 0 < (-1 : ℝ) + 2 ∧
    ((-1 : ℝ) * servedScore 0 1 Decision.freeze +
      2 * servedScore 0 1 Decision.adapt) / (-1 + 2) >
    ((-1 : ℝ) * 1 + 2 * 1) / (-1 + 2) := by
  simp only [servedScore, reduceCtorEq, if_false, if_true]
  norm_num

-- The unnormalized empty sum is covered without inventing a mean on no cells.
example (w frozen adapted : Unit → ℝ) (a : Unit → Decision) :
    ∑ i ∈ (∅ : Finset Unit), w i * servedScore (frozen i) (adapted i) (a i) ≤
      ∑ i ∈ (∅ : Finset Unit), w i * adapted i :=
  KBound.WeightedHelpful.helpful_weighted_sum ∅ w frozen adapted a (by simp) (by simp)
