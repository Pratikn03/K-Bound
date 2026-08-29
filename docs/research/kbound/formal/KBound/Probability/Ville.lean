import KBound.Probability.EProcess
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp

/-!
# Pointwise betting / Markov layer used by the anytime argument

This file does not define a filtration, conditional expectation, nonnegative
supermartingale, maximal event, or stopping time. Consequently
`ville_bound_false_adapt` is the pointwise indicator inequality behind Markov's
inequality, not a full formalization of Ville's maximal inequality or optional
stopping. The manuscript-level anytime theorem still relies on those external
probability results.
-/

namespace KBound

/-- Deterministic null-update wealth step (not a conditional-expectation theorem). -/
theorem betting_wealth_supermartingale_step (w lam x : ℝ)
    (hw : 0 ≤ w) (hl : 0 ≤ lam) (hx : x ≤ 0) (hb : lam * x ≥ -1) :
    w * bettingFactor lam x ≤ w :=
  betting_wealth_step_le w lam x hw hl hx hb

/-- **Pointwise Markov indicator bound** (algebraic core used inside Ville's theorem).
For a nonnegative weight `W` and level `0 < α`,
`1{W ≥ 1/α} ≤ α · W`. -/
theorem ville_bound_false_adapt (W alpha : ℝ)
    (hW : 0 ≤ W) (hα : 0 < alpha) :
    (if 1 / alpha ≤ W then (1 : ℝ) else 0) ≤ alpha * W := by
  split_ifs with h
  · -- 1/α ≤ W ⇒ 1 ≤ α W
    have hαne : alpha ≠ 0 := hα.ne'
    have : alpha * (1 / alpha) ≤ alpha * W :=
      mul_le_mul_of_nonneg_left h hα.le
    have h1 : alpha * (1 / alpha) = 1 := by field_simp [hαne]
    rw [h1] at this
    exact this
  · exact mul_nonneg hα.le hW

end KBound
