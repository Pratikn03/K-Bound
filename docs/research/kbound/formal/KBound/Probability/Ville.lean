import KBound.Probability.EProcess
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp

/-!
# Discrete e-process / Ville layer (`thm:anytime`)

Paper: anytime-valid false-adapt control via a one-sided betting supermartingale
and Ville's inequality (Markov form).
-/

namespace KBound

/-- Null-update supermartingale step (paper `thm:anytime` one-step core). -/
theorem betting_wealth_supermartingale_step (w lam x : ℝ)
    (hw : 0 ≤ w) (hl : 0 ≤ lam) (hx : x ≤ 0) (hb : lam * x ≥ -1) :
    w * bettingFactor lam x ≤ w :=
  betting_wealth_step_le w lam x hw hl hx hb

/-- **Discrete Ville / Markov bound** (pointwise core of the anytime certificate).
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
