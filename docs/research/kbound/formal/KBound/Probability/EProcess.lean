import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

/-!
# Discrete betting e-process core (`thm:evalue`, `thm:anytime`)
-/

namespace KBound

def bettingFactor (lam x : ℝ) : ℝ := 1 + lam * x

theorem bettingFactor_pos (lam x a : ℝ) (hl : 0 ≤ lam) (hx : a ≤ x) (ha : lam * a > -1) :
    0 < bettingFactor lam x := by
  unfold bettingFactor
  have h1 : lam * a ≤ lam * x := mul_le_mul_of_nonneg_left hx hl
  linarith

theorem bettingFactor_le_one (lam x : ℝ) (hl : 0 ≤ lam) (hx : x ≤ 0) (_hb : lam * x ≥ -1) :
    bettingFactor lam x ≤ 1 := by
  unfold bettingFactor
  have hlx : lam * x ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hl hx
  linarith

/-- Multiplicative wealth is non-increasing under a one-step null update. -/
theorem betting_wealth_step_le (w lam x : ℝ) (hw : 0 ≤ w) (hl : 0 ≤ lam) (hx : x ≤ 0)
    (hb : lam * x ≥ -1) :
    w * bettingFactor lam x ≤ w := by
  nlinarith [bettingFactor_le_one lam x hl hx hb]

end KBound
