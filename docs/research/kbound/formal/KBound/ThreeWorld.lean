import KBound.Basics
import KBound.Disagreement
import Mathlib.Tactic.Linarith

/-!
# Three-world exact minimax constant (`thm:t1c-exact`)

Algebraic core: for `μ_D > 0`, multiclass harm `p_a ≤ p_0` is equivalent to `Δ ≤ 0`.
This module records the finite algebraic identifiability link. A full Gaussian
sample-complexity layer is outside the short-paper mechanization.
-/

namespace KBound

/-- Multiclass false-harm (`p_a ≤ p_0` on `D`) matches non-positive benefit when `μ_D > 0`. -/
theorem multiclass_harm_iff_nonpos {muD pa p0 : ℝ} (hmu : 0 < muD) :
    pa ≤ p0 ↔ multiclassBenefit muD pa p0 ≤ 0 := by
  unfold multiclassBenefit
  constructor
  · intro h
    nlinarith
  · intro h
    nlinarith [hmu]

/-- Strict improvement on `D` implies strictly positive multiclass benefit. -/
theorem multiclass_benefit_pos_of_pa_gt {muD pa p0 : ℝ} (hmu : 0 < muD) (h : p0 < pa) :
    0 < multiclassBenefit muD pa p0 := by
  unfold multiclassBenefit
  nlinarith

end KBound
