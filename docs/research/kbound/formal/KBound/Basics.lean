import Mathlib.Data.Real.Basic

/-!
# Shared K-Bound definitions

Paper labels: adapt / freeze / abstain trichotomy used throughout the theory spine.
-/

namespace KBound

inductive Decision where
  | adapt
  | freeze
  | abstain
  deriving DecidableEq, Repr

open Decision

/-- Sign of a real benefit; zero is treated as neither positive nor negative. -/
noncomputable def signReal (x : ℝ) : ℤ :=
  if 0 < x then 1 else if x < 0 then -1 else 0

theorem signReal_pos {x : ℝ} (hx : 0 < x) : signReal x = 1 := by
  simp [signReal, hx]

theorem signReal_neg {x : ℝ} (hx : x < 0) : signReal x = -1 := by
  simp [signReal, hx, not_lt.mpr (le_of_lt hx)]

theorem signReal_mul_pos {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    signReal (a * b) = signReal a := by
  rw [signReal_pos (mul_pos ha hb), signReal_pos ha]

theorem signReal_mul_neg_left {a b : ℝ} (ha : a < 0) (hb : 0 < b) :
    signReal (a * b) = -1 :=
  signReal_neg (mul_neg_of_neg_of_pos ha hb)

theorem signReal_pos_mul {c x : ℝ} (hc : 0 < c) :
    signReal (c * x) = signReal x := by
  by_cases hxpos : 0 < x
  · rw [signReal_pos (mul_pos hc hxpos), signReal_pos hxpos]
  · by_cases hxneg : x < 0
    · rw [signReal_neg (mul_neg_of_pos_of_neg hc hxneg), signReal_neg hxneg]
    · have hx0 : x = 0 := le_antisymm (le_of_not_gt hxpos) (le_of_not_gt hxneg)
      simp [hx0, signReal]

theorem signReal_mul_pos_right {x c : ℝ} (hc : 0 < c) :
    signReal (x * c) = signReal x := by
  rw [mul_comm, signReal_pos_mul hc]

end KBound
