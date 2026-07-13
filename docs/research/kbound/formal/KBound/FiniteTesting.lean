import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Two-point total variation and Le Cam testing

Paper labels: `thm:imp-quant`, `prop:lecam-finite`.
-/

namespace KBound

def tvDist2 (p0 q0 : ℝ) : ℝ :=
  |p0 - q0|

def mixedError2 : Bool → ℝ → ℝ → ℝ
  | true, p0, q0 => q0 + (1 - p0)
  | false, p0, q0 => p0 + (1 - q0)

theorem mixedError2_true (p0 q0 : ℝ) :
    mixedError2 true p0 q0 = q0 + (1 - p0) := rfl

theorem mixedError2_false (p0 q0 : ℝ) :
    mixedError2 false p0 q0 = p0 + (1 - q0) := rfl

theorem mixedError2_min (p0 q0 : ℝ) :
    min (mixedError2 true p0 q0) (mixedError2 false p0 q0) = 1 - tvDist2 p0 q0 := by
  rw [mixedError2_true, mixedError2_false, tvDist2]
  rcases lt_trichotomy p0 q0 with h | h | h
  · have hneg : p0 - q0 < 0 := sub_neg.mpr h
    have habs : |p0 - q0| = q0 - p0 := by
      rw [abs_of_neg hneg]
      ring
    have e2 : p0 + (1 - q0) = 1 - (q0 - p0) := by ring
    have hmin : p0 + (1 - q0) ≤ q0 + (1 - p0) := by linarith
    rw [min_eq_right hmin, e2, habs]
  · simp [h]
  · have hpos : 0 < p0 - q0 := sub_pos.mpr h
    have habs : |p0 - q0| = p0 - q0 := abs_of_pos hpos
    have e2 : q0 + (1 - p0) = 1 - (p0 - q0) := by ring
    have hmin : q0 + (1 - p0) ≤ p0 + (1 - q0) := by linarith
    rw [min_eq_left hmin, e2, habs]

theorem lecam_testing_two_point (p0 q0 : ℝ) :
    min (mixedError2 true p0 q0) (mixedError2 false p0 q0) = 1 - tvDist2 p0 q0 :=
  mixedError2_min p0 q0

end KBound
