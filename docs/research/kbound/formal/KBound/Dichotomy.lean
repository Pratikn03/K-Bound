import KBound.Disagreement
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

/-!
# One-bit dichotomy algebraic core (`thm:conj1-dichotomy`)

Negating the accuracy coordinate flips the benefit sign on the disagreement region.
This is the finite-dimensional shadow of the evidence-preserving swap involution.
-/

namespace KBound

/-- Swapping accuracy to its complement negates binary benefit. -/
theorem binary_benefit_neg_accuracy (muD aBar : ℝ) :
    binaryBenefit muD (1 - aBar) = -binaryBenefit muD aBar := by
  unfold binaryBenefit
  ring

theorem signReal_neg_eq (x : ℝ) : signReal (-x) = -signReal x := by
  by_cases hx : 0 < x
  · have hnx : -x < 0 := neg_neg_of_pos hx
    simp [signReal, hx, hnx, not_lt.mpr (le_of_lt hx)]
  · by_cases hx0 : x < 0
    · have hnx : 0 < -x := neg_pos.mpr hx0
      simp [signReal, hx, hx0, hnx, not_lt.mpr (le_of_lt hnx)]
    · have hxz : x = 0 := le_antisymm (le_of_not_gt hx) (le_of_not_gt hx0)
      simp [hxz, signReal]

/-- Under positive mass on `D`, flipping accuracy flips the benefit sign. -/
theorem binary_sign_flip_on_accuracy_complement {muD aBar : ℝ} (_hmu : 0 < muD) :
    signReal (binaryBenefit muD (1 - aBar)) = -signReal (binaryBenefit muD aBar) := by
  rw [binary_benefit_neg_accuracy, signReal_neg_eq]

/-- Multiclass swap of accuracy coordinates flips benefit when `p0` and `pa` exchange roles. -/
theorem multiclass_benefit_swap_pa_p0 (muD pa p0 : ℝ) :
    multiclassBenefit muD p0 pa = -multiclassBenefit muD pa p0 := by
  unfold multiclassBenefit
  ring

end KBound
