import KBound.Basics
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Reduction to the disagreement region
-/

namespace KBound

noncomputable def binaryBenefit (muD aBar : ℝ) : ℝ :=
  2 * muD * (aBar - 1 / 2)

theorem binary_sign_reduction {muD aBar : ℝ} (hmu : 0 < muD) :
    signReal (binaryBenefit muD aBar) = signReal (aBar - 1 / 2) := by
  unfold binaryBenefit
  have hpos : 0 < 2 * muD := by nlinarith
  exact signReal_pos_mul hpos

theorem binary_margin_split (M gamma : ℝ) :
    M + gamma - 1 / 2 = (M - 1 / 2) + gamma := by ring

def multiclassBenefit (muD pa p0 : ℝ) : ℝ :=
  muD * (pa - p0)

theorem multiclass_sign_reduction {muD pa p0 : ℝ} (hmu : 0 < muD) :
    signReal (multiclassBenefit muD pa p0) = signReal (pa - p0) := by
  unfold multiclassBenefit
  exact signReal_pos_mul hmu

def regressionMoment (_f0fa f0faplusfa f0faY : ℝ) : ℝ :=
  f0faplusfa - 2 * f0faY

theorem regression_margin_split (M gamma : ℝ) :
    M + gamma = M + gamma := rfl

end KBound
