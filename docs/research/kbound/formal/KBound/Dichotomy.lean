import KBound.Disagreement
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# One-bit dichotomy + evidence-preserving swap involution (`thm:conj1-dichotomy`)

Negating the accuracy coordinate flips the benefit sign on the disagreement region.
The `EvidenceState` swap map is the finite-dimensional evidence-preserving involution
used by the paper's one-bit dichotomy: label-free score evidence is fixed while the
latent accuracy bit is flipped.
-/

namespace KBound

/-- Label-free / latent coordinates for the one-bit swap construction. -/
structure EvidenceState where
  /-- Mass on the disagreement region. -/
  muD : ℝ
  /-- Latent accuracy on the disagreement region. -/
  aBar : ℝ
  /-- Label-free score mean (observable evidence). -/
  scoreMean : ℝ

/-- Evidence-preserving swap: flip latent accuracy, keep label-free evidence. -/
def evidenceSwap (s : EvidenceState) : EvidenceState :=
  { muD := s.muD
    aBar := 1 - s.aBar
    scoreMean := s.scoreMean }

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
      simp [signReal, hx, hx0, hnx]
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

/-- The swap map is an involution. -/
theorem evidence_swap_involution (s : EvidenceState) :
    evidenceSwap (evidenceSwap s) = s := by
  cases s
  simp [evidenceSwap]

/-- Swap flips benefit sign while preserving label-free evidence coordinates. -/
theorem swap_flips_benefit_preserves_evidence (s : EvidenceState) :
    binaryBenefit s.muD (evidenceSwap s).aBar = -binaryBenefit s.muD s.aBar
      ∧ (evidenceSwap s).scoreMean = s.scoreMean
      ∧ (evidenceSwap s).muD = s.muD := by
  refine ⟨?_, rfl, rfl⟩
  simpa [evidenceSwap] using binary_benefit_neg_accuracy s.muD s.aBar

/-- Under positive disagreement mass, the swap flips the benefit *sign*. -/
theorem swap_flips_sign_of_benefit {s : EvidenceState} (hmu : 0 < s.muD) :
    signReal (binaryBenefit s.muD (evidenceSwap s).aBar)
      = -signReal (binaryBenefit s.muD s.aBar) := by
  simpa [evidenceSwap] using
    binary_sign_flip_on_accuracy_complement (muD := s.muD) (aBar := s.aBar) hmu

end KBound
