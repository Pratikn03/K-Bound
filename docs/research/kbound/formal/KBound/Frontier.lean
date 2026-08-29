import KBound.Basics
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
-- `frontier_band_zero_witness` discharges `M + -M = 0` with `ring`.  Older Mathlib
-- re-exported `ring` transitively through Linarith; current Mathlib does not, so the
-- tactic must be imported explicitly.  No proof term is changed by this import.
import Mathlib.Tactic.Ring

/-!
# Exact benefit-sign frontier (`thm:frontier`)

Paper: `paper/sections/main_theory_5.tex`, Theorem `thm:frontier`.
-/

namespace KBound

open Decision

/-- Paper frontier committal rule on observable margin `M` and budget `β`. -/
noncomputable def frontierDecision (M beta : ℝ) : Decision :=
  if M > beta then
    adapt
  else if M < -beta then
    freeze
  else
    abstain

theorem frontier_positive_raw {M gamma beta : ℝ}
    (hmargin : beta < M)
    (hdrift : |gamma| ≤ beta) :
    0 < M + gamma := by
  have hlow : -beta ≤ gamma := (abs_le.mp hdrift).1
  linarith

theorem frontier_negative_raw {M gamma beta : ℝ}
    (hmargin : beta < -M)
    (hdrift : |gamma| ≤ beta) :
    M + gamma < 0 := by
  have hup : gamma ≤ beta := (abs_le.mp hdrift).2
  linarith

theorem frontier_positive {M gamma beta : ℝ}
    (hM : 0 < M)
    (hfrontier : beta < |M|)
    (hdrift : |gamma| ≤ beta) :
    0 < M + gamma := by
  have hmargin : beta < M := by simpa [abs_of_pos hM] using hfrontier
  exact frontier_positive_raw hmargin hdrift

theorem frontier_negative {M gamma beta : ℝ}
    (hM : M < 0)
    (hfrontier : beta < |M|)
    (hdrift : |gamma| ≤ beta) :
    M + gamma < 0 := by
  have hmargin : beta < -M := by simpa [abs_of_neg hM] using hfrontier
  exact frontier_negative_raw hmargin hdrift

/-- Paper `thm:frontier` (ii): if `|M| > β`, sign is `sign M`. -/
theorem frontier_identifiable_positive {M gamma beta : ℝ}
    (hM : 0 < M) (hfrontier : beta < |M|) (hdrift : |gamma| ≤ beta) :
    signReal (M + gamma) = signReal M := by
  have hpos := frontier_positive hM hfrontier hdrift
  rw [signReal_pos hpos, signReal_pos hM]

theorem frontier_identifiable_negative {M gamma beta : ℝ}
    (hM : M < 0) (hfrontier : beta < |M|) (hdrift : |gamma| ≤ beta) :
    signReal (M + gamma) = signReal M := by
  have hneg := frontier_negative hM hfrontier hdrift
  rw [signReal_neg hneg, signReal_neg hM]

/-- The frontier rule commits adapt when `M > β`. -/
theorem frontier_decision_adapt {M beta : ℝ} (hM : beta < M) :
    frontierDecision M beta = adapt := by
  simp [frontierDecision, hM]

/-- The frontier rule commits freeze when `M < -β`. -/
theorem frontier_decision_freeze {M beta : ℝ} (hbeta : 0 ≤ beta) (hM : M < -beta) :
    frontierDecision M beta = freeze := by
  have hnot : ¬ beta < M := by linarith
  simp [frontierDecision, hnot, hM]

/-- Inside the band `|M| ≤ β`, the frontier rule abstains. -/
theorem frontier_decision_abstain {M beta : ℝ}
    (hleft : M ≤ beta) (hright : -beta ≤ M) :
    frontierDecision M beta = abstain := by
  simp [frontierDecision, not_lt.mpr hleft, not_lt.mpr hright]

/-- Every closed-band margin admits an allowed drift with exactly zero benefit.
This is the algebraic zero-versus-strict witness used by frontier necessity. -/
theorem frontier_band_zero_witness {M beta : ℝ} (hband : |M| ≤ beta) :
    ∃ gamma : ℝ, |gamma| ≤ beta ∧ M + gamma = 0 := by
  refine ⟨-M, ?_, by ring⟩
  simpa only [abs_neg] using hband

/-- In the open band, two allowed drift values yield opposite strict signs. -/
theorem frontier_open_band_opposite_witnesses {M beta : ℝ} (hband : |M| < beta) :
    ∃ gammaPos gammaNeg : ℝ,
      |gammaPos| ≤ beta ∧ |gammaNeg| ≤ beta ∧
      0 < M + gammaPos ∧ M + gammaNeg < 0 := by
  have hbeta : 0 < beta := lt_of_le_of_lt (abs_nonneg M) hband
  have hb := (abs_lt.mp hband)
  refine ⟨beta, -beta, ?_, ?_, ?_, ?_⟩
  · simp [abs_of_pos hbeta]
  · simp [abs_of_pos hbeta]
  · linarith
  · linarith

/-- At the positive boundary `M = β > 0`, an allowed drift realizes zero benefit
while another allowed drift realizes a strict positive benefit. -/
theorem frontier_positive_boundary_zero_strict {M beta : ℝ}
    (hbeta : 0 < beta) (hM : M = beta) :
    ∃ gammaZero gammaStrict : ℝ,
      |gammaZero| ≤ beta ∧ |gammaStrict| ≤ beta ∧
      M + gammaZero = 0 ∧ 0 < M + gammaStrict := by
  refine ⟨-beta, 0, ?_, ?_, ?_, ?_⟩
  · simp [abs_of_pos hbeta]
  · simp [le_of_lt hbeta]
  · linarith
  · linarith

/-- At the negative boundary `M = -β < 0`, allowed drifts realize zero and a
strict negative benefit. -/
theorem frontier_negative_boundary_zero_strict {M beta : ℝ}
    (hbeta : 0 < beta) (hM : M = -beta) :
    ∃ gammaZero gammaStrict : ℝ,
      |gammaZero| ≤ beta ∧ |gammaStrict| ≤ beta ∧
      M + gammaZero = 0 ∧ M + gammaStrict < 0 := by
  refine ⟨beta, 0, ?_, ?_, ?_, ?_⟩
  · simp [abs_of_pos hbeta]
  · simp [le_of_lt hbeta]
  · linarith
  · linarith

/-- β = 0 face used by ATC / DoC / GDE / COT / AETTA. -/
theorem zero_budget_positive_face {M gamma : ℝ}
    (hM : 0 < M) (hdrift : |gamma| ≤ 0) :
    0 < M + gamma :=
  frontier_positive hM (by simpa [abs_of_pos hM]) hdrift

theorem zero_budget_negative_face {M gamma : ℝ}
    (hM : M < 0) (hdrift : |gamma| ≤ 0) :
    M + gamma < 0 :=
  frontier_negative hM (by simpa [abs_of_neg hM]) hdrift

end KBound
