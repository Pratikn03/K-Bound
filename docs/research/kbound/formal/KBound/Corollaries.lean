import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

import KBound.Certificate

/-!
# Split-conformal and sample-complexity corollaries
-/

namespace KBound

open Decision

/-- Paper `cor:samplecomp` (one-sided): on the coverage event, `ε < Δ/2` implies adapt. -/
theorem one_sided_commit_when_radius_small {dhat delta eps : ℝ}
    (hcov : |dhat - delta| ≤ eps)
    (hstrict : eps < delta / 2)
    (_hpos : 0 < delta) :
    certificate dhat eps = adapt := by
  unfold certificate
  have h1 : 0 < dhat - eps := by
    have hlower : delta - eps ≤ dhat := by linarith [(abs_le.mp hcov).1]
    linarith
  simp [h1]

/-- Two-sided sign certification needs `ε < |Δ|/2`. -/
theorem two_sided_sign_certified {dhat delta eps : ℝ}
    (hcov : |dhat - delta| ≤ eps)
    (hmargin : eps < |delta| / 2) :
    (0 < delta → certificate dhat eps = adapt) ∧
    (delta < 0 → certificate dhat eps = freeze) := by
  constructor
  · intro hpos
    have hhalf : eps < delta / 2 := by
      simpa [abs_of_pos hpos] using hmargin
    simpa using one_sided_commit_when_radius_small hcov hhalf hpos
  · intro hneg
    unfold certificate
    have hupper : dhat ≤ delta + eps := by linarith [(abs_le.mp hcov).2]
    have hhalf : eps < (-delta) / 2 := by simpa [abs_of_neg hneg] using hmargin
    have hfreeze : dhat + eps < 0 := by linarith
    have hpos : ¬ 0 < dhat - eps := by linarith
    simp [hpos, hfreeze]

end KBound
