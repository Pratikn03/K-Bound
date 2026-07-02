import KBound.Corollaries
import KBound.Conformal
import Mathlib.Data.Real.Basic

/-!
# Finite-sample rates (`thm:ev-rate`, `cor:samplecomp`)
-/

namespace KBound

open Decision

theorem rate_implies_commit {dhat delta eps : ℝ}
    (hcov : |dhat - delta| ≤ eps) (hmargin : eps < |delta| / 2) :
    (0 < delta → certificate dhat eps = adapt) ∧
    (delta < 0 → certificate dhat eps = freeze) :=
  two_sided_sign_certified hcov hmargin

theorem rate_conformal_miss {n k : ℕ} {alpha : ℝ}
    (h : (1 - alpha) * (((n + 1 : ℕ) : ℝ)) ≤ (k : ℝ)) :
    finiteUniformRankMiss n k ≤ alpha :=
  finite_uniform_rank_miss_le_alpha h

end KBound
