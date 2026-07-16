import KBound.Corollaries
import KBound.Conformal
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Positivity

/-!
# Finite-sample rate corollaries (`thm:ev-rate`, `cor:samplecomp`)

Connects a Hoeffding-style radius to the adapt/freeze commit rule. The classical
concentration inequality itself is the standard Hoeffding formula; the
kernel-checked content is the radius definition and the commit implication.
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

/-- Hoeffding radius for a mean of `n` bounded scores at level `α`
(order `√(log(2/α)/(2n))`). -/
noncomputable def hoeffdingRadius (n : ℕ) (alpha : ℝ) : ℝ :=
  Real.sqrt (Real.log (2 / alpha) / (2 * (n : ℝ)))

/-- **Hoeffding radius is nonnegative** for nontrivial `n` and `α ∈ (0,1]`. -/
theorem hoeffding_radius_le (n : ℕ) {alpha : ℝ}
    (_hn : 0 < n) (_hα : 0 < alpha) (_hα1 : alpha ≤ 1) :
    0 ≤ hoeffdingRadius n alpha := by
  unfold hoeffdingRadius
  exact Real.sqrt_nonneg _

/-- If a concentration radius covers the benefit and clears half the margin,
the certificate commits the correct strict action. -/
theorem rate_commit_from_concentration {dhat delta eps : ℝ}
    (hcov : |dhat - delta| ≤ eps) (hmargin : eps < |delta| / 2) :
    (0 < delta → certificate dhat eps = adapt) ∧
    (delta < 0 → certificate dhat eps = freeze) :=
  rate_implies_commit hcov hmargin

end KBound
