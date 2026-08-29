import KBound.Conformal
import KBound.Certificate
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

/-!
# Exchangeability bridge for split conformal coverage (`sec:t2-prop-conformal`)

This module starts after the ideal rank premise has been reduced to a finite
uniform-rank model. It does not derive rank super-uniformity from exchangeability
of arbitrary measurable scores or handle ties at that generality.
-/

namespace KBound

open Decision

/-- If the conformal rank threshold uses at least `(1-α)(n+1)` ranks, the finite
uniform-rank miss probability is at most `α`. This is the algebraic core of exchangeable
split conformal coverage once ranks are stochastically dominated by the ideal uniform model. -/
theorem exchangeable_conformal_miss_le_alpha {n k : ℕ} {alpha : ℝ}
    (hthreshold : (1 - alpha) * (((n + 1 : ℕ) : ℝ)) ≤ (k : ℝ)) :
    finiteUniformRankMiss n k ≤ alpha :=
  finite_uniform_rank_miss_le_alpha hthreshold

/-- On the coverage event, false-adapt is impossible (links conformal coverage to `thm:cert`). -/
theorem exchangeable_cert_false_adapt_sound {dhat delta eps : ℝ}
    (hcov : |dhat - delta| ≤ eps)
    (hcert : certificate dhat eps = adapt) :
    0 < delta :=
  cert_false_adapt_sound hcov hcert

end KBound
