import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Finite conformal rank algebra

This module mechanizes the finite uniform-rank arithmetic behind split
conformal coverage.  It is not the full measure-theoretic exchangeability
theorem; that remains a probability-layer formalization task.
-/

namespace KBound

/-- Coverage mass of a threshold `k` when the test rank is uniform on
`1, ..., n+1`, in the unclipped case `k ≤ n+1`. -/
noncomputable def finiteUniformRankCoverage (n k : ℕ) : ℝ :=
  (k : ℝ) / ((n + 1 : ℕ) : ℝ)

/-- Miss mass complementary to `finiteUniformRankCoverage`. -/
noncomputable def finiteUniformRankMiss (n k : ℕ) : ℝ :=
  (((n + 1 : ℕ) : ℝ) - (k : ℝ)) / ((n + 1 : ℕ) : ℝ)

/-- The uniform-rank coverage and miss masses sum to one. -/
theorem finite_uniform_rank_coverage_add_miss (n k : ℕ) :
    finiteUniformRankCoverage n k + finiteUniformRankMiss n k = 1 := by
  unfold finiteUniformRankCoverage finiteUniformRankMiss
  have hpos : 0 < ((n + 1 : ℕ) : ℝ) := by exact_mod_cast Nat.succ_pos n
  have hden : ((n + 1 : ℕ) : ℝ) ≠ 0 := ne_of_gt hpos
  calc
    (k : ℝ) / ((n + 1 : ℕ) : ℝ) +
        ((((n + 1 : ℕ) : ℝ) - (k : ℝ)) / ((n + 1 : ℕ) : ℝ))
        = (((n + 1 : ℕ) : ℝ)) / ((n + 1 : ℕ) : ℝ) := by
          rw [← add_div]
          ring
    _ = 1 := div_self hden

/-- Equivalently, miss mass is `1 - coverage mass`. -/
theorem finite_uniform_rank_miss_eq_one_sub_coverage (n k : ℕ) :
    finiteUniformRankMiss n k = 1 - finiteUniformRankCoverage n k := by
  have h := finite_uniform_rank_coverage_add_miss n k
  linarith

/-- If the conformal threshold uses at least `(1-α)(n+1)` ranks, then the
finite uniform-rank miss mass is at most `α`. -/
theorem finite_uniform_rank_miss_le_alpha {n k : ℕ} {alpha : ℝ}
    (hthreshold : (1 - alpha) * (((n + 1 : ℕ) : ℝ)) ≤ (k : ℝ)) :
    finiteUniformRankMiss n k ≤ alpha := by
  unfold finiteUniformRankMiss
  have hpos : 0 < ((n + 1 : ℕ) : ℝ) := by exact_mod_cast Nat.succ_pos n
  have hmiss : (((n + 1 : ℕ) : ℝ) - (k : ℝ)) ≤ alpha * (((n + 1 : ℕ) : ℝ)) := by
    linarith
  exact (div_le_iff₀ hpos).mpr hmiss

end KBound
