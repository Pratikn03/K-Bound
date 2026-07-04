import Mathlib.Data.Real.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Order.MinMax
import Mathlib.Tactic.Linarith

/-!
# Rank counting for conformal coverage (`thm:certificate`, combinatorial core)

Paper: split-conformal coverage, rank step.  Mathlib (checked 2026-07-03 via Loogle)
has **no** order-statistic/rank counting lemma of this shape, so it is proved here
from scratch by a min-witness argument.

Main result: for any score vector `R : Fin m → ℝ` and any `k`, at most `m − k`
indices have strict rank at least `k`, where the strict rank of `j` counts the
indices with strictly smaller score.  Ties are handled automatically because the
rank is strict.  This is the deterministic heart of the `(n+1)α`-quantile coverage
bound; the probability layer is `KBound.Probability.UniformConformal`.
-/

namespace KBound

open Finset

/-- Strict rank of index `j` in the score vector `R`: how many indices carry a
strictly smaller score. -/
noncomputable def strictRank {m : ℕ} (R : Fin m → ℝ) (j : Fin m) : ℕ :=
  (Finset.univ.filter fun i => R i < R j).card

open scoped Classical in
/-- **Min-witness counting bound.**  At most `m − k` indices have strict rank `≥ k`:
take a minimizer `j⋆` of `R` over the high-rank set `S`; its `≥ k` strict witnesses
all have scores below every member of `S`, hence lie outside `S`, so `|Sᶜ| ≥ k`. -/
theorem card_high_strictRank_le {m : ℕ} (R : Fin m → ℝ) (k : ℕ) :
    (Finset.univ.filter fun j => k ≤ strictRank R j).card ≤ m - k := by
  by_cases hne : (Finset.univ.filter fun j => k ≤ strictRank R j).Nonempty
  · obtain ⟨jstar, hjS, hmin⟩ :=
      Finset.exists_min_image (Finset.univ.filter fun j => k ≤ strictRank R j) R hne
    -- the ≥ k strict witnesses of jstar
    have hWcard : k ≤ (Finset.univ.filter fun i => R i < R jstar).card :=
      (Finset.mem_filter.mp hjS).2
    -- witnesses avoid the high-rank set, by minimality of jstar
    have hWsub : (Finset.univ.filter fun i => R i < R jstar)
        ⊆ Finset.univ \ (Finset.univ.filter fun j => k ≤ strictRank R j) := by
      intro i hiW
      have hilt : R i < R jstar := (Finset.mem_filter.mp hiW).2
      refine Finset.mem_sdiff.mpr ⟨Finset.mem_univ i, fun hiS => ?_⟩
      exact absurd hilt (not_lt.mpr (hmin i hiS))
    have hcard1 : (Finset.univ.filter fun i => R i < R jstar).card
        ≤ (Finset.univ \ (Finset.univ.filter fun j => k ≤ strictRank R j)).card :=
      Finset.card_le_card hWsub
    have hcard2 : (Finset.univ \ (Finset.univ.filter fun j => k ≤ strictRank R j)).card
        = m - (Finset.univ.filter fun j => k ≤ strictRank R j).card := by
      rw [Finset.card_sdiff (Finset.subset_univ _), Finset.card_univ, Fintype.card_fin]
    have hle : (Finset.univ.filter fun j => k ≤ strictRank R j).card ≤ m := by
      calc (Finset.univ.filter fun j => k ≤ strictRank R j).card
          ≤ Fintype.card (Fin m) := Finset.card_le_univ _
        _ = m := Fintype.card_fin m
    omega
  · rw [Finset.not_nonempty_iff_eq_empty.mp hne]
    simp

/-- Complementary form: at least `k` indices have strict rank `< k` (equivalently,
score at most the `k`-th smallest).  Direct consequence of the counting bound. -/
theorem card_low_strictRank_ge {m : ℕ} (R : Fin m → ℝ) (k : ℕ) (hk : k ≤ m) :
    k ≤ (Finset.univ.filter fun j => strictRank R j < k).card := by
  classical
  have hsplit :
      (Finset.univ.filter fun j => strictRank R j < k).card
        + (Finset.univ.filter fun j => ¬ strictRank R j < k).card
        = Fintype.card (Fin m) := by
    simpa using Finset.filter_card_add_filter_neg_card_eq_card
      (s := (Finset.univ : Finset (Fin m))) (p := fun j => strictRank R j < k)
  have hhigh : (Finset.univ.filter fun j => ¬ strictRank R j < k).card ≤ m - k := by
    have hfilter : (Finset.univ.filter fun j => ¬ strictRank R j < k)
        = (Finset.univ.filter fun j => k ≤ strictRank R j) :=
      Finset.filter_congr (fun j _ => by simp [not_lt])
    rw [hfilter]
    exact card_high_strictRank_le R k
  have hcardfin : Fintype.card (Fin m) = m := Fintype.card_fin m
  omega

end KBound
