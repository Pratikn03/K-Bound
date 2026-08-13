import Mathlib.Data.Finset.Card
import Mathlib.Data.Fintype.Card
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators

/-!
# The jackknife+ counting bound: where the factor two comes from

`sec:fa-identity` declines to claim jackknife+ and says so; the paper's grids use a
leave-one-out radius with no distribution-free finite-sample guarantee. The reason
jackknife+ buys `1 - 2*alpha` rather than `1 - alpha` is a purely combinatorial fact
about comparison matrices, due to Barber, Candes, Ramdas and Tibshirani, and that fact
is what this file proves.

Set `A i j` to record that leave-two-out residual `i` strictly exceeds residual `j`.
Exchangeability makes `A` antisymmetric off the diagonal: at most one of `A i j`,
`A j i` holds. `two_mul_pairs_le` is the double count -- each unordered pair contributes
at most one ordered pair, so `2 * |P| + |S| <= |S|^2` on any `S`. `card_le_two_mul` then
shows that the set of indices whose row count is within `k` of the maximum has at most
`2k` elements. Taking `k` as the miscoverage budget, that is the factor two.

What is NOT claimed here: the reduction from a jackknife+ prediction interval to this
matrix, which needs the exchangeability argument over the `n+1` augmented points, and is
the same modelling step `UniformConformal.lean` performs for split conformal. This file
supplies the combinatorial core those arguments consume, not the end-to-end coverage
theorem.

Everything is stated over `ℕ` so no real analysis is involved and truncated subtraction
never appears in a hypothesis.
-/

namespace KBoundJK
open Finset

variable {m : ℕ}

/-- How many indices the row `i` dominates. -/
def rowCount (A : Fin m → Fin m → Bool) (i : Fin m) : ℕ :=
  (univ.filter (fun j => A i j = true)).card

/-- **Double count.** At most one ordered pair per unordered pair, nothing on the
diagonal: the ordered pairs inside `S` on which `A` holds number at most
`(|S|^2 - |S|)/2`. -/
theorem two_mul_pairs_le (A : Fin m → Fin m → Bool)
    (hdiag : ∀ i, A i i = false)
    (hpair : ∀ i j, A i j = true → A j i = false)
    (S : Finset (Fin m)) :
    2 * (((S ×ˢ S).filter (fun p => A p.1 p.2 = true)).card) + S.card ≤ S.card * S.card := by
  set P := (S ×ˢ S).filter (fun p => A p.1 p.2 = true) with hP
  have hinj : Function.Injective (Prod.swap : Fin m × Fin m → Fin m × Fin m) :=
    fun a b h => by simpa using congrArg Prod.swap h
  have hcard : (P.image Prod.swap).card = P.card := card_image_of_injective _ hinj
  have hdisj : Disjoint P (P.image Prod.swap) := by
    rw [disjoint_right]
    rintro ⟨i, j⟩ hmem hp
    simp only [hP, mem_image, mem_filter, mem_product, Prod.exists, Prod.swap_prod_mk,
      Prod.mk.injEq] at hmem hp
    obtain ⟨a, b, ⟨_, hab⟩, ha, hb⟩ := hmem
    subst ha; subst hb
    exact absurd hp.2 (by simp [hpair _ _ hab])
  have hsub : P ∪ P.image Prod.swap ⊆ S.offDiag := by
    rintro ⟨i, j⟩ hp
    simp only [hP, mem_union, mem_image, mem_filter, mem_product, Prod.exists,
      Prod.swap_prod_mk, Prod.mk.injEq, mem_offDiag] at hp ⊢
    rcases hp with ⟨⟨h1, h2⟩, h3⟩ | ⟨a, b, ⟨⟨h1, h2⟩, hab⟩, ha, hb⟩
    · refine ⟨h1, h2, ?_⟩
      rintro rfl
      rw [hdiag] at h3
      exact absurd h3 (by simp)
    · cases ha; cases hb
      refine ⟨h2, h1, ?_⟩
      rintro rfl
      rw [hdiag] at hab
      exact absurd hab (by simp)
  have hunion : P.card + (P.image Prod.swap).card ≤ S.offDiag.card :=
    (card_union_of_disjoint hdisj) ▸ card_le_card hsub
  rw [hcard] at hunion
  have hoff : S.offDiag.card = S.card * S.card - S.card := offDiag_card S
  have hs_le : S.card ≤ S.card * S.card := by
    rcases Nat.eq_zero_or_pos S.card with h | h
    · simp [h]
    · exact Nat.le_mul_of_pos_left _ h
  rw [hoff] at hunion
  omega

/-- **Jackknife+ counting bound.** If every index in `S` has row count within `k` of the
maximum `m`, then `|S| ≤ 2k`. This is the step that turns a budget of `k` misses into a
guarantee at `1 - 2*alpha` rather than `1 - alpha`. -/
theorem card_le_two_mul (A : Fin m → Fin m → Bool)
    (hdiag : ∀ i, A i i = false)
    (hpair : ∀ i j, A i j = true → A j i = false)
    (S : Finset (Fin m)) (k : ℕ)
    (hS : ∀ i ∈ S, m ≤ rowCount A i + k) :
    S.card ≤ 2 * k := by
  rcases Nat.eq_zero_or_pos S.card with h0 | hpos
  · omega
  set s := S.card with hs
  set P := (S ×ˢ S).filter (fun p => A p.1 p.2 = true) with hP
  have hpairs : 2 * P.card + s ≤ s * s := two_mul_pairs_le A hdiag hpair S
  have hsm : s ≤ m := by
    rw [hs]; simpa using card_le_card (subset_univ S)
  have hd : s + (Finset.univ \ S).card = m := by
    rw [card_sdiff, inter_univ, card_univ, Fintype.card_fin]
    omega
  have hsplit : ∀ i, rowCount A i ≤ (S.filter (fun j => A i j = true)).card
      + (Finset.univ \ S).card := by
    intro i
    have hsub : univ.filter (fun j => A i j = true)
        ⊆ (S.filter (fun j => A i j = true)) ∪ (univ \ S) := by
      intro j hj
      simp only [mem_filter, mem_univ, true_and, mem_union, mem_sdiff] at hj ⊢
      by_cases hjs : j ∈ S
      · exact Or.inl ⟨hjs, hj⟩
      · exact Or.inr hjs
    exact le_trans (card_le_card hsub) (card_union_le _ _)
  have hPsum : P.card = ∑ i ∈ S, (S.filter (fun j => A i j = true)).card := by
    rw [hP, card_filter, sum_product]
    simp [card_filter]
  have hlow : s * m ≤ (∑ i ∈ S, rowCount A i) + s * k := by
    have : ∑ i ∈ S, m ≤ ∑ i ∈ S, (rowCount A i + k) := sum_le_sum hS
    simpa [sum_add_distrib, sum_const, smul_eq_mul, mul_comm] using this
  have hhigh : (∑ i ∈ S, rowCount A i) ≤ P.card + s * (Finset.univ \ S).card := by
    calc ∑ i ∈ S, rowCount A i
        ≤ ∑ i ∈ S, ((S.filter (fun j => A i j = true)).card + (univ \ S).card) :=
          sum_le_sum (fun i _ => hsplit i)
      _ = P.card + s * (univ \ S).card := by
          rw [sum_add_distrib, ← hPsum, sum_const, smul_eq_mul]
  set d := (Finset.univ \ S).card with hdd
  set q := s * s with hq
  set r := s * k with hr
  set e := s * d with he
  have hm : s * m = q + e := by rw [hq, he, ← hd]; ring
  have key : q + s ≤ 2 * r := by omega
  have hfin : s * s ≤ s * (2 * k) := by
    have hqr : q ≤ 2 * r := by omega
    simp only [hq, hr] at hqr
    calc s * s ≤ 2 * (s * k) := hqr
      _ = s * (2 * k) := by ring
  exact Nat.le_of_mul_le_mul_left hfin hpos

end KBoundJK

-- Axiom audit, Lean 4.29.1 + Mathlib v4.29.1: both depend only on
-- [propext, Classical.choice, Quot.sound]; never sorryAx.
#print axioms KBoundJK.two_mul_pairs_le
#print axioms KBoundJK.card_le_two_mul
