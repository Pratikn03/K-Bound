import KBound.Probability.MeasureConformal
import Mathlib.MeasureTheory.Measure.Dirac
import Mathlib.Tactic.NormNum

/-! # Nine perfectly dependent fair signs: an exact sign-flip counterexample

This is the mathematical null witness in the maintained supplement1280–1287,
not a model fitted to the observed benchmark. Coordinate exchangeability and
global sign symmetry do not imply independent-coordinate sign-flip invariance.
-/

namespace KBound.DependentSignFlip

open MeasureTheory
open scoped BigOperators ENNReal

def sign (s : Bool) : ℝ := if s then 1 else -1
def gaps (s : Bool) : Fin 9 → ℝ := fun _ => sign s

noncomputable def fairLaw : Measure Bool :=
  (1 / 2 : ENNReal) • Measure.dirac true + (1 / 2 : ENNReal) • Measure.dirac false

instance : IsProbabilityMeasure fairLaw := by
  constructor
  norm_num [fairLaw]
  exact ENNReal.inv_two_add_inv_two

noncomputable def gapLaw : Measure (Fin 9 → ℝ) := fairLaw.map gaps

instance : IsProbabilityMeasure gapLaw :=
  Measure.isProbabilityMeasure_map (measurable_of_countable gaps).aemeasurable

theorem coordinate_exchangeable : ScoreLawExchangeable gapLaw := by
  intro sigma
  rw [gapLaw, Measure.map_map (measurable_score_reindex sigma) (measurable_of_countable gaps)]
  rfl

theorem positive_coordinate_probability (i : Fin 9) :
    gapLaw {v | v i = 1} = (1 / 2 : ENNReal) := by
  rw [gapLaw, Measure.map_apply (measurable_of_countable gaps)
    (measurableSet_eq_fun (measurable_pi_apply i) measurable_const)]
  norm_num [fairLaw, gaps, sign]

theorem exchangeable_fair_witness : ∃ P : Measure (Fin 9 → ℝ),
    IsProbabilityMeasure P ∧ ScoreLawExchangeable P ∧
    P {v | v 0 = 1} = (1 / 2 : ENNReal) :=
  ⟨gapLaw, inferInstance, coordinate_exchangeable, positive_coordinate_probability 0⟩

lemma fair_map (f : Bool → (Fin 9 → ℝ)) :
    fairLaw.map f = (1 / 2 : ENNReal) • Measure.dirac (f true) +
      (1 / 2 : ENNReal) • Measure.dirac (f false) := by
  rw [fairLaw, Measure.map_add _ _ (measurable_of_countable f)]
  simp only [Measure.map_smul, Measure.map_dirac' (measurable_of_countable f)]

def globalFlip (v : Fin 9 → ℝ) : Fin 9 → ℝ := fun i => -v i

lemma measurable_globalFlip : Measurable globalFlip :=
  measurable_pi_lambda _ fun i => (measurable_pi_apply i).neg

theorem global_sign_symmetric : gapLaw.map globalFlip = gapLaw := by
  rw [gapLaw, Measure.map_map measurable_globalFlip (measurable_of_countable gaps)]
  rw [fair_map, fair_map]
  have hp : globalFlip ∘ gaps = fun s => gaps (!s) := by
    funext s i
    cases s <;> norm_num [globalFlip, gaps, sign]
  rw [hp]
  simp only [Bool.not_true, Bool.not_false]
  exact add_comm _ _

theorem marginal_sign_symmetric (i : Fin 9) :
    (gapLaw.map (fun v => v i)).map (fun x : ℝ => -x) = gapLaw.map (fun v => v i) := by
  have h := congrArg (fun P : Measure (Fin 9 → ℝ) => P.map (fun v => v i)) global_sign_symmetric
  dsimp only at h
  rw [Measure.map_map (measurable_pi_apply i) measurable_globalFlip] at h
  rw [Measure.map_map (measurable_neg : Measurable (fun x : ℝ => -x))
    (measurable_pi_apply i)]
  exact h

def flipFirst (v : Fin 9 → ℝ) : Fin 9 → ℝ :=
  fun i => if i = 0 then -v i else v i

lemma measurable_flipFirst : Measurable flipFirst := by
  apply measurable_pi_lambda
  intro i
  by_cases hi : i = 0
  · simpa only [flipFirst, hi, if_pos] using
      (measurable_pi_apply i : Measurable (fun v : Fin 9 → ℝ => v i)).neg
  · simpa only [flipFirst, hi, if_neg] using
      (measurable_pi_apply i : Measurable (fun v : Fin 9 → ℝ => v i))

theorem equal_coordinates_probability : gapLaw {v | v 0 = v 1} = 1 := by
  rw [gapLaw, Measure.map_apply (measurable_of_countable gaps)
    (measurableSet_eq_fun (measurable_pi_apply 0) (measurable_pi_apply 1))]
  have he : {s : Bool | gaps s 0 = gaps s 1} = Set.univ := by
    ext s
    simp [gaps]
  change fairLaw {s : Bool | gaps s 0 = gaps s 1} = 1
  rw [he, measure_univ]

theorem flipped_equal_coordinates_probability :
    (gapLaw.map flipFirst) {v | v 0 = v 1} = 0 := by
  rw [gapLaw, Measure.map_map measurable_flipFirst (measurable_of_countable gaps)]
  rw [Measure.map_apply (measurable_flipFirst.comp (measurable_of_countable gaps))
    (measurableSet_eq_fun (measurable_pi_apply 0) (measurable_pi_apply 1))]
  norm_num [fairLaw, flipFirst, gaps, sign, Set.indicator_apply,
    show (1 : Fin 9) ≠ 0 by decide]

theorem not_coordinate_sign_symmetric : gapLaw.map flipFirst ≠ gapLaw := by
  intro h
  have he := congrArg (fun P : Measure (Fin 9 → ℝ) => P {v | v 0 = v 1}) h
  dsimp only at he
  rw [flipped_equal_coordinates_probability, equal_coordinates_probability] at he
  exact zero_ne_one he

lemma sign_le_one (s : Bool) : sign s ≤ 1 := by cases s <;> norm_num [sign]

lemma all_positive_iff (s : Fin 9 → Bool) :
    (9 : ℝ) ≤ ∑ i, sign (s i) ↔ ∀ i, s i = true := by
  constructor
  · intro hs i
    by_contra hi
    have hfalse : s i = false := Bool.eq_false_of_not_eq_true hi
    have hsum : (∑ j, sign (s j)) < ∑ _j : Fin 9, (1 : ℝ) := by
      apply Finset.sum_lt_sum (fun j _ => sign_le_one (s j))
      exact ⟨i, Finset.mem_univ i, by norm_num [sign, hfalse]⟩
    norm_num at hsum
    linarith
  · intro hs
    simp [hs, sign]

open Classical in
noncomputable def upperTailCount (v : Fin 9 → ℝ) : ℕ :=
  (Finset.univ.filter fun s : Fin 9 → Bool =>
    (∑ i, v i) ≤ ∑ i, sign (s i) * v i).card

noncomputable def referenceP (v : Fin 9 → ℝ) : ℝ :=
  (upperTailCount v : ℝ) / Fintype.card (Fin 9 → Bool)

/-- The enumeration defined using sums is exactly the printed mean-based
one-sided comparison, since both means divide by the same positive nine. -/
theorem mean_comparison_iff (v : Fin 9 → ℝ) (s : Fin 9 → Bool) :
    ((∑ i, v i) / 9 ≤ (∑ i, sign (s i) * v i) / 9) ↔
      (∑ i, v i) ≤ ∑ i, sign (s i) * v i :=
  div_le_div_iff_of_pos_right (by norm_num)

theorem sign_vector_count : Fintype.card (Fin 9 → Bool) = 512 := by
  norm_num [Fintype.card_fun]

theorem positive_upper_tail_count : upperTailCount (gaps true) = 1 := by
  classical
  have he : (Finset.univ.filter fun s : Fin 9 → Bool =>
      (9 : ℝ) ≤ ∑ i, sign (s i)) = {fun _ => true} := by
    ext s
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, all_positive_iff,
      Finset.mem_singleton, funext_iff]
  simp only [upperTailCount, show gaps true = (fun _ => 1) by rfl, mul_one,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
  simpa only [Nat.cast_ofNat, Finset.card_singleton] using congrArg Finset.card he

theorem negative_upper_tail_count : upperTailCount (gaps false) = 512 := by
  classical
  have hall (s : Fin 9 → Bool) : (∑ _i : Fin 9, (-1 : ℝ)) ≤ ∑ i, sign (s i) * -1 := by
    apply Finset.sum_le_sum
    intro i _
    have h := sign_le_one (s i)
    linarith
  simp only [upperTailCount, show gaps false = (fun _ => -1) by rfl]
  rw [Finset.filter_eq_self.mpr (fun s _ => hall s)]
  exact sign_vector_count

theorem positive_reference_value : referenceP (gaps true) = 1 / 512 := by
  rw [referenceP, positive_upper_tail_count, sign_vector_count]
  norm_num

theorem negative_reference_value : referenceP (gaps false) = 1 := by
  rw [referenceP, negative_upper_tail_count, sign_vector_count]
  norm_num

theorem small_reference_probability :
    fairLaw {s | referenceP (gaps s) = 1 / 512} = (1 / 2 : ENNReal) := by
  norm_num [fairLaw, Set.indicator_apply, positive_reference_value, negative_reference_value]

end KBound.DependentSignFlip

#print axioms KBound.DependentSignFlip.coordinate_exchangeable
#print axioms KBound.DependentSignFlip.global_sign_symmetric
#print axioms KBound.DependentSignFlip.marginal_sign_symmetric
#print axioms KBound.DependentSignFlip.not_coordinate_sign_symmetric
#print axioms KBound.DependentSignFlip.positive_upper_tail_count
#print axioms KBound.DependentSignFlip.positive_reference_value
#print axioms KBound.DependentSignFlip.mean_comparison_iff
#print axioms KBound.DependentSignFlip.small_reference_probability
