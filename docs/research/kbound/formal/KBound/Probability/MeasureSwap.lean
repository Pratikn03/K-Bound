import Mathlib.Probability.Kernel.Composition.MeasureComp
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

/-!
# General measurable evidence-preserving label swap

The targets below are actual measures on an arbitrary measurable input/label
space.  Exchanging the two predicted labels is a measurable involution.  It
preserves every label-free evidence law and negates the population 0/1-loss
difference.  No discrete-input, two-point-law, or constant-evidence premise is
used.  The negative result concerns evidence-definable classes; it does not
assert that choosing an arbitrary representative of every swap pair identifies
the sign on a larger evidence fibre.
-/

namespace KBound

open MeasureTheory ProbabilityTheory Set
open scoped ProbabilityTheory

variable {X Y E : Type*} [MeasurableSpace X] [MeasurableSpace Y]
  [MeasurableSpace E] [MeasurableEq Y]

/-- Swap the predicted labels, leaving all other labels and the input unchanged. -/
noncomputable def predictionSwap (f₀ fₐ : X → Y) (xy : X × Y) : X × Y := by
  classical
  exact (xy.1, if xy.2 = fₐ xy.1 then f₀ xy.1
    else if xy.2 = f₀ xy.1 then fₐ xy.1 else xy.2)

omit [MeasurableSpace X] [MeasurableSpace Y] [MeasurableEq Y] in
theorem predictionSwap_fst (f₀ fₐ : X → Y) (xy : X × Y) :
    (predictionSwap f₀ fₐ xy).1 = xy.1 := rfl

omit [MeasurableSpace X] [MeasurableSpace Y] [MeasurableEq Y] in
theorem predictionSwap_involutive (f₀ fₐ : X → Y) :
    Function.Involutive (predictionSwap f₀ fₐ) := by
  classical
  rintro ⟨x, y⟩
  by_cases h : f₀ x = fₐ x
  · by_cases hy : y = fₐ x <;> simp [predictionSwap, h, hy]
  · by_cases ha : y = fₐ x
    · simp [predictionSwap, ha, h]
    · by_cases h₀ : y = f₀ x
      · simp [predictionSwap, h₀, h]
      · simp [predictionSwap, ha, h₀]

theorem measurable_predictionSwap (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ) :
    Measurable (predictionSwap f₀ fₐ) := by
  classical
  exact measurable_fst.prodMk
    (Measurable.ite (measurableSet_eq_fun measurable_snd (hₐ.comp measurable_fst))
      (h₀.comp measurable_fst)
      (Measurable.ite (measurableSet_eq_fun measurable_snd (h₀.comp measurable_fst))
        (hₐ.comp measurable_fst) measurable_snd))

/-- The signed improvement in 0/1 loss at one labeled input. -/
noncomputable def zeroOneBenefit (f₀ fₐ : X → Y) (xy : X × Y) : ℝ := by
  classical
  exact (if xy.2 = f₀ xy.1 then 0 else 1) - (if xy.2 = fₐ xy.1 then 0 else 1)

theorem measurable_zeroOneBenefit (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ) :
    Measurable (zeroOneBenefit f₀ fₐ) := by
  classical
  exact (Measurable.ite (measurableSet_eq_fun measurable_snd (h₀.comp measurable_fst))
    measurable_const measurable_const).sub
    (Measurable.ite (measurableSet_eq_fun measurable_snd (hₐ.comp measurable_fst))
      measurable_const measurable_const)

omit [MeasurableSpace X] [MeasurableSpace Y] [MeasurableEq Y] in
theorem zeroOneBenefit_swap (f₀ fₐ : X → Y) (xy : X × Y) :
    zeroOneBenefit f₀ fₐ (predictionSwap f₀ fₐ xy) = -zeroOneBenefit f₀ fₐ xy := by
  classical
  rcases xy with ⟨x, y⟩
  by_cases h : f₀ x = fₐ x
  · by_cases hy : y = fₐ x <;> simp [predictionSwap, zeroOneBenefit, h, hy]
  · by_cases ha : y = fₐ x
    · simp [predictionSwap, zeroOneBenefit, ha, h, Ne.symm h]
    · by_cases h₀ : y = f₀ x
      · simp [predictionSwap, zeroOneBenefit, h₀, h, Ne.symm h]
      · simp [predictionSwap, zeroOneBenefit, ha, h₀]

omit [MeasurableSpace X] [MeasurableSpace Y] [MeasurableEq Y] in
theorem abs_zeroOneBenefit_le_one (f₀ fₐ : X → Y) (xy : X × Y) :
    |zeroOneBenefit f₀ fₐ xy| ≤ 1 := by
  classical
  unfold zeroOneBenefit
  split_ifs <;> norm_num

/-- Population improvement is the expectation of the actual loss difference. -/
noncomputable def populationBenefit (f₀ fₐ : X → Y) (P : Measure (X × Y)) : ℝ :=
  ∫ xy, zeroOneBenefit f₀ fₐ xy ∂P

theorem integrable_zeroOneBenefit (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (P : Measure (X × Y)) [IsFiniteMeasure P] : Integrable (zeroOneBenefit f₀ fₐ) P := by
  apply Integrable.mono' (integrable_const (1 : ℝ))
    (measurable_zeroOneBenefit f₀ fₐ h₀ hₐ).aestronglyMeasurable
  exact Filter.Eventually.of_forall fun xy => by
    simpa only [Real.norm_eq_abs] using abs_zeroOneBenefit_le_one f₀ fₐ xy

/-- The pushforward is an involution on the space of probability laws. -/
theorem predictionSwap_law_involutive (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ) (P : Measure (X × Y)) :
    (P.map (predictionSwap f₀ fₐ)).map (predictionSwap f₀ fₐ) = P := by
  rw [Measure.map_map (measurable_predictionSwap f₀ fₐ h₀ hₐ)
    (measurable_predictionSwap f₀ fₐ h₀ hₐ)]
  simp only [Function.Involutive.comp_self (predictionSwap_involutive f₀ fₐ),
    Measure.map_id]

theorem predictionSwap_preserves_evidence (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ) (P : Measure (X × Y))
    (g : X → E) (hg : Measurable g) :
    (P.map (predictionSwap f₀ fₐ)).map (fun xy => g xy.1) =
      P.map (fun xy => g xy.1) := by
  change (P.map (predictionSwap f₀ fₐ)).map (g ∘ Prod.fst) = P.map (g ∘ Prod.fst)
  rw [Measure.map_map (hg.comp measurable_fst) (measurable_predictionSwap f₀ fₐ h₀ hₐ)]
  rfl

theorem predictionSwap_preserves_input (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ) (P : Measure (X × Y)) :
    (P.map (predictionSwap f₀ fₐ)).fst = P.fst :=
  predictionSwap_preserves_evidence f₀ fₐ h₀ hₐ P id measurable_id

/-- Randomized label-free evidence channels are preserved as well. -/
theorem predictionSwap_preserves_channel (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ) (P : Measure (X × Y))
    (K : Kernel X E) :
    K ∘ₘ (P.map (predictionSwap f₀ fₐ)).fst = K ∘ₘ P.fst := by
  rw [predictionSwap_preserves_input f₀ fₐ h₀ hₐ P]

/-- Exact sign reversal of population risk for arbitrary measurable target laws. -/
theorem predictionSwap_negates_populationBenefit (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ) (P : Measure (X × Y)) :
    populationBenefit f₀ fₐ (P.map (predictionSwap f₀ fₐ)) =
      -populationBenefit f₀ fₐ P := by
  unfold populationBenefit
  rw [integral_map (measurable_predictionSwap f₀ fₐ h₀ hₐ).aemeasurable
    (measurable_zeroOneBenefit f₀ fₐ h₀ hₐ).aestronglyMeasurable]
  simp_rw [zeroOneBenefit_swap]
  exact integral_neg _

/-- Membership is determined by the law of a label-free evidence channel. -/
def EvidenceDefinableTargets (C : Set (Measure (X × Y))) (K : Kernel X E) : Prop :=
  ∀ P Q, K ∘ₘ P.fst = K ∘ₘ Q.fst → (P ∈ C ↔ Q ∈ C)

/-- Every nonzero-benefit probability law in an evidence-definable class has a
valid, evidence-identical, opposite-benefit probability law in that same class. -/
theorem evidence_definable_opposite_target (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (C : Set (Measure (X × Y))) (K : Kernel X E)
    (hC : EvidenceDefinableTargets C K)
    (P : Measure (X × Y)) [IsProbabilityMeasure P]
    (hP : P ∈ C) (hne : populationBenefit f₀ fₐ P ≠ 0) :
    ∃ Q : Measure (X × Y), IsProbabilityMeasure Q ∧ Q ∈ C ∧ Q ≠ P ∧
      K ∘ₘ Q.fst = K ∘ₘ P.fst ∧
      populationBenefit f₀ fₐ Q = -populationBenefit f₀ fₐ P := by
  let Q := P.map (predictionSwap f₀ fₐ)
  have he := predictionSwap_preserves_channel f₀ fₐ h₀ hₐ P K
  have hb := predictionSwap_negates_populationBenefit f₀ fₐ h₀ hₐ P
  have hprob : IsProbabilityMeasure Q :=
    Measure.isProbabilityMeasure_map (measurable_predictionSwap f₀ fₐ h₀ hₐ).aemeasurable
  refine ⟨Q, hprob, (hC Q P he).mpr hP, ?_, he, hb⟩
  intro hQP
  have h := hb
  change populationBenefit f₀ fₐ Q = -populationBenefit f₀ fₐ P at h
  rw [hQP] at h
  exact hne (by linarith)

end KBound
