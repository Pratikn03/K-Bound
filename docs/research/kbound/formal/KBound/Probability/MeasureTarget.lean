import KBound.Probability.MeasureSwap
import Mathlib.Probability.Kernel.Composition.IntegralCompProd
import Mathlib.Probability.Kernel.Composition.MeasureComp
import Mathlib.MeasureTheory.Constructions.BorelSpace.Order
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FieldSimp

/-!
# Measurable label-kernel freedom

This file constructs actual probability measures over an arbitrary measurable
input space.  The conditional law off disagreement is kept fixed.  The
correctness field on disagreement may be any measurable `[0,1]`-valued field.
It is not a two-atom input construction or an assumed richness predicate.
-/

namespace KBound

open MeasureTheory ProbabilityTheory Set
open scoped ENNReal ProbabilityTheory

variable {X Y E : Type*} [MeasurableSpace X] [MeasurableSpace Y]
  [MeasurableSpace E] [MeasurableSingletonClass Y] [MeasurableEq Y]

/-- A measurable mixture of the candidate and reference predictions. -/
noncomputable def correctnessKernel (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (η : X → ℝ≥0∞) (hη : Measurable η) : Kernel X Y where
  toFun x := η x • Measure.dirac (fₐ x) + (1 - η x) • Measure.dirac (f₀ x)
  measurable' := by
    refine Measure.measurable_of_measurable_coe _ fun s hs => ?_
    simp only [Measure.add_apply, Measure.smul_apply, smul_eq_mul]
    exact (hη.mul ((Kernel.deterministic fₐ hₐ).measurable_coe hs)).add
      ((measurable_const.sub hη).mul ((Kernel.deterministic f₀ h₀).measurable_coe hs))

instance correctnessKernel_isMarkov (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (η : X → ℝ≥0∞) (hη : Measurable η) (hη₁ : ∀ x, η x ≤ 1) :
    IsMarkovKernel (correctnessKernel f₀ fₐ h₀ hₐ η hη) := by
  constructor
  intro x
  constructor
  change (η x • Measure.dirac (fₐ x) + (1 - η x) • Measure.dirac (f₀ x)) univ = 1
  simp only [Measure.add_apply, Measure.smul_apply, Measure.dirac_apply_of_mem (mem_univ _),
    smul_eq_mul, mul_one]
  exact add_tsub_cancel_of_le (hη₁ x)

omit [MeasurableEq Y] in
/-- The mixture has the prescribed candidate correctness wherever the two
predictions disagree. -/
theorem correctnessKernel_candidate_mass (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (η : X → ℝ≥0∞) (hη : Measurable η) {x : X} (hx : f₀ x ≠ fₐ x) :
    correctnessKernel f₀ fₐ h₀ hₐ η hη x {fₐ x} = η x := by
  classical
  change (η x • Measure.dirac (fₐ x) + (1 - η x) • Measure.dirac (f₀ x)) {fₐ x} = η x
  simp [Measure.add_apply, Measure.smul_apply, hx]

omit [MeasurableSingletonClass Y] in
/-- The input-dependent disagreement set is measurable. -/
theorem measurable_disagreement (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ) :
    MeasurableSet {x | f₀ x ≠ fₐ x} :=
  (measurableSet_eq_fun h₀ hₐ).compl

/-- Retain the given conditional law outside disagreement, replacing only the
unobserved label kernel on disagreement. -/
noncomputable def targetLabelKernel (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) (η : X → ℝ≥0∞) (hη : Measurable η) : Kernel X Y := by
  classical
  exact Kernel.piecewise (measurable_disagreement f₀ fₐ h₀ hₐ)
    (correctnessKernel f₀ fₐ h₀ hₐ η hη) κ₀

theorem targetLabelKernel_on_disagreement (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) (η : X → ℝ≥0∞) (hη : Measurable η)
    {x : X} (hx : f₀ x ≠ fₐ x) :
    targetLabelKernel f₀ fₐ h₀ hₐ κ₀ η hη x {fₐ x} = η x := by
  classical
  simpa [targetLabelKernel, Kernel.piecewise_apply, hx] using
    correctnessKernel_candidate_mass f₀ fₐ h₀ hₐ η hη hx

omit [MeasurableSingletonClass Y] in
theorem targetLabelKernel_off_disagreement (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) (η : X → ℝ≥0∞) (hη : Measurable η)
    {x : X} (hx : f₀ x = fₐ x) :
    targetLabelKernel f₀ fₐ h₀ hₐ κ₀ η hη x = κ₀ x := by
  classical
  simp [targetLabelKernel, Kernel.piecewise_apply, hx]

omit [MeasurableSingletonClass Y] in
theorem targetLabelKernel_isMarkov (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀]
    (η : X → ℝ≥0∞) (hη : Measurable η) (hη₁ : ∀ x, η x ≤ 1) :
    IsMarkovKernel (targetLabelKernel f₀ fₐ h₀ hₐ κ₀ η hη) := by
  classical
  letI := correctnessKernel_isMarkov f₀ fₐ h₀ hₐ η hη hη₁
  unfold targetLabelKernel
  infer_instance

omit [MeasurableSingletonClass Y] [MeasurableEq Y] in
/-- Any Markov label kernel produces a genuine joint probability law with the
specified input marginal. -/
theorem joint_target_probability (μ : Measure X) [IsProbabilityMeasure μ]
    (κ : Kernel X Y) [IsMarkovKernel κ] :
    IsProbabilityMeasure (μ ⊗ₘ κ) ∧ (μ ⊗ₘ κ).fst = μ := by
  exact ⟨inferInstance, Measure.fst_compProd μ κ⟩

omit [MeasurableSingletonClass Y] [MeasurableEq Y] in
/-- Every measurable label-free observable has exactly its original input law,
not just a matching expectation or a constant evidence value. -/
theorem target_label_free_law (μ : Measure X) [IsProbabilityMeasure μ]
    (κ : Kernel X Y) [IsMarkovKernel κ] (g : X → E) (hg : Measurable g) :
    (μ ⊗ₘ κ).map (fun xy => g xy.1) = μ.map g := by
  change (μ ⊗ₘ κ).map (g ∘ Prod.fst) = μ.map g
  rw [← Measure.map_map hg measurable_fst]
  change ((μ ⊗ₘ κ).fst).map g = μ.map g
  rw [Measure.fst_compProd]

/-- Arbitrary measurable correctness fields, fixed off-disagreement labels,
valid joint laws, and all measurable label-free evidence laws coexist. -/
theorem measurable_label_kernel_freedom (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀]
    (η : X → ℝ≥0∞) (hη : Measurable η) (hη₁ : ∀ x, η x ≤ 1) :
    ∃ κ : Kernel X Y, IsMarkovKernel κ ∧
      (∀ x, f₀ x ≠ fₐ x → κ x {fₐ x} = η x) ∧
      (∀ x, f₀ x = fₐ x → κ x = κ₀ x) ∧
      (μ ⊗ₘ κ).fst = μ ∧
      (∀ (g : X → E), Measurable g → (μ ⊗ₘ κ).map (fun xy => g xy.1) = μ.map g) := by
  let κ := targetLabelKernel f₀ fₐ h₀ hₐ κ₀ η hη
  letI := targetLabelKernel_isMarkov f₀ fₐ h₀ hₐ κ₀ η hη hη₁
  refine ⟨κ, inferInstance, ?_, ?_, Measure.fst_compProd μ κ, ?_⟩
  · exact fun _ hx => targetLabelKernel_on_disagreement f₀ fₐ h₀ hₐ κ₀ η hη hx
  · exact fun _ hx => targetLabelKernel_off_disagreement f₀ fₐ h₀ hₐ κ₀ η hη hx
  · exact fun g hg => target_label_free_law μ κ g hg

/-- The paper permits a field defined only on the disagreement subtype.  A
measurable zero extension gives precisely that version, without requiring an
unmentioned extension assumption. -/
theorem measurable_label_kernel_freedom_subtype (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀]
    (η : {x : X | f₀ x ≠ fₐ x} → ℝ≥0∞) (hη : Measurable η)
    (hη₁ : ∀ x, η x ≤ 1) :
    ∃ κ : Kernel X Y, IsMarkovKernel κ ∧
      (∀ (x : X) (hx : f₀ x ≠ fₐ x), κ x {fₐ x} = η ⟨x, hx⟩) ∧
      (∀ x, f₀ x = fₐ x → κ x = κ₀ x) ∧
      (μ ⊗ₘ κ).fst = μ ∧
      (∀ (g : X → E), Measurable g → (μ ⊗ₘ κ).map (fun xy => g xy.1) = μ.map g) := by
  classical
  let η' : X → ℝ≥0∞ := fun x => if hx : f₀ x ≠ fₐ x then η ⟨x, hx⟩ else 0
  have hm : Measurable η' :=
    hη.dite measurable_const (measurable_disagreement f₀ fₐ h₀ hₐ)
  have hb : ∀ x, η' x ≤ 1 := by
    intro x
    by_cases hx : f₀ x ≠ fₐ x
    · simpa only [η', dif_pos hx] using hη₁ ⟨x, hx⟩
    · simp only [η', dif_neg hx]
      exact zero_le _
  obtain ⟨κ, hκ, hon, hoff, hfst, he⟩ :=
    measurable_label_kernel_freedom (E := E) μ f₀ fₐ h₀ hₐ κ₀ η' hm hb
  refine ⟨κ, hκ, ?_, hoff, hfst, he⟩
  intro x hx
  simpa [η', hx] using hon x hx

omit [MeasurableEq Y] in
/-- The constructed conditional loss difference is the genuine signed risk
difference `2η-1`, not an independently supplied benefit coordinate. -/
theorem correctnessKernel_integral_benefit (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (η : X → ℝ≥0∞) (hη : Measurable η) (hη₁ : ∀ x, η x ≤ 1)
    {x : X} (hx : f₀ x ≠ fₐ x) :
    (∫ y, zeroOneBenefit f₀ fₐ (x, y) ∂correctnessKernel f₀ fₐ h₀ hₐ η hη x) =
      2 * (η x).toReal - 1 := by
  classical
  have hfin : η x ≠ ∞ := ne_top_of_le_ne_top ENNReal.one_ne_top (hη₁ x)
  have hfin' : 1 - η x ≠ ∞ := ne_top_of_le_ne_top ENNReal.one_ne_top tsub_le_self
  have hi (y : Y) : Integrable (fun y => zeroOneBenefit f₀ fₐ (x, y)) (Measure.dirac y) :=
    integrable_dirac (by simp)
  change (∫ y, zeroOneBenefit f₀ fₐ (x, y)
    ∂(η x • Measure.dirac (fₐ x) + (1 - η x) • Measure.dirac (f₀ x))) = _
  rw [integral_add_measure ((hi _).smul_measure hfin) ((hi _).smul_measure hfin')]
  simp [integral_smul_measure, zeroOneBenefit, hx, Ne.symm hx,
    ENNReal.toReal_sub_of_le (hη₁ x) ENNReal.one_ne_top]
  ring

open Classical in
theorem targetLabelKernel_integral_benefit (f₀ fₐ : X → Y)
    (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) (η : X → ℝ≥0∞) (hη : Measurable η) (hη₁ : ∀ x, η x ≤ 1)
    (x : X) :
    (∫ y, zeroOneBenefit f₀ fₐ (x, y)
      ∂targetLabelKernel f₀ fₐ h₀ hₐ κ₀ η hη x) =
      if f₀ x ≠ fₐ x then 2 * (η x).toReal - 1 else 0 := by
  classical
  by_cases hx : f₀ x ≠ fₐ x
  · simpa [targetLabelKernel, Kernel.piecewise_apply, hx] using
      correctnessKernel_integral_benefit f₀ fₐ h₀ hₐ η hη hη₁ hx
  · have heq : f₀ x = fₐ x := not_not.mp hx
    simp [zeroOneBenefit, heq]

/-- Exact population risk of the constructed joint law, with the input
probability and arbitrary off-disagreement label law kept fixed. -/
theorem constructed_target_population_benefit (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀]
    (η : X → ℝ≥0∞) (hη : Measurable η) (hη₁ : ∀ x, η x ≤ 1) :
    populationBenefit f₀ fₐ (μ ⊗ₘ targetLabelKernel f₀ fₐ h₀ hₐ κ₀ η hη) =
      ∫ x in {x | f₀ x ≠ fₐ x}, (2 * (η x).toReal - 1) ∂μ := by
  classical
  letI := targetLabelKernel_isMarkov f₀ fₐ h₀ hₐ κ₀ η hη hη₁
  unfold populationBenefit
  rw [Measure.integral_compProd (integrable_zeroOneBenefit f₀ fₐ h₀ hₐ _)]
  simp_rw [targetLabelKernel_integral_benefit f₀ fₐ h₀ hₐ κ₀ η hη hη₁]
  simpa only [Set.indicator_apply, Set.mem_setOf_eq] using
    (integral_indicator (μ := μ) (f := fun x => 2 * (η x).toReal - 1)
      (measurable_disagreement f₀ fₐ h₀ hₐ))

/-- Every feasible constant correctness value is attained by an actual joint
law.  Endpoints zero and one are included. -/
theorem constant_target_population_benefit (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀] (p : ℝ) (hp₀ : 0 ≤ p) (hp₁ : p ≤ 1) :
    populationBenefit f₀ fₐ
      (μ ⊗ₘ targetLabelKernel f₀ fₐ h₀ hₐ κ₀ (fun _ => ENNReal.ofReal p) measurable_const) =
      2 * μ.real {x | f₀ x ≠ fₐ x} * (p - 1 / 2) := by
  rw [constructed_target_population_benefit μ f₀ fₐ h₀ hₐ κ₀ _ measurable_const
    (fun _ => ENNReal.ofReal_le_one.mpr hp₁)]
  simp only [ENNReal.toReal_ofReal hp₀, setIntegral_const, smul_eq_mul]
  ring

/-- The full measurable correctness-field class, with range conditions stored
as data rather than a postulated target-richness premise. -/
structure CorrectnessField (X : Type*) [MeasurableSpace X] where
  value : X → ℝ≥0∞
  measurable_value : Measurable value
  value_le_one : ∀ x, value x ≤ 1

noncomputable def CorrectnessField.constant (p : ℝ) (_hp₀ : 0 ≤ p) (hp₁ : p ≤ 1) :
    CorrectnessField X where
  value := fun _ => ENNReal.ofReal p
  measurable_value := measurable_const
  value_le_one := fun _ => ENNReal.ofReal_le_one.mpr hp₁

/-- Disagreement-conditional correctness, with positive disagreement mass
required in the theorems that interpret it as a conditional mean. -/
noncomputable def disagreementMean (μ : Measure X) (D : Set X) (η : CorrectnessField X) : ℝ :=
  (∫ x in D, (η.value x).toReal ∂μ) / μ.real D

omit [MeasurableSingletonClass Y] [MeasurableEq Y] in
theorem correctnessField_integrable (ν : Measure X) [IsFiniteMeasure ν]
    (η : CorrectnessField X) : Integrable (fun x => (η.value x).toReal) ν := by
  apply Integrable.mono' (integrable_const (1 : ℝ))
    (η.measurable_value.ennreal_toReal).aestronglyMeasurable
  refine Filter.Eventually.of_forall fun x => ?_
  rw [Real.norm_eq_abs, abs_of_nonneg ENNReal.toReal_nonneg]
  simpa using ENNReal.toReal_mono ENNReal.one_ne_top (η.value_le_one x)

omit [MeasurableSingletonClass Y] [MeasurableEq Y] in
theorem disagreementMean_bounds (μ : Measure X) [IsProbabilityMeasure μ]
    (D : Set X) (hD : 0 < μ.real D) (η : CorrectnessField X) :
    0 ≤ disagreementMean μ D η ∧ disagreementMean μ D η ≤ 1 := by
  constructor
  · exact div_nonneg (integral_nonneg fun _ => ENNReal.toReal_nonneg) hD.le
  · apply (div_le_one hD).mpr
    calc
      (∫ x in D, (η.value x).toReal ∂μ) ≤ ∫ _x in D, (1 : ℝ) ∂μ := by
        apply integral_mono (correctnessField_integrable _ η) (integrable_const _)
        intro x
        simpa using ENNReal.toReal_mono ENNReal.one_ne_top (η.value_le_one x)
      _ = μ.real D := by simp

omit [MeasurableSingletonClass Y] [MeasurableEq Y] in
theorem disagreementMean_constant (μ : Measure X) [IsProbabilityMeasure μ]
    (D : Set X) (hD : 0 < μ.real D) (p : ℝ) (hp₀ : 0 ≤ p) (hp₁ : p ≤ 1) :
    disagreementMean μ D (CorrectnessField.constant p hp₀ hp₁) = p := by
  simp only [disagreementMean, CorrectnessField.constant, ENNReal.toReal_ofReal hp₀,
    setIntegral_const, smul_eq_mul]
  exact mul_div_cancel_left₀ p (ne_of_gt hD)

/-- Exact population reduction with the residual defined from the measurable
correctness field.  Positive disagreement mass is explicit. -/
theorem measurable_target_benefit_reduction (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀]
    (hD : 0 < μ.real {x | f₀ x ≠ fₐ x}) (η : CorrectnessField X) :
    populationBenefit f₀ fₐ
      (μ ⊗ₘ targetLabelKernel f₀ fₐ h₀ hₐ κ₀ η.value η.measurable_value) =
      2 * μ.real {x | f₀ x ≠ fₐ x} *
        (disagreementMean μ {x | f₀ x ≠ fₐ x} η - 1 / 2) := by
  rw [constructed_target_population_benefit μ f₀ fₐ h₀ hₐ κ₀ _ _ η.value_le_one]
  rw [integral_sub ((correctnessField_integrable _ η).const_mul 2) (integrable_const 1)]
  rw [integral_const_mul]
  simp only [setIntegral_const, smul_eq_mul, mul_one, disagreementMean]
  field_simp

omit [MeasurableSingletonClass Y] [MeasurableEq Y] in
/-- The exact feasible identified interval is derived, including its clipping
by valid correctness probabilities.  Unlike the old `RichAt` premise, this does
not demand infeasible residual values when the budget is large. -/
theorem measurable_correctness_identified_interval (μ : Measure X) [IsProbabilityMeasure μ]
    (D : Set X) (hD : 0 < μ.real D) (M β z : ℝ) :
    (∃ η : CorrectnessField X,
      |disagreementMean μ D η - 1 / 2 - M| ≤ β ∧
      disagreementMean μ D η - 1 / 2 = z) ↔
      max (-1 / 2) (M - β) ≤ z ∧ z ≤ min (1 / 2) (M + β) := by
  constructor
  · rintro ⟨η, hβ, hz⟩
    obtain ⟨hlo, hhi⟩ := disagreementMean_bounds μ D hD η
    rw [hz] at hβ
    rw [max_le_iff, le_min_iff]
    obtain ⟨hβlo, hβhi⟩ := abs_le.mp hβ
    constructor <;> constructor <;> linarith
  · rintro ⟨hzlo, hzhi⟩
    obtain ⟨hlo, hbudgetlo⟩ := max_le_iff.mp hzlo
    obtain ⟨hhi, hbudgethi⟩ := le_min_iff.mp hzhi
    have hp₀ : 0 ≤ z + 1 / 2 := by linarith
    have hp₁ : z + 1 / 2 ≤ 1 := by linarith
    refine ⟨CorrectnessField.constant (z + 1 / 2) hp₀ hp₁, ?_, ?_⟩
    · rw [disagreementMean_constant μ D hD]
      apply abs_le.mpr
      constructor <;> linarith
    · rw [disagreementMean_constant μ D hD]
      ring

/-- Every point of the identified interval is realized by an actual target
probability law with fixed off-disagreement labels and fixed evidence law. -/
theorem measurable_target_frontier_attainment (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀]
    (hD : 0 < μ.real {x | f₀ x ≠ fₐ x}) (M β z : ℝ)
    (hz : max (-1 / 2) (M - β) ≤ z ∧ z ≤ min (1 / 2) (M + β)) :
    ∃ (η : CorrectnessField X) (κ : Kernel X Y),
      IsMarkovKernel κ ∧ IsProbabilityMeasure (μ ⊗ₘ κ) ∧
      |disagreementMean μ {x | f₀ x ≠ fₐ x} η - 1 / 2 - M| ≤ β ∧
      (∀ x, f₀ x ≠ fₐ x → κ x {fₐ x} = η.value x) ∧
      (∀ x, f₀ x = fₐ x → κ x = κ₀ x) ∧
      (μ ⊗ₘ κ).fst = μ ∧
      (∀ (g : X → E), Measurable g → (μ ⊗ₘ κ).map (fun xy => g xy.1) = μ.map g) ∧
      populationBenefit f₀ fₐ (μ ⊗ₘ κ) = 2 * μ.real {x | f₀ x ≠ fₐ x} * z := by
  obtain ⟨η, hβ, hzη⟩ := (measurable_correctness_identified_interval μ _ hD M β z).mpr hz
  let κ := targetLabelKernel f₀ fₐ h₀ hₐ κ₀ η.value η.measurable_value
  letI := targetLabelKernel_isMarkov f₀ fₐ h₀ hₐ κ₀ η.value η.measurable_value η.value_le_one
  refine ⟨η, κ, inferInstance, inferInstance, hβ, ?_, ?_, Measure.fst_compProd μ κ, ?_, ?_⟩
  · exact fun _ hx => targetLabelKernel_on_disagreement f₀ fₐ h₀ hₐ κ₀ _ η.measurable_value hx
  · exact fun _ hx => targetLabelKernel_off_disagreement f₀ fₐ h₀ hₐ κ₀ _ η.measurable_value hx
  · exact fun g hg => target_label_free_law μ κ g hg
  · rw [measurable_target_benefit_reduction μ f₀ fₐ h₀ hₐ κ₀ hD η, hzη]

end KBound
