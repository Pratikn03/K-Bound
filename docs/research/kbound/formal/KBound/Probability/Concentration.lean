import KBound.Probability.Rates
import Mathlib.Probability.Moments.SubGaussian
import Mathlib.MeasureTheory.Measure.Real
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity

/-!
# Probability concentration for calibrated scores

Unlike the deterministic radius lemmas in `Rates`, this module proves actual
probability bounds. The bounded-independent result uses Hoeffding's lemma and
independence to control the moment generating function of the sum. The adapted
result assumes conditional sub-Gaussian moment generating functions with respect
to the declared filtration and derives an Azuma-Hoeffding tail bound.

These are population-mean bounds under their explicit sampling assumptions. They
do not transfer an empirical cell-score interval to a population-risk target, and
do not prove a nonlinear estimator's Lipschitz constant, empirical-Bernstein
adaptivity, or an unlisted model-specific minimax rate.
-/

open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal BigOperators ProbabilityTheory

namespace KBound

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}

/-- Two-sided Chernoff bound derived from both one-sided sub-Gaussian tails. -/
theorem subgaussian_abs_tail [IsProbabilityMeasure μ]
    {X : Ω → ℝ} {c : ℝ≥0} (hX : HasSubgaussianMGF X c μ)
    {t : ℝ} (ht : 0 ≤ t) :
    μ.real {ω | t ≤ |X ω|} ≤ 2 * Real.exp (-t ^ 2 / (2 * (c : ℝ))) := by
  have hs : {ω | t ≤ |X ω|} ⊆ {ω | t ≤ X ω} ∪ {ω | t ≤ -X ω} := by
    intro ω hω
    rcases le_total 0 (X ω) with hpos | hneg
    · exact Or.inl (by simpa only [Set.mem_setOf_eq, abs_of_nonneg hpos] using hω)
    · exact Or.inr (by simpa only [Set.mem_setOf_eq, abs_of_nonpos hneg] using hω)
  calc
    _ ≤ μ.real ({ω | t ≤ X ω} ∪ {ω | t ≤ -X ω}) := measureReal_mono hs
    _ ≤ μ.real {ω | t ≤ X ω} + μ.real {ω | t ≤ -X ω} := measureReal_union_le _ _
    _ ≤ Real.exp (-t ^ 2 / (2 * (c : ℝ))) + Real.exp (-t ^ 2 / (2 * (c : ℝ))) :=
      add_le_add (hX.measure_ge_le ht) (hX.neg.measure_ge_le ht)
    _ = _ := by ring

/-- Hoeffding's two-sided inequality for independent, possibly non-identically
distributed bounded variables. Centering is by each variable's true expectation;
`c i = ((b i - a i)/2)^2` is the proved Hoeffding variance proxy. -/
theorem bounded_independent_sum_tail [IsProbabilityMeasure μ]
    {ι : Type*} {X : ι → Ω → ℝ} (h_indep : iIndepFun X μ)
    {s : Finset ι} {a b : ι → ℝ}
    (h_meas : ∀ i ∈ s, AEMeasurable (X i) μ)
    (h_bounds : ∀ i ∈ s, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (a i) (b i))
    {t : ℝ} (ht : 0 ≤ t) :
    μ.real {ω | t ≤ |∑ i ∈ s, (X i ω - ∫ x, X i x ∂μ)|} ≤
      2 * Real.exp (-t ^ 2 /
        (2 * ((∑ i ∈ s, ((‖b i - a i‖₊ / 2) ^ 2)) : ℝ≥0))) := by
  have h_centered : iIndepFun (fun i ω ↦ X i ω - ∫ x, X i x ∂μ) μ :=
    h_indep.comp (fun i x ↦ x - ∫ ω, X i ω ∂μ) (fun _ ↦ measurable_id.sub_const _)
  have h_subG : ∀ i ∈ s, HasSubgaussianMGF
      (fun ω ↦ X i ω - ∫ x, X i x ∂μ) ((‖b i - a i‖₊ / 2) ^ 2) μ := by
    intro i hi
    exact hasSubgaussianMGF_of_mem_Icc (h_meas i hi) (h_bounds i hi)
  exact subgaussian_abs_tail (HasSubgaussianMGF.sum_of_iIndepFun h_centered h_subG) ht

/-- Hoeffding concentration for the centered mean of independent `[0,1]` scores.
The sample count is positive; the right side is `2 exp(-2 n eps²)`. -/
theorem unit_interval_mean_tail [IsProbabilityMeasure μ]
    {X : ℕ → Ω → ℝ} (h_indep : iIndepFun X μ) {n : ℕ} (hn : 0 < n)
    (h_meas : ∀ i < n, AEMeasurable (X i) μ)
    (h_bounds : ∀ i < n, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    {eps : ℝ} (heps : 0 ≤ eps) :
    μ.real {ω | eps ≤ |(∑ i ∈ Finset.range n, (X i ω - ∫ x, X i x ∂μ)) / n|} ≤
      2 * Real.exp (-2 * n * eps ^ 2) := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have h_centered : iIndepFun (fun i ω ↦ X i ω - ∫ x, X i x ∂μ) μ :=
    h_indep.comp (fun i x ↦ x - ∫ ω, X i ω ∂μ) (fun _ ↦ measurable_id.sub_const _)
  have h_subG : ∀ i ∈ Finset.range n, HasSubgaussianMGF
      (fun ω ↦ X i ω - ∫ x, X i x ∂μ) ((1 / 2 : ℝ≥0) ^ 2) μ := by
    intro i hi
    simpa using hasSubgaussianMGF_of_mem_Icc
      (h_meas i (Finset.mem_range.mp hi)) (h_bounds i (Finset.mem_range.mp hi))
  have htail := subgaussian_abs_tail
    (HasSubgaussianMGF.sum_of_iIndepFun h_centered h_subG) (mul_nonneg hnR.le heps)
  have hevent : {ω | eps ≤ |(∑ i ∈ Finset.range n, (X i ω - ∫ x, X i x ∂μ)) / n|} =
      {ω | (n : ℝ) * eps ≤ |∑ i ∈ Finset.range n, (X i ω - ∫ x, X i x ∂μ)|} := by
    ext ω
    simp only [Set.mem_setOf_eq, abs_div, abs_of_pos hnR, le_div_iff₀ hnR]
    rw [mul_comm]
  rw [hevent]
  convert htail using 1
  congr 2
  simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul, NNReal.coe_mul,
    NNReal.coe_natCast, NNReal.coe_pow, NNReal.coe_div, NNReal.coe_one, NNReal.coe_ofNat]
  field_simp

/-- The radius used in `Rates` has level-alpha population-mean coverage for
independent unit-range scores. The width-one hypothesis is essential: a raw
paired benefit in `[-1,1]` has width two and needs twice this radius. -/
theorem unit_interval_hoeffding_coverage [IsProbabilityMeasure μ]
    {X : ℕ → Ω → ℝ} (h_indep : iIndepFun X μ) {n : ℕ} (hn : 0 < n)
    (h_meas : ∀ i < n, AEMeasurable (X i) μ)
    (h_bounds : ∀ i < n, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    {alpha : ℝ} (hα : 0 < alpha) (hα1 : alpha ≤ 1) :
    1 - alpha ≤ μ.real {ω |
      |(∑ i ∈ Finset.range n, (X i ω - ∫ x, X i x ∂μ)) / n| ≤
        hoeffdingRadius n alpha} := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hlog : 0 ≤ Real.log (2 / alpha) :=
    Real.log_nonneg ((le_div_iff₀ hα).2 (by linarith))
  have hsquare : (hoeffdingRadius n alpha) ^ 2 = Real.log (2 / alpha) / (2 * n) :=
    Real.sq_sqrt (div_nonneg hlog (by positivity))
  have hexponent : -2 * (n : ℝ) * (hoeffdingRadius n alpha) ^ 2 =
      -Real.log (2 / alpha) := by
    rw [hsquare]
    field_simp
  have hconstant : 2 * Real.exp (-2 * (n : ℝ) * (hoeffdingRadius n alpha) ^ 2) = alpha := by
    rw [hexponent, Real.exp_neg, Real.exp_log (div_pos (by norm_num) hα)]
    field_simp
  have htail := unit_interval_mean_tail h_indep hn h_meas h_bounds
    (eps := hoeffdingRadius n alpha) (Real.sqrt_nonneg _)
  rw [hconstant] at htail
  let M : Ω → ℝ := fun ω ↦
    (∑ i ∈ Finset.range n, (X i ω - ∫ x, X i x ∂μ)) / n
  have hM : AEMeasurable M μ := by
    apply AEMeasurable.div_const
    exact Finset.aemeasurable_fun_sum (Finset.range n) (fun i hi ↦
      (h_meas i (Finset.mem_range.mp hi)).sub_const (∫ x, X i x ∂μ))
  have hsubset : {ω | hoeffdingRadius n alpha < |M ω|} ⊆
      {ω | hoeffdingRadius n alpha ≤ |M ω|} := by
    intro ω hω
    exact (show hoeffdingRadius n alpha ≤ |M ω| from le_of_lt hω)
  have hbad : μ.real {ω | hoeffdingRadius n alpha < |M ω|} ≤ alpha :=
    (measureReal_mono (μ := μ) hsubset).trans htail
  have hcompl := probReal_compl_eq_one_sub₀ (μ := μ)
    (s := {ω | hoeffdingRadius n alpha < |M ω|})
    (nullMeasurableSet_lt aemeasurable_const hM.abs)
  have hgood : {ω | hoeffdingRadius n alpha < |M ω|}ᶜ =
      {ω | |M ω| ≤ hoeffdingRadius n alpha} := by
    ext ω
    simp
  rw [hgood] at hcompl
  change 1 - alpha ≤ μ.real {ω | |M ω| ≤ hoeffdingRadius n alpha}
  linarith

/-- Population-mean formulation when the independent unit-range variables share
the declared mean `delta`. Common means and independence suffice; identical
distributions are not required. -/
theorem common_mean_hoeffding_coverage [IsProbabilityMeasure μ]
    {X : ℕ → Ω → ℝ} (h_indep : iIndepFun X μ) {n : ℕ} (hn : 0 < n)
    (h_meas : ∀ i < n, AEMeasurable (X i) μ)
    (h_bounds : ∀ i < n, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    {delta : ℝ} (h_mean : ∀ i < n, (∫ x, X i x ∂μ) = delta)
    {alpha : ℝ} (hα : 0 < alpha) (hα1 : alpha ≤ 1) :
    1 - alpha ≤ μ.real {ω |
      |(∑ i ∈ Finset.range n, X i ω) / n - delta| ≤ hoeffdingRadius n alpha} := by
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  have hsum : (∑ i ∈ Finset.range n, ∫ x, X i x ∂μ) = (n : ℝ) * delta := by
    calc
      _ = ∑ _i ∈ Finset.range n, delta :=
        Finset.sum_congr rfl (fun i hi ↦ h_mean i (Finset.mem_range.mp hi))
      _ = _ := by simp
  have hform (ω : Ω) :
      (∑ i ∈ Finset.range n, (X i ω - ∫ x, X i x ∂μ)) / n =
        (∑ i ∈ Finset.range n, X i ω) / n - delta := by
    rw [Finset.sum_sub_distrib, hsum]
    field_simp
  simpa only [hform] using
    unit_interval_hoeffding_coverage h_indep hn h_meas h_bounds hα hα1

/-- Independent paired benefits in `[-1,1]` have population-mean coverage using
the correct width-two radius `2 * hoeffdingRadius n alpha`. This is not the
smaller radius applicable to unit-range accuracies. -/
theorem paired_benefit_hoeffding_coverage [IsProbabilityMeasure μ]
    {X : ℕ → Ω → ℝ} (h_indep : iIndepFun X μ) {n : ℕ} (hn : 0 < n)
    (h_meas : ∀ i < n, AEMeasurable (X i) μ)
    (h_bounds : ∀ i < n, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (-1 : ℝ) 1)
    {delta : ℝ} (h_mean : ∀ i < n, (∫ x, X i x ∂μ) = delta)
    {alpha : ℝ} (hα : 0 < alpha) (hα1 : alpha ≤ 1) :
    1 - alpha ≤ μ.real {ω |
      |(∑ i ∈ Finset.range n, X i ω) / n - delta| ≤ 2 * hoeffdingRadius n alpha} := by
  let Y : ℕ → Ω → ℝ := fun i ω ↦ (X i ω + 1) / 2
  have hY_indep : iIndepFun Y μ :=
    h_indep.comp (fun _ x ↦ (x + 1) / 2) (fun _ ↦ (measurable_id.add_const 1).div_const 2)
  have hY_meas : ∀ i < n, AEMeasurable (Y i) μ :=
    fun i hi ↦ ((h_meas i hi).add_const 1).div_const 2
  have hY_bounds : ∀ i < n, ∀ᵐ ω ∂μ, Y i ω ∈ Set.Icc (0 : ℝ) 1 := by
    intro i hi
    filter_upwards [h_bounds i hi] with ω hω
    change 0 ≤ (X i ω + 1) / 2 ∧ (X i ω + 1) / 2 ≤ 1
    constructor <;> linarith [hω.1, hω.2]
  have hY_mean : ∀ i < n, (∫ x, Y i x ∂μ) = (delta + 1) / 2 := by
    intro i hi
    have hXi : Integrable (X i) μ :=
      Integrable.of_mem_Icc (-1) 1 (h_meas i hi) (h_bounds i hi)
    change (∫ x, (X i x + 1) / 2 ∂μ) = (delta + 1) / 2
    rw [integral_div, integral_add hXi (integrable_const 1), h_mean i hi]
    simp
  have hcov := common_mean_hoeffding_coverage hY_indep hn hY_meas hY_bounds hY_mean hα hα1
  have hnR : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  have hform (ω : Ω) : (∑ i ∈ Finset.range n, Y i ω) / n - (delta + 1) / 2 =
      ((∑ i ∈ Finset.range n, X i ω) / n - delta) / 2 := by
    simp only [Y, ← Finset.sum_div, Finset.sum_add_distrib,
      Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one]
    field_simp
    ring
  have hevent : {ω | |(∑ i ∈ Finset.range n, Y i ω) / n - (delta + 1) / 2| ≤
      hoeffdingRadius n alpha} =
      {ω | |(∑ i ∈ Finset.range n, X i ω) / n - delta| ≤ 2 * hoeffdingRadius n alpha} := by
    ext ω
    rw [Set.mem_setOf_eq, Set.mem_setOf_eq, hform, abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
    rw [div_le_iff₀ (by norm_num : (0 : ℝ) < 2), mul_comm]
  rwa [hevent] at hcov

/-- A declared adapted sequence of conditional sub-Gaussian increments satisfies
the two-sided Azuma-Hoeffding tail bound. The conditional-MGF hypothesis is with
respect to the past sigma-algebra, not a marginal variance or an observed sign. -/
theorem adapted_subgaussian_sum_tail [IsProbabilityMeasure μ] [StandardBorelSpace Ω]
    {Y : ℕ → Ω → ℝ} {c : ℕ → ℝ≥0} {ℱ : Filtration ℕ mΩ}
    (h_adapted : StronglyAdapted ℱ Y)
    (h0 : HasSubgaussianMGF (Y 0) (c 0) μ) (n : ℕ)
    (h_cond : ∀ i < n - 1,
      HasCondSubgaussianMGF (ℱ i) (ℱ.le i) (Y (i + 1)) (c (i + 1)) μ)
    {t : ℝ} (ht : 0 ≤ t) :
    μ.real {ω | t ≤ |∑ i ∈ Finset.range n, Y i ω|} ≤
      2 * Real.exp (-t ^ 2 / (2 * ((∑ i ∈ Finset.range n, c i) : ℝ≥0))) := by
  exact subgaussian_abs_tail
    (HasSubgaussianMGF.sum_of_hasCondSubgaussianMGF h_adapted h0 n h_cond) ht

/-- Conditional Hoeffding's lemma: boundedness and a zero conditional mean imply
the conditional-MGF bound. Thus the martingale tail theorem below does not take
the desired concentration behavior as an unexplained premise. -/
theorem conditional_hoeffding_of_bounded_zero_mean
    [IsProbabilityMeasure μ] [StandardBorelSpace Ω]
    {m : MeasurableSpace Ω} (hm : m ≤ mΩ) {X : Ω → ℝ} {a b : ℝ}
    (h_meas : Measurable[mΩ] X) (h_bounds : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc a b)
    (h_zero : μ[X | m] =ᵐ[μ] (0 : Ω → ℝ)) :
    HasCondSubgaussianMGF m hm X ((‖b - a‖₊ / 2) ^ 2) μ := by
  letI : MeasurableSpace Ω := mΩ
  have hint : Integrable X μ := Integrable.of_mem_Icc a b h_meas.aemeasurable h_bounds
  have h_kernel_bounds : ∀ᵐ ω ∂(μ.trim hm),
      ∀ᵐ x ∂(condExpKernel μ m ω), X x ∈ Set.Icc a b := by
    apply Measure.ae_ae_of_ae_comp
    rw [condExpKernel_comp_trim hm]
    exact h_bounds
  have h_zero_trim : μ[X | m] =ᵐ[μ.trim hm] (0 : Ω → ℝ) :=
    StronglyMeasurable.ae_eq_trim_of_stronglyMeasurable hm
      stronglyMeasurable_condExp stronglyMeasurable_zero h_zero
  have h_cond := condExp_ae_eq_trim_integral_condExpKernel hm hint
  constructor
  · intro t
    rw [condExpKernel_comp_trim hm]
    exact integrable_exp_mul_of_mem_Icc h_meas.aemeasurable h_bounds
  · filter_upwards [h_kernel_bounds, h_zero_trim, h_cond] with ω hb hz hc
    have hmean : (∫ x, X x ∂condExpKernel μ m ω) = 0 := hc.symm.trans hz
    exact (hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero h_meas.aemeasurable hb hmean).mgf_le

/-- **Azuma-Hoeffding from bounded martingale differences.** The adapted
increments have zero conditional mean given the past, and are almost surely in
their stated deterministic intervals. No independence assumption is used. -/
theorem bounded_martingale_difference_tail [IsProbabilityMeasure μ] [StandardBorelSpace Ω]
    {Y : ℕ → Ω → ℝ} {a b : ℕ → ℝ} {ℱ : Filtration ℕ mΩ}
    (h_adapted : StronglyAdapted ℱ Y)
    (h_bounds : ∀ i, ∀ᵐ ω ∂μ, Y i ω ∈ Set.Icc (a i) (b i))
    (h0 : (∫ ω, Y 0 ω ∂μ) = 0)
    (h_cond_zero : ∀ i, μ[Y (i + 1) | ℱ i] =ᵐ[μ] (0 : Ω → ℝ))
    (n : ℕ) {t : ℝ} (ht : 0 ≤ t) :
    μ.real {ω | t ≤ |∑ i ∈ Finset.range n, Y i ω|} ≤
      2 * Real.exp (-t ^ 2 /
        (2 * ((∑ i ∈ Finset.range n, ((‖b i - a i‖₊ / 2) ^ 2)) : ℝ≥0))) := by
  apply adapted_subgaussian_sum_tail h_adapted
    (hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero
      h_adapted.stronglyMeasurable.aemeasurable (h_bounds 0) h0) n _ ht
  intro i _
  exact conditional_hoeffding_of_bounded_zero_mean (ℱ.le i)
    h_adapted.stronglyMeasurable.measurable (h_bounds (i + 1)) (h_cond_zero i)

end KBound
