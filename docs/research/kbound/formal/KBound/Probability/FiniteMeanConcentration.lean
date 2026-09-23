import KBound.Probability.Concentration

/-! Finite-index common-mean concentration. This removes the need to extend a
finite selected subfamily to an infinite independent sequence. -/
namespace KBound
open MeasureTheory ProbabilityTheory
open scoped NNReal ENNReal BigOperators
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

theorem finset_unit_interval_mean_tail [IsProbabilityMeasure μ]
    {ι : Type*} {X : ι → Ω → ℝ} (h_indep : iIndepFun X μ) {s : Finset ι} (hn : 0 < s.card)
    (h_meas : ∀ i ∈ s, AEMeasurable (X i) μ)
    (h_bounds : ∀ i ∈ s, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    {eps : ℝ} (heps : 0 ≤ eps) :
    μ.real {ω | eps ≤ |(∑ i ∈ s, (X i ω - ∫ x, X i x ∂μ)) / s.card|} ≤
      2 * Real.exp (-2 * s.card * eps ^ 2) := by
  have hnR : (0 : ℝ) < s.card := by exact_mod_cast hn
  have h_centered : iIndepFun (fun i ω ↦ X i ω - ∫ x, X i x ∂μ) μ :=
    h_indep.comp (fun i x ↦ x - ∫ ω, X i ω ∂μ) (fun _ ↦ measurable_id.sub_const _)
  have h_subG : ∀ i ∈ s, HasSubgaussianMGF
      (fun ω ↦ X i ω - ∫ x, X i x ∂μ) ((1 / 2 : ℝ≥0) ^ 2) μ := by
    intro i hi
    simpa using hasSubgaussianMGF_of_mem_Icc
      (h_meas i hi) (h_bounds i hi)
  have htail := subgaussian_abs_tail
    (HasSubgaussianMGF.sum_of_iIndepFun h_centered h_subG) (mul_nonneg hnR.le heps)
  have hevent : {ω | eps ≤ |(∑ i ∈ s, (X i ω - ∫ x, X i x ∂μ)) / s.card|} =
      {ω | (s.card : ℝ) * eps ≤ |∑ i ∈ s, (X i ω - ∫ x, X i x ∂μ)|} := by
    ext ω
    simp only [Set.mem_setOf_eq, abs_div, abs_of_pos hnR, le_div_iff₀ hnR]
    rw [mul_comm]
  rw [hevent]
  convert htail using 1
  congr 2
  simp only [Finset.sum_const, nsmul_eq_mul, NNReal.coe_mul,
    NNReal.coe_natCast, NNReal.coe_pow, NNReal.coe_div, NNReal.coe_one, NNReal.coe_ofNat]
  field_simp

/-- The radius used in `Rates` has level-alpha population-mean coverage for
independent unit-range scores. The width-one hypothesis is essential: a raw
paired benefit in `[-1,1]` has width two and needs twice this radius. -/
theorem finset_unit_interval_hoeffding_coverage [IsProbabilityMeasure μ]
    {ι : Type*} {X : ι → Ω → ℝ} (h_indep : iIndepFun X μ) {s : Finset ι} (hn : 0 < s.card)
    (h_meas : ∀ i ∈ s, AEMeasurable (X i) μ)
    (h_bounds : ∀ i ∈ s, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    {alpha : ℝ} (hα : 0 < alpha) (hα1 : alpha ≤ 1) :
    1 - alpha ≤ μ.real {ω |
      |(∑ i ∈ s, (X i ω - ∫ x, X i x ∂μ)) / s.card| ≤
        hoeffdingRadius s.card alpha} := by
  have hnR : (0 : ℝ) < s.card := by exact_mod_cast hn
  have hlog : 0 ≤ Real.log (2 / alpha) :=
    Real.log_nonneg ((le_div_iff₀ hα).2 (by linarith))
  have hsquare : (hoeffdingRadius s.card alpha) ^ 2 = Real.log (2 / alpha) / (2 * s.card) :=
    Real.sq_sqrt (div_nonneg hlog (by positivity))
  have hexponent : -2 * (s.card : ℝ) * (hoeffdingRadius s.card alpha) ^ 2 =
      -Real.log (2 / alpha) := by
    rw [hsquare]
    field_simp
  have hconstant : 2 * Real.exp (-2 * (s.card : ℝ) * (hoeffdingRadius s.card alpha) ^ 2) = alpha := by
    rw [hexponent, Real.exp_neg, Real.exp_log (div_pos (by norm_num) hα)]
    field_simp
  have htail := finset_unit_interval_mean_tail h_indep hn h_meas h_bounds
    (eps := hoeffdingRadius s.card alpha) (Real.sqrt_nonneg _)
  rw [hconstant] at htail
  let M : Ω → ℝ := fun ω ↦
    (∑ i ∈ s, (X i ω - ∫ x, X i x ∂μ)) / s.card
  have hM : AEMeasurable M μ := by
    apply AEMeasurable.div_const
    exact Finset.aemeasurable_fun_sum (s) (fun i hi ↦
      (h_meas i hi).sub_const (∫ x, X i x ∂μ))
  have hsubset : {ω | hoeffdingRadius s.card alpha < |M ω|} ⊆
      {ω | hoeffdingRadius s.card alpha ≤ |M ω|} := by
    intro ω hω
    exact (show hoeffdingRadius s.card alpha ≤ |M ω| from le_of_lt hω)
  have hbad : μ.real {ω | hoeffdingRadius s.card alpha < |M ω|} ≤ alpha :=
    (measureReal_mono (μ := μ) hsubset).trans htail
  have hcompl := probReal_compl_eq_one_sub₀ (μ := μ)
    (s := {ω | hoeffdingRadius s.card alpha < |M ω|})
    (nullMeasurableSet_lt aemeasurable_const hM.abs)
  have hgood : {ω | hoeffdingRadius s.card alpha < |M ω|}ᶜ =
      {ω | |M ω| ≤ hoeffdingRadius s.card alpha} := by
    ext ω
    simp
  rw [hgood] at hcompl
  change 1 - alpha ≤ μ.real {ω | |M ω| ≤ hoeffdingRadius s.card alpha}
  linarith

/-- Population-mean formulation when the independent unit-range variables share
the declared mean `delta`. Common means and independence suffice; identical
distributions are not required. -/
theorem finset_common_mean_hoeffding_coverage [IsProbabilityMeasure μ]
    {ι : Type*} {X : ι → Ω → ℝ} (h_indep : iIndepFun X μ) {s : Finset ι} (hn : 0 < s.card)
    (h_meas : ∀ i ∈ s, AEMeasurable (X i) μ)
    (h_bounds : ∀ i ∈ s, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    {delta : ℝ} (h_mean : ∀ i ∈ s, (∫ x, X i x ∂μ) = delta)
    {alpha : ℝ} (hα : 0 < alpha) (hα1 : alpha ≤ 1) :
    1 - alpha ≤ μ.real {ω |
      |(∑ i ∈ s, X i ω) / s.card - delta| ≤ hoeffdingRadius s.card alpha} := by
  have hnR : (s.card : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  have hsum : (∑ i ∈ s, ∫ x, X i x ∂μ) = (s.card : ℝ) * delta := by
    calc
      _ = ∑ _i ∈ s, delta :=
        Finset.sum_congr rfl (fun i hi ↦ h_mean i hi)
      _ = _ := by simp
  have hform (ω : Ω) :
      (∑ i ∈ s, (X i ω - ∫ x, X i x ∂μ)) / s.card =
        (∑ i ∈ s, X i ω) / s.card - delta := by
    rw [Finset.sum_sub_distrib, hsum]
    field_simp
  simpa only [hform] using
    finset_unit_interval_hoeffding_coverage h_indep hn h_meas h_bounds hα hα1


end KBound
