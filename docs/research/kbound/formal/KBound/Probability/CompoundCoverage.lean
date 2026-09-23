import KBound.Probability.MeasureConformal
import KBound.Probability.CalibrationOrderStatistic
import KBound.Probability.Concentration
import KBound.Probability.RandomRadiusCertificate
import Mathlib.Probability.Kernel.Composition.MeasureCompProd
import Mathlib.MeasureTheory.Integral.Lebesgue.Basic

/-!
# Compound population coverage with an explicit conditional sampling kernel

The environment law and conditional evaluation kernel are actual measures.
The concentration premise is bounded independent sampling with a shared mean,
not a postulated tail bound. Conformal scores are exchangeable on the joint
experiment; no conditional conformal coverage given the environment is assumed.
-/

namespace KBound
open MeasureTheory ProbabilityTheory Set
open scoped ENNReal BigOperators ProbabilityTheory

variable {Ω E : Type*} [MeasurableSpace Ω] [MeasurableSpace E]

/-- The common probability argument converting real coverage to an ENNReal miss bound. -/
theorem real_coverage_miss_le {μ : Measure Ω} [IsProbabilityMeasure μ]
    {s : Set Ω} (hs : MeasurableSet s) {a : ℝ}
    (hcov : 1 - a ≤ μ.real s) : μ sᶜ ≤ ENNReal.ofReal a := by
  have hc := probReal_compl_eq_one_sub (μ := μ) hs
  have hb : μ.real sᶜ ≤ a := by linarith
  rw [← ofReal_measureReal (μ := μ) (s := sᶜ) (measure_ne_top μ sᶜ)]
  exact ENNReal.ofReal_le_ofReal hb

/-- Intersecting an empirical-cell interval and a sampling interval transfers
coverage to the population mean, allowing the empirical radius to be infinite. -/
theorem extended_compound_containment {estimate cell target b : ℝ}
    {eps : ℝ≥0∞} (hb : 0 ≤ b)
    (hc : ENNReal.ofReal |estimate - cell| ≤ eps)
    (hs : |cell - target| ≤ b) :
    ENNReal.ofReal |estimate - target| ≤ eps + ENNReal.ofReal b := by
  have ht : |estimate - target| ≤ |estimate - cell| + b := by
    calc
      |estimate - target| = |(estimate - cell) + (cell - target)| := by congr 1; ring
      _ ≤ |estimate - cell| + |cell - target| := abs_add_le _ _
      _ ≤ |estimate - cell| + b := add_le_add le_rfl hs
  calc
    ENNReal.ofReal |estimate - target| ≤ ENNReal.ofReal (|estimate - cell| + b) :=
      ENNReal.ofReal_le_ofReal ht
    _ = ENNReal.ofReal |estimate - cell| + ENNReal.ofReal b := ENNReal.ofReal_add (abs_nonneg _) hb
    _ ≤ eps + ENNReal.ofReal b := add_le_add hc le_rfl

omit [MeasurableSpace Ω] in
/-- The compound interval's miss event is contained in the union of two misses;
no independence between the two sources of uncertainty is used. -/
theorem compound_miss_subset (estimate cell target : Ω → ℝ)
    (eps : Ω → ℝ≥0∞) (b : Ω → ℝ) (hb : ∀ ω, 0 ≤ b ω) :
    {ω | ¬ ENNReal.ofReal |estimate ω - target ω| ≤ eps ω + ENNReal.ofReal (b ω)} ⊆
      {ω | ¬ ENNReal.ofReal |estimate ω - cell ω| ≤ eps ω} ∪
      {ω | b ω < |cell ω - target ω|} := by
  intro ω hm
  by_cases hc : ENNReal.ofReal |estimate ω - cell ω| ≤ eps ω
  · right
    by_contra hs
    exact hm (extended_compound_containment (hb ω) hc (le_of_not_gt hs))
  · exact Or.inl hc

/-- Probability assembly for a compound interval, including infinite radii. -/
theorem compound_miss_le {μ : Measure Ω} [IsProbabilityMeasure μ]
    (estimate cell target : Ω → ℝ) (eps : Ω → ℝ≥0∞) (b : Ω → ℝ)
    (hb : ∀ ω, 0 ≤ b ω) {a d : ℝ≥0∞}
    (hcell : μ {ω | ¬ ENNReal.ofReal |estimate ω - cell ω| ≤ eps ω} ≤ a)
    (hsample : μ {ω | b ω < |cell ω - target ω|} ≤ d) :
    μ {ω | ¬ ENNReal.ofReal |estimate ω - target ω| ≤ eps ω + ENNReal.ofReal (b ω)} ≤ a + d := by
  exact (measure_mono (compound_miss_subset estimate cell target eps b hb)).trans
    ((measure_union_le _ _).trans (add_le_add hcell hsample))

/-- Averaging conditional event bounds over an actual environment/kernel law. -/
theorem kernel_event_bound {ν : Measure E} [IsProbabilityMeasure ν]
    (κ : Kernel E Ω) [IsMarkovKernel κ] {s : Set (E × Ω)}
    (hs : MeasurableSet s) {a : ℝ≥0∞}
    (hcond : ∀ e, κ e {ω | (e, ω) ∈ s} ≤ a) :
    (ν ⊗ₘ κ) s ≤ a := by
  rw [Measure.compProd_apply hs]
  calc
    (∫⁻ e, κ e (Prod.mk e ⁻¹' s) ∂ν) ≤ ∫⁻ _e, a ∂ν := lintegral_mono hcond
    _ = a := by simp

/-- Conditional paired-accuracy concentration, integrated over any environment
law. The kernel assumptions concern fresh evaluation observations after the
pair/environment has been fixed. -/
theorem kernel_paired_sampling_miss_le {ν : Measure E} [IsProbabilityMeasure ν]
    (κ : Kernel E Ω) [IsMarkovKernel κ]
    (W : ℕ → E × Ω → ℝ) (hW : ∀ i, Measurable (W i))
    (delta : E → ℝ) (hd : Measurable delta) {m : ℕ} (hm : 0 < m)
    (hindep : ∀ e, iIndepFun (fun i ω => W i (e, ω)) (κ e))
    (hbound : ∀ e i, i < m → ∀ᵐ ω ∂κ e, W i (e, ω) ∈ Icc (-1 : ℝ) 1)
    (hmean : ∀ e i, i < m → (∫ ω, W i (e, ω) ∂κ e) = delta e)
    {a : ℝ} (ha : 0 < a) (ha1 : a ≤ 1) :
    (ν ⊗ₘ κ) {z | 2 * hoeffdingRadius m a <
      |(∑ i ∈ Finset.range m, W i z) / m - delta z.1|} ≤ ENNReal.ofReal a := by
  have hB : Measurable (fun z => (∑ i ∈ Finset.range m, W i z) / (m : ℝ)) := by fun_prop
  have hbad : MeasurableSet {z | 2 * hoeffdingRadius m a <
      |(∑ i ∈ Finset.range m, W i z) / m - delta z.1|} := by
    exact measurableSet_lt measurable_const ((hB.sub (hd.comp measurable_fst)).abs)
  apply kernel_event_bound κ hbad
  intro e
  have hc := paired_benefit_hoeffding_coverage (hindep e) hm
    (fun i _ => ((hW i).comp measurable_prodMk_left).aemeasurable)
    (hbound e) (hmean e) ha ha1
  have hgood : MeasurableSet {ω |
      |(∑ i ∈ Finset.range m, W i (e, ω)) / m - delta e| ≤
        2 * hoeffdingRadius m a} :=
    measurableSet_le (((hB.comp measurable_prodMk_left).sub_const (delta e)).abs) measurable_const
  have hh := real_coverage_miss_le hgood hc
  simpa only [compl_setOf, not_le] using hh

/-- A finite average preserves the paired-benefit range. -/
theorem paired_average_bounds {m : ℕ} (hm : 0 < m) (x : ℕ → ℝ)
    (hx : ∀ i < m, x i ∈ Icc (-1 : ℝ) 1) :
    (∑ i ∈ Finset.range m, x i) / m ∈ Icc (-1 : ℝ) 1 := by
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hlo : ∑ i ∈ Finset.range m, (-1 : ℝ) ≤ ∑ i ∈ Finset.range m, x i :=
    Finset.sum_le_sum fun i hi => (hx i (Finset.mem_range.mp hi)).1
  have hhi : ∑ i ∈ Finset.range m, x i ≤ ∑ i ∈ Finset.range m, (1 : ℝ) :=
    Finset.sum_le_sum fun i hi => (hx i (Finset.mem_range.mp hi)).2
  simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_neg, mul_one] at hlo hhi
  constructor
  · exact (le_div_iff₀ hmR).mpr (by linarith)
  · exact (div_le_iff₀ hmR).mpr (by linarith)

/-- Clipping the width-two radius at two is justified by the support, not a
heuristic replacement of the concentration bound. -/
theorem paired_sampling_clipped_miss_le {μ : Measure Ω} [IsProbabilityMeasure μ]
    {W : ℕ → Ω → ℝ} (hW : ∀ i, Measurable (W i)) {m : ℕ} (hm : 0 < m)
    (hindep : iIndepFun W μ)
    (hbound : ∀ i, i < m → ∀ᵐ ω ∂μ, W i ω ∈ Icc (-1 : ℝ) 1)
    {delta : ℝ} (hmean : ∀ i, i < m → (∫ ω, W i ω ∂μ) = delta)
    {a : ℝ} (ha : 0 < a) (ha1 : a ≤ 1) :
    μ {ω | min 2 (2 * hoeffdingRadius m a) <
      |(∑ i ∈ Finset.range m, W i ω) / m - delta|} ≤ ENNReal.ofReal a := by
  have hB : Measurable (fun ω => (∑ i ∈ Finset.range m, W i ω) / (m : ℝ)) := by fun_prop
  have hc := paired_benefit_hoeffding_coverage hindep hm
    (fun i _ => (hW i).aemeasurable) hbound hmean ha ha1
  have hgood : MeasurableSet {ω | |(∑ i ∈ Finset.range m, W i ω) / m - delta| ≤
      2 * hoeffdingRadius m a} := measurableSet_le ((hB.sub_const delta).abs) measurable_const
  by_cases hr : 2 * hoeffdingRadius m a ≤ 2
  · simpa only [min_eq_right hr, compl_setOf, not_le] using real_coverage_miss_le hgood hc
  · have hi : Integrable (W 0) μ := Integrable.of_mem_Icc (-1) 1
      (hW 0).aemeasurable (hbound 0 hm)
    have hdelta : delta ∈ Icc (-1 : ℝ) 1 := by
      rw [← hmean 0 hm]
      constructor
      · have hh := integral_mono_ae (integrable_const (-1 : ℝ)) hi
          ((hbound 0 hm).mono fun _ h => h.1)
        simpa using hh
      · have hh := integral_mono_ae hi (integrable_const (1 : ℝ))
          ((hbound 0 hm).mono fun _ h => h.2)
        simpa using hh
    have hall : ∀ᵐ ω ∂μ, ∀ i, i < m → W i ω ∈ Icc (-1 : ℝ) 1 :=
      ae_all_iff.mpr fun i => ae_all_iff.mpr fun hi => hbound i hi
    have hz : μ {ω | min 2 (2 * hoeffdingRadius m a) <
        |(∑ i ∈ Finset.range m, W i ω) / m - delta|} = 0 := by
      apply measure_eq_zero_iff_ae_notMem.mpr
      filter_upwards [hall] with ω hw
      have hb := paired_average_bounds hm (fun i => W i ω) hw
      have habs : |(∑ i ∈ Finset.range m, W i ω) / m - delta| ≤ 2 :=
        abs_le.mpr ⟨by linarith [hb.1,hdelta.2], by linarith [hb.2,hdelta.1]⟩
      simpa only [Set.mem_setOf_eq, min_eq_left (le_of_not_ge hr), not_lt] using habs
    rw [hz]
    exact zero_le _

/-- The clipped fresh-sample bound averaged over the environment kernel. -/
theorem kernel_paired_clipped_sampling_miss_le {ν : Measure E} [IsProbabilityMeasure ν]
    (κ : Kernel E Ω) [IsMarkovKernel κ]
    (W : ℕ → E × Ω → ℝ) (hW : ∀ i, Measurable (W i))
    (delta : E → ℝ) (hd : Measurable delta) {m : ℕ} (hm : 0 < m)
    (hindep : ∀ e, iIndepFun (fun i ω => W i (e, ω)) (κ e))
    (hbound : ∀ e i, i < m → ∀ᵐ ω ∂κ e, W i (e, ω) ∈ Icc (-1 : ℝ) 1)
    (hmean : ∀ e i, i < m → (∫ ω, W i (e, ω) ∂κ e) = delta e)
    {a : ℝ} (ha : 0 < a) (ha1 : a ≤ 1) :
    (ν ⊗ₘ κ) {z | min 2 (2 * hoeffdingRadius m a) <
      |(∑ i ∈ Finset.range m, W i z) / m - delta z.1|} ≤ ENNReal.ofReal a := by
  have hB : Measurable (fun z => (∑ i ∈ Finset.range m, W i z) / (m : ℝ)) := by fun_prop
  apply kernel_event_bound κ (measurableSet_lt measurable_const ((hB.sub (hd.comp measurable_fst)).abs))
  intro e
  exact paired_sampling_clipped_miss_le (fun i => (hW i).comp measurable_prodMk_left)
    hm (hindep e) (hbound e) (hmean e) ha ha1

/-- Split conformal's insufficient-calibration branch is genuinely infinite. -/
noncomputable def splitExtendedRadius (n k : ℕ) (q : Ω → ℝ) (ω : Ω) : ℝ≥0∞ :=
  if k ≤ n then ENNReal.ofReal (q ω) else ∞

/-- The exchangeable-score theorem implies cell miss control for both the finite
quantile branch and the unavailable-rank branch. No cell coverage is postulated. -/
theorem split_extended_miss_le {μ : Measure Ω} [IsProbabilityMeasure μ] {n : ℕ}
    {R : Ω → Fin (n + 1) → ℝ} (hR : Measurable R)
    (hexch : ExchangeableScores μ R) (j : Fin (n + 1)) (k : ℕ)
    (estimate cell q : Ω → ℝ)
    (hquantile : k ≤ n → ∀ ω, CalibrationThreshold (R ω) j k (q ω))
    (hresidual : ∀ ω, R ω j = |estimate ω - cell ω|) {a : ℝ≥0∞}
    (hk : (((n + 1 - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal)) ≤ a) :
    μ {ω | ¬ ENNReal.ofReal |estimate ω - cell ω| ≤ splitExtendedRadius n k q ω} ≤ a := by
  by_cases hkn : k ≤ n
  · have hsub : {ω | ¬ ENNReal.ofReal |estimate ω - cell ω| ≤ splitExtendedRadius n k q ω} ⊆
        {ω | q ω < R ω j} := by
      intro ω hbad
      change q ω < R ω j
      rw [hresidual]
      by_contra hn
      have hh := ENNReal.ofReal_le_ofReal (le_of_not_gt hn)
      exact hbad (by simpa [splitExtendedRadius, hkn] using hh)
    exact (measure_mono hsub).trans
      ((exchangeable_calibration_threshold_miss_le hR hexch j k (hquantile hkn)).trans hk)
  · simp [splitExtendedRadius, hkn]

/-- Main conditional-episode capstone: exchangeable empirical residuals give
cell coverage, and fresh conditionally independent bounded evaluation draws give
population concentration. They may be dependent on the same experiment.
The radius is the width-two Hoeffding radius clipped at its support bound two. -/
theorem conditional_episode_compound_miss_le {ν : Measure E} [IsProbabilityMeasure ν]
    (κ : Kernel E Ω) [IsMarkovKernel κ]
    (W : ℕ → E × Ω → ℝ) (hW : ∀ i, Measurable (W i))
    (delta : E → ℝ) (hd : Measurable delta) {m : ℕ} (hm : 0 < m)
    (hindep : ∀ e, iIndepFun (fun i ω => W i (e, ω)) (κ e))
    (hbound : ∀ e i, i < m → ∀ᵐ ω ∂κ e, W i (e, ω) ∈ Icc (-1 : ℝ) 1)
    (hmean : ∀ e i, i < m → (∫ ω, W i (e, ω) ∂κ e) = delta e)
    {n : ℕ} {R : E × Ω → Fin (n + 1) → ℝ} (hR : Measurable R)
    (hexch : ExchangeableScores (ν ⊗ₘ κ) R) (j : Fin (n + 1)) (k : ℕ)
    (estimate q : E × Ω → ℝ)
    (hquantile : k ≤ n → ∀ z, CalibrationThreshold (R z) j k (q z))
    (hresidual : ∀ z, R z j = |estimate z - (∑ i ∈ Finset.range m, W i z) / m|)
    {a : ℝ≥0∞} (hk : (((n + 1 - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal)) ≤ a)
    {d : ℝ} (hd0 : 0 < d) (hd1 : d ≤ 1) :
    (ν ⊗ₘ κ) {z | ¬ ENNReal.ofReal |estimate z - delta z.1| ≤
      splitExtendedRadius n k q z + ENNReal.ofReal (min 2 (2 * hoeffdingRadius m d))} ≤
      a + ENNReal.ofReal d := by
  apply compound_miss_le estimate (fun z => (∑ i ∈ Finset.range m, W i z) / m)
    (fun z => delta z.1) (splitExtendedRadius n k q) (fun _ => min 2 (2 * hoeffdingRadius m d))
  · intro z
    exact le_min (by norm_num) (mul_nonneg (by norm_num) (Real.sqrt_nonneg _))
  · exact split_extended_miss_le hR hexch j k estimate _ q hquantile hresidual hk
  · exact kernel_paired_clipped_sampling_miss_le κ W hW delta hd hm hindep hbound hmean hd0 hd1

/-- The same capstone controls both strict directions together, not 2(alpha+delta). -/
theorem conditional_episode_wrong_direction_le {ν : Measure E} [IsProbabilityMeasure ν]
    (κ : Kernel E Ω) [IsMarkovKernel κ]
    (W : ℕ → E × Ω → ℝ) (hW : ∀ i, Measurable (W i))
    (delta : E → ℝ) (hd : Measurable delta) {m : ℕ} (hm : 0 < m)
    (hindep : ∀ e, iIndepFun (fun i ω => W i (e, ω)) (κ e))
    (hbound : ∀ e i, i < m → ∀ᵐ ω ∂κ e, W i (e, ω) ∈ Icc (-1 : ℝ) 1)
    (hmean : ∀ e i, i < m → (∫ ω, W i (e, ω) ∂κ e) = delta e)
    {n : ℕ} {R : E × Ω → Fin (n + 1) → ℝ} (hR : Measurable R)
    (hexch : ExchangeableScores (ν ⊗ₘ κ) R) (j : Fin (n + 1)) (k : ℕ)
    (estimate q : E × Ω → ℝ)
    (hquantile : k ≤ n → ∀ z, CalibrationThreshold (R z) j k (q z))
    (hresidual : ∀ z, R z j = |estimate z - (∑ i ∈ Finset.range m, W i z) / m|)
    {a : ℝ≥0∞} (hk : (((n + 1 - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal)) ≤ a)
    {d : ℝ} (hd0 : 0 < d) (hd1 : d ≤ 1) :
    (ν ⊗ₘ κ) {z |
      (extendedCertificate (estimate z) (splitExtendedRadius n k q z +
        ENNReal.ofReal (min 2 (2 * hoeffdingRadius m d))) = Decision.adapt ∧ delta z.1 ≤ 0) ∨
      (extendedCertificate (estimate z) (splitExtendedRadius n k q z +
        ENNReal.ofReal (min 2 (2 * hoeffdingRadius m d))) = Decision.freeze ∧ 0 ≤ delta z.1)} ≤
      a + ENNReal.ofReal d := by
  apply (measure_mono (randomRadiusEitherError_subset_compl estimate (fun z => delta z.1)
    (fun z => splitExtendedRadius n k q z + ENNReal.ofReal (min 2 (2 * hoeffdingRadius m d))))).trans
  exact conditional_episode_compound_miss_le κ W hW delta hd hm hindep hbound hmean
    hR hexch j k estimate q hquantile hresidual hk hd0 hd1

/-- Any miss bound gives the matching coverage bound; the argument uses only
subadditivity and total probability, so it also applies to outer measures. -/
theorem coverage_ge_of_miss_le {μ : Measure Ω} [IsProbabilityMeasure μ]
    {s : Set Ω} {a : ℝ≥0∞} (hmiss : μ sᶜ ≤ a) : 1 - a ≤ μ s := by
  apply tsub_le_iff_right.mpr
  calc
    1 = μ (s ∪ sᶜ) := by simp
    _ ≤ μ s + μ sᶜ := measure_union_le _ _
    _ ≤ μ s + a := add_le_add le_rfl hmiss

/-- A single frozen measurable residual map preserves episode exchangeability.
This is how the frozen-development protocol supplies the score-exchangeability
premise; it is not an assumption of uniform residual ranks. -/
theorem exchangeable_scores_of_exchangeable_episodes
    {H : Type*} [MeasurableSpace H] {μ : Measure Ω} {m : ℕ}
    (episodes : Ω → Fin m → H) (hep : Measurable episodes)
    (hexch : ∀ σ : Equiv.Perm (Fin m),
      (μ.map episodes).map (fun x => x ∘ σ) = μ.map episodes)
    (residual : H → ℝ) (hres : Measurable residual) :
    ExchangeableScores μ (fun ω i => residual (episodes ω i)) := by
  let F : (Fin m → H) → (Fin m → ℝ) := fun e i => residual (e i)
  have hF : Measurable F := measurable_pi_lambda _ fun i => hres.comp (measurable_pi_apply i)
  change ScoreLawExchangeable (μ.map (F ∘ episodes))
  rw [← Measure.map_map hF hep]
  intro σ
  have hr : Measurable (fun x : Fin m → H => x ∘ σ) :=
    measurable_pi_lambda _ fun i => measurable_pi_apply (σ i)
  rw [Measure.map_map (measurable_score_reindex σ) hF]
  have heq : (fun x : Fin m → ℝ => x ∘ σ) ∘ F = F ∘ (fun x => x ∘ σ) := rfl
  rw [heq, ← Measure.map_map hF hr, hexch σ]

/-- The displayed ceiling rank meets the ENNReal rank-error budget, including
ranks beyond the calibration sample size. -/
theorem ceiling_rank_miss_le (n : ℕ) {a : ℝ} (_ha : 0 ≤ a) :
    (((n + 1 - ⌈(1-a) * ((n+1 : ℕ) : ℝ)⌉₊ : ℕ) : ENNReal) /
      ((n+1 : ℕ) : ENNReal)) ≤ ENNReal.ofReal a := by
  let k := ⌈(1-a) * ((n+1 : ℕ) : ℝ)⌉₊
  change ((n+1-k : ℕ) : ENNReal) / ((n+1 : ℕ) : ENNReal) ≤ ENNReal.ofReal a
  by_cases hk : k ≤ n+1
  · have h := finite_uniform_rank_miss_le_alpha (n := n) (k := k) (alpha := a)
      (Nat.le_ceil ((1-a) * ((n+1 : ℕ) : ℝ)))
    have hpos : (0 : ℝ) < (n+1 : ℕ) := by positivity
    have hh := ENNReal.ofReal_le_ofReal h
    unfold finiteUniformRankMiss at hh
    rw [ENNReal.ofReal_div_of_pos hpos, ← Nat.cast_sub hk] at hh
    simpa only [ENNReal.ofReal_natCast] using hh
  · have hz : n+1-k=0 := Nat.sub_eq_zero_of_le (le_of_not_ge hk)
    simp [hz]

/-- The formal width-two radius is exactly the square-root expression printed
in the manuscript, rather than a different concentration constant. -/
theorem paired_radius_eq_printed {m : ℕ} (hm : 0 < m) {a : ℝ}
    (ha : 0 < a) (ha1 : a ≤ 1) :
    2 * hoeffdingRadius m a = Real.sqrt (2 * Real.log (2/a) / m) := by
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hl : 0 ≤ Real.log (2/a) := Real.log_nonneg ((le_div_iff₀ ha).mpr (by linarith))
  have harg1 : 0 ≤ Real.log (2/a) / (2*(m:ℝ)) := div_nonneg hl (by positivity)
  have harg2 : 0 ≤ 2 * Real.log (2/a) / (m:ℝ) := div_nonneg (mul_nonneg (by norm_num) hl) hmR.le
  have h1 := Real.sq_sqrt harg1
  have h2 := Real.sq_sqrt harg2
  have hid : (2:ℝ)^2 * (Real.log (2/a) / (2*m)) = 2 * Real.log (2/a) / m := by ring
  unfold hoeffdingRadius
  have hnon1 := Real.sqrt_nonneg (Real.log (2/a)/(2*m))
  have hnon2 := Real.sqrt_nonneg (2*Real.log (2/a)/m)
  nlinarith

/-- The actual conservative split-conformal radius formed from a finite episode
vector and a frozen residual map. The ceiling and unavailable-rank branch are
part of the definition, not separate assumptions. -/
noncomputable def canonicalEpisodeRadius {H : Type*} {n : ℕ}
    (episodes : Ω → Fin (n+1) → H) (residual : H → ℝ)
    (j : Fin (n+1)) (a : ℝ) (ω : Ω) : ℝ≥0∞ :=
  let k := ⌈(1-a) * ((n+1 : ℕ) : ℝ)⌉₊
  splitExtendedRadius n k (fun ω =>
    calibrationOrderStatistic (fun i => residual (episodes ω i)) j k) ω

/-- Full fixed-development conditional-episode assembly with the constructed
order statistic and printed clipped Hoeffding radius. Exchangeable episode
records pass through one frozen residual function; conditional fresh sampling
is integrated over the environment law. Neither cell coverage, rank uniformity,
a tail bound, nor a quantile-validity premise is assumed. -/
theorem conditional_episode_order_statistic_certificate
    {H : Type*} [MeasurableSpace H]
    {ν : Measure E} [IsProbabilityMeasure ν] (κ : Kernel E Ω) [IsMarkovKernel κ]
    (W : ℕ → E × Ω → ℝ) (hW : ∀ i, Measurable (W i))
    (delta : E → ℝ) (hd : Measurable delta) {m : ℕ} (hm : 0 < m)
    (hindep : ∀ e, iIndepFun (fun i ω => W i (e, ω)) (κ e))
    (hbound : ∀ e i, i < m → ∀ᵐ ω ∂κ e, W i (e, ω) ∈ Icc (-1 : ℝ) 1)
    (hmean : ∀ e i, i < m → (∫ ω, W i (e, ω) ∂κ e) = delta e)
    {n : ℕ} (episodes : E × Ω → Fin (n+1) → H) (hep : Measurable episodes)
    (hexch : ∀ σ : Equiv.Perm (Fin (n+1)),
      ((ν ⊗ₘ κ).map episodes).map (fun x => x ∘ σ) = (ν ⊗ₘ κ).map episodes)
    (residual : H → ℝ) (hres : Measurable residual) (j : Fin (n+1))
    (estimate : E × Ω → ℝ)
    (hresidual : ∀ z, residual (episodes z j) =
      |estimate z - (∑ i ∈ Finset.range m, W i z) / m|)
    {a d : ℝ} (ha : 0 ≤ a) (hd0 : 0 < d) (hd1 : d ≤ 1) :
    (1 - (ENNReal.ofReal a + ENNReal.ofReal d) ≤
      (ν ⊗ₘ κ) {z | ENNReal.ofReal |estimate z - delta z.1| ≤
        canonicalEpisodeRadius episodes residual j a z +
          ENNReal.ofReal (min 2 (Real.sqrt (2 * Real.log (2/d) / m)))}) ∧
    ((ν ⊗ₘ κ) {z |
      (extendedCertificate (estimate z) (canonicalEpisodeRadius episodes residual j a z +
        ENNReal.ofReal (min 2 (Real.sqrt (2 * Real.log (2/d) / m)))) = Decision.adapt ∧
          delta z.1 ≤ 0) ∨
      (extendedCertificate (estimate z) (canonicalEpisodeRadius episodes residual j a z +
        ENNReal.ofReal (min 2 (Real.sqrt (2 * Real.log (2/d) / m)))) = Decision.freeze ∧
          0 ≤ delta z.1)} ≤ ENNReal.ofReal a + ENNReal.ofReal d) := by
  let R := fun z i => residual (episodes z i)
  let k := ⌈(1-a) * ((n+1 : ℕ) : ℝ)⌉₊
  let q := fun z => calibrationOrderStatistic (R z) j k
  have hR : Measurable R := measurable_pi_lambda _ fun i =>
    hres.comp ((measurable_pi_apply i).comp hep)
  have hx : ExchangeableScores (ν ⊗ₘ κ) R :=
    exchangeable_scores_of_exchangeable_episodes episodes hep hexch residual hres
  have hq : k ≤ n → ∀ z, CalibrationThreshold (R z) j k (q z) :=
    fun hk z => calibrationOrderStatistic_threshold (R z) j hk
  have hk := ceiling_rank_miss_le n ha
  have hmiss := conditional_episode_compound_miss_le κ W hW delta hd hm hindep hbound hmean
    hR hx j k estimate q hq hresidual hk hd0 hd1
  have herr := conditional_episode_wrong_direction_le κ W hW delta hd hm hindep hbound hmean
    hR hx j k estimate q hq hresidual hk hd0 hd1
  rw [paired_radius_eq_printed hm hd0 hd1] at hmiss herr
  refine ⟨?_, herr⟩
  exact coverage_ge_of_miss_le hmiss

end KBound
