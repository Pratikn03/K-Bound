import KBound.Probability.CompoundCoverage
import KBound.Frontier
import KBound.Probability.FiniteMeanConcentration
import Mathlib.Probability.Independence.Basic

/-!
# Random-count sampled disagreement decisions

The count-kernel result handles the random positive sample count and zero-count
abstention. The separate rectangular-conditioning lemmas derive conditional
independence and coordinate laws from the original independent inputs, rather
than assuming that conditioning always preserves independence.
-/

namespace KBound
open MeasureTheory ProbabilityTheory Set Decision
open scoped ENNReal BigOperators ProbabilityTheory

variable {Ω X ι : Type*} [MeasurableSpace Ω] [MeasurableSpace X]

/-- Coordinatewise conditioning of independent inputs preserves independence
of their fixed measurable scores. This covers every positive-probability
membership pattern in D and its complement. -/
theorem independent_scores_given_membership_pattern [Finite ι]
    {μ : Measure Ω} (T : ι → Ω → X) (hT : ∀ i, Measurable (T i))
    (h_indep : iIndepFun T μ) (score : X → ℝ) (hs : Measurable score)
    (cells : ι → Set X) (hcells : ∀ i, MeasurableSet (cells i))
    (hpositive : ∀ i, μ (T i ⁻¹' cells i) ≠ 0) :
    iIndepFun (fun i ω => score (T i ω)) μ[|⋂ i, T i ⁻¹' cells i] := by
  exact iIndepFun.cond hT
    (h_indep.comp (fun _ x => (score x, x)) (fun _ => hs.prodMk measurable_id))
    hpositive hcells

/-- In a rectangular membership pattern, a score coordinate's conditional law
is its law conditional only on its own membership event. This is the key
identity used to obtain identical selected-coordinate distributions for iid inputs. -/
theorem score_law_given_pattern_eq_own_condition [Finite ι]
    {μ : Measure Ω} (T : ι → Ω → X) (hT : ∀ i, Measurable (T i))
    (h_indep : iIndepFun T μ) (score : X → ℝ) (hs : Measurable score)
    (cells : ι → Set X) (hcells : ∀ i, MeasurableSet (cells i))
    (hpositive : ∀ i, μ (T i ⁻¹' cells i) ≠ 0) (j : ι) :
    (μ[|⋂ i, T i ⁻¹' cells i]).map (fun ω => score (T j ω)) =
      (μ[|T j ⁻¹' cells j]).map (fun ω => score (T j ω)) := by
  classical
  apply Measure.ext
  intro A hA
  change ((μ[|⋂ i, T i ⁻¹' cells i]).map (score ∘ T j)) A =
    ((μ[|T j ⁻¹' cells j]).map (score ∘ T j)) A
  rw [Measure.map_apply (hs.comp (hT j)) hA, Measure.map_apply (hs.comp (hT j)) hA]
  have hind := h_indep.comp (fun _ x => (score x, x)) (fun _ => hs.prodMk measurable_id)
  have hh := cond_iInter (s := ({j} : Finset ι)) (f := fun i => (fun ω => score (T i ω)) ⁻¹' A)
    hT hind (fun i _ => ⟨A, hA, rfl⟩) (fun i _ => hpositive i) hcells
  simpa using hh

noncomputable def sampledMarginDecision (n : ℕ) (estimate beta alpha : ℝ) : Decision :=
  if n = 0 then abstain else certificate estimate (beta + hoeffdingRadius n alpha)

@[simp] theorem sampledMarginDecision_zero (estimate beta alpha : ℝ) :
    sampledMarginDecision 0 estimate beta alpha = abstain := by simp [sampledMarginDecision]

omit [MeasurableSpace Ω] in
/-- A sampled-margin error requires a mean-estimation error when the declared
structural residual bound holds. It uses the binary benefit identity explicitly. -/
theorem sampled_margin_error_subset {n : ℕ} (hn : 0 < n)
    {M gamma beta d alpha : ℝ} (hd : 0 < d) (hgamma : |gamma| ≤ beta)
    (estimate : Ω → ℝ) :
    {ω | (sampledMarginDecision n (estimate ω) beta alpha = adapt ∧
            2 * d * (M + gamma) ≤ 0) ∨
          (sampledMarginDecision n (estimate ω) beta alpha = freeze ∧
            0 ≤ 2 * d * (M + gamma))} ⊆
      {ω | hoeffdingRadius n alpha < |estimate ω - M|} := by
  intro ω herr
  by_contra hmiss
  change ¬ hoeffdingRadius n alpha < |estimate ω - M| at hmiss
  have hcov := abs_le.mp (le_of_not_gt hmiss)
  have hg := abs_le.mp hgamma
  rcases herr with ⟨ha,hbad⟩ | ⟨hf,hbad⟩
  · have hpos : 0 < estimate ω - (beta + hoeffdingRadius n alpha) := by
      unfold sampledMarginDecision at ha
      simp only [Nat.ne_of_gt hn, ↓reduceIte] at ha
      unfold certificate at ha
      split_ifs at ha
      simp_all
    have hsign : 0 < M + gamma := by linarith
    have := mul_pos (mul_pos (by norm_num : (0 : ℝ) < 2) hd) hsign
    linarith
  · have hneg : estimate ω + (beta + hoeffdingRadius n alpha) < 0 := by
      unfold sampledMarginDecision at hf
      simp only [Nat.ne_of_gt hn, ↓reduceIte] at hf
      unfold certificate at hf
      split_ifs at hf
      simp_all
    have hsign : M + gamma < 0 := by linarith
    have := mul_neg_of_pos_of_neg (mul_pos (by norm_num : (0 : ℝ) < 2) hd) hsign
    linarith

/-- An actual random-count kernel experiment: given positive count k, the first
k selected scores have common mean M+1/2 and are conditionally independent.
The zero-count branch abstains, and conditional concentration is averaged over
an arbitrary probability law of the count. No conditional tail bound is assumed. -/
theorem random_count_sampled_frontier_error_le
    {ν : Measure ℕ} [IsProbabilityMeasure ν] (κ : Kernel ℕ Ω) [IsMarkovKernel κ]
    (W : ℕ → ℕ × Ω → ℝ) (hW : ∀ i, Measurable (W i))
    {M gamma beta d alpha : ℝ} (hd : 0 < d) (hgamma : |gamma| ≤ beta)
    (ha : 0 < alpha) (ha1 : alpha ≤ 1)
    (hindep : ∀ k, 0 < k → iIndepFun (fun i ω => W i (k, ω)) (κ k))
    (hbound : ∀ k i, i < k → ∀ᵐ ω ∂κ k, W i (k, ω) ∈ Icc (0 : ℝ) 1)
    (hmean : ∀ k i, i < k → (∫ ω, W i (k, ω) ∂κ k) = M + 1/2)
    (hmeas : MeasurableSet {z : ℕ × Ω |
      (sampledMarginDecision z.1 ((∑ i ∈ Finset.range z.1, W i z) / z.1 - 1/2) beta alpha = adapt ∧
        2 * d * (M + gamma) ≤ 0) ∨
      (sampledMarginDecision z.1 ((∑ i ∈ Finset.range z.1, W i z) / z.1 - 1/2) beta alpha = freeze ∧
        0 ≤ 2 * d * (M + gamma))}) :
    (ν ⊗ₘ κ) {z : ℕ × Ω |
      (sampledMarginDecision z.1 ((∑ i ∈ Finset.range z.1, W i z) / z.1 - 1/2) beta alpha = adapt ∧
        2 * d * (M + gamma) ≤ 0) ∨
      (sampledMarginDecision z.1 ((∑ i ∈ Finset.range z.1, W i z) / z.1 - 1/2) beta alpha = freeze ∧
        0 ≤ 2 * d * (M + gamma))} ≤ ENNReal.ofReal alpha := by
  apply kernel_event_bound κ hmeas
  intro k
  by_cases hk : k = 0
  · subst k
    simp [sampledMarginDecision, show abstain ≠ adapt by decide, show abstain ≠ freeze by decide]
  · have hkpos : 0 < k := Nat.pos_of_ne_zero hk
    have hc := common_mean_hoeffding_coverage (hindep k hkpos) hkpos
      (fun i _ => ((hW i).comp measurable_prodMk_left).aemeasurable)
      (hbound k) (hmean k) ha ha1
    have hB : Measurable (fun ω => (∑ i ∈ Finset.range k, W i (k, ω)) / (k : ℝ)) := by fun_prop
    have hg : MeasurableSet {ω | |(∑ i ∈ Finset.range k, W i (k, ω)) / k - (M+1/2)| ≤
        hoeffdingRadius k alpha} := measurableSet_le ((hB.sub_const _).abs) measurable_const
    have hh := real_coverage_miss_le hg hc
    have hsub := sampled_margin_error_subset (Ω := Ω) (M := M) hkpos hd hgamma
      (fun ω => (∑ i ∈ Finset.range k, W i (k, ω)) / k - 1/2) (alpha := alpha)
    apply (measure_mono hsub).trans
    have hevent : {ω | hoeffdingRadius k alpha <
        |(∑ i ∈ Finset.range k, W i (k, ω)) / k - 1/2 - M|} =
        {ω | |(∑ i ∈ Finset.range k, W i (k, ω)) / k - (M+1/2)| ≤ hoeffdingRadius k alpha}ᶜ := by
      ext ω
      simp only [Set.mem_setOf_eq, Set.mem_compl_iff, not_le]
      rw [show (∑ i ∈ Finset.range k, W i (k, ω)) / k - 1/2 - M =
        (∑ i ∈ Finset.range k, W i (k, ω)) / k - (M+1/2) by ring]
    rw [hevent]
    exact hh

/-- Conditioning a law on a measurable coordinate event commutes with pushing
that coordinate forward. This identifies the selected score mean from the
original input marginal, rather than postulating a conditional mean. -/
theorem map_cond_preimage {μ : Measure Ω} (T : Ω → X) (hT : Measurable T)
    {D : Set X} (hD : MeasurableSet D) :
    (μ[|T ⁻¹' D]).map T = (μ.map T)[|D] := by
  apply Measure.ext
  intro A hA
  rw [Measure.map_apply hT hA, cond_apply' (hT hA), cond_apply' hA,
    Measure.map_apply hT hD, Measure.map_apply hT (hD.inter hA)]
  rfl

/-- A common iid input marginal determines every selected coordinate's mean
on each positive membership pattern. -/
theorem selected_pattern_mean [Finite ι] {μ : Measure Ω}
    (T : ι → Ω → X) (hT : ∀ i, Measurable (T i)) (h_indep : iIndepFun T μ)
    (score : X → ℝ) (hs : Measurable score) {ν : Measure X}
    (hlaw : ∀ i, μ.map (T i) = ν)
    (cells : ι → Set X) (hcells : ∀ i, MeasurableSet (cells i))
    (hpositive : ∀ i, μ (T i ⁻¹' cells i) ≠ 0)
    (j : ι) {D : Set X} (hj : cells j = D) (hD : MeasurableSet D) :
    (∫ ω, score (T j ω) ∂μ[|⋂ i, T i ⁻¹' cells i]) = ∫ x, score x ∂ν[|D] := by
  have heq := score_law_given_pattern_eq_own_condition T hT h_indep score hs cells hcells hpositive j
  have hi (mu0 : Measure Ω) : (∫ x, x ∂mu0.map (fun ω => score (T j ω))) =
      ∫ ω, score (T j ω) ∂mu0 :=
    integral_map (hs.comp (hT j)).aemeasurable measurable_id.aestronglyMeasurable
  rw [← hi, heq, hi]
  rw [hj]
  rw [← integral_map (hT j).aemeasurable hs.aestronglyMeasurable,
    map_cond_preimage (T j) (hT j) hD, hlaw j]

end KBound
