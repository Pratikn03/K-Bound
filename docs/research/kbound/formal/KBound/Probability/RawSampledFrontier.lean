import KBound.Probability.SampledDisagreement
import KBound.Probability.MeasureFrontier

/-! Raw iid inputs to a sampled disagreement guarantee by conditioning on the
finite membership pattern. No iid assertion after arbitrary selection is assumed. -/
namespace KBound
open MeasureTheory ProbabilityTheory Set Decision
open scoped ENNReal BigOperators ProbabilityTheory
variable {Ω X ι : Type*} [MeasurableSpace Ω] [MeasurableSpace X] [Fintype ι]

noncomputable def rawMembership (T : ι → Ω → X) (D : Set X) (ω : Ω) (i : ι) : Bool := by
  classical
  exact if T i ω ∈ D then true else false

noncomputable def selectedPattern (p : ι → Bool) : Finset ι := by
  classical
  exact Finset.univ.filter fun i => p i = true

noncomputable def patternCells (D : Set X) (p : ι → Bool) (i : ι) : Set X :=
  if p i = true then D else Dᶜ

omit [Fintype ι] in
theorem rawMembership_measurable (T : ι → Ω → X) (hT : ∀ i, Measurable (T i))
    {D : Set X} (hD : MeasurableSet D) : Measurable (rawMembership T D) := by
  classical
  apply measurable_pi_lambda
  intro i
  exact Measurable.ite ((hT i) hD) measurable_const measurable_const

omit [MeasurableSpace Ω] [MeasurableSpace X] [Fintype ι] in
theorem membership_fiber_eq_pattern (T : ι → Ω → X) (D : Set X) (p : ι → Bool) :
    rawMembership T D ⁻¹' {p} = ⋂ i, T i ⁻¹' patternCells D p i := by
  classical
  ext ω
  simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_iInter]
  constructor
  · intro h i
    have hi := congrFun h i
    change T i ω ∈ patternCells D p i
    cases hp : p i <;> simp [rawMembership, patternCells, hp] at hi ⊢ <;> simpa using hi
  · intro h
    funext i
    have hi := h i
    cases hp : p i <;> simp [rawMembership, patternCells, hp] at hi ⊢ <;> simp_all

/-- Total probability averages conditional bounds over a finite observable
partition, with zero-probability fibres contributing zero. -/
theorem finite_partition_event_bound {A : Type*} [Fintype A] [MeasurableSpace A]
    [DiscreteMeasurableSpace A] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (J : Ω → A) (hJ : Measurable J) (bad : Set Ω) {a : ℝ≥0∞}
    (hbad : ∀ j, μ (J ⁻¹' {j}) ≠ 0 → μ[bad | J ← j] ≤ a) : μ bad ≤ a := by
  classical
  calc
    μ bad = (∑ j, μ (J ⁻¹' {j}) • μ[|J ← j]) bad := by rw [sum_meas_smul_cond_fiber hJ μ]
    _ = ∑ j, μ (J ⁻¹' {j}) * μ[bad | J ← j] := by
      simp only [Measure.coe_finset_sum, Finset.sum_apply, Measure.smul_apply, smul_eq_mul]
    _ ≤ ∑ j, μ (J ⁻¹' {j}) * a := by
      apply Finset.sum_le_sum
      intro j _
      by_cases hj : μ (J ⁻¹' {j}) = 0
      · simp [hj]
      · exact mul_le_mul_right (hbad j hj) _
    _ = (∑ j, μ (J ⁻¹' {j})) * a := by rw [Finset.sum_mul]
    _ = a := by
      rw [sum_measure_preimage_singleton Finset.univ (fun j _ => hJ (measurableSet_singleton j))]
      simp

noncomputable def rawSampledMargin (T : ι → Ω → X) (D : Set X)
    (score : X → ℝ) (ω : Ω) : ℝ :=
  (∑ i ∈ selectedPattern (rawMembership T D ω), score (T i ω)) /
    (selectedPattern (rawMembership T D ω)).card - 1/2

noncomputable def rawSampledDecision (T : ι → Ω → X) (D : Set X)
    (score : X → ℝ) (beta alpha : ℝ) (ω : Ω) : Decision :=
  sampledMarginDecision (selectedPattern (rawMembership T D ω)).card
    (rawSampledMargin T D score ω) beta alpha

/-- The raw observations are independent with a common law. Conditioning on the
entire observed membership pattern derives independence and the common conditional
mean of the selected coordinates. Finite-pattern averaging then establishes the
marginal guarantee for the actual random disagreement count, including zero-count
abstention. No conditional iid premise about the selected sample is assumed. -/
theorem raw_iid_sampled_frontier_error_le
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    (T : ι → Ω → X) (hT : ∀ i, Measurable (T i)) (hindep : iIndepFun T μ)
    (score : X → ℝ) (hs : Measurable score)
    (hscore : ∀ x, score x ∈ Icc (0 : ℝ) 1)
    {ν : Measure X} (hlaw : ∀ i, μ.map (T i) = ν)
    {D : Set X} (hD : MeasurableSet D)
    {M gamma beta d alpha : ℝ}
    (hM : (∫ x, score x ∂ν[|D]) = M + 1/2)
    (hd : 0 < d) (hgamma : |gamma| ≤ beta) (ha : 0 < alpha) (ha1 : alpha ≤ 1) :
    μ {ω | (rawSampledDecision T D score beta alpha ω = adapt ∧
              2 * d * (M + gamma) ≤ 0) ∨
            (rawSampledDecision T D score beta alpha ω = freeze ∧
              0 ≤ 2 * d * (M + gamma))} ≤ ENNReal.ofReal alpha := by
  classical
  let J := rawMembership T D
  let bad := {ω | (rawSampledDecision T D score beta alpha ω = adapt ∧
              2 * d * (M + gamma) ≤ 0) ∨
            (rawSampledDecision T D score beta alpha ω = freeze ∧
              0 ≤ 2 * d * (M + gamma))}
  have hJ : Measurable J := rawMembership_measurable T hT hD
  apply finite_partition_event_bound J hJ bad
  intro p hp
  let S := selectedPattern p
  let C := ⋂ i, T i ⁻¹' patternCells D p i
  have hC : J ⁻¹' {p} = C := membership_fiber_eq_pattern T D p
  have hCm : MeasurableSet C := by rw [← hC]; exact hJ (measurableSet_singleton p)
  have hCp : μ C ≠ 0 := by rw [← hC]; exact hp
  letI : IsProbabilityMeasure μ[|C] := cond_isProbabilityMeasure hCp
  have hpat : ∀ᵐ ω ∂μ[|C], J ω = p := by
    filter_upwards [ae_cond_mem hCm] with ω hω
    rw [← hC] at hω
    exact hω
  have hcells : ∀ i, MeasurableSet (patternCells D p i) := by
    intro i
    unfold patternCells
    split_ifs
    · exact hD
    · exact hD.compl
  have hpositive : ∀ i, μ (T i ⁻¹' patternCells D p i) ≠ 0 := by
    intro i hi
    apply hCp
    apply le_antisymm _ (zero_le _)
    calc μ C ≤ μ (T i ⁻¹' patternCells D p i) := measure_mono (Set.iInter_subset _ i)
         _ = 0 := hi
  change μ[bad | J ← p] ≤ _
  rw [hC]
  by_cases hk : S.card = 0
  · have hzero : ∀ᵐ ω ∂μ[|C], ω ∉ bad := by
      filter_upwards [hpat] with ω hω
      change rawMembership T D ω = p at hω
      have hd0 : rawSampledDecision T D score beta alpha ω = abstain := by
        unfold rawSampledDecision
        rw [hω, show (selectedPattern p).card = 0 from hk]
        exact sampledMarginDecision_zero _ _ _
      simp [bad, hd0]
    rw [measure_eq_zero_iff_ae_notMem.mpr hzero]
    exact zero_le _
  · have hkpos : 0 < S.card := Nat.pos_of_ne_zero hk
    have hi : iIndepFun (fun i ω => score (T i ω)) μ[|C] :=
      independent_scores_given_membership_pattern T hT hindep score hs
        (patternCells D p) hcells hpositive
    have hm : ∀ i ∈ S, (∫ ω, score (T i ω) ∂μ[|C]) = M + 1/2 := by
      intro i hiS
      have hpi : p i = true := by simpa [S, selectedPattern] using hiS
      exact (selected_pattern_mean T hT hindep score hs hlaw
        (patternCells D p) hcells hpositive i (by simp [patternCells, hpi]) hD).trans hM
    have hc := finset_common_mean_hoeffding_coverage (s := S) hi hkpos
      (fun i _ => (hs.comp (hT i)).aemeasurable)
      (fun i _ => Filter.Eventually.of_forall (fun ω => hscore (T i ω))) hm ha ha1
    have hB : Measurable (fun ω => (∑ i ∈ S, score (T i ω)) / (S.card : ℝ)) := by fun_prop
    have hg : MeasurableSet {ω | |(∑ i ∈ S, score (T i ω)) / S.card - (M + 1/2)| ≤
        hoeffdingRadius S.card alpha} := measurableSet_le ((hB.sub_const _).abs) measurable_const
    have hh := real_coverage_miss_le hg hc
    have hsub := sampled_margin_error_subset (Ω := Ω) (M := M) hkpos hd hgamma
      (fun ω => (∑ i ∈ S, score (T i ω)) / S.card - 1/2) (alpha := alpha)
    have haesub : bad ≤ᵐ[μ[|C]]
        {ω | hoeffdingRadius S.card alpha <
          |(∑ i ∈ S, score (T i ω)) / S.card - 1/2 - M|} := by
      filter_upwards [hpat] with ω hω
      intro herr
      apply hsub
      change rawMembership T D ω = p at hω
      change (rawSampledDecision T D score beta alpha ω = adapt ∧ _) ∨
        (rawSampledDecision T D score beta alpha ω = freeze ∧ _) at herr
      simpa only [rawSampledDecision, rawSampledMargin, hω, S] using herr
    apply (measure_mono_ae haesub).trans
    have hevent : {ω | hoeffdingRadius S.card alpha <
          |(∑ i ∈ S, score (T i ω)) / S.card - 1/2 - M|} =
        {ω | |(∑ i ∈ S, score (T i ω)) / S.card - (M + 1/2)| ≤
          hoeffdingRadius S.card alpha}ᶜ := by
      ext ω
      simp only [Set.mem_setOf_eq, Set.mem_compl_iff, not_le]
      rw [show (∑ i ∈ S, score (T i ω)) / S.card - 1/2 - M =
        (∑ i ∈ S, score (T i ω)) / S.card - (M + 1/2) by ring]
    rw [hevent]
    exact hh

/-- If the disagreement region has null input probability, a finite raw sample
contains no disagreements almost surely and the implemented rule abstains. -/
theorem rawSampledDecision_abstain_of_null
    {μ : Measure Ω} (T : ι → Ω → X) (hT : ∀ i, Measurable (T i))
    {ν : Measure X} (hlaw : ∀ i, μ.map (T i) = ν)
    {D : Set X} (hD : MeasurableSet D) (hnull : ν D = 0)
    (score : X → ℝ) (beta alpha : ℝ) :
    ∀ᵐ ω ∂μ, rawSampledDecision T D score beta alpha ω = abstain := by
  classical
  have hz : ∀ i, μ (T i ⁻¹' D) = 0 := by
    intro i
    rw [← Measure.map_apply (hT i) hD, hlaw i, hnull]
  have hall : ∀ᵐ ω ∂μ, ∀ i, T i ω ∉ D :=
    ae_all_iff.mpr fun i => measure_eq_zero_iff_ae_notMem.mp (hz i)
  filter_upwards [hall] with ω hω
  have he : selectedPattern (rawMembership T D ω) = ∅ := by
    ext i
    simp [selectedPattern, rawMembership, hω i]
  simp [rawSampledDecision, he]

/-- Population-law specialization of the raw iid result. The target is the
actual zero-one risk difference of the measurable binary-support law, not an
arbitrarily named real parameter. The residual budget is explicit. Null
population disagreement is handled by almost-sure abstention, so no positive
mass hypothesis is needed by this final theorem. -/
theorem raw_iid_target_population_error_le
    {Y : Type*} [MeasurableSpace Y] [MeasurableSingletonClass Y] [MeasurableEq Y]
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    (T : ι → Ω → X) (hT : ∀ i, Measurable (T i)) (hindep : iIndepFun T μ)
    (ν : Measure X) [IsProbabilityMeasure ν] (hlaw : ∀ i, μ.map (T i) = ν)
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀] (η : CorrectnessField X)
    (score : X → ℝ) (hs : Measurable score) (hscore : ∀ x, score x ∈ Icc (0 : ℝ) 1)
    {beta alpha : ℝ} (ha : 0 < alpha) (ha1 : alpha ≤ 1)
    (hbudget : |disagreementMean ν {x | f₀ x ≠ fₐ x} η -
      (∫ x, score x ∂ν[|{x | f₀ x ≠ fₐ x}])| ≤ beta) :
    μ {ω |
      (rawSampledDecision T {x | f₀ x ≠ fₐ x} score beta alpha ω = adapt ∧
        populationBenefit f₀ fₐ (correctnessFieldTarget ν f₀ fₐ h₀ hₐ κ₀ η) ≤ 0) ∨
      (rawSampledDecision T {x | f₀ x ≠ fₐ x} score beta alpha ω = freeze ∧
        0 ≤ populationBenefit f₀ fₐ (correctnessFieldTarget ν f₀ fₐ h₀ hₐ κ₀ η))} ≤
      ENNReal.ofReal alpha := by
  let D := {x | f₀ x ≠ fₐ x}
  have hD : MeasurableSet D := (measurableSet_eq_fun h₀ hₐ).compl
  by_cases hd : 0 < ν.real D
  · let M := (∫ x, score x ∂ν[|D]) - 1/2
    let gamma := disagreementMean ν D η - (∫ x, score x ∂ν[|D])
    have he : populationBenefit f₀ fₐ (correctnessFieldTarget ν f₀ fₐ h₀ hₐ κ₀ η) =
        2 * ν.real D * (M + gamma) := by
      rw [correctnessFieldTarget_benefit ν f₀ fₐ h₀ hₐ κ₀ hd η]
      dsimp [M, gamma, D]
      ring
    rw [he]
    exact raw_iid_sampled_frontier_error_le T hT hindep score hs hscore hlaw hD
      (M := M) (gamma := gamma) (by dsimp [M]; ring) hd hbudget ha ha1
  · have hnull : ν D = 0 := (measureReal_eq_zero_iff).mp
      (le_antisymm (le_of_not_gt hd) (measureReal_nonneg))
    have habstain := rawSampledDecision_abstain_of_null T hT hlaw hD hnull score beta alpha
    have hz : μ {ω |
      (rawSampledDecision T D score beta alpha ω = adapt ∧
        populationBenefit f₀ fₐ (correctnessFieldTarget ν f₀ fₐ h₀ hₐ κ₀ η) ≤ 0) ∨
      (rawSampledDecision T D score beta alpha ω = freeze ∧
        0 ≤ populationBenefit f₀ fₐ (correctnessFieldTarget ν f₀ fₐ h₀ hₐ κ₀ η))} = 0 := by
      apply measure_eq_zero_iff_ae_notMem.mpr
      filter_upwards [habstain] with ω hω
      simp [hω]
    rw [hz]
    exact zero_le _

end KBound
