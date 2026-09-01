import KBound.Probability.RankCounting
import KBound.Probability.MeasureCertificate
import Mathlib.MeasureTheory.Integral.Lebesgue.Add
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise

/-!
# Split-conformal coverage from a permutation-invariant measurable score law

Unlike the earlier conditional uniform-index bridge, this module *derives* the
rank bound from exchangeability of the full score-vector distribution. Scores may
have atoms and ties. No conditional distribution, uniformly distributed rank,
independence assumption, or almost-sure absence of ties is a premise.

The proof first shows that relabeling a score vector relabels its strict ranks.
Permutation invariance then makes every coordinate's high-rank event equally
probable. Integrating the pointwise rank-counting bound bounds their common
probability by `(n + 1 - k) / (n + 1)`. The result is transported through any
measurable random score vector and composed with measure-level certificate
soundness. This is a one-shot marginal guarantee: it does not establish
conditional-on-adaptation safety or time-uniform repeated-deployment safety.
-/

namespace KBound

open MeasureTheory
open scoped BigOperators ENNReal

/-- The law of a finite real score vector is exchangeable when every permutation
of its coordinates preserves that law. This is a hypothesis on the *scores*, not
on a rank or a held-out-index random variable. -/
def ScoreLawExchangeable {m : ℕ} (μ : Measure (Fin m → ℝ)) : Prop :=
  ∀ σ : Equiv.Perm (Fin m), Measure.map (fun R => R ∘ σ) μ = μ

/-- Reindexing a finite score vector is measurable for the product Borel space. -/
lemma measurable_score_reindex {m : ℕ} (σ : Equiv.Perm (Fin m)) :
    Measurable (fun R : Fin m → ℝ => R ∘ σ) := by
  exact measurable_pi_lambda _ fun i => measurable_pi_apply (σ i)

/-- Strict ranks are measurable, including for score laws with atoms or ties. -/
lemma measurable_strictRank {m : ℕ} (j : Fin m) :
    Measurable (fun R : Fin m → ℝ => strictRank R j) := by
  classical
  simp only [strictRank, Finset.card_filter]
  exact Finset.measurable_fun_sum _ fun i _ =>
    Measurable.ite (measurableSet_lt (measurable_pi_apply i) (measurable_pi_apply j))
      measurable_const measurable_const

/-- The high strict-rank event is a Borel set. -/
lemma measurableSet_strictRank_miss {m : ℕ} (j : Fin m) (k : ℕ) :
    MeasurableSet {R : Fin m → ℝ | k ≤ strictRank R j} :=
  measurableSet_le measurable_const (measurable_strictRank j)

/-- Permuting coordinates preserves every strict-rank count after relabeling the
distinguished coordinate. The bijection proof does not break or randomize ties. -/
theorem strictRank_reindex {m : ℕ} (R : Fin m → ℝ)
    (σ : Equiv.Perm (Fin m)) (j : Fin m) :
    strictRank (R ∘ σ) j = strictRank R (σ j) := by
  classical
  unfold strictRank
  apply Finset.card_bij (fun i _ => σ i)
  · intro i hi
    simpa only [Finset.mem_filter, Finset.mem_univ, true_and, Function.comp_apply] using hi
  · intro i hi l hl hil
    exact σ.injective hil
  · intro i hi
    refine ⟨σ.symm i, ?_, σ.apply_symm_apply i⟩
    simpa only [Finset.mem_filter, Finset.mem_univ, true_and, Function.comp_apply,
      σ.apply_symm_apply] using hi

/-- Exchangeability of the score law, rather than a postulated uniform rank law,
implies identical high-rank event probabilities at any two coordinates. -/
theorem exchangeable_rank_miss_prob_eq {m : ℕ} {μ : Measure (Fin m → ℝ)}
    (hexch : ScoreLawExchangeable μ) (k : ℕ) (i j : Fin m) :
    μ {R | k ≤ strictRank R i} = μ {R | k ≤ strictRank R j} := by
  have h := congrArg (fun ν : Measure (Fin m → ℝ) =>
    ν {R | k ≤ strictRank R j}) (hexch (Equiv.swap j i))
  dsimp only at h
  rw [Measure.map_apply (measurable_score_reindex _) (measurableSet_strictRank_miss j k)] at h
  simpa only [Set.preimage_setOf_eq, strictRank_reindex, Equiv.swap_apply_left] using h

/-- Integrating the deterministic counting bound bounds the sum of miss
probabilities. This step holds for *any* score law, without exchangeability. -/
theorem sum_rank_miss_prob_le {m : ℕ} {μ : Measure (Fin m → ℝ)}
    [IsProbabilityMeasure μ] (k : ℕ) :
    (∑ j : Fin m, μ {R | k ≤ strictRank R j}) ≤ ((m - k : ℕ) : ENNReal) := by
  classical
  let f : Fin m → (Fin m → ℝ) → ENNReal :=
    fun j => {R | k ≤ strictRank R j}.indicator (fun _ => 1)
  have hf : ∀ j, Measurable (f j) := fun j =>
    measurable_const.indicator (measurableSet_strictRank_miss j k)
  have hcount (R : Fin m → ℝ) :
      (∑ j, f j R) = ((Finset.univ.filter fun j => k ≤ strictRank R j).card : ENNReal) := by
    calc
      (∑ j, f j R) = ∑ j : Fin m, if k ≤ strictRank R j then (1 : ENNReal) else 0 := by
        apply Finset.sum_congr rfl
        intro j _
        by_cases h : k ≤ strictRank R j <;> simp [f, h]
      _ = _ := Finset.sum_boole (R := ENNReal) (fun j : Fin m => k ≤ strictRank R j) Finset.univ
  calc
    (∑ j : Fin m, μ {R | k ≤ strictRank R j})
        = ∑ j : Fin m, ∫⁻ R, f j R ∂μ := by
          apply Finset.sum_congr rfl
          intro j _
          exact (lintegral_indicator_one (measurableSet_strictRank_miss j k)).symm
    _ = ∫⁻ R, ∑ j : Fin m, f j R ∂μ :=
      (lintegral_finset_sum Finset.univ (fun j _ => hf j)).symm
    _ ≤ ∫⁻ _R : Fin m → ℝ, ((m - k : ℕ) : ENNReal) ∂μ := by
      apply lintegral_mono
      intro R
      dsimp only
      rw [hcount]
      exact_mod_cast card_high_strictRank_le R k
    _ = ((m - k : ℕ) : ENNReal) := by simp

/-- **General measurable exchangeable-score conformal miss bound.** A specified
coordinate of an arbitrary permutation-invariant score law has high strict-rank
probability at most `(n + 1 - k)/(n + 1)`, with ties allowed. Uniformity of its
rank is a conclusion in neither the assumptions nor the proof: only the stated
upper bound is needed, and ties can make it conservative. -/
theorem exchangeable_scoreLaw_miss_le {n : ℕ} {μ : Measure (Fin (n + 1) → ℝ)}
    [IsProbabilityMeasure μ] (hexch : ScoreLawExchangeable μ)
    (j : Fin (n + 1)) (k : ℕ) :
    μ {R | k ≤ strictRank R j}
      ≤ (((n + 1 - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal)) := by
  have hsum := sum_rank_miss_prob_le (μ := μ) k
  have heq : (∑ i : Fin (n + 1), μ {R | k ≤ strictRank R i})
      = ((n + 1 : ℕ) : ENNReal) * μ {R | k ≤ strictRank R j} := by
    simp_rw [exchangeable_rank_miss_prob_eq hexch k _ j]
    simp [nsmul_eq_mul]
  rw [heq] at hsum
  apply (ENNReal.le_div_iff_mul_le (Or.inl (by simp)) (Or.inl (by simp))).2
  simpa only [mul_comm] using hsum

/-- Exchangeability of a measurable random score vector means permutation
invariance of its pushforward law on the product Borel score space. -/
def ExchangeableScores {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (μ : Measure Ω) (R : Ω → Fin m → ℝ) : Prop :=
  ScoreLawExchangeable (Measure.map R μ)

/-- Split-conformal miss control on an arbitrary underlying probability space.
Measurability of the score vector and exchangeability of its law are explicit;
the test coordinate is fixed before observing the scores. -/
theorem exchangeable_scores_rank_miss_le {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {n : ℕ}
    {R : Ω → Fin (n + 1) → ℝ} (hR : Measurable R)
    (hexch : ExchangeableScores μ R) (j : Fin (n + 1)) (k : ℕ) :
    μ {ω | k ≤ strictRank (R ω) j}
      ≤ (((n + 1 - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal)) := by
  letI : IsProbabilityMeasure (Measure.map R μ) := Measure.isProbabilityMeasure_map hR.aemeasurable
  have h := exchangeable_scoreLaw_miss_le hexch j k
  rw [Measure.map_apply hR (measurableSet_strictRank_miss j k)] at h
  exact h

/-- One-shot marginal coverage under general measurable score exchangeability.
The threshold may exceed the available score count, yielding an empty miss event
(the usual conservative infinite-radius convention). -/
theorem exchangeable_scores_rank_coverage_ge {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {n : ℕ}
    {R : Ω → Fin (n + 1) → ℝ} (hR : Measurable R)
    (hexch : ExchangeableScores μ R) (j : Fin (n + 1)) (k : ℕ) {alpha : ENNReal}
    (hk : (((n + 1 - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal)) ≤ alpha) :
    1 - alpha ≤ μ {ω | ¬ k ≤ strictRank (R ω) j} := by
  have hmiss := (exchangeable_scores_rank_miss_le hR hexch j k).trans hk
  have hmeas : MeasurableSet {ω | k ≤ strictRank (R ω) j} :=
    (measurableSet_strictRank_miss j k).preimage hR
  change 1 - alpha ≤ μ ({ω | k ≤ strictRank (R ω) j}ᶜ)
  rw [prob_compl_eq_one_sub hmeas]
  exact tsub_le_tsub_left hmiss 1

/-- **General-score certificate capstone, false-adapt direction.** The conformal
rank-cover event must imply coverage of the declared benefit target; the theorem
does not change an observed batch target into an unobserved population risk. -/
theorem exchangeable_scores_false_adapt_le {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {n : ℕ}
    {R : Ω → Fin (n + 1) → ℝ} (hR : Measurable R)
    (hexch : ExchangeableScores μ R) (j : Fin (n + 1)) (k : ℕ)
    {dhat delta : Ω → ℝ} {eps : ℝ} {alpha : ENNReal}
    (hdhat : Measurable dhat) (hdelta : Measurable delta)
    (hsub : {ω | ¬ k ≤ strictRank (R ω) j} ⊆ coverageEvent dhat delta eps)
    (hk : (((n + 1 - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal)) ≤ alpha) :
    μ (falseAdaptEvent dhat delta eps) ≤ alpha := by
  exact measure_false_adapt_le_alpha (measurableSet_coverageEvent hdhat hdelta eps)
    ((exchangeable_scores_rank_coverage_ge hR hexch j k hk).trans (measure_mono hsub))

/-- Mirror capstone for false-freeze under the same score-exchangeability and
target-coverage assumptions. -/
theorem exchangeable_scores_false_freeze_le {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {n : ℕ}
    {R : Ω → Fin (n + 1) → ℝ} (hR : Measurable R)
    (hexch : ExchangeableScores μ R) (j : Fin (n + 1)) (k : ℕ)
    {dhat delta : Ω → ℝ} {eps : ℝ} {alpha : ENNReal}
    (hdhat : Measurable dhat) (hdelta : Measurable delta)
    (hsub : {ω | ¬ k ≤ strictRank (R ω) j} ⊆ coverageEvent dhat delta eps)
    (hk : (((n + 1 - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal)) ≤ alpha) :
    μ (falseFreezeEvent dhat delta eps) ≤ alpha := by
  exact measure_false_freeze_le_alpha (measurableSet_coverageEvent hdhat hdelta eps)
    ((exchangeable_scores_rank_coverage_ge hR hexch j k hk).trans (measure_mono hsub))

/-- A finite calibration threshold contains at least `k` calibration scores at or
below it, excluding the held-out coordinate `j`. This is the deterministic
property of a conservative `k`-th calibration order statistic. It is not a
coverage-probability or rank-distribution assumption. -/
def CalibrationThreshold {m : ℕ} (R : Fin m → ℝ) (j : Fin m) (k : ℕ) (q : ℝ) : Prop :=
  k ≤ ((Finset.univ.erase j).filter fun i => R i ≤ q).card

/-- A finite threshold based on `n` calibration scores cannot realize a rank
above `n`. In particular, an insufficient-data quantile cannot be certified by
silently treating a finite placeholder radius as the required infinite radius. -/
theorem calibrationThreshold_rank_le {n : ℕ}
    {R : Fin (n + 1) → ℝ} {j : Fin (n + 1)} {k : ℕ} {q : ℝ}
    (hq : CalibrationThreshold R j k q) : k ≤ n := by
  have h := hq.trans (Finset.card_filter_le (Finset.univ.erase j) (fun i => R i ≤ q))
  simpa only [Finset.card_erase_of_mem (Finset.mem_univ j), Finset.card_univ,
    Fintype.card_fin, Nat.add_sub_cancel] using h

/-- If a held-out score exceeds a calibration threshold, the `k` calibration
witnesses below the threshold are all strictly below the held-out score. Thus
the threshold's miss event is contained in the high strict-rank event, including
when calibration scores tie at the threshold. -/
theorem calibrationThreshold_miss_implies_high_rank {m : ℕ}
    {R : Fin m → ℝ} {j : Fin m} {k : ℕ} {q : ℝ}
    (hq : CalibrationThreshold R j k q) (hmiss : q < R j) :
    k ≤ strictRank R j := by
  classical
  apply hq.trans
  apply Finset.card_le_card
  intro i hi
  exact Finset.mem_filter.mpr ⟨Finset.mem_univ i,
    ((Finset.mem_filter.mp hi).2).trans_lt hmiss⟩

/-- The observable finite calibration-quantile property, together with score
exchangeability, yields an actual threshold miss bound; no separate
rank-to-error coverage hypothesis is assumed. -/
theorem exchangeable_calibration_threshold_miss_le {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {n : ℕ}
    {R : Ω → Fin (n + 1) → ℝ} (hR : Measurable R)
    (hexch : ExchangeableScores μ R) (j : Fin (n + 1)) (k : ℕ)
    {q : Ω → ℝ} (hquantile : ∀ ω, CalibrationThreshold (R ω) j k (q ω)) :
    μ {ω | q ω < R ω j}
      ≤ (((n + 1 - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal)) := by
  exact (measure_mono (fun ω hω =>
    calibrationThreshold_miss_implies_high_rank (hquantile ω) hω)).trans
      (exchangeable_scores_rank_miss_le hR hexch j k)

/-- Split-conformal coverage at a random calibration threshold. The threshold is
measurable and may vary with the calibration data. The inequality is marginal
over the entire experiment, not conditional on the realized calibration data. -/
theorem exchangeable_calibration_threshold_coverage_ge {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {n : ℕ}
    {R : Ω → Fin (n + 1) → ℝ} (hR : Measurable R)
    (hexch : ExchangeableScores μ R) (j : Fin (n + 1)) (k : ℕ)
    {q : Ω → ℝ} (hq : Measurable q)
    (hquantile : ∀ ω, CalibrationThreshold (R ω) j k (q ω)) {alpha : ENNReal}
    (hk : (((n + 1 - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal)) ≤ alpha) :
    1 - alpha ≤ μ {ω | R ω j ≤ q ω} := by
  have hmiss := (exchangeable_calibration_threshold_miss_le hR hexch j k hquantile).trans hk
  have hmeas : MeasurableSet {ω | q ω < R ω j} :=
    measurableSet_lt hq ((measurable_pi_apply j).comp hR)
  have hcompl : {ω | R ω j ≤ q ω} = {ω | q ω < R ω j}ᶜ := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_compl_iff, not_lt]
  rw [hcompl, prob_compl_eq_one_sub hmeas]
  exact tsub_le_tsub_left hmiss 1

/-- Coverage of the declared benefit target when the held-out conformity score
is its absolute residual. Unlike the old fixed-radius bridge, this theorem
allows the conformal radius `q` to be a random calibration-data function. -/
theorem exchangeable_residual_coverage_ge {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {n : ℕ}
    {R : Ω → Fin (n + 1) → ℝ} (hR : Measurable R)
    (hexch : ExchangeableScores μ R) (j : Fin (n + 1)) (k : ℕ)
    {q dhat delta : Ω → ℝ} (hq : Measurable q)
    (hquantile : ∀ ω, CalibrationThreshold (R ω) j k (q ω))
    (hresidual : ∀ ω, R ω j = |dhat ω - delta ω|) {alpha : ENNReal}
    (hk : (((n + 1 - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal)) ≤ alpha) :
    1 - alpha ≤ μ {ω | |dhat ω - delta ω| ≤ q ω} := by
  simpa only [hresidual] using
    exchangeable_calibration_threshold_coverage_ge hR hexch j k hq hquantile hk

/-- **End-to-end split-conformal false-adapt control with a data-dependent
radius.** Genuine measurable score exchangeability and the finite calibration
quantile rule imply the unconditional erroneous-decision probability bound for
the declared residual target. No ideal rank distribution or coverage probability
is supplied as a premise. -/
theorem exchangeable_residual_false_adapt_le {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {n : ℕ}
    {R : Ω → Fin (n + 1) → ℝ} (hR : Measurable R)
    (hexch : ExchangeableScores μ R) (j : Fin (n + 1)) (k : ℕ)
    {q dhat delta : Ω → ℝ} (hq : Measurable q)
    (hquantile : ∀ ω, CalibrationThreshold (R ω) j k (q ω))
    (hresidual : ∀ ω, R ω j = |dhat ω - delta ω|) {alpha : ENNReal}
    (hk : (((n + 1 - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal)) ≤ alpha) :
    μ {ω | certificate (dhat ω) (q ω) = Decision.adapt ∧ delta ω ≤ 0} ≤ alpha := by
  have hmeas : MeasurableSet {ω | |dhat ω - delta ω| ≤ q ω} := by
    simpa only [Function.comp_apply, hresidual] using
      measurableSet_le ((measurable_pi_apply j).comp hR) hq
  apply measure_le_alpha_of_subset_compl hmeas _
    (exchangeable_residual_coverage_ge hR hexch j k hq hquantile hresidual hk)
  rintro ω ⟨hcert, hbad⟩ hcover
  exact cert_false_adapt_implies_coverage_failure hcert (not_lt.mpr hbad) hcover

/-- Mirror end-to-end split-conformal false-freeze control with a random
calibration radius and the same declared residual target. -/
theorem exchangeable_residual_false_freeze_le {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {n : ℕ}
    {R : Ω → Fin (n + 1) → ℝ} (hR : Measurable R)
    (hexch : ExchangeableScores μ R) (j : Fin (n + 1)) (k : ℕ)
    {q dhat delta : Ω → ℝ} (hq : Measurable q)
    (hquantile : ∀ ω, CalibrationThreshold (R ω) j k (q ω))
    (hresidual : ∀ ω, R ω j = |dhat ω - delta ω|) {alpha : ENNReal}
    (hk : (((n + 1 - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal)) ≤ alpha) :
    μ {ω | certificate (dhat ω) (q ω) = Decision.freeze ∧ 0 ≤ delta ω} ≤ alpha := by
  have hmeas : MeasurableSet {ω | |dhat ω - delta ω| ≤ q ω} := by
    simpa only [Function.comp_apply, hresidual] using
      measurableSet_le ((measurable_pi_apply j).comp hR) hq
  apply measure_le_alpha_of_subset_compl hmeas _
    (exchangeable_residual_coverage_ge hR hexch j k hq hquantile hresidual hk)
  rintro ω ⟨hcert, hbad⟩ hcover
  exact cert_false_freeze_implies_coverage_failure hcert (not_lt.mpr hbad) hcover

/-- Both erroneous certified directions are controlled by the *same* residual
coverage-failure event, so their union also has probability at most `alpha`, not
merely twice `alpha`. This remains an unconditional one-experiment bound. -/
theorem exchangeable_residual_either_error_le {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] {n : ℕ}
    {R : Ω → Fin (n + 1) → ℝ} (hR : Measurable R)
    (hexch : ExchangeableScores μ R) (j : Fin (n + 1)) (k : ℕ)
    {q dhat delta : Ω → ℝ} (hq : Measurable q)
    (hquantile : ∀ ω, CalibrationThreshold (R ω) j k (q ω))
    (hresidual : ∀ ω, R ω j = |dhat ω - delta ω|) {alpha : ENNReal}
    (hk : (((n + 1 - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal)) ≤ alpha) :
    μ {ω | (certificate (dhat ω) (q ω) = Decision.adapt ∧ delta ω ≤ 0) ∨
      (certificate (dhat ω) (q ω) = Decision.freeze ∧ 0 ≤ delta ω)} ≤ alpha := by
  have hmeas : MeasurableSet {ω | |dhat ω - delta ω| ≤ q ω} := by
    simpa only [Function.comp_apply, hresidual] using
      measurableSet_le ((measurable_pi_apply j).comp hR) hq
  apply measure_le_alpha_of_subset_compl hmeas _
    (exchangeable_residual_coverage_ge hR hexch j k hq hquantile hresidual hk)
  rintro ω (⟨hcert, hbad⟩ | ⟨hcert, hbad⟩) hcover
  · exact cert_false_adapt_implies_coverage_failure hcert (not_lt.mpr hbad) hcover
  · exact cert_false_freeze_implies_coverage_failure hcert (not_lt.mpr hbad) hcover

end KBound
