import Mathlib.MeasureTheory.Measure.Decomposition.Hahn
import Mathlib.MeasureTheory.Integral.Bochner.Set
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.InformationTheory.KullbackLeibler.ChainRule
import KBound.Probability.InformationBound
import Mathlib.Tactic.Linarith

/-!
# General measurable Le Cam testing identity

This module works with arbitrary probability laws on an arbitrary measurable
observation space, not the historical two-point input-law encoding. A randomized
binary test is a measurable acceptance probability in `[0,1]`. Total variation
is defined by the standard supremum of differences on measurable events.

The key measure-theoretic argument is Hahn decomposition: every randomized test
is dominated, in its difference of expectations, by the indicator of a Hahn
set. Thus the optimal sum of testing errors is attained by a deterministic
measurable test and equals `1 - TV`. This is an exact testing statement; no KL/TV
inequality is assumed as a premise.
-/

namespace KBound

open MeasureTheory Set
open scoped ENNReal

/-- A general measurable randomized test. Its value is the probability of
choosing the second hypothesis after observing the evidence. -/
structure MeasurableBinaryTest (Ω : Type*) [MeasurableSpace Ω] where
  toFun : Ω → ℝ
  measurable : Measurable toFun
  nonneg : ∀ ω, 0 ≤ toFun ω
  le_one : ∀ ω, toFun ω ≤ 1

namespace MeasurableBinaryTest

variable {Ω : Type*} [MeasurableSpace Ω]

/-- Bounded measurable tests are integrable under any finite measure. -/
lemma integrable (φ : MeasurableBinaryTest Ω) (μ : Measure Ω) [IsFiniteMeasure μ] :
    Integrable φ.toFun μ := by
  refine ⟨φ.measurable.aestronglyMeasurable, HasFiniteIntegral.of_mem_Icc 0 1 ?_⟩
  exact ae_of_all μ fun ω => ⟨φ.nonneg ω, φ.le_one ω⟩

/-- A deterministic decision region is a special case of a randomized test. -/
noncomputable def indicator (s : Set Ω) (hs : MeasurableSet s) : MeasurableBinaryTest Ω where
  toFun := s.indicator (fun _ => 1)
  measurable := measurable_const.indicator hs
  nonneg := by
    intro ω
    by_cases hω : ω ∈ s <;> simp [hω]
  le_one := by
    intro ω
    by_cases hω : ω ∈ s <;> simp [hω]

/-- The expected acceptance probability of an indicator test is its event mass. -/
lemma integral_indicator (s : Set Ω) (hs : MeasurableSet s) (μ : Measure Ω) :
    (∫ ω, (indicator s hs).toFun ω ∂μ) = μ.real s := by
  change (∫ ω, s.indicator (fun _ => (1 : ℝ)) ω ∂μ) = μ.real s
  rw [MeasureTheory.integral_indicator_const (1 : ℝ) hs]
  simp only [smul_eq_mul, mul_one]

/-- The zero test makes the type of admissible randomized tests nonempty. -/
instance : Nonempty (MeasurableBinaryTest Ω) :=
  ⟨⟨fun _ => 0, measurable_const, fun _ => le_rfl, fun _ => zero_le_one⟩⟩

end MeasurableBinaryTest

/-- Sum of the two testing errors: probability of choosing hypothesis one under
law zero, plus probability of choosing hypothesis zero under law one. -/
noncomputable def measurableTestingError {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) (φ : MeasurableBinaryTest Ω) : ℝ :=
  (∫ ω, φ.toFun ω ∂μ) + ∫ ω, 1 - φ.toFun ω ∂ν

/-- Total variation in the measurable-event supremum convention. For probability
laws this oriented supremum equals the supremum of absolute event differences. -/
noncomputable def measurableTotalVariation {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) : ℝ :=
  sSup (Set.range fun s : {s : Set Ω // MeasurableSet s} => ν.real s.val - μ.real s.val)

/-- A Hahn decision region dominates the difference of expectations of *all*
measurable `[0,1]`-valued tests. This is the measure-theoretic step that rules out
any advantage from randomized decisions. -/
theorem testing_difference_le_hahn {Ω : Type*} [MeasurableSpace Ω]
    {μ ν : Measure Ω} [IsFiniteMeasure μ] [IsFiniteMeasure ν]
    {s : Set Ω} (hs : IsHahnDecomposition μ ν s) (φ : MeasurableBinaryTest Ω) :
    (∫ ω, φ.toFun ω ∂ν) - (∫ ω, φ.toFun ω ∂μ) ≤ ν.real s - μ.real s := by
  have hcomp := integral_mono_measure hs.ge_on_compl
    (ae_of_all (μ.restrict sᶜ) φ.nonneg) (φ.integrable (μ.restrict sᶜ))
  have hpart := integral_mono_measure hs.le_on
    (ae_of_all (ν.restrict s) fun ω => sub_nonneg.mpr (φ.le_one ω))
    ((integrable_const (1 : ℝ)).sub (φ.integrable (ν.restrict s)))
  rw [integral_sub (integrable_const _) (φ.integrable (μ.restrict s)),
    integral_sub (integrable_const _) (φ.integrable (ν.restrict s))] at hpart
  simp only [integral_const, measureReal_restrict_apply_univ, smul_eq_mul, mul_one] at hpart
  have hμ := integral_add_compl hs.measurableSet (φ.integrable μ)
  have hν := integral_add_compl hs.measurableSet (φ.integrable ν)
  linarith

/-- The Hahn measurable region attains the measurable-event supremum. -/
theorem measurableTotalVariation_eq_hahn {Ω : Type*} [MeasurableSpace Ω]
    {μ ν : Measure Ω} [IsFiniteMeasure μ] [IsFiniteMeasure ν]
    {s : Set Ω} (hs : IsHahnDecomposition μ ν s) :
    measurableTotalVariation μ ν = ν.real s - μ.real s := by
  have hbound (t : {s : Set Ω // MeasurableSet s}) :
      ν.real t.val - μ.real t.val ≤ ν.real s - μ.real s := by
    simpa only [MeasurableBinaryTest.integral_indicator] using
      testing_difference_le_hahn hs (MeasurableBinaryTest.indicator t.val t.property)
  have hnonempty : (Set.range fun t : {s : Set Ω // MeasurableSet s} =>
      ν.real t.val - μ.real t.val).Nonempty :=
    ⟨0, ⟨⟨∅, MeasurableSet.empty⟩, by simp⟩⟩
  have hbdd : BddAbove (Set.range fun t : {s : Set Ω // MeasurableSet s} =>
      ν.real t.val - μ.real t.val) :=
    ⟨ν.real s - μ.real s, by rintro x ⟨t, rfl⟩; exact hbound t⟩
  apply le_antisymm
  · exact csSup_le hnonempty (by rintro x ⟨t, rfl⟩; exact hbound t)
  · exact le_csSup hbdd ⟨⟨s, hs.measurableSet⟩, rfl⟩

/-- General bounded-test characterization, upper-bound direction. -/
theorem testing_difference_le_totalVariation {Ω : Type*} [MeasurableSpace Ω]
    {μ ν : Measure Ω} [IsFiniteMeasure μ] [IsFiniteMeasure ν]
    (φ : MeasurableBinaryTest Ω) :
    (∫ ω, φ.toFun ω ∂ν) - (∫ ω, φ.toFun ω ∂μ) ≤ measurableTotalVariation μ ν := by
  obtain ⟨s, hs⟩ := exists_isHahnDecomposition μ ν
  rw [measurableTotalVariation_eq_hahn hs]
  exact testing_difference_le_hahn hs φ

/-- Every measurable event difference is bounded by total variation. -/
theorem event_difference_le_totalVariation {Ω : Type*} [MeasurableSpace Ω]
    {μ ν : Measure Ω} [IsFiniteMeasure μ] [IsFiniteMeasure ν]
    {s : Set Ω} (hs : MeasurableSet s) :
    ν.real s - μ.real s ≤ measurableTotalVariation μ ν := by
  simpa only [MeasurableBinaryTest.integral_indicator] using
    testing_difference_le_totalVariation (μ := μ) (ν := ν) (MeasurableBinaryTest.indicator s hs)

/-- Total variation is nonnegative, witnessed by the empty event. -/
theorem measurableTotalVariation_nonneg {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsFiniteMeasure μ] [IsFiniteMeasure ν] :
    0 ≤ measurableTotalVariation μ ν := by
  simpa only [measureReal_empty, sub_self] using
    event_difference_le_totalVariation (μ := μ) (ν := ν) MeasurableSet.empty

/-- Total variation between probability laws is at most one. -/
theorem measurableTotalVariation_le_one {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] :
    measurableTotalVariation μ ν ≤ 1 := by
  obtain ⟨s, hs⟩ := exists_isHahnDecomposition μ ν
  rw [measurableTotalVariation_eq_hahn hs]
  exact (sub_le_self _ measureReal_nonneg).trans measureReal_le_one

/-- For probability laws, both signs of an event difference are controlled by
the oriented supremum: the opposite sign is the difference of the complement. -/
theorem event_abs_difference_le_totalVariation {Ω : Type*} [MeasurableSpace Ω]
    {μ ν : Measure Ω} [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    {s : Set Ω} (hs : MeasurableSet s) :
    |ν.real s - μ.real s| ≤ measurableTotalVariation μ ν := by
  have hp := event_difference_le_totalVariation (μ := μ) (ν := ν) hs
  have hn := event_difference_le_totalVariation (μ := μ) (ν := ν) hs.compl
  have hμ := probReal_add_probReal_compl (μ := μ) hs
  have hν := probReal_add_probReal_compl (μ := ν) hs
  exact abs_le.mpr ⟨by linarith, hp⟩

/-- Agreement with the usual total-variation convention
`sup { |ν(A) - μ(A)| : A measurable }` for arbitrary probability laws. -/
theorem measurableTotalVariation_eq_abs_sup {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] :
    measurableTotalVariation μ ν =
      sSup (Set.range fun s : {s : Set Ω // MeasurableSet s} => |ν.real s.val - μ.real s.val|) := by
  have hnonempty : (Set.range fun s : {s : Set Ω // MeasurableSet s} =>
      |ν.real s.val - μ.real s.val|).Nonempty :=
    ⟨0, ⟨⟨∅, MeasurableSet.empty⟩, by simp⟩⟩
  have hbdd : BddAbove (Set.range fun s : {s : Set Ω // MeasurableSet s} =>
      |ν.real s.val - μ.real s.val|) :=
    ⟨measurableTotalVariation μ ν, by
      rintro x ⟨s, rfl⟩
      exact event_abs_difference_le_totalVariation s.property⟩
  apply le_antisymm
  · obtain ⟨s, hs⟩ := exists_isHahnDecomposition μ ν
    have hnonneg : 0 ≤ ν.real s - μ.real s := by
      rw [← measurableTotalVariation_eq_hahn hs]
      exact measurableTotalVariation_nonneg μ ν
    rw [measurableTotalVariation_eq_hahn hs]
    simpa only [abs_of_nonneg hnonneg] using
      le_csSup hbdd (show |ν.real s - μ.real s| ∈
        Set.range (fun t : {s : Set Ω // MeasurableSet s} => |ν.real t.val - μ.real t.val|)
        from ⟨⟨s, hs.measurableSet⟩, rfl⟩)
  · exact csSup_le hnonempty (by
      rintro x ⟨s, rfl⟩
      exact event_abs_difference_le_totalVariation s.property)

/-- Symmetry of total variation for general probability laws. -/
theorem measurableTotalVariation_symm {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] :
    measurableTotalVariation μ ν = measurableTotalVariation ν μ := by
  rw [measurableTotalVariation_eq_abs_sup, measurableTotalVariation_eq_abs_sup]
  simp only [abs_sub_comm]

/-- Deterministic measurable processing cannot increase total variation. This
applies, in particular, to any measurable vector of monitoring features. -/
theorem measurableTotalVariation_map_le {Ω Ξ : Type*} [MeasurableSpace Ω] [MeasurableSpace Ξ]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    {f : Ω → Ξ} (hf : Measurable f) :
    measurableTotalVariation (Measure.map f μ) (Measure.map f ν) ≤ measurableTotalVariation μ ν := by
  letI : IsProbabilityMeasure (Measure.map f μ) := Measure.isProbabilityMeasure_map hf.aemeasurable
  letI : IsProbabilityMeasure (Measure.map f ν) := Measure.isProbabilityMeasure_map hf.aemeasurable
  obtain ⟨s, hs⟩ := exists_isHahnDecomposition (Measure.map f μ) (Measure.map f ν)
  rw [measurableTotalVariation_eq_hahn hs]
  simpa only [Measure.real, Measure.map_apply hf hs.measurableSet] using
    event_difference_le_totalVariation (μ := μ) (ν := ν) (hs.measurableSet.preimage hf)

/-- Writing the sum of testing errors as one minus the expectation gap. -/
theorem measurableTestingError_eq_one_sub_difference {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (φ : MeasurableBinaryTest Ω) :
    measurableTestingError μ ν φ = 1 - ((∫ ω, φ.toFun ω ∂ν) - ∫ ω, φ.toFun ω ∂μ) := by
  unfold measurableTestingError
  rw [integral_sub (integrable_const _) (φ.integrable ν)]
  simp only [integral_const, probReal_univ, smul_eq_mul, mul_one]
  ring

/-- **Le Cam lower bound for arbitrary measurable randomized tests.** -/
theorem general_lecam_testing_error_ge {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (φ : MeasurableBinaryTest Ω) :
    1 - measurableTotalVariation μ ν ≤ measurableTestingError μ ν φ := by
  rw [measurableTestingError_eq_one_sub_difference]
  exact sub_le_sub_left (testing_difference_le_totalVariation φ) 1

/-- The lower bound is attained by a deterministic measurable Hahn-region test,
even though the optimization class allows arbitrary randomized tests. -/
theorem exists_lecam_optimal_test {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] :
    ∃ φ : MeasurableBinaryTest Ω,
      measurableTestingError μ ν φ = 1 - measurableTotalVariation μ ν := by
  obtain ⟨s, hs⟩ := exists_isHahnDecomposition μ ν
  refine ⟨MeasurableBinaryTest.indicator s hs.measurableSet, ?_⟩
  rw [measurableTestingError_eq_one_sub_difference, measurableTotalVariation_eq_hahn hs]
  simp only [MeasurableBinaryTest.integral_indicator]

/-- **Exact general measurable Le Cam testing identity.** The infimum over all
measurable randomized tests equals `1 - TV`; neither input law is restricted to
a finite or two-point observation space. -/
theorem general_lecam_inf_testing_error {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] :
    (⨅ φ : MeasurableBinaryTest Ω, measurableTestingError μ ν φ)
      = 1 - measurableTotalVariation μ ν := by
  obtain ⟨φ, hφ⟩ := exists_lecam_optimal_test μ ν
  have hbdd : BddBelow (Set.range (measurableTestingError μ ν)) :=
    ⟨1 - measurableTotalVariation μ ν, by
      rintro x ⟨ψ, rfl⟩
      exact general_lecam_testing_error_ge μ ν ψ⟩
  apply le_antisymm
  · exact (ciInf_le hbdd φ).trans hφ.le
  · exact le_ciInf fun ψ => general_lecam_testing_error_ge μ ν ψ

/-- Any randomized test's worst-case error is at least half the affinity. The
factor `1/2` arises from converting the sum of errors to their maximum. -/
theorem general_lecam_worst_case_error_ge {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (φ : MeasurableBinaryTest Ω) :
    (1 - measurableTotalVariation μ ν) / 2 ≤
      max (∫ ω, φ.toFun ω ∂μ) (∫ ω, 1 - φ.toFun ω ∂ν) := by
  have h := general_lecam_testing_error_ge μ ν φ
  have hμ := le_max_left (∫ ω, φ.toFun ω ∂μ) (∫ ω, 1 - φ.toFun ω ∂ν)
  have hν := le_max_right (∫ ω, φ.toFun ω ∂μ) (∫ ω, 1 - φ.toFun ω ∂ν)
  unfold measurableTestingError at h
  linarith

/-- If either wrong decision costs at least `Lambda ≥ 0`, its two-world
worst-case regret is at least `Lambda/2 * (1-TV)`. -/
theorem general_lecam_regret_floor {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (φ : MeasurableBinaryTest Ω) {Lambda : ℝ} (hLambda : 0 ≤ Lambda) :
    (Lambda / 2) * (1 - measurableTotalVariation μ ν) ≤
      Lambda * max (∫ ω, φ.toFun ω ∂μ) (∫ ω, 1 - φ.toFun ω ∂ν) := by
  have h := mul_le_mul_of_nonneg_left (general_lecam_worst_case_error_ge μ ν φ) hLambda
  nlinarith

/-- An actual independent `n`-observation product experiment on the product
measurable space. This is not a pair of Bernoulli parameters. -/
noncomputable def iidObservationLaw {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (n : ℕ) : Measure (Fin n → Ω) :=
  Measure.pi (fun _ => μ)

instance iidObservationLaw_isProbabilityMeasure {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (n : ℕ) :
    IsProbabilityMeasure (iidObservationLaw μ n) := by
  unfold iidObservationLaw
  infer_instance

/-- The exact general testing identity applies to the actual product laws of
an independent sample of any finite size, including `n=0`. -/
theorem general_lecam_iid_testing_identity {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] (n : ℕ) :
    (⨅ φ : MeasurableBinaryTest (Fin n → Ω),
      measurableTestingError (iidObservationLaw μ n) (iidObservationLaw ν n) φ)
      = 1 - measurableTotalVariation (iidObservationLaw μ n) (iidObservationLaw ν n) :=
  general_lecam_inf_testing_error (iidObservationLaw μ n) (iidObservationLaw ν n)

/-- Restricting both laws to a measurable event leaves their likelihood ratio
unchanged almost everywhere inside that event. This is derived from the
Radon--Nikodym chain rule and the two restriction derivative identities. -/
theorem llr_restrict_event_eq {Ω : Type*} [MeasurableSpace Ω]
    {μ ν : Measure Ω} [IsFiniteMeasure μ] [IsFiniteMeasure ν]
    (hμν : μ ≪ ν) {s : Set Ω} (hs : MeasurableSet s) :
    llr (μ.restrict s) (ν.restrict s) =ᵐ[μ.restrict s] llr μ ν := by
  have hchain := Measure.rnDeriv_mul_rnDeriv (κ := ν) (hμν.restrict s)
  have hleft := Measure.rnDeriv_restrict μ ν hs
  have hright := Measure.rnDeriv_restrict_self ν hs
  have hder : (μ.restrict s).rnDeriv (ν.restrict s) =ᵐ[ν.restrict s] μ.rnDeriv ν := by
    filter_upwards [ae_restrict_of_ae hchain, ae_restrict_of_ae hleft,
      ae_restrict_of_ae hright, ae_restrict_mem hs] with ω hchainω hleftω hrightω hω
    simpa only [Pi.mul_apply, hleftω, hrightω, Set.indicator_of_mem hω, Pi.one_apply,
      mul_one] using hchainω
  filter_upwards [(hμν.restrict s).ae_le hder] with ω hω
  simp only [llr, hω]

/-- The eventwise log-sum inequality for the actual log-likelihood ratio of
arbitrary finite measures. It follows from Gibbs/Jensen for the restricted laws,
not from an assumed binary data-processing inequality. -/
theorem event_mul_log_ratio_le_integral_llr {Ω : Type*} [MeasurableSpace Ω]
    {μ ν : Measure Ω} [IsFiniteMeasure μ] [IsFiniteMeasure ν]
    (hμν : μ ≪ ν) (h_int : Integrable (llr μ ν) μ)
    {s : Set Ω} (hs : MeasurableSet s) :
    μ.real s * Real.log (μ.real s / ν.real s) ≤ ∫ ω in s, llr μ ν ω ∂μ := by
  have hllr := llr_restrict_event_eq hμν hs
  have h_int_s : Integrable (llr (μ.restrict s) (ν.restrict s)) (μ.restrict s) :=
    (integrable_congr hllr).2 h_int.integrableOn
  have h := InformationTheory.mul_log_le_toReal_klDiv (hμν.restrict s) h_int_s
  rw [InformationTheory.toReal_klDiv (hμν.restrict s) h_int_s,
    integral_congr_ae hllr] at h
  simp only [measureReal_restrict_apply_univ] at h
  linarith

/-- **Binary-partition KL lower bound for arbitrary probability measures.**
The reduction is proved by summing the two eventwise log-sum inequalities. -/
theorem binary_partition_kl_le {Ω : Type*} [MeasurableSpace Ω]
    {μ ν : Measure Ω} [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hKL : InformationTheory.klDiv μ ν ≠ ∞)
    {s : Set Ω} (hs : MeasurableSet s) :
    μ.real s * Real.log (μ.real s / ν.real s) +
      (1 - μ.real s) * Real.log ((1 - μ.real s) / (1 - ν.real s))
        ≤ (InformationTheory.klDiv μ ν).toReal := by
  obtain ⟨hμν, h_int⟩ := InformationTheory.klDiv_ne_top_iff.mp hKL
  have hsbound := event_mul_log_ratio_le_integral_llr hμν h_int hs
  have hcbound := event_mul_log_ratio_le_integral_llr hμν h_int hs.compl
  have hsum := add_le_add hsbound hcbound
  rw [integral_add_compl hs h_int] at hsum
  rw [InformationTheory.toReal_klDiv hμν h_int]
  simpa only [measureReal_compl hs, probReal_univ, add_sub_cancel_right] using hsum

/-- The finite-KL support conditions required at the binary partition's boundary
are consequences of absolute continuity, not extra empirical assumptions. -/
theorem binary_partition_support {Ω : Type*} [MeasurableSpace Ω]
    {μ ν : Measure Ω} [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hKL : InformationTheory.klDiv μ ν ≠ ∞)
    {s : Set Ω} (hs : MeasurableSet s) :
    (ν.real s = 0 → μ.real s = 0) ∧ (ν.real s = 1 → μ.real s = 1) := by
  have hμν := (InformationTheory.klDiv_ne_top_iff.mp hKL).1
  constructor
  · intro hν
    apply (measureReal_eq_zero_iff).2
    exact hμν ((measureReal_eq_zero_iff).1 hν)
  · intro hν
    have hνc : ν.real sᶜ = 0 := by rw [measureReal_compl hs, probReal_univ, hν]; ring
    have hμc : μ.real sᶜ = 0 := (measureReal_eq_zero_iff).2
      (hμν ((measureReal_eq_zero_iff).1 hνc))
    have hμsum := probReal_add_probReal_compl (μ := μ) hs
    linarith

/-- KL is invariant under a bimeasurable relabeling of the observation space.
The proof uses the actual Radon--Nikodym transport identity and the nonnegative
integral form of KL, so it includes infinite-divergence cases. -/
theorem klDiv_map_measurableEquiv {Ω Ξ : Type*} [MeasurableSpace Ω] [MeasurableSpace Ξ]
    (μ ν : Measure Ω) [IsFiniteMeasure μ] [IsFiniteMeasure ν] (e : Ω ≃ᵐ Ξ) :
    InformationTheory.klDiv (Measure.map e μ) (Measure.map e ν) = InformationTheory.klDiv μ ν := by
  have hac : Measure.map e μ ≪ Measure.map e ν ↔ μ ≪ ν := by
    constructor
    · intro h
      have hback := e.symm.measurableEmbedding.absolutelyContinuous_map h
      simpa only [MeasurableEquiv.map_symm_map] using hback
    · exact e.measurableEmbedding.absolutelyContinuous_map
  rw [InformationTheory.klDiv_eq_lintegral_klFun,
    InformationTheory.klDiv_eq_lintegral_klFun, hac]
  by_cases hμν : μ ≪ ν
  · simp only [hμν, if_pos]
    rw [e.measurableEmbedding.lintegral_map]
    apply lintegral_congr_ae
    filter_upwards [e.measurableEmbedding.rnDeriv_map μ ν] with ω hω
    simp only [hω]
  · simp only [hμν, if_false]

/-- Appending the same independent experiment to both worlds leaves KL
unchanged. This follows from the full kernel composition-product chain rule. -/
theorem klDiv_prod_same_right {Ω Ξ : Type*} [MeasurableSpace Ω] [MeasurableSpace Ξ]
    (μ ν : Measure Ω) (ρ : Measure Ξ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    [IsProbabilityMeasure ρ] :
    InformationTheory.klDiv (μ.prod ρ) (ν.prod ρ) = InformationTheory.klDiv μ ν := by
  simpa only [Measure.compProd_const] using
    InformationTheory.klDiv_compProd_left μ ν (ProbabilityTheory.Kernel.const Ω ρ)

/-- Mirror invariance for an identical independent first component. -/
theorem klDiv_prod_same_left {Ω Ξ : Type*} [MeasurableSpace Ω] [MeasurableSpace Ξ]
    (ρ : Measure Ω) (μ ν : Measure Ξ) [IsProbabilityMeasure ρ] [IsProbabilityMeasure μ]
    [IsProbabilityMeasure ν] :
    InformationTheory.klDiv (ρ.prod μ) (ρ.prod ν) = InformationTheory.klDiv μ ν := by
  have hswap := klDiv_map_measurableEquiv (ρ.prod μ) (ρ.prod ν) MeasurableEquiv.prodComm
  change InformationTheory.klDiv (Measure.map Prod.swap (ρ.prod μ))
    (Measure.map Prod.swap (ρ.prod ν)) = _ at hswap
  rw [Measure.prod_swap, Measure.prod_swap] at hswap
  rw [← hswap]
  exact klDiv_prod_same_right μ ν ρ

/-- **KL additivity for two independent measurable experiments.** The identity
holds in extended nonnegative reals, including failures of absolute continuity
or integrability, without pretending an infinite KL has real value zero. -/
theorem klDiv_prod_add {Ω Ξ : Type*} [MeasurableSpace Ω] [MeasurableSpace Ξ]
    (μ ν : Measure Ω) (ρ σ : Measure Ξ) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    [IsProbabilityMeasure ρ] [IsProbabilityMeasure σ] :
    InformationTheory.klDiv (μ.prod ρ) (ν.prod σ)
      = InformationTheory.klDiv μ ν + InformationTheory.klDiv ρ σ := by
  have h := InformationTheory.klDiv_compProd_eq_add μ ν
    (ProbabilityTheory.Kernel.const Ω ρ) (ProbabilityTheory.Kernel.const Ω σ)
  simpa only [Measure.compProd_const, klDiv_prod_same_left] using h

/-- **Exact finite-sample KL factorization on actual independent product laws.**
The proof splits one coordinate by a measurable product equivalence and uses
the extended-real two-experiment chain rule. -/
theorem klDiv_iidObservationLaw {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] (n : ℕ) :
    InformationTheory.klDiv (iidObservationLaw μ n) (iidObservationLaw ν n)
      = n * InformationTheory.klDiv μ ν := by
  induction n with
  | zero =>
    have heq : iidObservationLaw μ 0 = iidObservationLaw ν 0 := by
      unfold iidObservationLaw
      congr 1
      funext i
      exact Fin.elim0 i
    rw [heq, InformationTheory.klDiv_self]
    simp
  | succ n ih =>
    let e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => Ω) 0
    have hμ := (measurePreserving_piFinSuccAbove (fun _ : Fin (n + 1) => μ) 0).map_eq
    have hν := (measurePreserving_piFinSuccAbove (fun _ : Fin (n + 1) => ν) 0).map_eq
    have heq := klDiv_map_measurableEquiv (iidObservationLaw μ (n + 1))
      (iidObservationLaw ν (n + 1)) e
    change InformationTheory.klDiv (Measure.map e (Measure.pi fun _ => μ))
      (Measure.map e (Measure.pi fun _ => ν)) = _ at heq
    rw [hμ, hν] at heq
    rw [← heq]
    change InformationTheory.klDiv (μ.prod (iidObservationLaw μ n))
      (ν.prod (iidObservationLaw ν n)) = _
    rw [klDiv_prod_add, ih, Nat.cast_add, Nat.cast_one, add_mul, one_mul]
    exact add_comm _ _

/-- **Bretagnolle--Huber for arbitrary measurable probability laws, finite-KL
case.** The binary information inequality is applied to a proved Hahn-optimal
decision region; the binary-partition KL lower bound is itself derived above
from the actual log-likelihood ratio and Gibbs/Jensen for restricted measures. -/
theorem general_bretagnolle_huber_finite {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hKL : InformationTheory.klDiv μ ν ≠ ∞) :
    Real.exp (-(InformationTheory.klDiv μ ν).toReal) / 2
      ≤ 1 - measurableTotalVariation μ ν := by
  obtain ⟨s, hs⟩ := exists_isHahnDecomposition μ ν
  have hsupport := binary_partition_support hKL hs.measurableSet.compl
  have hbinary := binary_bretagnolle_huber
    (p := μ.real sᶜ) (q := ν.real sᶜ) measureReal_nonneg measureReal_le_one
    measureReal_nonneg measureReal_le_one hsupport.1 hsupport.2
  have hdata := binary_partition_kl_le hKL hs.measurableSet.compl
  have hexp := Real.exp_le_exp.mpr (neg_le_neg hdata)
  calc
    Real.exp (-(InformationTheory.klDiv μ ν).toReal) / 2
        ≤ Real.exp (-(μ.real sᶜ * Real.log (μ.real sᶜ / ν.real sᶜ) +
          (1 - μ.real sᶜ) * Real.log ((1 - μ.real sᶜ) / (1 - ν.real sᶜ)))) / 2 := by
            linarith
    _ ≤ (1 - μ.real sᶜ) + ν.real sᶜ := hbinary
    _ = 1 - measurableTotalVariation μ ν := by
      rw [measurableTotalVariation_eq_hahn hs, measureReal_compl hs.measurableSet,
        measureReal_compl hs.measurableSet]
      simp only [probReal_univ]
      ring

/-- The mathematically correct extension of `exp(-D)` to infinite divergence.
In particular, `D=∞` maps to zero, rather than to the erroneous value obtained
by applying the totalized `ENNReal.toReal` first. -/
noncomputable def extendedKLExpNeg (D : ENNReal) : ℝ :=
  if D = ∞ then 0 else Real.exp (-D.toReal)

/-- **General measurable Bretagnolle--Huber bound**, including singular laws and
nonintegrable log-likelihood ratios through the correct infinite-KL convention. -/
theorem general_bretagnolle_huber {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] :
    extendedKLExpNeg (InformationTheory.klDiv μ ν) / 2
      ≤ 1 - measurableTotalVariation μ ν := by
  by_cases hKL : InformationTheory.klDiv μ ν = ∞
  · simp only [extendedKLExpNeg, hKL, if_pos, zero_div]
    exact sub_nonneg.mpr (measurableTotalVariation_le_one μ ν)
  · simpa only [extendedKLExpNeg, hKL, if_false] using
      general_bretagnolle_huber_finite μ ν hKL

/-- The general-measure Le Cam regret floor and its KL exponential lower bound,
with the two factors of two kept distinct: one for max versus sum of testing
errors, one for Bretagnolle--Huber. -/
theorem general_lecam_exponential_regret_floor {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (φ : MeasurableBinaryTest Ω) {Lambda : ℝ} (hLambda : 0 ≤ Lambda) :
    (Lambda / 4) * extendedKLExpNeg (InformationTheory.klDiv μ ν) ≤
      Lambda * max (∫ ω, φ.toFun ω ∂μ) (∫ ω, 1 - φ.toFun ω ∂ν) := by
  calc
    (Lambda / 4) * extendedKLExpNeg (InformationTheory.klDiv μ ν)
        = (Lambda / 2) * (extendedKLExpNeg (InformationTheory.klDiv μ ν) / 2) := by ring
    _ ≤ (Lambda / 2) * (1 - measurableTotalVariation μ ν) :=
      mul_le_mul_of_nonneg_left (general_bretagnolle_huber μ ν) (by positivity)
    _ ≤ _ := general_lecam_regret_floor μ ν φ hLambda

/-- **Finite-sample product-experiment capstone.** For `n` actual independent
observations, the worst-case regret floor is `Lambda/4 * exp(-n KL(μ,ν))`.
The product law, KL additivity, general TV testing identity, and exponential
information inequality are all proved here, including `n=0` and infinite KL. -/
theorem general_lecam_iid_exponential_regret_floor {Ω : Type*} [MeasurableSpace Ω]
    (μ ν : Measure Ω) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν] (n : ℕ)
    (φ : MeasurableBinaryTest (Fin n → Ω)) {Lambda : ℝ} (hLambda : 0 ≤ Lambda) :
    (Lambda / 4) * extendedKLExpNeg (n * InformationTheory.klDiv μ ν) ≤
      Lambda * max (∫ ω, φ.toFun ω ∂iidObservationLaw μ n)
        (∫ ω, 1 - φ.toFun ω ∂iidObservationLaw ν n) := by
  simpa only [klDiv_iidObservationLaw] using
    general_lecam_exponential_regret_floor (iidObservationLaw μ n)
      (iidObservationLaw ν n) φ hLambda

end KBound
