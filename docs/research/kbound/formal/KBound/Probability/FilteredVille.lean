import Mathlib.Probability.Martingale.OptionalStopping
import Mathlib.MeasureTheory.Measure.Real
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp

/-!
# Filtered optional stopping and Ville's inequality

This module works on an actual probability space with a discrete-time filtration,
strong adaptation, integrability, and conditional-expectation supermartingale
inequalities. It does not infer these hypotheses from an observed null update.

The maximal event ranges over every natural-number time, not a fixed finite horizon.
Optional stopping of expectations is stated for bounded stopping times; the
probability guarantee at an arbitrary finite time follows from the maximal event
and does not assert an unbounded optional-stopping equality.
-/

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal BigOperators ProbabilityTheory

namespace KBound

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}
  {ℱ : Filtration ℕ mΩ} {W : ℕ → Ω → ℝ}

/-- A supermartingale's expectation cannot increase at a bounded stopping time.
The stopping time may depend on all information available in the filtration. -/
theorem filtered_optional_stopping_le [IsFiniteMeasure μ]
    (hW : Supermartingale W ℱ μ) {τ : Ω → ℕ∞}
    (hτ : IsStoppingTime ℱ τ) {N : ℕ} (hN : ∀ ω, τ ω ≤ N) :
    (∫ ω, stoppedValue W τ ω ∂μ) ≤ ∫ ω, W 0 ω ∂μ := by
  have h := hW.neg.expected_stoppedValue_mono
    (isStoppingTime_const ℱ 0) hτ (fun _ ↦ bot_le) hN
  change (∫ ω, -W 0 ω ∂μ) ≤ ∫ ω, -stoppedValue W τ ω ∂μ at h
  simpa only [integral_neg, neg_le_neg_iff] using h

/-- Finite-horizon Ville inequality obtained by stopping at the first crossing.
The nonnegative wealth assumption is separate from the supermartingale property. -/
theorem filtered_ville_finite [IsFiniteMeasure μ]
    (hW : Supermartingale W ℱ μ) (h_nonneg : ∀ n ω, 0 ≤ W n ω)
    {a : ℝ} (ha : 0 < a) (N : ℕ) :
    μ.real {ω | ∃ n ≤ N, a ≤ W n ω} ≤ (∫ ω, W 0 ω ∂μ) / a := by
  let τ : Ω → ℕ∞ := fun ω ↦ (hittingBtwn W (Set.Ici a) 0 N ω : ℕ)
  have hτ : IsStoppingTime ℱ τ :=
    hW.stronglyAdapted.adapted.isStoppingTime_hittingBtwn measurableSet_Ici
  have hN : ∀ ω, τ ω ≤ N := fun ω ↦ by
    change ((hittingBtwn W (Set.Ici a) 0 N ω : ℕ) : ℕ∞) ≤ (N : ℕ∞)
    exact_mod_cast hittingBtwn_le (u := W) (s := Set.Ici a) (n := 0) (m := N) ω
  have hi : Integrable (stoppedValue W τ) μ :=
    integrable_stoppedValue ℕ hτ hW.integrable hN
  have hn : 0 ≤ᵐ[μ] stoppedValue W τ :=
    ae_of_all _ fun ω ↦ h_nonneg _ ω
  have hs : {ω | ∃ n ≤ N, a ≤ W n ω} ⊆ {ω | a ≤ stoppedValue W τ ω} := by
    rintro ω ⟨n, hnN, han⟩
    exact stoppedValue_hittingBtwn_mem ⟨n, ⟨Nat.zero_le n, hnN⟩, han⟩
  have hMarkov := mul_meas_ge_le_integral_of_nonneg hn hi a
  have hstop := filtered_optional_stopping_le hW hτ hN
  apply (le_div_iff₀ ha).2
  calc
    μ.real {ω | ∃ n ≤ N, a ≤ W n ω} * a
        ≤ μ.real {ω | a ≤ stoppedValue W τ ω} * a :=
      mul_le_mul_of_nonneg_right (measureReal_mono hs) ha.le
    _ ≤ ∫ ω, W 0 ω ∂μ := by nlinarith

/-- **Ville's inequality**, with the crossing event over the entire discrete time
axis. The proof passes from bounded hitting times to a countable increasing union. -/
theorem filtered_ville [IsFiniteMeasure μ]
    (hW : Supermartingale W ℱ μ) (h_nonneg : ∀ n ω, 0 ≤ W n ω)
    {a : ℝ} (ha : 0 < a) :
    μ.real {ω | ∃ n : ℕ, a ≤ W n ω} ≤ (∫ ω, W 0 ω ∂μ) / a := by
  let A : ℕ → Set Ω := fun N ↦ {ω | ∃ n ≤ N, a ≤ W n ω}
  have hA : Monotone A := by
    intro N M hNM ω hω
    obtain ⟨n, hnN, han⟩ := hω
    exact ⟨n, hnN.trans hNM, han⟩
  have h_union : {ω | ∃ n : ℕ, a ≤ W n ω} = ⋃ N, A N := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, A]
    constructor
    · rintro ⟨n, hn⟩
      exact ⟨n, n, le_rfl, hn⟩
    · rintro ⟨N, n, _, hn⟩
      exact ⟨n, hn⟩
  have hc : 0 ≤ (∫ ω, W 0 ω ∂μ) / a :=
    div_nonneg (integral_nonneg (h_nonneg 0)) ha.le
  apply ENNReal.toReal_le_of_le_ofReal hc
  rw [h_union, hA.measure_iUnion]
  apply iSup_le
  intro N
  exact (ENNReal.le_ofReal_iff_toReal_le (measure_ne_top μ (A N)) hc).2
    (filtered_ville_finite hW h_nonneg ha N)

/-- Unit-capital Ville guarantee. This is an unconditional crossing probability,
not a conditional error rate among the paths that cross. -/
theorem filtered_ville_alpha [IsProbabilityMeasure μ]
    (hW : Supermartingale W ℱ μ) (h_nonneg : ∀ n ω, 0 ≤ W n ω)
    (h_initial : (∫ ω, W 0 ω ∂μ) ≤ 1) {alpha : ℝ} (hα : 0 < alpha) :
    μ.real {ω | ∃ n : ℕ, 1 / alpha ≤ W n ω} ≤ alpha := by
  have h := filtered_ville hW h_nonneg (one_div_pos.mpr hα)
  calc
    _ ≤ (∫ ω, W 0 ω ∂μ) / (1 / alpha) := h
    _ ≤ 1 / (1 / alpha) := div_le_div_of_nonneg_right h_initial (one_div_pos.mpr hα).le
    _ = alpha := by simp

/-- Any process dominated by the same nonnegative supermartingale inherits its
time-uniform crossing guarantee. This covers e-processes defined by domination;
the dominating process and filtration must be justified under the null law. -/
theorem dominated_eprocess_ville [IsProbabilityMeasure μ]
    (hW : Supermartingale W ℱ μ) (h_nonneg : ∀ n ω, 0 ≤ W n ω)
    (h_initial : (∫ ω, W 0 ω ∂μ) ≤ 1) {E : ℕ → Ω → ℝ}
    (h_dom : ∀ n ω, E n ω ≤ W n ω) {alpha : ℝ} (hα : 0 < alpha) :
    μ.real {ω | ∃ n : ℕ, 1 / alpha ≤ E n ω} ≤ alpha := by
  refine (measureReal_mono ?_).trans (filtered_ville_alpha hW h_nonneg h_initial hα)
  rintro ω ⟨n, hn⟩
  exact ⟨n, hn.trans (h_dom n ω)⟩

/-- The same probability bound holds at any finite data-dependent time, since
its crossing event is a subset of the full maximal event. No optional-stopping
expectation identity, boundedness, or independence of that time is asserted. -/
theorem eprocess_finite_time_crossing [IsProbabilityMeasure μ]
    (hW : Supermartingale W ℱ μ) (h_nonneg : ∀ n ω, 0 ≤ W n ω)
    (h_initial : (∫ ω, W 0 ω ∂μ) ≤ 1) {E : ℕ → Ω → ℝ}
    (h_dom : ∀ n ω, E n ω ≤ W n ω) (τ : Ω → ℕ)
    {alpha : ℝ} (hα : 0 < alpha) :
    μ.real {ω | 1 / alpha ≤ E (τ ω) ω} ≤ alpha := by
  refine (measureReal_mono ?_).trans (dominated_eprocess_ville hW h_nonneg h_initial h_dom hα)
  intro ω hω
  exact ⟨τ ω, hω⟩

/-- Predictable nonnegative betting stakes turn conditionally nonpositive
increments into a genuine supermartingale. Wealth adaptation and integrability
are explicit obligations of the betting construction; a pointwise observed
negative increment is not substituted for the conditional-null assumption. -/
theorem filtered_betting_supermartingale [IsFiniteMeasure μ]
    (h_adapted : StronglyAdapted ℱ W) (h_integrable : ∀ n, Integrable (W n) μ)
    (h_nonneg : ∀ n ω, 0 ≤ W n ω) {lam X : ℕ → Ω → ℝ}
    (h_predictable : ∀ n, StronglyMeasurable[ℱ n] (lam n))
    (h_lam : ∀ n ω, 0 ≤ lam n ω) (hX : ∀ n, Integrable (X n) μ)
    (h_update : ∀ n ω, W (n + 1) ω = W n ω * (1 + lam n ω * X n ω))
    (h_null : ∀ n, μ[X n | ℱ n] ≤ᵐ[μ] (0 : Ω → ℝ)) :
    Supermartingale W ℱ μ := by
  apply supermartingale_nat h_adapted h_integrable
  intro n
  let stake : Ω → ℝ := fun ω ↦ W n ω * lam n ω
  have hdifference : (fun ω ↦ stake ω * X n ω) = W (n + 1) - W n := by
    funext ω
    simp only [Pi.sub_apply, h_update, stake]
    ring
  have hi : Integrable (stake * X n) μ := by
    change Integrable (fun ω ↦ stake ω * X n ω) μ
    rw [hdifference]
    exact (h_integrable (n + 1)).sub (h_integrable n)
  have h_pull := condExp_mul_of_stronglyMeasurable_left
    ((h_adapted n).mul (h_predictable n)) hi (hX n)
  have hdiff := condExp_sub (h_integrable (n + 1)) (h_integrable n) (ℱ n)
  have hself := condExp_of_stronglyMeasurable (ℱ.le n) (h_adapted n) (h_integrable n)
  change μ[(fun ω ↦ stake ω * X n ω) | ℱ n] =ᵐ[μ] stake * μ[X n | ℱ n] at h_pull
  rw [hdifference] at h_pull
  filter_upwards [h_pull, hdiff, h_null n] with ω hp hd hn
  simp only [Pi.sub_apply, Pi.mul_apply, Pi.zero_apply, hself] at hp hd hn ⊢
  have hs : 0 ≤ stake ω := mul_nonneg (h_nonneg n ω) (h_lam n ω)
  have hprod : stake ω * (μ[X n | ℱ n]) ω ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hs hn
  linarith

/-- Time-uniform false-crossing control for predictable betting under the
conditional null, obtained from the constructed supermartingale and Ville. -/
theorem filtered_betting_anytime [IsProbabilityMeasure μ]
    (h_adapted : StronglyAdapted ℱ W) (h_integrable : ∀ n, Integrable (W n) μ)
    (h_nonneg : ∀ n ω, 0 ≤ W n ω) (h_initial : (∫ ω, W 0 ω ∂μ) ≤ 1)
    {lam X : ℕ → Ω → ℝ}
    (h_predictable : ∀ n, StronglyMeasurable[ℱ n] (lam n))
    (h_lam : ∀ n ω, 0 ≤ lam n ω) (hX : ∀ n, Integrable (X n) μ)
    (h_update : ∀ n ω, W (n + 1) ω = W n ω * (1 + lam n ω * X n ω))
    (h_null : ∀ n, μ[X n | ℱ n] ≤ᵐ[μ] (0 : Ω → ℝ))
    {alpha : ℝ} (hα : 0 < alpha) :
    μ.real {ω | ∃ n : ℕ, 1 / alpha ≤ W n ω} ≤ alpha :=
  filtered_ville_alpha
    (filtered_betting_supermartingale h_adapted h_integrable h_nonneg
      h_predictable h_lam hX h_update h_null) h_nonneg h_initial hα

/-- Wealth after `n` bets. Increment `X i` is observed at time `i+1`, while
`lam i` must be measurable at time `i`, before that increment is observed. -/
noncomputable def predictableBettingWealth (lam X : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  ∏ i ∈ Finset.range n, (1 + lam i ω * X i ω)

@[simp] theorem predictable_betting_wealth_zero (lam X : ℕ → Ω → ℝ) (ω : Ω) :
    predictableBettingWealth lam X 0 ω = 1 := by
  simp [predictableBettingWealth]

theorem predictable_betting_wealth_succ (lam X : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω) :
    predictableBettingWealth lam X (n + 1) ω =
      predictableBettingWealth lam X n ω * (1 + lam n ω * X n ω) := by
  simp [predictableBettingWealth, Finset.prod_range_succ]

/-- Capped bets on normalized bounded increments have factors in `[0,2]`. -/
theorem capped_betting_factor_bounds {lam x : ℝ}
    (hlam : lam ∈ Set.Icc (0 : ℝ) 1) (hx : x ∈ Set.Icc (-1 : ℝ) 1) :
    1 + lam * x ∈ Set.Icc (0 : ℝ) 2 := by
  have hlow := mul_le_mul_of_nonneg_left hx.1 hlam.1
  have hhigh := mul_le_mul_of_nonneg_left hx.2 hlam.1
  constructor <;> nlinarith [hlam.2]

/-- Finite-horizon wealth is deterministically bounded by `2^n`. This derives
integrability rather than assuming it as an optional-stopping shortcut. -/
theorem predictable_betting_wealth_bounds {lam X : ℕ → Ω → ℝ}
    (h_lam : ∀ n ω, lam n ω ∈ Set.Icc (0 : ℝ) 1)
    (hX : ∀ n ω, X n ω ∈ Set.Icc (-1 : ℝ) 1) (n : ℕ) (ω : Ω) :
    predictableBettingWealth lam X n ω ∈ Set.Icc (0 : ℝ) (2 ^ n) := by
  constructor
  · apply Finset.prod_nonneg
    intro i _
    exact (capped_betting_factor_bounds (h_lam i ω) (hX i ω)).1
  · calc
      _ ≤ ∏ _i ∈ Finset.range n, (2 : ℝ) := by
        apply Finset.prod_le_prod
        · intro i _
          exact (capped_betting_factor_bounds (h_lam i ω) (hX i ω)).1
        · intro i _
          exact (capped_betting_factor_bounds (h_lam i ω) (hX i ω)).2
      _ = _ := by simp

/-- Correct information timing makes the constructed wealth adapted. -/
theorem predictable_betting_wealth_adapted {lam X : ℕ → Ω → ℝ}
    (h_lam : ∀ n, StronglyMeasurable[ℱ n] (lam n))
    (hX : ∀ n, StronglyMeasurable[ℱ (n + 1)] (X n)) :
    StronglyAdapted ℱ (predictableBettingWealth lam X) := by
  intro n
  apply Finset.stronglyMeasurable_fun_prod
  intro i hi
  have hin : i < n := Finset.mem_range.mp hi
  exact stronglyMeasurable_const.add
    (((h_lam i).mono (ℱ.mono hin.le)).mul ((hX i).mono (ℱ.mono (Nat.succ_le_of_lt hin))))

/-- A fully constructed predictable betting process is anytime valid for
normalized bounded increments satisfying the conditional null. Nonnegativity,
adaptation, and integrability are proved from the bounds and timing assumptions.
The probability space and its null law remain fixed throughout deployment. -/
theorem bounded_predictable_betting_anytime [IsProbabilityMeasure μ]
    {lam X : ℕ → Ω → ℝ}
    (h_lam_meas : ∀ n, StronglyMeasurable[ℱ n] (lam n))
    (hX_meas : ∀ n, StronglyMeasurable[ℱ (n + 1)] (X n))
    (h_lam_bounds : ∀ n ω, lam n ω ∈ Set.Icc (0 : ℝ) 1)
    (hX_bounds : ∀ n ω, X n ω ∈ Set.Icc (-1 : ℝ) 1)
    (h_null : ∀ n, μ[X n | ℱ n] ≤ᵐ[μ] (0 : Ω → ℝ))
    {alpha : ℝ} (hα : 0 < alpha) :
    μ.real {ω | ∃ n : ℕ, 1 / alpha ≤ predictableBettingWealth lam X n ω} ≤ alpha := by
  have hadapt := predictable_betting_wealth_adapted h_lam_meas hX_meas
  have hwealth := predictable_betting_wealth_bounds h_lam_bounds hX_bounds
  have hint (n : ℕ) : Integrable (predictableBettingWealth lam X n) μ := by
    apply Integrable.of_bound (hadapt.stronglyMeasurable.aestronglyMeasurable) (2 ^ n)
    exact ae_of_all _ fun ω ↦ by
      simpa only [Real.norm_eq_abs, abs_of_nonneg (hwealth n ω).1] using (hwealth n ω).2
  have hXint (n : ℕ) : Integrable (X n) μ := by
    apply Integrable.of_bound (((hX_meas n).mono (ℱ.le (n + 1))).aestronglyMeasurable) 1
    exact ae_of_all _ fun ω ↦ by
      simpa only [Real.norm_eq_abs, abs_le] using hX_bounds n ω
  apply filtered_betting_anytime hadapt hint (fun n ω ↦ (hwealth n ω).1) _
    h_lam_meas (fun n ω ↦ (h_lam_bounds n ω).1) hXint
    (predictable_betting_wealth_succ lam X) h_null hα
  simp

end KBound
