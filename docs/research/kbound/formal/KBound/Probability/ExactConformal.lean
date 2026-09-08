import KBound.Probability.MeasureConformal
import KBound.Probability.ExtendedRadiusCertificate
import Mathlib.Data.Finset.Powerset
import Mathlib.Algebra.Order.Floor.Semiring

/-!
# Literal split-conformal order statistic, with the infinity convention

The k-th order statistic of nonnegative calibration scores is the minimum,
over k-element calibration subsets, of the maximum score in that subset.
This finite min/max representation keeps repeated scores and gives infinity
when fewer than k calibration indices exist. No quantile property, absence of
ties or rank-distribution conclusion is postulated.
-/

namespace KBound.ExactConformal

open MeasureTheory
open scoped ENNReal

noncomputable def orderRadius {m : ℕ} (R : Fin m → ℝ) (j : Fin m) (k : ℕ) : ℝ≥0∞ :=
  ((Finset.univ.erase j).powersetCard k).inf (fun A => A.sup (fun i => ENNReal.ofReal (R i)))

noncomputable def rank (n : ℕ) (alpha : ℝ) : ℕ :=
  Nat.ceil ((n + 1 : ℕ) * (1 - alpha))

noncomputable def radius {n : ℕ} (R : Fin (n + 1) → ℝ)
    (j : Fin (n + 1)) (alpha : ℝ) : ℝ≥0∞ := orderRadius R j (rank n alpha)

theorem orderRadius_le_iff {m : ℕ} (R : Fin m → ℝ) (j : Fin m)
    (k : ℕ) {r : ℝ≥0∞} (hr : r < ⊤) :
    orderRadius R j k ≤ r ↔
      k ≤ ((Finset.univ.erase j).filter (fun i => ENNReal.ofReal (R i) ≤ r)).card := by
  classical
  rw [orderRadius, Finset.inf_le_iff hr]
  constructor
  · rintro ⟨A, hA, hsup⟩
    rcases Finset.mem_powersetCard.mp hA with ⟨hsub, hcard⟩
    rw [← hcard]
    apply Finset.card_le_card
    intro i hi
    exact Finset.mem_filter.mpr ⟨hsub hi, (Finset.sup_le_iff.mp hsup) i hi⟩
  · intro hk
    obtain ⟨A, hA, hcard⟩ := Finset.exists_subset_card_eq hk
    refine ⟨A, Finset.mem_powersetCard.mpr ⟨?_, hcard⟩, ?_⟩
    · exact hA.trans (Finset.filter_subset _ _)
    · exact Finset.sup_le_iff.mpr fun i hi => (Finset.mem_filter.mp (hA hi)).2

theorem orderRadius_eq_top {n : ℕ} (R : Fin (n + 1) → ℝ)
    (j : Fin (n + 1)) {k : ℕ} (hk : n < k) : orderRadius R j k = ⊤ := by
  classical
  have hcard : (Finset.univ.erase j).card = n := by simp
  have hempty : (Finset.univ.erase j).powersetCard k = ∅ :=
    Finset.powersetCard_eq_empty.mpr (hcard.symm ▸ hk)
  simp [orderRadius, hempty]

theorem orderRadius_real_characterization {m : ℕ} (R : Fin m → ℝ)
    (j : Fin m) (k : ℕ) {q : ℝ} (hq : 0 ≤ q) :
    orderRadius R j k ≤ ENNReal.ofReal q ↔ CalibrationThreshold R j k q := by
  simpa only [CalibrationThreshold, ENNReal.ofReal_le_ofReal_iff hq] using
    orderRadius_le_iff R j k (r := ENNReal.ofReal q) ENNReal.ofReal_lt_top

theorem orderRadius_ignores_heldout {m : ℕ} (R S : Fin m → ℝ)
    (j : Fin m) (k : ℕ) (heq : ∀ i, i ≠ j → R i = S i) :
    orderRadius R j k = orderRadius S j k := by
  classical
  apply Finset.inf_congr rfl
  intro A hA
  apply Finset.sup_congr rfl
  intro i hi
  rw [heq i (Finset.mem_erase.mp ((Finset.mem_powersetCard.mp hA).1 hi)).1]

theorem orderRadius_at_full_rank {n : ℕ} (R : Fin (n + 1) → ℝ)
    (j : Fin (n + 1)) :
    orderRadius R j n = (Finset.univ.erase j).sup (fun i => ENNReal.ofReal (R i)) := by
  classical
  have hcard : (Finset.univ.erase j).card = n := by simp
  have hpower : (Finset.univ.erase j).powersetCard n = {Finset.univ.erase j} := by
    simpa only [hcard] using Finset.powersetCard_self (Finset.univ.erase j)
  simp [orderRadius, hpower]

theorem orderRadius_lt_top {n : ℕ} (R : Fin (n + 1) → ℝ)
    (j : Fin (n + 1)) {k : ℕ} (hk : k ≤ n) : orderRadius R j k < ⊤ := by
  classical
  have hcard : (Finset.univ.erase j).card = n := by simp
  obtain ⟨A, hA⟩ := Finset.powersetCard_nonempty.mpr (hcard.symm ▸ hk)
  apply Finset.inf_lt_iff.mpr
  refine ⟨A, hA, (Finset.sup_lt_iff (show (⊥ : ℝ≥0∞) < ⊤ from bot_lt_top)).mpr ?_⟩
  intro i _
  exact ENNReal.ofReal_lt_top

theorem finite_orderRadius_threshold {n : ℕ} (R : Fin (n + 1) → ℝ)
    (j : Fin (n + 1)) {k : ℕ} (hk : k ≤ n) :
    CalibrationThreshold R j k (orderRadius R j k).toReal := by
  classical
  have hr := orderRadius_lt_top R j hk
  have h := (orderRadius_le_iff R j k hr).mp le_rfl
  simpa only [CalibrationThreshold, ENNReal.ofReal_le_iff_le_toReal hr.ne] using h

theorem measurable_orderRadius {m : ℕ} (j : Fin m) (k : ℕ) :
    Measurable (fun R : Fin m → ℝ => orderRadius R j k) := by
  classical
  have hsup (A : Finset (Fin m)) :
      Measurable (fun R : Fin m → ℝ => A.sup (fun i => ENNReal.ofReal (R i))) := by
    induction A using Finset.induction_on with
    | empty => simp
    | @insert i A hi ih =>
      simpa only [Finset.sup_insert] using
        (ENNReal.measurable_ofReal.comp (measurable_pi_apply i)).sup ih
  unfold orderRadius
  generalize (Finset.univ.erase j).powersetCard k = S
  induction S using Finset.induction_on with
  | empty => simp
  | @insert A S hA ih => simpa only [Finset.inf_insert] using (hsup A).inf ih

theorem rank_positive (n : ℕ) {alpha : ℝ} (hα1 : alpha < 1) : 0 < rank n alpha := by
  apply Nat.one_le_ceil_iff.mpr
  exact mul_pos (by positivity) (sub_pos.mpr hα1)

theorem rank_le_total (n : ℕ) {alpha : ℝ} (hα0 : 0 ≤ alpha) : rank n alpha ≤ n + 1 := by
  apply Nat.ceil_le.mpr
  have hn : (0 : ℝ) ≤ (n + 1 : ℕ) := by positivity
  nlinarith

theorem ceiling_rank_budget (n : ℕ) {alpha : ℝ} (hα0 : 0 ≤ alpha) :
    (((n + 1 - rank n alpha : ℕ) : ℝ≥0∞) / ((n + 1 : ℕ) : ℝ≥0∞)) ≤
      ENNReal.ofReal alpha := by
  have hn : (0 : ℝ) < (n + 1 : ℕ) := by positivity
  have hk := rank_le_total n hα0
  have hceil : ((n + 1 : ℕ) : ℝ) * (1 - alpha) ≤ (rank n alpha : ℝ) :=
    Nat.le_ceil _
  have hreal : ((n + 1 - rank n alpha : ℕ) : ℝ) / ((n + 1 : ℕ) : ℝ) ≤ alpha := by
    apply (div_le_iff₀ hn).mpr
    rw [Nat.cast_sub hk]
    nlinarith
  calc
    _ = ENNReal.ofReal (((n + 1 - rank n alpha : ℕ) : ℝ) /
        ((n + 1 : ℕ) : ℝ)) := by
      rw [ENNReal.ofReal_div_of_pos hn]
      simp only [ENNReal.ofReal_natCast]
    _ ≤ ENNReal.ofReal alpha := ENNReal.ofReal_le_ofReal hreal

theorem no_calibration_radius (R : Fin 1 → ℝ) (j : Fin 1)
    {alpha : ℝ} (_hα0 : 0 ≤ alpha) (hα1 : alpha < 1) : radius R j alpha = ⊤ :=
  orderRadius_eq_top R j (rank_positive 0 hα1)

/-- At a feasible positive rank, the min/max value is one of the calibration
scores. Together with `orderRadius_le_iff`, this exactly characterizes the
k-th order statistic including repeated scores. For nonnegative residuals,
`ofReal` preserves the original real score values. -/
theorem orderRadius_attained {n : ℕ} (R : Fin (n + 1) → ℝ)
    (j : Fin (n + 1)) {k : ℕ} (hk0 : 0 < k) (hk : k ≤ n) :
    ∃ i ∈ Finset.univ.erase j, orderRadius R j k = ENNReal.ofReal (R i) := by
  classical
  have hcard : (Finset.univ.erase j).card = n := by simp
  have hS := Finset.powersetCard_nonempty.mpr (hcard.symm ▸ hk)
  obtain ⟨A, hA, hmin⟩ := Finset.exists_mem_eq_inf _ hS
    (fun A => A.sup (fun i => ENNReal.ofReal (R i)))
  rcases Finset.mem_powersetCard.mp hA with ⟨hsub, hAk⟩
  have hAnonempty : A.Nonempty := Finset.card_pos.mp (hAk ▸ hk0)
  obtain ⟨i, hi, hmax⟩ := Finset.exists_mem_eq_sup A hAnonempty (fun i => ENNReal.ofReal (R i))
  exact ⟨i, hsub hi, hmin.trans hmax⟩

theorem real_order_statistic_attained {n : ℕ} (R : Fin (n + 1) → ℝ)
    (j : Fin (n + 1)) (hR0 : ∀ i, 0 ≤ R i) {k : ℕ}
    (hk0 : 0 < k) (hk : k ≤ n) :
    ∃ i ∈ Finset.univ.erase j, (orderRadius R j k).toReal = R i := by
  obtain ⟨i, hi, hvalue⟩ := orderRadius_attained R j hk0 hk
  exact ⟨i, hi, by rw [hvalue, ENNReal.toReal_ofReal (hR0 i)]⟩

/-- One-shot marginal coverage for the literal ceiling-rank construction.
The fixed held-out coordinate may be any coordinate; ties are allowed. No
quantile property or finite-radius assumption is an input. -/
theorem coverage {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} {R : Ω → Fin (n + 1) → ℝ}
    (hR : Measurable R) (hexch : ExchangeableScores μ R)
    (j : Fin (n + 1)) {alpha : ℝ} (hα0 : 0 ≤ alpha) (_hα1 : alpha < 1) :
    1 - ENNReal.ofReal alpha ≤ μ {ω | ENNReal.ofReal (R ω j) ≤ radius (R ω) j alpha} := by
  by_cases hk : rank n alpha ≤ n
  · have hq : Measurable (fun ω => (orderRadius (R ω) j (rank n alpha)).toReal) :=
      ((measurable_orderRadius j (rank n alpha)).comp hR).ennreal_toReal
    have hcov := exchangeable_calibration_threshold_coverage_ge hR hexch j (rank n alpha)
      hq (fun ω => finite_orderRadius_threshold (R ω) j hk) (ceiling_rank_budget n hα0)
    apply hcov.trans
    apply measure_mono
    intro ω hω
    exact (ENNReal.ofReal_le_iff_le_toReal (orderRadius_lt_top (R ω) j hk).ne).mpr hω
  · have htop (ω : Ω) : radius (R ω) j alpha = ⊤ :=
      orderRadius_eq_top (R ω) j (Nat.lt_of_not_ge hk)
    simpa only [htop, le_top, Set.setOf_true, measure_univ] using
      (tsub_le_self : (1 : ℝ≥0∞) - ENNReal.ofReal alpha ≤ 1)

theorem residual_directional_error {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] {n : ℕ}
    {R : Ω → Fin (n + 1) → ℝ} (hR : Measurable R)
    (hexch : ExchangeableScores μ R) (j : Fin (n + 1))
    {estimate truth : Ω → ℝ} (he : Measurable estimate) (ht : Measurable truth)
    (hresidual : ∀ ω, R ω j = |estimate ω - truth ω|)
    {alpha : ℝ} (hα0 : 0 ≤ alpha) (hα1 : alpha < 1) :
    μ (ExtendedRadiusCertificate.falseDirection estimate truth
      (fun ω => radius (R ω) j alpha)) ≤ ENNReal.ofReal alpha := by
  apply ExtendedRadiusCertificate.directional_error_probability μ he ht
    ((measurable_orderRadius j (rank n alpha)).comp hR)
  simpa only [ExtendedRadiusCertificate.coverage, hresidual] using coverage hR hexch j hα0 hα1

/-- End-to-end mathematical composition at the paper's literal absolute
residual scores. The probability law includes all fitting/calibration/evaluation
randomness. Exchangeability is explicit and is not inferred from cross-fitting.
This does not certify operational availability before new-unit label access. -/
theorem literal_residual_certificate {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] {n : ℕ}
    {estimate truth : Ω → Fin (n + 1) → ℝ}
    (he : Measurable estimate) (ht : Measurable truth)
    (hexch : ExchangeableScores μ (fun ω i => |estimate ω i - truth ω i|))
    (j : Fin (n + 1)) {alpha : ℝ} (hα0 : 0 ≤ alpha) (hα1 : alpha < 1) :
    let r := fun ω => radius (fun i => |estimate ω i - truth ω i|) j alpha
    1 - ENNReal.ofReal alpha ≤ μ (ExtendedRadiusCertificate.coverage
      (fun ω => estimate ω j) (fun ω => truth ω j) r) ∧
    μ (ExtendedRadiusCertificate.falseDirection (fun ω => estimate ω j)
      (fun ω => truth ω j) r) ≤ ENNReal.ofReal alpha := by
  have hR : Measurable (fun ω i => |estimate ω i - truth ω i|) :=
    measurable_pi_lambda _ fun i =>
      continuous_abs.measurable.comp
        (((measurable_pi_apply i).comp he).sub ((measurable_pi_apply i).comp ht))
  exact ⟨coverage hR hexch j hα0 hα1,
    residual_directional_error μ hR hexch j ((measurable_pi_apply j).comp he)
      ((measurable_pi_apply j).comp ht) (fun _ => rfl) hα0 hα1⟩

end KBound.ExactConformal
