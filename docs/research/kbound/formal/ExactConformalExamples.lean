import KBound

open MeasureTheory

-- These checks must fail if only an assumed quantile threshold remains: the
-- literal measurable construction, its infinity branch and its probability
-- guarantee must all be present.
#check KBound.ExactConformal.orderRadius_le_iff
#check KBound.ExactConformal.orderRadius_eq_top
#check KBound.ExactConformal.measurable_orderRadius
#check KBound.ExactConformal.ceiling_rank_budget

example {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {n : ℕ} {R : Ω → Fin (n + 1) → ℝ}
    (hR : Measurable R) (hexch : KBound.ExchangeableScores μ R)
    (j : Fin (n + 1)) {alpha : ℝ} (hα0 : 0 ≤ alpha) (hα1 : alpha < 1) :
    1 - ENNReal.ofReal alpha ≤ μ {ω | ENNReal.ofReal (R ω j) ≤
      KBound.ExactConformal.radius (R ω) j alpha} := by
  exact KBound.ExactConformal.coverage hR hexch j hα0 hα1

example (R : Fin 1 → ℝ) :
    KBound.ExactConformal.radius R 0 (1 / 10) = ⊤ := by
  exact KBound.ExactConformal.no_calibration_radius R 0 (by norm_num) (by norm_num)

-- One calibration cell is insufficient for alpha = .1; no finite placeholder
-- can replace the exact infinity convention.
example (R : Fin 2 → ℝ) : KBound.ExactConformal.radius R 1 (1 / 10) = ⊤ := by
  apply KBound.ExactConformal.orderRadius_eq_top
  apply Nat.lt_ceil.mpr
  norm_num

example {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] {n : ℕ} {estimate truth : Ω → Fin (n + 1) → ℝ}
    (he : Measurable estimate) (ht : Measurable truth)
    (hexch : KBound.ExchangeableScores μ (fun ω i => |estimate ω i - truth ω i|))
    (j : Fin (n + 1)) {alpha : ℝ} (hα0 : 0 ≤ alpha) (hα1 : alpha < 1) :
    let r := fun ω => KBound.ExactConformal.radius
      (fun i => |estimate ω i - truth ω i|) j alpha
    μ (KBound.ExtendedRadiusCertificate.falseDirection (fun ω => estimate ω j)
      (fun ω => truth ω j) r) ≤ ENNReal.ofReal alpha :=
  (KBound.ExactConformal.literal_residual_certificate μ he ht hexch j hα0 hα1).2

-- The boundary n = 9, alpha = .1 selects the calibration maximum exactly.
example (R : Fin 10 → ℝ) (j : Fin 10) :
    KBound.ExactConformal.radius R j (1 / 10) =
      (Finset.univ.erase j).sup (fun i => ENNReal.ofReal (R i)) := by
  have hk : KBound.ExactConformal.rank 9 (1 / 10) = 9 := by
    norm_num [KBound.ExactConformal.rank]
  simpa only [KBound.ExactConformal.radius, hk] using
    KBound.ExactConformal.orderRadius_at_full_rank R j

-- Repeated scores do not require tie breaking or continuous score laws.
example : KBound.ExactConformal.radius (fun _ : Fin 4 => (2 : ℝ)) 3 (1 / 2) =
    ENNReal.ofReal 2 := by
  have hk : KBound.ExactConformal.rank 3 (1 / 2) = 2 := by
    norm_num [KBound.ExactConformal.rank]
  obtain ⟨i, _, hi⟩ := KBound.ExactConformal.orderRadius_attained
    (fun _ : Fin 4 => (2 : ℝ)) 3 (show 0 < 2 by norm_num) (show 2 ≤ 3 by norm_num)
  simpa only [KBound.ExactConformal.radius, hk] using hi
