import KBound.Probability.MeasureConformal
import Mathlib.Data.Finset.Max

/-! The kth calibration order statistic defined as the smallest observed score
with at least k calibration scores at or below it. This definition keeps ties.
The zero/unavailable ranks are totalized only here; the actual split radius uses
infinity when the required rank exceeds the available calibration size. -/
namespace KBound
open MeasureTheory
open scoped BigOperators ENNReal

noncomputable def calibrationCandidates {m : ℕ} (R : Fin m → ℝ)
    (j : Fin m) (k : ℕ) : Finset ℝ := by
  classical
  exact ((Finset.univ.erase j).image R).filter (fun q => CalibrationThreshold R j k q)

noncomputable def calibrationOrderStatistic {m : ℕ} (R : Fin m → ℝ)
    (j : Fin m) (k : ℕ) : ℝ := by
  classical
  exact if h : (calibrationCandidates R j k).Nonempty then
    (calibrationCandidates R j k).min' h else 0

/-- Every positive available rank has an observed threshold, including ties. -/
theorem calibrationCandidates_nonempty {n : ℕ} (R : Fin (n+1) → ℝ)
    (j : Fin (n+1)) {k : ℕ} (hk0 : 0 < k) (hkn : k ≤ n) :
    (calibrationCandidates R j k).Nonempty := by
  classical
  let s := Finset.univ.erase j
  have hs : s.card = n := by simp [s]
  have hsn : s.Nonempty := Finset.card_pos.mp (by omega)
  have hvals : (s.image R).Nonempty := hsn.image R
  let q := (s.image R).max' hvals
  have hall : ∀ i ∈ s, R i ≤ q := by
    intro i hi
    exact Finset.le_max' _ _ (Finset.mem_image.mpr ⟨i, hi, rfl⟩)
  have hfilt : s.filter (fun i => R i ≤ q) = s := by
    apply Finset.filter_true_of_mem
    exact hall
  refine ⟨q, ?_⟩
  apply Finset.mem_filter.mpr
  refine ⟨Finset.max'_mem _ hvals, ?_⟩
  change k ≤ (s.filter (fun i => R i ≤ q)).card
  rw [hfilt, hs]
  exact hkn

/-- The constructed order statistic supplies the deterministic quantile property;
the property is proved from the observed finite scores, not supplied as a premise. -/
theorem calibrationOrderStatistic_threshold {n : ℕ} (R : Fin (n+1) → ℝ)
    (j : Fin (n+1)) {k : ℕ} (hkn : k ≤ n) :
    CalibrationThreshold R j k (calibrationOrderStatistic R j k) := by
  classical
  by_cases hk : k = 0
  · subst k
    exact Nat.zero_le _
  · have hn := calibrationCandidates_nonempty R j (Nat.pos_of_ne_zero hk) hkn
    unfold calibrationOrderStatistic
    rw [dif_pos hn]
    exact (Finset.mem_filter.mp (Finset.min'_mem _ hn)).2

/-- The threshold is the least observed value satisfying the rank criterion.
This identifies the construction with the usual kth order statistic under ties. -/
theorem calibrationOrderStatistic_le_candidate {n : ℕ} (R : Fin (n+1) → ℝ)
    (j : Fin (n+1)) {k : ℕ} (hk0 : 0 < k) (hkn : k ≤ n)
    {q : ℝ} (hq : q ∈ (Finset.univ.erase j).image R)
    (hcount : CalibrationThreshold R j k q) : calibrationOrderStatistic R j k ≤ q := by
  classical
  have hn := calibrationCandidates_nonempty R j hk0 hkn
  unfold calibrationOrderStatistic
  rw [dif_pos hn]
  exact Finset.min'_le _ _ (Finset.mem_filter.mpr ⟨hq, hcount⟩)

end KBound
