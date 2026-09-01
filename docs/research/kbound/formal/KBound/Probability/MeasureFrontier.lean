import KBound.Probability.MeasureTarget
import Mathlib.Tactic.Linarith

/-!
# The strict frontier on a full measurable correctness-field class

The input probability space, predictors, and off-disagreement label kernel are
fixed. A world is the actual joint law constructed from an arbitrary measurable
correctness field with values in `[0,1]`. Its benefit is the integral of the
zero-one loss difference, not a freely assigned scalar.
On disagreement these constructed kernels place their mass on the two predicted
labels. This is the full measurable correctness-field class for that support,
not the class of every unconstrained multiclass label kernel.

The capstones below derive richness from `MeasureTarget`'s label-kernel
construction. Feasible margins, nonnegative budgets, and positive disagreement
mass are explicit. Necessity uses the clipped identified interval, so it never
asks for impossible correctness probabilities when the budget is large.
-/

namespace KBound

open MeasureTheory ProbabilityTheory Set
open scoped ENNReal ProbabilityTheory

variable {X Y E : Type*} [MeasurableSpace X] [MeasurableSpace Y]
  [MeasurableSpace E] [MeasurableSingletonClass Y] [MeasurableEq Y]

/-- The genuine joint target law associated with a measurable correctness
field. Off disagreement, the original label kernel is unchanged. -/
noncomputable def correctnessFieldTarget (μ : Measure X)
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) (η : CorrectnessField X) : Measure (X × Y) :=
  μ ⊗ₘ targetLabelKernel f₀ fₐ h₀ hₐ κ₀ η.value η.measurable_value

/-- Every field in the quantified class produces a probability measure with
the prescribed input marginal, off-disagreement conditional law, and every
measurable label-free evidence law. -/
theorem correctnessFieldTarget_properties (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀] (η : CorrectnessField X) :
    IsProbabilityMeasure (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η) ∧
    (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η).fst = μ ∧
    (∀ x, f₀ x ≠ fₐ x →
      targetLabelKernel f₀ fₐ h₀ hₐ κ₀ η.value η.measurable_value x {fₐ x} = η.value x) ∧
    (∀ x, f₀ x = fₐ x →
      targetLabelKernel f₀ fₐ h₀ hₐ κ₀ η.value η.measurable_value x = κ₀ x) ∧
    (∀ g : X → E, Measurable g →
      (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η).map (fun xy => g xy.1) = μ.map g) := by
  letI := targetLabelKernel_isMarkov f₀ fₐ h₀ hₐ κ₀ η.value η.measurable_value η.value_le_one
  refine ⟨?_, Measure.fst_compProd μ _, ?_, ?_, ?_⟩
  · unfold correctnessFieldTarget
    infer_instance
  · exact fun _ hx => targetLabelKernel_on_disagreement f₀ fₐ h₀ hₐ κ₀ _ _ hx
  · exact fun _ hx => targetLabelKernel_off_disagreement f₀ fₐ h₀ hₐ κ₀ _ _ hx
  · exact fun g hg => target_label_free_law μ _ g hg

/-- The quantity tested by the strict frontier is the true population benefit
of the constructed target probability law. -/
theorem correctnessFieldTarget_benefit (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀]
    (hD : 0 < μ.real {x | f₀ x ≠ fₐ x}) (η : CorrectnessField X) :
    populationBenefit f₀ fₐ (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η) =
      2 * μ.real {x | f₀ x ≠ fₐ x} *
        (disagreementMean μ {x | f₀ x ≠ fₐ x} η - 1 / 2) :=
  measurable_target_benefit_reduction μ f₀ fₐ h₀ hₐ κ₀ hD η

private theorem clipped_frontier_nonempty {M beta : ℝ}
    (hMlo : -1 / 2 ≤ M) (hMhi : M ≤ 1 / 2) (hbeta : 0 ≤ beta) :
    max (-1 / 2) (M - beta) ≤ min (1 / 2) (M + beta) := by
  exact (max_le hMlo (by linarith)).trans (le_min hMhi (by linarith))

/-- The admissible measurable-field class is nonempty at every feasible margin
and nonnegative budget; the strict-frontier equivalences are not vacuous. -/
theorem measurable_frontier_class_nonempty (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀]
    (hD : 0 < μ.real {x | f₀ x ≠ fₐ x}) {M beta : ℝ}
    (hMlo : -1 / 2 ≤ M) (hMhi : M ≤ 1 / 2) (hbeta : 0 ≤ beta) :
    ∃ η : CorrectnessField X,
      IsProbabilityMeasure (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η) ∧
      |disagreementMean μ {x | f₀ x ≠ fₐ x} η - 1 / 2 - M| ≤ beta := by
  have hz : max (-1 / 2) (M - beta) ≤ M ∧ M ≤ min (1 / 2) (M + beta) :=
    ⟨max_le hMlo (by linarith), le_min hMhi (by linarith)⟩
  obtain ⟨η, hbudget, _⟩ :=
    (measurable_correctness_identified_interval μ _ hD M beta M).mpr hz
  exact ⟨η, (correctnessFieldTarget_properties (E := Unit) μ f₀ fₐ h₀ hₐ κ₀ η).1, hbudget⟩

/-- **Exact strict ADAPT frontier** over the full measurable correctness-field
class. The left side quantifies actual target population benefits. -/
theorem measurable_frontier_adapt_iff (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀]
    (hD : 0 < μ.real {x | f₀ x ≠ fₐ x}) {M beta : ℝ}
    (hMlo : -1 / 2 ≤ M) (hMhi : M ≤ 1 / 2) (hbeta : 0 ≤ beta) :
    (∀ η : CorrectnessField X,
      |disagreementMean μ {x | f₀ x ≠ fₐ x} η - 1 / 2 - M| ≤ beta →
      0 < populationBenefit f₀ fₐ (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η)) ↔
      beta < M := by
  constructor
  · intro h_all
    by_contra h_margin
    have hMbeta : M ≤ beta := le_of_not_gt h_margin
    have hz : max (-1 / 2) (M - beta) ≤ max (-1 / 2) (M - beta) ∧
        max (-1 / 2) (M - beta) ≤ min (1 / 2) (M + beta) :=
      ⟨le_rfl, clipped_frontier_nonempty hMlo hMhi hbeta⟩
    obtain ⟨η, hbudget, hη⟩ :=
      (measurable_correctness_identified_interval μ _ hD M beta _).mpr hz
    have hpos := h_all η hbudget
    rw [correctnessFieldTarget_benefit μ f₀ fₐ h₀ hₐ κ₀ hD η, hη] at hpos
    have hz_nonpos : max (-1 / 2) (M - beta) ≤ 0 :=
      max_le (by norm_num) (sub_nonpos.mpr hMbeta)
    have hnonpos := mul_nonpos_of_nonneg_of_nonpos
      (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) hD.le) hz_nonpos
    linarith
  · intro h_margin η hbudget
    have hlo := (abs_le.mp hbudget).1
    rw [correctnessFieldTarget_benefit μ f₀ fₐ h₀ hₐ κ₀ hD η]
    exact mul_pos (mul_pos (by norm_num) hD) (by linarith)

/-- **Exact strict FREEZE frontier** over actual target laws. On the boundary,
an admissible zero-benefit target prevents a strict negative certificate. -/
theorem measurable_frontier_freeze_iff (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀]
    (hD : 0 < μ.real {x | f₀ x ≠ fₐ x}) {M beta : ℝ}
    (hMlo : -1 / 2 ≤ M) (hMhi : M ≤ 1 / 2) (hbeta : 0 ≤ beta) :
    (∀ η : CorrectnessField X,
      |disagreementMean μ {x | f₀ x ≠ fₐ x} η - 1 / 2 - M| ≤ beta →
      populationBenefit f₀ fₐ (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η) < 0) ↔
      M < -beta := by
  constructor
  · intro h_all
    by_contra h_margin
    have hMbeta : -beta ≤ M := le_of_not_gt h_margin
    have hz : max (-1 / 2) (M - beta) ≤ min (1 / 2) (M + beta) ∧
        min (1 / 2) (M + beta) ≤ min (1 / 2) (M + beta) :=
      ⟨clipped_frontier_nonempty hMlo hMhi hbeta, le_rfl⟩
    obtain ⟨η, hbudget, hη⟩ :=
      (measurable_correctness_identified_interval μ _ hD M beta _).mpr hz
    have hneg := h_all η hbudget
    rw [correctnessFieldTarget_benefit μ f₀ fₐ h₀ hₐ κ₀ hD η, hη] at hneg
    have hz_nonneg : 0 ≤ min (1 / 2) (M + beta) := le_min (by norm_num) (by linarith)
    have hnonneg := mul_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) hD.le) hz_nonneg
    linarith
  · intro h_margin η hbudget
    have hhi := (abs_le.mp hbudget).2
    rw [correctnessFieldTarget_benefit μ f₀ fₐ h₀ hₐ κ₀ hD η]
    exact mul_neg_of_pos_of_neg (mul_pos (by norm_num) hD) (by linarith)

/-- Every point in the closed abstention band is realized by a genuine
zero-benefit target with the original input/evidence laws and unchanged labels
off disagreement. This includes both strict-frontier boundaries. -/
theorem measurable_closed_band_zero_target (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀]
    (hD : 0 < μ.real {x | f₀ x ≠ fₐ x}) {M beta : ℝ} (hband : |M| ≤ beta) :
    ∃ η : CorrectnessField X,
      IsProbabilityMeasure (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η) ∧
      |disagreementMean μ {x | f₀ x ≠ fₐ x} η - 1 / 2 - M| ≤ beta ∧
      populationBenefit f₀ fₐ (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η) = 0 ∧
      (∀ x, f₀ x = fₐ x →
        targetLabelKernel f₀ fₐ h₀ hₐ κ₀ η.value η.measurable_value x = κ₀ x) ∧
      (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η).fst = μ ∧
      (∀ g : X → E, Measurable g →
        (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ η).map (fun xy => g xy.1) = μ.map g) := by
  obtain ⟨hbandlo, hbandhi⟩ := abs_le.mp hband
  have hz : max (-1 / 2) (M - beta) ≤ (0 : ℝ) ∧ 0 ≤ min (1 / 2) (M + beta) :=
    ⟨max_le (by norm_num) (by linarith), le_min (by norm_num) (by linarith)⟩
  obtain ⟨η, hbudget, hη⟩ :=
    (measurable_correctness_identified_interval μ _ hD M beta 0).mpr hz
  obtain ⟨hprob, hfst, _, hoff, hevidence⟩ :=
    correctnessFieldTarget_properties (E := E) μ f₀ fₐ h₀ hₐ κ₀ η
  refine ⟨η, hprob, hbudget, ?_, hoff, hfst, hevidence⟩
  rw [correctnessFieldTarget_benefit μ f₀ fₐ h₀ hₐ κ₀ hD η, hη, mul_zero]

/-- In the open band there are actual evidence-identical target probability
laws with opposite strict benefits. Their correctness fields attain the clipped
interval endpoints, so all probability constraints remain valid for large budgets. -/
theorem measurable_open_band_opposite_targets (μ : Measure X) [IsProbabilityMeasure μ]
    (f₀ fₐ : X → Y) (h₀ : Measurable f₀) (hₐ : Measurable fₐ)
    (κ₀ : Kernel X Y) [IsMarkovKernel κ₀]
    (hD : 0 < μ.real {x | f₀ x ≠ fₐ x}) {M beta : ℝ} (hband : |M| < beta) :
    ∃ ηPos ηNeg : CorrectnessField X,
      IsProbabilityMeasure (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ ηPos) ∧
      IsProbabilityMeasure (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ ηNeg) ∧
      |disagreementMean μ {x | f₀ x ≠ fₐ x} ηPos - 1 / 2 - M| ≤ beta ∧
      |disagreementMean μ {x | f₀ x ≠ fₐ x} ηNeg - 1 / 2 - M| ≤ beta ∧
      0 < populationBenefit f₀ fₐ (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ ηPos) ∧
      populationBenefit f₀ fₐ (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ ηNeg) < 0 ∧
      (∀ x, f₀ x = fₐ x →
        targetLabelKernel f₀ fₐ h₀ hₐ κ₀ ηPos.value ηPos.measurable_value x = κ₀ x) ∧
      (∀ x, f₀ x = fₐ x →
        targetLabelKernel f₀ fₐ h₀ hₐ κ₀ ηNeg.value ηNeg.measurable_value x = κ₀ x) ∧
      (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ ηPos).fst = μ ∧
      (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ ηNeg).fst = μ ∧
      (∀ g : X → E, Measurable g →
        (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ ηPos).map (fun xy => g xy.1) =
        (correctnessFieldTarget μ f₀ fₐ h₀ hₐ κ₀ ηNeg).map (fun xy => g xy.1)) := by
  obtain ⟨hbandlo, hbandhi⟩ := abs_lt.mp hband
  have hzNeg : max (-1 / 2) (M - beta) < 0 := max_lt (by norm_num) (by linarith)
  have hzPos : 0 < min (1 / 2) (M + beta) := lt_min (by norm_num) (by linarith)
  have hinterval : max (-1 / 2) (M - beta) ≤ min (1 / 2) (M + beta) :=
    hzNeg.le.trans hzPos.le
  obtain ⟨ηPos, hbudgetPos, hηPos⟩ :=
    (measurable_correctness_identified_interval μ _ hD M beta _).mpr ⟨hinterval, le_rfl⟩
  obtain ⟨ηNeg, hbudgetNeg, hηNeg⟩ :=
    (measurable_correctness_identified_interval μ _ hD M beta _).mpr ⟨le_rfl, hinterval⟩
  obtain ⟨hprobPos, hfstPos, _, hoffPos, hevidencePos⟩ :=
    correctnessFieldTarget_properties (E := E) μ f₀ fₐ h₀ hₐ κ₀ ηPos
  obtain ⟨hprobNeg, hfstNeg, _, hoffNeg, hevidenceNeg⟩ :=
    correctnessFieldTarget_properties (E := E) μ f₀ fₐ h₀ hₐ κ₀ ηNeg
  refine ⟨ηPos, ηNeg, hprobPos, hprobNeg, hbudgetPos, hbudgetNeg, ?_, ?_,
    hoffPos, hoffNeg, hfstPos, hfstNeg, ?_⟩
  · rw [correctnessFieldTarget_benefit μ f₀ fₐ h₀ hₐ κ₀ hD ηPos, hηPos]
    exact mul_pos (mul_pos (by norm_num) hD) hzPos
  · rw [correctnessFieldTarget_benefit μ f₀ fₐ h₀ hₐ κ₀ hD ηNeg, hηNeg]
    exact mul_neg_of_pos_of_neg (mul_pos (by norm_num) hD) hzNeg
  · exact fun g hg => (hevidencePos g hg).trans (hevidenceNeg g hg).symm

end KBound
