import KBound.Probability.JointKernelScore
import KBound.Probability.MeasureFrontier
import KBound.Frontier
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure

/-!
# Score-defined frontier over actual arbitrary binary joint laws

The quantified class consists of all probability measures with the fixed input
marginal and an actual conditional-kernel score residual bounded by beta. A
correctness field is used to construct witnesses, not to restrict the universal
class. The off-disagreement kernel can be any fixed Markov kernel in those
witnesses. Sampling and randomized decision composition are separate.
-/

namespace KBound.ActualWorldFrontier

open MeasureTheory ProbabilityTheory Set JointKernelScore
open scoped ENNReal ProbabilityTheory

variable {X : Type*} [MeasurableSpace X]

noncomputable def fieldWorld (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (η : CorrectnessField X) :
    ProbabilityMeasure (X × Bool) :=
  ⟨correctnessFieldTarget μ f0 fa h0 ha κ0 η,
    (correctnessFieldTarget_properties (E := Unit) μ f0 fa h0 ha κ0 η).1⟩

theorem fieldWorld_fst (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (η : CorrectnessField X) :
    (fieldWorld μ f0 fa h0 ha κ0 η : Measure (X × Bool)).fst = μ :=
  (correctnessFieldTarget_properties (E := Unit) μ f0 fa h0 ha κ0 η).2.1

/-- The constructed world's actual disintegrated residual has the prescribed
conditional mean. No pointwise uniqueness of conditional versions is assumed. -/
theorem fieldWorld_residual (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score η : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) :
    scoreResidual (fieldWorld μ f0 fa h0 ha κ0 η : Measure (X × Bool)) f0 fa ha score =
      disagreementMean μ {x | f0 x ≠ fa x} η - 1 / 2 -
        scoreMargin μ {x | f0 x ≠ fa x} score := by
  have hf := fieldWorld_fst μ f0 fa h0 ha κ0 η
  have h := joint_score_benefit (fieldWorld μ f0 fa h0 ha κ0 η : Measure (X × Bool))
    f0 fa h0 ha score (by rw [hf]; exact hD)
  rw [hf] at h
  change populationBenefit f0 fa (correctnessFieldTarget μ f0 fa h0 ha κ0 η) = _ at h
  rw [correctnessFieldTarget_benefit μ f0 fa h0 ha κ0 hD η] at h
  have he := (mul_left_cancel₀ (mul_ne_zero (by norm_num : (2 : ℝ) ≠ 0) (ne_of_gt hD))) h
  linarith

/-- The full class uses genuine joint probability laws and the actual residual. -/
def admissible (μ : Measure X) (f0 fa : X → Bool) (ha : Measurable fa)
    (score : CorrectnessField X) (beta : ℝ) (P : ProbabilityMeasure (X × Bool)) : Prop :=
  (P : Measure (X × Bool)).fst = μ ∧
    |scoreResidual (P : Measure (X × Bool)) f0 fa ha score| ≤ beta

/-- Strict ADAPT soundness iff, universally over actual joint laws. -/
theorem actual_frontier_adapt_iff (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta) :
    (∀ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P →
      0 < populationBenefit f0 fa (P : Measure (X × Bool))) ↔
      beta < scoreMargin μ {x | f0 x ≠ fa x} score := by
  have hM := scoreMargin_bounds μ _ score hD
  constructor
  · intro hall
    apply (measurable_frontier_adapt_iff μ f0 fa h0 ha κ0 hD hM.1 hM.2 hb).mp
    intro η hη
    exact hall (fieldWorld μ f0 fa h0 ha κ0 η)
      ⟨fieldWorld_fst μ f0 fa h0 ha κ0 η, by
        rw [fieldWorld_residual μ f0 fa h0 ha κ0 score η hD]
        exact hη⟩
  · intro hm P hP
    rcases hP with ⟨hf, hg⟩
    rw [joint_score_benefit (P : Measure (X × Bool)) f0 fa h0 ha score
      (by rw [hf]; exact hD), hf]
    exact mul_pos (mul_pos (by norm_num) hD) (by linarith [(abs_le.mp hg).1])

/-- Strict FREEZE soundness iff over the same full class. -/
theorem actual_frontier_freeze_iff (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta) :
    (∀ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P →
      populationBenefit f0 fa (P : Measure (X × Bool)) < 0) ↔
      scoreMargin μ {x | f0 x ≠ fa x} score < -beta := by
  have hM := scoreMargin_bounds μ _ score hD
  constructor
  · intro hall
    apply (measurable_frontier_freeze_iff μ f0 fa h0 ha κ0 hD hM.1 hM.2 hb).mp
    intro η hη
    exact hall (fieldWorld μ f0 fa h0 ha κ0 η)
      ⟨fieldWorld_fst μ f0 fa h0 ha κ0 η, by
        rw [fieldWorld_residual μ f0 fa h0 ha κ0 score η hD]
        exact hη⟩
  · intro hm P hP
    rcases hP with ⟨hf, hg⟩
    rw [joint_score_benefit (P : Measure (X × Bool)) f0 fa h0 ha score
      (by rw [hf]; exact hD), hf]
    exact mul_neg_of_pos_of_neg (mul_pos (by norm_num) hD) (by linarith [(abs_le.mp hg).2])

/-- Exact clipped interval for normalized actual population benefit. -/
theorem actual_identified_interval (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta z : ℝ) :
    (∃ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P ∧
      populationBenefit f0 fa (P : Measure (X × Bool)) = 2 * μ.real {x | f0 x ≠ fa x} * z) ↔
      max (-1 / 2) (scoreMargin μ {x | f0 x ≠ fa x} score - beta) ≤ z ∧
      z ≤ min (1 / 2) (scoreMargin μ {x | f0 x ≠ fa x} score + beta) := by
  rw [← measurable_correctness_identified_interval μ _ hD _ beta z]
  constructor
  · rintro ⟨P, ⟨hf, hg⟩, hp⟩
    let η := jointCorrectness (P : Measure (X × Bool)) fa ha
    have hi := joint_score_identity (P : Measure (X × Bool)) f0 fa h0 ha score
    rw [joint_conditional_accuracy (P : Measure (X × Bool)) f0 fa h0 ha, hf] at hi
    have hb := joint_score_benefit (P : Measure (X × Bool)) f0 fa h0 ha score
      (by rw [hf]; exact hD)
    rw [hf, hp] at hb
    have he := (mul_left_cancel₀ (mul_ne_zero (by norm_num : (2 : ℝ) ≠ 0) (ne_of_gt hD))) hb
    refine ⟨η, ?_, ?_⟩
    · have hr : disagreementMean μ {x | f0 x ≠ fa x} η - 1 / 2 -
          scoreMargin μ {x | f0 x ≠ fa x} score =
          scoreResidual (P : Measure (X × Bool)) f0 fa ha score := by
        change _ at hi
        linarith
      rwa [hr]
    · change _ at hi
      linarith
  · rintro ⟨η, hη, hz⟩
    refine ⟨fieldWorld μ f0 fa h0 ha κ0 η, ⟨fieldWorld_fst μ f0 fa h0 ha κ0 η, ?_⟩, ?_⟩
    · rw [fieldWorld_residual μ f0 fa h0 ha κ0 score η hD]
      exact hη
    · change populationBenefit f0 fa (correctnessFieldTarget μ f0 fa h0 ha κ0 η) = _
      rw [correctnessFieldTarget_benefit μ f0 fa h0 ha κ0 hD η, hz]

/-- The actual class is nonempty for every nonnegative budget. -/
theorem actual_class_nonempty (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta) :
    ∃ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P := by
  have hm := scoreMargin_bounds μ _ score hD
  obtain ⟨P, hP, _⟩ := (actual_identified_interval μ f0 fa h0 ha κ0 score hD beta
    (scoreMargin μ {x | f0 x ≠ fa x} score)).mpr
      ⟨max_le hm.1 (by linarith), le_min hm.2 (by linarith)⟩
  exact ⟨P, hP⟩

/-- Zero benefit is attained throughout the closed band, including both endpoints. -/
theorem actual_closed_band_zero_target (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ)
    (hband : |scoreMargin μ {x | f0 x ≠ fa x} score| ≤ beta) :
    ∃ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P ∧
      populationBenefit f0 fa (P : Measure (X × Bool)) = 0 := by
  obtain ⟨P, hP, hp⟩ := (actual_identified_interval μ f0 fa h0 ha κ0 score hD beta 0).mpr
    ⟨max_le (by norm_num) (by linarith [(abs_le.mp hband).2]),
      le_min (by norm_num) (by linarith [(abs_le.mp hband).1])⟩
  exact ⟨P, hP, by simpa using hp⟩

/-- At zero margin and zero budget, every admissible actual law has zero benefit. -/
theorem actual_zero_margin_budget (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (score : CorrectnessField X) (hD : 0 < μ.real {x | f0 x ≠ fa x})
    (hm : scoreMargin μ {x | f0 x ≠ fa x} score = 0)
    (P : ProbabilityMeasure (X × Bool)) (hP : admissible μ f0 fa ha score 0 P) :
    populationBenefit f0 fa (P : Measure (X × Bool)) = 0 := by
  obtain ⟨hf, hg⟩ := hP
  have hz := abs_eq_zero.mp (le_antisymm hg (abs_nonneg _))
  rw [joint_score_benefit (P : Measure (X × Bool)) f0 fa h0 ha score
    (by rw [hf]; exact hD), hf, hm, hz]
  ring

/-- A uniform strict direction exists exactly outside the closed band. -/
theorem actual_strict_direction_iff (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta) :
    ((∀ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P →
        0 < populationBenefit f0 fa (P : Measure (X × Bool))) ∨
      (∀ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P →
        populationBenefit f0 fa (P : Measure (X × Bool)) < 0)) ↔
      beta < |scoreMargin μ {x | f0 x ≠ fa x} score| := by
  rw [actual_frontier_adapt_iff μ f0 fa h0 ha κ0 score hD beta hb,
    actual_frontier_freeze_iff μ f0 fa h0 ha κ0 score hD beta hb, lt_abs]
  apply or_congr Iff.rfl
  constructor <;> intro h <;> linarith

/-- The printed rule commits precisely in each uniformly sound strict direction
and abstains precisely when neither is sound: definition-level maximality. -/
theorem actual_pointwise_maximal_rule (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta) :
    (frontierDecision (scoreMargin μ {x | f0 x ≠ fa x} score) beta = Decision.adapt ↔
      ∀ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P →
        0 < populationBenefit f0 fa (P : Measure (X × Bool))) ∧
    (frontierDecision (scoreMargin μ {x | f0 x ≠ fa x} score) beta = Decision.freeze ↔
      ∀ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P →
        populationBenefit f0 fa (P : Measure (X × Bool)) < 0) ∧
    (frontierDecision (scoreMargin μ {x | f0 x ≠ fa x} score) beta = Decision.abstain ↔
      |scoreMargin μ {x | f0 x ≠ fa x} score| ≤ beta) := by
  rw [actual_frontier_adapt_iff μ f0 fa h0 ha κ0 score hD beta hb,
    actual_frontier_freeze_iff μ f0 fa h0 ha κ0 score hD beta hb]
  by_cases hp : beta < scoreMargin μ {x | f0 x ≠ fa x} score
  · have hn : ¬ scoreMargin μ {x | f0 x ≠ fa x} score < -beta := by linarith
    have hnb : ¬ |scoreMargin μ {x | f0 x ≠ fa x} score| ≤ beta := by
      intro h
      linarith [(abs_le.mp h).2]
    simp [frontierDecision, hp, hn, hnb]
  · by_cases hn : scoreMargin μ {x | f0 x ≠ fa x} score < -beta
    · have hnb : ¬ |scoreMargin μ {x | f0 x ≠ fa x} score| ≤ beta := by
        intro h
        linarith [(abs_le.mp h).1]
      simp [frontierDecision, hp, hn, hnb]
    · have hab : |scoreMargin μ {x | f0 x ≠ fa x} score| ≤ beta :=
        abs_le.mpr ⟨le_of_not_gt hn, le_of_not_gt hp⟩
      simp [frontierDecision, hp, hn, hab]

end KBound.ActualWorldFrontier

#print axioms KBound.ActualWorldFrontier.fieldWorld_fst
#print axioms KBound.ActualWorldFrontier.fieldWorld_residual
#print axioms KBound.ActualWorldFrontier.actual_frontier_adapt_iff
#print axioms KBound.ActualWorldFrontier.actual_frontier_freeze_iff
#print axioms KBound.ActualWorldFrontier.actual_identified_interval
#print axioms KBound.ActualWorldFrontier.actual_class_nonempty
#print axioms KBound.ActualWorldFrontier.actual_closed_band_zero_target
#print axioms KBound.ActualWorldFrontier.actual_zero_margin_budget
#print axioms KBound.ActualWorldFrontier.actual_strict_direction_iff
#print axioms KBound.ActualWorldFrontier.actual_pointwise_maximal_rule
