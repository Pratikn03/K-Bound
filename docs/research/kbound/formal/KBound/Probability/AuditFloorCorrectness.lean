import KBound.Probability.AuditFloor
import KBound.Probability.MeasureFrontier

/-!
# The radius of the full feasible measurable correctness-field class

This is a separate application of the supremum radius definition. The radius
equals the declared population budget for feasible margins and `0 ≤ beta ≤ 1/2`.
Positive disagreement mass permits the existing conditional-correctness
construction. The target-law corollary retains measurable predictors and a
Markov off-disagreement kernel and preserves all measurable input evidence laws.
No richness of a restricted deployment subclass or estimation of beta follows.
-/

namespace KBound

open MeasureTheory ProbabilityTheory Set
open scoped ENNReal ProbabilityTheory

variable {X : Type*} [MeasurableSpace X]

noncomputable def correctnessResidual (mu : Measure X) (D : Set X) (M : ℝ)
    (eta : CorrectnessField X) : ℝ := disagreementMean mu D eta - 1 / 2 - M

def correctnessFibre (mu : Measure X) (D : Set X) (M beta : ℝ) : Set (CorrectnessField X) :=
  {eta | |correctnessResidual mu D M eta| ≤ beta}

/-- One feasible sign of the residual attains the absolute budget, including
the endpoints. Both signs need not be feasible. -/
theorem correctness_fibre_abs_residual_attained (mu : Measure X) [IsProbabilityMeasure mu]
    (D : Set X) (hD : 0 < mu.real D) (M beta : ℝ)
    (hMlo : -1 / 2 ≤ M) (hMhi : M ≤ 1 / 2)
    (hbeta0 : 0 ≤ beta) (hbeta1 : beta ≤ 1 / 2) :
    ∃ eta ∈ correctnessFibre mu D M beta, |correctnessResidual mu D M eta| = beta := by
  by_cases hM : M ≤ 0
  · have hz : max (-1 / 2) (M - beta) ≤ M + beta ∧
        M + beta ≤ min (1 / 2) (M + beta) := by
      constructor
      · exact max_le (by linarith) (by linarith)
      · exact le_min (by linarith) le_rfl
    obtain ⟨eta, hbudget, heta⟩ :=
      (measurable_correctness_identified_interval mu D hD M beta (M + beta)).mpr hz
    refine ⟨eta, hbudget, ?_⟩
    simp only [correctnessResidual, heta, add_sub_cancel_left, abs_of_nonneg hbeta0]
  · have hz : max (-1 / 2) (M - beta) ≤ M - beta ∧
        M - beta ≤ min (1 / 2) (M + beta) := by
      constructor
      · exact max_le (by linarith) le_rfl
      · exact le_min (by linarith) (by linarith)
    obtain ⟨eta, hbudget, heta⟩ :=
      (measurable_correctness_identified_interval mu D hD M beta (M - beta)).mpr hz
    refine ⟨eta, hbudget, ?_⟩
    unfold correctnessResidual
    rw [heta, show M - beta - M = -beta by ring, abs_neg, abs_of_nonneg hbeta0]

/-- The actual full feasible correctness-field fibre has radius beta under
the printed small-budget restriction, without finite enumeration. Predictor and
kernel assumptions are unnecessary for this conditional-mean identity itself. -/
theorem correctness_fibreRadius_eq_beta (mu : Measure X) [IsProbabilityMeasure mu]
    (D : Set X) (hD : 0 < mu.real D) (M beta : ℝ)
    (hMlo : -1 / 2 ≤ M) (hMhi : M ≤ 1 / 2)
    (hbeta0 : 0 ≤ beta) (hbeta1 : beta ≤ 1 / 2) :
    fibreRadius (correctnessFibre mu D M beta) (correctnessResidual mu D M) = beta := by
  obtain ⟨eta, hmem, heta⟩ :=
    correctness_fibre_abs_residual_attained mu D hD M beta hMlo hMhi hbeta0 hbeta1
  exact fibreRadius_eq_of_bound_and_witness _ _ beta (fun p => p.property)
    ⟨⟨eta, hmem⟩, heta⟩

/-- The extremal correctness field gives an actual joint target probability
law with fixed off-disagreement labels and every measurable label-free evidence
law unchanged. This connects the radius identity to the declared target class. -/
theorem correctness_target_fibreRadius_eq_beta
    {Y E : Type*} [MeasurableSpace Y] [MeasurableSpace E]
    [MeasurableSingletonClass Y] [MeasurableEq Y]
    (mu : Measure X) [IsProbabilityMeasure mu]
    (f0 fa : X → Y) (h0 : Measurable f0) (ha : Measurable fa)
    (k0 : Kernel X Y) [IsMarkovKernel k0]
    (hD : 0 < mu.real {x | f0 x ≠ fa x}) (M beta : ℝ)
    (hMlo : -1 / 2 ≤ M) (hMhi : M ≤ 1 / 2)
    (hbeta0 : 0 ≤ beta) (hbeta1 : beta ≤ 1 / 2) :
    fibreRadius (correctnessFibre mu {x | f0 x ≠ fa x} M beta)
      (correctnessResidual mu {x | f0 x ≠ fa x} M) = beta ∧
    ∃ eta ∈ correctnessFibre mu {x | f0 x ≠ fa x} M beta,
      |correctnessResidual mu {x | f0 x ≠ fa x} M eta| = beta ∧
      IsProbabilityMeasure (correctnessFieldTarget mu f0 fa h0 ha k0 eta) ∧
      (correctnessFieldTarget mu f0 fa h0 ha k0 eta).fst = mu ∧
      (∀ x, f0 x = fa x →
        targetLabelKernel f0 fa h0 ha k0 eta.value eta.measurable_value x = k0 x) ∧
      (∀ g : X → E, Measurable g →
        (correctnessFieldTarget mu f0 fa h0 ha k0 eta).map (fun xy => g xy.1) = mu.map g) := by
  refine ⟨correctness_fibreRadius_eq_beta mu _ hD M beta hMlo hMhi hbeta0 hbeta1, ?_⟩
  obtain ⟨eta, hmem, heta⟩ :=
    correctness_fibre_abs_residual_attained mu _ hD M beta hMlo hMhi hbeta0 hbeta1
  obtain ⟨hprob, hfst, _, hoff, hevidence⟩ :=
    correctnessFieldTarget_properties (E := E) mu f0 fa h0 ha k0 eta
  exact ⟨eta, hmem, heta, hprob, hfst, hoff, hevidence⟩

end KBound
