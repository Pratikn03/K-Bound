import KBound

-- RED: these assembled actual-world declarations are missing from the current root.
#check KBound.ActualWorldFrontier.actual_frontier_adapt_iff
#check KBound.ActualWorldFrontier.actual_frontier_freeze_iff
#check KBound.ActualWorldFrontier.actual_identified_interval

open MeasureTheory ProbabilityTheory Set KBound KBound.JointKernelScore
open KBound.ActualWorldFrontier
open scoped ENNReal ProbabilityTheory

-- The quantified world class is every probability law, not only fieldWorld.
example {X : Type*} [MeasurableSpace X] (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta) :
    (∀ P : ProbabilityMeasure (X × Bool), admissible μ f0 fa ha score beta P →
      0 < populationBenefit f0 fa (P : Measure (X × Bool))) ↔
      beta < scoreMargin μ {x | f0 x ≠ fa x} score :=
  actual_frontier_adapt_iff μ f0 fa h0 ha κ0 score hD beta hb

local notation "mu" => (Measure.dirac () : Measure Unit)
local notation "f0" => (fun _ : Unit => false)
local notation "fa" => (fun _ : Unit => true)
local notation "kernel0" => (Kernel.const Unit (Measure.dirac false))

-- Strictly outside the positive band: actual universal positivity.
example : ∀ P : ProbabilityMeasure (Unit × Bool),
    admissible mu f0 fa measurable_const
      (CorrectnessField.constant (3 / 4) (by norm_num) (by norm_num)) (1 / 8) P →
      0 < populationBenefit f0 fa (P : Measure (Unit × Bool)) := by
  apply (actual_frontier_adapt_iff mu f0 fa measurable_const measurable_const kernel0
    (CorrectnessField.constant (3 / 4) (by norm_num) (by norm_num))
    (by norm_num [Measure.real]) (1 / 8) (by norm_num)).mpr
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]

-- Positive boundary: zero remains admissible, precluding strict positivity.
example : ∃ P : ProbabilityMeasure (Unit × Bool),
    admissible mu f0 fa measurable_const
      (CorrectnessField.constant (3 / 4) (by norm_num) (by norm_num)) (1 / 4) P ∧
      populationBenefit f0 fa (P : Measure (Unit × Bool)) = 0 := by
  apply actual_closed_band_zero_target mu f0 fa measurable_const measurable_const kernel0
    (CorrectnessField.constant (3 / 4) (by norm_num) (by norm_num))
    (by norm_num [Measure.real]) (1 / 4)
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]

-- Negative boundary is also included.
example : ∃ P : ProbabilityMeasure (Unit × Bool),
    admissible mu f0 fa measurable_const
      (CorrectnessField.constant (1 / 4) (by norm_num) (by norm_num)) (1 / 4) P ∧
      populationBenefit f0 fa (P : Measure (Unit × Bool)) = 0 := by
  apply actual_closed_band_zero_target mu f0 fa measurable_const measurable_const kernel0
    (CorrectnessField.constant (1 / 4) (by norm_num) (by norm_num))
    (by norm_num [Measure.real]) (1 / 4)
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]

-- Large budgets never admit impossible benefit beyond the probability clip.
example : ¬ ∃ P : ProbabilityMeasure (Unit × Bool),
    admissible mu f0 fa measurable_const
      (CorrectnessField.constant (1 / 2) (by norm_num) (by norm_num)) 100 P ∧
      populationBenefit f0 fa (P : Measure (Unit × Bool)) =
        2 * (mu).real {x | f0 x ≠ fa x} * (3 / 5) := by
  rw [actual_identified_interval mu f0 fa measurable_const measurable_const kernel0
    (CorrectnessField.constant (1 / 2) (by norm_num) (by norm_num))
    (by norm_num [Measure.real]) 100 (3 / 5)]
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]

-- Zero budget/zero margin is universally zero, not an empty-class shortcut.
example (P : ProbabilityMeasure (Unit × Bool))
    (hP : admissible mu f0 fa measurable_const
      (CorrectnessField.constant (1 / 2) (by norm_num) (by norm_num)) 0 P) :
    populationBenefit f0 fa (P : Measure (Unit × Bool)) = 0 := by
  apply actual_zero_margin_budget mu f0 fa measurable_const measurable_const
    (CorrectnessField.constant (1 / 2) (by norm_num) (by norm_num))
    (by norm_num [Measure.real]) ?_ P hP
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]

-- A score of one does not constrain correctness when the budget is large.
example : ∃ P : ProbabilityMeasure (Unit × Bool),
    admissible mu f0 fa measurable_const
      (CorrectnessField.constant 1 (by norm_num) (by norm_num)) 100 P ∧
      populationBenefit f0 fa (P : Measure (Unit × Bool)) =
        2 * (mu).real {x | f0 x ≠ fa x} * (-1 / 2) := by
  rw [actual_identified_interval mu f0 fa measurable_const measurable_const kernel0
    (CorrectnessField.constant 1 (by norm_num) (by norm_num))
    (by norm_num [Measure.real]) 100 (-1 / 2)]
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]

-- Conversely score zero permits the positive endpoint at a large budget.
example : ∃ P : ProbabilityMeasure (Unit × Bool),
    admissible mu f0 fa measurable_const
      (CorrectnessField.constant 0 (by norm_num) (by norm_num)) 100 P ∧
      populationBenefit f0 fa (P : Measure (Unit × Bool)) =
        2 * (mu).real {x | f0 x ≠ fa x} * (1 / 2) := by
  rw [actual_identified_interval mu f0 fa measurable_const measurable_const kernel0
    (CorrectnessField.constant 0 (by norm_num) (by norm_num))
    (by norm_num [Measure.real]) 100 (1 / 2)]
  norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real]
