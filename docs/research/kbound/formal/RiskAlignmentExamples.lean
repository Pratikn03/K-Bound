import KBound

open MeasureTheory ProbabilityTheory KBound KBound.JointKernelScore
open KBound.ActualWorldFrontier KBound.ActualWorldEvidence
open scoped ENNReal ProbabilityTheory

#check KBound.RiskAlignment.aligned_of_nonnegative
#check KBound.RiskAlignment.not_strict_of_zero
#check KBound.RiskAlignment.actual_positive_boundary_nonnegative

-- "Weaker" includes this direction as well as a counterexample to its converse.
example {World Evidence : Type*} (C : World → Prop) (e : World → Evidence)
    (b : World → ℝ) (h : KBound.RiskAlignment.StrictDirection C b) :
    KBound.RiskAlignment.Aligned C e b :=
  KBound.RiskAlignment.strict_implies_aligned C e b h

-- Definition-level actual-class witness, not a free pair of scalar benefits.
example {X Z : Type*} [MeasurableSpace X] [MeasurableSpace Z]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 < beta)
    (hm : scoreMargin μ {x | f0 x ≠ fa x} score = beta) :
    KBound.RiskAlignment.Aligned (admissible μ f0 fa ha score beta)
        (augmentedLaw observation f0 fa score)
        (fun P => populationBenefit f0 fa (P : Measure (X × Bool))) ∧
      ¬ KBound.RiskAlignment.StrictDirection (admissible μ f0 fa ha score beta)
        (fun P => populationBenefit f0 fa (P : Measure (X × Bool))) ∧
      ∃ Pzero Ppos : ProbabilityMeasure (X × Bool),
        admissible μ f0 fa ha score beta Pzero ∧ admissible μ f0 fa ha score beta Ppos ∧
        populationBenefit f0 fa (Pzero : Measure (X × Bool)) = 0 ∧
        0 < populationBenefit f0 fa (Ppos : Measure (X × Bool)) ∧
        augmentedLaw observation f0 fa score Pzero = augmentedLaw observation f0 fa score Ppos :=
  KBound.RiskAlignment.actual_boundary_risk_aligned_not_strict μ f0 fa h0 ha κ0
    score observation hD beta hb hm

-- Constant evidence with benefits0 and1 is risk aligned, yet neither strict
-- direction is universally valid. This exercises the literal definitions.
example : KBound.RiskAlignment.Aligned (fun _ : Bool => True) (fun _ => ())
      (fun b => if b then (1 : ℝ) else 0) ∧
    ¬ KBound.RiskAlignment.StrictDirection (fun _ : Bool => True)
      (fun b => if b then (1 : ℝ) else 0) := by
  constructor
  · apply KBound.RiskAlignment.aligned_of_nonnegative
    intro b _
    split_ifs <;> norm_num
  · exact KBound.RiskAlignment.not_strict_of_zero (fun _ : Bool => True)
      (fun b => if b then (1 : ℝ) else 0) false trivial (by norm_num)

-- The alignment predicate must reject actual opposite nonzero scalar signs.
example : ¬ KBound.RiskAlignment.Aligned (fun _ : Bool => True) (fun _ => ())
    (fun b => if b then (1 : ℝ) else -1) := by
  intro h
  exact h false true trivial trivial rfl (by norm_num)

-- The actual positive-boundary hypotheses are jointly realizable: fixed score
--3/4, beta1/4 and full disagreement on a one-input probability space.
example : ∃ Pzero Ppos : ProbabilityMeasure (Unit × Bool),
    admissible (Measure.dirac ()) (fun _ => false) (fun _ => true) measurable_const
      (CorrectnessField.constant (3 / 4) (by norm_num) (by norm_num)) (1 / 4) Pzero ∧
    admissible (Measure.dirac ()) (fun _ => false) (fun _ => true) measurable_const
      (CorrectnessField.constant (3 / 4) (by norm_num) (by norm_num)) (1 / 4) Ppos ∧
    populationBenefit (fun _ => false) (fun _ => true) (Pzero : Measure (Unit × Bool)) = 0 ∧
    0 < populationBenefit (fun _ => false) (fun _ => true) (Ppos : Measure (Unit × Bool)) := by
  let observation : ProbabilityMeasure Unit → ProbabilityMeasure Unit :=
    fun _ => ⟨Measure.dirac (), inferInstance⟩
  have h := KBound.RiskAlignment.actual_boundary_risk_aligned_not_strict
    (Measure.dirac ()) (fun _ => false) (fun _ => true) measurable_const measurable_const
    (Kernel.const Unit (Measure.dirac false))
    (CorrectnessField.constant (3 / 4) (by norm_num) (by norm_num)) observation
    (by norm_num [Measure.real]) (1 / 4) (by norm_num)
    (by norm_num [scoreMargin, disagreementMean, CorrectnessField.constant, Measure.real])
  obtain ⟨_, _, Pzero, Ppos, hz, hp, bz, bp, _⟩ := h
  exact ⟨Pzero, Ppos, hz, hp, bz, bp⟩
