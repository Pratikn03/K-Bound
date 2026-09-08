import KBound.Probability.InformationRefinement

/-! Regression boundaries: zero disagreement needs no positive mass; refinement
uses a joint law, not paired marginal equalities; real radii require the refined
fibre nonempty and the coarse residual range bounded; deterministic extra
information must be the same measurable recoding in every compatible world. -/

open Classical MeasureTheory ProbabilityTheory Set KBound KBound.InformationRefinement
open scoped ENNReal ProbabilityTheory

section ZeroDisagreement

variable {X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y]
  (P : Measure (X × Y)) [IsFiniteMeasure P] (f0 fa : X → Y)
  (hD : P.fst.real {x | f0 x ≠ fa x} = 0)

-- Would fail if the null case were discarded by a positive-disagreement premise.
example : (∫ xy, (if xy.2 = f0 xy.1 then (0 : ℝ) else 1) ∂P) =
    ∫ xy, (if xy.2 = fa xy.1 then (0 : ℝ) else 1) ∂P := by
  classical
  exact zero_disagreement_risks_eq P f0 fa hD

example : populationBenefit f0 fa P = 0 := by
  exact zero_disagreement_populationBenefit P f0 fa hD

end ZeroDisagreement

-- A strict fibre inclusion must give the correct (non-increasing) orientation.
example {World : Type*} (F G : Set World) (gamma : World → ℝ)
    (hFG : F ⊆ G) (hF : F.Nonempty)
    (hbounded : BddAbove (range fun p : G => |gamma p.1|)) :
    fibreRadius F gamma ≤ fibreRadius G gamma := by
  exact fibreRadius_mono F G gamma hFG hF hbounded

section Observation

variable {Ω Z E : Type*} [MeasurableSpace Ω] [MeasurableSpace Z] [MeasurableSpace E]
  (C : Set (ProbabilityMeasure Ω)) (W : Ω → Z) (H : Ω → E)
  (hW : Measurable W) (hH : Measurable H) (nu : Measure (Z × E))

-- Equality of the entire joint observation law yields equality after projection.
example : {P | P ∈ C ∧ (P : Measure Ω).map (fun ω => (W ω, H ω)) = nu} ⊆
    {P | P ∈ C ∧ (P : Measure Ω).map W = nu.fst} := by
  exact observable_fibre_refinement C W H hW hH nu

example (gamma : ProbabilityMeasure Ω → ℝ)
    (hne : {P | P ∈ C ∧ (P : Measure Ω).map (fun ω => (W ω, H ω)) = nu}.Nonempty)
    (hb : BddAbove (range fun p : {P | P ∈ C ∧ (P : Measure Ω).map W = nu.fst} =>
      |gamma p.1|)) :
    fibreRadius {P | P ∈ C ∧ (P : Measure Ω).map (fun ω => (W ω, H ω)) = nu} gamma ≤
      fibreRadius {P | P ∈ C ∧ (P : Measure Ω).map W = nu.fst} gamma := by
  exact observable_fibreRadius_mono C W H hW hH nu gamma hne hb

-- A realized compatible joint law is required; arbitrary incompatible nu is not enough.
example (g : Z → E) (hg : Measurable g) (hdet : ∀ ω, H ω = g (W ω))
    (hcompatible : ∃ Q ∈ C, (Q : Measure Ω).map (fun ω => (W ω, H ω)) = nu) :
    {P | P ∈ C ∧ (P : Measure Ω).map (fun ω => (W ω, H ω)) = nu} =
      {P | P ∈ C ∧ (P : Measure Ω).map W = nu.fst} := by
  exact deterministic_observable_fibre_eq C W H hW g hg hdet nu hcompatible

end Observation

-- The monotonicity theorem accepts the existing actual conditional-kernel residual.
example {X Z E : Type*} [MeasurableSpace X] [MeasurableSpace Z] [MeasurableSpace E]
    (C : Set (ProbabilityMeasure (X × Bool))) (W : X × Bool → Z) (H : X × Bool → E)
    (hW : Measurable W) (hH : Measurable H) (nu : Measure (Z × E))
    (f0 fa : X → Bool) (ha : Measurable fa) (score : CorrectnessField X)
    (hne : {P | P ∈ C ∧ (P : Measure (X × Bool)).map (fun ω => (W ω, H ω)) = nu}.Nonempty)
    (hb : BddAbove (range fun p : {P | P ∈ C ∧
        (P : Measure (X × Bool)).map W = nu.fst} =>
      |JointKernelScore.scoreResidual (p.1 : Measure (X × Bool)) f0 fa ha score|)) :
    fibreRadius {P | P ∈ C ∧ (P : Measure (X × Bool)).map (fun ω => (W ω, H ω)) = nu}
        (fun P => JointKernelScore.scoreResidual (P : Measure (X × Bool)) f0 fa ha score) ≤
      fibreRadius {P | P ∈ C ∧ (P : Measure (X × Bool)).map W = nu.fst}
        (fun P => JointKernelScore.scoreResidual (P : Measure (X × Bool)) f0 fa ha score) := by
  exact observable_fibreRadius_mono C W H hW hH nu _ hne hb
