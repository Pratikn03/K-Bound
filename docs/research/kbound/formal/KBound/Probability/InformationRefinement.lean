import KBound.Probability.JointKernelScore
import KBound.Probability.AuditFloor
import Mathlib.MeasureTheory.Measure.ProbabilityMeasure

/-!
# Null disagreement and refinement of actual observation-law fibres

The null-disagreement result uses the actual joint measure and its input
marginal; it does not divide by the disagreement mass. Refinement below is
defined by equality of pushforward laws on the full joint observation space,
not equality of separate marginals. The real supremum is used only with the
stated nonempty/bounded premises. Deterministic recoding requires a compatible
joint law and the same measurable recoding at every sample point.
-/

namespace KBound.InformationRefinement

open Classical MeasureTheory ProbabilityTheory Set
open scoped ENNReal ProbabilityTheory

private theorem predictions_ae_of_zero_disagreement
    {X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y]
    (P : Measure (X × Y)) [IsFiniteMeasure P] (f0 fa : X → Y)
    (hD : P.fst.real {x | f0 x ≠ fa x} = 0) :
    ∀ᵐ xy ∂P, f0 xy.1 = fa xy.1 := by
  have hin : ∀ᵐ x ∂P.fst, f0 x = fa x :=
    ae_iff.mpr ((measureReal_eq_zero_iff (measure_ne_top P.fst _)).mp hD)
  exact ae_of_ae_map measurable_fst.aemeasurable hin

/-- Zero input disagreement gives equality of the actual zero-one risk
integrals. Finiteness excludes the `ENNReal.toReal infinity = 0` degeneracy.
No positive-disagreement or conditional-score premise is needed. -/
theorem zero_disagreement_risks_eq
    {X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y]
    (P : Measure (X × Y)) [IsFiniteMeasure P] (f0 fa : X → Y)
    (hD : P.fst.real {x | f0 x ≠ fa x} = 0) :
    (∫ xy, (if xy.2 = f0 xy.1 then (0 : ℝ) else 1) ∂P) =
      ∫ xy, (if xy.2 = fa xy.1 then (0 : ℝ) else 1) ∂P := by
  apply integral_congr_ae
  filter_upwards [predictions_ae_of_zero_disagreement P f0 fa hD] with xy hxy
  rw [hxy]

/-- The existing populationBenefit is zero at null input disagreement,
including laws whose classifiers differ on a null set of inputs. -/
theorem zero_disagreement_populationBenefit
    {X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y]
    (P : Measure (X × Y)) [IsFiniteMeasure P] (f0 fa : X → Y)
    (hD : P.fst.real {x | f0 x ≠ fa x} = 0) :
    populationBenefit f0 fa P = 0 := by
  have hzero : zeroOneBenefit f0 fa =ᵐ[P] (fun _ => (0 : ℝ)) := by
    filter_upwards [predictions_ae_of_zero_disagreement P f0 fa hD] with xy hxy
    simp only [zeroOneBenefit, hxy, sub_self]
  rw [populationBenefit, integral_congr_ae hzero, integral_zero]

/-- Inclusion decreases the real residual radius. Only the smaller fibre
needs an explicit witness; only the larger residual range needs a bound.
Neither a finite fibre nor attainment of either supremum is assumed. -/
theorem fibreRadius_mono {World : Type*} (F G : Set World) (gamma : World → ℝ)
    (hFG : F ⊆ G) (hF : F.Nonempty)
    (hbounded : BddAbove (range fun p : G => |gamma p.1|)) :
    fibreRadius F gamma ≤ fibreRadius G gamma := by
  apply csSup_le (fibreResidualRange_nonempty hF gamma)
  rintro _ ⟨p, rfl⟩
  exact constant_fibreRadius_valid G gamma hbounded ⟨p.1, hFG p.2⟩

/-- Projecting the actual joint observation law onto its first coordinate
gives the original observation law, hence the fibre inclusion. -/
theorem observable_fibre_refinement
    {Ω Z E : Type*} [MeasurableSpace Ω] [MeasurableSpace Z] [MeasurableSpace E]
    (C : Set (ProbabilityMeasure Ω)) (W : Ω → Z) (H : Ω → E)
    (hW : Measurable W) (hH : Measurable H) (nu : Measure (Z × E)) :
    {P | P ∈ C ∧ (P : Measure Ω).map (fun ω => (W ω, H ω)) = nu} ⊆
      {P | P ∈ C ∧ (P : Measure Ω).map W = nu.fst} := by
  intro P hP
  refine ⟨hP.1, ?_⟩
  have hp := congrArg (fun law : Measure (Z × E) => law.fst) hP.2
  change ((P : Measure Ω).map (fun ω => (W ω, H ω))).map Prod.fst = nu.fst at hp
  rw [Measure.map_map measurable_fst (hW.prodMk hH)] at hp
  exact hp

/-- Radius monotonicity for actual joint-observation refinement, specialized
to any fixed residual functional, including the actual scoreResidual. -/
theorem observable_fibreRadius_mono
    {Ω Z E : Type*} [MeasurableSpace Ω] [MeasurableSpace Z] [MeasurableSpace E]
    (C : Set (ProbabilityMeasure Ω)) (W : Ω → Z) (H : Ω → E)
    (hW : Measurable W) (hH : Measurable H) (nu : Measure (Z × E))
    (gamma : ProbabilityMeasure Ω → ℝ)
    (hne : {P | P ∈ C ∧ (P : Measure Ω).map (fun ω => (W ω, H ω)) = nu}.Nonempty)
    (hb : BddAbove (range fun p : {P | P ∈ C ∧ (P : Measure Ω).map W = nu.fst} =>
      |gamma p.1|)) :
    fibreRadius {P | P ∈ C ∧ (P : Measure Ω).map (fun ω => (W ω, H ω)) = nu} gamma ≤
      fibreRadius {P | P ∈ C ∧ (P : Measure Ω).map W = nu.fst} gamma :=
  fibreRadius_mono _ _ gamma (observable_fibre_refinement C W H hW hH nu) hne hb

/-- For a compatible joint law, adding H = g(W) leaves the fibre unchanged.
Compatibility is witnessed by an actual world in C. The augmented law is
derived by pushforward composition, not postulated to depend only on W's law. -/
theorem deterministic_observable_fibre_eq
    {Ω Z E : Type*} [MeasurableSpace Ω] [MeasurableSpace Z] [MeasurableSpace E]
    (C : Set (ProbabilityMeasure Ω)) (W : Ω → Z) (H : Ω → E)
    (hW : Measurable W) (g : Z → E) (hg : Measurable g)
    (hdet : ∀ ω, H ω = g (W ω)) (nu : Measure (Z × E))
    (hcompatible : ∃ Q ∈ C, (Q : Measure Ω).map (fun ω => (W ω, H ω)) = nu) :
    {P | P ∈ C ∧ (P : Measure Ω).map (fun ω => (W ω, H ω)) = nu} =
      {P | P ∈ C ∧ (P : Measure Ω).map W = nu.fst} := by
  have hH : Measurable H := by
    have he : H = g ∘ W := funext hdet
    rw [he]
    exact hg.comp hW
  have hrec (P : ProbabilityMeasure Ω) :
      (P : Measure Ω).map (fun ω => (W ω, H ω)) =
        ((P : Measure Ω).map W).map (fun z => (z, g z)) := by
    have hgraph : Measurable (fun z : Z => (z, g z)) := measurable_id.prodMk hg
    calc
      (P : Measure Ω).map (fun ω => (W ω, H ω)) =
          (P : Measure Ω).map ((fun z : Z => (z, g z)) ∘ W) := by
        congr 1
        funext ω
        exact Prod.ext rfl (hdet ω)
      _ = _ := (Measure.map_map (μ := (P : Measure Ω)) hgraph hW).symm
  have hnu : nu = nu.fst.map (fun z => (z, g z)) := by
    obtain ⟨Q, _, hQ⟩ := hcompatible
    rw [← hQ, Measure.fst_map_prodMk hH]
    exact hrec Q
  apply Set.Subset.antisymm (observable_fibre_refinement C W H hW hH nu)
  intro P hP
  exact ⟨hP.1, by rw [hrec P, hP.2, ← hnu]⟩

end KBound.InformationRefinement

#print axioms KBound.InformationRefinement.zero_disagreement_risks_eq
#print axioms KBound.InformationRefinement.zero_disagreement_populationBenefit
#print axioms KBound.InformationRefinement.fibreRadius_mono
#print axioms KBound.InformationRefinement.observable_fibre_refinement
#print axioms KBound.InformationRefinement.observable_fibreRadius_mono
#print axioms KBound.InformationRefinement.deterministic_observable_fibre_eq
