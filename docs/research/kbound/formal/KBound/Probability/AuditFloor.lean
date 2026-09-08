import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.MeasureTheory.Constructions.BorelSpace.Order
import Mathlib.Topology.Order.IsLUB

/-!
# Exact randomized audit floor on an arbitrary residual fibre

One probability measure represents the common joint law of evidence and audit
randomization. Uniform validity at every attainable residual forces coverage at
the real supremum. The fibre may be uncountable and its supremum unattained.
Boundedness and audit measurability are explicit; neither is inferred from data.
-/

namespace KBound

open MeasureTheory Set Filter
open scoped ENNReal Topology

/-- The radius of the attainable absolute residuals. Its interpretation requires
the nonemptiness and boundedness premises supplied by the theorems below. -/
noncomputable def fibreRadius {P : Type*} (F : Set P) (gamma : P → ℝ) : ℝ :=
  sSup (Set.range fun p : F => |gamma p.1|)

theorem fibreResidualRange_nonempty {P : Type*} {F : Set P}
    (hF : F.Nonempty) (gamma : P → ℝ) :
    (Set.range fun p : F => |gamma p.1|).Nonempty := by
  obtain ⟨p, hp⟩ := hF
  exact ⟨|gamma p|, ⟨⟨p, hp⟩, rfl⟩⟩

/-- A constant radius dominates every world, without requiring a maximizer. -/
theorem constant_fibreRadius_valid {P : Type*} (F : Set P) (gamma : P → ℝ)
    (hbounded : BddAbove (Set.range fun p : F => |gamma p.1|)) (p : F) :
    |gamma p.1| ≤ fibreRadius F gamma :=
  le_csSup hbounded (Set.mem_range_self p)

theorem fibreRadius_nonneg {P : Type*} (F : Set P) (hF : F.Nonempty)
    (gamma : P → ℝ)
    (hbounded : BddAbove (Set.range fun p : F => |gamma p.1|)) :
    0 ≤ fibreRadius F gamma := by
  obtain ⟨p, hp⟩ := hF
  exact (abs_nonneg _).trans (constant_fibreRadius_valid F gamma hbounded ⟨p, hp⟩)

/-- Uniform validity over an arbitrary nonempty bounded fibre implies the
same coverage at its supremum, including when no world attains that supremum.
Nonnegativity and `delta < 1` retain the audit interface; the continuity argument
also works without these two semantic restrictions. -/
theorem measurable_audit_floor {P Ω : Type*} [MeasurableSpace Ω]
    (F : Set P) (hF : F.Nonempty) (gamma : P → ℝ)
    (hbounded : BddAbove (Set.range fun p : F => |gamma p.1|))
    (mu : Measure Ω) [IsProbabilityMeasure mu]
    (betaHat : Ω → ℝ) (hbetaHat : Measurable betaHat)
    (_hbetaHat_nonneg : ∀ omega, 0 ≤ betaHat omega)
    (delta : ℝ≥0∞) (_hdelta : delta < 1)
    (hvalid : ∀ p : F, 1 - delta ≤ mu {omega | |gamma p.1| ≤ betaHat omega}) :
    1 - delta ≤ mu {omega | fibreRadius F gamma ≤ betaHat omega} := by
  obtain ⟨a, ha_mono, ha_lim, ha_mem⟩ :=
    exists_seq_tendsto_sSup (fibreResidualRange_nonempty hF gamma) hbounded
  let E : ℕ → Set Ω := fun n => {omega | a n ≤ betaHat omega}
  have hE_mono : Antitone E := by
    intro i j hij omega homega
    exact (ha_mono hij).trans homega
  have hE_meas : ∀ n, MeasurableSet (E n) := fun _ =>
    measurableSet_le measurable_const hbetaHat
  have hE_valid : ∀ n, 1 - delta ≤ mu (E n) := by
    intro n
    obtain ⟨p, hp⟩ := ha_mem n
    simpa only [E, ← hp] using hvalid p
  have hE_inter : (⋂ n, E n) = {omega | fibreRadius F gamma ≤ betaHat omega} := by
    ext omega
    simp only [Set.mem_iInter, Set.mem_setOf_eq, E]
    constructor
    · intro homega
      exact le_of_tendsto' ha_lim homega
    · intro homega n
      exact (le_csSup hbounded (ha_mem n)).trans homega
  rw [← hE_inter, hE_mono.measure_iInter
    (fun n => (hE_meas n).nullMeasurableSet) ⟨0, measure_ne_top _ _⟩]
  exact le_iInf hE_valid

/-- World-specific joint observation/seed laws are explicitly equal to the
common fibre law. No independence or equality-of-marginals inference is hidden. -/
theorem fibrewise_randomized_audit_floor {P Ω : Type*} [MeasurableSpace Ω]
    (F : Set P) (hF : F.Nonempty) (gamma : P → ℝ)
    (hbounded : BddAbove (Set.range fun p : F => |gamma p.1|))
    (law : F → Measure Ω) (mu : Measure Ω) [IsProbabilityMeasure mu]
    (hcommon : ∀ p, law p = mu)
    (betaHat : Ω → ℝ) (hbetaHat : Measurable betaHat)
    (hbetaHat_nonneg : ∀ omega, 0 ≤ betaHat omega)
    (delta : ℝ≥0∞) (hdelta : delta < 1)
    (hvalid : ∀ p : F, 1 - delta ≤ law p {omega | |gamma p.1| ≤ betaHat omega}) :
    1 - delta ≤ mu {omega | fibreRadius F gamma ≤ betaHat omega} := by
  apply measurable_audit_floor F hF gamma hbounded mu betaHat hbetaHat
    hbetaHat_nonneg delta hdelta
  intro p
  simpa only [hcommon p] using hvalid p

/-- The oracle constant radius is measurable, nonnegative and valid with
probability one simultaneously for every fibre world. This does not estimate it. -/
theorem constant_fibreRadius_audit_valid {P Ω : Type*} [MeasurableSpace Ω]
    (F : Set P) (hF : F.Nonempty) (gamma : P → ℝ)
    (hbounded : BddAbove (Set.range fun p : F => |gamma p.1|))
    (mu : Measure Ω) [IsProbabilityMeasure mu] :
    Measurable (fun _ : Ω => fibreRadius F gamma) ∧
    0 ≤ fibreRadius F gamma ∧
    ∀ p : F, mu {omega : Ω | |gamma p.1| ≤ (fun _ => fibreRadius F gamma) omega} = 1 := by
  refine ⟨measurable_const, fibreRadius_nonneg F hF gamma hbounded, ?_⟩
  intro p
  simp only [constant_fibreRadius_valid F gamma hbounded p, Set.setOf_true, measure_univ]

/-- On the floor event a strict audit-based commitment is possible only
outside the fibre radius. This is a deterministic consequence of the floor. -/
theorem audit_floor_frontier_inert {Gamma betaHat M : ℝ}
    (hfloor : Gamma ≤ betaHat) (hcommit : betaHat < |M|) : Gamma < |M| :=
  hfloor.trans_lt hcommit

/-- Exact attainment is useful for model-specific applications but is not an
assumption of the general randomized-audit theorem. -/
theorem fibreRadius_eq_of_bound_and_witness {P : Type*} (F : Set P)
    (gamma : P → ℝ) (beta : ℝ)
    (hupper : ∀ p : F, |gamma p.1| ≤ beta)
    (hwitness : ∃ p : F, |gamma p.1| = beta) : fibreRadius F gamma = beta := by
  obtain ⟨p, hp⟩ := hwitness
  have hbounded : BddAbove (Set.range fun p : F => |gamma p.1|) :=
    ⟨beta, by rintro _ ⟨q, rfl⟩; exact hupper q⟩
  apply le_antisymm
  · apply csSup_le ⟨|gamma p.1|, Set.mem_range_self p⟩
    rintro _ ⟨q, rfl⟩
    exact hupper q
  · rw [← hp]
    exact constant_fibreRadius_valid F gamma hbounded p

end KBound
