import KBound.Probability.AuditFloorCorrectness
import Mathlib.MeasureTheory.Measure.Dirac
import Mathlib.Order.Interval.Set.Infinite
import Mathlib.Tactic.NormNum

/-!
# Audit-floor contract examples

The open unit interval has infinitely many worlds and no maximizing world.
Boundary checks and a concrete two-point seed law exercise the same general
theorem. These synthetic mathematical examples do not validate empirical laws.
-/

namespace KBound.AuditFloorExamples

open MeasureTheory Set
open scoped ENNReal

theorem openUnit_infinite : (Set.Ioo (0 : ℝ) 1).Infinite :=
  Set.Ioo_infinite (by norm_num)

theorem openUnit_residual_range :
    (Set.range fun p : Set.Ioo (0 : ℝ) 1 => |p.1|) = Set.Ioo (0 : ℝ) 1 := by
  ext x
  constructor
  · rintro ⟨p, rfl⟩
    simpa only [abs_of_pos p.property.1] using p.property
  · intro hx
    exact ⟨⟨x, hx⟩, abs_of_pos hx.1⟩

theorem openUnit_radius : fibreRadius (Set.Ioo (0 : ℝ) 1) id = 1 := by
  unfold fibreRadius
  change sSup (Set.range fun p : Set.Ioo (0 : ℝ) 1 => |p.1|) = 1
  rw [openUnit_residual_range, csSup_Ioo (by norm_num : (0 : ℝ) < 1)]

theorem openUnit_no_maximizer :
    ∀ p : Set.Ioo (0 : ℝ) 1, |p.1| < fibreRadius (Set.Ioo (0 : ℝ) 1) id := by
  intro p
  rw [openUnit_radius, abs_of_pos p.property.1]
  exact p.property.2

/-- The nonattained radius still has the uniform coverage probability. -/
theorem openUnit_audit_floor {Ω : Type*} [MeasurableSpace Ω]
    (mu : Measure Ω) [IsProbabilityMeasure mu]
    (b : Ω → ℝ) (hb : Measurable b) (hb0 : ∀ omega, 0 ≤ b omega)
    (delta : ℝ≥0∞) (hd : delta < 1)
    (hv : ∀ p : Set.Ioo (0 : ℝ) 1, 1 - delta ≤ mu {omega | |p.1| ≤ b omega}) :
    1 - delta ≤ mu {omega | 1 ≤ b omega} := by
  have hbounded : BddAbove (Set.range fun p : Set.Ioo (0 : ℝ) 1 => |p.1|) := by
    rw [openUnit_residual_range]
    exact bddAbove_Ioo
  simpa only [openUnit_radius] using
    measurable_audit_floor (Set.Ioo (0 : ℝ) 1) (nonempty_Ioo.mpr (by norm_num))
      id hbounded mu b hb hb0 delta hd hv

theorem singleton_radius : fibreRadius ({(3 : ℝ)} : Set ℝ) id = 3 := by
  apply fibreRadius_eq_of_bound_and_witness _ _ 3
  · intro p
    have hp : p.1 = 3 := p.property
    simp [hp]
  · exact ⟨⟨3, by simp⟩, by norm_num⟩

theorem finite_two_world_radius :
    fibreRadius (Set.univ : Set Bool) (fun b => if b then (1 : ℝ) else 1 / 2) = 1 := by
  apply fibreRadius_eq_of_bound_and_witness _ _ 1
  · rintro ⟨b, _⟩
    cases b <;> norm_num
  · exact ⟨⟨true, Set.mem_univ _⟩, by norm_num⟩

/-- Zero residual budget, zero failure budget and a constant zero audit. -/
theorem zero_audit {Ω : Type*} [MeasurableSpace Ω]
    (mu : Measure Ω) [IsProbabilityMeasure mu] :
    1 ≤ mu {_omega : Ω | fibreRadius (Set.univ : Set Unit) (fun _ => 0) ≤ (0 : ℝ)} := by
  have hbounded : BddAbove (Set.range fun _p : (Set.univ : Set Unit) => |(0 : ℝ)|) := by
    exact ⟨0, by simp⟩
  simpa only [tsub_zero] using measurable_audit_floor (Set.univ : Set Unit)
    ⟨(), Set.mem_univ _⟩ (fun _ => 0) hbounded mu (fun _ => 0)
    measurable_const (fun _ => le_rfl) 0 (by norm_num)
    (by intro p; simp)

/-- Every real residual cannot satisfy the bounded-range premise. -/
theorem unbounded_family_rejected :
    ¬ BddAbove (Set.range fun p : (Set.univ : Set ℝ) => |p.1|) := by
  rintro ⟨b, hb⟩
  have h := hb (Set.mem_range_self (⟨|b| + 1, Set.mem_univ _⟩ : (Set.univ : Set ℝ)))
  change |(|b| + 1)| ≤ b at h
  rw [abs_of_pos (by positivity)] at h
  have := le_abs_self b
  linarith

/-- Removing measurability or boundedness from a call leaves an unprovable
contract obligation. The correctly supplied premises close the very same goal. -/
example {P Ω : Type*} [MeasurableSpace Ω]
    (F : Set P) (hF : F.Nonempty) (gamma : P → ℝ)
    (hbounded : BddAbove (Set.range fun p : F => |gamma p.1|))
    (mu : Measure Ω) [IsProbabilityMeasure mu]
    (b : Ω → ℝ) (hb : Measurable b) (hb0 : ∀ omega, 0 ≤ b omega)
    (delta : ℝ≥0∞) (hd : delta < 1)
    (hv : ∀ p : F, 1 - delta ≤ mu {omega | |gamma p.1| ≤ b omega}) :
    1 - delta ≤ mu {omega | fibreRadius F gamma ≤ b omega} := by
  fail_if_success
    exact measurable_audit_floor F hF gamma hbounded mu b
      (by clear hb; assumption) hb0 delta hd hv
  fail_if_success
    exact measurable_audit_floor F hF gamma (by clear hbounded; assumption)
      mu b hb hb0 delta hd hv
  exact measurable_audit_floor F hF gamma hbounded mu b hb hb0 delta hd hv

/-- Syntactically distinct world laws need the supplied common-law equalities. -/
example {P Ω : Type*} [MeasurableSpace Ω]
    (F : Set P) (hF : F.Nonempty) (gamma : P → ℝ)
    (hbounded : BddAbove (Set.range fun p : F => |gamma p.1|))
    (law : F → Measure Ω) (mu : Measure Ω) [IsProbabilityMeasure mu]
    (hc : ∀ p, law p = mu) (b : Ω → ℝ) (hb : Measurable b)
    (hb0 : ∀ omega, 0 ≤ b omega) (delta : ℝ≥0∞) (hd : delta < 1)
    (hv : ∀ p : F, 1 - delta ≤ law p {omega | |gamma p.1| ≤ b omega}) :
    1 - delta ≤ mu {omega | fibreRadius F gamma ≤ b omega} := by
  fail_if_success
    exact fibrewise_randomized_audit_floor F hF gamma hbounded law mu
      (by clear hc; assumption) b hb hb0 delta hd hv
  exact fibrewise_randomized_audit_floor F hF gamma hbounded law mu hc b hb hb0 delta hd hv

section ProductSeed

local instance : MeasurableSpace Unit := ⊤
local instance : MeasurableSpace Bool := ⊤

noncomputable def productSeedLaw : Measure (Unit × Bool) :=
  (1 / 2 : ℝ≥0∞) • Measure.dirac ((), false) +
    (1 / 2 : ℝ≥0∞) • Measure.dirac ((), true)

instance productSeed_probability : IsProbabilityMeasure productSeedLaw := by
  constructor
  simp [productSeedLaw, ENNReal.inv_two_add_inv_two]

def seedAudit (omega : Unit × Bool) : ℝ := if omega.2 then 1 else 0

theorem seedAudit_randomized :
    productSeedLaw {omega | seedAudit omega = 0} = 1 / 2 ∧
    productSeedLaw {omega | seedAudit omega = 1} = 1 / 2 := by
  norm_num [productSeedLaw, seedAudit, Measure.dirac_apply]

theorem product_seed_audit_floor :
    1 - (1 / 2 : ℝ≥0∞) ≤
      productSeedLaw {omega | fibreRadius (Set.univ : Set Unit) (fun _ => 1) ≤ seedAudit omega} := by
  have hbounded : BddAbove (Set.range fun _p : (Set.univ : Set Unit) => |(1 : ℝ)|) := by
    exact ⟨1, by simp⟩
  apply measurable_audit_floor (Set.univ : Set Unit) ⟨(), Set.mem_univ _⟩
    (fun _ => 1) hbounded productSeedLaw seedAudit (measurable_of_countable _)
    (fun omega => by unfold seedAudit; split <;> norm_num) (1 / 2) (by norm_num)
  intro p
  norm_num [productSeedLaw, seedAudit, Measure.dirac_apply]

end ProductSeed

section CorrectnessBoundaries

variable {X : Type*} [MeasurableSpace X]
    (mu : Measure X) [IsProbabilityMeasure mu] (D : Set X) (hD : 0 < mu.real D)

example : fibreRadius (correctnessFibre mu D (-1 / 2) (1 / 2))
    (correctnessResidual mu D (-1 / 2)) = 1 / 2 := by
  apply correctness_fibreRadius_eq_beta mu D hD <;> norm_num

example : fibreRadius (correctnessFibre mu D (1 / 2) (1 / 2))
    (correctnessResidual mu D (1 / 2)) = 1 / 2 := by
  apply correctness_fibreRadius_eq_beta mu D hD <;> norm_num

example : fibreRadius (correctnessFibre mu D 0 0) (correctnessResidual mu D 0) = 0 := by
  apply correctness_fibreRadius_eq_beta mu D hD <;> norm_num

end CorrectnessBoundaries

end KBound.AuditFloorExamples
