import KBound.Probability.PairedTransport
import Mathlib.Topology.Order.Compact
import Mathlib.Topology.Instances.Real.Lemmas
import Mathlib.Tactic.FunProp

/-!
# Exact extrema of the finite paired-transport LP

The literal linear constraint set is closed and lies in a compact coordinate
cube. Thus the finite linear objective has attained minimum and maximum on
every nonempty feasible set. No compactness or solver-attainment premise is
supplied by the caller. This is an exact real-arithmetic statement, not a
claim about floating-point optimization.
-/

namespace KBound

noncomputable section

open Set
open scoped BigOperators

abbrev PairedCoordinates (R Y : Type*) :=
  ((R → Y → ℝ) × (R → Y → ℝ)) × ((Y → ℝ) × (R → Y → ℝ))

def pairedDecode {R Y : Type*} (x : PairedCoordinates R Y) : PairedLPPoint R Y :=
  ⟨x.1.1, x.1.2, x.2.1, x.2.2⟩

def pairedEncode {R Y : Type*} (w : PairedLPPoint R Y) : PairedCoordinates R Y :=
  ((w.targetJoint, w.sourceJoint), (w.prior, w.slack))

@[simp] theorem paired_decode_encode {R Y : Type*} (w : PairedLPPoint R Y) :
    pairedDecode (pairedEncode w) = w := by
  cases w
  rfl

variable {R Y : Type*} [Fintype R] [Fintype Y]

def pairedFeasibleCoordinates (LA UA : R → Y → ℝ) (Lq Uq : R → ℝ) (rho : ℝ) :
    Set (PairedCoordinates R Y) :=
  {x | PairedLPFeasible LA UA Lq Uq rho (pairedDecode x)}

/-- Every equality and inequality in the stated LP is a closed condition. -/
theorem paired_feasible_coordinates_closed
    (LA UA : R → Y → ℝ) (Lq Uq : R → ℝ) (rho : ℝ) :
    IsClosed (pairedFeasibleCoordinates LA UA Lq Uq rho) := by
  have hiff : ∀ x : PairedCoordinates R Y,
      PairedLPFeasible LA UA Lq Uq rho (pairedDecode x) ↔
      (∑ y, x.2.1 y = 1) ∧
      (∀ y, ∑ r, x.1.1 r y = x.2.1 y) ∧
      (∀ y, ∑ r, x.1.2 r y = x.2.1 y) ∧
      (∀ r y, LA r y * x.2.1 y ≤ x.1.2 r y ∧ x.1.2 r y ≤ UA r y * x.2.1 y) ∧
      (∀ r, Lq r ≤ ∑ y, x.1.1 r y ∧ (∑ y, x.1.1 r y) ≤ Uq r) ∧
      (∀ r y, x.1.1 r y - x.1.2 r y ≤ x.2.2 r y) ∧
      (∀ r y, x.1.2 r y - x.1.1 r y ≤ x.2.2 r y) ∧
      ((1 / 2 : ℝ) * ∑ r, ∑ y, x.2.2 r y ≤ rho) ∧
      (∀ r y, 0 ≤ x.1.1 r y ∧ x.1.1 r y ≤ 1) ∧
      (∀ r y, 0 ≤ x.1.2 r y ∧ x.1.2 r y ≤ 1) ∧
      (∀ y, 0 ≤ x.2.1 y ∧ x.2.1 y ≤ 1) ∧
      (∀ r y, 0 ≤ x.2.2 r y ∧ x.2.2 r y ≤ 1) := by
    intro x
    constructor
    · intro h
      exact ⟨h.prior_sum, h.target_columns, h.source_columns, h.source_boxes,
        h.target_boxes, h.slack_forward, h.slack_backward, h.tv_budget,
        h.target_unit, h.source_unit, h.prior_unit, h.slack_unit⟩
    · rintro ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12⟩
      exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12⟩
  unfold pairedFeasibleCoordinates
  simp_rw [hiff, setOf_and, setOf_forall]
  repeat' first
    | apply IsClosed.inter
    | apply isClosed_iInter; intro i
    | rw [setOf_and]; apply IsClosed.inter
  all_goals first
    | apply isClosed_eq <;> fun_prop
    | apply isClosed_le (α := ℝ) <;> fun_prop

/-- The coordinate bounds explicitly present in the LP give its compact cube. -/
theorem paired_feasible_coordinates_subset_cube
    (LA UA : R → Y → ℝ) (Lq Uq : R → ℝ) (rho : ℝ) :
    pairedFeasibleCoordinates LA UA Lq Uq rho ⊆
      Icc (0 : PairedCoordinates R Y) 1 := by
  intro x hx
  constructor
  · exact ⟨⟨fun r y => (hx.target_unit r y).1, fun r y => (hx.source_unit r y).1⟩,
      ⟨fun y => (hx.prior_unit y).1, fun r y => (hx.slack_unit r y).1⟩⟩
  · exact ⟨⟨fun r y => (hx.target_unit r y).2, fun r y => (hx.source_unit r y).2⟩,
      ⟨fun y => (hx.prior_unit y).2, fun r y => (hx.slack_unit r y).2⟩⟩

theorem paired_feasible_coordinates_compact
    (LA UA : R → Y → ℝ) (Lq Uq : R → ℝ) (rho : ℝ) :
    IsCompact (pairedFeasibleCoordinates LA UA Lq Uq rho) :=
  isCompact_Icc.of_isClosed_subset
    (paired_feasible_coordinates_closed LA UA Lq Uq rho)
    (paired_feasible_coordinates_subset_cube LA UA Lq Uq rho)

theorem paired_objective_continuous (c : R → Y → ℝ) :
    Continuous (fun x : PairedCoordinates R Y => pairedObjective c (pairedDecode x)) := by
  unfold pairedObjective pairedDecode
  fun_prop

/-- The exact objective-value set is a continuous image of the actual LP. -/
theorem paired_objective_values_compact
    (c : R → Y → ℝ) (LA UA : R → Y → ℝ) (Lq Uq : R → ℝ) (rho : ℝ) :
    IsCompact (pairedObjectiveValues c LA UA Lq Uq rho) := by
  have himage : pairedObjectiveValues c LA UA Lq Uq rho =
      (fun x : PairedCoordinates R Y => pairedObjective c (pairedDecode x)) ''
        pairedFeasibleCoordinates LA UA Lq Uq rho := by
    ext z
    constructor
    · rintro ⟨w, hw, rfl⟩
      exact ⟨pairedEncode w, by simpa [pairedFeasibleCoordinates] using hw, by simp⟩
    · rintro ⟨x, hx, rfl⟩
      exact ⟨pairedDecode x, hx, rfl⟩
  rw [himage]
  exact (paired_feasible_coordinates_compact LA UA Lq Uq rho).image
    (paired_objective_continuous c)

/-- Every nonempty feasible paired LP attains both exact endpoints. -/
theorem paired_objective_extrema_attained
    (c : R → Y → ℝ) (LA UA : R → Y → ℝ) (Lq Uq : R → ℝ) (rho : ℝ)
    (hne : ∃ w, PairedLPFeasible LA UA Lq Uq rho w) :
    ∃ wmin wmax,
      PairedLPFeasible LA UA Lq Uq rho wmin ∧
      PairedLPFeasible LA UA Lq Uq rho wmax ∧
      pairedObjective c wmin = sInf (pairedObjectiveValues c LA UA Lq Uq rho) ∧
      pairedObjective c wmax = sSup (pairedObjectiveValues c LA UA Lq Uq rho) := by
  have hnonempty : (pairedObjectiveValues c LA UA Lq Uq rho).Nonempty := by
    obtain ⟨w, hw⟩ := hne
    exact ⟨pairedObjective c w, w, hw, rfl⟩
  have hcompact := paired_objective_values_compact c LA UA Lq Uq rho
  obtain ⟨zmin, hzmin⟩ := hcompact.exists_isLeast hnonempty
  obtain ⟨zmax, hzmax⟩ := hcompact.exists_isGreatest hnonempty
  obtain ⟨wmin, hwmin, rfl⟩ := hzmin.1
  obtain ⟨wmax, hwmax, rfl⟩ := hzmax.1
  exact ⟨wmin, wmax, hwmin, hwmax, hzmin.csInf_eq.symm, hzmax.csSup_eq.symm⟩

/-- Box containment and the supplied TV budget imply existence of true feasible
mass, and therefore attainment without a separately assumed nonempty LP. -/
theorem paired_boxes_imply_extrema_attained
    (m : PairedTransportModel R Y) (c : R → Y → ℝ)
    {LA UA : R → Y → ℝ} {Lq Uq : R → ℝ} {rho : ℝ}
    (hA : ∀ r y, LA r y ≤ (m.source y).value r ∧ (m.source y).value r ≤ UA r y)
    (hq : ∀ r, Lq r ≤ m.targetMarginal r ∧ m.targetMarginal r ≤ Uq r)
    (hTV : m.aggregateTV ≤ rho) :
    ∃ wmin wmax,
      PairedLPFeasible LA UA Lq Uq rho wmin ∧
      PairedLPFeasible LA UA Lq Uq rho wmax ∧
      pairedObjective c wmin = sInf (pairedObjectiveValues c LA UA Lq Uq rho) ∧
      pairedObjective c wmax = sSup (pairedObjectiveValues c LA UA Lq Uq rho) :=
  paired_objective_extrema_attained c LA UA Lq Uq rho
    ⟨m.truePoint, paired_true_point_feasible m hA hq hTV⟩

end
end KBound
