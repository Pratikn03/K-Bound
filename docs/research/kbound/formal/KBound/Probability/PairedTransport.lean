import KBound.Probability.MeasureCertificate
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Real.Archimedean
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Conditional paired transport: true table, LP containment and confidence

The finite source/target conditional distributions and target class prior are
explicit normalized probability vectors. The true LP witness is constructed
as t=B*pi, s=A*pi and v=|t-s|, and every printed constraint is derived from
probability normalization, interval containment and the external TV budget.
Individual probability-box coverage is a supplied statistical premise; this
module does not prove Clopper--Pearson coverage or numerical quantile accuracy.
-/

namespace KBound

noncomputable section

open MeasureTheory Set
open scoped BigOperators ENNReal

structure FiniteProbabilityVector (I : Type*) [Fintype I] where
  value : I → ℝ
  nonneg : ∀ i, 0 ≤ value i
  sum_one : ∑ i, value i = 1

theorem FiniteProbabilityVector.le_one {I : Type*} [Fintype I]
    (p : FiniteProbabilityVector I) (i : I) : p.value i ≤ 1 := by
  classical
  rw [← p.sum_one]
  exact Finset.single_le_sum (fun j _ => p.nonneg j) (Finset.mem_univ i)

structure PairedTransportModel (R Y : Type*) [Fintype R] [Fintype Y] where
  source : Y → FiniteProbabilityVector R
  target : Y → FiniteProbabilityVector R
  prior : FiniteProbabilityVector Y

structure PairedLPPoint (R Y : Type*) where
  targetJoint : R → Y → ℝ
  sourceJoint : R → Y → ℝ
  prior : Y → ℝ
  slack : R → Y → ℝ

variable {R Y : Type*} [Fintype R] [Fintype Y]

def PairedTransportModel.trueTarget (m : PairedTransportModel R Y) (r : R) (y : Y) : ℝ :=
  (m.target y).value r * m.prior.value y

def PairedTransportModel.trueSource (m : PairedTransportModel R Y) (r : R) (y : Y) : ℝ :=
  (m.source y).value r * m.prior.value y

def PairedTransportModel.targetMarginal (m : PairedTransportModel R Y) (r : R) : ℝ :=
  ∑ y, m.trueTarget r y

def PairedTransportModel.aggregateTV (m : PairedTransportModel R Y) : ℝ :=
  ∑ y, m.prior.value y * ((1 / 2) * ∑ r, |(m.target y).value r - (m.source y).value r|)

def PairedTransportModel.truePoint (m : PairedTransportModel R Y) : PairedLPPoint R Y where
  targetJoint := m.trueTarget
  sourceJoint := m.trueSource
  prior := m.prior.value
  slack := fun r y => |m.trueTarget r y - m.trueSource r y|

/-- The literal constraints in the printed paired polytope. -/
structure PairedLPFeasible
    (LA UA : R → Y → ℝ) (Lq Uq : R → ℝ) (rho : ℝ) (w : PairedLPPoint R Y) : Prop where
  prior_sum : ∑ y, w.prior y = 1
  target_columns : ∀ y, ∑ r, w.targetJoint r y = w.prior y
  source_columns : ∀ y, ∑ r, w.sourceJoint r y = w.prior y
  source_boxes : ∀ r y, LA r y * w.prior y ≤ w.sourceJoint r y ∧
    w.sourceJoint r y ≤ UA r y * w.prior y
  target_boxes : ∀ r, Lq r ≤ ∑ y, w.targetJoint r y ∧ (∑ y, w.targetJoint r y) ≤ Uq r
  slack_forward : ∀ r y, w.targetJoint r y - w.sourceJoint r y ≤ w.slack r y
  slack_backward : ∀ r y, w.sourceJoint r y - w.targetJoint r y ≤ w.slack r y
  tv_budget : (1 / 2) * ∑ r, ∑ y, w.slack r y ≤ rho
  target_unit : ∀ r y, 0 ≤ w.targetJoint r y ∧ w.targetJoint r y ≤ 1
  source_unit : ∀ r y, 0 ≤ w.sourceJoint r y ∧ w.sourceJoint r y ≤ 1
  prior_unit : ∀ y, 0 ≤ w.prior y ∧ w.prior y ≤ 1
  slack_unit : ∀ r y, 0 ≤ w.slack r y ∧ w.slack r y ≤ 1

theorem paired_true_target_unit (m : PairedTransportModel R Y) (r : R) (y : Y) :
    0 ≤ m.trueTarget r y ∧ m.trueTarget r y ≤ 1 := by
  constructor
  · exact mul_nonneg ((m.target y).nonneg r) (m.prior.nonneg y)
  · exact mul_le_one₀ ((m.target y).le_one r) (m.prior.nonneg y) (m.prior.le_one y)

theorem paired_true_source_unit (m : PairedTransportModel R Y) (r : R) (y : Y) :
    0 ≤ m.trueSource r y ∧ m.trueSource r y ≤ 1 := by
  constructor
  · exact mul_nonneg ((m.source y).nonneg r) (m.prior.nonneg y)
  · exact mul_le_one₀ ((m.source y).le_one r) (m.prior.nonneg y) (m.prior.le_one y)

/-- The LP slack budget equals the declared prior-weighted conditional TV. -/
theorem paired_true_slack_tv (m : PairedTransportModel R Y) :
    (1 / 2) * ∑ r, ∑ y, m.truePoint.slack r y = m.aggregateTV := by
  have hslack : ∀ r y, m.truePoint.slack r y =
      |(m.target y).value r - (m.source y).value r| * m.prior.value y := by
    intro r y
    change |(m.target y).value r * m.prior.value y -
      (m.source y).value r * m.prior.value y| = _
    rw [← sub_mul, abs_mul, abs_of_nonneg (m.prior.nonneg y)]
  simp_rw [hslack]
  rw [Finset.sum_comm]
  simp only [PairedTransportModel.aggregateTV, ← Finset.sum_mul]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro y _
  ring

/-- Simultaneous source/target box containment gives an explicit true feasible
point. In particular exact nonemptiness follows on this event; it is not
inferred from numerical solver tolerances. -/
theorem paired_true_point_feasible (m : PairedTransportModel R Y)
    {LA UA : R → Y → ℝ} {Lq Uq : R → ℝ} {rho : ℝ}
    (hA : ∀ r y, LA r y ≤ (m.source y).value r ∧ (m.source y).value r ≤ UA r y)
    (hq : ∀ r, Lq r ≤ m.targetMarginal r ∧ m.targetMarginal r ≤ Uq r)
    (hTV : m.aggregateTV ≤ rho) :
    PairedLPFeasible LA UA Lq Uq rho m.truePoint := by
  constructor
  · exact m.prior.sum_one
  · intro y
    change (∑ r, (m.target y).value r * m.prior.value y) = m.prior.value y
    rw [← Finset.sum_mul, (m.target y).sum_one, one_mul]
  · intro y
    change (∑ r, (m.source y).value r * m.prior.value y) = m.prior.value y
    rw [← Finset.sum_mul, (m.source y).sum_one, one_mul]
  · intro r y
    exact ⟨mul_le_mul_of_nonneg_right (hA r y).1 (m.prior.nonneg y),
      mul_le_mul_of_nonneg_right (hA r y).2 (m.prior.nonneg y)⟩
  · exact hq
  · intro r y
    exact le_abs_self _
  · intro r y
    have h := neg_le_abs (m.trueTarget r y - m.trueSource r y)
    change m.trueSource r y - m.trueTarget r y ≤ |m.trueTarget r y - m.trueSource r y|
    linarith
  · exact (paired_true_slack_tv m).trans_le hTV
  · exact paired_true_target_unit m
  · exact paired_true_source_unit m
  · intro y
    exact ⟨m.prior.nonneg y, m.prior.le_one y⟩
  · intro r y
    have ht := paired_true_target_unit m r y
    have hs := paired_true_source_unit m r y
    exact ⟨abs_nonneg _, abs_le.mpr ⟨by linarith, by linarith⟩⟩

def pairedObjective (c : R → Y → ℝ) (w : PairedLPPoint R Y) : ℝ :=
  ∑ r, ∑ y, c r y * w.targetJoint r y

def pairedObjectiveValues (c : R → Y → ℝ)
    (LA UA : R → Y → ℝ) (Lq Uq : R → ℝ) (rho : ℝ) : Set ℝ :=
  {z | ∃ w, PairedLPFeasible LA UA Lq Uq rho w ∧ z = pairedObjective c w}

def pairedAccuracy [DecidableEq Y] (m : PairedTransportModel R Y) (f : R → Y) : ℝ :=
  ∑ r, ∑ y, (if f r = y then (1 : ℝ) else 0) * m.trueTarget r y

def pairedContrast [DecidableEq Y] (f₀ fₐ : R → Y) (r : R) (y : Y) : ℝ :=
  (if fₐ r = y then 1 else 0) - (if f₀ r = y then 1 else 0)

/-- The paired objective is the difference of the two accuracies on the same
true finite joint distribution. -/
theorem paired_objective_is_benefit [DecidableEq Y]
    (m : PairedTransportModel R Y) (f₀ fₐ : R → Y) :
    pairedObjective (pairedContrast f₀ fₐ) m.truePoint =
      pairedAccuracy m fₐ - pairedAccuracy m f₀ := by
  simp only [pairedObjective, pairedContrast, PairedTransportModel.truePoint,
    sub_mul, Finset.sum_sub_distrib, pairedAccuracy]

/-- Bounded LP coordinates make the exact objective set bounded. -/
theorem paired_objective_abs_bound {c : R → Y → ℝ}
    {LA UA : R → Y → ℝ} {Lq Uq : R → ℝ} {rho : ℝ} {w : PairedLPPoint R Y}
    (hw : PairedLPFeasible LA UA Lq Uq rho w) :
    |pairedObjective c w| ≤ ∑ r, ∑ y, |c r y| := by
  unfold pairedObjective
  calc
    |∑ r, ∑ y, c r y * w.targetJoint r y| ≤ ∑ r, |∑ y, c r y * w.targetJoint r y| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ r, ∑ y, |c r y * w.targetJoint r y| := by
      apply Finset.sum_le_sum
      intro r _
      exact Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ r, ∑ y, |c r y| := by
      apply Finset.sum_le_sum
      intro r _
      apply Finset.sum_le_sum
      intro y _
      rw [abs_mul, abs_of_nonneg (hw.target_unit r y).1]
      exact mul_le_of_le_one_right (abs_nonneg _) (hw.target_unit r y).2

/-- Exact objective containment is derived from the explicit true witness,
using infimum/supremum endpoints; no solver-supplied feasibility is assumed. -/
theorem paired_true_benefit_in_objective_interval (m : PairedTransportModel R Y)
    (c : R → Y → ℝ) {LA UA : R → Y → ℝ} {Lq Uq : R → ℝ} {rho : ℝ}
    (hA : ∀ r y, LA r y ≤ (m.source y).value r ∧ (m.source y).value r ≤ UA r y)
    (hq : ∀ r, Lq r ≤ m.targetMarginal r ∧ m.targetMarginal r ≤ Uq r)
    (hTV : m.aggregateTV ≤ rho) :
    sInf (pairedObjectiveValues c LA UA Lq Uq rho) ≤ pairedObjective c m.truePoint ∧
    pairedObjective c m.truePoint ≤ sSup (pairedObjectiveValues c LA UA Lq Uq rho) := by
  have hmem : pairedObjective c m.truePoint ∈ pairedObjectiveValues c LA UA Lq Uq rho :=
    ⟨m.truePoint, paired_true_point_feasible m hA hq hTV, rfl⟩
  have hlo : BddBelow (pairedObjectiveValues c LA UA Lq Uq rho) := by
    refine ⟨-(∑ r, ∑ y, |c r y|), ?_⟩
    rintro z ⟨w, hw, rfl⟩
    exact (abs_le.mp (paired_objective_abs_bound hw)).1
  have hhi : BddAbove (pairedObjectiveValues c LA UA Lq Uq rho) := by
    refine ⟨(∑ r, ∑ y, |c r y|), ?_⟩
    rintro z ⟨w, hw, rfl⟩
    exact (abs_le.mp (paired_objective_abs_bound hw)).2
  exact ⟨csInf_le hlo hmem, le_csSup hhi hmem⟩

/-- Individual interval coverage budgets combine without independence of the
events. This is the exact union-bound layer, not a binomial-coverage theorem. -/
theorem finite_boxes_miss_le
    {Ω J : Type*} [MeasurableSpace Ω] [Fintype J]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (boxes : J → Set Ω)
    (hboxes : ∀ j, MeasurableSet (boxes j)) (budget : J → ℝ≥0∞)
    {alpha : ℝ≥0∞} (hsum : ∑ j, budget j ≤ alpha)
    (hcov : ∀ j, 1 - budget j ≤ μ (boxes j)) :
    μ (⋂ j, boxes j)ᶜ ≤ alpha := by
  have hmiss : ∀ j, μ (boxes j)ᶜ ≤ budget j := by
    intro j
    exact measure_le_alpha_of_subset_compl (hboxes j) (fun _ h => h) (hcov j)
  rw [compl_iInter]
  calc
    μ (⋃ j, (boxes j)ᶜ) ≤ ∑ j, μ (boxes j)ᶜ := measure_iUnion_fintype_le μ _
    _ ≤ ∑ j, budget j := Finset.sum_le_sum fun j _ => hmiss j
    _ ≤ alpha := hsum

/-- Equal allocation over every declared source and target probability box
has exactly the requested total error budget. -/
theorem uniform_probability_box_budget {J : Type*} [Fintype J] [Nonempty J]
    (alpha : ℝ≥0∞) :
    (∑ _j : J, alpha / (Fintype.card J : ℝ≥0∞)) = alpha := by
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_comm]
  exact ENNReal.div_mul_cancel (by exact_mod_cast Fintype.card_ne_zero) (by simp)

def pairedProbabilityBoxEvent {Ω : Type*} (m : PairedTransportModel R Y)
    (LA UA : Ω → R → Y → ℝ) (Lq Uq : Ω → R → ℝ) :
    (R × Y) ⊕ R → Set Ω
  | .inl (r, y) => {ω | LA ω r y ≤ (m.source y).value r ∧ (m.source y).value r ≤ UA ω r y}
  | .inr r => {ω | Lq ω r ≤ m.targetMarginal r ∧ m.targetMarginal r ≤ Uq ω r}

/-- On simultaneous probability-box containment, the true finite target law
constructs a feasible LP table and its paired benefit is enclosed. -/
theorem paired_box_event_implies_objective_containment
    {Ω : Type*} (m : PairedTransportModel R Y) (c : R → Y → ℝ)
    (LA UA : Ω → R → Y → ℝ) (Lq Uq : Ω → R → ℝ)
    {rho : ℝ} (hTV : m.aggregateTV ≤ rho) {ω : Ω}
    (hgood : ω ∈ ⋂ j, pairedProbabilityBoxEvent m LA UA Lq Uq j) :
    sInf (pairedObjectiveValues c (LA ω) (UA ω) (Lq ω) (Uq ω) rho) ≤
        pairedObjective c m.truePoint ∧
      pairedObjective c m.truePoint ≤
        sSup (pairedObjectiveValues c (LA ω) (UA ω) (Lq ω) (Uq ω) rho) := by
  apply paired_true_benefit_in_objective_interval m c
  · intro r y
    exact Set.mem_iInter.mp hgood (.inl (r, y))
  · intro r
    exact Set.mem_iInter.mp hgood (.inr r)
  · exact hTV

/-- Conditional paired-benefit coverage from separately supplied individual
box guarantees and the external transport restriction. -/
theorem paired_transport_coverage_from_boxes
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (m : PairedTransportModel R Y) (c : R → Y → ℝ)
    (LA UA : Ω → R → Y → ℝ) (Lq Uq : Ω → R → ℝ)
    (hmeas : ∀ j, MeasurableSet (pairedProbabilityBoxEvent m LA UA Lq Uq j))
    (budget : (R × Y) ⊕ R → ℝ≥0∞) {alpha : ℝ≥0∞}
    (hsum : ∑ j, budget j ≤ alpha)
    (hcov : ∀ j, 1 - budget j ≤ μ (pairedProbabilityBoxEvent m LA UA Lq Uq j))
    {rho : ℝ} (hTV : m.aggregateTV ≤ rho) :
    1 - alpha ≤ μ {ω |
      sInf (pairedObjectiveValues c (LA ω) (UA ω) (Lq ω) (Uq ω) rho) ≤
          pairedObjective c m.truePoint ∧
      pairedObjective c m.truePoint ≤
        sSup (pairedObjectiveValues c (LA ω) (UA ω) (Lq ω) (Uq ω) rho)} := by
  let good := ⋂ j, pairedProbabilityBoxEvent m LA UA Lq Uq j
  have hgood : MeasurableSet good := MeasurableSet.iInter hmeas
  have hmiss : μ goodᶜ ≤ alpha :=
    finite_boxes_miss_le μ _ hmeas budget hsum hcov
  have hgoodprob : 1 - alpha ≤ μ good := by
    calc
      1 - alpha ≤ 1 - μ goodᶜ := tsub_le_tsub_left hmiss 1
      _ = μ good := by
        rw [prob_compl_eq_one_sub hgood,
          ENNReal.sub_sub_cancel ENNReal.one_ne_top (prob_le_one : μ good ≤ 1)]
  apply hgoodprob.trans
  apply measure_mono
  intro ω hω
  exact paired_box_event_implies_objective_containment m c LA UA Lq Uq hTV hω

/-- Either wrong strict direction requires failure of the same simultaneous
box event. The union, not merely each direction, has probability at most alpha.
This outer-measure bound also applies when endpoint measurability is supplied
separately; numerical optimization is not part of its hypotheses. -/
theorem paired_transport_either_error_le
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (m : PairedTransportModel R Y) (c : R → Y → ℝ)
    (LA UA : Ω → R → Y → ℝ) (Lq Uq : Ω → R → ℝ)
    (hmeas : ∀ j, MeasurableSet (pairedProbabilityBoxEvent m LA UA Lq Uq j))
    (budget : (R × Y) ⊕ R → ℝ≥0∞) {alpha : ℝ≥0∞}
    (hsum : ∑ j, budget j ≤ alpha)
    (hcov : ∀ j, 1 - budget j ≤ μ (pairedProbabilityBoxEvent m LA UA Lq Uq j))
    {rho : ℝ} (hTV : m.aggregateTV ≤ rho) :
    μ {ω |
      (0 < sInf (pairedObjectiveValues c (LA ω) (UA ω) (Lq ω) (Uq ω) rho) ∧
        pairedObjective c m.truePoint ≤ 0) ∨
      (sSup (pairedObjectiveValues c (LA ω) (UA ω) (Lq ω) (Uq ω) rho) < 0 ∧
        0 ≤ pairedObjective c m.truePoint)} ≤ alpha := by
  apply le_trans (measure_mono ?_)
    (finite_boxes_miss_le μ _ hmeas budget hsum hcov)
  intro ω hbad hgood
  have hbound := paired_box_event_implies_objective_containment m c LA UA Lq Uq hTV hgood
  rcases hbad with ⟨hpos, hnonpos⟩ | ⟨hneg, hnonneg⟩ <;> linarith

/-- The strict interval rule, with explicit abstention for an empty feasible
objective set. -/
def pairedTransportRule (S : Set ℝ) : Decision := by
  classical
  exact if S.Nonempty then
    if 0 < sInf S then .adapt else if sSup S < 0 then .freeze else .abstain
  else .abstain

@[simp] theorem paired_transport_rule_empty : pairedTransportRule ∅ = .abstain := by
  simp [pairedTransportRule]

/-- A zero-touching/straddling exact interval does not certify a strict sign. -/
theorem paired_transport_rule_zero_in_interval {S : Set ℝ}
    (hlo : sInf S ≤ 0) (hhi : 0 ≤ sSup S) : pairedTransportRule S = .abstain := by
  simp [pairedTransportRule, not_lt.mpr hlo, not_lt.mpr hhi]

theorem paired_rule_error_implies_endpoint_error {S : Set ℝ} {benefit : ℝ}
    (hbad : (pairedTransportRule S = .adapt ∧ benefit ≤ 0) ∨
      (pairedTransportRule S = .freeze ∧ 0 ≤ benefit)) :
    (0 < sInf S ∧ benefit ≤ 0) ∨ (sSup S < 0 ∧ 0 ≤ benefit) := by
  by_cases hS : S.Nonempty
  · by_cases hlo : 0 < sInf S
    · simp [pairedTransportRule, hS, hlo] at hbad
      exact Or.inl ⟨hlo, hbad⟩
    · by_cases hhi : sSup S < 0
      · simp [pairedTransportRule, hS, hlo, hhi] at hbad
        exact Or.inr ⟨hhi, hbad⟩
      · simp [pairedTransportRule, hS, hlo, hhi] at hbad
  · simp [pairedTransportRule, hS] at hbad

/-- The declared three-way paired-transport rule inherits the single total
error budget, including its empty-set abstention branch. -/
theorem paired_transport_rule_either_error_le
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (m : PairedTransportModel R Y) (c : R → Y → ℝ)
    (LA UA : Ω → R → Y → ℝ) (Lq Uq : Ω → R → ℝ)
    (hmeas : ∀ j, MeasurableSet (pairedProbabilityBoxEvent m LA UA Lq Uq j))
    (budget : (R × Y) ⊕ R → ℝ≥0∞) {alpha : ℝ≥0∞}
    (hsum : ∑ j, budget j ≤ alpha)
    (hcov : ∀ j, 1 - budget j ≤ μ (pairedProbabilityBoxEvent m LA UA Lq Uq j))
    {rho : ℝ} (hTV : m.aggregateTV ≤ rho) :
    μ {ω |
      (pairedTransportRule (pairedObjectiveValues c (LA ω) (UA ω) (Lq ω) (Uq ω) rho) =
          .adapt ∧ pairedObjective c m.truePoint ≤ 0) ∨
      (pairedTransportRule (pairedObjectiveValues c (LA ω) (UA ω) (Lq ω) (Uq ω) rho) =
          .freeze ∧ 0 ≤ pairedObjective c m.truePoint)} ≤ alpha := by
  apply le_trans (measure_mono ?_)
    (paired_transport_either_error_le μ m c LA UA Lq Uq hmeas budget hsum hcov hTV)
  intro ω hbad
  exact paired_rule_error_implies_endpoint_error hbad

/-- On the finite/countable count-sample spaces used by exact binomial
intervals, the objective coverage and rule-error events are measurable. This
separates ordinary probability claims from the more general outer-measure
bounds above without assuming measurability of an optimization solver. -/
theorem paired_count_sample_events_measurable
    {Ω : Type*} [MeasurableSpace Ω] [MeasurableSingletonClass Ω] [Countable Ω]
    (m : PairedTransportModel R Y) (c : R → Y → ℝ)
    (LA UA : Ω → R → Y → ℝ) (Lq Uq : Ω → R → ℝ) (rho : ℝ) :
    MeasurableSet {ω |
      sInf (pairedObjectiveValues c (LA ω) (UA ω) (Lq ω) (Uq ω) rho) ≤
          pairedObjective c m.truePoint ∧
      pairedObjective c m.truePoint ≤
        sSup (pairedObjectiveValues c (LA ω) (UA ω) (Lq ω) (Uq ω) rho)} ∧
    MeasurableSet {ω |
      (pairedTransportRule (pairedObjectiveValues c (LA ω) (UA ω) (Lq ω) (Uq ω) rho) =
          .adapt ∧ pairedObjective c m.truePoint ≤ 0) ∨
      (pairedTransportRule (pairedObjectiveValues c (LA ω) (UA ω) (Lq ω) (Uq ω) rho) =
          .freeze ∧ 0 ≤ pairedObjective c m.truePoint)} := by
  constructor <;> exact (Set.to_countable _).measurableSet

end

end KBound
