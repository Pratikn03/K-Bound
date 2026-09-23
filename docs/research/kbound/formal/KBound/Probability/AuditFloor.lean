import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.MeasureTheory.Constructions.BorelSpace.Order
import Mathlib.MeasureTheory.Measure.Prod
import Mathlib.Topology.Order.IsLUB
import KBound.Probability.MeasureTarget

/-!
# Exact evidence-fibre residual audit floor

An arbitrary nonempty bounded set of real residual magnitudes need not attain
its supremum. The proof constructs an increasing attainable sequence and uses
continuity of probability from above, preserving the single miss budget. A
common independent seed is then used to derive equality of audit laws from
equality of observable laws. No finite support or attained-maximum premise is
substituted for the printed arbitrary-fibre theorem.
-/

namespace KBound

open MeasureTheory Set Filter
open scoped ENNReal Topology

/-- Uniform lower-tail coverage of every attainable value forces coverage of
their supremum, even when no member attains it. -/
theorem audit_floor_of_uniform_coverage
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {S : Set ℝ} (hne : S.Nonempty) (hbd : BddAbove S)
    {audit : Ω → ℝ} (haudit : Measurable audit) {delta : ℝ≥0∞}
    (hcov : ∀ r ∈ S, 1 - delta ≤ μ {ω | r ≤ audit ω}) :
    1 - delta ≤ μ {ω | sSup S ≤ audit ω} := by
  obtain ⟨u, hmono, hlim, hu⟩ := exists_seq_tendsto_sSup hne hbd
  have hevent : {ω | sSup S ≤ audit ω} = ⋂ n : ℕ, {ω | u n ≤ audit ω} := by
    ext ω
    simp only [mem_setOf_eq, mem_iInter]
    constructor
    · intro h n
      exact (le_csSup hbd (hu n)).trans h
    · intro h
      exact le_of_tendsto hlim (Eventually.of_forall h)
  have hanti : Antitone (fun n : ℕ => {ω | u n ≤ audit ω}) := by
    intro i j hij ω hω
    exact (hmono hij).trans hω
  rw [hevent, hanti.measure_iInter
    (fun n => (measurableSet_le measurable_const haudit).nullMeasurableSet)
    ⟨0, measure_ne_top μ _⟩]
  exact le_iInf fun n => hcov (u n) (hu n)

/-- The constant supremum is a valid zero-error oracle audit; this does not say
that the radius can be learned from a finite batch. -/
theorem constant_supremum_audit_valid
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {S : Set ℝ} (hbd : BddAbove S) {r : ℝ} (hr : r ∈ S) :
    μ {_ω : Ω | r ≤ sSup S} = 1 := by
  have h := le_csSup hbd hr
  simp [h]

/-- Nonnegative attainable residual magnitudes have a nonnegative supremum. -/
theorem residual_supremum_nonneg {S : Set ℝ} (hne : S.Nonempty)
    (hbd : BddAbove S) (hnonneg : ∀ r ∈ S, 0 ≤ r) :
    0 ≤ sSup S := by
  obtain ⟨r, hr⟩ := hne
  exact (hnonneg r hr).trans (le_csSup hbd hr)

/-- The no-improvement implication holds on the audit-floor coverage event. -/
theorem audited_frontier_no_extra_commitment {M radius audit : ℝ}
    (hfloor : radius ≤ audit) (hcommit : audit < |M|) : radius < |M| :=
  hfloor.trans_lt hcommit

/-- Independent randomization preserves equality of evidence laws. The product
measure is the joint law with a common independent seed. -/
theorem common_independent_seed_audit_law
    {Ω₁ Ω₂ E U : Type*} [MeasurableSpace Ω₁] [MeasurableSpace Ω₂]
    [MeasurableSpace E] [MeasurableSpace U]
    (P : Measure Ω₁) (Q : Measure Ω₂) (ν : Measure U)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q] [IsProbabilityMeasure ν]
    {W₁ : Ω₁ → E} {W₂ : Ω₂ → E}
    (hW₁ : Measurable W₁) (hW₂ : Measurable W₂)
    {audit : E × U → ℝ} (haudit : Measurable audit)
    (hevidence : P.map W₁ = Q.map W₂) :
    (P.prod ν).map (fun p => audit (W₁ p.1, p.2)) =
      (Q.prod ν).map (fun p => audit (W₂ p.1, p.2)) := by
  have h₁ : (P.prod ν).map (fun p => audit (W₁ p.1, p.2)) =
      ((P.map W₁).prod ν).map audit := by
    have hp := Measure.map_prod_map P ν hW₁ measurable_id
    simp only [Measure.map_id] at hp
    rw [hp, Measure.map_map haudit (hW₁.prodMap measurable_id)]
    rfl
  have h₂ : (Q.prod ν).map (fun p => audit (W₂ p.1, p.2)) =
      ((Q.map W₂).prod ν).map audit := by
    have hp := Measure.map_prod_map Q ν hW₂ measurable_id
    simp only [Measure.map_id] at hp
    rw [hp, Measure.map_map haudit (hW₂.prodMap measurable_id)]
    rfl
  rw [h₁, h₂, hevidence]

/-- Full arbitrary-fibre audit floor: all worlds share the observable law,
and the common audit reads only that evidence plus an independent seed. Its
uniform worldwise validity therefore forces the exact residual supremum.
The index type represents the nonempty evidence fibre itself. -/
theorem evidence_fibre_audit_floor
    {I Ω E U : Type*} [Nonempty I] [MeasurableSpace Ω]
    [MeasurableSpace E] [MeasurableSpace U]
    (P : I → Measure Ω) [∀ i, IsProbabilityMeasure (P i)]
    (ν : Measure U) [IsProbabilityMeasure ν]
    (W : I → Ω → E) (hW : ∀ i, Measurable (W i))
    (hevidence : ∀ i j, (P i).map (W i) = (P j).map (W j))
    (gamma : I → ℝ) (hbd : BddAbove (Set.range fun i => |gamma i|))
    (audit : E × U → ℝ) (haudit : Measurable audit) {delta : ℝ≥0∞}
    (hcov : ∀ i, 1 - delta ≤ (P i).prod ν {p | |gamma i| ≤ audit (W i p.1, p.2)}) :
    ∀ i, 1 - delta ≤ (P i).prod ν
      {p | sSup (Set.range fun j => |gamma j|) ≤ audit (W i p.1, p.2)} := by
  intro i
  have hai : Measurable (fun p : Ω × U => audit (W i p.1, p.2)) :=
    haudit.comp ((hW i).comp measurable_fst |>.prodMk measurable_snd)
  apply audit_floor_of_uniform_coverage ((P i).prod ν) (Set.range_nonempty _) hbd hai
  rintro r ⟨j, rfl⟩
  have hj := hcov j
  have haj : Measurable (fun p : Ω × U => audit (W j p.1, p.2)) :=
    haudit.comp ((hW j).comp measurable_fst |>.prodMk measurable_snd)
  have heq := common_independent_seed_audit_law (P i) (P j) ν
    (hW i) (hW j) haudit (hevidence i j)
  have hevent : ((P i).prod ν).map (fun p => audit (W i p.1, p.2)) (Ici |gamma j|) =
      ((P j).prod ν).map (fun p => audit (W j p.1, p.2)) (Ici |gamma j|) := by rw [heq]
  rw [Measure.map_apply hai measurableSet_Ici,
      Measure.map_apply haj measurableSet_Ici] at hevent
  change ((P i).prod ν) {p | |gamma j| ≤ audit (W i p.1, p.2)} =
    ((P j).prod ν) {p | |gamma j| ≤ audit (W j p.1, p.2)} at hevent
  rwa [hevent]

/-- All residual magnitudes attainable by the declared full correctness-field
class on a fixed disagreement region. -/
def fullCorrectnessResidualMagnitudes
    {X : Type*} [MeasurableSpace X] (μ : Measure X) (D : Set X) (M beta : ℝ) : Set ℝ :=
  {r | ∃ η : CorrectnessField X,
    |disagreementMean μ D η - 1 / 2 - M| ≤ beta ∧
    r = |disagreementMean μ D η - 1 / 2 - M|}

/-- The printed Γ(C_beta)=beta claim for 0≤beta≤1/2 is proved by constructing
an admissible constant correctness field at the appropriate residual endpoint.
No equality of an unattained supremum is assumed. -/
theorem full_correctness_residual_radius_exact
    {X : Type*} [MeasurableSpace X] (μ : Measure X) [IsProbabilityMeasure μ]
    (D : Set X) (hD : 0 < μ.real D) {M beta : ℝ}
    (hMlo : -1 / 2 ≤ M) (hMhi : M ≤ 1 / 2)
    (hbeta₀ : 0 ≤ beta) (hbeta₁ : beta ≤ 1 / 2) :
    sSup (fullCorrectnessResidualMagnitudes μ D M beta) = beta := by
  have hatt : beta ∈ fullCorrectnessResidualMagnitudes μ D M beta := by
    by_cases hM : 0 ≤ M
    · have hp₀ : 0 ≤ M - beta + 1 / 2 := by linarith
      have hp₁ : M - beta + 1 / 2 ≤ 1 := by linarith
      have heq : |disagreementMean μ D
          (CorrectnessField.constant (M - beta + 1 / 2) hp₀ hp₁) - 1 / 2 - M| = beta := by
        rw [disagreementMean_constant μ D hD]
        have hid : M - beta + 1 / 2 - 1 / 2 - M = -beta := by ring
        rw [hid, abs_neg, abs_of_nonneg hbeta₀]
      exact ⟨_, heq.le, heq.symm⟩
    · have hMneg : M < 0 := lt_of_not_ge hM
      have hp₀ : 0 ≤ M + beta + 1 / 2 := by linarith
      have hp₁ : M + beta + 1 / 2 ≤ 1 := by linarith
      have heq : |disagreementMean μ D
          (CorrectnessField.constant (M + beta + 1 / 2) hp₀ hp₁) - 1 / 2 - M| = beta := by
        rw [disagreementMean_constant μ D hD]
        have hid : M + beta + 1 / 2 - 1 / 2 - M = beta := by ring
        rw [hid, abs_of_nonneg hbeta₀]
      exact ⟨_, heq.le, heq.symm⟩
  have hupper : ∀ r ∈ fullCorrectnessResidualMagnitudes μ D M beta, r ≤ beta := by
    rintro r ⟨η, hb, rfl⟩
    exact hb
  exact le_antisymm (csSup_le ⟨beta, hatt⟩ hupper) (le_csSup ⟨beta, hupper⟩ hatt)

end KBound
