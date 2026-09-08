import KBound.Probability.RandomizedActionLaw
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Probability.Kernel.Composition.MeasureComp

/-!
# From common label-free evidence to common randomized action laws

The policy is fixed across worlds. A Markov policy kernel may depend on the
observed evidence, but not on an unobserved world identifier. Alternatively,
randomness is an independent seed with the same law in both worlds. Equal seed
marginals without the product/common-joint-law premise are not enough.

The finite iid batch theorem is a precise sampling-law lift. It does not infer
arbitrary batch dependence from equality of one-input marginals. For a specified
common non-iid batch law, the general pushforward composition theorem applies.
-/

namespace KBound.EvidenceTransport

open MeasureTheory ProbabilityTheory
open scoped ProbabilityTheory

theorem common_observable_pushforward
    {Ω₁ Ω₂ Z A : Type*} [MeasurableSpace Ω₁] [MeasurableSpace Ω₂]
    [MeasurableSpace Z] [MeasurableSpace A]
    {μ₁ : Measure Ω₁} {μ₂ : Measure Ω₂} {e₁ : Ω₁ → Z} {e₂ : Ω₂ → Z}
    (he₁ : Measurable e₁) (he₂ : Measurable e₂) {rule : Z → A}
    (hr : Measurable rule) (hobs : μ₁.map e₁ = μ₂.map e₂) :
    μ₁.map (rule ∘ e₁) = μ₂.map (rule ∘ e₂) := by
  rw [← Measure.map_map hr he₁, ← Measure.map_map hr he₂, hobs]

/-- Label-kernel changes cannot affect any input-batch distribution under the
declared finite iid sampling design. -/
theorem iid_input_batch_law {X Y ι : Type*} [MeasurableSpace X] [MeasurableSpace Y]
    [Fintype ι] {P Q : Measure (X × Y)} [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (hinput : P.fst = Q.fst) :
    (Measure.pi (fun _ : ι => P)).map (fun batch i => (batch i).1) =
      (Measure.pi (fun _ : ι => Q)).map (fun batch i => (batch i).1) := by
  letI : IsProbabilityMeasure (P.map Prod.fst) :=
    Measure.isProbabilityMeasure_map measurable_fst.aemeasurable
  letI : IsProbabilityMeasure (Q.map Prod.fst) :=
    Measure.isProbabilityMeasure_map measurable_fst.aemeasurable
  rw [Measure.pi_map_pi (fun _ => measurable_fst.aemeasurable),
    Measure.pi_map_pi (fun _ => measurable_fst.aemeasurable)]
  change Measure.pi (fun _ : ι => P.fst) = Measure.pi (fun _ : ι => Q.fst)
  rw [hinput]

theorem iid_batch_observable_law {X Y Z ι : Type*}
    [MeasurableSpace X] [MeasurableSpace Y] [MeasurableSpace Z] [Fintype ι]
    {P Q : Measure (X × Y)} [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (hinput : P.fst = Q.fst) (evidence : (ι → X) → Z) (he : Measurable evidence) :
    (Measure.pi (fun _ : ι => P)).map (fun batch => evidence (fun i => (batch i).1)) =
      (Measure.pi (fun _ : ι => Q)).map (fun batch => evidence (fun i => (batch i).1)) := by
  have hbatch : Measurable (fun batch : ι → X × Y => fun i => (batch i).1) :=
    measurable_pi_lambda _ fun i => measurable_fst.comp (measurable_pi_apply i)
  exact common_observable_pushforward hbatch hbatch he (iid_input_batch_law hinput)

theorem independent_seed_joint_law {Ω₁ Ω₂ Z U : Type*}
    [MeasurableSpace Ω₁] [MeasurableSpace Ω₂] [MeasurableSpace Z] [MeasurableSpace U]
    {μ₁ : Measure Ω₁} {μ₂ : Measure Ω₂} [SFinite μ₁] [SFinite μ₂]
    (seed : Measure U) [SFinite seed] {e₁ : Ω₁ → Z} {e₂ : Ω₂ → Z}
    (he₁ : Measurable e₁) (he₂ : Measurable e₂) (hobs : μ₁.map e₁ = μ₂.map e₂) :
    (μ₁.prod seed).map (fun ωu => (e₁ ωu.1, ωu.2)) =
      (μ₂.prod seed).map (fun ωu => (e₂ ωu.1, ωu.2)) := by
  change (μ₁.prod seed).map (Prod.map e₁ id) = (μ₂.prod seed).map (Prod.map e₂ id)
  rw [← Measure.map_prod_map μ₁ seed he₁ measurable_id,
    ← Measure.map_prod_map μ₂ seed he₂ measurable_id, hobs]

theorem independent_seed_action_law {Ω₁ Ω₂ Z U A : Type*}
    [MeasurableSpace Ω₁] [MeasurableSpace Ω₂] [MeasurableSpace Z]
    [MeasurableSpace U] [MeasurableSpace A]
    {μ₁ : Measure Ω₁} {μ₂ : Measure Ω₂} [SFinite μ₁] [SFinite μ₂]
    (seed : Measure U) [SFinite seed] {e₁ : Ω₁ → Z} {e₂ : Ω₂ → Z}
    (he₁ : Measurable e₁) (he₂ : Measurable e₂)
    (rule : Z × U → A) (hr : Measurable rule) (hobs : μ₁.map e₁ = μ₂.map e₂) :
    (μ₁.prod seed).map (fun ωu => rule (e₁ ωu.1, ωu.2)) =
      (μ₂.prod seed).map (fun ωu => rule (e₂ ωu.1, ωu.2)) :=
  common_observable_pushforward ((he₁.comp measurable_fst).prodMk measurable_snd)
    ((he₂.comp measurable_fst).prodMk measurable_snd) hr
    (independent_seed_joint_law seed he₁ he₂ hobs)

theorem kernel_action_law {Z A : Type*} [MeasurableSpace Z] [MeasurableSpace A]
    (policy : Kernel Z A) {μ₁ μ₂ : Measure Z} (hobs : μ₁ = μ₂) :
    policy ∘ₘ μ₁ = policy ∘ₘ μ₂ := by rw [hobs]

section Decisions

variable [MeasurableSpace Decision] [MeasurableSingletonClass Decision]

/-- General fixed randomized policy, represented by a Markov kernel rather than
assuming an already-equal action law. -/
theorem kernel_randomized_abstention {Z : Type*} [MeasurableSpace Z]
    {μneg μpos : Measure Z} [IsProbabilityMeasure μneg] [IsProbabilityMeasure μpos]
    (policy : Kernel Z Decision) [IsMarkovKernel policy]
    (hobs : μneg = μpos) {dneg dpos : ℝ} {alpha : ENNReal}
    (hn : dneg < 0) (hp : 0 < dpos)
    (ha : (policy ∘ₘ μneg) {a | a = Decision.adapt ∧ dneg ≤ 0} ≤ alpha)
    (hf : (policy ∘ₘ μpos) {a | a = Decision.freeze ∧ 0 ≤ dpos} ≤ alpha) :
    1 - 2 * alpha ≤ (policy ∘ₘ μneg) {Decision.abstain} ∧
    1 - 2 * alpha ≤ (policy ∘ₘ μpos) {Decision.abstain} :=
  RandomizedActionLaw.common_law_forces_abstention hn hp
    (kernel_action_law policy hobs) ha hf

/-- The independent-seed realization measures the actual error events in the
two world/seed product spaces, not free action-probability variables. -/
theorem independent_seed_randomized_abstention {Ωneg Ωpos Z U : Type*}
    [MeasurableSpace Ωneg] [MeasurableSpace Ωpos] [MeasurableSpace Z] [MeasurableSpace U]
    {μneg : Measure Ωneg} {μpos : Measure Ωpos}
    [IsProbabilityMeasure μneg] [IsProbabilityMeasure μpos]
    (seed : Measure U) [IsProbabilityMeasure seed]
    {eneg : Ωneg → Z} {epos : Ωpos → Z} (hen : Measurable eneg) (hep : Measurable epos)
    (rule : Z × U → Decision) (hr : Measurable rule)
    (hobs : μneg.map eneg = μpos.map epos) {dneg dpos : ℝ} {alpha : ENNReal}
    (hn : dneg < 0) (hp : 0 < dpos)
    (ha : (μneg.prod seed) {ωu | rule (eneg ωu.1, ωu.2) = Decision.adapt ∧ dneg ≤ 0} ≤ alpha)
    (hf : (μpos.prod seed) {ωu | rule (epos ωu.1, ωu.2) = Decision.freeze ∧ 0 ≤ dpos} ≤ alpha) :
    1 - 2 * alpha ≤ (μneg.prod seed) {ωu | rule (eneg ωu.1, ωu.2) = Decision.abstain} ∧
    1 - 2 * alpha ≤ (μpos.prod seed) {ωu | rule (epos ωu.1, ωu.2) = Decision.abstain} :=
  RandomizedActionLaw.randomized_rules_force_abstention
    (hr.comp ((hen.comp measurable_fst).prodMk measurable_snd))
    (hr.comp ((hep.comp measurable_fst).prodMk measurable_snd)) hn hp
    (independent_seed_action_law seed hen hep rule hr hobs) ha hf

end Decisions
end KBound.EvidenceTransport
