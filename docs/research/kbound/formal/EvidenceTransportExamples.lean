import KBound
import Mathlib.Probability.Kernel.Composition.MeasureComp

open MeasureTheory ProbabilityTheory
open scoped ProbabilityTheory

-- The old interface assumes a common action law. These regressions require
-- its derivation from actual observation laws and a fixed randomized rule.
#check KBound.EvidenceTransport.iid_input_batch_law
#check KBound.EvidenceTransport.independent_seed_joint_law

example {Z : Type*} [MeasurableSpace Z]
    [MeasurableSpace KBound.Decision] [MeasurableSingletonClass KBound.Decision]
    {μneg μpos : Measure Z} [IsProbabilityMeasure μneg] [IsProbabilityMeasure μpos]
    (policy : Kernel Z KBound.Decision) [IsMarkovKernel policy]
    (hobs : μneg = μpos) {dneg dpos : ℝ} {alpha : ENNReal}
    (hn : dneg < 0) (hp : 0 < dpos)
    (ha : (policy ∘ₘ μneg) {a | a = KBound.Decision.adapt ∧ dneg ≤ 0} ≤ alpha)
    (hf : (policy ∘ₘ μpos) {a | a = KBound.Decision.freeze ∧ 0 ≤ dpos} ≤ alpha) :
    1 - 2 * alpha ≤ (policy ∘ₘ μneg) {KBound.Decision.abstain} ∧
    1 - 2 * alpha ≤ (policy ∘ₘ μpos) {KBound.Decision.abstain} := by
  exact KBound.EvidenceTransport.kernel_randomized_abstention policy hobs hn hp ha hf

example {X Y Z : Type*} [MeasurableSpace X] [MeasurableSpace Y] [MeasurableSpace Z]
    {P Q : Measure (X × Y)} [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (hinput : P.fst = Q.fst) (evidence : (Fin 7 → X) → Z) (he : Measurable evidence) :
    (Measure.pi (fun _ : Fin 7 => P)).map (fun b => evidence (fun i => (b i).1)) =
    (Measure.pi (fun _ : Fin 7 => Q)).map (fun b => evidence (fun i => (b i).1)) :=
  KBound.EvidenceTransport.iid_batch_observable_law hinput evidence he

-- The seed is independent of each world experiment; its marginal alone would
-- not justify these product-space action-law equalities.
example {Ω₁ Ω₂ Z U : Type*} [MeasurableSpace Ω₁] [MeasurableSpace Ω₂]
    [MeasurableSpace Z] [MeasurableSpace U]
    [MeasurableSpace KBound.Decision] [MeasurableSingletonClass KBound.Decision]
    {μ₁ : Measure Ω₁} {μ₂ : Measure Ω₂} [IsProbabilityMeasure μ₁] [IsProbabilityMeasure μ₂]
    (seed : Measure U) [IsProbabilityMeasure seed] {e₁ : Ω₁ → Z} {e₂ : Ω₂ → Z}
    (he₁ : Measurable e₁) (he₂ : Measurable e₂)
    (rule : Z × U → KBound.Decision) (hr : Measurable rule)
    (hobs : μ₁.map e₁ = μ₂.map e₂) :
    (μ₁.prod seed).map (fun ωu => rule (e₁ ωu.1, ωu.2)) =
    (μ₂.prod seed).map (fun ωu => rule (e₂ ωu.1, ωu.2)) :=
  KBound.EvidenceTransport.independent_seed_action_law seed he₁ he₂ rule hr hobs
