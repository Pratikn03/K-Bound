import KBound.Frontier
import KBound.Impossibility
import KBound.Probability.MeasureCertificate

/-!
# Finite measurable target-law realization and distributional frontier lift

This module kernel-checks the canonical two-label construction used by the
paper's necessity proof.  The sample space is finite and discrete, so the
label-free evidence map is measurable without additional regularity axioms.
The final lift states the target-class richness premise explicitly.
-/

namespace KBound

open Decision

/-- The two possible label outcomes on a binary disagreement region. -/
inductive DisagreementAtom where
  | adaptedCorrect
  | sourceCorrect
  deriving DecidableEq, Fintype, Repr

instance : MeasurableSpace DisagreementAtom := ⊤

/-- A probability law on the finite disagreement-label space. -/
structure FiniteTargetLaw where
  mass : DisagreementAtom → ℝ
  mass_nonneg : ∀ a, 0 ≤ mass a
  mass_total : ∑ a, mass a = 1

/-- Positive-label world: the adapted predictor is correct with mass `1/2+δ`. -/
noncomputable def positiveTargetLaw (delta : ℝ) (hdelta0 : 0 ≤ delta)
    (hdeltaHalf : delta ≤ 1 / 2) : FiniteTargetLaw where
  mass
    | .adaptedCorrect => 1 / 2 + delta
    | .sourceCorrect => 1 / 2 - delta
  mass_nonneg := by
    intro a
    cases a <;> simp <;> linarith
  mass_total := by
    -- `simp` alone cannot expand a Fintype sum over a derived 2-constructor enum.
    have huniv : (Finset.univ : Finset DisagreementAtom) =
        {DisagreementAtom.adaptedCorrect, DisagreementAtom.sourceCorrect} := by decide
    rw [huniv, Finset.sum_pair (by decide)]
    ring

/-- Negative-label world: the adapted predictor is correct with mass `1/2-δ`. -/
noncomputable def negativeTargetLaw (delta : ℝ) (hdelta0 : 0 ≤ delta)
    (hdeltaHalf : delta ≤ 1 / 2) : FiniteTargetLaw where
  mass
    | .adaptedCorrect => 1 / 2 - delta
    | .sourceCorrect => 1 / 2 + delta
  mass_nonneg := by
    intro a
    cases a <;> simp <;> linarith
  mass_total := by
    -- `simp` alone cannot expand a Fintype sum over a derived 2-constructor enum.
    have huniv : (Finset.univ : Finset DisagreementAtom) =
        {DisagreementAtom.adaptedCorrect, DisagreementAtom.sourceCorrect} := by decide
    rw [huniv, Finset.sum_pair (by decide)]
    ring

/-- Label-free evidence is constant on the two label outcomes: it cannot read `Y`. -/
def finiteEvidence (_ : DisagreementAtom) : Unit := ()

/-- The finite discrete evidence map is measurable. -/
theorem finiteEvidence_measurable : Measurable finiteEvidence := by
  exact measurable_const

/-- Pushforward mass assigned to an evidence value. -/
noncomputable def evidenceMass (P : FiniteTargetLaw) (u : Unit) : ℝ :=
  ∑ a, if finiteEvidence a = u then P.mass a else 0

/-- Every valid finite target law induces the same unit evidence distribution. -/
theorem evidenceMass_eq_one (P : FiniteTargetLaw) (u : Unit) :
    evidenceMass P u = 1 := by
  cases u
  simpa [evidenceMass, finiteEvidence] using P.mass_total

/-- The positive and negative label worlds have identical label-free evidence laws. -/
theorem finite_target_laws_matched_evidence
    (delta : ℝ) (hdelta0 : 0 ≤ delta) (hdeltaHalf : delta ≤ 1 / 2) :
    evidenceMass (positiveTargetLaw delta hdelta0 hdeltaHalf) () =
      evidenceMass (negativeTargetLaw delta hdelta0 hdeltaHalf) () := by
  rw [evidenceMass_eq_one, evidenceMass_eq_one]

/-- Adaptation benefit of a finite disagreement law with disagreement mass `muD`. -/
noncomputable def lawBenefit (muD : ℝ) (P : FiniteTargetLaw) : ℝ :=
  2 * muD * (P.mass .adaptedCorrect - 1 / 2)

theorem positiveTargetLaw_benefit
    {muD delta : ℝ} (hdelta0 : 0 ≤ delta) (hdeltaHalf : delta ≤ 1 / 2) :
    lawBenefit muD (positiveTargetLaw delta hdelta0 hdeltaHalf) = 2 * muD * delta := by
  -- `simp` closes this outright; the trailing `ring` had no goal left to act on,
  -- which made Lean mark the whole declaration as `sorry`.
  simp [lawBenefit, positiveTargetLaw]

theorem negativeTargetLaw_benefit
    {muD delta : ℝ} (hdelta0 : 0 ≤ delta) (hdeltaHalf : delta ≤ 1 / 2) :
    lawBenefit muD (negativeTargetLaw delta hdelta0 hdeltaHalf) = -(2 * muD * delta) := by
  -- Same as the positive case: `simp` discharges it, the trailing `ring` did not.
  simp [lawBenefit, negativeTargetLaw]

/-- A target world packages a concrete law with its observable margin and latent drift. -/
structure TargetWorld where
  law : FiniteTargetLaw
  margin : ℝ
  drift : ℝ
  benefit : ℝ
  benefit_eq : benefit = margin + drift
  benefit_law : benefit = lawBenefit (1 / 2) law

def TargetWorld.Admissible (beta : ℝ) (w : TargetWorld) : Prop := |w.drift| ≤ beta

def SameEvidence (w₁ w₂ : TargetWorld) : Prop :=
  w₁.margin = w₂.margin ∧
    ∀ u, evidenceMass w₁.law u = evidenceMass w₂.law u

/-- Concrete positive/negative target worlds realizing the manuscript's finite
label-kernel construction.  The drift inequalities are stated explicitly so
class membership cannot be smuggled into the construction. -/
theorem finite_target_world_pair
    {M beta delta : ℝ} (hdelta0 : 0 < delta) (hdeltaHalf : delta ≤ 1 / 2)
    (hposDrift : |delta - M| ≤ beta) (hnegDrift : |-delta - M| ≤ beta) :
    ∃ wPos wNeg : TargetWorld,
      wPos.Admissible beta ∧ wNeg.Admissible beta ∧
      SameEvidence wPos wNeg ∧
      wPos.margin = M ∧ wNeg.margin = M ∧
      wPos.benefit = delta ∧ wNeg.benefit = -delta := by
  have hdeltaNonneg : 0 ≤ delta := le_of_lt hdelta0
  let pLaw := positiveTargetLaw delta hdeltaNonneg hdeltaHalf
  let nLaw := negativeTargetLaw delta hdeltaNonneg hdeltaHalf
  let wPos : TargetWorld := {
    law := pLaw
    margin := M
    drift := delta - M
    benefit := delta
    benefit_eq := by ring
    benefit_law := by
      dsimp [pLaw]
      rw [positiveTargetLaw_benefit hdeltaNonneg hdeltaHalf]
      ring
  }
  let wNeg : TargetWorld := {
    law := nLaw
    margin := M
    drift := -delta - M
    benefit := -delta
    benefit_eq := by ring
    benefit_law := by
      dsimp [nLaw]
      rw [negativeTargetLaw_benefit hdeltaNonneg hdeltaHalf]
      ring
  }
  refine ⟨wPos, wNeg, ?_, ?_, ?_, rfl, rfl, rfl, rfl⟩
  · exact hposDrift
  · exact hnegDrift
  · constructor
    · rfl
    · intro u
      rw [evidenceMass_eq_one, evidenceMass_eq_one]

/-- The declared class is rich at `(M,β)` when every allowed algebraic drift is
realized by a concrete target law in the same evidence fibre. -/
def RichAt (worlds : Set TargetWorld) (M beta : ℝ) : Prop :=
  ∀ gamma, |gamma| ≤ beta →
    ∃ w ∈ worlds, w.margin = M ∧ w.drift = gamma ∧ w.benefit = M + gamma

/-- Uniform directional soundness over a concrete target-world class. -/
def UniformlySoundOn (a : Decision) (worlds : Set TargetWorld) : Prop :=
  ∀ w ∈ worlds, SoundForBenefit a w.benefit

/-- Rich target classes make either strict action unsound throughout the closed band. -/
theorem rich_closed_band_forces_abstain
    {worlds : Set TargetWorld} {M beta : ℝ}
    (hrich : RichAt worlds M beta) (hband : |M| ≤ beta)
    {a : Decision} (hsound : UniformlySoundOn a worlds) :
    a = abstain := by
  obtain ⟨w, hw, _, _, hbenefit⟩ := hrich (-M) (by simpa [abs_neg] using hband)
  have hs := hsound w hw
  have hzero : w.benefit = 0 := by linarith
  cases a with
  | adapt =>
      have : 0 < w.benefit := hs.1 rfl
      linarith
  | freeze =>
      have : w.benefit < 0 := hs.2 rfl
      linarith
  | abstain => rfl

/-- Outside the band the frontier action is uniformly sound on every admissible world. -/
theorem frontierDecision_uniformly_sound
    {worlds : Set TargetWorld} {M beta : ℝ} (hbeta : 0 ≤ beta)
    (hfixed : ∀ w ∈ worlds, w.margin = M)
    (hadm : ∀ w ∈ worlds, w.Admissible beta)
    (hout : beta < |M|) :
    UniformlySoundOn (frontierDecision M beta) worlds := by
  intro w hw
  have hm := hfixed w hw
  have hd := hadm w hw
  constructor
  · intro ha
    have hMpos : 0 < M := by
      by_contra hn
      have hMnonpos : M ≤ 0 := le_of_not_gt hn
      have hfreeze : frontierDecision M beta = freeze := by
        apply frontier_decision_freeze hbeta
        rw [abs_of_nonpos hMnonpos] at hout
        linarith
      rw [hfreeze] at ha
      contradiction
    have hpos : 0 < M + w.drift := frontier_positive hMpos hout hd
    linarith [w.benefit_eq, hm]
  · intro hf
    have hMneg : M < 0 := by
      by_contra hn
      have hMnonneg : 0 ≤ M := le_of_not_gt hn
      have hadapt : frontierDecision M beta = adapt := by
        apply frontier_decision_adapt
        rw [abs_of_nonneg hMnonneg] at hout
        linarith
      rw [hadapt] at hf
      contradiction
    have hneg : M + w.drift < 0 := frontier_negative hMneg hout hd
    linarith [w.benefit_eq, hm]

/-- Distributional necessity/maximality under the exact richness premise. -/
theorem distributional_frontier_maximal
    {worlds : Set TargetWorld} {M beta : ℝ} (hbeta : 0 ≤ beta)
    (hrich : RichAt worlds M beta)
    (hfixed : ∀ w ∈ worlds, w.margin = M)
    (hadm : ∀ w ∈ worlds, w.Admissible beta) :
    (beta < |M| → UniformlySoundOn (frontierDecision M beta) worlds) ∧
    (|M| ≤ beta → ∀ a, UniformlySoundOn a worlds → a = abstain) := by
  constructor
  · exact frontierDecision_uniformly_sound hbeta hfixed hadm
  · intro hband a hsound
    exact rich_closed_band_forces_abstain hrich hband hsound

end KBound
