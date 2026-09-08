import KBound.Probability.ActualWorldEvidence

/-!
# The actual frontier on construction-closed restricted classes

Interior closure requires one legal positive/negative tilt pair, not a zero
world. Boundary closure requires the constant-one-half zero construction only
when |M| = beta, including M = beta = 0. Residuals and signs are derived from
the actual target laws, not assumed in the closure conditions. Outside the band,
sufficiency holds for every subclass. Individual directional iff statements
additionally expose nonemptiness; both signs otherwise hold vacuously.
-/

namespace KBound.ActualWorldSubclass

open MeasureTheory ProbabilityTheory Set JointKernelScore ActualWorldFrontier ActualWorldEvidence
open scoped ENNReal ProbabilityTheory

variable {X : Type*} [MeasurableSpace X]

noncomputable def zeroWorld (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] : ProbabilityMeasure (X × Bool) :=
  fieldWorld μ f0 fa h0 ha κ0 (CorrectnessField.constant (1 / 2) (by norm_num) (by norm_num))

def Tilt := {δ : ℝ // 0 < δ ∧ δ < 1 / 2}

noncomputable def tiltWorld (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (δ : Tilt) (positive : Bool) :
    ProbabilityMeasure (X × Bool) :=
  fieldWorld μ f0 fa h0 ha κ0 (CorrectnessField.constant
    (1 / 2 + if positive then δ.val else -δ.val)
    (by split_ifs <;> linarith [δ.property.1, δ.property.2])
    (by split_ifs <;> linarith [δ.property.1, δ.property.2]))

/-- Membership of one legal instance of the paper's two interior constructions.
No sign predicate, zero-world membership or full-field richness is assumed. -/
def InteriorClosed (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (beta : ℝ) (W : ProbabilityMeasure (X × Bool) → Prop) : Prop :=
  |scoreMargin μ {x | f0 x ≠ fa x} score| < beta →
    ∃ δ : Tilt, δ.val < beta - |scoreMargin μ {x | f0 x ≠ fa x} score| ∧
      W (tiltWorld μ f0 fa h0 ha κ0 δ true) ∧ W (tiltWorld μ f0 fa h0 ha κ0 δ false)

def BoundaryClosed (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (beta : ℝ) (W : ProbabilityMeasure (X × Bool) → Prop) : Prop :=
  |scoreMargin μ {x | f0 x ≠ fa x} score| = beta → W (zeroWorld μ f0 fa h0 ha κ0)

theorem zeroWorld_residual (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) :
    scoreResidual (zeroWorld μ f0 fa h0 ha κ0 : Measure (X × Bool)) f0 fa ha score =
      -scoreMargin μ {x | f0 x ≠ fa x} score := by
  unfold zeroWorld
  rw [fieldWorld_residual μ f0 fa h0 ha κ0 score _ hD,
    disagreementMean_constant μ _ hD]
  ring

theorem zeroWorld_benefit (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0]
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) :
    populationBenefit f0 fa (zeroWorld μ f0 fa h0 ha κ0 : Measure (X × Bool)) = 0 := by
  change populationBenefit f0 fa (correctnessFieldTarget μ f0 fa h0 ha κ0
    (CorrectnessField.constant (1 / 2) (by norm_num) (by norm_num))) = 0
  rw [correctnessFieldTarget_benefit μ f0 fa h0 ha κ0 hD,
    disagreementMean_constant μ _ hD]
  ring

/-- The closure target is an actual admissible target exactly on the closed band. -/
theorem zeroWorld_admissible_iff (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) :
    admissible μ f0 fa ha score beta (zeroWorld μ f0 fa h0 ha κ0) ↔
      |scoreMargin μ {x | f0 x ≠ fa x} score| ≤ beta := by
  have hf : (zeroWorld μ f0 fa h0 ha κ0 : Measure (X × Bool)).fst = μ :=
    fieldWorld_fst μ f0 fa h0 ha κ0 _
  simp only [admissible, hf, true_and, zeroWorld_residual μ f0 fa h0 ha κ0 score hD,
    abs_neg]

theorem tiltWorld_residual (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (δ : Tilt) (positive : Bool) :
    scoreResidual (tiltWorld μ f0 fa h0 ha κ0 δ positive : Measure (X × Bool)) f0 fa ha score =
      (if positive then δ.val else -δ.val) - scoreMargin μ {x | f0 x ≠ fa x} score := by
  unfold tiltWorld
  rw [fieldWorld_residual μ f0 fa h0 ha κ0 score _ hD, disagreementMean_constant μ _ hD]
  ring

theorem tiltWorld_benefit (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0]
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (δ : Tilt) (positive : Bool) :
    populationBenefit f0 fa (tiltWorld μ f0 fa h0 ha κ0 δ positive : Measure (X × Bool)) =
      2 * μ.real {x | f0 x ≠ fa x} * (if positive then δ.val else -δ.val) := by
  change populationBenefit f0 fa (correctnessFieldTarget μ f0 fa h0 ha κ0
    (CorrectnessField.constant _ _ _)) = _
  rw [correctnessFieldTarget_benefit μ f0 fa h0 ha κ0 hD, disagreementMean_constant μ _ hD]
  ring

theorem tiltWorld_admissible (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (δ : Tilt) (positive : Bool)
    (hgap : δ.val < beta - |scoreMargin μ {x | f0 x ≠ fa x} score|) :
    admissible μ f0 fa ha score beta (tiltWorld μ f0 fa h0 ha κ0 δ positive) := by
  refine ⟨fieldWorld_fst μ f0 fa h0 ha κ0 _, ?_⟩
  rw [tiltWorld_residual μ f0 fa h0 ha κ0 score hD]
  have hm := abs_le.mp (le_refl |scoreMargin μ {x | f0 x ≠ fa x} score|)
  split_ifs <;> apply abs_le.mpr <;> constructor <;> linarith [δ.property.1, hm.1, hm.2]

/-- The full actual class contains a legal pair, so the closure interface is
realized by genuine target laws whenever the band has nonzero interior. -/
theorem full_class_tilt_witnesses (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) :
    InteriorClosed μ f0 fa h0 ha κ0 score beta (admissible μ f0 fa ha score beta) := by
  intro hband
  have hm : 0 < min (beta - |scoreMargin μ {x | f0 x ≠ fa x} score|) (1 / 2 : ℝ) :=
    lt_min (by linarith) (by norm_num)
  obtain ⟨d, hd0, hdm⟩ := exists_between hm
  let δ : Tilt := ⟨d, hd0, (lt_min_iff.mp hdm).2⟩
  have hg : δ.val < beta - |scoreMargin μ {x | f0 x ≠ fa x} score| :=
    (lt_min_iff.mp hdm).1
  exact ⟨δ, hg, tiltWorld_admissible μ f0 fa h0 ha κ0 score hD beta δ true hg,
    tiltWorld_admissible μ f0 fa h0 ha κ0 score hD beta δ false hg⟩

/-- Interior pairs or the boundary zero construction supply both non-strict
obstructions. The two targets may coincide only in the boundary branch. -/
theorem subclass_closed_band_obstructions (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ)
    (W : ProbabilityMeasure (X × Bool) → Prop)
    (hi : InteriorClosed μ f0 fa h0 ha κ0 score beta W)
    (he : BoundaryClosed μ f0 fa h0 ha κ0 score beta W)
    (hband : |scoreMargin μ {x | f0 x ≠ fa x} score| ≤ beta) :
    ∃ Pneg Ppos : ProbabilityMeasure (X × Bool), W Pneg ∧ W Ppos ∧
      populationBenefit f0 fa (Pneg : Measure (X × Bool)) ≤ 0 ∧
      0 ≤ populationBenefit f0 fa (Ppos : Measure (X × Bool)) := by
  rcases lt_or_eq_of_le hband with hopen | hedge
  · obtain ⟨δ, _, hp, hn⟩ := hi hopen
    refine ⟨tiltWorld μ f0 fa h0 ha κ0 δ false, tiltWorld μ f0 fa h0 ha κ0 δ true,
      hn, hp, ?_, ?_⟩
    · rw [tiltWorld_benefit μ f0 fa h0 ha κ0 hD δ false]
      change 2 * μ.real {x | f0 x ≠ fa x} * (-δ.val) ≤ 0
      exact mul_nonpos_of_nonneg_of_nonpos (by positivity) (by linarith [δ.property.1])
    · rw [tiltWorld_benefit μ f0 fa h0 ha κ0 hD δ true]
      change 0 ≤ 2 * μ.real {x | f0 x ≠ fa x} * δ.val
      exact mul_nonneg (by positivity) δ.property.1.le
  · refine ⟨zeroWorld μ f0 fa h0 ha κ0, zeroWorld μ f0 fa h0 ha κ0,
      he hedge, he hedge, ?_, ?_⟩ <;> rw [zeroWorld_benefit μ f0 fa h0 ha κ0 hD]

/-- The closure conditions mention only the actual constructions, not freely
assigned risks or membership of every correctness-field target. -/
theorem subclass_strict_direction_iff (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta)
    (W : ProbabilityMeasure (X × Bool) → Prop)
    (hsub : ∀ P, W P → admissible μ f0 fa ha score beta P)
    (hi : InteriorClosed μ f0 fa h0 ha κ0 score beta W)
    (he : BoundaryClosed μ f0 fa h0 ha κ0 score beta W) :
    ((∀ P : ProbabilityMeasure (X × Bool), W P →
        0 < populationBenefit f0 fa (P : Measure (X × Bool))) ∨
      (∀ P : ProbabilityMeasure (X × Bool), W P →
        populationBenefit f0 fa (P : Measure (X × Bool)) < 0)) ↔
      beta < |scoreMargin μ {x | f0 x ≠ fa x} score| := by
  constructor
  · intro hs
    by_contra hnot
    obtain ⟨Pneg, Ppos, hn, hp, hneg, hpos⟩ :=
      subclass_closed_band_obstructions μ f0 fa h0 ha κ0 score hD beta W hi he (le_of_not_gt hnot)
    rcases hs with hp | hn
    · have h := hp Pneg hn
      linarith
    · have h := hn Ppos hp
      linarith
  · intro hmargin
    rcases lt_abs.mp hmargin with hp | hn
    · exact Or.inl (fun P hP =>
        (actual_frontier_adapt_iff μ f0 fa h0 ha κ0 score hD beta hb).mpr hp P (hsub P hP))
    · exact Or.inr (fun P hP =>
        (actual_frontier_freeze_iff μ f0 fa h0 ha κ0 score hD beta hb).mpr
          (by linarith) P (hsub P hP))

/-- Nonemptiness is explicit for individual direction iff: the empty subclass
would make both universal direction predicates true. -/
theorem subclass_adapt_iff_of_nonempty (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta)
    (W : ProbabilityMeasure (X × Bool) → Prop)
    (hsub : ∀ P, W P → admissible μ f0 fa ha score beta P)
    (hi : InteriorClosed μ f0 fa h0 ha κ0 score beta W)
    (he : BoundaryClosed μ f0 fa h0 ha κ0 score beta W) (hne : ∃ P, W P) :
    (∀ P : ProbabilityMeasure (X × Bool), W P →
      0 < populationBenefit f0 fa (P : Measure (X × Bool))) ↔
      beta < scoreMargin μ {x | f0 x ≠ fa x} score := by
  constructor
  · intro hp
    have ho := (subclass_strict_direction_iff μ f0 fa h0 ha κ0 score hD beta hb
      W hsub hi he).mp (Or.inl hp)
    rcases lt_abs.mp ho with hm | hm
    · exact hm
    · obtain ⟨P, hP⟩ := hne
      have hn := (actual_frontier_freeze_iff μ f0 fa h0 ha κ0 score hD beta hb).mpr
        (by linarith) P (hsub P hP)
      have hpos := hp P hP
      linarith
  · exact fun hm P hP =>
      (actual_frontier_adapt_iff μ f0 fa h0 ha κ0 score hD beta hb).mpr hm P (hsub P hP)

theorem subclass_freeze_iff_of_nonempty (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta)
    (W : ProbabilityMeasure (X × Bool) → Prop)
    (hsub : ∀ P, W P → admissible μ f0 fa ha score beta P)
    (hi : InteriorClosed μ f0 fa h0 ha κ0 score beta W)
    (he : BoundaryClosed μ f0 fa h0 ha κ0 score beta W) (hne : ∃ P, W P) :
    (∀ P : ProbabilityMeasure (X × Bool), W P →
      populationBenefit f0 fa (P : Measure (X × Bool)) < 0) ↔
      scoreMargin μ {x | f0 x ≠ fa x} score < -beta := by
  constructor
  · intro hn
    have ho := (subclass_strict_direction_iff μ f0 fa h0 ha κ0 score hD beta hb
      W hsub hi he).mp (Or.inr hn)
    rcases lt_abs.mp ho with hm | hm
    · obtain ⟨P, hP⟩ := hne
      have hp := (actual_frontier_adapt_iff μ f0 fa h0 ha κ0 score hD beta hb).mpr
        hm P (hsub P hP)
      have hneg := hn P hP
      linarith
    · linarith
  · exact fun hm P hP =>
      (actual_frontier_freeze_iff μ f0 fa h0 ha κ0 score hD beta hb).mpr hm P (hsub P hP)

/-- Soundness of each printed commitment, and abstention iff no strict direction
is sound, need no nonemptiness premise beyond the concrete band closure. -/
theorem subclass_pointwise_maximal_rule (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta)
    (W : ProbabilityMeasure (X × Bool) → Prop)
    (hsub : ∀ P, W P → admissible μ f0 fa ha score beta P)
    (hi : InteriorClosed μ f0 fa h0 ha κ0 score beta W)
    (he : BoundaryClosed μ f0 fa h0 ha κ0 score beta W) :
    (frontierDecision (scoreMargin μ {x | f0 x ≠ fa x} score) beta = Decision.adapt →
      ∀ P : ProbabilityMeasure (X × Bool), W P →
        0 < populationBenefit f0 fa (P : Measure (X × Bool))) ∧
    (frontierDecision (scoreMargin μ {x | f0 x ≠ fa x} score) beta = Decision.freeze →
      ∀ P : ProbabilityMeasure (X × Bool), W P →
        populationBenefit f0 fa (P : Measure (X × Bool)) < 0) ∧
    (frontierDecision (scoreMargin μ {x | f0 x ≠ fa x} score) beta = Decision.abstain ↔
      ¬((∀ P : ProbabilityMeasure (X × Bool), W P →
          0 < populationBenefit f0 fa (P : Measure (X × Bool))) ∨
        (∀ P : ProbabilityMeasure (X × Bool), W P →
          populationBenefit f0 fa (P : Measure (X × Bool)) < 0))) := by
  obtain ⟨hpa, hpf, hpb⟩ := actual_pointwise_maximal_rule μ f0 fa h0 ha κ0 score hD beta hb
  refine ⟨fun hd P hP => hpa.mp hd P (hsub P hP),
    fun hd P hP => hpf.mp hd P (hsub P hP), ?_⟩
  rw [subclass_strict_direction_iff μ f0 fa h0 ha κ0 score hD beta hb W hsub hi he,
    not_lt]
  exact hpb

/-- The restricted-class iff on a fixed actual augmented-evidence-law fibre. -/
theorem subclass_augmented_fibre_strict_direction_iff {Z : Type*} [MeasurableSpace Z]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 ≤ beta)
    (W : ProbabilityMeasure (X × Bool) → Prop)
    (hsub : ∀ P, W P → admissible μ f0 fa ha score beta P)
    (hi : InteriorClosed μ f0 fa h0 ha κ0 score beta W)
    (he : BoundaryClosed μ f0 fa h0 ha κ0 score beta W)
    (Q : ProbabilityMeasure (X × Bool)) (hQ : (Q : Measure (X × Bool)).fst = μ) :
    ((∀ P : ProbabilityMeasure (X × Bool), W P ∧
        augmentedLaw observation f0 fa score P = augmentedLaw observation f0 fa score Q →
        0 < populationBenefit f0 fa (P : Measure (X × Bool))) ∨
      (∀ P : ProbabilityMeasure (X × Bool), W P ∧
        augmentedLaw observation f0 fa score P = augmentedLaw observation f0 fa score Q →
        populationBenefit f0 fa (P : Measure (X × Bool)) < 0)) ↔
      beta < |scoreMargin μ {x | f0 x ≠ fa x} score| := by
  have heq (P : ProbabilityMeasure (X × Bool)) (hP : W P) :=
    augmentedLaw_eq observation f0 fa score ((hsub P hP).1.trans hQ.symm)
  rw [← subclass_strict_direction_iff μ f0 fa h0 ha κ0 score hD beta hb W hsub hi he]
  constructor
  · rintro (hp | hn)
    · exact Or.inl (fun P hP => hp P ⟨hP, heq P hP⟩)
    · exact Or.inr (fun P hP => hn P ⟨hP, heq P hP⟩)
  · rintro (hp | hn)
    · exact Or.inl (fun P hP => hp P hP.1)
    · exact Or.inr (fun P hP => hn P hP.1)

/-- Restricted uniform event control implies the same closed-band abstention
bound, even when the interior subclass contains no zero-benefit world. -/
theorem subclass_closed_band_abstention {Z : Type*} [MeasurableSpace Z]
    [MeasurableSpace Decision] [MeasurableSingletonClass Decision]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (policy : Kernel (Z × ℝ) Decision) [IsMarkovKernel policy]
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ)
    (W : ProbabilityMeasure (X × Bool) → Prop)
    (hsub : ∀ P, W P → admissible μ f0 fa ha score beta P)
    (hi : InteriorClosed μ f0 fa h0 ha κ0 score beta W)
    (he : BoundaryClosed μ f0 fa h0 ha κ0 score beta W)
    (hband : |scoreMargin μ {x | f0 x ≠ fa x} score| ≤ beta) (alpha : ENNReal)
    (hcontrol : ∀ P : ProbabilityMeasure (X × Bool), W P →
      (policy ∘ₘ (augmentedLaw observation f0 fa score P : Measure (Z × ℝ)))
        {a | a = Decision.adapt ∧ populationBenefit f0 fa (P : Measure (X × Bool)) ≤ 0} ≤ alpha ∧
      (policy ∘ₘ (augmentedLaw observation f0 fa score P : Measure (Z × ℝ)))
        {a | a = Decision.freeze ∧ 0 ≤ populationBenefit f0 fa (P : Measure (X × Bool))} ≤ alpha)
    (P : ProbabilityMeasure (X × Bool)) (hP : W P) :
    1 - 2 * alpha ≤ (policy ∘ₘ
      (augmentedLaw observation f0 fa score P : Measure (Z × ℝ))) {Decision.abstain} := by
  obtain ⟨Pneg, Ppos, hn, hp, hneg, hpos⟩ :=
    subclass_closed_band_obstructions μ f0 fa h0 ha κ0 score hD beta W hi he hband
  have ha := (hcontrol Pneg hn).1
  have hf := (hcontrol Ppos hp).2
  simp only [hneg, and_true, setOf_eq_eq_singleton] at ha
  simp only [hpos, and_true, setOf_eq_eq_singleton] at hf
  have hen := augmentedLaw_eq observation f0 fa score
    ((hsub P hP).1.trans (hsub Pneg hn).1.symm)
  have hep := augmentedLaw_eq observation f0 fa score
    ((hsub P hP).1.trans (hsub Ppos hp).1.symm)
  apply RandomizedActionLaw.abstention_lower_bound
  · rw [hen]
    exact ha
  · rw [hep]
    exact hf

end KBound.ActualWorldSubclass

#print axioms KBound.ActualWorldSubclass.zeroWorld_residual
#print axioms KBound.ActualWorldSubclass.zeroWorld_benefit
#print axioms KBound.ActualWorldSubclass.zeroWorld_admissible_iff
#print axioms KBound.ActualWorldSubclass.tiltWorld_residual
#print axioms KBound.ActualWorldSubclass.tiltWorld_benefit
#print axioms KBound.ActualWorldSubclass.tiltWorld_admissible
#print axioms KBound.ActualWorldSubclass.full_class_tilt_witnesses
#print axioms KBound.ActualWorldSubclass.subclass_closed_band_obstructions
#print axioms KBound.ActualWorldSubclass.subclass_strict_direction_iff
#print axioms KBound.ActualWorldSubclass.subclass_adapt_iff_of_nonempty
#print axioms KBound.ActualWorldSubclass.subclass_freeze_iff_of_nonempty
#print axioms KBound.ActualWorldSubclass.subclass_pointwise_maximal_rule
#print axioms KBound.ActualWorldSubclass.subclass_augmented_fibre_strict_direction_iff
#print axioms KBound.ActualWorldSubclass.subclass_closed_band_abstention
