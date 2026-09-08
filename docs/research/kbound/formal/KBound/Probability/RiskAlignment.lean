import KBound.Probability.ActualWorldEvidence

/-! # Risk alignment is weaker than strict identification

The definitions separate exclusion of opposite nonzero signs at equal evidence
from a uniform strict direction. The actual-world capstone uses the full
scoreResidual-defined class at M=beta>0: all benefits are nonnegative, but zero
and positive actual target laws have the same fixed augmented-evidence law.
No empirical risk-alignment claim is made.
-/

namespace KBound.RiskAlignment

open MeasureTheory ProbabilityTheory JointKernelScore ActualWorldFrontier ActualWorldEvidence
open scoped ENNReal ProbabilityTheory

def Aligned {World Evidence : Type*} (C : World → Prop)
    (evidence : World → Evidence) (benefit : World → ℝ) : Prop :=
  ∀ P Q, C P → C Q → evidence P = evidence Q → ¬ (benefit P < 0 ∧ 0 < benefit Q)

def StrictDirection {World : Type*} (C : World → Prop) (benefit : World → ℝ) : Prop :=
  (∀ P, C P → 0 < benefit P) ∨ (∀ P, C P → benefit P < 0)

theorem strict_implies_aligned {World Evidence : Type*} (C : World → Prop)
    (evidence : World → Evidence) (benefit : World → ℝ)
    (h : StrictDirection C benefit) : Aligned C evidence benefit := by
  intro P Q hP hQ _ hbad
  rcases h with hp | hn
  · exact (lt_asymm (hp P hP)) hbad.1
  · exact (lt_asymm (hn Q hQ)) hbad.2

theorem aligned_of_nonnegative {World Evidence : Type*} (C : World → Prop)
    (evidence : World → Evidence) (benefit : World → ℝ)
    (h : ∀ P, C P → 0 ≤ benefit P) : Aligned C evidence benefit := by
  intro P _ hP _ _ hbad
  exact (not_lt_of_ge (h P hP)) hbad.1

theorem not_strict_of_zero {World : Type*} (C : World → Prop) (benefit : World → ℝ)
    (Pzero : World) (hzero : C Pzero) (hz : benefit Pzero = 0) :
    ¬ StrictDirection C benefit := by
  rintro (hp | hn)
  · have h := hp Pzero hzero
    rw [hz] at h
    exact (lt_irrefl (0 : ℝ)) h
  · have h := hn Pzero hzero
    rw [hz] at h
    exact (lt_irrefl (0 : ℝ)) h

theorem actual_positive_boundary_nonnegative {X : Type*} [MeasurableSpace X]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (score : CorrectnessField X) (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ)
    (hm : scoreMargin μ {x | f0 x ≠ fa x} score = beta)
    (P : ProbabilityMeasure (X × Bool)) (hP : admissible μ f0 fa ha score beta P) :
    0 ≤ populationBenefit f0 fa (P : Measure (X × Bool)) := by
  obtain ⟨hf, hg⟩ := hP
  rw [joint_score_benefit (P : Measure (X × Bool)) f0 fa h0 ha score
    (by rw [hf]; exact hD), hf, hm]
  exact mul_nonneg (by positivity) (by linarith [(abs_le.mp hg).1])

theorem actual_boundary_risk_aligned_not_strict {X Z : Type*}
    [MeasurableSpace X] [MeasurableSpace Z]
    (μ : Measure X) [IsProbabilityMeasure μ]
    (f0 fa : X → Bool) (h0 : Measurable f0) (ha : Measurable fa)
    (κ0 : Kernel X Bool) [IsMarkovKernel κ0] (score : CorrectnessField X)
    (observation : ProbabilityMeasure X → ProbabilityMeasure Z)
    (hD : 0 < μ.real {x | f0 x ≠ fa x}) (beta : ℝ) (hb : 0 < beta)
    (hm : scoreMargin μ {x | f0 x ≠ fa x} score = beta) :
    Aligned (admissible μ f0 fa ha score beta) (augmentedLaw observation f0 fa score)
        (fun P => populationBenefit f0 fa (P : Measure (X × Bool))) ∧
      ¬ StrictDirection (admissible μ f0 fa ha score beta)
        (fun P => populationBenefit f0 fa (P : Measure (X × Bool))) ∧
      ∃ Pzero Ppos : ProbabilityMeasure (X × Bool),
        admissible μ f0 fa ha score beta Pzero ∧ admissible μ f0 fa ha score beta Ppos ∧
        populationBenefit f0 fa (Pzero : Measure (X × Bool)) = 0 ∧
        0 < populationBenefit f0 fa (Ppos : Measure (X × Bool)) ∧
        augmentedLaw observation f0 fa score Pzero = augmentedLaw observation f0 fa score Ppos := by
  have hband : |scoreMargin μ {x | f0 x ≠ fa x} score| ≤ beta := by
    rw [hm, abs_of_pos hb]
  obtain ⟨Pzero, hzero, hz⟩ := actual_closed_band_zero_target μ f0 fa h0 ha κ0 score hD beta hband
  have bounds := scoreMargin_bounds μ {x | f0 x ≠ fa x} score hD
  have hi : max (-1 / 2) (scoreMargin μ {x | f0 x ≠ fa x} score - beta) ≤ beta ∧
      beta ≤ min (1 / 2) (scoreMargin μ {x | f0 x ≠ fa x} score + beta) := by
    rw [hm]
    constructor
    · exact max_le (by linarith) (by linarith)
    · exact le_min (by linarith [bounds.2]) (by linarith)
  obtain ⟨Ppos, hpos, hp⟩ := (actual_identified_interval μ f0 fa h0 ha κ0 score hD beta beta).mpr hi
  refine ⟨aligned_of_nonnegative _ _ _ (fun P hP =>
      actual_positive_boundary_nonnegative μ f0 fa h0 ha score hD beta hm P hP),
    not_strict_of_zero _ _ Pzero hzero hz, Pzero, Ppos, hzero, hpos, hz, ?_, ?_⟩
  · rw [hp]
    exact mul_pos (mul_pos (by norm_num) hD) hb
  · exact augmentedLaw_eq observation f0 fa score (hzero.1.trans hpos.1.symm)

end KBound.RiskAlignment

#print axioms KBound.RiskAlignment.strict_implies_aligned
#print axioms KBound.RiskAlignment.aligned_of_nonnegative
#print axioms KBound.RiskAlignment.not_strict_of_zero
#print axioms KBound.RiskAlignment.actual_positive_boundary_nonnegative
#print axioms KBound.RiskAlignment.actual_boundary_risk_aligned_not_strict
