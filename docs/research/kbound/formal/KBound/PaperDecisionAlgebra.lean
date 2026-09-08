import KBound.Certificate
import Mathlib.Algebra.Order.Group.MinMax
import Mathlib.Tactic.NormNum

/-! # Score-defined regret and interval branch consequences in the current paper

These are deterministic identities/monotonicity claims, not statistical or
empirical assertions. Both FREEZE and ABSTAIN serve the frozen score.
-/

namespace KBound.PaperDecisionAlgebra

noncomputable def servedScore (frozen adapted : ℝ) (action : Decision) : ℝ :=
  if action = Decision.adapt then adapted else frozen

noncomputable def oracleRegret (frozen adapted : ℝ) (action : Decision) : ℝ :=
  max frozen adapted - servedScore frozen adapted action

theorem frozen_oracle_regret (frozen adapted : ℝ) :
    oracleRegret frozen adapted Decision.freeze = max (adapted - frozen) 0 := by
  simp only [oracleRegret, servedScore, reduceCtorEq, if_false]
  rw [← max_sub_sub_right]
  simp [max_comm]

theorem adapted_oracle_regret (frozen adapted : ℝ) :
    oracleRegret frozen adapted Decision.adapt = max (-(adapted - frozen)) 0 := by
  simp only [oracleRegret, servedScore, if_true]
  rw [← max_sub_sub_right]
  simp [neg_sub]

theorem abstain_regret_eq_frozen (frozen adapted : ℝ) :
    oracleRegret frozen adapted Decision.abstain = oracleRegret frozen adapted Decision.freeze := by
  simp [oracleRegret, servedScore]

theorem equal_score_zero_regret (score : ℝ) (action : Decision) :
    oracleRegret score score action = 0 := by
  simp [oracleRegret, servedScore]

theorem adapt_iff (estimate radius : ℝ) :
    certificate estimate radius = Decision.adapt ↔ 0 < estimate - radius := by
  unfold certificate
  split_ifs <;> simp_all

theorem freeze_iff (estimate radius : ℝ) (hradius : 0 ≤ radius) :
    certificate estimate radius = Decision.freeze ↔ estimate + radius < 0 := by
  unfold certificate
  split_ifs with ha hf
  · have hn : ¬estimate + radius < 0 := by linarith
    simp [hn]
  · simp [hf]
  · simp [hf]

theorem larger_radius_adapt_subset {estimate small large : ℝ} (h : small ≤ large)
    (ha : certificate estimate large = Decision.adapt) :
    certificate estimate small = Decision.adapt := by
  apply (adapt_iff _ _).mpr
  have hb := (adapt_iff _ _).mp ha
  linarith

theorem larger_radius_freeze_subset {estimate small large : ℝ}
    (hs : 0 ≤ small) (h : small ≤ large)
    (ha : certificate estimate large = Decision.freeze) :
    certificate estimate small = Decision.freeze := by
  apply (freeze_iff _ _ hs).mpr
  have hb := (freeze_iff _ _ (hs.trans h)).mp ha
  linarith

theorem larger_radius_commit_subset {estimate small large : ℝ}
    (hs : 0 ≤ small) (h : small ≤ large)
    (hc : certificate estimate large ≠ Decision.abstain) :
    certificate estimate small ≠ Decision.abstain := by
  cases ha : certificate estimate large with
  | adapt => rw [larger_radius_adapt_subset h ha]; decide
  | freeze => rw [larger_radius_freeze_subset hs h ha]; decide
  | abstain => exact False.elim (hc ha)

/-- In a helpful-only panel no routing between the two fixed scores can beat
the always-adapt score on any cell, hence none can improve a positive-weight mean. -/
theorem helpful_only_no_score_improvement {frozen adapted : ℝ} (h : frozen ≤ adapted)
    (action : Decision) : servedScore frozen adapted action ≤ adapted := by
  unfold servedScore
  split_ifs
  · exact le_rfl
  · exact h

theorem radius_expansion_can_increase_regret :
    oracleRegret 0 1 (certificate 1 0) = 0 ∧
    oracleRegret 0 1 (certificate 1 2) = 1 := by
  norm_num [oracleRegret, servedScore, certificate]
  decide

theorem printed_numeric_branches :
    certificate (4 / 100) (1 / 100) = Decision.adapt ∧
    certificate (-(4 / 100)) (1 / 100) = Decision.freeze ∧
    certificate (5 / 1000) (1 / 100) = Decision.abstain := by
  norm_num [certificate]

/-- The interval assumptions alone always certify the coarse residual bound1.
Thus an unqualified claim that *no* label-free bound can be verified is false;
the valid audit-floor obstruction concerns bounds below the fibre radius. -/
theorem universal_unit_residual_bound {correctnessMean scoreMean : ℝ}
    (ha0 : 0 ≤ correctnessMean) (ha1 : correctnessMean ≤ 1)
    (hs0 : 0 ≤ scoreMean) (hs1 : scoreMean ≤ 1) :
    |correctnessMean - scoreMean| ≤ 1 := by
  apply abs_le.mpr
  constructor <;> linarith

end KBound.PaperDecisionAlgebra

#print axioms KBound.PaperDecisionAlgebra.frozen_oracle_regret
#print axioms KBound.PaperDecisionAlgebra.adapted_oracle_regret
#print axioms KBound.PaperDecisionAlgebra.larger_radius_commit_subset
#print axioms KBound.PaperDecisionAlgebra.helpful_only_no_score_improvement
#print axioms KBound.PaperDecisionAlgebra.radius_expansion_can_increase_regret
#print axioms KBound.PaperDecisionAlgebra.printed_numeric_branches
#print axioms KBound.PaperDecisionAlgebra.universal_unit_residual_bound
