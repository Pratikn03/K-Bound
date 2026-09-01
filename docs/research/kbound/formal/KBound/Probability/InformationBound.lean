import Mathlib.Analysis.SpecialFunctions.BinaryEntropy
import Mathlib.Tactic.Linarith

/-!
# Binary testing information bound

This is the scalar Bretagnolle--Huber bound for the two-outcome distributions
`(p,1-p)` and `(q,1-q)`. Support conditions are explicit because Lean's real
logarithm is totalized at zero; an invalid finite-KL support case must not be
silently assigned the value produced by `Real.log 0`.

The proof uses concavity of the logarithm and the binary-entropy bound by `log 2`.
Connecting this scalar result to general measures requires a separate proved
binary-partition/data-processing inequality for the actual KL divergence.
-/

namespace KBound

private theorem weighted_log_division (x y : ℝ) (hy : y ≠ 0) :
    x * Real.log (x / y) = x * Real.log x - x * Real.log y := by
  by_cases hx : x = 0
  · simp [hx]
  · rw [Real.log_div hx hy, mul_sub]

/-- **Bretagnolle--Huber for a binary test.** Here `p` and `q` are the two
probabilities of the test choosing its first action, so `(1-p)+q` is the sum
of its errors. The conditions on zero/one `q` are precisely binary absolute
continuity and retain all supported boundary cases. -/
theorem binary_bretagnolle_huber {p q : ℝ}
    (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (h_support_zero : q = 0 → p = 0) (h_support_one : q = 1 → p = 1) :
    Real.exp (-(p * Real.log (p / q) +
      (1 - p) * Real.log ((1 - p) / (1 - q)))) / 2 ≤ (1 - p) + q := by
  by_cases hqz : q = 0
  · have hpz := h_support_zero hqz
    subst p
    subst q
    norm_num
  by_cases hqo : q = 1
  · have hpo := h_support_one hqo
    subst p
    subst q
    norm_num
  have hqpos : 0 < q := lt_of_le_of_ne hq0 (Ne.symm hqz)
  have hqsub : 0 < 1 - q := sub_pos.mpr (lt_of_le_of_ne hq1 hqo)
  have hpq : 0 ≤ p * q := mul_nonneg hp0 hq0
  have hrest : 0 ≤ (1 - p) * (1 - q) :=
    mul_nonneg (sub_nonneg.mpr hp1) hqsub.le
  have hmixpos : 0 < p * q + (1 - p) * (1 - q) := by
    by_cases hpz : p = 0
    · simpa [hpz] using hqsub
    · exact add_pos_of_pos_of_nonneg
        (mul_pos (lt_of_le_of_ne hp0 (Ne.symm hpz)) hqpos) hrest
  have hmix_le : p * q + (1 - p) * (1 - q) ≤ (1 - p) + q := by
    nlinarith [mul_nonneg hq0 (sub_nonneg.mpr hp1)]
  have herrpos : 0 < (1 - p) + q := hmixpos.trans_le hmix_le
  have h_jensen : p * Real.log q + (1 - p) * Real.log (1 - q) ≤
      Real.log (p * q + (1 - p) * (1 - q)) := by
    exact strictConcaveOn_log_Ioi.concaveOn.2 hqpos hqsub hp0
      (sub_nonneg.mpr hp1) (by ring)
  have h_cross : p * Real.log q + (1 - p) * Real.log (1 - q) ≤
      Real.log ((1 - p) + q) :=
    h_jensen.trans (Real.log_le_log hmixpos hmix_le)
  have h_entropy := Real.binEntropy_le_log_two (p := p)
  have h_formula : p * Real.log (p / q) + (1 - p) * Real.log ((1 - p) / (1 - q)) =
      -Real.binEntropy p - (p * Real.log q + (1 - p) * Real.log (1 - q)) := by
    rw [weighted_log_division p q hqz,
      weighted_log_division (1 - p) (1 - q) hqsub.ne']
    simp only [Real.binEntropy, Real.log_inv]
    ring
  have h_exp_arg : -(p * Real.log (p / q) +
      (1 - p) * Real.log ((1 - p) / (1 - q))) ≤
      Real.log 2 + Real.log ((1 - p) + q) := by
    rw [h_formula]
    linarith
  have h_exp := Real.exp_le_exp.mpr h_exp_arg
  rw [Real.exp_add, Real.exp_log (by norm_num : (0 : ℝ) < 2), Real.exp_log herrpos] at h_exp
  apply (div_le_iff₀ (by norm_num : (0 : ℝ) < 2)).2
  nlinarith

end KBound
