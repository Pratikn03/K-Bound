import KBound.Probability.RankCounting
import KBound.Probability.MeasureCertificate
import Mathlib.Probability.Distributions.Uniform
import Mathlib.MeasureTheory.MeasurableSpace.Instances

/-!
# Uniform-index conformal coverage (`thm:certificate`, probability layer)

Paper: split-conformal coverage, conditional-on-the-bag form.

The standard proof of split-conformal coverage conditions on the multiset (bag) of
the `n + 1` scores; exchangeability then makes the held-out index uniform.  This
module formalizes exactly that conditional model with an actual probability
measure: the held-out index `J` is uniform on `Fin (n+1)`
(`PMF.uniformOfFintype`), the miss event is `k ≤ strictRank R J`, and the miss
probability is bounded by `(n + 1 − k)/(n + 1)` via the min-witness counting bound
of `KBound.Probability.RankCounting` — hence `≤ α` at the conformal threshold
`k ≥ (1 − α)(n + 1)`.

The capstone theorem composes this with the measure-level certificate bound
(`KBound.Probability.MeasureCertificate`): in the uniform-index model, the
certificate's unconditional false-adapt probability is `≤ α`, machine-checked
end-to-end.  Lifting from the conditional bag model to arbitrary exchangeable
score processes (Layer 4) is future work and is *not* claimed here.

If `MeasurableSet.of_discrete` or the `Fin` measurable-space instance fails to
resolve at the pinned Mathlib revision, replace the import above with
`import Mathlib` and/or add `instance : MeasurableSpace (Fin (n+1)) := ⊤` locally.
-/

namespace KBound

open MeasureTheory

variable {n : ℕ}

/-- The uniform held-out-index measure on `Fin (n+1)`: the conditional-on-the-bag
form of exchangeability. -/
noncomputable def uniformIndexMeasure (n : ℕ) : Measure (Fin (n + 1)) :=
  (PMF.uniformOfFintype (Fin (n + 1))).toMeasure

instance : IsProbabilityMeasure (uniformIndexMeasure n) :=
  PMF.toMeasure.isProbabilityMeasure _

open scoped Classical in
/-- Miss probability of the rank-`k` conformal set under the uniform held-out
index: exactly `#{j : rank ≥ k} / (n + 1)`. -/
theorem uniformIndex_miss_eq (R : Fin (n + 1) → ℝ) (k : ℕ) :
    uniformIndexMeasure n {j | k ≤ strictRank R j}
      = ((Finset.univ.filter fun j => k ≤ strictRank R j).card : ENNReal)
          / ((n + 1 : ℕ) : ENNReal) := by
  rw [uniformIndexMeasure,
    PMF.toMeasure_uniformOfFintype_apply _ (MeasurableSet.of_discrete)]
  simp [Set.coe_setOf, Fintype.card_subtype, Fintype.card_fin]

open scoped Classical in
/-- **Uniform-rank miss bound.**  Under the uniform held-out index, the miss
probability is at most `(n + 1 − k)/(n + 1)`. -/
theorem uniformIndex_miss_le (R : Fin (n + 1) → ℝ) (k : ℕ) :
    uniformIndexMeasure n {j | k ≤ strictRank R j}
      ≤ (((n + 1) - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal) := by
  rw [uniformIndex_miss_eq]
  apply ENNReal.div_le_div_right
  exact_mod_cast card_high_strictRank_le R k

/-- **Split-conformal coverage in the uniform-index model.**  If the numeric
threshold satisfies `(n + 1 − k)/(n + 1) ≤ α` (equivalently `k ≥ (1 − α)(n + 1)`,
the conformal quantile rule), the coverage event `rank < k` has probability at
least `1 − α`. -/
theorem uniformIndex_coverage_ge (R : Fin (n + 1) → ℝ) (k : ℕ) {alpha : ENNReal}
    (hk : (((n + 1) - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal) ≤ alpha) :
    1 - alpha ≤ uniformIndexMeasure n {j | ¬ k ≤ strictRank R j} := by
  have hmiss : uniformIndexMeasure n {j | k ≤ strictRank R j} ≤ alpha :=
    le_trans (uniformIndex_miss_le R k) hk
  have hcompl : {j : Fin (n + 1) | ¬ k ≤ strictRank R j}
      = {j : Fin (n + 1) | k ≤ strictRank R j}ᶜ := by
    ext j; simp
  rw [hcompl, prob_compl_eq_one_sub (MeasurableSet.of_discrete)]
  exact tsub_le_tsub_left hmiss 1

/-- **Capstone: certificate false-adapt `≤ α` in the uniform-index model,
end-to-end.**  Hypotheses: the per-index benefit estimate is within `eps` of the
per-index true benefit whenever the index's score rank clears the conformal
threshold (`hsub` — the conformal-score/coverage link, supplied by the
instantiation), and the threshold satisfies the quantile inequality (`hk`).
Conclusion: the unconditional false-adapt probability under the uniform held-out
index is at most `α`.  Composes `uniformIndex_coverage_ge` with
`measure_false_adapt_le_alpha`. -/
theorem uniformIndex_false_adapt_le
    (dhat delta : Fin (n + 1) → ℝ) (eps : ℝ) (R : Fin (n + 1) → ℝ) (k : ℕ)
    {alpha : ENNReal}
    (hsub : {j | ¬ k ≤ strictRank R j} ⊆ coverageEvent dhat delta eps)
    (hk : (((n + 1) - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal) ≤ alpha) :
    uniformIndexMeasure n (falseAdaptEvent dhat delta eps) ≤ alpha := by
  have hcov : 1 - alpha ≤ uniformIndexMeasure n (coverageEvent dhat delta eps) :=
    le_trans (uniformIndex_coverage_ge R k hk) (measure_mono hsub)
  exact measure_false_adapt_le_alpha (MeasurableSet.of_discrete) hcov

/-- Mirror capstone for false-freeze. -/
theorem uniformIndex_false_freeze_le
    (dhat delta : Fin (n + 1) → ℝ) (eps : ℝ) (R : Fin (n + 1) → ℝ) (k : ℕ)
    {alpha : ENNReal}
    (hsub : {j | ¬ k ≤ strictRank R j} ⊆ coverageEvent dhat delta eps)
    (hk : (((n + 1) - k : ℕ) : ENNReal) / ((n + 1 : ℕ) : ENNReal) ≤ alpha) :
    uniformIndexMeasure n (falseFreezeEvent dhat delta eps) ≤ alpha := by
  have hcov : 1 - alpha ≤ uniformIndexMeasure n (coverageEvent dhat delta eps) :=
    le_trans (uniformIndex_coverage_ge R k hk) (measure_mono hsub)
  exact measure_false_freeze_le_alpha (MeasurableSet.of_discrete) hcov

end KBound
