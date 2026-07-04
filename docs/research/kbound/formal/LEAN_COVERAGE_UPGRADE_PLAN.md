# Lean upgrade: coverage → FA_u ≤ α with an actual probability measure

Date: 2026-07-03. Status: Layers 1–3 drafted (this commit), iterate on `lake build`.
Prior-art check (web, Loogle, GitHub, Zulip, arXiv, AFP): **no existing formalization of
conformal-prediction coverage in Lean/Isabelle/Coq** — this is a first, citable on its own.

## What exists vs what this adds

Current `KBound.Conformal` + `KBound.Probability.ConformalExchangeability` are *algebraic
stubs*: `finiteUniformRankMiss n k` is a real-number formula; "exchangeability" is narrative.
This upgrade replaces the narrative with kernel-checked probability, in three layers plus one
staged stretch layer.

## Layers

**L1 — Measure-theoretic certificate bound** (`KBound/Probability/MeasureCertificate.lean`).
For any measurable space Ω, probability measure μ, functions dhat, delta : Ω → ℝ:
if μ{ω : |dhat ω − delta ω| ≤ eps} ≥ 1 − α then
μ{ω : certificate (dhat ω) eps = adapt ∧ delta ω ≤ 0} ≤ α (and the freeze mirror).
Proof: pointwise containment (reusing `cert_false_adapt_implies_coverage_failure`) +
`measure_mono` + `prob_compl_eq_one_sub` + ℝ≥0∞ tsub monotonicity. α : ℝ≥0∞.
Only the *coverage* event needs measurability (measure_mono needs none), so the main theorems
take a `MeasurableSet` hypothesis, with a helper deriving it from `Measurable dhat/delta`.

**L2 — Rank counting, no shortcut exists in Mathlib** (`KBound/Probability/RankCounting.lean`).
`strictRank R j = #{i : R i < R j}` on `Fin (n+1)`.
Theorem: `#{j : k ≤ strictRank R j} ≤ (n+1) − k`.
Proof (min-witness, ~30 lines): take j* minimizing R over the high-rank set S; its ≥ k strict
witnesses all lie outside S (minimality), so |Sᶜ| ≥ k. Ties handled automatically (strict `<`).

**L3 — Uniform-index conformal coverage** (`KBound/Probability/UniformConformal.lean`).
Exchangeable-by-construction model: held-out index J ~ `PMF.uniformOfFintype (Fin (n+1))`
(the conditional-on-the-score-bag form of split conformal — the standard proof's core).
P(k ≤ strictRank R J) = #S/(n+1) ≤ (n+1−k)/(n+1) ≤ α at k ≥ (1−α)(n+1),
via `PMF.toMeasure_uniformOfFintype_apply` + L2 + ℝ≥0∞ division monotonicity.
Capstone corollary: L3 coverage + L1 ⇒ FA_u ≤ α in the uniform model, machine-checked
end-to-end.

**L4 (stretch, not in this commit) — general exchangeability.** Define
`Exchangeable` (none exists in Mathlib: for every σ : Perm (Fin (n+1)), the joint law is
σ-invariant; engine: `measurePreserving_piCongrLeft` on constant `Measure.pi` families),
push forward to the uniform-index model, recover L3 for arbitrary exchangeable score vectors,
then full split-conformal statement with the threshold computed from the n calibration
coordinates (rank identity). This is the multi-week part; L1–L3 already upgrade the claim
honestly.

## Claim wording after L1–L3 build green

"Kernel-checked: the certificate's unconditional false-adapt/false-freeze bound over an
arbitrary probability measure (L1), and split-conformal coverage in the
conditional-on-the-bag uniform-rank model (L2–L3). General exchangeable-process coverage
remains a stated assumption (L4 in progress)." Do NOT claim the full exchangeability theorem
until L4 is done.

## Mathlib API used (verified via Loogle/mathlib4_docs 2026-07-03)

`MeasureTheory.IsProbabilityMeasure`, `measure_mono`, `prob_compl_eq_one_sub`,
`prob_le_one`, `tsub_le_tsub_left`, `tsub_tsub_le` (fallback: by_cases on α ≤ 1),
`measurableSet_le`/`measurableSet_lt`, `Measurable.sub`/`.abs`/`.sub_const`,
`Finset.exists_min_image`, `Finset.card_sdiff`, `Fintype.card_subtype`,
`PMF.uniformOfFintype` + `PMF.toMeasure_uniformOfFintype_apply` +
`PMF.toMeasure.isProbabilityMeasure`, `ENNReal.div_le_div_right`, `Fintype.card_fin`.
Known drift risk: exact module paths at the pinned mathlib tag. If an `import` fails,
first fix: replace the file's Mathlib imports with the single umbrella `import Mathlib`
(slower build, always correct), then narrow later.

## Build loop

```
cd /Volumes/T9/uav/AutoML_Flagship_V8/docs/research/kbound/formal
lake exe cache get   # mathlib cache (needed once; also fixes ProofWidgets JS)
lake build 2>&1 | tail -40
```
Paste errors back; expect 1–3 iterations (new heavy Mathlib imports are the usual suspects).
