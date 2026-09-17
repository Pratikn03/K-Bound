# K-Bound Theory Audit — Phase 1

Scope: `paper/sections/theory_setup.tex` (shared), `paper/sections/theory_core_main.tex`
(short paper), `paper/sections/main_theory_5.tex` (long paper). Audited against the
manuscript-revision brief §7. Every claim below was checked against the definitions actually
used in source, not against a prior draft. Sign convention verified consistent everywhere:
Δ = R_T(f₀) − R_T(f_a), so Δ>0 ⇔ adapting helps.

## A. Results verified CORRECT (no change)

1. **Disagreement-region identity** (`lem:reduction`). With ā := Pr_T(f_a=Y|D), on D exactly one
   of f₀,f_a is correct under 0/1 loss, giving Δ = 2μ_T(D)(ā − ½). The reduction sign(Δ)=sign(M+γ)
   is valid **once η_a is the pointwise correctness** (see correction B1).
2. **Frontier IFF** (`thm:headline`/`thm:frontier` (iv)/(ii)): sign(Δ) is label-free identifiable
   over C_β **iff |M|>β**. Correct. At |M|>β and |γ|≤β, sign(Δ)=sign(M). The headline result is
   sound and is **not** affected by the boundary fix.
3. **Finite-sample certificate** (`thm:certificate`/`thm:cert`): stated under the coverage
   hypothesis Pr[|Δ̂−Δ|≤ε] ≥ 1−α; the adapt-iff-(Δ̂−ε>0) rule then has marginal false-adapt ≤ α.
   The proof (a false adapt requires the complement of the coverage event) is correct. The theorem
   is stated as a *conditional* on coverage — correct; over-claiming is avoided.
4. **Marginal vs conditional** (`rem:fa-marginal`): correctly bounds FA_u = Pr(adapt, Δ≤0) and
   explicitly does NOT bound FA_c = Pr(Δ≤0 | adapt) (§7.7 satisfied at the theorem level).

## B. Corrections APPLIED this phase

**B1 — η_a definition (§7.2). REAL FIX.**
`theory_setup.tex` defined η_a(x) := Pr_T(Y=1 | f_a(x)=1) — a *class-conditional* posterior, exactly
the form §7.2 warns against. The lemma step ā−½ = E[η_a−s|D] = M+γ only holds if η_a is the
*pointwise correctness*. Averaging the class-conditional over D does **not** equal the correctness
rate ā, so the bridge from the (correct) identity Δ=2μ_T(D)(ā−½) to sign(M+γ) did not follow from
the definitions.
- Fix: η_a(x) := Pr_T(f_a(X)=Y | X=x) (pointwise correctness); score s redefined as a calibrated
  estimate of that correctness. Then E_{μ_T|D}[η_a] = ā and M+γ = E[η_a|D]−½ = ā−½. Identity now
  valid. (Edited `theory_setup.tex`; propagates to both papers.)

**B2 — |M|=β boundary (§7.3). REAL FIX.**
`lem:nonid` and `thm:headline`(iii) asserted opposite *nonzero* benefit signs for the closed region
|M|≤β. At the boundary M=β>0, M+γ ∈ [0, 2β]: the sign can be + or 0, but **never** −, so opposite
nonzero signs are impossible at equality.
- Fix (applied to `theory_core_main.tex` and mirrored in `main_theory_5.tex`
  `thm:frontier`(ii)):
  - The opposite-nonzero-sign witness / minimax-½ statement now requires **strict |M|<β**.
  - At |M|=β, the admissible drift γ = −sign(M)·β forces Δ=0, which cannot be certified as strictly
    positive/negative; hence sign(Δ) is still undetermined and **abstention remains the
    maximal-sound action on the closed region |M|≤β**.
  - The headline IFF (identifiable iff |M|>β) is unchanged and remains correct: at |M|=β the sign
    is + or 0 (not determined), so |M|≤β is the non-identifiable region.

Both edits recompile clean (short paper: latexmk exit 0, 0 undefined refs, 20 pp).

## C. Items already satisfied in source (verified, no edit needed)

- **§7.4 β=0 wording.** `main_theory_5.tex` `rem:beta-zero` frames prior heuristics (ATC, GDE, COT,
  AETTA, AGL) as sound only on C₀ (γ=0); the short-paper Method/Limitations already state β=0 is the
  "strongest zero-drift assumption … not an assumption-free or conservative default." Correct.
- **§7.5 β vs ε.** `theory_setup.tex` `rem:four-quantities` explicitly separates M, γ, β, ε and
  states "ε is not an estimate of β." Correct.
- **§7.6 finite-sample coverage.** The Method section already distinguishes exact split-conformal
  (clean split) from the leave-one-out empirical radius, and states the jackknife+ finite-sample
  counterpart is "noted but not implemented." No over-claim of jackknife+ found. Correct — keep.
- **§7.9 multiclass bridge.** `lem:reduction` (long) states Δ=μ_T(D)(p_a−p_0) for K-class; the
  short paper carries a "Multiclass and Regression Scope" subsection with the same relation and does
  NOT reuse p₀=1−p_a. Binary scope of the frontier is explicit. Adequate for the main paper.
- **§7.10 generality.** The β=0 "face" is stated as prior heuristics being *sound on C₀* and sharing
  the γ≠0 failure mode — appropriately qualified, not an identity claim.

## D. Open theory items (flagged, not yet actioned)

- **DRAFT TODO [THEORY-DUP-LABELS]:** `theory_core_main.tex` and `main_theory_5.tex` reuse the same
  \label keys (`lem:reduction`, `lem:nonid`/`thm:imp`, `thm:headline`/`thm:frontier`,
  `thm:certificate`). They never co-compile (short vs long paper), so there is no duplicate-label
  error today, but if any future file \inputs both, labels will collide. Recommend namespacing the
  long-paper variants (e.g. `thm:frontier-long`).
- **DRAFT TODO [THEORY-RISK-ALIGN]:** `def:risk-align` is used for the frontier identifiability
  claim; confirm the main text never states the empirical benefit model *establishes* risk
  alignment (§7.8). Prose pass pending in Phase 2/3.
- The pointwise-correctness score s now needs one sentence in the Method tying the implemented
  confidence feature to s (Phase 2).

## E. Status
Phase 1 theory corrections applied and compiling. Headline results intact; two genuine rigor bugs
(η_a class-conditional; |M|=β boundary) fixed and verified.
