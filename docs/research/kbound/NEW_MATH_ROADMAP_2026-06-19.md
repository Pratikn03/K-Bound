# K-Bound — New-Math Roadmap & the Knowability-Capacity Result (2026-06-19)

Output of an ideation loop (architect+search → prover) run to find mathematics that
could raise K-Bound's novelty, and to rigorously attempt the best tractable candidate.
Integrity note: the result below is real and validated; its scope is stated honestly
and is **not** oversold as paradigm-level.

## 1. The candidate agenda (ranked novelty × tractability)

Prior-art anchors: Ben-David et al. 2010 (domain-adaptation impossibility);
Kpotufe–Martinet (transfer-exponent γ — graded label-*benefit* rate, COLT'18 / AoS'21);
Wu et al. JMLR 2024 (causal vs anticausal, information-theoretic); POEM (NeurIPS'24) &
Schirmer et al. 2025 (e-process / betting TTA monitoring); Dong–Liu–Mao 2025 (PTLU,
average-case UDA hardness).

1. **Knowability capacity** — a graded invariant whose threshold marks label-free
   *sign-of-benefit decidability*, with matched achievability + converse. Distinct from
   γ (rate, not decidability) and PTLU (average-case, no converse). **PICKED + proven in
   1-D (§2).**
2. **k-bit hierarchy** of shifts by side-information needed — high ceiling, but
   formalization-fragile; attempt after the capacity base case is solid.
3. **Minimax-optimal safe-adaptation rates** — provable but crowded by Kpotufe–Martinet;
   novelty hinges on a safety-induced rate separation that may not exist.
4. **Causal characterization** of knowability — strong-if-true but heavily preempted by
   Wu et al. 2024; likely only a caveated biconditional.
5. **Composable continual certificates** — tractable but incremental (the "another
   certificate" trap; the certificate is already not the novel part).

## 2. The proven result — an exact Knowability Capacity (1-D Gaussian-location)

**Invariant.** K = (mass-margin between the calibrated boundary and the sign-flip
locus) / (concept drift-budget ε) — closed-form, computable from observables (known
source, fixed rule, unlabeled target sample).

**Converse (PROVEN, Le Cam two-point).** K < 1 ⟹ two admissible target concepts with
*identical* observable Q_X but *opposite* benefit-sign ⟹ label-free minimax error = ½
for all n. (The graded reading of Thm 1's TV = 0 face, triggered exactly on {K<1}.)

**Achievability (PROVEN, matches).** K > 1 ⟹ the plug-in certificate is correct for
*every* admissible concept. Converse and achievability **meet at the same τ = 1**
(population problem) — no gap.

**Finite-n (PROVEN).** A *located* O(1/√n) Le Cam boundary layer around the flip
(width ~1/√n, constant error inside), not a τ-value gap.

**Validated.** `val_knowability_capacity.py` → `results_knowability_capacity.json`
(`all_ok=true`, seed 20260619): 40,000 random draws, **0** "K>1 ⇔ identifiable"
mismatches; phase transition pinned at K=1 (acc →1 above, =½ below); Le Cam layer meets
the Bretagnolle–Huber floor; benefit identity matches MC to 4.2e-4; equivalence survives
soft/logistic concepts. Write-up: `paper/sections/knowability_capacity.tex` (refs resolve).

## 3. Honest scope — what it is and is not

- **IS:** the first *exact, closed-form capacity* for label-free sign-of-benefit
  decidability — a graded invariant turning the binary impossibility into a threshold,
  with matched converse + achievability — in one clean model.
- **IS NOT:** a general-distribution capacity. τ = 1 depends on the natural mass-unit
  drift budget; it does not enlarge the distribution-free guarantee set. It should ride
  as a **subsection under the benefit-sign frontier**, not a sixth headline theorem.

## 4. The path to genuine paradigm-level novelty (the prize — honestly hard)

The 1-D result proves the graded invariant *exists and is exact* in a clean case. That
is how such results start; it is not yet field-changing. The prize:

1. **General knowability capacity** — define K and prove a *matched* converse +
   achievability for broad classes (multivariate, exponential families, nonparametric).
   This is the genuine open problem and a multi-month research program, not a one-shot
   proof. The architect rated the general version hard/long-shot, honestly.
2. **Tie K to the k-bit hierarchy and/or causal structure** — if K both grades
   decidability *and* counts the side-information bits (or maps to causal direction),
   it becomes field-organizing. That combination would be the paradigm-level object.

## 5. Verdict

The loop produced **real new mathematics** — a new object plus two proven theorems and a
clean numerical confirmation — a genuine step up in novelty. But it is a **worked
instance, not a paradigm shift**: an exact capacity in one model, scoped honestly as a
refinement of the existing frontier. The value of the exercise is that it **identified
and de-risked the right hard problem**: the *general* knowability capacity. If the goal
is field-changing novelty, that is now the precise target to attack — and we have shown
the clean case is provable, which is the honest, non-trivial first step toward it.
