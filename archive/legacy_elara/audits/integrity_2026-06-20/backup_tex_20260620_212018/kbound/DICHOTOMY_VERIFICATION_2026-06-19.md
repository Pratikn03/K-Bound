# Adversarial Verification — "Knowability Dichotomy" for K-Bound
**Date:** 2026-06-19  **Role:** adversarial verifier (mandate: try to break it, expose overclaim)
**Artifacts audited:**
- `docs/research/kbound/paper/sections/knowability_dichotomy.tex`
- `docs/research/kbound/paper/sections/knowability_capacity_general.tex` (companion it leans on)
- `experiments/kbound/theory_validation/val_knowability_dichotomy.py`
- `experiments/kbound/theory_validation/results_knowability_dichotomy.json`

All numbers below were independently re-run in a clean sandbox (numpy/scipy, seed 20260619)
and via my own adversarial scripts. The substantive `.tex`/`.py` files were **not** modified.

---

## TL;DR VERDICT

| Claim | Verdict | One-line reason |
|---|---|---|
| **1. Dichotomy: capacity exists ⟺ Φ, τ=1, capacity=κ** | **HOLDS — but the τ=1 half is NEAR-DEFINITIONAL** | Converse is real (Le Cam). "Universal τ=1" is the algebra `dist/ε>1 ⟺ dist>ε`. The non-trivial content is per-family (the flip *is* a function of Q_X) + finite-n recoverability, not the threshold. |
| **2. Hierarchy collapse: τ=1 survives multi-flip + curved frontier** | **ONLY-NUMERICAL (and the numerics certify an identity, not the geometry)** | The `.tex` labels Lemma multiflip **(NUMERICAL)** honestly. But Blocks B/D test the *same algebraic tautology* as Claim 1 — they do **not** independently certify the frontier shape. |
| **3. Causal map: Φ⟺anticausal is FALSE unconditionally; holds under G1∧G2** | **HOLDS as stated, but the witnesses are NOMINAL** | The "unconditional biconditional is false" conclusion is correct and the `.tex` is honest about genericity. C3/C4 are real *logical* counterexamples but carry **no SCM content** in code (C3 = the Gaussian location test re-labelled "causal"; C4 = the A2 nuisance witness re-labelled "anticausal"). |
| **Overall framing** | **Honest. NOT a paradigm theorem — and the `.tex` says so.** | Line 370–371 explicitly: *"not an unconditional paradigm theorem."* No fabricated proofs found. The result is a correct **framing + per-family + conditional-causal** contribution, not a deep new threshold phenomenon. |

**My own break attempts (fat-Cantor pathological margin; try-to-beat-½; finite-n at true κ>1) all FAILED to break the dichotomy.** It is correct. The issue is not correctness — it is *how much is theorem vs definition*.

---

## ATTACK 1 — TAUTOLOGY AUDIT (the decisive one)

**Finding: the "universal threshold τ=1" is definitional, not discovered.**

The capacity is *defined* as `κ(O) := |m(O)|/ε`, where `m(O) = dist(admissible band at O, frontier Γ)`.
The headline equivalence is then

> `κ(O) > 1  ⟺  sign Δ is label-free identifiable at O`.

But "identifiable at O" is itself *defined* (Thm A proof, lines 165–168 of the `.tex`) as
"the admissible band of mass-radius ε around the calibrated point does **not straddle** Γ",
i.e. `dist(band, Γ) > ε`. So the equivalence reads

> `|m|/ε > 1  ⟺  |m| > ε`,

which is the arithmetic of dividing by ε. **I confirmed this is model-free:** feeding a *completely
arbitrary* flip location (no benefit model at all) into `(κ>1)` vs `(band excludes flip)` gives
**0 mismatches over 20,000 random draws** — and the *same* 0 mismatches when I replace Block B's
frontier with a garbage `0.5 + 0.4 sin(11μ)cos(7μ)`. The "0 mismatches" reported by the validator
in Blocks A1, B, C1, C3, D are **re-verifications of this algebraic identity**, because in every
block `kappa(·)` and `ident(·)` consume the *same* `flip_mass(·)`; they cannot disagree except on
the measure-zero boundary `|m|=ε`.

**Honest fraction (theorem vs definition):**
- The **converse (C)** — *real theorem*, but a shallow one: it is verbatim the prior
  `Thm gen-df-conv`, a two-point Le Cam bound. Its content is "two worlds with identical
  observable law and opposite answers are indistinguishable ⇒ minimax ½." TV(Q^⊗n,Q^⊗n)=0 is true
  *by construction*, so this is close to a restatement of indistinguishability. Call it **~20% novel
  math, 80% standard.**
- The **τ=1 / capacity=κ statement** — **~90% definitional.** Once you decide to measure the margin
  in units of ε, τ=1 is forced. There is essentially no theorem here; it is a units convention.
- The **genuinely non-trivial content** is *not* in the dichotomy statement at all. It is the
  *per-family* fact that **the flip locus actually factors through the observable reduct O** (e.g.
  for Gaussian/Laplace/logistic location, and for the identifiable Gaussian mixture), plus the
  *finite-n recoverability of O*. I verified the substantive per-family check directly: sweeping the
  hidden concept within budget at κ>1, the sign predicted from O alone never fails
  (**0 failures / 14,472 draws**). *That* is the real (but family-specific, and already essentially
  in the companion §knowcap-gen) content.

**Bottom line for Claim 1:** the *dichotomy framing* ("one property Φ controls existence") is a
genuine and clean organizing insight; the *converse* is a correct standard bound; the *"universal
threshold τ=1 / capacity = frontier margin"* is a definition dressed as a theorem. **HOLDS-BUT-
NEAR-DEFINITIONAL.**

---

## ATTACK 2 — PROVEN vs NUMERICAL (Claim 2)

**The `.tex` is honest about the label** (Lemma `lem:dich-multiflip` is tagged **NUMERICAL**, and
Remark `rmk:dich-scope` repeats it). So this is not an overclaim of *proof status*. But two things
must be said bluntly:

1. **"τ=1 survives multi-flip / curved frontier" is not proven in the `.tex`.** The proof of
   Thm `dich-main(A)` asserts (lines 175–178) that "no monotonicity of H is used" — which is *true
   and is a legitimate observation* — but it is an observation about the *definition* of the local
   band-vs-frontier event, not a theorem requiring the multi-flip/2-D Monte-Carlo. The MC (Blocks
   B, D) does not add rigor; per Attack 1 it certifies the same `|m|/ε>1 ⟺ |m|>ε` algebra. So the
   slogan **"the single Φ collapses the R1∧R2 hierarchy"** rests on: (i) the *definitional*
   observation that the threshold doesn't reference monotonicity, plus (ii) numerics that don't test
   what they appear to test. It is **conjecture-grade as a geometric statement, asserted-grade as a
   definitional one** — *not* an independently proven collapse.

2. **The achievability direction (Φ ⇒ capacity) has a real, named, open sliver.** The `.tex` is
   candid: Remark `rmk:dich-compute` + Conjecture `conj:dich-compute` admit that **Φ (set-theoretic
   factorization) ⇒ Φ-computability is *unproven*.** So "achievability PROVEN under computability" is
   honest but the qualifier is load-bearing: the clean statement "capacity exists ⟺ Φ" is **not
   proven** unconditionally; only "⟺ Φ-computably" is. I tried to *exploit* this gap with a
   pathological fat-Cantor sign set (positive-measure, nowhere-dense frontier) to make m(O)
   non-recoverable — **the dichotomy survived** (margin is still a measurable distance-to-closed-set;
   plug-in works in the interior, 0 failures / 317 interior draws). This matches the authors' own
   admission that they have *no* natural non-computable example. So the sliver is real but
   **plausibly empty**, exactly as claimed.

**Verdict Claim 2: ONLY-NUMERICAL for the geometric content, with the further caveat that the
numerics certify an identity rather than the frontier geometry. Honestly labelled, but the
"collapse" should be read as definitional+conjectural, not a proven theorem.**

---

## ATTACK 3 — TRY TO BREAK Φ (my own counterexamples)

I ran three independent break attempts (sandbox MC). **None broke the dichotomy.**

1. **Φ holds, pathological non-recoverable margin (fat-Cantor frontier).** Goal: sign factors
   through O=μ but m(O) is a nowhere-dense-boundary distance that a plug-in can't track.
   *Result:* 0 plug-in failures in the interior (κ≫1). The margin is a legitimate measurable
   distance; achievability holds. **No break.**

2. **Φ fails, try to make a label-free rule work.** Two worlds, identical Q_X=N(0,1), opposite
   sign. Tested const±, mean>0, var>1, and a nonlinear `sin(Σx)` statistic.
   *Result:* every rule has worst-case error ≥ ~0.50 (const rules → 1.00). **No break** — the
   converse is airtight (because the sample laws are literally identical; this is Le Cam, not magic).

3. **Finite-n achievability at a genuine κ=2.415.** *Result:* plug-in sign-error = 0.0000 even at
   n=20. No hidden finite-n gap when O is √n-recoverable. **No break.**

**Conclusion: the dichotomy is correct and I could not break it.** Its robustness, however, is
*because* the load-bearing equivalence is near-definitional (Attack 1) — there is little surface
area to attack. A tautology is hard to falsify.

---

## ATTACK 4 — CAUSAL CLAIMS (Block C)

**Re-ran C1–C4 (full and quick): all four reproduce.** The *logical* structure of Claim 3 is
**correct**: to disprove the unconditional slogan "Φ ⟺ anticausal" you need one causal-but-Φ witness
(C3) and one anticausal-but-¬Φ witness (C4), and both exist. The `.tex` is honest that the
biconditional is only conditional (G1∧G2) and that each genericity is necessary.

**But the witnesses are nominal, not structural:**

- **C3 ("causal-but-Φ") carries no SCM content.** In code it calls
  `_capacity_test_locfamily(N(x−μ))` — **byte-for-byte the A1-gaussian computation** (I confirmed
  C3 mismatches = A1-gaussian mismatches = 0, same machinery). The word "causal" is a re-labelling
  of the single Gaussian location capacity test. As a *logical* counterexample it is valid (a
  covariate-shift model genuinely is X→Y with stable η, and Φ holds), but it does **not** instantiate
  or stress any causal mechanism.

- **C4 ("anticausal-but-¬Φ") is degenerate by design.** It sets μ0=μ1 so P(X|Y) ignores Y — i.e.
  *there is no anticausal structure at all*; Y is irrelevant to X. It is the **A2 hidden-nuisance
  witness re-labelled "anticausal."** Sign = sign(1−2π) flips while Q_X=N(0,1) is fixed. Valid
  Φ-fails witness; correctly shows "uninformative mechanism ⇒ ¬Φ." But it does **not** exhibit a
  *non-degenerate* anticausal model failing Φ.

- **C1 ("anticausal+invariant ⇒ Φ") assumes its key hypothesis.** It fixes μ0,μ1 = −1.2, 1.2
  (well-separated ⇒ mixture identifiable). The real anticausal content — *mixture identifiability is
  what makes P(Y|X) recoverable from Q_X* — is **assumed by construction**, never stress-tested.
  Then the same band-vs-flip algebra runs on the mixture CDF. So C1 ⇒ "Φ holds" is, again, the
  Attack-1 identity applied to a mixture.

**Are the genericities G1/G2 substantive or do they assume the conclusion?** They are **honestly
substantive as stated** (the `.tex` proves each inclusion and shows each genericity is necessary via
the witnesses), but the proof of inclusion (i) (anticausal+informative ⇒ Φ) is essentially
*"if P(Y|X) is a functional of Q_X then the flip locus is a functional of Q_X"* — true, and not
circular, but shallow: it is the definition of "informative mechanism" doing the work. **The causal
correspondence is a correct re-description in causal vocabulary, with the unconditional slogan
correctly disproven; it is not a new causal theorem.**

**Verdict Claim 3: HOLDS (logically correct, honestly conditional) — but the C3/C4 witnesses are
nominal re-labellings and the proven inclusions are near-definitional.**

---

## ATTACK 5 — INDEPENDENT RE-RUN (headline numbers)

Re-ran the validator (quick driver + full Blocks B and D + full Block C) in the sandbox, seed
20260619. **Every headline number reproduces exactly:**

| Quantity | Published JSON | My re-run | Match |
|---|---|---|---|
| A1 mismatches (gaussian/laplace/logistic/locscale) | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | ✅ |
| A2 benefit signs at same Q_X / minimax | [−1,1] / 0.5 | [−1,1] / 0.5 | ✅ |
| B sign-flips / grid / mismatches | 3 / 6000 / 0 | 3 / 6000 / 0 | ✅ |
| C1 mismatches / C2 signs / C3 mismatches / C4 signs | 0 / [−1,1] / 0 / [−1,1] | identical | ✅ |
| unconditional biconditional FAILS | true | true | ✅ |
| D tested / identifiable / mismatches / near-bdry | 29990 / 21740 / 0 / 330/330 | 29990 / 21740 / 0 / 330/330 | ✅ |

**No number failed to reproduce.** The JSON is faithful to the code and the run is deterministic.
*(Caveat already established in Attacks 1–2: these zeros certify an algebraic identity, not the deep
content the prose attributes to them.)*

---

## ATTACK 6 — OVERCLAIM AUDIT of the `.tex`

**Status labels are present and honest on every theorem.** Each carries an explicit
PROVEN / PROVEN-UNDER-COND / NUMERICAL / CONJECTURE tag; the header of Thm `dich-main` even splits
"converse PROVEN distribution-free, achievability PROVEN under computability." Remark `rmk:dich-scope`
is a model of candor and **explicitly disavows the paradigm framing the user wants** (line 370–371:
*"This is a strong, fundamental, and honestly-bounded step, **not an unconditional paradigm
theorem**"*).

Sentences that read as *slightly* more than was proven (editorializing, not false):
- L122 "Theorem … shows it is **exactly the right one**." — value judgment; the math shows Φ is
  *sufficient framing*, not uniquely forced.
- L374–376 "The fundamental object … is, we believe, **the right one**." — flagged as belief, but
  primes the reader toward "paradigm."
- L138/146 "**universal** threshold τ=1 / the universal constant 1" — technically accurate but, per
  Attack 1, "universal" here means "definitional," which a casual reader will over-read as "deep."
- Lemma `lem:dich-multiflip` prose ("τ=1 **survives** … regardless of the frontier's shape or the
  reduct's dimension") slightly oversells: it survives *because the threshold is defined not to
  reference shape*, and the numerics don't independently test shape (Attack 2).

**No fabricated proof, no mislabelled NUMERICAL-as-PROVEN, no invented experiment.** This is an
honestly-scoped document.

---

## FINAL ASSESSMENT — is this a correct, substantive, honestly-bounded contribution?

**Correct:** Yes. I could not break it; the converse is a valid Le Cam bound, the achievability is
operationally real for the families treated, the causal slogan is correctly disproven, and the
numerics reproduce exactly.

**Honestly bounded:** Yes, unusually so. The document labels every claim, names its open sliver
(`conj:dich-compute`), names its causal genericities (G1/G2) and their necessity, and *explicitly
refuses* the "paradigm theorem" label.

**Substantive — how much is real math vs restatement?** This is where the skeptical verdict bites:

- The **"complete knowability dichotomy with universal τ=1"** headline is **~70–80% framing/definition
  and ~20–30% theorem.** The theorem-grade content is (a) the standard distribution-free converse
  (inherited, shallow), and (b) the *already-established* per-family fact that the flip locus factors
  through Q_X. The eye-catching parts — "universal threshold τ=1," "capacity = frontier margin,"
  "single property collapses the hierarchy" — are a **units convention and a definitional
  observation**, validated by Monte-Carlo runs that (I proved) certify the algebra `dist/ε>1 ⟺
  dist>ε` rather than any frontier geometry.
- The **causal correspondence** is a **correct re-description in causal vocabulary** whose only
  theorem-grade output is "the unconditional biconditional is false," supported by two **nominal**
  witnesses (C3 = Gaussian location test re-labelled; C4 = the nuisance witness re-labelled). The
  proven inclusions are near-definitional.

**Verdict on the "paradigm-shaped" framing the user is hoping for:** **It is not a paradigm result,
and the math does not support promoting it to one.** What is genuinely valuable and defensible:
1. The **single-property *framing*** Φ ("the benefit-relevant shift must be visible in Q_X") is a
   clean, correct, useful organizing principle — a good *definition/lens*, not a deep theorem.
2. The **honest map** of where the threshold travels and stops, with explicit breakers.
3. The **explicit disproof** of the seductive "knowable ⟺ anticausal" slogan — a genuinely useful
   negative result.

These are a solid, publishable *conceptual* contribution. They are **not** a new threshold phenomenon
or a deep dichotomy theorem. Anyone marketing this as "a complete knowability dichotomy" should be
told, plainly: the dichotomy is **near-definitional**; the substance is the framing, the per-family
identifications (mostly already in the companion section), and the conditional causal map with its
slogan-killing counterexamples. The current `.tex` already says almost exactly this in
`rmk:dich-scope` — the risk is purely that the *abstract/headline* prose ("universal," "complete,"
"the right one") will be read as more than the body proves.

**Recommendation:** keep the result, keep the (excellent) honesty remarks, and **down-tune the words
"universal / complete / the right one" in the headline** to match `rmk:dich-scope`; **add a one-line
caveat to Blocks B/D** noting that the 0-mismatch certifies the band-vs-frontier *identity* given the
analytic frontier, not the frontier's geometry independently; and **relabel C3/C4 as "logical
witnesses (nominal SCM)"** so no reader mistakes them for stress-tested causal mechanisms. With those
three edits the contribution is correct, substantive-as-framing, and bullet-proof against exactly the
adversarial reading I just performed.
