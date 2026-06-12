# Closing Conjecture 1: the testability dichotomy

**Claim being closed.** Conjecture 1 (kbound.tex, `conj:gen`) asks for a label-free
*bracketing* of the ordinal comparison on D (p_a vs p_0 for K≥3; sign E[(f0−fa)Y|D] for
regression) "without target labels — a reliability-model assumption, as in the binary
case." The γ-meter section resolves it *relative to checkable agreement structure* and
leaves the unconditional question open. Here we close the unconditional question, in the
only direction it can close: **negatively, with an exact characterization**.

Every numeric claim below is produced by `val_conj1_closure.py` →
`conj1_closure_results.json` (keys cited inline).

---

## Setup

Observables (the *evidence channel*): the labeled source P_S, the fixed predictor maps,
and the unlabeled target X-stream — hence every label-free statistic of any order,
including the full joint law of the prediction vector (f_0(X), f_a(X), g_2(X), …, g_M(X))
and any source-calibrated score. Write L(P) for the induced law of ALL observables under
target P (note: P enters L(P) only through its X-marginal, since predictions are fixed
maps of X and P_S is fixed).

**Definition (evidence-definable class).** A class 𝒞 of targets is *evidence-definable*
(label-free testable in the population limit) if membership depends on P only through
L(P): L(P)=L(Q) ⇒ (P∈𝒞 ⟺ Q∈𝒞).

This is the natural formalization of "checkable assumption": any test with asymptotic
power for membership must be a functional of the evidence law.

---

## Lemma 1 (The swap, three output spaces)

For each output space there is an involution P ↦ P★ on targets that (i) preserves the
X-marginal (hence preserves L(P) exactly: TV(L(P), L(P★)) = 0), and (ii) flips the
benefit sign: Δ(P★) = −Δ(P) whenever Δ(P) ≠ 0.

**(a) Binary / multiclass (conditional swap on D).** On D = {f_0 ≠ f_a} define P★ by
swapping, pointwise in x ∈ D, the conditional masses of the two *disagreeing predicted
values*:
P★(Y = f_a(x) | x) = P(Y = f_0(x) | x), P★(Y = f_0(x) | x) = P(Y = f_a(x) | x),
all other values' masses unchanged; off D, P★ = P.

*Proof.* P★ is a valid conditional law (a transposition of two masses). The X-marginal
is untouched, predictions are maps of X, and P_S is untouched, so L(P★) = L(P) — at
every moment order, for every label-free feature. For the benefit: off D the per-sample
benefit δ = 0 under both. On D, p★_a := P★(f_a = Y | x) = P(f_0 = Y | x) and symmetrically,
so the swap exchanges p_a(x) ↔ p_0(x) pointwise; taking E_{μ|D}: p★_a = p_0, p★_0 = p_a,
hence Δ★ = P(D)(p★_a − p★_0) = −Δ. ∎
[Validated: `multiclass_swap`: evidence stats identical to 0 (exact); p′_a = p_0 and
p′_0 = p_a to machine precision; Δ′ = −Δ.]

**(b) Regression (midpoint reflection on D).** Define Y★ = f_0(X) + f_a(X) − Y
conditionally on X ∈ D (the pushforward of P(Y|x) under reflection about the prediction
midpoint m(x) = (f_0(x)+f_a(x))/2); off D, unchanged.

*Proof.* Valid law (pushforward of an affine involution). X-marginal untouched ⇒ L
preserved. Squared-loss benefit density δ = (f_0 − f_a)(f_0 + f_a − 2Y): under Y★,
(f_0 + f_a − 2Y★) = −(f_0 + f_a − 2Y), so δ★ = −δ pointwise on D and δ = 0 off D; hence
Δ★ = −Δ. (Note: reflection negates the *conditional-mean term* E[(f_0−f_a)(f_0+f_a−2m)]
and preserves the noise law's contribution to risks symmetrically; the validator checks
the full Δ computed from risks, not only the mean term.) ∎
[Validated: `regression_reflection`: Δ′ = −Δ to 1e−16; evidence stats identical.]

**(c) Binary, H-preserving (D-local complement).** For binary Y the swap in (a) is the
D-local label complement; it maps the class-conditional accuracies on D as q_j ↦ 1−q_j,
hence preserves CEI and per-class symmetry (hypothesis H) and maps b ↦ −b on D.
[Validated: `binary_H_swap`: agreement matrix c invariant (exact), b′ = −b, τ stays 0.]

---

## Theorem (Resolution of Conjecture 1: the testability dichotomy)

**(i) No testable bracketing — in any output space.** Let 𝒞 be any evidence-definable
class containing at least one target P with Δ(P) ≠ 0. Then sign Δ is not identifiable on
𝒞: by Lemma 1, P★ ∈ 𝒞 (same evidence law), and the pair (P, P★) shares one evidence law
with opposite benefit signs. Consequently *no label-free-checkable assumption brackets
the ordinal comparison of Conjecture 1* — for binary, multiclass, or regression, and not
merely for pairwise-agreement evidence but for the full observable algebra of every
order, even with unlimited labeled source data.

**(ii) The minimal untestable supplement is one bit, and it suffices (binary).** The
swap is an involution pairing each Δ≠0 target with an opposite-sign twin on the same
evidence fiber; identifying sign Δ on a class 𝒞 is exactly selecting at most one member
of each swap pair — a one-bit selection, irreducible by (i). Conversely, under the
falsifiable (never verifiable — by (i), necessarily) structural class H, one declared
bit identifies sign Δ for *every* candidate simultaneously at the minimax m^{−1/2} rate
(Theorems T-I, T-III of THEORY_V2_PROOFS.md). For multiclass one-coin / tensor-regular
families the same sufficiency holds by the γ-meter section's Theorem (multiclass
bracketing) with its stated anchors; the impossibility half (i) is unconditional.

*Proof.* (i) is Lemma 1 + the definition of evidence-definability (membership and any
decision rule are functionals of L; a rule outputs one action per evidence law, and the
two worlds demand opposite committal actions — the committal-regret argument of Theorem 1
applies verbatim, with the pair now constructed *inside* 𝒞). (ii) Pairing: the swap is
its own inverse and changes the sign, so orbits on {Δ≠0} have size 2 with opposite
signs; a sign-identifying class meets each orbit at most once (else two signs on one
fiber); selecting one member per orbit is by definition one bit of non-evidence
information. Sufficiency under H + bit is T-I(a)+(d) (identification up to flip; the bit
resolves the flip) with the rate from T-III. ∎

---

## Remark (the algebraic heart: observables are the even subalgebra)

For binary Y write v = 2Y−1 (latent) and u_j = 2g_j−1 (observable); correctness signs
satisfy s_j = u_j v. Every label-free statistic is a function of the u's: the
*even* part of the correctness algebra (products of an even number of s's, e.g.
s_i s_j = u_i u_j). The decision target sign Δ ∝ E[s_a − s_0] is *odd* (one factor of v).
The swap of Lemma 1(c) is exactly v ↦ −v on D: it fixes the even subalgebra (all
observables, all orders) and negates every odd element (every benefit sign). The
dichotomy is therefore structural: **what is observable is the even algebra; what is
decided is odd; the quotient is exactly one bit.** This also clarifies the γ-meter
section's reach hierarchy: odd-order correctness moments (e.g. E[s_1 s_2 s_3]) are
label-carrying — estimating them already presupposes the bit (an anchored frame) — so
the hierarchy r_2 > r_3 > ⋯ quantifies residual *magnitude* ambiguity within the
anchored frame, while the raw observable channel retains the sign bit at every order:
lim_k r_k-style programs can shrink magnitude ambiguity, never the bit. That answers the
question the γ-meter section leaves open ("how fast r_k shrinks beyond third order"):
whatever its rate, its limit cannot break the swap orbit.

## Remark (positioning — verified June 2026)

Rosenfeld & Garg (NeurIPS 2023, arXiv 2306.00312, Remark 3.9) assert informally that
error bounds under shift are impossible without assumptions; our Theorem (i) is the
formal version, with the closure-under-swap mechanism, the exact minimal residual (one
bit), and coverage of the full observable algebra. Ben-David–Lu–Luu–Pál (AISTATS 2010)
construct observably indistinguishable DA tasks for *learnability*, not benefit-sign
identification over testable classes. The testable-learning-with-shift line (Klivans et
al., arXiv 2311.15142) gives *positive* runtime certification for agnostic learning — a
different estimand; we cite to distinguish. The algebraic-evaluation program
(Corrada-Emmanuel, arXiv 2312.05392/2409.11052) notes a residual two-solution ambiguity
but proves no statement that *no testable condition resolves it*. The closest conceptual
precedent is Manski-style partial identification, where sign identification of a
treatment effect requires untestable monotonicity; our result is its exact analogue for
label-free adaptation, with the supplement quantified as one bit. A check for prior
statements of the dichotomy as a theorem over testable classes found none (live search,
June 2026).

## Weight (honest)

The swap constructions are elementary (each proof is three lines); the value is the
*characterization*: (a) it upgrades Corollary `irreducible` from "some supplied bound is
needed" to "no testable condition of any kind, at any evidence order, suffices — and the
untestable content is exactly one bit"; (b) it converts Conjecture 1 from "open" to
"resolved as a dichotomy" — the hoped-for unconditional label-free bracketing provably
does not exist, and the conditional resolutions (calibration-transfer Prop, γ-meter
multiclass theorem, H + bit) are now known to be *of the minimal possible form*:
falsifiable structure + one declared bit. Nothing here weakens the conditional positive
results; it certifies they cannot be improved in kind.
