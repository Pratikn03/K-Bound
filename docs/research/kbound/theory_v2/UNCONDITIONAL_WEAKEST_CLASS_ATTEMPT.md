CLOSED

# Unconditional characterization of the weakest falsifiable one-bit class

*Resolving the open piece of `conj:gen` / `rmk:genpos`: minimality of the one-bit
relative-calibration class **without** Assumption `asm:genpos` (General Position).*

Author: theory pass (claude-opus). Date: 2026-06-23.
Companion validator: `val_unconditional_weakest.py` (fixed seeds; fails loudly).

---

## 0. One-paragraph summary

The open problem (Remark `rmk:genpos`) asks for the weakest falsifiable class on which **one
declared structural bit** certifies `sign(Delta)`, *across the entire lattice of dominant-region
thresholds `rho`*, with General Position removed. We close it. The answer is an **exact,
unconditional, necessary-and-sufficient inequality** (a *dominance margin* on the unobserved
benefit magnitudes), and the weakest classes are an **explicit finite family of "dominance
polytopes."** The `rho`-dependent family `C_dom(rho)` that obstructed an unconditional statement
is not a new phenomenon: every `C_dom(rho<=1)` is a *proper subclass* of a single maximal
polytope `W*`, and `rho=1` is exactly the polytope's boundary. General Position is precisely the
hypothesis that collapses the finite family down to the single representative `C_mono`. The honest
nuance — stated up front, not hidden — is that there is **no unique weakest class**: the maxima
are pairwise incomparable. But that incomparability is now *fully characterized* (a finite,
explicit, combinatorial set), which is what closing the conjecture requires; it is no longer an
open "continuum of unknown thresholds." All claims are machine-checked by exact-arithmetic oracles
over `>=1e5` members; the criterion is shown to coincide with ground truth on `~95k` box fibres
and `~3.2k` arbitrary non-box polytope fibres, with `0` errors, and the validator fails loudly
(exit 1, 2395 mismatches) under a one-character corruption of the criterion.

---

## 1. Setup and exact problem statement (notation of `weakest_class.tex`)

Fix an **evidence fibre** `E`: the observable base measure `mu` on a finite partition of the
disagreement region `D` into cells `i = 1..n` (`mu_i > 0`, `sum_i mu_i = mu(D)`), and the
observable relative-calibration field `c_i = w_i (m_i - m*)`. A **target** is an *unobservable*
benefit vector `a in [-1,1]^n` with `Delta = sum_i mu_i a_i`; `sign(Delta)` is what one declared
bit must certify. We normalise `mu(D)=1` WLOG (it scales `Delta` positively and does not affect
its sign).

Two facts from the paper are load-bearing and used verbatim:

* **(Reduction.)** `Delta = sum_i mu_i a_i` (Lemma `lem:reduction`); only `sign(Delta)` matters.
* **(Swap involution.)** Within a fibre, on any cell whose benefit sign is label-free
  *unidentified*, flipping `sign(a_i)` is an **evidence-preserving (`TV=0`) operation**
  (Theorem `thm:conj1-dichotomy`(iii)): the label-free law is a function of `x` alone and never
  sees `a`. Magnitudes `|a_i|` are label-*identifiable* (a finite labelled probe estimates
  `eta_i = Pr_T(f_a=Y | cell i)`, hence `|a_i| = |2 eta_i - 1|`); the **global orientation** is
  the one thing labels never fix. This is the paper's own "falsifiable but not verifiable."

### 1.1 Canonical form of a falsifiable one-bit class (Lemma 1 below)

> **Definition (orientation pattern; canonical class).** An *orientation pattern* on `E` is a
> choice of a **tied set** `P` (a subset of `{i : c_i != 0}`) together with a **tied sign-pattern**
> `s in {+,-}^P` (each tied cell independently *aligned* `s_i = +` or *anti-aligned* `s_i = -`
> with `c_i`); the **free set** is `Z = {1..n} \ P`. The associated **canonical class** with
> magnitude region `W subset [0,1]^n` is
> ```
> C(P, s, W) = { a :  sign(a_i) = sigma * s_i * sign(c_i)  for i in P,
>                     (|a_j|)_j in W },       sigma in {+1,-1} the single declared bit.   (CANON)
> ```
> (Aligned ties `s_i=+` are the `C_mono`-type ties of the paper; anti-aligned ties `s_i=-` are
> *also* falsifiable — a labelled probe cannot reject either, since `sigma` is unknown — and they
> contribute additional incomparable maxima. The global flip `sigma -> -sigma` flips the whole
> sign-pattern at once, so the maxima are indexed by `s in {+,-}^P` *up to* that global flip; the
> decoder's one declared bit is `sigma`.)

**Lemma 1 (every falsifiable one-bit class is canonical).**
A class `C` is *falsifiable* (label-free testable, i.e. evidence-definable up to the single
declared bit) **iff** it has the form `(CANON)` for some orientation pattern `(P, pi)` with
`P subset {c != 0}` and some evidence-definable magnitude region `W`.

*Proof.* `(<=)` `(CANON)` is rejected by a finite labelled probe exactly when an estimated
magnitude vector lands outside `W` or a tied cell's estimated sign contradicts
`sigma*sign(c_i)` for **both** values of `sigma` (the bit is existentially quantified). Both are
checkable from labelled magnitudes, so `(CANON)` is falsifiable.

`(=>)` Let `C` be falsifiable. Falsifiability means membership is decided by an
evidence-definable predicate applied to the label-free observables together with the one declared
bit. By the swap involution, for any cell `i` whose sign is not pinned by `c` *and the bit*,
flipping `sign(a_i)` is `TV=0`; an evidence-definable predicate cannot distinguish `a` from its
flip, so the predicate restricted to such cells depends only on `|a_i|`. Hence the cells split
into (a) cells where the predicate forces a sign — necessarily `sign(a_i)=sigma*sign(c_i)`, which
is the *only* sign information a single global bit plus the observable `c` can encode (forcing any
other sign is not evidence-definable, see Lemma 2) — call these `P`; and (b) the rest `Z`, on
which the predicate sees only `|a_i|`. The residual constraint is therefore a predicate on
`(|a_j|)_j`, i.e. a region `W`. This is `(CANON)`. ∎

**Lemma 2 (anchors are necessarily free).** If `c_i = 0` then `i` cannot lie in `P` of any
falsifiable class. *Proof.* `c_i=0` carries no observable margin signal, so the sign of `a_i` is
unidentified and flipping it is `TV=0`. A class that forced `sign(a_i)` would contain `a` but not
its `TV=0` flip, contradicting evidence-definability. ∎ (Machine-checked: test **E1**.)

> **Interpretation.** Lemma 1 is the crucial generalisation over the prior treatment, which
> implicitly assumed `W` is a **box** `prod_i [lo_i, hi_i]` (per-cell magnitude windows). Lemma 1
> shows the *most general* falsifiable class allows `W` to be **any** evidence-definable region —
> in particular a non-product region that *couples* magnitudes across cells (e.g.
> `{ |a_1| <= |a_0| }`). This is exactly the freedom that General Position suppresses, and it is
> where the unconditional answer lives. Coupling of *signs* across free cells, or of a free sign
> to the bit, is **impossible** for a falsifiable class (Lemma 2's argument), so the box
> restriction is a real loss of generality only in the **magnitude** directions.

---

## 2. The exact unconditional criterion

Fix a fibre and an orientation pattern `(P, s, Z)`. At declared bit `sigma = +1` put
```
T(r) = sum_{i in P} s_i sign(c_i) mu_i r_i    (tied block, signed by c and the chosen pattern s)
G(r) = sum_{i in Z} mu_i r_i  >= 0            (free block half-width)
```
where `r = (|a_i|)_i in W`. For **fixed magnitudes** `r`, the tied signs are determined by
`sigma` while each free sign ranges independently over `{+,-}` (swap involution), so
`Delta` ranges over the interval `[ T(r) - G(r),  T(r) + G(r) ]`. Taking the union over `r in W`,
`Delta` ranges over `[ inf_{W}(T - G),  sup_{W}(T + G) ]`; the `sigma=-1` fibre is the exact
negation (tied block negates, free block is already sign-symmetric). Hence:

> **Theorem 1 (unconditional one-bit criterion — dominance margin).**
> One declared bit certifies `sign(Delta)` on `C(P, W)` **iff**
> ```
>   [ inf_{r in W} ( T(r) - G(r) ) >= 0   AND   sup_{r in W} ( T(r) + G(r) ) > 0 ]    (pin +)
>     OR
>   [ sup_{r in W} ( T(r) + G(r) ) <= 0   AND   inf_{r in W} ( T(r) - G(r) ) < 0 ].   (pin -)
> ```
> (The second conjunct in each line is the non-vacuity caveat: some member must have
> `Delta != 0`, else the bit is unused and the question is degenerate.)

*Proof.* `Delta` is linear, hence its range over the box-fibered set is the displayed interval;
"one bit certifies" means each `sigma`-fibre's *strict* sign set has size `<= 1`, which for an
interval is exactly the displayed sign condition, and the `sigma=-1` fibre mirrors `sigma=+1` so a
single inequality suffices. ∎

This is **tautological once the interval form is granted**, which is the point: the only
non-trivial content is Lemma 1 (that *every* falsifiable class is captured by some `(P, Z, W)`).
The criterion is **exact** and uses **no assumption** on `W`.

**Specialisation to boxes (reconciliation with the prior result).** If `W = prod_i [lo_i, hi_i]`,
the linear `inf/sup` are attained at explicit corners:
`inf_W(T-G) = sum_{P,c>0} mu_i lo_i - sum_{P,c<0} mu_i hi_i - sum_Z mu_i hi_i = Umin - F`, and
`sup_W(T+G) = Umax + F`, recovering **verbatim** the prior `DOMINANCE MARGIN`
`(Umin - F >= 0) or (Umax + F <= 0)` of the earlier box-only validator. So Theorem 1 *contains*
the prior characterization as its box case and extends it to arbitrary falsifiable classes.
(Machine-checked: test **G**, `0` mismatches on `1e5` box fibres.)

---

## 3. The weakest classes (minimality, unconditionally)

> **Theorem 2 (the weakest falsifiable one-bit class).**
> Fix a fibre and an orientation pattern `(P, Z)` with pin `+`. Among all falsifiable one-bit
> classes with that pattern, there is a **unique maximal** (weakest) one, the **dominance
> polytope**
> ```
>   W* = { r in [0,1]^n :  T(r) >= G(r) },        C* = C(P, W*).        (pin +)
> ```
> (For pin `-`: `W* = { r : T(r) <= -G(r) }`.) Concretely:
> 1. **`C*` is one-bit:** `inf_{W*}(T - G) >= 0` by construction, so Theorem 1 certifies it.
> 2. **`C*` is maximal:** for any `r0` with `T(r0) < G(r0)`, adding `r0` to `W*` admits a member
>    with `Delta < 0` (take the free signs opposing the tied block) while `W*` already admits
>    `Delta > 0`; both signs now occur at fixed bit, so one bit no longer certifies.
> 3. **Subsumption:** `C_mono` (the case `W = {r : r_i = |c_i| on P, r_j = 0 on Z}`), every
>    `C_dom(rho<=1)`, and the prior "continuum of incomparable box maxima" are all **proper
>    subclasses of the single `C*`**. (They are sub-boxes / sub-rays of the polytope.)
> 4. **The set of weakest classes is the finite, explicit family** indexed by a tied set
>    `P subset {c != 0}` and a tied sign-pattern `s in {+,-}^P` (each tied cell aligned or
>    anti-aligned with `c`), modulo the global flip `sigma -> -sigma`. Distinct patterns give
>    **pairwise incomparable** maxima in general (Theorem 3), so there is **no unique weakest
>    class** — but the family is finite (at most `sum_{k} C(|{c!=0}|, k) 2^k = 3^{|{c!=0}|}`
>    patterns) and fully enumerated. `C_mono` is the all-cells-tied, all-aligned member
>    (`P = {c != 0}`, `s = +`).

*Proof.* (1)–(2) are immediate from Theorem 1 and linearity (the maximality witness is explicit).
(3) For `C_mono` the magnitudes are `r_i = |c_i|` on `P` and `0` on `Z`, so `G = 0 <= T`, hence
`r in W*`; `C_mono` is the slice `Z`-magnitudes`=0`. For `C_dom(rho)` on the canonical fibre
`c=(+1,0)`, `mu=(1/2,1/2)`, `P={0}`, `Z={1}`: membership is `r_0 = 1`, `r_1 <= rho`, and
`T - G = (r_0 - r_1)/2 = (1 - r_1)/2 >= 0` iff `r_1 <= 1`, i.e. `C_dom(rho) subset W*` iff
`rho <= 1`, with `rho = 1` the boundary. The prior box maxima `C_t = { r_0 = t, r_1 <= t }` lie in
`W* = { r_0 >= r_1 }` and are proper sub-boxes. (4) is Theorem 3. ∎ (Machine-checked: **C, D, E2,
E3, F**.)

> **Theorem 3 (no unique weakest class — but the family is explicit).**
> Distinct orientation patterns generally give incomparable maxima. Example (fibre `c=(1,1,0)`,
> `mu=(1/3,1/3,1/3)`): pattern **A** = tie `{0,1}` (free `{2}`) and pattern **B** = tie `{0}`
> (free `{1,2}`) have target sets `C*_A`, `C*_B` with `C*_A !subset C*_B` and `C*_B !subset C*_A`.
> Witnesses: `a = (0, 1/3, 0) in C*_A \ C*_B` (cell 1 aligned with magnitude `B` forbids, since
> `mu_0·0 < mu_1·(1/3)`); `a = (1, -1/3, 1/3) in C*_B \ C*_A` (cell 1 sign-flipped, which `A`
> forbids). Hence no single class dominates; the weakest classes form the finite family of
> Theorem 2(4). ∎ (Machine-checked: test **E3**, with these exact witnesses.)

### 3.1 What General Position was doing all along

Assumption `asm:genpos` requires that the free regions carry "comparable, not-fully-`E`-pinned"
benefit mass with no `E`-certifiable dominant subcollection. In the language above, that is exactly
the requirement that the **free block can overpower the tied block** on the fibre, i.e. that the
dominance margin `inf_{W}(T - G)` is *not* automatically `>= 0` unless `Z`-magnitudes vanish. Under
that requirement the only one-bit class is `C_mono` (`Z`-magnitudes `= 0`), recovering Theorem
`thm:cmono-weakest`(iii). Dropping it reveals the **whole dominance polytope** `W*` (and the finite
family across patterns). So:

* The prior **conditional** statement is *correct as stated* (it is the General-Position face).
* The **obstruction** named in `rmk:genpos` — "characterising minimality across the entire lattice
  of dominant-region thresholds `rho`" — is resolved: the `rho`-lattice is the 1-parameter slice
  `{ r_0 = 1, r_1 = rho }` of the single polytope `W* = {r_0 >= r_1}`; `rho = 1` is the boundary;
  there is no mysterious threshold family, only one polytope per pattern.

---

## 4. Adversarial analysis (actively trying to break it)

Per the discipline, for every proposed characterization I tried to construct two
**evidence-identical (`TV=0`)** targets sharing the declared bit but with **opposite**
`sign(Delta)`, *inside* a class the criterion calls one-bit. Findings:

1. **Coupling counterexample to the box-only claim (this is what forces Theorem 1, and is *not* a
   flaw).** The class `{ |a_1| <= |a_0| }` on `c=(+1,0)`, `mu=(1/2,1/2)` is falsifiable
   (swap-closed: depends only on magnitudes), one-bit (`Delta = (a_0 + a_1)/2`, `|a_1| <= |a_0|`
   forces `sign = sigma`), yet its **bounding box violates** the prior `DOMINANCE` inequality
   (`Umin - F = 0 - 1/2 < 0`). This is a genuine falsifiable one-bit class the *box-only*
   characterization wrongly rejects, proving the box characterization is **not** the unconditional
   answer. Theorem 1 (exact `inf/sup`, here `inf_W(T-G) = inf{(r_0-r_1)/2 : r_1<=r_0} = 0 >= 0`)
   correctly accepts it. This example is exactly `C* = W*` for this fibre.

2. **Anchor-tie / sign-coupling attempts fail to be falsifiable (Lemma 2).** Any attempt to beat
   `W*` by forcing a free/anchor sign (e.g. "force `sign(a_1) = +`") is **not** swap-closed: the
   `TV=0` anchor flip `a_1 -> -a_1` produces an evidence-identical target *outside* the class, so
   the class is not evidence-definable, hence not falsifiable. It therefore cannot be realised as a
   structural class and does not count as a one-bit class. (Test **E1**: member in `= True`,
   `TV=0`-flip in `= False`.)

3. **The only "breaks" found were artifacts of *numerical* `inf/sup`, not of the math.** A
   first-pass adversarial sweep that approximated `inf/sup` by random *sampling* of the polytope
   reported breaks on mixed-sign tied classes with linear constraints. Exact LP on the worst case
   (`c=(1,1,-1)`, polytope `A r <= b`) showed the class **genuinely is not one-bit** (`phi(r)`
   ranges over `[-0.0365, 0.797]`, attaining both signs at the vertex `r=(0,0,0.18)` that sampling
   missed), and the **exact** criterion correctly rejects it. Lesson encoded in the validator:
   `inf/sup` over `W` must be computed **exactly** (closed-form corner for boxes; LP for
   polytopes), never by sampling. With exact `inf/sup`, the corrected adversarial hunt finds
   **`0` breaks** over the criterion-positive classes (test **H**).

4. **Direct large-scale hunt (test H).** Over arbitrary `n=2..4` fibres with random
   integer/Gaussian half-space magnitude regions (box and non-box), using the **exact-LP**
   criterion, `91` classes were declared one-bit and **`0`** admitted a same-bit opposite-sign
   `TV=0` pair under `~5.5e5` member draws.

5. **Self-caught refinement (anti-aligned ties enlarge the finite family).** A further
   adversarial pass asked whether a falsifiable class may tie a cell *anti-aligned*
   (`sign(a_i) = sigma * (-sign(c_i))`). It can: a labelled probe cannot reject it (since `sigma`
   is unknown), so anti-aligned ties are falsifiable and contribute **additional incomparable
   maxima**. On `c=(1,1)` the four tied sign-patterns `{+,-}^2` give four pairwise-incomparable
   maximal target sets. This does **not** threaten any claim — the family is still finite and
   explicit (`<= 3^{|{c != 0}|}` patterns) — but it means the correct index set is the tied
   **sign-pattern** `s in {+,-}^P`, not merely "tie to `sigma*sign(c)`". The definitions and
   theorems above already use `s`; this is recorded so the index set is stated correctly and not
   understated. Theorem 1's criterion is unchanged (it uses the chosen tied signs in `T`).

No counterexample survives. The characterization stands.

---

## 5. Validator and ACTUAL numerical results

`val_unconditional_weakest.py` (seeds fixed at `SEED=20260623`; numpy + scipy + fractions). It
uses **exact** oracles for every load-bearing equivalence (rational vertex enumeration and
exact-LP `inf/sup`), and **fails loudly** (nonzero exit) on any violation. Run in sandbox
(`python3`, scipy 1.15.3), total wall time **38.1 s**:

```
[A] (CRIT) interval-criterion == exact vertex-truth, BOX classes
    non-vacuous box fibres tested : 94882
    criterion vs vertex-truth mism: 0        -> require 0      PASS

[B] (CRIT) exact-LP criterion == exact vertex-truth, NON-BOX integer polytopes
    non-vacuous polytope fibres tested : 3216  (with inequalities: 2140)
    exact-LP criterion vs vertex-truth : 0   -> require 0      PASS

[C] SUFFICIENCY on the weakest class W*={T(r)>=G(r)}
    members of W* tested   : 120000
    one-bit decision errors: 0               -> require 0      PASS

[D] NECESSITY: enlarging W* past the dominance margin loses one bit
    just-outside members tested : 36221
    one-bit decision error rate : 0.50269    -> require > 0    PASS

[E] STRUCTURE
    (E1) anchor-tie class: member in=True, TV=0-flip in=False -> not falsifiable: True
    (E2) C_mono,C_dom(1) subset W* and W* strictly larger:    True
    (E3) two tie-patterns give incomparable maxima:          True            PASS

[F] C_dom(rho) lattice (collapses rmk:genpos's rho-family into W*)
    rho= 0.0000: error rate 0.00000   (subset W* / ONE-BIT)
    rho= 0.5000: error rate 0.00000   (subset W* / ONE-BIT)
    rho= 0.9000: error rate 0.00000   (subset W* / ONE-BIT)
    rho= 1.0000: error rate 0.00000   (subset W* / ONE-BIT)   <- boundary
    rho= 1.0001: error rate 0.00003   (leaves W* / LOST)
    rho= 1.5000: error rate 0.16681   (leaves W* / LOST)
    rho= 3.0000: error rate 0.33487   (leaves W* / LOST)                     PASS

[G] RECONCILIATION: on BOX classes (CRIT) == prior DOMINANCE-MARGIN
    box fibres compared        : 100000
    (CRIT) vs prior DOMINANCE  : 0           -> require 0      PASS

[H] ADVERSARIAL: same-bit opposite-sign TV=0 pair inside a criterion-one-bit class
    classes declared ONE-BIT (exact LP): 91
    same-bit opposite-sign TV=0 breaks : 0   -> require 0      PASS

ALL CHECKS PASSED -- the unconditional characterization holds on every test.
```

**Loud-failure check (anti "passes-by-construction").** Corrupting the box criterion by a single
character (`inf_TmG >= 0` -> `inf_TmG > 0`) makes the validator report
`criterion vs vertex-truth mism: 2395` and `(CRIT) vs prior DOMINANCE: 2335`, prints
`!!! FAIL ...`, and **exits 1**. The validator genuinely tests the claim.

---

## 6. Verdict, scope, and exactly what is (and is not) claimed

**VERDICT: CLOSED.** Theorem 1 gives an exact, unconditional, necessary-and-sufficient condition
for one-bit certifiability on *any* falsifiable class; Theorem 2 identifies the weakest such
classes (the dominance polytopes `C*`); Theorem 3 + Theorem 2(4) characterise the full set of
maxima as an explicit finite family. This resolves the open piece of `conj:gen` /
`rmk:genpos` *without* General Position, which is recovered as the special face on which the family
collapses to `C_mono`.

**Honest scope — read before merging into the paper:**

* The closure is **not** "there is a unique weakest class." There is **not** one; the maxima are
  pairwise incomparable (Theorem 3). The closure is "the necessary-and-sufficient one-bit condition
  is exactly the dominance margin, and the weakest classes are exactly the dominance polytopes,
  forming an explicit finite family." That *is* a complete characterization of minimality — which
  is what the conjecture asked — but it must be **stated as the finite-family characterization**,
  not as a uniqueness claim. Overstating it as a unique class would be false.
* Theorem 1 is exact and assumption-free; its **only** non-tautological input is **Lemma 1** (every
  falsifiable class is canonical). Lemma 1 rests on the paper's own swap-involution
  (`thm:conj1-dichotomy`(iii)) and on magnitudes being label-identifiable while the global
  orientation is not — both already in the manuscript. A reviewer should sanity-check Lemma 1's
  `(=>)` direction against the paper's exact definition of "falsifiable/evidence-definable class";
  it is the place where the generality lives.
* The validator's `inf/sup` are **exact** (closed-form corner for boxes; LP for polytopes). The
  characterization is *false if implemented with sampled `inf/sup`* — this is a numerical, not a
  mathematical, caveat (Section 4.3), and is the reason the prior validator soundly restricted to
  boxes (closed-form corners). It is documented so nobody re-introduces sampling.

---

## 7. Drop-in LaTeX (augments part (iii) of `thm:cmono-weakest`)

The following is written in the existing notation and is ready to drop into
`weakest_class.tex` (it strengthens (iii) from conditional to unconditional and adds the
dominance-margin theorem). It is **only** to be used because the result is closed; it states the
finite-family characterization honestly and does not claim uniqueness.

```latex
% ---- Drop-in: unconditional weakest one-bit class (augments thm:cmono-weakest(iii)) ----
\begin{definition}[Orientation pattern and canonical class]\label{def:orient}
On a fibre $E$ an \emph{orientation pattern} is a tied set $P\subseteq\{i:c_i\neq0\}$ with a tied
sign-pattern $s\in\{\pm\}^{P}$; the free set is $Z=D\setminus P$. For an observable magnitude region
$W\subseteq[0,1]^{|D|}$ the \emph{canonical class} is
$\mathcal C(P,s,W)=\{a:\ \sign a_i=\sigma\,s_i\sign c_i\ (i\in P),\ (|a_j|)_j\in W\}$ with one
declared bit $\sigma$. Write $T(r)=\sum_{i\in P}s_i\sign(c_i)\mu_i r_i$ and
$G(r)=\sum_{i\in Z}\mu_i r_i$.
\end{definition}

\begin{lemma}[Canonical form of a falsifiable class]\label{lem:canonform}
A class is falsifiable \emph{iff} it equals some $\mathcal C(P,s,W)$; moreover anchors ($c_i=0$)
lie necessarily in $Z$. \textnormal{(Swap-closure (Thm.~\ref{thm:conj1-dichotomy}(iii)) forces the
constraint on $Z$ to depend only on $|a|$; forcing an anchor sign is not evidence-definable. Both
aligned ($s_i{=}+$) and anti-aligned ($s_i{=}-$) ties are falsifiable since $\sigma$ is
unidentified.)}
\end{lemma}

\begin{theorem}[Unconditional one-bit criterion; weakest classes]\label{thm:uncond-weakest}
Let $\mathcal C(P,s,W)$ be falsifiable. \textnormal{(i) (Dominance margin.)} One declared bit
certifies $\sign\Delta$ on $\mathcal C(P,s,W)$ \emph{iff}
$\inf_{r\in W}\!\big(T(r)-G(r)\big)\ge0$ (with some member $\Delta>0$) or
$\sup_{r\in W}\!\big(T(r)+G(r)\big)\le0$. \textnormal{(ii) (Weakest class.)} For a fixed pattern
$(P,s)$ the unique maximal one-bit class is the dominance polytope
$\mathcal C^\star=\mathcal C(P,s,W^\star)$, $W^\star=\{r\in[0,1]^{|D|}:T(r)\ge G(r)\}$;
$\mathcal C_{\mathrm{mono}}$, every $\mathcal C_{\mathrm{dom}}(\rho\le1)$ and every box one-bit
class are proper subclasses of one $\mathcal C^\star$. \textnormal{(iii) (Finite family.)} The set
of weakest falsifiable one-bit classes is the explicit finite family $\{\mathcal C^\star\}$ indexed
by $(P\subseteq\{c\neq0\},\,s\in\{\pm\}^{P})$ modulo the global flip $\sigma\mapsto-\sigma$ (at most
$3^{|\{c\neq0\}|}$ patterns); distinct patterns give pairwise incomparable maxima, so there is no
\emph{unique} weakest class. $\mathcal C_{\mathrm{mono}}$ is the all-tied all-aligned member, and
Assumption~\ref{asm:genpos} is exactly the face on which the family collapses to
$\mathcal C_{\mathrm{mono}}$, recovering Thm.~\ref{thm:cmono-weakest}(iii).
\end{theorem}
\begin{proof}
$\Delta$ is linear, so over $\mathcal C(P,s,W)$ at bit $\sigma=+1$ it ranges over
$[\inf_W(T-G),\sup_W(T+G)]$ (free signs independent by swap-closure; the $\sigma=-1$ fibre is its
negation); a single bit certifies iff each fibre's strict sign set is a singleton, giving (i). For
$W^\star$, $\inf_{W^\star}(T-G)\ge0$ by construction (one-bit), and any $r_0$ with $T(r_0)<G(r_0)$
admits, with opposing free signs, a member of $\Delta<0$ while $W^\star$ admits $\Delta>0$, so
$W^\star$ is maximal; $\mathcal C_{\mathrm{dom}}(\rho)$ sits in $W^\star$ iff $\rho\le1$, with
$\rho=1$ the boundary, giving (ii). Incomparability (iii) is witnessed on $c=(1,1,0)$ by
$a=(0,\tfrac13,0)$ and $a=(1,-\tfrac13,\tfrac13)$. On the general-position face the free block can
overpower the tied block unless $Z$-magnitudes vanish, leaving only $\mathcal C_{\mathrm{mono}}$.
\end{proof}
% ---- end drop-in ----
```

*Note for the LaTeX integrator:* keep the existing `\begin{remark}[...]{rmk:genpos}` but replace
its final two sentences ("We leave this open and do not claim it ... sidesteps by fiat.") with a
forward reference to `Theorem~\ref{thm:uncond-weakest}`, e.g. "This is resolved unconditionally in
Theorem~\ref{thm:uncond-weakest}: the $\rho$-lattice is the slice $\{r_0=1,r_1=\rho\}$ of the
single polytope $W^\star=\{r_0\ge r_1\}$, with $\rho=1$ its boundary." Update `conj:gen`'s last
sentence from "remains open" to "is resolved (Theorem~\ref{thm:uncond-weakest})." Do **not** change
parts (i)–(ii) of `thm:cmono-weakest`, which are already unconditional.
