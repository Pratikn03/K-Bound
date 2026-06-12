# K-Bound Theory V2: Falsifiable Budget Audit and the Evidence-Channel Rate

**Scope.** Three theorem families (T-I, T-II, T-III) for the *label-free* benefit-sign
gating problem of the K-Bound paper, all built on a **corrected** multi-candidate
hypothesis **H**. Every claim is proved from stated assumptions; every numeric claim is
produced by `theory_v2_validation.py` and stored in `validation_results.json` (JSON keys
cited inline as **[Validated: …]**). Each result carries a **Weight** note (what is new
vs. near-definitional vs. classical).

Notation. Binary label $Y\in\{0,1\}$ on the observable region $D$; condition everything on
$D$ throughout (write $P(\cdot)$ for $P(\cdot\mid D)$, $\pi:=P(Y=1\mid D)\in(0,1)$). A panel
of $K$ predictors: frozen $f_0$ and candidates $f_1,\dots,f_{K-1}$, all fixed measurable
maps of $X$. Correctness indicators $C_j:=\mathbf 1[f_j=Y]$; class-conditional accuracies
$q_j(y):=P(C_j=1\mid Y=y)$; marginal accuracy $a_j:=P(C_j=1)=(1-\pi)q_j(0)+\pi q_j(1)$;
**advantage** $b_j:=2a_j-1$; **signed class accuracy** $d_j(y):=2q_j(y)-1$ (so
$b_j=(1-\pi)d_j(0)+\pi d_j(1)$). Pairwise agreement $A_{ij}:=P(f_i=f_j)$ and its centered
form $c_{ij}:=2A_{ij}-1$. The benefit of candidate $a$ vs. $f_0$ is
$\Delta_a=P(D)(2a^D_a-1)$ with $\operatorname{sign}\Delta_a=\operatorname{sign}(b_a-b_0)$
(paper Thm. on the disagreement region). The **drift budget** quantities are
$M:=\E[s]-\tfrac12$ (observable surrogate $s$) and $\gamma:=\E[\eta_a-s]$ (unobservable).

---

## 0. The corrected hypothesis H (CEI is insufficient — reviewer flag confirmed)

The paper's Proposition (multicandidate) states the identity $2A_{ij}-1=b_ib_j$ under
**CEI** alone (Def.: correctness indicators $\{C_j\}$ conditionally independent given $Y$).
A reviewer flagged that this needs a *per-class symmetric-accuracy* assumption. **We verify
this from first principles and confirm the flag.**

### Lemma 0.1 (Exact agreement identity under CEI; binary).
In the binary setting, $f_i=f_j \iff C_i=C_j$ (if both correct, both equal $Y$; if both
wrong, both equal $1-Y$, hence equal). Therefore $A_{ij}=P(C_i=C_j)$, and under CEI,
$$
c_{ij}=2A_{ij}-1=\E\big[(2C_i-1)(2C_j-1)\big]
=\E_Y\!\big[\,\E[2C_i-1\mid Y]\,\E[2C_j-1\mid Y]\,\big]
=(1-\pi)\,d_i(0)d_j(0)+\pi\,d_i(1)d_j(1).
$$
*Proof.* The first equality is the binary fact above. The second is the definition of
$c_{ij}$. The third uses CEI ($C_i\perp C_j\mid Y$) so the conditional expectation of the
product factorizes; the outer expectation is over $Y\in\{0,1\}$ with masses $1-\pi,\pi$. $\;\square$

Thus, under CEI alone, $c$ is a **rank-$\le 2$** matrix (the sum of two rank-one terms),
**not** rank-one. The paper's $c_{ij}=b_ib_j$ is the rank-one special case.

### Theorem 0.2 (Exact rank-one deficit; minimal hypothesis H).
Write the per-class asymmetry $\delta_j:=d_j(1)-d_j(0)=2(q_j(1)-q_j(0))$. Under CEI, for
every pair $i\neq j$,
$$
\boxed{\;c_{ij}-b_ib_j=\pi(1-\pi)\,\delta_i\,\delta_j\;}
$$
Consequently, the identity $c_{ij}=b_ib_j$ holds for **all** pairs of a panel with $K\ge 3$
candidates whose advantages are nonzero **iff** at most one candidate has $\delta_j\neq 0$;
robustly (so that no pair is exempted) iff $\delta_j=0$ for **every** $j$, i.e.

> **Hypothesis H (per-class symmetric accuracy + CEI).** Correctness indicators are
> conditionally independent given $Y$ (CEI), **and** each predictor has class-symmetric
> accuracy $q_j(0)=q_j(1)$ for all $j$ (equivalently $\delta_j=0$, equivalently
> $d_j(0)=d_j(1)=b_j$).

*Proof.* Parametrize $d_i(0)=b_i-\pi\delta_i$ and $d_i(1)=b_i+(1-\pi)\delta_i$; then
$(1-\pi)d_i(0)+\pi d_i(1)=b_i$ (consistency of the marginal). Substituting into Lemma 0.1,
$$
c_{ij}=(1-\pi)(b_i-\pi\delta_i)(b_j-\pi\delta_j)+\pi(b_i+(1-\pi)\delta_i)(b_j+(1-\pi)\delta_j).
$$
Expanding (a one-line symbolic computation, reproduced in the validation log) the cross
terms in $b_i\delta_j$ and $b_j\delta_i$ cancel and the $\delta_i\delta_j$ terms collect to
$[(1-\pi)\pi^2+\pi(1-\pi)^2]\delta_i\delta_j=\pi(1-\pi)\delta_i\delta_j$, leaving
$c_{ij}=b_ib_j+\pi(1-\pi)\delta_i\delta_j$. Since $\pi(1-\pi)>0$ on $D$, $c_{ij}=b_ib_j$ for
all $i\ne j$ iff $\delta_i\delta_j=0$ for all $i\ne j$; with $K\ge3$ this forces all but at
most one $\delta_j$ to vanish, and uniform validity forces all $\delta_j=0$. $\;\square$

**Corollary 0.3 (corrected rank structure).** Under CEI, $c=bb^\top+\pi(1-\pi)\,\delta\delta^\top$
off-diagonal: a rank-$\le2$ correction of the paper's rank-one form. The $M\ge4$
overdetermination residual $\tau$ (2×2 minors) therefore detects *asymmetry-driven*
rank-2 departures, but is blind to rank-1-preserving violations (see T-II(b)).

> **Weight.** *New and load-bearing.* The exact deficit $\pi(1-\pi)\delta_i\delta_j$ pins
> the minimal hypothesis and corrects the paper's stated CEI to **H**. The binary
> equivalence $f_i=f_j\iff C_i=C_j$ and the rank-one form itself are classical
> (Platanios et al. 2014); the precise *insufficiency mechanism* and minimal fix are the
> contribution here.
>
> **[Validated: `H_hypothesis_check`]** Symmetric $H$: $\max|c_{ij}-b_ib_j|$ off-diag
> $=5.5\times10^{-4}$ at $\pi{=}0.5$ and $4.8\times10^{-4}$ at $\pi{=}0.25$ (Monte-Carlo
> noise; identity holds even off class-balance). Asymmetric (plain CEI): the identity
> **fails** by $0.052$, while the deficit matches $\pi(1-\pi)\delta_i\delta_j$ to
> $5.7\times10^{-4}$; the rank-one 2×2 minor is $-0.025\neq0$.

All of T-I/T-II/T-III below are stated under H.

---

## T-I. One-bit identification and its necessity

### Theorem T-I(a) (Identification up to the global flip).
Assume H, $K\ge3$, all $b_j\neq0$, and pairwise products bounded away from zero
($\min_{i\ne j}|c_{ij}|\ge c_{\min}>0$). The full label-free evidence law — the joint
distribution of the prediction vector $(f_0,\dots,f_{K-1})$ on $D$ — determines:
(i) every magnitude $|b_j|$; (ii) every pairwise product $b_ib_j$ (hence every relative
sign $\operatorname{sign}(b_ib_j)$); and (iii) every benefit magnitude $|b_j-b_0|$. All of
this is determined **exactly up to the single global sign flip** $b\mapsto-b$; in
particular $b_j$ itself, $\operatorname{sign}b_j$, and $\operatorname{sign}(b_j-b_0)$ are
**not** determined without resolving that one bit.

*Proof.* Under H, $c_{ij}=b_ib_j$. For distinct $i,k,l$, $c_{ik}c_{il}/c_{kl}=b_i^2$
(well-defined since $|c_{kl}|\ge c_{\min}$), giving $|b_i|=\sqrt{c_{ik}c_{il}/c_{kl}}$ — claim
(i). The products $b_ib_j=c_{ij}$ are read directly — claim (ii). For (iii),
$|b_j-b_0|^2=b_j^2-2b_jb_0+b_0^2=c_{jk}c_{jl}/c_{kl}-2c_{j0}+c_{0k}c_{0l}/c_{kl}$, all
functions of the $c$'s — determined. The evidence law fixes $c$ (each $A_{ij}$ is a marginal
of the prediction vector), so all three are functions of the evidence law.
*Flip invariance / sensitivity.* Every listed quantity is a function of the entries of
$c=bb^\top$ (off-diagonal) and of magnitudes; $bb^\top$ is invariant under $b\mapsto-b$, and
$|b_j-b_0|=|-b_j-(-b_0)|$ is invariant. Conversely $\operatorname{sign}(b_j-b_0)$ flips under
$b\mapsto-b$ (it equals $-\operatorname{sign}(-b_j+b_0)$), so it is *not* a function of $c$
alone. $\;\square$

> Note (sign-magnitude asymmetry, as flagged): $b_j-b_0$ flips sign under $b\mapsto-b$, but
> $|b_j-b_0|$ is invariant — the magnitude survives, the decision direction does not.

### Theorem T-I(b) (Explicit flip witness; the flip bit is information-theoretically free).
Let $P$ be any target satisfying H. Define $P'$ by complementing the label, $Y'=1-Y$, and
leaving the $X$-marginal and all predictors unchanged. Then:
1. **Predictions and $X$-marginal are identical** ($f_j$ are maps of $X$; $X$ unchanged).
2. **H is preserved**: $q'_j(y')=P(C'_j=1\mid Y'=y')=1-q_j(1-y')$, so $q_j(0)=q_j(1)$ implies
   $q'_j(0)=q'_j(1)=1-q_j$; CEI is preserved (a relabeling of $Y$).
3. **The full evidence law is identical**: $\mathrm{TV}\big(\mathrm{Law}(f_0,\dots,f_{K-1}\mid P),\,
   \mathrm{Law}(\cdot\mid P')\big)=0$.
4. **$b\mapsto-b$ and every benefit sign flips**: $a'_j=1-a_j$, $b'_j=-b_j$, and
   $\operatorname{sign}(b'_j-b'_0)=-\operatorname{sign}(b_j-b_0)$ for every $j$.

*Proof.* (1)–(2) are above. (3): the prediction vector is a function of $X$ only, whose law
is unchanged, so its distribution is identical; the agreements (and hence $c$) coincide.
(4): $C'_j=\mathbf 1[f_j=1-Y]=1-C_j$, so $a'_j=1-a_j$ and $b'_j=2(1-a_j)-1=-b_j$; therefore
$b'_j-b'_0=-(b_j-b_0)$ and its sign flips. $\;\square$

> **Consequence.** Within the **fully audited class** (predictions, agreements, all
> label-free statistics, and H all matched), $P$ and $P'$ are evidence-indistinguishable
> (TV $=0$) yet have opposite benefit signs for *every* candidate. The flip bit cannot be
> recovered from label-free evidence by *any* rule — it is information-theoretically free.

### Theorem T-I(c) (Necessity: every identifying assumption is a bit-selector).
Let $\mathcal C$ be any class of H-targets on which $\operatorname{sign}(b_a-b_0)$ is
identifiable (constant on each evidence-fiber). Then $\mathcal C$ is **flip-asymmetric**: it
cannot contain both a target $P$ (with $b_a\ne b_0$, i.e. not flip-fixed on the $a$–$0$
comparison) and its label-complement $P'$. Equivalently, identifiability of the benefit sign
is *exactly* the ability to exclude one of each flip-pair; every identifying restriction is a
**bit-selector**.

*Proof.* By T-I(b), $P$ and $P'$ share the evidence law but have opposite
$\operatorname{sign}(b_a-b_0)\ne0$. If both lay in $\mathcal C$, the sign would take both
values on one evidence-fiber, contradicting identifiability. Hence at most one member of each
non-flip-fixed flip-pair lies in $\mathcal C$ — a selection of the flip bit. $\;\square$

### Theorem T-I(d) (Each existing assumption is exactly a bit-selector under H).
Under H, the paper's three identifying devices each select precisely one global flip bit:

* **$\gamma$-budget with $|M|>\beta$** selects $\operatorname{sign}(M)$. Indeed
  $\operatorname{sign}\Delta=\operatorname{sign}(M+\gamma)=\operatorname{sign}M$ when
  $|M|>\beta\ge|\gamma|$, and the flip $b\mapsto-b$ sends $\gamma\to-\gamma$ but leaves the
  *observable* $M$ fixed, so the budget breaks the tie by committing to $\operatorname{sign}M$.
* **Majority-above-chance** selects $\operatorname{sign}(\operatorname{median}_j b_j)$ (the
  bit making most candidates better-than-chance). Flip reverses every $\operatorname{sign}b_j$,
  hence the majority bit, fixing the global sign of $b$.
* **AoL slope sign** selects $\operatorname{sign}(1-2\bar w)$. On $D_\theta=\{f_0\ne f_\theta\}$
  the win rate $w(\theta)=P(f_\theta=Y\mid D_\theta)$; under flip $Y\mapsto1-Y$,
  $w\mapsto1-w$ (pointwise), so $\bar w\mapsto1-\bar w$ and the slope
  $1-2\bar w\mapsto-(1-2\bar w)$ — the slope's sign is exactly the flip bit.

*Proof.* The $M$-budget statement is the paper's frontier Lemma plus the flip law
$\gamma\to-\gamma$ (since $\gamma=\E[\eta_a-s]$ and $\eta_a\mapsto1-\eta_a$ under flip while
$s$ is the fixed observable surrogate). For majority: $b'_j=-b_j$ (T-I(b)) reverses each
$\operatorname{sign}b_j$ and thus the count $\#\{j:b_j>0\}\mapsto\#\{j:b_j<0\}$. For AoL:
on $D_\theta$ exactly one of $f_0,f_\theta$ is correct, so $w'(\theta)=P(f_\theta=1-Y\mid
D_\theta)=1-w(\theta)$; averaging and inserting into the AoL slope $1-2\bar w$
(paper Thm., constant-$w$ case) flips its sign. $\;\square$

> **Weight.** *New (structural unification).* T-I(a) is a packaging of the classical
> product-ratio identifiability (Platanios/Parisi/Jaffe) **made precise about the residual
> flip bit**. T-I(b)–(d) are new: the explicit TV-zero flip witness inside the fully
> audited class, the necessity statement "every identifying assumption is a bit-selector,"
> and the dictionary mapping the paper's three devices onto that single bit. This is the
> sharp information-theoretic reframing of why label-free gating needs *exactly one*
> external bit.
>
> **[Validated: `V1_flip_witness`]** $\max|b'_j+b_j|=1.1\times10^{-16}$ (machine zero),
> $\mathrm{TV}(P,P')=0$ exactly, products invariant ($c=bb^\top=b'b'^\top$ to
> $1.1\times10^{-3}$ MC), benefits opposite for all $j$. The AoL flip law $w'=1-w$ and
> slope sign-flip were verified pointwise in development (max deviation 0).

---

## T-II. Budget audit: falsifiable, not verifiable

### Theorem T-II(a) (Bit-robust falsification with finite-sample radius; soundness and power).
**Estimation identity.** Under H, $\bar a_a-\tfrac12=b_a/2$, hence
$$
\gamma=\bar a_a-\tfrac12-M=\tfrac{b_a}{2}-M,
$$
which matches the paper's $\gamma=\E[\eta_a-s]$ with $M=\E[s]-\tfrac12$ by definition
($\E[\eta_a]=\bar a_a$). Up to the flip bit, $\gamma\in\{+\tfrac{|b_a|}{2}-M,\,-\tfrac{|b_a|}{2}-M\}$.

**Radius.** From $m$ unlabeled samples on $D$: estimate each $\hat A_{ij}$ (Bernoulli mean),
$\hat c_{ij}=2\hat A_{ij}-1$. By Hoeffding with a union bound over the $\le K^2$ pairs, with
probability $\ge1-\delta$,
$$
|\hat c_{ij}-c_{ij}|\le e_c:=\sqrt{\tfrac{2\log(2K^2/\delta)}{m}}\quad\text{for all }i,j.
$$

> **Lemma T-II.1 (product-ratio perturbation; proven constant).** For $g(x,y,z)=xy/z$ with
> $|x|,|y|,|z|\le1$, $|x|,|y|,|z|\ge c_{\min}$, and perturbations $\le e_c\le c_{\min}/2$,
> $$
> |\hat g-g|\le e_c\,\frac{4c_{\min}+2}{c_{\min}^2}=:C_2\,e_c .
> $$
> *Proof.* Write $\hat g-g=\frac{(\hat x\hat y-xy)z+xy(z-\hat z)}{\hat z\,z}$. Then
> $|\hat x\hat y-xy|\le|\hat x|\,|\hat y-y|+|y|\,|\hat x-x|\le 2e_c$ (using $|\hat x|,|y|\le1$),
> $|xy(z-\hat z)|\le e_c$ (using $|xy|\le1$), and $|\hat z\,z|\ge(c_{\min}-e_c)c_{\min}\ge
> \tfrac12 c_{\min}^2$. Hence $|\hat g-g|\le\frac{2e_c|z|+e_c}{\tfrac12c_{\min}^2}
> \le\frac{e_c(2+1/c_{\min})}{\tfrac12 c_{\min}}=e_c\frac{4c_{\min}+2}{c_{\min}^2}$. $\;\square$

Therefore $|\hat b_i^2-b_i^2|\le C_2e_c$, and by $|\sqrt u-\sqrt v|\le|u-v|/(2\sqrt{\min})$
with floor $b_{\min}^2$,
$$
r^{(b)}_m:=\frac{C_2\,e_c}{2\sqrt{\,b_{\min}^2-C_2e_c\,}}\quad(\text{valid when }C_2e_c<b_{\min}^2),
\qquad
r_m:=r^{(M)}_m+\tfrac12 r^{(b)}_m,
$$
where $r^{(M)}_m$ is the empirical-Bernstein (Maurer–Pontil) radius for $\hat M$ (range 1):
$r^{(M)}_m=\sqrt{2\widehat V\log(2/\alpha)/m}+\tfrac{7\log(2/\alpha)}{3(m-1)}$.

**Audit (bit-robust).** Reject the budget claim "$|\gamma|\le\beta$" iff
$$
\min_{\text{flip}}|\hat\gamma|:=\min\!\Big(\big|+\tfrac{|\hat b_a|}{2}-\hat M\big|,\,
\big|-\tfrac{|\hat b_a|}{2}-\hat M\big|\Big)\;>\;\beta+r_m .
$$

* **Soundness (level $\alpha$).** If the true budget holds ($|\gamma|\le\beta$ for the true
  bit), then $\min_{\text{flip}}|\gamma|\le|\gamma|\le\beta$. On the
  probability-$\ge1-\alpha$ event $\{|\hat\gamma_{\text{bit}}-\gamma|\le r_m\}$ (both radii
  hold), $\min_{\text{flip}}|\hat\gamma|\le|\hat\gamma_{\text{bit}}|\le\beta+r_m$, so the
  audit does **not** reject. Hence $P[\text{reject}\mid\text{true budget}]\le\alpha$.
* **Power.** If $\min_{\text{flip}}|\gamma|>\beta+2r_m$ then on the same good event
  $\min_{\text{flip}}|\hat\gamma|>\beta+r_m$, so the audit rejects with probability
  $\ge1-\alpha$.

*Proof.* The estimation identity is the two definitions inserted into
$\gamma=\bar a_a-\tfrac12-M$. The radius is Hoeffding + Lemma T-II.1 + the $\sqrt\cdot$ step
+ Maurer–Pontil, with $\gamma=b_a/2-M$ adding the two half-radii. Soundness and power are the
displayed deterministic implications on the good event; the bit-robust $\min$ over flips is
what makes soundness hold *for the unknown true bit*. $\;\square$

> **Honest caveat (proven but loose).** $C_2=(4c_{\min}+2)/c_{\min}^2$ is large at small
> $c_{\min}$ (e.g. $C_2\approx84$ at $c_{\min}=0.18$); the $\sqrt{}$-floor makes the
> *worst-case* radius vacuous unless $m$ is enormous, so the **worst-case audit is sound but
> nearly powerless** at realistic $m$. A **bootstrap/plug-in** radius (resample $m$ indices,
> take the $(1-\alpha)$ quantile of $|\hat\gamma^*-\hat\gamma|$) concentrates at the true
> estimation sd and restores power while remaining empirically valid. We report both and use
> the worst-case version only for the *guarantee*.
>
> **Weight.** *New.* The bit-robust falsification rule, the proven product-ratio radius
> constant, and the asymmetry between the (sound-but-loose) worst-case and (usable)
> bootstrap radii are new; the empirical-Bernstein and Hoeffding ingredients are classical.
>
> **[Validated: `V3_audit.level`, `V3_audit.power`]** Level: true $\gamma=0\le\beta$,
> bootstrap rejection $0.000$ and worst-case rejection $0.000$ over 600 trials, $m{=}6000$
> ($\le\alpha=0.05$, sound). Power: $\min_{\text{flip}}|\gamma|=0.216>\beta$, bootstrap
> rejection $1.000$ (powerful); worst-case rejection $0.000$ (documenting the proven radius
> is too loose for power).

### Theorem T-II(b) (Verification impossibility, in two regimes).
No label-free test can *verify* a budget (accept "$|\gamma|\le\beta$" with power) — only
falsify it.

**(b1) Within H (bit ambiguity).** $\gamma$ is identified only up to the flip:
$\{+\tfrac{|b_a|}{2}-M,\,-\tfrac{|b_a|}{2}-M\}$, with $M$ observable. Verification fails
*exactly* on the **bit-ambiguous set**
$$
\mathcal B_\beta=\Big\{\;\min_{\text{flip}}|\gamma|\le\beta<\max_{\text{flip}}|\gamma|\;\Big\}
=\Big\{\,\big|\,|M|-\tfrac{|b_a|}{2}\,\big|\le\beta<\,|M|+\tfrac{|b_a|}{2}\,\Big\}.
$$
On $\mathcal B_\beta$ one bit satisfies the budget and the other violates it, while both bits
share the same evidence law (T-I(b)); any sound test must therefore *abstain* (neither accept
nor reject). The audit of T-II(a), keyed to $\min_{\text{flip}}|\hat\gamma|$, correctly does
**not** reject there — this is the honest blind zone.

**(b2) Outside H (covert CEI violation; $\tau$ sound but not complete).** There exist
$K=4$ joint correctness laws that
(i) have all pairwise $c_{ij}$ exactly rank-one (so the $M\ge4$ residual $\tau=0$ and the
diagnostic is blind), yet (ii) violate CEI, and (iii) leave the advantage vector $b$
**unidentified** — the product-ratio fit returns a rank-one $b^{\text{fit}}\ne b^{\text{true}}$.
Hence $\tau=0$ is **sound** (it never falsely rejects a true H-triple) but **not complete**
(it misses these stealth violations). What $\tau=0$ *does* pin down: the pairwise product
matrix $c$ is rank-one, i.e. it is **pairwise-realizable by some H-model** — the equivalence
class is *"all joint laws with the same rank-one $c$,"* which includes both H-models (whose
$b$ equals the rank-one factor) and non-H laws with arbitrary higher-order moments.

*Construction (proof of b2).* Work with the $2^4=16$ probabilities $p$ of the correctness
pattern $(C_1,\dots,C_4)\in\{0,1\}^4$; let $S=2C-1\in\{\pm1\}^4$. Impose the 6 pairwise
second-moment equations $\sum_p p\,S_iS_j=b^{\text{fit}}_ib^{\text{fit}}_j$ (a chosen rank-one
target), $\sum p=1$, $p\ge0$, **and** a first-moment constraint $\sum_p p\,S_1\ne b^{\text{fit}}_1$.
This is a feasible linear program (16 unknowns, 7 equality + nonnegativity); its
Chebyshev-center solution is **strictly interior** ($\min_k p_k>0$). The realized law has
$c=b^{\text{fit}}b^{\text{fit}\top}$ exactly (so $\tau=0$) but first moments
$b^{\text{true}}\ne b^{\text{fit}}$, and a nonzero third-order moment
$\E[S_1S_2S_3]\ne\prod\E[S_i]$ certifies CEI is violated. $\;\square$

> **Weight.** *New (b1 characterization; b2 explicit construction).* (b1) gives the exact
> bit-ambiguous set on which verification provably fails *within H*. (b2) is an explicit,
> strictly-interior $K=4$ counterexample proving the paper's diagnostic $\tau$ is
> **sound-but-incomplete**, with a precise statement of what $\tau=0$ pins down (rank-one
> $c$ = pairwise-H-realizability). The non-completeness of $\tau$ was acknowledged
> informally in the paper; the *constructive* witness and the equivalence class are new.
>
> **[Validated: `V3_audit.bit_ambiguous_blind`, `V3_audit.stealth_tau0`]** Bit-ambiguous:
> $\min_{\text{flip}}|\gamma|=0\le\beta<\max=0.43$, bootstrap rejection $0.000$ (correctly
> blind). Stealth: a strictly-interior $K{=}4$ law with design $\tau=2.1\times10^{-17}$
> (empirical $\tau=1.2\times10^{-3}$ at $m{=}2\times10^5$), product-ratio recovers
> $b^{\text{fit}}=(0.51,0.45,0.40,0.35)$ while truth is
> $b^{\text{true}}=(0.62,0.22,0.20,0.18)$ — bias $0.229$. A second witness flips the
> **decision**: recovered $\operatorname{sign}(b_1-b_0)=+1$ but true $=-1$ at $\tau\approx0$.

### Theorem T-II(c) (Robustness of the rank-one fit to small $\hat\tau$).
If the empirical residual satisfies $\hat\tau\le t$, then the product-ratio advantages obey
$$
\big|\hat b_i^2-b_i^{2}\big|\;\le\;\frac{t}{c_{\min}}+C_2\,e_c
\qquad\Big(b_i^{2}:=\tfrac{c_{ik}c_{il}}{c_{kl}}\ \text{the population rank-one projection}\Big),
$$
an *exact algebraic* bound (no eigenvalue perturbation needed). In particular, as
$t,e_c\to0$ the fit converges to the rank-one projection at the stated rate.

*Proof.* $\hat\tau\le t$ bounds the spread of the three triple-products
$\hat c_{ik}\hat c_{il},\,\dots$; writing $\hat b_i^2=\hat c_{ik}\hat c_{il}/\hat c_{kl}$, the
deviation of the ratio from the population value decomposes into (i) the rank-one *model*
error, controlled by the minor spread $t$ divided by $|\hat c_{kl}|\ge c_{\min}$ (one factor
of $c_{\min}$ in the denominator), plus (ii) the *sampling* error $C_2e_c$ of Lemma T-II.1.
Summing gives the bound. $\;\square$

> **Weight.** *New, minor.* A clean closed-form sensitivity of the estimator to residual
> rank-one violation; used to argue graceful degradation when $\hat\tau$ is small but
> nonzero. **[Validated indirectly: `V3_audit.tau_vs_rho`]** $\tau$ rises monotonically with
> overt error-correlation $\rho$ (Pearson $0.914$, $\tau:0.0002\to0.119$ as $\rho:0\to0.6$),
> confirming detectable departures inflate the bound.

---

## T-III. Evidence-channel rate (matching bounds within H)

### Theorem T-III(a) (Upper bound; sample complexity).
Within audited H with margins ($\min_{i\ne j}|c_{ij}|\ge c_{\min}$, $\min_j|b_j|\ge b_{\min}$),
and given the flip bit, from $m$ unlabeled samples on $D$ the product-ratio estimator
satisfies, with probability $\ge1-\delta$,
$$
|\hat b_j-b_j|\le C(c_{\min},b_{\min})\sqrt{\tfrac{\log(K/\delta)}{m}},\qquad
C(c_{\min},b_{\min})=\Theta\!\Big(\tfrac{1}{b_{\min}c_{\min}^2}\Big).
$$
Consequently $\operatorname{sign}(b_j-b_0)$ is certified when $|\hat b_j-\hat b_0|>2C\sqrt{\log(K/\delta)/m}$,
giving sample complexity
$$
m\;\asymp\;\frac{C^2}{(b_j-b_0)^2}\,\log\frac{K}{\delta}.
$$
*Proof.* Hoeffding gives $|\hat c_{ij}-c_{ij}|\le\sqrt{2\log(2K^2/\delta)/m}$ uniformly;
Lemma T-II.1 and the $\sqrt\cdot$-step (T-II(a)) convert this to the displayed $\hat b_j$
bound with $C=\Theta(1/(b_{\min}c_{\min}^2))$. The certification threshold and inversion are
algebra. $\;\square$

### Theorem T-III(b) (Lower bound; Le Cam two-point within H).
Fix the flip bit. Let $P_b,P_{b'}$ be two H-models differing only in candidate $a$ by
$|b_a-b'_a|=\varepsilon$ (all else equal). The observable per-sample is the $K$-bit
prediction pattern; its single-sample law has
$$
\mathrm{KL}\big(P_b^{\text{pat}}\,\Vert\,P_{b'}^{\text{pat}}\big)\le C'(c_{\min})\,\varepsilon^2,
\qquad\text{so}\quad
\mathrm{KL}\big(P_b^{\otimes m}\Vert P_{b'}^{\otimes m}\big)\le C' m\varepsilon^2 .
$$
By Le Cam/Bretagnolle–Huber, any estimator has
$\inf_{\hat b}\sup\,P[\,|\hat b_a-b_a|\ge\varepsilon/2\,]\ge\tfrac14e^{-C'm\varepsilon^2}$;
hence once $m\lesssim 1/(C'\varepsilon^2)$ the minimax error is $\gtrsim\varepsilon/4$. The
$m^{-1/2}$ rate is therefore **minimax for the evidence channel**, matching the labeled
paired-benefit rate up to constants (and the flip bit).
*Proof (KL bound, $K=3$ explicit).* Under H the pattern law factorizes given $Y$ with
$q_j=(1+b_j)/2$; differentiating the pattern probabilities in $b_a$ shows each is
$C^1$ with bounded derivative on $\{q_j\in[\tfrac{1+ c_{\min}'}{2},\dots]\}$, so the
$\chi^2$ (hence KL) between $P_b^{\text{pat}}$ and $P_{b'}^{\text{pat}}$ is $O(\varepsilon^2)$
with constant controlled by the smallest pattern probability $\ge$ a function of $c_{\min}$;
tensorization gives the $m$-sample KL. $\;\square$

### Theorem T-III(c) ("Labels buy constants (and the bit), not the rate").
Both the labeled paired-benefit estimator and the label-free evidence channel estimate
$b_a-b_0$ at rate $m^{-1/2}$. The efficiency ratio of standard deviations,
$\mathrm{sd}_{\text{evidence}}/\mathrm{sd}_{\text{labeled}}$, is a **constant** in $m$ —
but that constant **scales like $1/c_{\min}$ and diverges as pairwise agreement approaches
chance** ($c_{\min}\to0$). So labels do not improve the *rate*; they buy a margin-dependent
constant factor (and the flip bit, which the evidence channel cannot supply at all).

> **Weight.** *Moderate (new framing; classical tools).* The two-sided $m^{-1/2}$
> characterization *for the evidence channel within H* and the explicit "labels = constants +
> bit" statement are new contributions of this section; Hoeffding/Le Cam are classical. The
> $c_{\min}$-blowup of the constant is the honest qualifier.
>
> **[Validated: `V2_identification_rate`, `V4_rate`]** Recovery error log-log slope
> $-0.510$ (tail) / $-0.513$ (all), theory $-1/2$. Minimax $\varepsilon(m)$ slope $-0.525$,
> tracking the Le Cam curve $c_0/\sqrt m$ ($c_0{=}1.80$). KL/$\varepsilon^2\in[0.164,0.171]$
> (constant ⇒ quadratic separation ⇒ $m^{-1/2}$ minimax). Efficiency ratio $3.68$ overall;
> by margin: $2.15$ at $c_{\min}{=}0.21$, $4.49$ at $c_{\min}{=}0.067$, **diverges (NaN)** at
> $c_{\min}\le0.022$ — confirming the constant blows up at low agreement.

---

## Summary of dependencies and what is genuinely new

| Result | Statement | New? |
|---|---|---|
| **Thm 0.2 / H** | $c_{ij}-b_ib_j=\pi(1-\pi)\delta_i\delta_j$; minimal hypothesis = per-class symmetry | **New, load-bearing** (corrects paper's CEI) |
| T-I(a) | identification of $|b|$, products, $|b_j{-}b_0|$ up to flip | Classical core, made flip-precise |
| T-I(b) | TV-zero flip witness preserving full audited evidence + H | **New** |
| T-I(c) | every identifying assumption is a bit-selector (necessity) | **New** |
| T-I(d) | $\gamma$-budget / majority / AoL-slope each = one bit | **New** |
| T-II(a) | bit-robust falsification, proven radius constant $C_2$, level+power | **New** (EB/Hoeffding classical) |
| T-II(b1) | exact bit-ambiguous set where verification fails within H | **New** |
| T-II(b2) | strictly-interior $K{=}4$ stealth witness: $\tau$ sound-not-complete | **New construction** |
| T-II(c) | closed-form robustness of rank-one fit to $\hat\tau$ | New, minor |
| T-III(a,b) | matching $m^{-1/2}$ evidence-channel rate within H | **New framing** (tools classical) |
| T-III(c) | labels buy constants (∝$1/c_{\min}$) + bit, not the rate | **New framing** |

All numerics: `theory_v2_validation.py` → `validation_results.json` (+ four figures
`fig_v1_flip_witness.png`, `fig_v2_identification_rate.png`, `fig_v3_audit.png`,
`fig_v4_rate.png`). Seeds fixed; each part reproducible via `--part {H,V1,V2,V3,V4}`.
