# K-Bound — Theory Extensions (4 new results)

Four results that strengthen the paper, each **proved here and numerically validated this
session** (`scripts/theory_extensions_validation.py` → `results/theory/theory_extensions_validation.json`).
Notation follows the paper: frozen model $f_0$, candidate $f_a$, target risk $R_T(\cdot)$,
benefit $\Delta = R_T(f_0) - R_T(f_a)$ (so $\Delta>0 \Rightarrow$ adapting helps), label-free
evidence $Z=\phi(X_{1:m},f_0,f_a)$, decision $g(Z)\in\{\text{adapt},\text{freeze},\text{abstain}\}$.

## Status note (honest)
- **Proved + numerically validated this session:** A1 (Le Cam lower bound, tight), A2 (forced
  abstention), A3 (regret decomposition, exact identity), A4 (multiclass sign identity, with witness).
- **Needs advisor/peer review:** all proofs are elementary but should be read; in particular A1's
  matching upper bound (tightness) and the TV‑general statement.
- **Still open (kept as future work, not claimed):** the **regression‑loss** case of A4 (the
  remainder of Conjecture 1); a clean minimax statement of A1/A2 over a richer family than the two‑point worst case.
- **Empirical #5 (CIFAR‑10‑C + Tent/SAR/EATA):** harness ready (`scripts/cifar_tent_mps_v2.py`),
  runs on your M5 — **not run here**.

## Honest weighting (what these are worth)
A1 and A2 are the **impossibility side** — A1 instantiates the *standard* two‑point/Le Cam method,
A2 follows from it; they add rigor, not a new technique, and should not be sold as the centerpiece.
A3 is a **decomposition (lemma)**. **A4 is the scientific contribution of this batch** — a *positive*
identifiability result that closes the classification half of Conjecture 1.

## Positioning (prior art credited)
The two‑point / Le Cam method is the standard minimax lower‑bound tool (Le Cam; Tsybakov). The
closest "when to adapt" works — **AETTA** (label‑free accuracy estimation for TTA, CVPR 2024) and
**Monitoring Risks in TTA** (2025) — are *post‑hoc monitoring / estimation*, not a *pre‑adaptation*
adapt/freeze/abstain certificate with a lower bound. Disagreement‑based error prediction
(**Agreement‑on‑the‑Line**, Baek et al. 2022; **Disagreement Discrepancy**, Rosenfeld & Garg 2023)
targets *absolute* error; A4 uses the disagreement region only for the **sign of a risk difference**,
a strictly weaker target than absolute risk estimation (Steinhardt & Liang 2016).

---

## A1 — Le Cam minimax lower bound for label-free adaptation
*(strengthens §5.1: turns "a bad pair exists" into "every label-free rule pays a price")*

**Theorem A1.** Let $P_T^1,P_T^2$ be two target worlds with $\Delta_1>0$ (adapt correct) and
$\Delta_2<0$ (freeze correct), inducing evidence laws $\mu_i=\mathrm{Law}(Z\mid P_T^i)$. Let
$\delta=\min(|\Delta_1|,|\Delta_2|)$ and $\mathrm{TV}=\mathrm{TV}(\mu_1,\mu_2)$. Then every
(possibly randomized) committal label‑free rule $g:Z\to\{\text{adapt},\text{freeze}\}$ obeys
$$\max_i \mathbb{E}_i[\text{committal regret}] \;\ge\; \tfrac{\delta}{2}\,(1-\mathrm{TV}).$$
When the worlds are observationally equivalent ($\mu_1=\mu_2$, $\mathrm{TV}=0$) this is $\delta/2$;
the bound is **achieved** by the likelihood‑ratio test, so it is tight.

**Proof.** Committal regret is $|\Delta_1|\mathbf 1[g{=}\text{freeze}]$ in world 1 and
$|\Delta_2|\mathbf 1[g{=}\text{adapt}]$ in world 2. Under a uniform prior the Bayes regret is
$\tfrac12(|\Delta_1|P_1(g{=}\text{freeze})+|\Delta_2|P_2(g{=}\text{adapt}))
\ge \tfrac{\delta}{2}\big(P_1(g{=}\text{freeze})+P_2(g{=}\text{adapt})\big).$
The parenthesis is the type‑I+type‑II error sum of a test of $\mu_1$ vs $\mu_2$, whose minimum over
all $Z$‑measurable tests equals $1-\mathrm{TV}(\mu_1,\mu_2)$ (Neyman–Pearson / Le Cam). Hence Bayes
regret $\ge \tfrac{\delta}{2}(1-\mathrm{TV})$, and the worst case is $\ge$ the average. The LR test
attains the $1-\mathrm{TV}$ error sum, giving the matching upper bound. $\qquad\blacksquare$

**Witness.** $X\sim\mathcal N(0,1)$ in both worlds; $f_0=\mathbf 1[x>0]$, $f_a=\mathbf 1[x<0]$, 0/1
loss; world 1 has $Y=\mathbf 1[X>0]$ ($\Delta_1=+1$), world 2 has $Y=\mathbf 1[X<0]$ ($\Delta_2=-1$).
$Z$ is a function of $X$ alone $\Rightarrow \mu_1=\mu_2\Rightarrow\mathrm{TV}=0,\ \delta=1$, so
worst‑case regret $\ge 1/2$.

**Validation.** At $\mathrm{TV}=0$ the optimal rule's worst‑case regret is **0.500** $=\delta/2$;
across a TV sweep the bound matches the Monte‑Carlo optimum to within **1.3e‑3** everywhere.
![Le Cam bound is tight](../figures/final/fig_lecam_bound.png)

---

## A2 — Forced abstention
*(new, sits right after Theorem 1: abstaining is necessary, not a hedge)*

**Theorem A2.** On an observationally‑equivalent pair ($\mu_1=\mu_2$, $\Delta_1>0,\Delta_2<0$), any
label‑free rule $g:Z\to\{\text{adapt},\text{freeze},\text{abstain}\}$ with false‑adapt rate $\le\alpha$
and false‑freeze rate $\le\alpha$ abstains with probability $\ge 1-2\alpha$. As $\alpha\to0$, the
abstention probability $\to 1$.

**Proof.** Since $\mu_1=\mu_2=\mu$ and $g$ depends only on $Z$ (plus independent randomness), the
action law $(a,f,s)=(P(\text{adapt}),P(\text{freeze}),P(\text{abstain}))$ under $\mu$ is identical in
both worlds and sums to 1. False‑freeze (world 1) $=f\le\alpha$; false‑adapt (world 2) $=a\le\alpha$;
hence $s=1-a-f\ge 1-2\alpha$. Achievability: $a=f=\alpha\Rightarrow s=1-2\alpha$. $\qquad\blacksquare$

**Validation.** Across $\alpha\in\{0,0.01,0.05,0.1,0.2,0.4\}$ the floor $s\ge 1-2\alpha$ is **never
violated**, and the real KGA conformal rule abstains on **100%** of the witness instances
(`results/witness/witness_clean.json`), the $\alpha\to0$ prediction.

---

## A3 — Regret decomposition
*(new lemma; the algebraic backbone of the §7 mixed‑regime story)*

**Lemma A3.** For benefit $\Delta=R_T(f_0)-R_T(f_a)$ and any policy whose abstentions default to the
frozen model, the regret to the label‑aware oracle decomposes **exactly** as
$$\mathrm{Regret}(g)=\underbrace{\mathbb E[|\Delta|\mathbf 1\{g{=}\text{adapt},\Delta{<}0\}]}_{\text{false-adapt}}
+\underbrace{\mathbb E[|\Delta|\mathbf 1\{g{=}\text{freeze},\Delta{>}0\}]}_{\text{false-freeze}}
+\underbrace{\mathbb E[|\Delta|\mathbf 1\{g{=}\text{abstain},\Delta{>}0\}]}_{\text{abstain-coverage}},$$
each term $\ge0$. Always‑adapt is pure false‑adapt $\mathbb E[|\Delta|\mathbf1\{\Delta<0\}]$;
always‑freeze is pure false‑freeze $\mathbb E[|\Delta|\mathbf1\{\Delta>0\}]$.

**Proof.** The oracle attains $\min(R_{f_0},R_{f_a})$. If the policy adapts, regret
$=R_{f_a}-\min=\max(0,R_{f_a}-R_{f_0})=|\Delta|\mathbf1\{\Delta<0\}$. If it freezes/abstains
($\to$ frozen), regret $=R_{f_0}-\min=|\Delta|\mathbf1\{\Delta>0\}$. Partitioning by the disjoint
decision events gives the three terms; abstaining when $\Delta<0$ costs $0$ (the frozen default is
already optimal), so only the $\Delta>0$ slice of abstention appears. $\qquad\blacksquare$

**Why it matters.** KGA buys safety by driving **false‑adapt $\to 0$** (the certificate) at the price
of some **abstain‑coverage**; it beats *both* trivial policies only when both their pure terms are
large — a genuinely mixed regime.

**Validation.** The identity holds exactly: total regret **49.3412** $=$ FA **0.257** $+$ FF
**0.189** $+$ AC **48.895** on a synthetic mixed set, where the trichotomy (49.34) beats both
always‑adapt (163.81) and always‑freeze (153.05).
![Regret decomposition](../figures/final/fig_regret_decomposition.png)

---

## A4 — Multiclass disagreement-region sign identifiability
*(replaces Conjecture 1's classification half in §5.4 with a proved theorem)*

**Theorem A4.** Let $Y\in\{1,\dots,K\}$ with 0/1 loss, $f_0,f_a$ known maps, and
$D=\{x:f_0(x)\ne f_a(x)\}$ the observable disagreement region. With
$a_0^D=P_T(f_0{=}Y\mid D)$ and $a_a^D=P_T(f_a{=}Y\mid D)$,
$$\Delta=R_T(f_0)-R_T(f_a)=P_T(D)\,\big(a_a^D-a_0^D\big),\qquad
\text{so}\quad \operatorname{sign}\Delta=\operatorname{sign}(a_a^D-a_0^D).$$
Thus $\operatorname{sign}\Delta$ is label‑free identifiable **iff** the evidence determines which
predictor is more accurate on $D$. Binary **Theorem 5** is the special case $a_0^D=1-a_a^D$ (exactly
one correct on $D$), collapsing the criterion to $a_a^D$ vs $\tfrac12$.

**Proof.** Off $D$, $f_0=f_a$ so the per‑sample loss difference $\delta(x)=\mathbf1\{f_0\text{ wrong}\}-\mathbf1\{f_a\text{ wrong}\}=0$.
On $D$, $\delta(x)=\mathbf1\{f_a{=}Y\}-\mathbf1\{f_0{=}Y\}\in\{-1,0,+1\}$ ($0$ when **both** are wrong —
the genuinely multiclass case, impossible when $K=2$). Hence
$\Delta=\mathbb E[\delta]=P(D)\,\mathbb E[\delta\mid D]=P(D)(a_a^D-a_0^D)$. $\qquad\blacksquare$

**Honest delta over Theorem 5 / Steinhardt–Liang.** Binary needs only an *absolute* judgment (is
$f_a$ above chance on $D$); multiclass needs the **sign of a difference** of two accuracies on $D$ — a
paired/ordinal comparison with no fixed $\tfrac12$ reference. This is still **strictly weaker** than
estimating the absolute risks $R(f_0),R(f_a)$, so the positive result survives into multiclass; the
residual burden is a reliability signal that brackets $a_a^D-a_0^D$. When none does, the regime
re‑enters the unknowable case of A1/Theorem 1.

**Resolves / open.** Closes the **classification** half of Conjecture 1 (the sign identity holds for
all $K$). The **regression‑loss** case — $\delta(x)=\ell(f_0)-\ell(f_a)$ real‑valued, no
$\{-1,0,+1\}$ structure — remains open and is kept as future work.

**Witness + validation.** Synthetic $K=5$. *Helpful:* $P(D)=0.582$, $a_a^D=0.542>a_0^D=0.283$ →
$\Delta=+0.1502$. *Harmful:* $P(D)=0.632$, $a_a^D=0.279<a_0^D=0.511$ → $\Delta=-0.1462$. The identity
$\Delta=P(D)(a_a^D-a_0^D)$ matches the direct $\Delta$ to machine precision (gap $\le 3\mathrm{e}{-}17$),
the sign tracks the relative accuracy in both, and the both‑wrong‑on‑$D$ fraction (**0.175** / 0.210)
confirms it is genuinely outside the binary regime.

---

## Where each result goes in the paper
| Result | Placement | Effect |
|---|---|---|
| A1 | §5.1 (augment the Theorem‑1 remark) | quantitative limit + `fig_lecam_bound` |
| A2 | §5.1 (new, after Theorem 1) | justifies the abstain action |
| A3 | end of §5 / start of §7 | names what the experiments measure + `fig_regret_decomposition` |
| A4 | §5.4 (replace Conjecture 1, classification half) | proved multiclass positive result |
