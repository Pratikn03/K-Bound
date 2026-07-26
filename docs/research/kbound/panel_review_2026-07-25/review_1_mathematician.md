# Reviewer 1 — Mathematician / Theoretical Statistician

*(All paths are relative to `/root/kb` unless absolute. Every numerical claim below was
re-derived with `python3` against the shipped artifacts; the exact commands are quoted in the
Evidence blocks.)*

## Bottom line

The population theory of K-Bound is, as far as I can check, **correct** — but it is also
**definitionally thin**: because $\gamma$ is *defined* as $(\bar a-\tfrac12)-M$, the "reduction
lemma" $\operatorname{sign}\Delta=\operatorname{sign}(M+\gamma)$ is an identity, and the headline
frontier $|M|>\beta$ is interval arithmetic on a *declared* constraint. That is defensible framing,
and the authors are unusually candid about it. What is **not** defensible in the current frozen
submission is the finite-sample layer. Theorem `thm:certificate` is a two-line tautology whose
entire content sits in an assumed coverage premise, and the empirical evidence offered for that
premise is, on the two genuine multi-seed tracks, an **arithmetic identity of an in-sample
quantile**: I verified that the reported "empirical interval-hit coverage" values
$0.8981481481481481$ (CIFAR-10-C, $n=432$) and $0.8888888888888888$ (ImageNet-C, $n=27$) are
*exactly* `np.quantile(x, 0.9)`-in-sample coverage for those $n$ and are independent of the data;
that the miscovered-cell count equals $n-k$ in **all 13** shipped per-condition files (the
deterministic maximum); and hence that $\mathrm{FA}_{\mathrm u}\le\alpha$ **cannot be violated** on
those tracks. The one track in the panel where coverage is genuinely held out (CIFAR-10.1,
$25/30=0.833$) **under-covers**, and is precisely the track where $\mathrm{FA}_{\mathrm u}=0.167>\alpha$.
Separately, the appendix per-seed ImageNet-C table still reports the *discarded* interpolated-quantile
numbers, contradicting the promoted panel by a factor of 2.5, and under the promoted exact-rank rule
KGA is bit-identical to always-freeze on 3 of 5 seeds — falsifying the appendix sentence "Point
estimates improve both fixed-policy regrets on every seed."

**Verdict: major revision required. The theory can stand (with sharpening); the finite-sample
"certificate" evidence must be rebuilt on a genuinely held-out calibration/test split before any
$\mathrm{FA}_{\mathrm u}\le\alpha$ claim is made.**

## What is done well

* **Definitional hygiene is genuinely excellent.** `paper/sections/theory_setup.tex:77–96`
  (`rem:four-quantities`) separates $M,\gamma,\beta,\varepsilon$ with an explicit table and the
  sentence "The radius $\varepsilon$ ... is *not* an estimate of $\beta$." I looked hard for a place
  where $\varepsilon$ is smuggled in as $\beta$ and did not find one in the compiled build.
* **`def:risk-align` is *not* smuggled in as a fact.** `theory_setup.tex:34–44` states it as a
  definition and immediately says "Fitting a benefit regressor does not certify risk alignment";
  `kbound_short.tex:207` (assumptions table) lists it as a condition with a declared fallback;
  `rem:fa-marginal` (`theory_core_main.tex:172–178`) explicitly says risk alignment is needed for the
  *frontier*, not for the interval rule. This is exactly right.
* **Boundary cases are handled**, and handled better than most papers of this kind: $|M|=\beta>0$,
  $M=\beta=0$, and $\beta=0$ each get their own clause (`thm:headline`(iii)–(iv),
  `prop:closed-band`, `rem:beta-zero`), and the paper is careful that *opposite nonzero signs* are
  claimed only on the open band while the boundary gets the weaker zero-versus-strict argument.
* **The Le Cam / two-point machinery in `theory_v2/minimax_optimality_theorem.tex` is real.** The
  three-way-to-two-way reduction ($\psi=\mathbf 1[g=\textsc{adapt}]$, with abstention counted as a
  miss so that $e_1(\psi)=\mathrm{miss}_+(g)$) is clean and correct, the Bretagnolle–Huber step is
  correctly applied, the Gaussian KL is right, and the paper is explicit that it proves
  **order**-optimality, not tight constants (`rmk:opt-scope`). This is a well-written minimax section.
* **No `sorry`, no `axiom`, no `admit` anywhere in the Lean tree** (verified by grep over all 27
  `.lean` files), and `kbound_short_appendix.tex:261–269` (`app:formal`) volunteers an accurate list
  of what Lean does *not* cover. Very few papers with a formalization appendix are this honest.
* The authors' own `SUBMISSION_LEDGER.md` correctly flags `thm:conj1-dichotomy` as near-vacuous and
  `\iffalse`-s it out of the build; I verified all five `\iffalse` blocks in
  `theory_appendix_ext.tex` really are excluded.

---

## Findings

### [BLOCKER] F1-1 — The reported interval coverage and $\mathrm{FA}_{\mathrm u}\le\alpha$ on the stress grids are arithmetic identities of an in-sample quantile, not measurements

**Location.** `docs/research/kbound/kbound_short_appendix.tex:23–32` (Algorithm 1, stress-grid
branch); `docs/research/kbound/scripts/g8_exactrank_regen.py:4–29`;
`docs/research/kbound/paper/generated/empirical_audit/decision_metrics.json`
(`interval_coverage_observed`); `kbound_short.tex:511` (RQ2), `:627` ("selective routing at $0\%$
false-adapt").

**Evidence.** Algorithm 1 computes $\varepsilon$ from the residuals of *all* $N$ cells and then
scores *those same* $N$ cells:

```
25  State Fit \widehat\Delta_{-i} on the other N-1 cells; \rho_i <- |\widehat\Delta_{-i}(Z_i)-\Delta_i|
28  State \varepsilon <- \rho_{(k)}, k=\lceil (N+1)(1-\alpha)\rceil
29  ForAll cells i \in C:  Run Algorithm 2 with (..., \widehat\Delta_{-i}, \varepsilon)
```

`g8_exactrank_regen.py` does the same in code (`rho=np.abs(bh-B); ee=cexact(rho); de=decide(bh,ee)`).
When $\varepsilon$ is the $k$-th order statistic of the very residuals being tested, the number of
miscovered cells is *identically* $N-k$ — no randomness, no assumption. I confirmed this on every
shipped file:

```
$ python3  # over experiments/kbound/results/**/per_condition_*.json
 imagenetc_sar_seed0..4 : n=27  k=26  miscovered=1  (max possible n-k = 1)
 cifar10c_tent_seed1..4 : n=432 k=390 miscovered=42 (max possible n-k = 42)
 cifar10c_eata_seed1..4 : n=432 k=390 miscovered=42 (max possible n-k = 42)
```

13/13 files hit the ceiling exactly. Consequently
$\mathrm{FA}_{\mathrm u}\le\#\{\rho_i>\varepsilon\}/N=(N-k)/N\le\alpha$ **holds for any data
whatsoever**. The same identity explains the shipped coverage numbers, which use the interpolated
variant:

```
$ python3 -c "import numpy as np; x=np.abs(np.random.default_rng(0).standard_normal(432));
              q=np.quantile(x,0.9); print((x<=q).mean())"
0.8981481481481481          # == decision_metrics.json CIFAR-10-C interval_coverage_observed.rate
# n=27 gives 0.8888888888888888  == decision_metrics.json ImageNet-C rate
```

These are functions of $n$ alone. They contain zero information about the data, the estimator, or
exchangeability.

**Why it matters.** RQ2 ("whether the interval controls marginal false adaptation",
`kbound_short.tex:511`) and the abstract's "keeps observed false adaptation near zero" are the
paper's *primary* claim ("Our primary evidence is a safety guarantee, not an accuracy gain",
`:41`). On the two tracks that carry that claim, the guarantee is untestable by construction. The
paper does disclose the weaker point once — "the leave-one-cell-out radius provides strong
*empirical* coverage, not the exact distribution-free validity of clean split conformal or
jackknife$+$" (`kbound_short.tex:691–693`) — but that is a statement about *validity*, not about the
much stronger fact that the reported statistic is a constant.

**Fix.** (a) Split each grid into disjoint calibration and evaluation cell sets (or use a
genuinely held-out seed) and recompute coverage and $\mathrm{FA}_{\mathrm u}$ there; (b) delete the
Wilson intervals on the in-sample coverage counts (F1-11); (c) state explicitly in the results
section that in-sample rank calibration makes $\mathrm{FA}_{\mathrm u}\le(N-k)/N$ an identity, so
the *informative* statistic is $\mathrm{FA}_{\mathrm u}=0$ versus the ceiling $(N-k)/N$, not
"$\mathrm{FA}_{\mathrm u}\le\alpha$".

---

### [BLOCKER] F1-2 — The appendix ImageNet-C per-seed table uses the *discarded* quantile rule and contradicts the promoted panel; under the promoted rule, "improves both fixed policies on every seed" is false

**Location.** `docs/research/kbound/kbound_short_appendix.tex:285–312` (`tab:imagenetc-perseed` and
the paragraph above it); `SUBMISSION_LEDGER.md:83–89` (gap G8);
`paper/generated/kbound_result_manifest.json` → `tracks/imagenetc_sar`.

**Evidence.** The ledger records G8 as `[RESOLVED = PASS]` with the action
"update panel numbers to exact-rank values; state FA_u/eps use the exact rank rule; drop
interpolated-quantile from headline path." The manifest for `imagenetc_sar` duly says
`"quantile_rule": "exact split-conformal rank k=ceil((n+1)(1-alpha)), alpha=0.10"` and
`regret [0.026422, 0.052933, 0.031894]`. I reproduced both conventions from the five per-seed files:

| seed | appendix table (`:303–309`) | recomputed **interpolated** | recomputed **exact-rank** | always-freeze |
|---|---|---|---|---|
| 0 | 0.0108 | 0.0108 | **0.0319** | 0.0319 |
| 1 | 0.0091 | 0.0091 | **0.0312** | 0.0312 |
| 2 | 0.0128 | 0.0128 | **0.0102** | 0.0284 |
| 3 | 0.0056 | 0.0056 | **0.0290** | 0.0290 |
| 4 | 0.0154 | 0.0154 | **0.0297** | 0.0389 |
| pooled | 0.0107 | 0.0107 | **0.0264** | 0.0319 |

The appendix table is *exactly* the interpolated rule (including `FA_u` seed 2 $=0.037=1/27$ and
pooled $0.007=1/135$, which are 0.000 and 0.000 under the exact rule). The promoted panel number is
0.0264. **The same quantity appears twice in the same frozen document with a 2.5× discrepancy.**

Worse, under the promoted exact-rank rule the action composition is

```
seed0 {FREEZE:3, ABSTAIN:24}   seed1 {FREEZE:3, ABSTAIN:24}   seed2 {FREEZE:2, ABSTAIN:16, ADAPT:9}
seed3 {FREEZE:3, ABSTAIN:24}   seed4 {FREEZE:3, ABSTAIN:21, ADAPT:3}
pooled {FREEZE:14, ABSTAIN:109, ADAPT:12}   # matches manifest n_cells 135, abstain_count 109
```

so on seeds 0, 1 and 3 KGA **never adapts** and its regret is *bit-identical* to always-freeze
(0.0319/0.0319, 0.0312/0.0312, 0.0290/0.0290). The appendix sentence
"Point estimates improve both fixed-policy regrets on every seed" (`:287–288`) and the claim that the
per-seed bootstrap "excludes zero on both gaps for seeds 0–1" (`:288–289`) are **false under the
promoted rule** — the freeze-side gap is identically zero on three of five seeds. The pooled
"beats-both" therefore rests on 12 adapt decisions, 9 of which come from a single seed.

The same staleness infects the CIFAR panel: `tracks/cifar10c_tent` in the manifest carries **no**
`quantile_rule` field and its regret 0.0015736 is the interpolated value (my 4-seed exact-rank
recomputation gives 0.0017 vs 0.0016 interpolated), so the panel mixes the two conventions across
tracks. And `decision_metrics.json`'s "ImageNet-C SAR" entry reports 12 adapt / 3 freeze / 12 abstain
from a *third* source (`win_hunt_v5/imagenetc_aggr`, seed 0 only), versus 0 adapt / 3 freeze /
24 abstain for seed 0 under the promoted rule.

**Why it matters.** This is the paper's headline "mixed + detectable ⇒ beats both" track. Three
shipped artifacts (compiled appendix, `kbound_result_manifest.json`, `decision_metrics.json`)
disagree with each other about it, and the promoted rule turns the claim into a single-seed effect.

**Fix.** Regenerate `tab:imagenetc-perseed`, the per-seed bootstrap CIs, `decision_metrics.json`, and
the CIFAR manifest entries under one declared rule; restate the per-seed sentence honestly
("on 3/5 seeds KGA degenerates to always-freeze; the pooled win is driven by seeds 2 and 4"); or
revert the whole track to the interpolated rule and drop the exact-rank claim. Either is fine —
mixing them is not.

---

### [MAJOR] F1-3 — `thm:certificate` is a tautology, and the protocol that is supposed to supply its premise is a jackknife, not split conformal

**Location.** `paper/sections/theory_core_main.tex:143–170`; `kbound_short_appendix.tex:28`, `:36`
(Algorithm 1); `kga/certificate.py:265–309`.

**Evidence.** The theorem *assumes* $\Pr[|\widehat\Delta-\Delta|\le\varepsilon]\ge1-\alpha$ and the
proof is two sentences. Everything rests on the premise; the paper's only comment is
"Exchangeability, or an explicit shift correction, is one route to this coverage assumption"
(`:169`). Algorithm 1 then labels the construction

```
28   \varepsilon <- \rho_{(k)}, k=\lceil (N+1)(1-\alpha)\rceil   \Comment{exact split-conformal rank quantile}
36   Form out-of-fold residuals on C_cal; \varepsilon <- \rho_{(k)} ... (exact split-conformal rank quantile)
37   Fit deployment estimator \widehat\Delta(Z) on ALL of C_cal; freeze (\widehat\Delta,\varepsilon)
```

Neither branch is split conformal. The grid branch is the **jackknife** (leave-one-out residuals,
no held-out point), which has *no* distribution-free coverage guarantee (Barber–Candès–Ramdas–Tibshirani
2021); jackknife$+$ would give $1-2\alpha$ and requires using the leave-one-out models at test time.
The natural-shift branch is worse in a specific way: $\varepsilon$ is calibrated on out-of-fold
residuals (models trained on $|\mathcal C_{\rm cal}|-1$ folds) but is then applied to a **full-data**
estimator $\widehat\Delta$ trained on all of $\mathcal C_{\rm cal}$ — the classic CV-vs-CV$+$
mismatch, for which no finite-sample guarantee exists. The paper concedes the point once at
`kbound_short.tex:691–693` but the algorithm caption, the module docstring
(`kga/certificate.py:23`, "the exact split-conformal order-statistic radius") and the manifest
(`"exact split-conformal rank"`) all assert the opposite.

Separately, exchangeability is implausible here on its own terms: the "units" are grid cells
indexed by (corruption type × severity × batch size × aggressiveness). Severity-5 and severity-1
cells are not exchangeable in any meaningful sense, and for the online/continual adapters
(`Camelyon17 EATA online`, `Office-Home SAR online`, `tab:adapter-hparams`) the candidate $f_a$ is
itself a function of the stream order, so calibration residuals and deployment residuals are not
even identically distributed.

**Why it matters.** "Finite-sample certificate" is the load-bearing phrase of the paper's safety
claim. As stated it is `if coverage then safety`, and the shipped route to coverage is not valid.

**Fix.** Either (i) rename Algorithm 1's radius honestly ("leave-one-cell-out empirical residual
radius; not exact split conformal" — the phrase already exists inside
`kbound_result_manifest.json` and should be promoted to the paper), or (ii) implement jackknife$+$
/ CV$+$ properly (score the test cell with the fold models) and state the $1-2\alpha$ guarantee, or
(iii) reserve a genuine held-out calibration split. Also state explicitly that
`thm:certificate` is an implication, not an achievability result.

---

### [MAJOR] F1-4 — The only empirical validation of the frontier theorem is circular by construction, and its "90.0% coverage" is `np.quantile`'s definition

**Location.** `docs/research/kbound/scripts/frontier_validation.py:16–23, 52–58, 62–70`;
`kbound_short.tex:596–612` (RQ1) and `fig:frontier-measured`.

**Evidence.** The generative model is

```python
54  M     = rng.uniform(m_lo, m_hi, n)
55  gamma = rng.uniform(-beta, beta, n)      # unobserved; |gamma| <= beta
56  B     = M + gamma                        # true benefit
57  Z     = np.column_stack([M + rng.normal(0, obs_noise, n) for _ in range(4)])
```

$Z$ is four noisy copies of $M$ and nothing else, so any consistent regressor returns
$\widehat B\approx M$ and its residual is *identically* $\gamma\sim U(-\beta,\beta)$. Hence
$\varepsilon=q_{0.9}(|\widehat B-B|)\approx 0.9\beta$ and the commit rule
$|\widehat B|>\varepsilon$ is $|M|\gtrsim0.9\beta$ **by algebra**. The script's own docstring says so:
"Because gamma is the irreducible residual, the split-conformal radius eps self-calibrates to
~beta, so the decision rule reproduces the |M|>beta frontier automatically."

The paper reports this as a *test*: "the commit rate jumps from $6.5\%$ for $|M|<\beta$ to $96.7\%$
for $|M|>\beta$—a transition *at* the predicted boundary… The frontier is not imposed; it appears
because the synthetic residual scale is controlled by the deliberately hidden drift"
(`:608–612`). The second sentence states the mechanism that makes the first one circular.

The recovery number is worse. Line 50 is `eps = float(np.quantile(np.abs(Bhat-B), 1-alpha))` and
line 68 is `cov = float(np.mean(np.abs(Bhat-B) <= eps))`. For $n=400$:

```
$ python3 -c "import numpy as np; x=np.abs(np.random.default_rng(0).standard_normal(400));
              print((x<=np.quantile(x,0.9)).mean())"
0.9
```

The paper's "$\widehat\Delta$ tracks $\Delta$ at $90.0\%$ empirical coverage, matching the nominal
target in this simulation" (`:602–603`) is the definition of the empirical quantile, not a
measurement.

**Why it matters.** This is the *only* place `thm:headline` is checked against ground truth
(`fig:frontier-measured`, RQ1), and $\beta$ appears in no other experiment.

**Fix.** Re-run with (a) $Z$ carrying information that is *not* a noisy copy of $M$ (so
$\varepsilon\neq\beta$ generically), (b) $\gamma$ with a distribution whose $0.9$-quantile is not
tied to $\beta$, and (c) a held-out evaluation set for coverage. Then the "transition at $|M|=\beta$"
becomes a real prediction. Otherwise drop the word "validation" and label the figure as an
illustration.

---

### [MAJOR] F1-5 — `split_conformal_rank_radius` silently returns an invalid level when $n<(1-\alpha)/\alpha$, contradicting the paper's own feasibility condition

**Location.** `kga/certificate.py:252–262`.

**Evidence.**

```python
261    k = min(n, int(math.ceil((n + 1) * (1.0 - alpha))))
262    return float(np.sort(arr)[k - 1])
```

The `min(n, ·)` clamp fires exactly when $\lceil(n+1)(1-\alpha)\rceil>n$, i.e. $n<(1-\alpha)/\alpha$
— for $\alpha=0.1$, whenever $n\le8$. In that regime the function returns $\max_i\rho_i$, whose
attainable coverage is at most $n/(n+1)<1-\alpha$; the correct conformal output is $+\infty$
(forced abstention). The docstring nevertheless promises "guarantees that a fresh $\widehat\Delta$
deviates … by at most $\varepsilon$ with probability at least $1-\alpha$".

This is not hypothetical: the paper *proves* the feasibility condition itself in
`kbound_short_appendix.tex:353–360` (`thm:short-audG`): "feasible iff
$\alpha\ge\tfrac1{K+1}+\delta$ … ($K=5$: nothing below $\alpha\approx0.17+\delta$)". The code
violates the paper's own theorem. Several natural-shift tracks operate near this boundary
(`kbound_result_manifest.json`: `camelyon17_ood` has `n_test 18` with `dev_seeds [0,1]`;
`officehome_M_v2` has `n_test 35` over 4 LODO domains).

**Why it matters.** A silently-degraded level turns a certificate into a point estimate with
decoration, on exactly the small-$n$ domain-level tracks the paper leans on for "no-harm".

**Fix.** Raise or return `float("inf")` when `ceil((n+1)*(1-alpha)) > n`, and report the effective
level $1-k/(n+1)$ alongside every $\varepsilon$ in the manifests.

---

### [MAJOR] F1-6 — `thm:imp` is compiled with **no proof**, and `lem:reduction`'s proof forward-references a Gaussian witness that does not exist in the build

**Location.** `paper/sections/theory_core_main.tex:60–61`; `paper/sections/theory_appendix_ext.tex:36–52`;
`SUBMISSION_LEDGER.md:45` ("thm:imp … COMPILED (FIX: xref+M(g) notation)").

**Evidence.** The proof of `lem:nonid` ends with

```
60  This completes the construction. An explicit Gaussian witness appears in
61  Appendix~\ref{app:theory-full} (Theorem~\ref{thm:imp}).
```

`app:theory-full` `\input`s `theory_appendix_ext.tex`, whose `thm:imp` (lines 36–44) is a bare
three-clause statement with **no proof environment at all**. I grepped the entire compiled stack:

```
$ grep -n "Gaussian" kbound_short.tex kbound_short_appendix.tex \
        paper/sections/theory_{setup,core_main,appendix_ext}.tex
theory_core_main.tex:60:  ... An explicit Gaussian witness appears in
theory_appendix_ext.tex:31: A Gaussian family with $d_n=c/\sqrt n$ ...   <-- inside \iffalse
kbound_short_appendix.tex:218: ... Gaussian noise over 13 severities       <-- the D33 experiment
```

The only Gaussian construction is inside the `\iffalse … \fi` block containing `prop:lecam-finite`
(lines 10–34), which the ledger itself records as NOT COMPILED. So the cross-reference points to
content the reader cannot see. Moreover `thm:imp`(ii) ("Le Cam identity: $\inf_g\mathrm{err}(g)=1-\TV$")
is a genuine mathematical statement that is never proved anywhere in the build, and
`cor:forced-abstain` (`:46–52`) is derived *from* `thm:imp`(iii), also unproved here.

Note the statement of (ii) is *correct* only under the reading $\mathrm{err}=e_0+e_1$ (sum of the two
errors); under the equal-prior average it is $(1-\TV)/2$. "Total test error" is doing a lot of work
and should be defined.

**Why it matters.** A compiled theorem with no proof and a dangling promise of a witness is exactly
what a referee flags first. The ledger claims the xref was fixed; it was not.

**Fix.** Either move the Gaussian witness into the compiled appendix, or delete the sentence at
`theory_core_main.tex:60–61` (the explicit Bernoulli kernel already in the proof of `lem:nonid` is a
perfectly good witness and needs no Gaussian companion). Give `thm:imp`(ii) a two-line proof
(Neyman–Pearson) and define $\mathrm{err}$.

---

### [MAJOR] F1-7 — `theory_v2/tight_constants_closure.tex`: the "all-rules" lower bound invokes the wrong extremal principle; the theorem is true but the proof is a non-sequitur

**Location.** `docs/research/kbound/theory_v2/tight_constants_closure.tex:97–115` (Step 1), `:116–124`
(Step 2).

**Evidence.** Step 1 reads:

> "Among all Borel sets $C$ with $\Pr_0(\bar X\in C)\le\alpha$, the one *maximizing*
> $\Pr_{+\Delta}(\bar X\in C)$ under a fixed $\Pr_0(C)$ is an interval
> $(-\infty,-\tau]\cup[\tau,\infty)$ by the monotone likelihood ratio (MLR) property … (Karlin–Rubin)"

This is false. The likelihood ratio $dP_{+\Delta}/dP_0$ is *increasing* in $x$, so by Neyman–Pearson
the maximizer is the **one-sided** upper tail $[\tau',\infty)$, not the symmetric two-sided set. The
symmetric set is strictly sub-optimal for that objective, so "MLR" cannot deliver
$2\Phi(-\tau/t)\le\alpha\Rightarrow\tau\ge z_{1-\alpha/2}t$.

The **conclusion is nevertheless correct**, but for a different reason, which the file does not
give: writing $p=\Pr_0(A_+)$, $q=\Pr_0(A_-)$ with $p+q=\Pr_0(C)\le\alpha$, Neyman–Pearson applied
*separately* to each side gives $\Delta/t\ge z_{1-\alpha}+\Phi^{-1}(1-p)$ and
$\Delta/t\ge z_{1-\alpha}+\Phi^{-1}(1-q)$; minimising $\max$ over $p+q\le\alpha$ forces
$p=q=\alpha/2$, whence $\Delta/t\ge z_{1-\alpha}+z_{1-\alpha/2}$, i.e. $n\ge\kappa(\alpha)n_{\rm opt}$.
I checked this budget-split argument and it is airtight; the MLR sentence should simply be replaced
by it. Step 2 ("randomization") is likewise hand-waved ("by convexity of the Gaussian tail
probabilities") where randomized Neyman–Pearson gives it immediately.

Minor but real notational bug in the same file: line ~25 defines $z_\alpha:=\Phi^{-1}(1-\alpha)$, yet
the very next displays use $z_{1-\alpha/2}$ meaning $\Phi^{-1}(1-\alpha/2)$ — under the file's own
convention $z_{1-\alpha/2}$ would be $\Phi^{-1}(\alpha/2)<0$. The header comment (line 16) writes
$\kappa=(1+z_{1-\alpha/2}/z_{1-\alpha})^2$ while Step 3 writes $\kappa=(1+z_{1-\alpha/2}/z_\alpha)^2$.
Two conventions, one formula.

**Why it matters.** This file is billed as *closing* the tight-constants program
(`cor:t1c-status`: "the tight-constants program is **closed**"). A closure argument whose extremal
step is wrong is not a closure.

**Fix.** Replace Step 1 with the $p+q\le\alpha$ budget-split + Neyman–Pearson argument; replace
Step 2 with the randomized-NP lemma; fix the $z$ convention once at the top of the file.

---

### [MAJOR] F1-8 — 8 of the 13 compiled theorem-level results have no proof in the compiled document, and one compiled proposition is missing from the ledger

**Location.** Compiled stack enumerated by stripping `\iffalse` blocks from `kbound_short.tex`,
`theory_setup.tex`, `theory_core_main.tex`, `kbound_short_appendix.tex`, `theory_appendix_ext.tex`.

**Evidence.** Compiled with a `\begin{proof}`: `lem:reduction`, `lem:nonid`, `cor:matched-abstain`,
`thm:headline`, `thm:certificate` (5). Compiled with **no** proof: `prop:multiclass`
(`kbound_short.tex:249`), `prop:closed-band` (`theory_core_main.tex:77`), `thm:imp` and
`cor:forced-abstain` (`theory_appendix_ext.tex:36,46`), `thm:short-audA/C/DE/G`
(`kbound_short_appendix.tex:321,329,344,353`), `prop:beatsboth-asym` (`:155`). The audit theorems
defer to "proofs in the long manuscript" (`:314–316`) and `prop:beatsboth-asym` defers to
"the repository note `theory_v2/minimax_frontier/`" (`:170–171`) — neither is a citable venue for a
conference submission.

Separately, `prop:multiclass` is **compiled in the main text** but does **not** appear in
`SUBMISSION_LEDGER.md §2`'s "TRUE COMPILED short-paper stack" list (`:58–60`), so the ledger's
inventory — the declared single source of truth — is incomplete.

I did verify the audit statements are mathematically right where they are checkable:
`thm:short-audA` is correct ($\sup|\gamma|=\tfrac12+|M|$ over unconstrained label kernels, attained
at $\eta_a\equiv0$ or $1$ on $D$, and $\widehat\beta\ge\tfrac12+|M|>|M|$ forces abstention);
`thm:short-audC`(i) has the right Hoeffding width for $u\in[-1,1]$ ($R\sqrt{\ln(2/\delta)/2n}
=\sqrt{2\ln(2/\delta)/n}$); `thm:short-audDE`'s union bound is right; `thm:short-audG`'s
$\alpha\ge\frac1{K+1}+\delta$ is the right feasibility threshold. The issue is availability of
proof, not correctness.

**Fix.** Either supply one-paragraph proofs in the appendix (each of Aud-A/C/DE/G is ≤ 6 lines) or
mark them explicitly as "stated here; proved in [long version]" with a stable citation.

---

### [MAJOR] F1-9 — The frontier is definitional, and $\beta$ never enters any deployed rule or any experiment

**Location.** `theory_setup.tex:24–26` (definition of $\gamma$); `theory_core_main.tex:12–23`
(proof of `lem:reduction`); `kbound_short.tex:41` (abstract), `:596–598`.

**Evidence.** $\gamma:=\E_{\mu_T}[\eta_a-s\mid D]$, so $M+\gamma=\E[\eta_a\mid D]-\tfrac12=\bar a-\tfrac12$
**by definition** — the proof of `lem:reduction` is literally the cancellation of $\E[s\mid D]$.
`thm:headline`(ii) is then "$|M|>\beta$ and $|\gamma|\le\beta$ imply $M+\gamma$ has the sign of $M$",
i.e. interval arithmetic; (iii) is the (genuinely nontrivial, and correct) statement that the
declared class is rich enough to realise the opposite drifts. So the mathematical content of the
headline theorem is: *the constraint set $|\gamma|\le\beta$ is tight*.

Meanwhile $\beta$ is a ghost parameter operationally:

```
$ grep -rn "beta" kga/*.py       # -> no matches
```

and the paper concedes "In the real-data tracks that follow, $\beta$ is *not* numerically supplied to
KGA" (`:596–598`), with the only $\beta$-instantiated study being the circular synthetic of F1-4.

**Why it matters.** The abstract's "we prove that … a strict adapt-or-freeze commitment is
uniformly sound if and only if the population evidence margin satisfies $|M|>\beta$" reads as a
substantive identifiability theorem. It is a correct but near-tautological statement about a
*declared* budget, and no deployment ever declares one.

**Fix.** Reframe: present `lem:reduction` explicitly as a *decomposition* (it is one), and put the
weight on `lem:nonid`'s richness construction, which is the real theorem. State in the abstract that
$\beta$ is declared, not estimated, and that KGA does not implement the $|M|>\beta$ test. Consider
adding the one result that would give the frontier teeth: conditions under which a *declared* $\beta$
is falsifiable from data (`thm:short-audG` is the seed of this).

---

### [MINOR] F1-10 — `prop:beatsboth-asym` is a sentence fragment, is unproved, and is applied to a rule it does not describe

**Location.** `kbound_short_appendix.tex:155–172`.

**Evidence.** The statement opens

```
156  Since $\Delta>0$ on $\{M>\beta\}$ and $\Delta<0$ on $\{M<-\beta\}$ for every admissible drift
157  (Theorem~\ref{thm:headline}).  The excess regret of always-freeze is exactly ...
```

— a subordinate clause with no main clause, inside a numbered proposition of a frozen submission.

The mathematics is *correct*: with $\delta^\star$ = frontier + abstain-to-freeze,
$R(\text{freeze})-R(\delta^\star)=\E_Q[|\Delta|\mathbf 1\{M>\beta\}]$ and
$R(\text{adapt})-R(\delta^\star)=\E_Q[|\Delta|\mathbf 1\{M<-\beta\}]+\E_Q[|\Delta|(\mathbf 1\{\Delta<0\}-\mathbf 1\{\Delta>0\})\mathbf 1\{|M|\le\beta\}]$;
I verified both partitions. But $\delta^\star$ is the **population** frontier rule, whereas the
sentence "This predicts the candidate-dependent Camelyon17 outcome (Table~\ref{tab:multiseed})"
(`:168–169`) applies it to the empirical $\widehat\Delta\pm\varepsilon$ rule — a different decision
function, on a track where $\beta$ is not even defined.

**Fix.** Repair the sentence; add the two-line partition proof (it fits); and downgrade the
Camelyon17 sentence to "is consistent with", or state and prove the analogue for the
$\widehat\Delta\pm\varepsilon$ rule (it holds with $\{M>\beta\}$ replaced by
$\{\widehat\Delta-\varepsilon>0\}$ *on the coverage event*).

---

### [MINOR] F1-11 — Wilson binomial intervals are reported on deterministic in-sample counts

**Location.** `paper/generated/empirical_audit/decision_metrics.json`
(`interval_coverage_observed.ci95_wilson`); described in `kbound_short_appendix.tex:271–283`.

**Evidence.** CIFAR-10-C: `{"count": 1940, "n": 2160, "rate": 0.8981481481481481,
"ci95_wilson": [0.8846780803898902, 0.9102045543745856]}`. By F1-1 the count 1940 is
$5\times\lfloor$in-sample interpolated-quantile coverage at $n=432\rfloor$ and is a function of $n$
alone; it is not a binomial draw from any coverage parameter. Its Wilson interval has no
frequentist interpretation. The same applies to the ImageNet-C `24/27` entry.

**Fix.** Delete these intervals, or replace them with intervals computed on a held-out partition
(cf. F1-1).

---

### [MINOR] F1-12 — `empirical_bernstein` estimates the range from the data by default, which invalidates the Maurer–Pontil bound

**Location.** `kga/certificate.py:171–178`.

**Evidence.**

```python
175    else:
176        rng = float(arr.max() - arr.min())
```

Maurer–Pontil (2009) requires $R=b-a$ to be an *a priori* bound on the support; substituting the
observed range makes the deviation term data-dependent and the stated $1-\alpha$ guarantee no longer
holds (the observed range under-estimates $R$, so $\varepsilon$ is anti-conservative). The docstring
warns "For $|p-y|$ paired losses the exact range is 2.0 and should be passed explicitly" but the
default silently does the wrong thing.

**Fix.** Make `benefit_range` a required argument, or default to 2.0 for $0/1$-loss paired benefits
and document that data-estimated ranges void the guarantee.

---

### [MINOR] F1-13 — The two "authoritative" generated artifacts disagree about CIFAR-10.1, the one track that breaches the level

**Location.** `paper/generated/kbound_result_manifest.json` → `tracks/cifar10_1_K`;
`paper/generated/empirical_audit/decision_metrics.json` → tracks "CIFAR-10.1 TENT/EATA/SAR".

**Evidence.** The manifest says `"false_adapt_unconditional": 0.1667,
"false_adapt_conditional": 0.4444, "n_test": 48`. `decision_metrics.json` says, for all three
adapters, `adapt.count = 0`, `false_adapt_unconditional {"count": 0, "n": 30, "rate": 0.0}`. A
conditional false-adapt rate of 0.4444 is undefined when the adapt count is zero, so the two
artifacts cannot both describe the same run; `app:claim-artifact` (`kbound_short_appendix.tex:271–283`)
presents both as the authoritative index.

This track is also the *only* place in the panel where the reported coverage is not the in-sample
identity: $25/30=0.8333$ for $n=30$, whereas the in-sample interpolated quantile at $n=30$ would give
$0.900$. So the single genuinely out-of-sample coverage measurement in the entire panel
**under-covers** ($0.833<0.90$) — and it is exactly the track where $\mathrm{FA}_{\mathrm u}=0.167>\alpha$.
That is the most informative datum in the paper about `thm:certificate`'s premise, and it is negative.

**Fix.** Reconcile the two artifacts; and promote the CIFAR-10.1 coverage failure from a "transfer
bar" footnote to a first-class discussion of when the coverage premise fails, since it is the only
honest test of it.

---

### [MINOR] F1-14 — Lean: the exchangeability bridge assumes its own conclusion, and `Impossibility.lean`'s headline lemma is `linarith`

**Location.** `formal/KBound/Probability/Exchangeable.lean:21–33`;
`formal/KBound/Impossibility.lean:48–55`; disclosed at `kbound_short_appendix.tex:261–269`.

**Evidence.** The "exchangeable-score miss bound" takes the uniform rank law as a *hypothesis*:

```lean
theorem exchangeable_scores_miss_le_alpha ... (hexch : μ = uniformIndexMeasure n) ...
```

and the hard step (exchangeability $\Rightarrow$ uniform held-out rank) is not formalized —
indeed there is no definition of exchangeability at all in the tree:

```
$ grep -rn "Exchange\|permut\|Perm" --include=*.lean formal/
# only docstrings / filenames / imports; no Perm-invariance definition
```

Likewise `forced_abstention_probability`, tagged "Paper `thm:imp` (iii): matched-evidence abstention
rate", is

```lean
theorem forced_abstention_probability {qa qf alpha : ℝ} (hfa : qa ≤ alpha) (hff : qf ≤ alpha)
    (_hprob : qa + qf ≤ 1) : 1 - qa - qf ≥ 1 - 2 * alpha := by linarith
```

— real-arithmetic, with no probability, no two worlds, and no matched evidence law.

**In fairness**, `app:formal` and `TheoremMap.lean`'s docstring both disclose precisely this
("Lean does not formalize … the lift from a conditional uniform-index model to arbitrary
exchangeable deployment processes"; "PEN-AND-PAPER: measurable target-label kernels, equality of
induced evidence laws, and membership of the constructed laws in the declared drift class"). This is
better disclosure than most formalization appendices. The problem is only that the *declaration
names* (`Impossibility.lean`, `exchangeable_scores_*`) overclaim relative to their content, and a
reader scanning the file tree will draw the wrong conclusion.

**Fix.** Rename to reflect content (`abstention_probability_arithmetic`,
`uniformIndex_scores_miss_le_alpha`), or formalize `Exchangeable`: define permutation invariance of
the score vector and derive the uniform-rank law. The latter is a genuinely tractable mathlib
exercise and would be the single highest-value addition to the formalization.

---

### [NIT] F1-15 — Minor statement imprecisions in the compiled theorems

**Location.** `kbound_short_appendix.tex:344–351`, `:36–44`; `theory_setup.tex:59–70`.

**Evidence and fix.**
* `thm:short-audDE`: "replacing $M$ by the batch estimate $\widehat M\pm t_M(m,\delta')$ gives a
  rule computable entirely from data" — the resulting decision rule is never written down. State it
  ($\textsc{adapt}$ iff $\widehat M-t_M>\widehat\beta$, etc.) so the $\delta+\delta'$ union bound is
  checkable.
* `thm:imp`(ii): define $\mathrm{err}(g)$ (sum of the two error probabilities, not the average) —
  the identity $\inf_g\mathrm{err}=1-\TV$ is false for the equal-prior average, where it is
  $(1-\TV)/2$.
* `def:strict-sound` is written as "$a=\textsc{adapt}\Rightarrow\Delta(P)>0$ for every $P$", which
  parses as a material conditional on the *action label* rather than a property of the rule. Rewrite
  as "$\textsc{adapt}$ is uniformly sound at $z$ iff $\Delta(P)>0$ for all $P\in\mathcal C_\beta(z)$".
* `kbound_short.tex` uses `\newcommand{\TV}{\mathrm{TV}}` (line 27) but $\TV$ is used in
  `thm:imp`(ii) without ever defining total variation distance in the short paper.

---

## What I checked and could NOT fault

* **`lem:reduction` algebra.** $\Delta=\mu_T(D)(\bar a-(1-\bar a))=2\mu_T(D)(\bar a-\tfrac12)$
  requires (i) binary $Y$, (ii) $0/1$ loss, (iii) $f_0(x)\ne f_a(x)\Rightarrow\{f_0(x),f_a(x)\}=\{0,1\}$,
  and (iv) $\mu_T(D)>0$ for the sign claim. All four are stated in `ass:deploy`
  (`theory_setup.tex:14–18`) — including $\mu_T(D)>0$, which is the one most papers forget. Correct.
* **`lem:nonid` construction.** $K^\theta$ is a valid Bernoulli kernel for $\delta<\tfrac12$;
  $\gamma_\theta=\theta\delta-M$ with $|\gamma_\theta|\le\delta+|M|<\beta$ for
  $\delta<\beta-|M|$, so both worlds are in $\mathcal C_\beta$; $Z$ is label-free so the evidence
  laws coincide; $\Delta_\theta=2\mu_T(D)\theta\delta\ne0$. The $\beta=0$ carve-out is correct and
  necessary. Quantifiers are in the right order ("for any $M$ with $|M|<\beta$ **there exist**
  $P^1,P^2$").
* **`cor:matched-abstain`.** Action probabilities agree across the two evidence-identical worlds;
  $\Pr[\textsc{adapt}]\le\alpha$ in the negative world, $\Pr[\textsc{freeze}]\le\alpha$ in the
  positive world, three probabilities sum to one $\Rightarrow\Pr[\textsc{abstain}]\ge1-2\alpha$.
  Correct, and correctly requires $\alpha<\tfrac12$ to be non-vacuous.
* **`prop:closed-band`.** At $|M|=\beta>0$ the drift $\gamma=-M$ satisfies $|\gamma|=\beta$ so the
  zero-benefit world *is* admissible; both directional error events $\{\textsc{adapt},\Delta\le0\}$
  and $\{\textsc{freeze},\Delta\ge0\}$ fire simultaneously at $\Delta=0$, so the $1-2\alpha$ bound
  follows in that single world without needing two. The $M=\beta=0$ case is handled separately and
  correctly.
* **`thm:headline` (i)–(iv).** Sufficiency, necessity and the iff are all correct *given* the
  richness caveat, which the theorem states up front ("For a narrower declared subclass not closed
  under these constructions, only the sufficiency clause follows"). This caveat is exactly right and
  is usually missing from papers of this type.
* **`thm:certificate` proof.** $\{g=\textsc{adapt},\Delta\le0\}\subseteq\{|\widehat\Delta-\Delta|>\varepsilon\}$
  is correct; the freeze case is symmetric; the two events are disjoint so each is $\le\alpha$.
  `rem:fa-marginal` correctly warns that $\mathrm{FA}_{\mathrm c}$ is *not* bounded.
* **`theory_v2/minimax_optimality_theorem.tex`.** Verified: $\mathrm{KL}(P_+^n\|P_-^n)=2n\Delta^2/\sigma^2$;
  Bretagnolle–Huber $1-\TV\ge\tfrac12e^{-\mathrm{KL}}$ with the stated elementary derivation;
  $2\alpha\ge\tfrac12e^{-2n\Delta^2/\sigma^2}\Rightarrow n\ge\frac{\sigma^2}{2\Delta^2}\log\frac1{4\alpha}$;
  the achievability side's $\varepsilon_n<\Delta/2$ (not $<\Delta$) requirement is *correctly*
  derived, which is the step most people get wrong; the constant ratio $\to16$ is right. The
  restriction $\alpha<\tfrac14$ and the explicit "order-optimal, not tight constants" scoping are
  honest.
* **`thm:short-audA`.** $\sup_{P}|\gamma(P)|=\tfrac12+|M|$ over label kernels with fixed
  $(\mu_T,f_0,f_a,s)$, attained at $\eta_a\equiv0$ (or $1$) on $D$; since the audit's law is identical
  across those worlds, $\Pr[\widehat\beta\ge\tfrac12+|M|]\ge1-\delta$; and $\widehat\beta>|M|$ forces
  abstention because $M\in[-\tfrac12,\tfrac12]$. Correct.
* **`prop:beatsboth-asym`'s regret partition** (see F1-10) — the algebra is right.
* **Lean tree hygiene.** `grep -rn "sorry\|admit\|axiom" --include=*.lean` over all 27 files:
  zero hits. Compilation is not possible here (no `.lake`), but the sources contain no escape hatches.
* **`\iffalse` accounting.** `lem:gate`, `prop:lecam-finite`, `prop:cert-sample`,
  `thm:conj1-dichotomy`, `conj:gen`, `thm:ev-rate` are all genuinely excluded from the build, matching
  `SUBMISSION_LEDGER.md:47–51`.

---

## Open questions for the author

1. **Why is $\varepsilon$ calibrated in-sample on the stress grids at all?** With 432 cells per seed
   and 5 seeds, a clean split (calibrate on seeds 0–2, deploy on seeds 3–4) costs almost nothing and
   would turn F1-1 from a fatal objection into a genuine result. Was there a reason not to?
2. **What is $\Pr[\textsc{abstain}]$ supposed to be?** `cor:matched-abstain` says any
   $\alpha$-safe rule must abstain with probability $\ge1-2\alpha=0.8$ *on matched evidence*, yet
   KGA's decision coverage on CIFAR-10-C is $0.67$/$0.63$ (`tab:decisive`), i.e. abstention
   $\approx0.35$. That is consistent (the evidence is not matched there), but it means the paper
   never exhibits a track where the impossibility bound bites. Is there one?
3. **Under the exact-rank rule, ImageNet-C SAR KGA $\equiv$ always-freeze on 3/5 seeds.** Do you
   regard the pooled "beats-both" as a real effect, or as a seed-2 artifact? A per-seed sign test
   ($2$ wins, $3$ ties, $0$ losses vs always-freeze) does not reach significance.
4. **Is there any deployment scenario in which $\beta$ would actually be declared?** If not, would
   you consider making `thm:short-audG` (domain-level verifiability at $\alpha\ge\frac1{K+1}+\delta$)
   the headline instead? It is the only result in the stack that says something a practitioner could
   act on, and it is currently buried unproved in the appendix.
5. **Exchangeability for online adapters.** For Camelyon17 EATA-online and Office-Home SAR-online,
   $f_a$ depends on stream order, so calibration and deployment residuals are not identically
   distributed. Do you have a shift correction in mind, or should those tracks be relabelled as
   episodic?
6. **`theory_v2/tight_constants_closure.tex`**: do you agree the Step-1 MLR argument should be
   replaced by the $p+q\le\alpha$ Neyman–Pearson budget split? If so, the theorem survives unchanged
   and the "closed" claim stands.
