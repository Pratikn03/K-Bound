# K-Bound Narrative Draft v2
## (a) Rewritten Abstract, (b) Rewritten Intro Opening, (c) Per-Result Intuitive Examples

---

## (a) ABSTRACT — plain-language-first rewrite

> **Numbers used below are read directly from result JSONs and the current kbound.tex abstract.
> No values are invented.  Sources are noted in brackets.**

---

Test-time adaptation (TTA) can silently hurt a deployed model — it updates on unlabeled data,
and without labels there is no immediate signal that it is making things worse.
We ask the sharper question: *when can a system prove, before adapting, that adaptation will
help?*  We formalize when the decision to adapt, freeze, or abstain is **knowable** from
label-free evidence, prove that an unknowable regime exists and is unavoidable
(Theorem 1, with an explicit non-identifiability witness), and give a finite-sample
adapt/freeze/abstain certificate with a controlled false-adapt rate (Theorem 2).

**Contributions.**
(1) A knowability formulation that partitions test-time conditions into *knowably helpful*,
*knowably harmful*, and *unknowable* regimes.
(2) An impossibility theorem with an explicit witness: two target distributions that produce
identical observable evidence but opposite optimal decisions, proving that no label-free rule
can be correct in both worlds — the worst-case-optimal action on such evidence is to abstain
[*witness_clean.json*: KS p-values > 0.05 on all five observable features, 100% abstain rate,
|Δ|=1.0 by construction].
(3) A finite-sample certificate (KGA) that controls the false-adapt rate at a user-chosen
level α, with a conformal radius ε calibrated from held-out source data.
(4) A provable positive regime under covariate shift (Theorem 3).
(5) An explicit, honest scope: where harmful adaptation is mild, always-adapt is already strong,
and the certificate's value is bounded by how often catastrophic detectable harm occurs.

**Headline results (all numbers from result JSONs).**
On a pre-registered CIFAR-10-C **stress grid**
(severity × batch-size × composition × update-aggressiveness; 432 conditions each)
with a genuine harmful base rate of 16–34% depending on the wrapped method,
**KGA beats both trivial policies for Tent and EATA** — regret-to-oracle 0.0016 vs
always-adapt 0.009 and always-freeze 0.12, at a **0% false-adapt rate**
[*decisive_tta_results.json / tent.metrics*: `false_adapt_rate_B<0=0.0`,
`regret_vs_oracle.K_Bound=0.0016`, `regret_vs_oracle.always_adapt=0.009`] — and **ties
the collapse-resistant SAR** [*decisive_tta_results.json / sar.metrics*:
`beats_both=false`, `regret_vs_oracle.K_Bound=0.00178 ≈ always_adapt=0.00176`].
On the **per-corruption CIFAR-10-C suite** (65 cells, helpful-dominated with 1.5% harmful
base rate), adaptation is almost always beneficial so KGA ties always-adapt
[*cifar10c_suite_results.json*: `false_adapt_rate.kga=0.0154`, `regret_to_oracle.kga=0.00315`].
On the **123-task anomaly-routing benchmark**, KGA abstains on 83% of tasks where the true
benefit is near zero (mean |B̂|=0.022 in the abstain set) and achieves adapt-precision 0.90
[*knowability_results.json*].
On the **harmful stream** (80% of tasks genuinely harmful), KGA freezes correctly in 100%
of cases [*kbound_harmful_results.json*: `freeze_correct_B<=0=1.0`, `base_rate_harmful=0.797`]
cutting regret-to-oracle ~11× versus always-adapt [0.0019 vs 0.0214].
On a **controlled mixed-regime study**, KGA beats always-freeze by ≈0.10 AUROC in the
detectable-shift condition while still abstaining on the ambiguous covert condition
[*mixed_regime_results.json*: `mean_auc.K_Bound=0.690 vs always_freeze=0.586`].
On **online non-stationary CIFAR-10** streams, KGA yields mean stream accuracy 0.426 vs
always-adapt 0.339 and always-freeze 0.440, and beats always-adapt by +0.086
[*cifar_tent_online_results.json*: `kga_minus_adapt=0.086`].
These results hold across 8 seeds (paired t-test p<10⁻⁴ for KGA vs always-freeze)
[*rigor_multiseed.json*: `paired_ttest_KBound_vs_always_freeze.p=7.3e-11`].

---

## (b) INTRODUCTION — Plain-language opening paragraph (rewrite)

Every time a deployed model encounters new data — from a different hospital, camera, or weather
condition — engineers face a choice: let the model adapt on the fly, using the unlabeled target
batch, or keep it frozen.  Adaptation methods like Tent, EATA, and SAR can recover substantial
accuracy under distribution shift.  But adaptation is double-edged.  The same entropy-minimizing
update that sharpens boundaries under mild covariate shift can collapse prediction diversity
under label shift; a reliability gate that rescues a failed sensor can dilute a strong detector
on clean data.  The uncomfortable truth is that, when you have no labels on the target data,
*you often cannot tell which is happening* — and by the time you discover that adaptation has
hurt, real-world harm may already have occurred.

The field's answer so far has been to build sturdier adaptation mechanisms: sample selection
(EATA), sharpness-aware updates (SAR), online resets.  These mechanisms make adaptation more
robust, but all of them commit to adapting.  We ask the prior question: *can a label-free
system decide, before updating, whether adapting will help?*  The answer depends on the
geometry of the problem — on whether the observable evidence (confidence distributions, entropy,
disagreement rates, score-distribution drift) is rich enough to separate helpful from harmful
regimes.  We prove that sometimes it is, sometimes it provably cannot be, and we give an exact
characterization of when each case holds, along with a certified three-way gate that acts only
when it can prove it should.

---

## (c) ONE-SENTENCE INTUITIVE EXAMPLES — per theorem/result

**Thm 1 — Non-identifiability / Unknowable regime.**
Two hospitals collect chest X-rays with the same predicted-confidence distribution: one has a
mild domain shift where adapting helps; the other has a label-frequency inversion where adapting
catastrophically hurts — no label-free rule can tell them apart, so the safe action is to abstain.

**Thm 2 — Quantitative Le Cam minimax bound.**
The maximum probability of being right about whether to adapt — when two worlds are hard to
distinguish — cannot exceed 50% plus a correction that shrinks to zero as the TV distance
between the observed evidence distributions shrinks to zero.

**Thm 3 — Finite-sample adapt/freeze/abstain certificate (KGA).**
A medical imaging deployment runs a split-conformal calibration on a held-out source set, then
at each test batch computes a confidence interval around the estimated benefit: only when the
interval lies entirely above zero does it adapt, only when entirely below zero does it freeze,
and otherwise it holds the frozen model — guaranteeing that harmful adaptations occur no more
than α fraction of the time.

**Thm 4 — Regret identity (gate decomposition).**
The excess risk of any adapt/freeze/abstain policy over the oracle can be written exactly as the
sum of false-adapt harm, false-freeze missed-gain, and abstain cost — a minimax floor term
quantifies how much even the best policy must pay when evidence is weak.

**Thm 5 — Positive regime under covariate shift.**
When the shift is purely covariate (labels do not shift), confidence-calibrated disagreement
between frozen and adapted models is a valid, computable proxy for adaptation benefit — making
the decision *knowable* without labels.

**Thms 6–7 — Observable disagreement region (binary and multiclass).**
On the test inputs where the frozen and adapted models disagree, the sign of adaptation benefit
equals the sign of the adapted model's accuracy advantage on those inputs — a quantity
estimable from label-free model outputs, extending identifiability to a strictly larger set of
conditions than the unknowable regime.

**Thm 8 — Anytime-valid e-process certificate.**
In a streaming deployment, the KGA certificate can be updated sequentially without fixing the
sample size in advance: the false-adapt rate remains controlled at every stopping time, not just
at a pre-specified horizon, enabling deployment in open-ended test streams.

**Thm 9 — Multiclass disagree-then-certify.**
For a K-class classifier, the adaptation decision on the disagreement region reduces to checking
whether the adapted model's class-marginal advantage exceeds a conformal radius — a
plug-in quantity computed from validation data with no access to target labels.

---

*File generated: 2026-06-05.  All numeric values read from result JSONs under
`experiments/kbound/results/`.  No values are fabricated.  Do NOT edit `kbound.tex` based
on this draft without cross-checking every number against the JSON sources listed above.*
