# Adversarial review — kbound_short (22 pp), 2026-07-04

Three-agent review: (A) award-winning-paper craft research (14 sourced traits from NeurIPS/ICLR
award citations, reviewer guidelines, Peyton Jones/Dreyer/Carlini); (B) full line-edit of the
manuscript; (C) 2024–2026 literature-coverage audit. Synthesis and rating below.

## Verdict and rating (conference-review scale)

| Axis | Score | One-line basis |
|---|---|---|
| Originality | 9/10 | Exact iff-frontier for the benefit sign + abstention necessity; first machine-checked conformal coverage |
| Soundness/Quality | 8.5/10 | Pre-registration, report-all-arms, withdrawals disclosed, validators + Lean; official-baseline arm still open |
| Clarity | 8/10 | Post-revision three-theorem staircase is strong; ~40 line-level consistency items remain |
| Significance | 7.5/10 | Safety layer + regime map; rises with external replication/official baselines |
| **Overall** | **8/10 — accept, top-venue viable** | Blocking gap before sending anywhere: missing citations (below) |

## Scorecard against the 14 award-paper traits

Strong (✓✓): one refutable idea stated dangerously plainly (|M|>β, boxed); skeptic pre-answered
(radius-ablation row, six-gate comparison, dual bootstrap designs); systematic evidence grid
(432×5 stress + 9 datasets, pre-registered); candor-as-strength (withdrawals, FAILs reported —
the trait NeurIPS guidelines explicitly reward); reproducible headline (research_lock, packet,
Lean).
Present (✓): question-title answered exactly; refutable contribution bullets; worked minimal
example; intuition-before-formalism; Monday-morning takeaway (wrap any adapter, tune α).
Weakest (~): Figure 1 is a flow diagram, not the result — award papers put the whole result in
Fig. 1 (candidate: promote the regime-map table or natural-forest plot); the intro never names
the *widely-held assumption being overturned* in one sentence ("more adaptation is better" /
"accuracy estimators suffice as gates"); stakes-beyond-benchmark stated only implicitly.

## Weakness 1 (the real miss in 22 pages): eleven missing references

Agent C verified all. In priority order:

MUST: (1) **MaNo** (Xie et al., NeurIPS 2024, arXiv:2405.18979) — the paper *uses* MaNo-style
features; uncited source of own signal is the most flag-worthy gap. (2) **TTA survey** (Liang,
He, Tan, IJCV 2025, arXiv:2303.15361). (3) **Barber et al. 2023, beyond exchangeability**
(arXiv:2202.13415) — their TV coverage-gap is the population form of the calibration-drift
budget; the FA≤α-under-shift claim must engage it. (4) **Tibshirani et al. 2019 weighted
conformal** (arXiv:1904.06019). (5) **PeTTA** (NeurIPS 2024, arXiv:2311.18193) — closest prior
"sense collapse, adjust commitment" gate. (6) **PAC prediction sets under covariate shift**
(Park et al., ICLR 2022, arXiv:2106.09848).
SHOULD: (7) Gibbs–Candès adaptive conformal (arXiv:2106.00170); (8) Learn-then-Test
(arXiv:2110.01052) + (12) RCPS (arXiv:2101.02703) as the risk-control lineage of FA≤α;
(9) nuclear-norm estimator (Deng et al., ICML 2023, arXiv:2302.01094); (10) DeYO (ICLR 2024,
arXiv:2403.07366); (11) reset-timing for long-term TTA (ICLR 2026, arXiv:2603.03796);
(13) Lean formalizations of learning bounds (arXiv:2503.19605; arXiv:2602.02285) — cite next to
the Lean artifact. OPTIONAL: (14) SoTTA/RoTTA grouped mention.
Placement: 3–4 sentences in Related Work (conformal-under-shift + risk-control lineage;
when-to-adapt cluster), one clause in Method (weighted conformal), one clause in
Reproducibility (Lean prior art), MaNo/nuclear-norm where evidence features are described.

## Weakness 2: consistency drift (top of agent B's 40 line items)

(1) K-Bound vs. \textsc{KGA} discipline — framework vs. method blurred at ~6 sites (incl. one
\textsc{K-Bound}); (2) M vs. $\widehat M$ — Fig. 1 caption uses $\widehat M$, Theorem 2 uses M;
one sentence defining $\widehat M$ as the finite-sample estimate of M resolves it;
(3) beats-both/no-harm hyphenation (noun vs. verb) inconsistent; (4) \textsc{adapt} vs.
\textsc{Adapt} vs. \text{adapt} across theorem/algorithm/FA definitions; (5) FA_u spacing
({=} forced vs. not). Full 40-item list in the session log; none is substantive.

## Weakness 3: flow (residual, minor)

Abstract sentence 1 is 61 words (split once); contribution bullet 2 spans six lines (split);
three em-dash pileups in Related Work; "supporting and negative regimes" is a roll call;
camera-protocol subsection interrupts results→guarantees (move to appendix at venue time);
"pooling artifact" is used before it is explained (add 4-word gloss at first use).

## Prioritized actions

P0 (before the professor email, ~1–2 h): add the 6 MUST + LTT/RCPS/nuclear-norm/Lean-prior-art
citations with the placements above; fix K-Bound/KGA drift; add the $\widehat M$ definition
sentence.
P1 (same pass, 30 min): split the three long sentences; standardize hyphenation, small-caps
decision tokens, FA_u spacing.
P2 (venue formatting): Fig. 1 upgrade to a result-carrying panel; camera subsection → appendix;
name the overturned assumption in intro paragraph 1.

## What the award-paper research says this paper already is

The single most common denominator of award winners: *one idea, stated dangerously plainly,
defended overwhelmingly* — the thesis extractable from the title, verifiable in one figure, with
the skeptic's experiment already run. This paper has the plain statement (|M|>β, boxed), the
overwhelming pre-registered defense, and an integrity record (withdrawals, FAILs, Lean) that
review committees explicitly say they reward. What separates it from the award tier is not
writing: it is the official-baseline arm, external replication, and a Figure 1 that carries the
result — two of which are evidence work, not editing.
