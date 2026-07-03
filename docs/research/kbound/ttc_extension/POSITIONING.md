# K-Bound / TTC extension — Knowable Test-time Compute

**One sentence:** Should this query get extra test-time compute (retrieval, longer
reasoning, a bigger model) — decided label-free, with a certified false-spend rate,
an explicit unknowable region, and a proof of exactly what no gate can ever decide?

This is the K-Bound benefit-sign frontier instantiated on the hottest decision in
ML systems. The theorems transfer verbatim (they are proven for any split-observable
functional); the contribution of THIS paper is the instantiation, the cost-indexed
frontier, the witness, and the empirical program.

## Gap map (searched 2026-07-02)

| Line of work | What they do | What they DON'T have |
|---|---|---|
| Adaptive RAG gates: [Self-RAG / DRAGIN / SeaKR](https://arxiv.org/pdf/2406.19215), [TARG "Retrieval as a Decision"](https://arxiv.org/pdf/2511.09803), [LLM-independent adaptive retrieval](https://ar5iv.labs.arxiv.org/html/2505.04253) | Threshold an uncertainty/logit signal to decide retrieval | **No error guarantee of any kind.** These are exactly the β=0 face of the frontier: sound iff calibration does not drift — the failure mode our Thm frontier(iv) characterizes |
| Conformal / calibrated routers: CP-Router, [UCCI](https://arxiv.org/html/2605.18796), [RouteNLP conformal cascading](https://arxiv.org/html/2604.23577v1), [NP-routing with distribution-free safety](https://arxiv.org/pdf/2603.14623), [early-abstention cascades](https://arxiv.org/html/2502.09054v1) | Distribution-free guarantees on the routing score / routed subset | Guarantees hold **only under exchangeability (γ=0)** — no drift budget, no identifiability analysis of the benefit SIGN, no impossibility floor, no abstain-forced region |
| TTC allocation: [constrained policy optimization](https://arxiv.org/abs/2604.14853), [selective verification](https://arxiv.org/abs/2606.19808v1), [overthinking analysis](https://arxiv.org/html/2604.10739v1), [survey](https://arxiv.org/html/2507.02076v1) | Optimize accuracy under a compute budget; document that extra thinking can HURT | Treat benefit as estimable everywhere; **no characterization of when the benefit sign is identifiable at all**, no certified decision rule under drift |

**The surviving gap:** nobody provides (a) an exact identifiability condition for the
benefit sign of extra compute under calibration drift, (b) a spend/answer/abstain
certificate with false-spend ≤ α that stays valid off the exchangeability face,
(c) the impossibility floor — pairs of deployment worlds indistinguishable from all
label-free observables with opposite benefit — or (d) the one-bit minimal-supplement
result. K-Bound owns all four; this paper carries them to the TTC audience.

**Motivating empirics already in the literature:** "overthinking" — extended
reasoning abandons previously correct answers ([2604.10739](https://arxiv.org/html/2604.10739v1));
"retrieval-state lock-in" — confidence rises while retrieval-poisoned answers are
wrong ([2606.22728](https://arxiv.org/pdf/2606.22728)). Harmful adaptation exists in
TTC and is precisely the regime the certificate is for.

## Honest relation to the K-Bound core
Same theory spine (cited, not re-proven). New here: the TTC instantiation and
reduction, the cost-indexed frontier M_λ (Prop 1), the certified willingness-to-pay
λ\*(query distribution), the TTC witness, and the TTC empirical program. Lives inside the K-Bound project as its generality showcase.
