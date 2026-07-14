# K-Bound Extension: Knowable Test-time Compute (TTC) — 2026-07-02

**Part of the K-Bound project** (folded in by author decision — one project). This is
the showcase application of the paper's stated split-observability generality: the
theorems already hold for any split-observable functional; this extension instantiates
them on the LLM test-time-compute decision. Use as an 'Extensions' section/appendix of
kbound.tex (wire with one \input when ready).

**Claim of this seed:** the K-Bound benefit-sign frontier, certificate, impossibility
witness, and one-bit result transfer verbatim to the decision *"should this query get
extra test-time compute (retrieval / longer reasoning / bigger model)?"* — plus one new
result (Prop 1): pricing compute shifts the observable margin, giving a **certified
willingness-to-pay** λ\* per deployment distribution, with no added unknowability.

Files: `POSITIONING.md` (gap map, searched 2026-07-02) · `THEORY.md` (instantiation +
Prop 1 with proof + open list) · `val_ktc_routing.py` → `ktc_results.json`,
`fig_ktc_regret.png`, `fig_ktc_lambda_frontier.png` (all real runs, seeded).

## Validated in this seed (synthetic, executed 2026-07-02)

| Block | Result |
|---|---|
| A — witness | Matched-observable gold/distractor worlds (opposite Δ, \|Δ\|≈0.15): certificate abstains 84%; any forced commit is wrong 52% (chance) |
| B — routing under drift, λ=0 | KTC regret **0.0019** vs never 0.089 / always 0.122 (**beats both**); FA **0.000** ≤ α. The β=0 entropy gate (Self-RAG/DRAGIN/TARG-style) false-spends **46%**; the exchangeability-assuming conformal router **42.5%** — both silently violate their nominal levels under calibration drift, exactly as Thm frontier(iv) predicts |
| B — λ=0.08 (priced compute) | KTC regret 0.0000, FA 0, beats both; entropy gate 37× worse regret |
| C — Prop 1 | sign(Δ_λ) = sign(μ(D)(M_λ+γ)) exact in 900/900 checks; \|M_λ\|>β frontier classification 72/72 correct where \|γ\|≤β |

## Honest scope (what this is NOT yet)
- **All synthetic.** No real LLM has been queried. The generative model encodes the
  documented phenomena (overthinking, retrieval lock-in) but real evidence maps, real
  drift, and real costs may behave differently — the K-Bound Camelyon lesson applies.
- The witness uses constructed observables; the real-LLM version (poisoned vs gold
  retrieval corpora, matched similarity) is the flagship experiment of the full paper.
- Sequential re-decisions within one query (think-again-after-thinking) need new
  theory (THEORY.md §5.2).

## Why this raises the ceiling
The K-Bound paper's novelty is capped by prior art on ITS question. Nobody owns THIS
question: none of the adaptive-RAG gates has any error guarantee; none of the conformal
routers survives drift; none of the TTC-allocation line characterizes when the benefit
sign is knowable at all. The theory is already proven and machine-checked; the seed's
experiments already run. Inside K-Bound it becomes the demonstration that the frontier is a general theory of
label-free deployment decisions, not a TTA trick — aimed at the field's largest
current audience.

## Next steps (ordered)
1. Real-LLM pilot: one open model + one QA set + gold/poisoned corpora; reproduce
   Blocks A/B with real evidence features (logits, self-consistency, retrieval scores).
2. Pre-register the protocol (research_lock style) BEFORE the decisive run.
3. Multicandidate arm: {direct, RAG, long-CoT} with family-wise false-spend ≤ α
   (K-Bound Bonferroni theorem, transfers unchanged).
4. Sequential re-decision theory (the genuinely new math).
