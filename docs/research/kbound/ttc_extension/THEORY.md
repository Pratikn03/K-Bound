# K-Bound / TTC extension — theory instantiation (draft, 2026-07-02)

Status ledger: everything in §1–§3 is PROVEN (it is the K-Bound spine, cited, applied
to a new functional after verifying split-observability). Prop 1 is NEW and proven
below (elementary). §5 lists what remains genuinely open before this section can expand.

## 1. Setup (the TTC decision as a K-Bound instance)

Query X ~ Q (deployment, unlabeled). Two answering policies:
- f0(X): direct answer (cheap; frozen model / no retrieval / short reasoning);
- fa(X): augmented answer (retrieval, extended reasoning, or a larger model),
  with observable per-query compute cost c(X) ≥ 0 (tokens / latency / $).

Correctness q0, qa ∈ {0,1} against the (unavailable) ground truth. Benefit
Δ = E_Q[qa − q0]. Cost-adjusted benefit at price λ ≥ 0:
  Δ_λ = Δ − λ·E_Q[c].
Decision: SPEND (use fa) / DIRECT (use f0) / ABSTAIN (route to safe default = f0,
flag). False-spend: SPEND while Δ_λ ≤ 0. This is verbatim the K-Bound
adapt/freeze/abstain problem with "adapt"=SPEND.

## 2. The reduction (K-Bound Lemma 1, multiclass form, applies verbatim)

Let D = {x : fa(x) ≠ f0(x)} — the observable disagreement set (answers differ as
strings/choices). Off D the answers coincide, so qa = q0 pointwise. Hence
  Δ = μ(D)·(p_a − p_0),   p_i = P(q_i = 1 | X ∈ D),
exactly K-Bound's multiclass reduction. With s a source-calibrated correctness
score for the augmented answer on D (verifier score, self-consistency vote,
calibrated logit margin):
  M := E[s_a − s_0 | D]  (observable),   γ := E[(p_a−p_0) − (s_a−s_0) | D]
  sign Δ = sign(M + γ),   |γ| ≤ β  (declared calibration-drift budget).
**The TTC benefit sign is a split-observable functional.** All five K-Bound
theorems therefore hold for it verbatim:

- **Frontier:** sign Δ identifiable over the drift class iff |M| > β; the
  SPEND/DIRECT/ABSTAIN rule at threshold β is the unique maximal sound certificate.
  Corollary (the gate audit): every uncertainty-threshold retrieval gate
  (Self-RAG-, DRAGIN-, TARG-style) is the β = 0 face — sound iff the confidence
  calibration does not drift between calibration and deployment, silently unsafe
  exactly when it does.
- **Impossibility:** there exist deployment worlds with identical label-free
  observables and opposite Δ (witness in §4); any rule with false-spend and
  false-direct ≤ α must abstain on matched evidence w.p. ≥ 1 − 2α.
- **Certificate:** a benefit estimate with a valid conformal radius yields
  false-spend ≤ α (finite-sample); anytime-valid streaming and multicandidate
  (route among {direct, RAG, long-CoT, big-model}: family-wise false-spend ≤ α by
  the K-Bound Bonferroni theorem) transfer unchanged.
- **One bit:** no label-free-testable assumption class identifies sign Δ; the
  minimal untestable supplement is one orientation bit (e.g., "retrieval quality
  is no worse at deployment than calibration" — falsifiable, not verifiable).

## 3. Prop 1 (NEW): the cost-indexed frontier and certified willingness-to-pay

Since c is observable, define the observable cost-adjusted margin
  M_λ := M − λ·c̄/μ(D),   c̄ := E_Q[c],  μ(D) = disagreement rate (observable).
Then sign Δ_λ = sign(M_λ + γ) with the SAME γ. Hence for every price λ:

  **sign Δ_λ is identifiable over the drift class iff |M_λ| > β.**

M_λ is strictly decreasing in λ, so the identifiable set is a union of at most two
intervals in λ, and
  λ*_spend := sup{ λ ≥ 0 : M_λ > β }  =  μ(D)·(M − β)/c̄  (when M > β)
is the **certified willingness-to-pay**: the largest price at which SPEND is still
certifiably beneficial under every drift within budget. Dually, prices with
M_λ < −β certify DIRECT ("not worth it at any drift"), and the band |M_λ| ≤ β is
the priced unknowable region.
*Proof.* Δ_λ = μ(D)(p_a − p_0) − λc̄ = μ(D)[(M + γ) − λc̄/μ(D)] = μ(D)(M_λ + γ);
μ(D) > 0; apply the K-Bound frontier to the split-observable pair (M_λ, γ);
monotonicity of M_λ in λ is linear with negative slope c̄/μ(D) > 0. ∎
(Cost enters the OBSERVABLE margin, not the drift term — pricing shifts the
frontier, it does not add unknowability.)

## 4. The TTC witness (impossibility instance)

Two deployment worlds for a RAG system, identical observables:
- W+ (gold world): retrieval returns passages containing the correct answer;
  fa flips wrong direct answers to right on D. Δ > 0.
- W− (distractor world): retrieval returns fluent, on-topic, plausible passages
  containing a coherent WRONG answer (documented as "retrieval-state lock-in");
  fa flips right answers to wrong on D. Δ < 0.
Retrieval scores (similarity), fluency, self-consistency of the augmented answer,
disagreement rate, and confidence are matched by construction — similarity and
fluency measure plausibility, not truth. All label-free observables share one law;
the benefit signs are opposite; no gate can decide; abstention is mandatory.
(§validated synthetically in val_ktc_routing.py; a real-LLM instantiation with
poisoned retrieval corpora is the flagship experiment of the full paper.)

## 5. Genuinely open for the full paper (do NOT claim yet)
1. Real-LLM validation: everything below §4 currently synthetic.
2. Sequential compounding: multi-STEP reasoning (spend again after observing the
   first thought) = optional-stopping version; K-Bound anytime theorem covers the
   stream over queries, not re-decisions within one query — new theory needed.
3. Verifier-in-the-loop: when s comes from a learned verifier, γ includes verifier
   drift; a self-normalized audit for verifier calibration (Wave-5 τ′ analogue).
4. The λ\*-curve estimator's own conformal validity (plug-in vs debiased).
