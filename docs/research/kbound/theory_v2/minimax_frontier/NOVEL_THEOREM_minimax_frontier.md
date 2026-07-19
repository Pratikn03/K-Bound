# Minimax optimality of the |M|>β frontier + beats-both characterization

STATUS: working note (2026-07-18). Proofs drafted here; adversarial verification +
numerical witness pending before any fold-in to the paper. Single-paper structure.

## Why this is the novel angle (positioning)

The two audit conjectures (maximal auditable class; poly-time certified sup) are
resolved/owned by the OT-statistics literature (Sriperumbudur 2012 IPM=Rademacher;
Nietert et al. max-sliced; Niles-Weed–Rigollet spiked transport; kernel-max-sliced
NP-hard+SDR, arXiv:2405.15441). So they are NOT where a novel K-Bound theorem lives.

The unoccupied intersection (literature scan, 2026-07-18) is:
**minimax-optimal ternary decision (adapt/freeze/abstain) driven by an UNIDENTIFIABLE
drift parameter, with a false-commit constraint** — abstention forced by
non-identifiability (Le Cam indistinguishability), not by noise/risk-cost (Chow).
Closest neighbor: "A Decision-Theoretic View of Test-Time Training" (arXiv:2606.15569,
Bayesian/PAC-Bayes, different mechanism) — cite & differentiate.

## Setup and definitions (from Assumption ass:deploy / Lemma reduction)

Per deployment condition c: observable margin M(c) = E[s|D] − 1/2 ∈ [−1/2,1/2]
(estimable label-free from unlabeled deployment); label-free evidence Z(c);
unknown drift γ(c) = E[u|D] with |γ(c)| ≤ β under drift class C_β.
Benefit sign: sign Δ(c) = sign(M(c)+γ(c))  (Lemma reduction).
Oracle: knows sign Δ, plays a*(c) = ADAPT if Δ>0, FREEZE if Δ<0; regret 0.

A label-free decision rule is a (possibly randomized) map δ: (M,Z) ↦ {ADAPT,FREEZE,ABSTAIN}.
ABSTAIN defaults to the safe fixed policy (freeze) and incurs regret = |Δ| only when
adaptation would have helped; a WRONG strict commitment incurs the full benefit gap |Δ|.
False strict commitment event: {δ commits to the sign opposite to sign Δ}.
Constraint (unconditional false-adapt, matching FA_u): sup_{P∈C_β} Pr_P(false commit) ≤ α,
with α < 1/2.

Drift class C_β (rotation-closed, symmetric) is rich enough to realize the Aud-A kernel:
for fixed observables (μ_T, f0, fa, s) it contains laws achieving every
γ ∈ [−min(β,1/2+M), +min(β,1/2−M)] at matched Law(Z,M)  (Thm Aud-A construction).

## Theorem 1 (Forced-abstention bound — necessity) [restated per adversary review]

Fix a condition with observables (M,Z), |M| < β. For any label-free rule δ with
sup_{P∈C_β} Pr_P(false commit) ≤ α: on this input δ commits to ADAPT with prob ≤ α and to
FREEZE with prob ≤ α, hence commits at all with prob ≤ 2α; in particular δ abstains almost
surely iff α = 0.

Proof. By the Aud-A construction there exist P_+, P_- ∈ C_β with (i) identical observable law
Law_{P_+}(Z,M) = Law_{P_-}(Z,M), and (ii) drifts γ_+ ∈ (−M, β], γ_- ∈ [−β, −M) — both
intervals nonempty since −M ∈ (−β,β) — giving M+γ_+ > 0 (sign Δ=+1 under P_+) and
M+γ_- < 0 (sign Δ=−1 under P_-). As δ is a measurable function of (M,Z), its output law is
identical under P_+ and P_-. Let p = Pr(δ=ADAPT | M,Z), q = Pr(δ=FREEZE | M,Z). ADAPT is a
false commit under P_- (FREEZE optimal), so the constraint at world P_- gives p ≤ α; FREEZE
is a false commit under P_+, so the constraint at P_+ gives q ≤ α. Hence commit prob
= p+q ≤ 2α, and = 0 forced only when α = 0. □

Remark (honest, per adversary). This is Aud-A repackaged for the decision; the content used
downstream is only the ≥1/2 indistinguishability fact and the ≤2α / ≤α-per-action budget.
The earlier "abstain a.s." statement was FALSE for α>0 and is withdrawn. No "point-mass on
(M,Z)" step is needed (it is superfluous and would require an inadmissible μ_T that changes
M). Optimality — that the ≤2α commit slack buys no regret advantage — is Theorem 2.

## Theorem 2 (Minimax optimality of the frontier)

Let δ* be the frontier rule: ADAPT iff M>β, FREEZE iff M<−β, else ABSTAIN. Then
(a) δ* is feasible: sup_{C_β} Pr(false commit) = 0 ≤ α (population); with the finite-sample
    certificate (Thm certificate) its plug-in KGA has false-commit ≤ α.
(b) δ* is minimax-optimal for regret-to-oracle: for every feasible δ and every deployment
    mixture Q over conditions,  R(δ*, Q) ≤ R(δ, Q) + α·(sup|Δ|),
    where R is expected regret-to-oracle. In the population/α→0 limit δ* is exactly optimal:
    no feasible rule attains strictly smaller regret.

Hypotheses (made explicit per adversary): (H1) C_β contains a single world adversarial across
the whole abstain region simultaneously (per-condition drift may be chosen jointly worst-case);
(H2) the feasibility class shares the deployment condition-marginal Q; (H3) |Δ|=|M+γ|≤1/2+β.

Proof. (a) If M>β then for all γ∈[−β,β], M+γ>0, so sign Δ=+1 in every world of C_β ⇒ zero
false commit; symmetric for M<−β. (b) On {|M|>β}, R(δ*)=0≤R(δ). On {|M|≤β}, let a=a(M,Z) be
δ's ADAPT-probability; then R(δ*)−R(δ)=E_Q[a·Δ·1{|M|≤β}] (signed Δ). Bounding by the worst
world P_- that makes all abstain-region conditions harmful (H1): E_Q[a·Δ·1{|M|≤β}] ≤
sup|Δ|·E_Q[a·1{|M|≤β,Δ>0}] ≤ sup|Δ|·E_Q[a·1{|M|≤β}] ≤ α·sup|Δ|, the last step because
E_Q[a·1{|M|≤β}] = Pr_{P_-}(false adapt) ≤ α. Hence R(δ*) ≤ R(δ)+α·sup|Δ| with the CORRECT
constant α (not 2α). As α→0, R(δ*) ≤ R(δ) for every feasible δ (δ* optimal); a suboptimal δ
has R(δ)>R(δ*), so this is domination, not equality. □

## Theorem 3 (Beats-both characterization) — primary novel result [CORRECTED]

Regret model: R(policy)=E_Q[|Δ|·1{action≠oracle}]; ABSTAIN⇒FREEZE. With δ* the frontier
rule and using that on {M>β} Δ>0 and on {M<−β} Δ<0 for all γ∈[−β,β] (certified signs):
  R(always-freeze) − R(δ*) = E_Q[|Δ|·1{M>β}]                                   (F)
  R(always-adapt)  − R(δ*) = E_Q[|Δ|·1{M<−β}] + E_Q[|Δ|·1{|M|≤β,Δ<0}]
                              − E_Q[|Δ|·1{|M|≤β,Δ>0}]                          (A)
(δ* abstains→freezes on the CLOSED set |M|≤β; assume Q(|M|=β)=0 so the boundary is null.)

Theorem 3a (freeze side, CLEAN). δ* strictly beats always-freeze  iff  Q(M>β)>0
(detectable-helpful mass exists). Proof: δ* and always-freeze differ only on {M>β}, where δ*
adapts correctly and freeze is wrong; (F)≥0, strict iff Q(M>β)>0. □

Theorem 3b (adapt side, ASYMMETRIC). δ* strictly beats always-adapt iff (A) > 0 — a
retrospective identity, since (A) = R(AA)−R(δ*) by definition and its abstain-region terms
depend on the world-dependent, label-requiring signs of Δ (so (A)'s sign is NOT evaluable at
deployment without labels). Interpretation: since ABSTAIN⇒FREEZE, δ* is freeze-biased, so the
certified harmful-mass win E_Q[|Δ|1{M<−β}] must exceed the net helpful adaptation forgone in
the abstain region. Detectable harmful mass is necessary UNLESS the abstain region is itself
net-harmful (if Q(M<−β)=0 and the abstain region is net-helpful, (A)<0 and always-adapt wins;
if net-neutral, (A)=0, a tie). A clean SUFFICIENT condition for a strict win:
E_Q[|Δ|1{M<−β}] > E_Q[|Δ|1{|M|≤β,Δ>0}].

Corollary (beats BOTH). δ* strictly beats both iff Q(M>β)>0 AND (A)>0. Detectable-helpful
mass makes the freeze side free; detectable-harmful mass (net of the abstain-region helpful
imbalance) is what earns the adapt side. Symmetric statement holds for an abstain⇒adapt
variant (adapt-biased), which flips which side is free — the default choice of the safe
action is a modeling decision that determines which fixed policy is the easy one to beat.

Corollary (retrospective consistency, NOT a deployment prediction). Because (A)'s sign needs
labels, this EXPLAINS observed regret patterns retrospectively; it does not forecast them
label-free. Consistency with our data (validation v2, 25 seeds, results_v2.json):
- Detectable-harmful-mass regime (ImageNet-C harmful-SAR analogue): (A)=+0.078±0.0002, δ*
  beats both on 25/25 seeds — consistent with Table XIII (beats-both, 5/5 seeds).
- Helpful-dominated regime (Camelyon17 analogue): (A) is SMALL and SIGN-SENSITIVE —
  net-helpful abstain ⇒ (A)=−0.0055 (δ* slightly LOSES to always-adapt); tiny-harmful/neutral
  ⇒ (A)=+0.006 (δ* slightly beats). This candidate-dependent near-tie MATCHES the real
  Camelyon rows (SAR: KGA<adapt; EATA/Tent: KGA>adapt; all tiny) far better than a clean tie.
Honest reading: the asymmetry explains WHY helpful-dominated shifts hover at parity with
always-adapt (sign of (A) flips with fine structure), while detectable-harmful shifts
robustly beat both. The earlier "falsifiable prediction, not post hoc" wording is withdrawn.

## Witnesses (non-vacuity)

W1 (beats-both holds): two conditions, M_1=+0.3, M_2=−0.3, β=0.1, γ≡0. Both |M|>β, opposite
signs ⇒ δ* regret 0, always-adapt regret |Δ_2|>0, always-freeze regret |Δ_1|>0. Strict.
W2 (ties, helpful-dominated): all conditions M_i∈[0.2,0.4]>β ⇒ always-adapt optimal, δ* ties.
W3 (forced abstention binds): M=0, β=0.2, Aud-A pair γ=±0.2 ⇒ any commit wrong w.p.≥1/2.

## Honest scope / limitations

- Thm 1 lower bound REUSES the Aud-A kernel (credit internal). Novelty is Thms 2–3 framing:
  optimality (upper meets lower) and the beats-both characterization.
- The α>0 "gamble" slack (≤α·sup|Δ|) is real: a rule MAY commit on ≤α mass in the abstain
  region; δ* is optimal up to this slack, exact only as α→0. State this, don't hide it.
- Regret model assumes ABSTAIN defaults to freeze and wrong-commit costs |Δ|; alternative
  cost models (abstention cost, asymmetric losses) may shift the boundary — scope explicitly.
- This is a decision-theoretic optimality result over the label-free rule class; it does NOT
  claim the drift class C_β is itself verifiable (Aud-A: it is not). Conditional on β.

## Adversary review outcome (2026-07-18) + validation

Independent red-team verdict: Thm 2, Thm 3a, and the (F)/(A) algebra SOUND; three required
fixes applied here: (1) Thm 1 "abstain a.s." was false for α>0 — restated as ≤α-per-action /
≤2α-total, a.s. only at α=0; (2) the "falsifiable prediction" corollary downgraded to
retrospective consistency (sign of (A) needs labels); (3) the Camelyon "clean tie" was a
symmetric-construction artifact — replaced by multi-seed v2 showing (A) is small and
sign-sensitive on helpful-dominated shifts (validate_v2.py → results_v2.json). Minor: Thm 2
hypotheses H1–H3 made explicit, constant is α (not 2α), "optimality" not "equality"; Thm 3b
boundary |M|≤β with Q(|M|=β)=0; "necessary" softened.

Net contribution (honest): Thm 2 (minimax optimality of the |M|>β frontier, up to α·sup|Δ|)
and Thm 3a–b (the freeze-side-clean / adapt-side-asymmetric beats-both characterization that
explains the candidate-dependent Camelyon parity vs ImageNet-C beats-both). Lower-bound
machinery reuses Aud-A. This is a "solid contribution" decision-theoretic result specific to
the adapt/freeze/abstain problem — NOT covered by the OT-statistics literature — validated
numerically. It is not a resolution of the transport conjectures (those are owned by that
literature) and does not by itself change the paper's tier; it removes the two hand-wavy
conjectures' worst attack surface and adds a genuine optimality + explanation theorem.
