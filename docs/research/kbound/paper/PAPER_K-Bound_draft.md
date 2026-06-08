# When Is Label-Free Adaptation Knowable?
### Helpful, Harmful, and Unknowable Regimes Under Distribution Shift

**Theory:** K-Bound — the Knowability Boundary of Label-Free Adaptation.
**Algorithm:** KGA — Knowability-Guided Adaptation.
**Author:** Pratik Niroula (independent researcher). **Draft:** v0.1, working.

---

## ⚠️ Status note (read first — separates proven from planned)

This is an honest working draft, not a finished submission. Status of each part:

| Component | Status |
|---|---|
| Theorem 1 (non-identifiability) | **Proved** (Le Cam two-point argument), §5.1 |
| Theorem 2 (optimal gate under alignment) | **Proved** (Bayes decision; near-tautology), §5.2 |
| Theorem 3 (finite-sample certificate) | **Proved conditional on an estimator assumption** — the assumption is the real burden, §5.3 |
| Theorem 4 (positive identifiable regimes) | Covariate-shift case **proved (classical)**; general reliability-gating case is a **sketch / partially open**, §5.4 |
| Anomaly-routing experiments (safety, failure-recovery, mixed-regime) | **Run, real numbers**, §7.1–7.3 |
| Empirical non-identifiability witness | **Partial** (clean vs covert share *most* of Z, not all), §7.4 |
| "Beats both trivial baselines under catastrophic harm" | **NOT demonstrated in-house**; requires deep-network TTA (a 2-D logistic did not collapse). Required future work, §8 |

No claim of "solved," "universal," or "field-shaping" is made; those are not self-assignable.

---

## Abstract

Label-free test-time adaptation (TTA) is widely used to improve models under
distribution shift, yet the same unlabeled objective that helps under one shift
can degrade performance under another. Existing methods propose increasingly
robust *adaptation mechanisms*; we instead study a prior question: **can a system
decide, without labels, whether adaptation should happen at all?** We formalize
adaptation benefit as the conditional excess risk between an adapted candidate and
a frozen baseline, and separate test-time regimes into *knowably helpful*,
*knowably harmful*, and *unknowable*. We prove (Le Cam) that when two target
worlds induce the same observable evidence but opposite benefit, no label-free
rule can certify the correct decision in both — establishing the unknowable regime
as fundamental and motivating an explicit **abstain** action. Under an *observable
risk-alignment* assumption we give a computable adapt/freeze/abstain certificate
that controls the false-adapt rate at a chosen level. We instantiate the framework
(KGA) on a 123-task anomaly-routing benchmark and a controlled mixed-regime
study. KGA abstains precisely where the true benefit is near zero, recovers large
performance when detectors fail (beating an always-freeze policy by ~0.11 AUROC),
and never adopts a harmful fusion path (false-adapt rate 0.065). We are explicit
about scope: in domains where harmful adaptation is mild, an always-adapt baseline
is already strong, so the certificate's distinctive value is bounded by how often
*catastrophic, detectable* harm occurs — itself the quantity our theory
characterizes.

---

## 1. Introduction

Deployed models meet distribution shift, and test-time adaptation promises to use
unlabeled target data to recover lost accuracy. But adaptation is double-edged:
entropy minimization that sharpens a good boundary under mild covariate shift can
collapse to a degenerate solution under label shift; a reliability gate that
rescues a failed sensor can also dilute a strong detector on clean data. The field
has responded with sturdier mechanisms (sample selection, sharpness-aware updates,
resets). We ask the prior question instead:

> **Can observable test-time evidence separate helpful, harmful, and unknowable
> regimes of label-free adaptation, yielding a computable rule for when a system
> should adapt, freeze, or abstain?**

Our position is that adaptation should be treated as a **decision under
identifiability constraints**, not only an optimization problem. The contribution
is a theory of *when the decision is even knowable from unlabeled evidence*, a
certificate that respects that boundary, and an honest empirical map of where each
regime lives.

**Contributions.**
1. **Knowability formulation** (§4): adaptation benefit as conditional excess risk;
   helpful / harmful / unknowable regimes.
2. **Impossibility** (§5.1): identical observable evidence + opposite benefit ⇒ no
   label-free rule certifies both; abstention is the minimax-safe action.
3. **Certificate** (§5.3): an adapt/freeze/abstain rule with a false-adapt
   guarantee under an observable-alignment assumption (whose burden we state plainly).
4. **KGA + experiments** (§6–7): real results on anomaly routing, including a
   controlled mixed regime; safety, failure-recovery, and a partial
   non-identifiability witness.
5. **Honest scope** (§8): the "beats every trivial baseline" headline requires a
   catastrophic-harm domain (deep TTA), which we do not yet demonstrate.

---

## 2. Related work and positioning

This question is **not virgin territory**; several lines are adjacent, and the
contribution must be stated as a delta over them.

| Prior line | What it establishes | What it does **not** do (our gap) |
|---|---|---|
| **TTA methods** — Tent (Wang 2021), SHOT (Liang 2020), TTT (Sun 2020), EATA (Niu 2022), SAR (Niu 2023) | Adaptation helps under some shifts; robust variants reduce collapse/forgetting | No *pre-adaptation* certificate separating helpful/harmful/**unknowable**; no abstain-on-the-decision state |
| **Protected TTA / betting** (online entropy matching, 2024) | Guards against harmful adaptation with a guarantee | Guards one direction during adaptation; not a 3-way decision with an explicit unknowable region |
| **Label-free accuracy estimation** — ATC (Garg 2022), Agreement-on-the-Line (Baek 2022), **AETTA (Lee, CVPR 2024)** | Estimate target accuracy / detect TTA failure without labels | Estimating *performance* ≠ certifying the *sign of adaptation benefit* with a false-adapt bound |
| **Unsupervised risk estimation** — **Steinhardt & Liang (2016)**, Ben-David (2010), Lipton (2018) | Risk is estimable without labels **under conditional-independence / structural** assumptions; impossible in general | Estimates a *single* risk; we need the sign of a *difference* of risks (adapt vs freeze) and the decision-theoretic trichotomy |
| **Selective prediction** — Geifman & El-Yaniv | Abstain on uncertain *predictions* | Abstention over the *adaptation decision* is underdeveloped |

**The surviving gap.** No prior work packages (i) an explicit *unknowable* region
with an impossibility proof, (ii) a single adapt/freeze/**abstain** certificate
with a false-adapt guarantee, and (iii) the sign-of-benefit (not single-risk)
target. Theorem 1 is close to known non-identifiability results and we credit them;
the novelty is the *unification into a decision certificate* and the positive
side as a delta over Steinhardt-Liang (estimating the sign of a risk *difference*).

---

## 3. Problem setup

Source $P_S(X,Y)$ with labels available at train/validation time; target $P_T(X,Y)$
with **only** $P_T(X)$ observed at test time. Loss $\ell$, frozen model $f_0$,
adapted/gated candidate $f_a$. Target risk $R_T(f)=\mathbb{E}_{P_T}[\ell(f(X),Y)]$.

**Adaptation benefit** (oracle, needs labels): $\Delta = R_T(f_0)-R_T(f_a)$, so
$\Delta>0$ means *adapt helps*. (In the AUROC experiments we report benefit as
$B=\mathrm{AUROC}(f_a)-\mathrm{AUROC}(f_0)$, higher-is-better, same sign convention.)

**Observable evidence** $Z=\phi(B_{\text{batch}}, f_0, f_a)$: any label-free
statistic — entropy, confidence, score-distribution drift (KS), detector
disagreement, predicted-class balance, update norm, density-ratio diagnostics.

**Conditional benefit** $\Delta(z)=\mathbb{E}[\ell(f_0(X),Y)-\ell(f_a(X),Y)\mid Z=z]$.
The central question: *when is $\mathrm{sign}\,\Delta(z)$ recoverable from $Z$ alone?*

---

## 4. Definitions (the three regimes)

- **Observable risk alignment.** $Z$ is risk-aligned if it identifies or bounds
  $\mathrm{sign}\,\Delta(z)$ — it need not reveal the label, only whether
  adaptation helps.
- **Knowably helpful:** $Z$ certifies $\Delta(z)>0$ → **adapt**.
- **Knowably harmful:** $Z$ certifies $\Delta(z)<0$ → **freeze**.
- **Unknowable:** the same $Z$ is compatible with both signs → **abstain / refuse
  certification**.

---

## 5. Theory

### 5.1 Theorem 1 (Non-identifiability ⇒ the unknowable regime). **[Proved]**

*Let $P_T^1,P_T^2$ be target worlds with $\mathrm{Law}(Z\mid P_T^1)=\mathrm{Law}(Z\mid P_T^2)$
but $\Delta^1$ and $\Delta^2$ of opposite sign.
Then for any (possibly randomized) label-free rule $g(Z)\in\{\text{adapt,freeze,abstain}\}$,
the distribution of $g$ is identical under both worlds; hence $g$ cannot output the
benefit-maximizing committal action in both. The minimax-optimal committal-risk
rule abstains on the shared evidence.*

*Proof.* Le Cam two-point. Since $g$ is a measurable function of $Z$ (plus
independent randomness) and $\mathrm{Law}(Z)$ coincides across the two worlds,
$\mathrm{Law}(g)$ coincides too. If $g$ commits to "adapt" with probability $p$,
that same $p$ holds in the world where adapting is harmful, incurring regret
$\ge p\,|\Delta|$; symmetrically for "freeze." The committal regret is minimized
on this evidence by placing zero mass on both committal actions, i.e. abstaining. ∎

**Weight.** This result is *close to* known non-identifiability of target risk
without labels (Steinhardt-Liang 2016; Ben-David 2010); it is not the hard part.
Its role is to *justify the abstain action*, not to claim the impossibility as new.

### 5.2 Theorem 2 (Optimal gate under alignment). **[Proved, near-tautology]**

*If $\Delta(z)$ is $Z$-measurable, the Bayes-optimal label-free gate is
$g^\*(z)=\mathbf{1}[\Delta(z)>0]$ (adapt iff benefit positive).*
*Proof.* Pointwise minimization of expected loss. ∎ (Stated for completeness; the
content is in the *assumption* $\Delta(z)$ is $Z$-measurable, addressed by Thm 4.)

### 5.3 Theorem 3 (Finite-sample adapt/freeze/abstain certificate). **[Proved, conditional]**

*Suppose an estimator $\widehat\Delta(z)$ admits a radius $\varepsilon(z)$ with
$\Pr[\,|\widehat\Delta(z)-\Delta(z)|\le\varepsilon(z)\,]\ge 1-\alpha$. Define*
$$\text{ADAPT if }\widehat\Delta-\varepsilon>0,\quad \text{FREEZE if }\widehat\Delta+\varepsilon<0,\quad \text{ABSTAIN otherwise.}$$
*Then $\Pr[\text{ADAPT}\wedge\Delta\le0]\le\alpha$ and $\Pr[\text{FREEZE}\wedge\Delta\ge0]\le\alpha$ (false-adapt and false-freeze controlled).*
*Proof.* If ADAPT fires then $\Delta\ge\widehat\Delta-\varepsilon>0$ on the
$1-\alpha$ event; the complement has mass $\le\alpha$. Symmetric for FREEZE. ∎

**The real burden (stated plainly).** The theorem *assumes* a valid label-free
$(\widehat\Delta,\varepsilon)$. Whether such an estimator exists is the whole
question; Theorem 3 does **not** discharge it. §7 uses a cross-task conformal
estimator whose validity rests on task exchangeability — an assumption we flag, not
prove in general.

### 5.4 Theorem 4 (Positive identifiable regimes). **[Covariate case proved; general open]**

*Covariate shift.* If $P_S(Y\mid X)=P_T(Y\mid X)$ and the density ratio
$r(x)=P_T(x)/P_S(x)$ is estimable and bounded with finite second moment, then
$R_T(f)=\mathbb{E}_{P_S}[r(X)\ell(f(X),Y)]$ is estimable from labeled source +
unlabeled target, hence $\mathrm{sign}\,\Delta$ is identifiable (classical;
Shimodaira 1999). 

*Delta over Steinhardt-Liang.* They estimate a *single* risk under
conditional-independence structure. We need the sign of a *difference*
$R_T(f_0)-R_T(f_a)$; this is *easier* in one respect (constants cancel) and the
open work is to give the structural condition on $(f_0,f_a,Z)$ under which the
difference's sign is identifiable even when each risk separately is not. The
reliability-gating instantiation (observable detector-failure ⇒ routing benefit)
is the worked candidate; a general proof is **open**.

---

## 6. Method: KGA (Knowability-Guided Adaptation)

A decision *wrapper* — it does not require a new adaptation mechanism.

```
Input: frozen f0, candidate adaptation A, unlabeled batch, evidence extractor phi,
       cross-task benefit estimator Delta_hat, radius eps, level alpha
1. Z       <- phi(batch, f0, A)            # label-free evidence
2. d_hat   <- Delta_hat(Z)                 # estimated benefit
3. eps     <- conformal_radius(level=alpha)
4. if d_hat - eps > 0:  return ADAPT
   elif d_hat + eps < 0: return FREEZE
   else:                 return ABSTAIN    # refuse certification
```

In our experiments $\widehat\Delta$ is a gradient-boosted regressor trained
leave-one-task-out on $(Z,\text{true }B)$ pairs; $\varepsilon$ is the
$(1-\alpha)$ quantile of leave-one-out residuals (split-conformal style, $\alpha=0.1$).

---

## 7. Experiments (real numbers; every value from a run in `scripts/`)

Data: the project's 123-task anomaly score archive (ADBench tabular, image-OOD,
text, cyber, fraud) — each task has 6 detector scores on labeled validation + test.
Frozen $f_0$ = best-validation detector ("auto-select"). Controlled-synthetic
corruptions validate the *mechanism*, not real sensor failure, and are labeled as such.

### 7.1 Safety on the clean suite — `knowability_results.json`
Candidate $f_a$ = rank-normalized logistic stack. KGA's abstentions land where the
**true** benefit is near zero (mean $|B|=0.021$ on abstained tasks vs $0.132$ on
acted tasks); adapt precision $0.90$. *But* this suite is helpful-dominated, so an
always-adapt policy (mean AUROC $0.777$) edges the safe policy ($0.766$); coverage
is $17\%$. Honest reading: when adaptation almost always helps, abstention mostly
costs coverage. (Fig. `fig_certificate.png`, `fig_false_adapt.png`.)

### 7.2 Harmful regime — `kbound_harmful_results.json`
Candidate $f_a$ = reliability-fusion (`elara_fuse`), which **hurts on 80% of tasks**.
KGA refuses it almost everywhere and matches the safe baseline (mean AUROC $0.748$)
while always-adapt drops to $0.728$; regret-vs-oracle falls from $0.0214$ to
$0.0019$ (~11× lower). Here it *ties* always-freeze — correct, since freezing is
right 80% of the time. (Fig. `fig_kbound_harmful.png`.)

### 7.3 Controlled mixed regime — `mixed_regime_results.json` (369 instances)
Three conditions per task: **clean** (adapt usually mildly harmful), **detectable
failure** (best detector corrupted with a distribution shift → KS drift observable),
**covert failure** (best detector permuted → signal destroyed but marginal, hence
$Z$, ~unchanged). Decisions by true regime:

| condition | true mean $B$ | ADAPT | FREEZE | ABSTAIN |
|---|---:|---:|---:|---:|
| clean | −0.041 | 8 | 5 | 110 |
| detectable failure | +0.193 | 71 | 3 | 49 |
| covert failure | +0.181 | 59 | 3 | 61 |

Adapt precision $0.935$, false-adapt rate $0.065$. Policy AUROC: always-adapt
$0.697$, always-freeze $0.586$, **KGA $0.690$**, oracle $0.719$. KGA **beats
always-freeze by 0.10** (failure recovery) and ties always-adapt — because in this
domain harmful adaptation is mild. (Figs. `fig_mixed_decisions.png`,
`fig_mixed_policies.png`.)

### 7.4 Empirical non-identifiability witness (partial)
Clean instances have mean $B=-0.041$; covert-failure instances have mean
$B=+0.181$ — **opposite sign**. Their evidence is *closer* than to detectable
failure (mean standardized $Z$-distance clean→covert $1.35$ vs clean→detectable
$3.10$), so $Z$ only *partially* separates them, weakening committal decisions on
that boundary — consistent with Theorem 1. It is a **partial** witness: permutation
preserves marginals but slightly perturbs the correlation structure, so $Z$ is not
perfectly identical. A clean witness needs a constructed pair with provably equal
$Z$-law.

### 7.5 Corroborating evidence already in the repository
- **MVTec-3D multimodal failure** (`experiments/elara_u/multimodal_reliability_results_mvtec3d.json`):
  clean → gating does no harm ($0.882=0.882$); under modality failure → gating
  recovers **+0.21 AUROC**, all hypotheses' CIs exclude zero (`reliability_validated: True`).
  This is the genuine mixed regime where discrimination pays.
- **Shift-stress crossover** (`shift_stress_ablation.json`): reliability routing is
  n.s./negative at severity 0 ($-0.0039$) and **significant only at severity 3**
  ($+0.039$, CI excludes 0) — the predicted knowability crossover.

---

## 8. Limitations & honest scope

1. **The "beats every trivial baseline" headline is not yet shown in-house.** It
   needs a domain with *catastrophic, detectable* harmful adaptation. A 2-D logistic
   TTA we built did **not** collapse (Tent helped mildly everywhere), so we do not
   claim it. Deep-network TTA (CIFAR-10-C / ImageNet-C with Tent/EATA/SAR), where
   collapse is documented and predicted-class balance is a label-free red flag, is
   **required future work**.
2. **Synthetic corruptions** validate the mechanism, not real sensor failure.
3. **Conformal validity** assumes task exchangeability; out-of-family transfer is untested here.
4. **Theorem 4 general case is open**; only the covariate-shift instance is proved.
5. **Non-identifiability witness is partial** (§7.4).
6. KGA's value scales with how often catastrophic, detectable harm occurs — which is
   precisely the quantity the theory characterizes. In benign domains, always-adapt
   is a strong baseline and the certificate's gain is mainly *safety insurance*.

---

## 9. Conclusion

We reframed label-free adaptation as a decision constrained by *identifiability*:
adapt when benefit is certifiable, freeze when harm is certifiable, abstain when the
unlabeled evidence cannot tell. We proved the unknowable regime is fundamental,
gave a false-adapt-controlled certificate under an alignment assumption we state
plainly, and showed on real anomaly-routing data that the certificate abstains where
benefit vanishes and recovers large performance when detectors fail. The honest
frontier is a catastrophic-harm benchmark that would let the trichotomy beat every
trivial policy — the experiment that would turn this from a solid, scoped
contribution into the broader claim.

---

## References (to be completed/verified before any submission)
Wang et al., *Tent*, ICLR 2021 · Liang et al., *SHOT*, ICML 2020 · Sun et al.,
*TTT*, ICML 2020 · Niu et al., *EATA*, ICML 2022 · Niu et al., *SAR*, ICLR 2023 ·
Lee et al., *AETTA*, CVPR 2024 · (Protected TTA, online entropy matching) 2024 ·
Steinhardt & Liang, *Unsupervised Risk Estimation under Conditional Independence*,
2016 · Ben-David et al., *A theory of learning from different domains*, ML 2010 ·
Lipton et al., *Detecting and correcting label shift (BBSE)*, ICML 2018 · Garg et
al., *ATC*, 2022 · Baek et al., *Agreement-on-the-Line*, 2022 · Geifman &
El-Yaniv, *SelectiveNet*, 2019 · Shimodaira, *Covariate shift / importance
weighting*, 1999.
*(Citation details to be verified against sources before submission.)*
