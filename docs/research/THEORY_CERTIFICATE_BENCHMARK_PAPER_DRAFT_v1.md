# When Does Reliability-Gated Multimodal Fusion Help? An Operating-Boundary Theory with a Computable Certificate and a Pre-Registered Benchmark

Draft v1 — 2026-06-02. Theory-forward reframe of the ELARA program. The
contribution is the *characterization of an entire method class*, not a single
method's gains. ELARA is the case study that confirms the theory.

---

## Abstract

Reliability-gated multimodal fusion — down-weighting a modality when it looks
unreliable — is widely assumed to improve robustness, yet it is rarely stated
*when* it can and cannot help. We give a complete operating-boundary theory for
score-level reliability-gated fusion against the parameter-free
confidence-weighted mean (CW). We prove three results. (i) **Stress lower bound
(T1/T3):** under modality corruption, quality-blind fusion is dominated, and a
reliability-aware rule has a positive, characterized advantage with a closed-form
crossover in the number of degraded modalities. (ii) **Clean-transfer
impossibility (T9):** on near-separable clean data the recoverable advantage of
*any* fusion rule over CW is bounded by the optimality gap of CW,
`Delta* <= eps_subopt = A* - A(CW)`, which vanishes when CW reaches the
Neyman--Pearson ceiling; we make this a *computable certificate* by estimating
`A*` with an unconstrained cross-fitted oracle. (iii) **An admissibility
certificate** that turns the impossibility into a constructive, validation-only
test predicting, before deployment, whether reliability-gated fusion can beat CW
on a given dataset. We pair the theory with a **degenerate-channel guard** (a
prerequisite that prevents a broken/inverted modality from masquerading as
recoverable headroom) and **CCRT**, a pre-registered benchmark for the
complementary-reliability regime where a clean-transfer win is theoretically
possible. On real industrial RGB+3D anomaly data we confirm all three regimes:
gated fusion ties CW on clean one-class data (Δ = 0.000) exactly as T9 predicts,
beats CW by +0.18 to +0.25 AUROC under degradation as T1/T3 predict, and the
admissibility certificate selects exactly the complementary-regime categories.
The practical message is a decision rule: deploy reliability gating in the
stress/complementary regime, and *not* on clean near-ceiling data, where we prove
it cannot help.

---

## 1. The question and why it matters

The literature treats reliability-aware fusion as a robustness improvement, but
provides no boundary: practitioners cannot tell in advance whether it will help
their data or waste effort. We answer the boundary question for the canonical
score-level setting (each modality emits an anomaly score; a fusion rule combines
them) with the confidence-weighted mean as the reference.

**Contributions.**
1. A stress-regime lower bound (T1/T3) with a closed-form degradation crossover.
2. A clean-transfer impossibility theorem (T9) with a computable certificate.
3. An *admissibility certificate*: a validation-only predictor of whether fusion
   can beat CW on a dataset (the impossibility run as a selection rule).
4. A degenerate-channel guard that is a necessary preprocessing step for any
   honest fusion comparison.
5. CCRT, a pre-registered benchmark for the complementary-reliability regime.
6. Empirical confirmation of all three regimes on real RGB+3D industrial data.

## 2. The operating-boundary theory

Let `A(g)` be the AUROC of fusion rule `g`, `A(CW)` the confidence-weighted mean,
`A* = sup_g A(g)` the Neyman--Pearson ceiling. Decompose the headroom-to-perfect:

```
1 - A(CW) = eps_Bayes + eps_subopt,   eps_subopt = A* - A(CW) >= 0.
```

**T9 (clean-transfer impossibility).** For every fusion class `G`,
`Delta*(G) = sup_{g in G} A(g) - A(CW) <= eps_subopt`. On near-separable clean
data `eps_subopt -> 0` (CW reaches the ceiling), so no rule — reliability gate
included — can be certified above CW. The recoverable advantage is the optimality
gap of CW, not the gap to perfect AUROC. *Certificate:* estimate `A*` by an
unconstrained cross-fitted oracle (gradient boosting on the joint scores with
labels — strictly more information than any gate); if `A_oracle - A(CW) < MDE`,
the gate provably cannot win. A soundness guard rejects an underpowered oracle so
the impossibility cannot be manufactured.

**T1/T3 (stress lower bound + crossover).** Under modality corruption,
quality-blind fusion is dominated; the mean-gate miss probability is
`P(miss) = Phi((mu_bar - tau)/sigma_bar)` with deterministic crossover
`k* = D(mu_h - tau)/(mu_h - mu_c)` in the number of degraded modalities `D`.
Reliability gating recovers a positive, characterized advantage in this regime.

**The envelope.** T9 caps the clean end at zero; T1 lifts the corrupted end above
zero; T3 locates the crossover. The gate's value lives strictly in the band T1/T3
open and T9 closes.

## 3. The admissibility certificate (impossibility as a tool)

The same `A_oracle` vs `A(CW)` comparison, run as a *selection* rule, predicts
where fusion can help: a dataset/category is **admissible** iff, after the
degenerate-channel guard, `A_oracle - A(CW) > MDE` (oracle credibly recovers
headroom). This is computed on development data only, with no test labels, and is
a drop-in pre-deployment test for any score-level fusion study.

## 4. The degenerate-channel guard (a necessary prerequisite)

A modality whose validation scores are sign-inverted (AUROC ~ 0) or saturated
(near-constant) is a *detector artifact*, not signal: CW (sharpness-weighted)
trusts it and collapses, so an oracle "wins" merely by routing around it — false
headroom. The guard rejects such channels using validation statistics only. It is
both a robustness improvement (it raises CW itself) and a prerequisite for an
honest admissibility verdict.

## 5. CCRT: a pre-registered complementary-regime benchmark

CCRT admits only categories where T9 permits a win (complementary modalities, CW
suboptimal), with the inclusion rule frozen before any official test and the
held-out split sealed. It does not redefine or relax the strict clean-transfer
gate, which remains closed by proof; it measures fusion where a win is real.

## 6. Empirical confirmation (real RGB+3D industrial anomaly data)

- **Clean (T9):** MVTec 3D-AD one-class, category-averaged — gated-CW = CW = RGA
  = 0.791, Δ = 0.000, not significant. Gating ties CW on clean data, as proven.
  Oracle ties CW on 3D-ADAM (eps_subopt ~ 0) and MulSen, confirming the
  certificate.
- **Stress (T1/T3):** under increasing corruption, gated-CW beats CW by +0.18 to
  +0.25 AUROC, significant — the predicted lower bound.
- **Guard / boundary:** guarding both sides shrinks an apparent +0.045 fusion
  margin to +0.005, surviving only where >=2 channels are genuinely reliable;
  most apparent "gating gains" were degenerate-channel artifacts.
- **Admissibility:** on a real multimodal RGB+photometric+point-cloud set, the
  certificate admits exactly the complementary-regime categories (3/11) and
  correctly excludes redundant/near-ceiling and artifact categories.
- **Detector study:** improving the point-cloud detector raises raw AUROC but
  does *not* widen the gating margin (it narrows it, +0.017 -> +0.011) — better
  detectors move data toward the T9 regime, a prediction of the theory.
- **Cross-modal feature-level fusion** on MVTec 3D-AD one-class (8 categories):
  mean image-AUROC **0.837** (rgb-only 0.795, depth-only 0.817, score-fusion
  gated-CW 0.791). Feature-level joint patch interaction improves over score-level
  fusion by **+0.046**, confirming it captures cross-modal structure score fusion
  cannot — but on a generic ImageNet backbone it remains **below specialized-3D
  SOTA** (BTF 0.865, PatchCore-3D 0.901, AST 0.937, M3DM 0.945). This isolates the
  gain attributable to *fusion* from that attributable to the *detector*, and
  positions the method honestly: the contribution is the theory and tools, not a
  SOTA detector.

## 7. Honest scope

We characterize *score-level* reliability-gated fusion against CW. We do **not**
claim universal superiority, clean-transfer dominance, or leaderboard SOTA; T9
proves the first is impossible on clean near-ceiling data, and our one-class
numbers position the score-fusion method below cross-modal-feature SOTA. The
contribution is the boundary and the tools, confirmed on real data.

## 8. Why this is the right framing

A negative result with a proof and a reusable certificate is a contribution in
its own right: it tells practitioners where reliability gating cannot help, where
it provably can, and gives a test to decide for a new dataset. The method
(ELARA) is the case study; the theory, certificate, guard, and benchmark are the
transferable contributions.

---

## Mapping to artifacts (for the camera-ready)

| Claim | Artifact |
|---|---|
| T9 + certificate | `src/elara/theory/t9_clean_transfer_ceiling.py`, `experiments/fusion/t9_clean_transfer_ceiling_validation.json` |
| T1/T3 | `src/elara/theory/t1_impossibility.py`, `src/elara/family_b/corruption.py`, `experiments/fusion/craf_real_k_domain_results.json` |
| Operating boundary synthesis | `docs/research/phase3/OPERATING_BOUNDARY_UNIFIED_T1_T3_T9_2026_06_02.md` |
| Admissibility certificate | `src/scripts/scenario_c/clean_complementary_admissibility.py`, `experiments/fusion/clean_complementary_admissibility.json` |
| Degenerate-channel guard | `src/elara/evaluation/degenerate_channel_guard.py`, `experiments/fusion/guarded_channel_dev_analysis.json` |
| CCRT benchmark | `research_lock/CLEAN_COMPLEMENTARY_TRANSFER_PROTOCOL_v1.yaml` |
| One-class clean / stress | `experiments/fusion/mvtec3d_one_class_degradation_results.json`, `docs/research/tables/mvtec3d_sota_demarcation.tex` |
| Detector study | `experiments/fusion/lever1c_dual_xyz_dev_rescoring.json` |
| Cross-modal (pending) | `experiments/fusion/cross_modal_mvtec3d_one_class.json` |
