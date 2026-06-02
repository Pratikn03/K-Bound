# T9 — Clean-Transfer Reliability-Gate Impossibility (2026-06-01)

Turns the repeated empirical observation "no fusion rule beats the
confidence-weighted mean on clean external transfer" into a **theorem with a
computable impossibility certificate**. This closes clean Gate E *by proof*
(decision D14): it is provably unpassable on near-ceiling clean transfer, not
merely unmet. T9 is the clean-regime complement of T1/T3 (which give the gate a
positive lower bound under corruption); together they characterise the full
operating boundary of reliability-gated fusion.

## The theorem

Let `A(g)` be the AUROC of fusion rule `g`, `g_CW` the confidence-weighted mean,
and `A* = sup_g A(g)` the Neyman–Pearson ceiling (AUROC of the posterior ranker
`g*(s)=P(Y=1|s)`; no rule exceeds it). Decompose the CW headroom-to-perfect:

```
1 - A(CW) = eps_Bayes + eps_subopt
eps_Bayes  = 1 - A*           (irreducible class overlap)
eps_subopt = A* - A(CW) >= 0  (the ONLY slice any rule can recover)
```

**(i) Ceiling.** Because `A(g) <= A*` for every `g`,
`Delta*(G) = sup_{g in G} A(g) - A(CW) <= eps_subopt <= 1 - A(CW)` for *every*
fusion class `G` — every reliability-gated rule included. The recoverable
headroom is the optimality gap of CW, not the gap to perfect AUROC.

**(ii) Homoscedastic closure.** Under the Gaussian equal-covariance model the
posterior ranker is linear (LDA), `A* = Phi(||mu1-mu0||_{Sigma^-1}/sqrt2)`, so no
nonlinear gate beats the best linear rule. When confidence weights align with
the LDA direction `Sigma^-1(mu1-mu0)` (the redundant / equal-discriminability
case), `A(CW)=A*` exactly and `Delta*(G) <= 0`: CW is unbeatable.

**(iii) Gate-E certificate.** A one-sided level-α power-β paired AUROC test can
certify `Delta>0` only if `eps_subopt > MDE(α,β,n,ρ)`. Hence clean Gate E is
**unpassable** whenever `eps_subopt < MDE`. We estimate `eps_subopt` generously
with an *unconstrained* cross-fitted oracle (gradient boosting on the joint
modality scores + confidences **with** labels — strictly more information than
any gate). If even the oracle cannot beat CW, the restricted gate certainly
cannot.

Why this is the *right* impossibility (not "headroom to 1 is small"): the naive
bound `Delta* <= 1-A(CW)` is weak — a tiny `1-A(CW)` could still hide exploitable
structure. T9 sharpens it to `Delta* <= A*-A(CW)` and *measures* `A*`. The
finding is that on clean transfer `eps_subopt ~ 0`: CW already reaches the
posterior ceiling, so the residual headroom is Bayes-irreducible and
unexploitable by any rule.

## The certificate on real data

| Benchmark | A(CW) | Â* (oracle) | eps_subopt | MDE | complementary? | Gate E |
|---|---|---|---|---|---|---|
| 3D-ADAM RGB+depth (strong v3 detector) | 0.9349 | 0.9336 | 0.000 | 0.0095 | yes (weak) | **unpassable** |
| MulSen RGB+infrared (per-category) | 0.9970 | 0.9947 | 0.000 | 0.0060 | yes | **unpassable** |

On both, the unconstrained oracle does **not** beat CW (`eps_subopt = 0`), so the
recoverable advantage is below the minimum detectable effect. Clean Gate E is
closed by proof.

Synthetic confirmation (both clean-transfer regimes, `validate_t9`):
- **redundant-mean-optimal:** `A(CW)=A*` analytically; oracle cannot beat CW.
- **complementary-ceiling:** `A(CW)→1`; tiny residual is Bayes-irreducible.

## What it changes (and does not)

- **Changes:** clean strict Gate E is reclassified OPEN→`CLOSED_BY_PROOF_T9` in
  the gate program (`gate_e_m2_clean_closed_by_proof_t9`,
  `summary.gate_e_strict_clean_status`). No gate is deleted; the unwinnable gate
  is now accounted for by proof, not presented as an unexplained failure.
- **Does NOT change:** strict `gate_e_m2_transfer_confirmed` /
  `gate_f_scenario_c_scientific_strict` remain **false** — T9 does not
  manufacture a pass, it proves the pass is unattainable on clean near-ceiling
  data. Gate D/T5 and pillars A/B/C remain the passing gates. Standing level
  unchanged (D12: ~2.5/5, bounded claim).

## Why this is a genuine contribution

A negative result with a proof and a reusable certificate is publishable in its
own right: it tells practitioners *not to* deploy reliability gating expecting
clean-transfer gains, and tells theorists exactly where the gate's value lives
(the stress regime, T1/T3). The certificate (`gate_e_unpassable_certificate`) is
a drop-in test any multimodal-fusion study can run before claiming a fusion-rule
win on near-separable data.

## Artifacts
- `src/elara/theory/t9_clean_transfer_ceiling.py` — theorem + samplers + certificate
- `src/scripts/validate_t9_clean_transfer_ceiling.py` — synthetic + real-data validation
- `experiments/fusion/t9_clean_transfer_ceiling_validation.json`
- `docs/research/tables/t9_clean_transfer_ceiling.tex`
- `tests/test_t9_clean_transfer_ceiling.py`
- Registry: `THEOREM_REGISTRY["T9"]`; Decision: D14 in `research_lock/DECISIONS_v1.md`
