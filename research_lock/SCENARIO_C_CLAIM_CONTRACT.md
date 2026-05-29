# SCENARIO C CLAIM CONTRACT (v1, frozen)

This is the signed-off definition of what evidence counts as success **before**
any final test set is evaluated. It is the binding contract for the ELARA
Scenario C program.

ELARA = *Reliability-Aware Multimodal Anomaly Fusion Under Degraded Evidence.*

## Central claim (DECISION REQUIRED — slot)

The one-sentence final claim is **not yet ratified**. Until the user ratifies
it, the only admissible claim is the current-manuscript claim (see
`BASELINE_STATE_v1.md`). Candidate target (NOT yet supported):

> *ELARA is a validated reliability-aware multimodal anomaly-fusion framework
> that improves over strong frozen baselines under clean, degraded, and
> transferred conditions across naturally co-observed domains, with formal
> switching conditions and deployment-style monitoring.*

This claim may only be used once the corresponding pillars below pass.

## The six claim pillars (Definition of Done)

| Pillar | Required evidence | Status (v1, grounded in current artifacts) |
| --- | --- | --- |
| **P1 Mechanism validity** | Base RGA activates under degradation, quiet on clean evidence. | PARTIAL — coherent-collapse works (B1 +0.0507); sensitive RGA-v2 gates G1/G2/G3 fail clean false-fire (FFR 1.000); domain-composition shift unresolved. |
| **P2 Strong-baseline superiority** | Final ELARA beats the strongest frozen non-ELARA baseline (not only static attention). | NOT ESTABLISHED — current evidence is vs fixed static attention only. |
| **P3 True multimodal generalization** | Positive results on multiple naturally co-observed datasets. | PARTIAL — MVTec 3D-AD is naturally paired; LOCO/VisA are derived-view proxies; insufficient overall. |
| **P4 Held-out transfer** | Confirmed improvement on unseen paired data / unseen degradation. | NOT CONFIRMED — Eyecandies transfer not confirmed (D-EYE-1 Δ −0.0010 p=0.3632; D-EYE-2 Δ −0.0109 p=0.4468). |
| **P5 Theory & certificate** | Formal conditions for when switching helps / fails / abstains. | PARTIAL — finite-sample retrospective certificate exists (max_attack certified, zero_attack not); population-level + calibration-transfer theory open. |
| **P6 Deployment auditability** | Temporal eval, calibration monitoring, abstention/fallback, license review, reproducible raw predictions. | PARTIAL — `PredictionArchive` + monitoring concept exist; full prospective evidence incomplete. |

A claim pillar flips to PASS only via a `NEW CONFIRMATORY` result under this
contract.

## Readiness tiers

**Tier 1 — Strong paper readiness** (bounded publishable ELARA paper):
- [ ] Current evidence frozen and auditable (this lock).
- [ ] Base RGA and RGA+ claims separated.
- [ ] Strongest comparator frozen before final test.
- [ ] Positive primary degraded-evidence result on real paired data.
- [ ] Clean false-fire controlled.
- [ ] Raw predictions + paired statistics archived.
- [ ] Family-D statistical ambiguity resolved (sealed or reclassified).
- [ ] Paper shortened and claims bounded.

**Tier 2 — Generalization readiness** (high-impact candidate):
- [ ] Confirmed held-out transfer to a new naturally paired dataset.
- [ ] Positive partial-domain (k-of-D) failure results.
- [ ] Improved calibration-transfer method.
- [ ] Positive vs strongest non-ELARA baseline.
- [ ] Strong upstream domain experts.
- [ ] Unknown-degradation test passed OR safe abstention shown.

**Tier 3 — Scenario C flagship readiness**:
- [ ] Success across ≥2 distinct naturally co-observed domain families.
- [ ] Same core ELARA interface, no architecture redesign.
- [ ] Theory explains switching benefit, false-fire cost, partial-failure boundary, transfer assumptions.
- [ ] Prospective / temporal monitoring validation.
- [ ] Safe fallback/abstention under uncertified shift or attack.
- [ ] Independent reproduction / external confirmation.
- [ ] All confirmatory endpoints frozen and statistically passed.

## Pass criterion for "strong-baseline superiority" (P2)

ELARA must win on the **pre-registered primary metric** against the strongest
**validation-selected, frozen** comparator, with: (a) bootstrap CI excluding
zero, (b) multiplicity-corrected significance, (c) practically meaningful effect
size, and (d) no worse major failure in calibration or false-fire behavior.

## Forbidden language (at every stage)

`universal anomaly detection`, `universal SOTA`, `always robust`, `works in all
domains`, `deployment ready everywhere`. (See `C_NO_UNIVERSAL` claim row.)

## Decisions that block confirmatory work

| # | Decision | Default until ratified |
| --- | --- | --- |
| D1 | Eyecandies: Policy A (sealed FAILED external test) vs Policy B (reclassify to development + acquire new untouched final transfer dataset). | Policy A (sealed FAILED). |
| D2 | Central claim wording (above). | Use current-manuscript claim only. |
| D3 | New untouched RGB+depth transfer dataset (T2). | None selected. |
| D4 | Non-vision naturally co-observed domain (M3). | None selected. |
| D5 | Strongest-baseline family freeze (Phase 5). | Not frozen. |

Ratify D1–D5 in a `DECISIONS_v1.md` (append-only) before evaluating any final
test set.
