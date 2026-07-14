<!--
PAPER_SECTION_deployment_draft.md  —  Markdown mirror of
PAPER_SECTION_deployment_draft.tex (same content, same placeholders).
-->

> **DRAFT — fold into `kbound_short.tex` only AFTER the real camera run.**
> Result tables (R2 / R3 / S1 / S3 / S4 / S5) hold placeholder "—" cells until
> populated from real held-out logs. **R1** and **S2** reflect the locked design
> and the code's verified guarantees and are complete now.
>
> Additive / edge-only. Do **not** modify `kbound.tex` or `kbound_short.tex`.

# Deployment: a certified camera adapter on a phone

We stress-test the certificate as a drop-in safety layer for test-time
adaptation (TTA) on a real phone-camera inspection task. A frozen
MobileNetV3-Small, *f₀*, produces the **official** per-window prediction that a
downstream consumer acts on. Each 32-frame window also proposes, in shadow, an
episodic TENT candidate *fₐ* that updates only the BatchNorm-affine parameters of
a deep copy of *f₀* for a single Adam step, so *f₀* is never mutated. From the
frozen and adapted softmax matrices we compute a 14-dimensional **label-free**
evidence vector, predict the candidate's benefit *B̂* with a calibration-fit
estimator, and attach a split-conformal radius *ε* (α = 0.10) to form the
interval [*B̂* − *ε*, *B̂* + *ε*]. The reused K-Bound rule then certifies
**adapt** (lower bound > 0), **freeze** (upper bound < 0), or **abstain**
(interval spans 0). Ground-truth labels never touch the online path; the
per-window benefit *Δ* (adapted minus frozen balanced accuracy) is revealed only
offline to score regret and false-adapt rates. We compare six policies on the
identical held-out stream and report decision-level metrics: regret,
unconditional and conditional false-adapt (FA_u, FA_c), adapt rate, and
abstention/coverage.

## Figure 1 — Pipeline (placeholder)

```
  ┌──────────────┐   ┌───────────────┐   ┌────────────────┐   ┌──────────────────┐   ┌───────────┐
  │ Phone camera │ → │ Frozen f0      │ → │ Episodic TENT  │ → │ KGA certificate  │ → │ adapt     │
  │ 32-frame     │   │ MobileNetV3-S  │   │ candidate fa   │   │ decide(B̂, ε)     │   │ freeze    │
  │ window       │   │ OFFICIAL       │   │ (shadow)       │   │                  │   │ abstain   │
  └──────────────┘   └───────────────┘   └────────────────┘   └────────▲─────────┘   └───────────┘
                                                                        │
                                     14 label-free features → B̂ (calib-fit) + split-conformal ε (α=0.10)
```

**Caption.** Certified TTA on a phone camera. Each 32-frame window is scored by
the frozen model *f₀* (the official output) and, in shadow, by an episodic TENT
candidate *fₐ* that touches only BatchNorm-affine parameters of a deep copy, so
*f₀* is never mutated. A 14-D label-free evidence vector yields a benefit
estimate *B̂* (calibration-fit) and a split-conformal radius *ε* (α = 0.10); the
reused K-Bound rule certifies *adapt* (*B̂* − *ε* > 0), *freeze* (*B̂* + *ε* < 0),
or *abstain* (interval spans 0). No ground-truth labels touch the online path.

## Table R1 — Protocol and split integrity (locked design; complete now)

The held-out camera run is fixed to these settings before any window is scored.
Values reflect the locked real-camera protocol and the code.

| Field | Setting |
|---|---|
| Task | 4-class package/label inspection: {ok, defect, empty, misaligned} |
| Camera / input | Phone (Continuity Camera / Camo / EpocCam), real objects, 224×224 frames |
| Frozen base model f₀ (official) | MobileNetV3-Small + 4-class head, BN-recalibrated |
| Candidate fₐ | Episodic TENT, BN-affine params only, 1 Adam step / window |
| Decision window | 32 frames |
| Label-free evidence | 14 features = 11 paper disagreement features + 3 edge (`mean_js_div`, `pred_flip_rate`, `post_top2_margin`) |
| Calibration-fit session | Benefit estimator training — distinct day |
| Calibration-conformal session | Split-conformal residuals → ε — distinct day |
| Held-out session | Primary evaluation replay — distinct day |
| Replication session | Repeat held-out — distinct day |
| Physical shift families | Lighting, shadow, blur, background, viewpoint, batch-composition |
| Policies compared | freeze, adapt, confidence gate, entropy gate, KGA-no-radius, KGA-certificate |
| Deployment labels used live | No |
| Miscoverage level α | 0.10 |
| Official live model | Frozen f₀ fallback unless candidate certified (adapt) |

## Table R2 — Primary held-out result (template; populate from real logs)

Six policies on the identical held-out window stream. Only the structural
adapt/abstain cells of the two constant policies are filled; every metric cell is
a placeholder until the camera run.

| Policy | Balanced acc ↑ | Macro-F1 ↑ | Regret ↓ | FA_u ↓ | FA_c ↓ | Adapt rate | Abstain rate | Mean latency ms ↓ |
|---|---|---|---|---|---|---|---|---|
| always-freeze | — | — | — | — | — | 0.00 | 0.00 | — |
| always-adapt | — | — | — | — | — | 1.00 | 0.00 | — |
| confidence gate | — | — | — | — | — | — | — | — |
| entropy gate | — | — | — | — | — | — | — | — |
| KGA-no-radius | — | — | — | — | — | — | — | — |
| KGA-certificate | — | — | — | — | — | — | — | — |

*FA_u = P(adapt ∧ Δ ≤ 0); FA_c = P(Δ ≤ 0 | adapt). Ground-truth labels are
unavailable during live decisions and are revealed only offline; Δ is the
per-window benefit (adapted minus frozen balanced accuracy).*

## Table R3 — Physical-shift breakdown (template; populate from real logs)

Regret and the certified decision mix per physical shift family on the held-out
stream.

| Shift family | Windows | Always-adapt regret ↓ | KGA regret ↓ | KGA decision pattern | Interpretation |
|---|---|---|---|---|---|
| Mild lighting | — | — | — | — | — |
| Strong shadow | — | — | — | — | — |
| Blur / vibration | — | — | — | — | — |
| Background / viewpoint | — | — | — | — | — |
| Batch-composition | — | — | — | — | — |

## Table S1 — Recording inventory (template; populate from real sessions)

One row per recording session; sessions are on distinct days.

| Session | Day | Clips | Windows | Classes | Shift families present | Notes |
|---|---|---|---|---|---|---|
| Source (clean) | — | — | — | — | — | — |
| Calibration-fit | — | — | — | — | — | — |
| Calibration-conformal | — | — | — | — | — | — |
| Held-out | — | — | — | — | — | — |
| Replication | — | — | — | — | — | — |

## Table S3 — Per-condition held-out breakdown (template; populate from real logs)

One row per recorded condition; enumerate to match the held-out session.

| Condition | Shift family | Windows | Mean Δ | Always-adapt regret | KGA regret | KGA adapt rate | KGA abstain rate |
|---|---|---|---|---|---|---|---|
| Lighting (mild) | Lighting | — | — | — | — | — | — |
| Lighting (strong) | Lighting | — | — | — | — | — | — |
| Shadow (soft) | Shadow | — | — | — | — | — | — |
| Shadow (hard) | Shadow | — | — | — | — | — | — |
| Motion blur | Blur/vibration | — | — | — | — | — | — |
| Camera vibration | Blur/vibration | — | — | — | — | — | — |
| New background | Background | — | — | — | — | — | — |
| New viewpoint | Viewpoint | — | — | — | — | — | — |
| Mixed batch | Batch-composition | — | — | — | — | — | — |

## Table S4 — Runtime profile (template; populate from real logs)

Per-window wall-clock by pipeline stage on the deployment device.

| Stage | Mean ms ↓ | p95 ms ↓ | Share % |
|---|---|---|---|
| Capture (frame acquisition) | — | — | — |
| Frozen inference f₀ | — | — | — |
| TENT update (1 step, BN-affine) | — | — | — |
| Candidate inference fₐ | — | — | — |
| Evidence (14 features) | — | — | — |
| Estimator + gate (B̂, ε, decide) | — | — | — |
| **Full window (end-to-end)** | — | — | — |

## Table S5 — Evidence/gate ablation (template; populate from real logs)

Each variant scored on the identical held-out stream.

| Variant | Regret ↓ | FA_u ↓ | FA_c ↓ | Adapt rate | Abstain rate | Δ vs full KGA |
|---|---|---|---|---|---|---|
| Full KGA (certificate) | — | — | — | — | — | — |
| No-radius (ε = 0) | — | — | — | — | — | — |
| No blur/brightness features | — | — | — | — | — | — |
| No disagreement feature | — | — | — | — | — | — |
| Confidence-only | — | — | — | — | — | — |
| Entropy-only | — | — | — | — | — | — |

## Table S2 — Anti-leakage audit (verified by tests; complete now)

Each guarantee is enforced by a unit test in `edge/tests/`.

| Check | Status | Verified by |
|---|---|---|
| Frozen checkpoint changed after candidate adaptation? | No | `test_candidate_isolation`: f₀ state-dict hash bit-identical before/after; candidate is a deep copy, BN-affine only |
| Test/held-out labels accessible to live runtime? | No | `test_no_live_labels`: payload and logger guards raise `LabelLeakError` on any label-like key |
| Feature schema changed after test lock? | No | `test_features`: `EDGE_EVIDENCE_NAMES` frozen at 14 features (first 11 identical to the paper) |
| ε calibrated only from the conformal split? | Yes | `test_conformal`: estimator fit on calibration-fit; residuals from calibration-conformal only; ε = conservative order statistic at α = 0.10 |
| Test session used to tune adapter/features? | No | Protocol split (calibration-fit/conformal disjoint from held-out; held-out is replay-only) + `test_no_live_labels` |
| Same test stream replayed for all policies? | Yes | `policy_comparison`: all six policies scored on identical windows/inputs |
| Config hash in every log row? | Yes | `test_log_integrity`: `WindowLogger` writes `config_hash` per record |
| Model hash in every log row? | Yes | `test_log_integrity`: `WindowLogger` writes `model_version` (= state-dict hash) per record |

---

## Author notes (layout guidance — not part of the section body)

**Recommended 1.5-page short-paper layout**

- Half-page pipeline figure (Figure 1) across the top.
- R1 ≈ one-third page directly beneath it.
- R2 ≈ half page.
- One interpretation paragraph tying regret / FA_u / abstention to the
  adapt / freeze / abstain story (no new tables).

**Minimum table package by result quality**

- *Modest / no-harm result* → R1 + R2 + S1–S4. Call it a feasibility / no-harm
  study: "the certificate never adapts harmfully, at bounded per-window latency."
- *Strong win* → additionally include R3 + S5. Call it a held-out physical-shift
  validation: regret reduced vs always-adapt, family by family, with the ablation
  isolating which evidence/gate parts carry it.
- *Abstain-heavy outcome* → frame as the certificate retaining the frozen
  fallback under hard shifts: the unknowability story. It abstains and keeps the
  official model rather than gamble when the evidence is insufficient.
