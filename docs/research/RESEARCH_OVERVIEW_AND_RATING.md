# ELARA — Research Overview, What It Does, and Honest Rating

**Date:** 2026-05-30
**Purpose:** one document that explains, in plain terms, what this research is,
what it actually does, why some datasets are used and others are not, and how
it rates across every axis used to judge research. Written to be readable when
you are switching between many threads.

---

## 1. What is this research, in one paragraph?

ELARA is a **reliability-aware multimodal anomaly-fusion** system. Anomaly
detectors for two modalities (e.g. RGB images + depth/3D) each output a score;
ELARA's contribution is the **fusion layer that decides how much to trust each
modality** at inference time. Its core mechanism, **RGA (Reliability-Gated
Attention)**, watches each modality's score distribution for *drift* away from
what it saw in validation, and **down-weights a modality when it looks
unreliable**. The central research question is not "can we detect anomalies"
(the upstream detectors do that) but **"when should a fusion layer stop trusting
a degraded modality, and does doing so actually help?"**

---

## 2. What does it actually DO? (the mechanism, plainly)

1. **Upstream experts** score each modality independently (now via true
   patch-level PatchCore, ~0.79–0.93 AUROC — competitive).
2. **Reliability estimator** computes, per modality, a trust weight from three
   signals: validation calibration (ECE), score-distribution drift (KS test),
   and score sharpness.
3. **The gate** compares mean reliability to a validation-calibrated threshold
   τ. If reliability is high (no drift) → behave like the strong baseline. If a
   modality has drifted (degraded) → re-weight toward the reliable modality.
4. **RGA-gated-CW** (the latest form) makes this safe: it **defaults to a
   confidence-weighted average** when nothing looks wrong, and only deviates
   under detected drift — so it cannot regress on clean data, and it recovers
   when a modality breaks.

**The honest finding (this is the actual result):** reliability gating is a
**stress-regime mechanism**. It *wins* in-domain and *wins* when a modality
degrades, but on *clean* data where both modalities are reliable, a simple
average is already near-optimal and gating adds little. So the contribution is
a precise characterization of **WHEN reliability gating helps**, not a universal
"our method is best everywhere" claim.

---

## 3. Why are some datasets used and others NOT? (your key question)

You now have **15 primary raw dataset directories on disk (~346 GB)**, plus
download caches. Not all feed the headline claim, and that is **by design**,
not neglect. The full authoritative map is maintained in
[`DATASET_USE_MATRIX.md`](./DATASET_USE_MATRIX.md). The compact map is below.

### Datasets and their role

| Dataset | Size | Modalities / views | Current use | Claim ceiling |
|---|---|---|---|---|
| **MVTec 3D-AD** | 26 GB | RGB + depth/XYZ | Primary M1/Gate D, T5, and controlled-stress replication | Bounded in-domain/stress evidence; not external transfer or one-class SOTA |
| **3D-ADAM** | 6.5 GB | RGB + depth | Sealed M2 external transfer and v3 stress-regime transfer | Opened/spent official M2; clean Gate E failed/tied, stress evidence retained |
| **Real-IAD** | 4.0 GB | RGB industrial images | D13 natural positive-transfer official attempt | Official D13 fail: beats CW, fails SAR; now opened development data |
| **Real-IAD D3** | 259 GB | RGB + pseudo-3D + point cloud | D16 natural-degradation/headroom audit | All-test CW positive, primary stress CI crosses zero; no Gate S pass |
| **MulSen-AD** | 19 GB | RGB + infrared + point cloud | D13 opened-development replication | Development only unless an unopened split/category is prelocked |
| **Eyecandies** | 25 GB | Synthetic RGB + depth + normals | Family D transfer failure record | Valid failed transfer; development/negative evidence only |
| **Real3D-AD** | 10 GB | 3D point-cloud benchmark | Earlier exploratory/mechanism benchmark cells | Tier-B/exploratory only; not v3 headline transfer |
| **MVTec LOCO-AD** | 12 GB | RGB + derived edge proxy | Family A secondary diagnostic | Derived-view proxy; no independent-modality claim |
| **VisA** | 4.3 GB | RGB + derived edge/noise proxy | Family A/C secondary and noise-floor checks | Diagnostic only; not independent multimodal evidence |
| **UNSW-NB15 / cyber** | 606 MB | Flow / connection / context event views | Non-vision structured event-view benchmark | Fusion machinery outside vision; small effect; not co-observed sensors |
| **Healthcare / BIDMC** | 354 MB | Clinical structured/time-series views | M3 development and deployment-audit gap checks | Development evidence; not clinical deployment validation |
| **Fraud / behavior / NLP** | ~252 MB | Tabular / text | ELARA-Bench-LA label-aligned components | Mechanism/stress only; not natural multimodal transfer |
| **Vision scaffold** | 256 KB | Small image utility data | Smoke/scaffolding | No research claim |
| **Download caches** | ~8.4 GB | `_downloads_*` staging data | Acquisition cache only | Not evidence |

### The principle behind "used vs. not used"

The headline claim is now **real-dataset-first**: primary evidence should come
from captured datasets such as **MVTec 3D-AD, 3D-ADAM, Real-IAD, Real-IAD D3,
MulSen-AD, Real3D-AD, UNSW-NB15, and BIDMC-healthcare**. Eyecandies is paired
RGB-D-normal data, but it is synthetic, so it remains a failure/development
record rather than positive primary evidence. Also, "genuinely paired" is not
the same as "fresh official transfer." 3D-ADAM, Real-IAD, Real-IAD D3,
MulSen-AD, and Eyecandies are now opened for at least one track, so they cannot
be reused as fresh evidence without a new prelocked unopened split/category.

The others are either:

- **derived-view proxies** (VisA, LOCO — one modality is computed from the
  other, so "fusion" is partly circular), or
- **label-aligned composites** (ELARA-Bench from fraud/cyber/NLP — different
  samples glued by class label, useful for *mechanism* tests but not a real
  multimodal claim), or
- **development / candidate** sets (Real-IAD after D13, Real-IAD D3 after D16,
  MulSen-AD opened-development, Eyecandies after failure, Healthcare M3).

So unused-in-headline ≠ wasted. They are **honestly scoped out** because
including a derived-view or label-aligned result in a "multimodal generalization"
claim would be overclaiming. This conservatism is *why the work is defensible*.

**The one genuine gap:** the next transfer upgrade needs either a new untouched
RGB+X dataset or a prelocked unopened split/category. It cannot be obtained by
retuning on Real-IAD, Real-IAD D3, 3D-ADAM, MulSen-AD, or Eyecandies and then
calling the same opened data fresh.

Real-IAD D3 natural degradation remains useful: it showed all-test headroom
versus CW and clean default safety, while also proving that the current
stress-subset gate is not statistically strong enough. It is evidence, not a
pass.

---

## 4. The actual results (what passed, what didn't)

### What is confirmed (real, significant, multi-split)
- **In-domain strong-baseline superiority (Gate D + T5): PASS.** On MVTec 3D-AD
  with the competitive detector, across **30 independent stratified splits**,
  RGA+ beats the frozen strongest baseline SAR by **Δ = +0.0240, 95% CI
  [+0.0218, +0.0261], p = 2.6×10⁻¹⁹, 30/30 splits**.
- **Stress-regime transfer (3D-ADAM external):** when a modality degrades, the
  gate beats confidence-weighting significantly (up to **+0.10 AUROC** at full
  corruption), and defaults safely to the baseline on clean data.
- **D13 opened-development natural transfer:** the validation-only residual
  stack beats both SAR and CW on opened 3D-ADAM, but this is development
  evidence only.
- **D16 Real-IAD D3 natural degradation:** the validation-selected stress router
  beats CW on all D3 holdout samples (**Δ = +0.0624, 95% CI [+0.0347,+0.0896]**)
  and defaults on clean samples, but this is not the primary gate endpoint.

### What did NOT pass (honestly)
- **Gate E (clean external transfer): FAIL.** RGA+ vs SAR on *clean* 3D-ADAM is a
  statistical **tie** (+0.0139, CI [−0.0007, +0.0286]). Worse, a parameter-free
  confidence-weighted mean (0.912) beats both RGA+ (0.886) and SAR (0.872) on
  clean transfer. So clean-regime fusion has no edge.
- **D13 Real-IAD natural-transfer official attempt: FAIL vs SAR.** The residual-stack
  candidate beats CW (**Δ = +0.0191, 95% CI [+0.0148, +0.0235]**) but loses to
  SAR (**Δ = −0.0858, 95% CI [−0.0928, −0.0787]**). Because D13 requires both
  SAR and CW to pass, `gate_e_positive_transfer_confirmed=false`. After this
  evaluation, the Real-IAD result is opened evidence, not a reusable fresh
  holdout.
- **D16 Real-IAD D3 natural degradation: OFFICIAL FAIL.** The selected
  stress rule (`score_disagreement`) has positive stress-subset point delta vs
  CW (**Δ = +0.0351**) but the 95% CI crosses zero
  (**[−0.0276,+0.0980]**). Clean no-regression passes, but the primary stress
  endpoint does not, so `gate_s_natural_degradation_confirmed=false`. The D3
  data remain valuable real natural-degradation evidence, but they are now
  opened after the D16 attempt.
- **Gate F (scientific flagship): FAIL** — blocked by Gate E.
- **MVTec degradation replication:** positive point estimates but CIs cross zero
  → "directionally supportive, statistically inconclusive."

### The decisive honesty rule (D7)
The v3 bounded result is **not** a substitute for the Master-C flagship claim,
which remains unmet. We report the bounded win *and* the preserved failure side
by side. No goalpost-moving.

---

## 5. RATING BY SECTION

Scale: F → D → C → B → A (A = top-tier). Each grade has its basis.

### Novelty — **B+ / A−**
- **What's novel:** (a) the *characterization* of when reliability gating helps
  vs. hurts (the clean-vs-stress crossover), validated on held-out external
  data; (b) **RGA-gated-CW**, a gate that provably defaults to a strong baseline
  and only deviates under calibrated drift; (c) the **GDR** coherence-certified
  decision rule with a minimax argument.
- **What's NOT novel:** the upstream detector is standard PatchCore; the fusion
  primitives (attention, reliability weighting) build on known components.
- **Verdict:** the *mechanism* is incremental; the *characterization +
  safe-default rule* is genuinely new. Reviewer-defensible novelty, not a
  paradigm shift.

### Benchmarks — **B**
- **Strengths:** uses real, naturally-paired RGB+depth (MVTec 3D-AD, 3D-ADAM);
  competitive upstream detector (~0.79–0.93); held-out external transfer;
  multi-split CIs.
- **Weaknesses:** headline is **supervised-paired**, not the one-class protocol
  the published leaderboard (M3DM/AST) uses → not directly leaderboard-
  comparable; only **2** genuinely-paired vision datasets carry the claim;
  degradation is **controlled synthetic**, not natural.
- **Verdict:** solid, honestly-scoped benchmarking; not yet leaderboard-grade.

### Idea / research question — **A−**
- "When should a fusion layer stop trusting a degraded modality?" is a sharp,
  well-motivated, and *answerable* question. The work actually answers it
  (stress-regime yes, clean-regime no). Strong framing.

### Statistical rigor — **A−**
- Paired + seed-level bootstrap, Holm correction, frozen comparators, 30-split
  CIs, negative results preserved, a real bug found and fixed (depth codec;
  fixed-split determinism). Among the strongest dimensions.

### Theory — **A−**
- 8-theorem stack (T1–T7 + GDR) with a code registry and a validator that
  returns `all_ok: true`. Includes the GDR minimax argument and the
  mean-gate-dilution closed form. Honest about which theorems are
  operationalizations vs. deep results.

### Reproducibility — **A−**
- Pinned lockfile, one-command rebuild, hash-verified data acquisition,
  checkpointed builds, **693 tests (687 pass / 6 skip)**, ruff syntax gate green,
  both PDFs build clean. (Operational risk: the project lives on an unstable
  exFAT external drive — move to APFS / clean clone.)

### Honesty / integrity — **A**
- Refused to force Gate E, refused goalpost-moving, preserved every negative,
  documented the bug-fixes, and wrote D7 to stop the bounded claim from
  masquerading as the flagship. This is the single strongest axis and it is what
  makes everything else credible.

### Empirical findings — **A− (bounded)**
- Real, significant in-domain win + real stress-regime transfer win, but bounded
  by protocol (supervised-paired) and scope (clean transfer is a tie).

### Presentation (paper + code) — **A−**
- Paper has 5 algorithm blocks (RGA, patch-PatchCore, degradation, RGA-gated-CW,
  GDR); thesis has condensed blocks; docstrings filled; manuscript-claim
  validator passes (0 forbidden tokens). Some legacy audit reports still lag.

---

## 6. Overall level

**Level 2.5 / 5** = **strong bounded paper / strong PhD thesis chapter / partial
generalization.**

- **NOT** Master-C flagship, **NOT** universal/SOTA, **NOT** independently
  reproduced, and **NOT** scientifically production-ready in the sense of
  strict clean-transfer deployment evidence.
- **Operational API production readiness is a separate engineering track:**
  `deploy/api` is the bounded production target with authenticated core routes,
  `/ready`, checksum-gated artifacts, fail-closed optional routes, production
  compose defaults, and a runbook. This does not make Gate E pass.
- **IS** a rigorous, honest, novel-in-framing measurement study with a confirmed
  in-domain win and a characterized stress-regime transfer result.

### Submission readiness
| Target | Verdict |
|---|---|
| arXiv preprint | Ready now |
| Workshop / short paper | Comfortable accept |
| Mid-tier conference | Plausible (frame as "when does reliability gating help") |
| Top-tier full paper | Borderline — needs one-class numbers + natural degradation |
| PhD thesis chapter | Strong — arguably thesis-defining |

---

## 7. What would raise the level (honest, none faked)

1. **One-class protocol evaluation** → leaderboard-comparable headline. (Level 3)
2. **Natural degradation** (real sensor artifacts) with a stricter confirmed
   stress endpoint. Real-IAD D3/D16 gives positive all-test evidence but not a
   stress-subset CI pass.
3. **A stronger pre-registered natural-transfer method plus another fresh
   holdout.** Real-IAD is now opened for D13 and failed SAR, so it can guide
   method design but cannot be reused as a new official pass.
4. **A stronger fusion mechanism** (cross-modal patch interaction) to beat the
   confidence-weighted baseline on *clean* transfer — the only path that turns
   Gate E from a tie into a real win.

See `docs/research/phase3/LEVEL_3_PLAN_one_class_and_natural_degradation.md`
for the concrete plan, and `research_lock/DECISIONS_v1.md` (D7/D12) for the
binding claim boundary.

---

## 8. One-sentence summary

**ELARA is an honest, rigorously-validated study showing that reliability-gated
multimodal fusion beats the strongest baseline in-domain and recovers from
modality degradation on held-out external data, while plainly reporting that it
offers no advantage on clean data — a bounded, defensible, PhD-grade result, not
a universal breakthrough.**
