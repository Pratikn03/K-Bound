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

You have **12 raw datasets on disk (~96 GB)**. Not all feed the headline claim,
and that is **by design**, not neglect. Here is the full map.

### Datasets and their role

| Dataset | Size | Modalities | Role | Used in headline? | Why / why not |
|---|---|---|---|---|---|
| **MVTec 3D-AD** | 26 GB | RGB + depth (XYZ) | **M1 primary** | **YES** | Naturally paired RGB+depth; the in-domain Gate-D/T5 win lives here |
| **3D-ADAM** | 6.5 GB | RGB + depth | **M2 external transfer** | **YES** | Held-out external set; the transfer / stress-regime evidence lives here |
| **Eyecandies** | 25 GB | RGB + depth | M2 development | Partial | Used for Family-D study; reclassified to *development* after it failed transfer (Policy B, D1) — kept for analysis, not for the final transfer claim |
| **MVTec LOCO-AD** | 12 GB | RGB → derived edge | M1 secondary | Diagnostic only | RGB-only with a *derived* second "modality" — not a true second sensor, so it can't carry a multimodal claim |
| **VisA** | 4.3 GB | RGB → derived edge | M1 secondary | Diagnostic only | Same reason: derived-view proxy, not genuine multimodality |
| **Real3D-AD** | 10 GB | RGB + 3D | candidate | Tier-B only | Used in an earlier mechanism replication; not part of the v3 headline |
| **UNSW-NB15** | 606 MB | network-flow views | M3 candidate (non-vision) | Family-A/B | Used to show the method isn't vision-only, but it's "structured views," not co-observed sensors |
| **Healthcare (GridPulse)** | 434 MB | clinical multimodal | M3 development | Built, not headline | Fusion inputs exist (`healthcare_*_inputs.csv`); sealed as a *development* M3 candidate (D4), not yet a confirmatory cell |
| **Fraud / Behavior / NLP** | ~250 MB | tabular / text | M0 components | Mechanism only | These feed the *synthetic label-aligned* ELARA-Bench benchmark (Family-B mechanism), not the natural-pairing transfer claim |
| **Vision (CIFAR-style)** | 256 KB | image | scaffolding | No | Tiny utility data, not a research benchmark |

### The principle behind "used vs. not used"

The headline claim is about **naturally co-observed multimodal data** (two real
sensors of the same object). Only a few datasets qualify as *genuinely* paired:
**MVTec 3D-AD and 3D-ADAM (RGB+depth), Real3D-AD, Eyecandies.** The others are
either:

- **derived-view proxies** (VisA, LOCO — one modality is computed from the
  other, so "fusion" is partly circular), or
- **label-aligned composites** (ELARA-Bench from fraud/cyber/NLP — different
  samples glued by class label, useful for *mechanism* tests but not a real
  multimodal claim), or
- **development / candidate** sets (Eyecandies after it failed; Healthcare M3,
  not yet executed at confirmatory level).

So unused-in-headline ≠ wasted. They are **honestly scoped out** because
including a derived-view or label-aligned result in a "multimodal generalization"
claim would be overclaiming. This conservatism is *why the work is defensible*.

**The one genuine gap:** a *second untouched naturally-paired RGB+depth* dataset
for the final transfer audit (registry slot `m2_new_untouched_transfer`,
`train_allowed: false_until_sealed`) has **not been acquired**. That is the
single missing dataset that would strengthen the transfer claim — not any of the
ones already on disk.

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

### What did NOT pass (honestly)
- **Gate E (clean external transfer): FAIL.** RGA+ vs SAR on *clean* 3D-ADAM is a
  statistical **tie** (+0.0139, CI [−0.0007, +0.0286]). Worse, a parameter-free
  confidence-weighted mean (0.912) beats both RGA+ (0.886) and SAR (0.872) on
  clean transfer. So clean-regime fusion has no edge.
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

- **NOT** Master-C flagship, **NOT** deployment-ready, **NOT** universal/SOTA,
  **NOT** independently reproduced.
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
2. **Natural degradation** (real sensor artifacts) instead of synthetic noise.
3. **A second untouched naturally-paired RGB+depth dataset** for the transfer
   audit (the one genuinely-missing dataset).
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
