# Closing the gap: external-benchmark scoping for a real multimodal beats-both

*Scoping only — no acquisition. Goal: find an external benchmark that meets the paper's named
condition so KGA can demonstrate a **significant** beats-both (the thing the on-disk 3D-ADAM hunt
missed by tying the safe baseline at P=0.64).*

## The named condition, as a hard checklist
A candidate must have ALL of:
1. **Multimodal / multi-sensor** with a deployable fusion vs a unimodal fallback.
2. **A genuine helpful↔harmful flip across conditions**: fusion *gains* in some conditions and *hurts*
   in others — and BOTH sides are material (so neither always-adapt nor always-freeze is already best).
3. **Label-free detectability** of the flip: per-modality reliability / confidence / disagreement
   drops when a modality fails (so the certificate's evidence can see it without test labels).
4. **Paired labels** to measure true benefit + false-adapt.
5. **Many conditions** (corruptions × severities × scenes) so a real margin can reach significance —
   the n=23 categories in 3D-ADAM were too few.

## Why this is findable (evidence from the literature)
The flip provably occurs: in AV, "when one modality experiences corruption — LiDAR occlusion or camera
obstruction — it can severely degrade the other modality" [MultiCorrupt; ReliFusion]. And fusion *can*
underperform unimodal: "in VB100, audiovisual fusion performs worse than vision-only because audio is
uninformative/distracting"; "multimodal benefits rapidly diminish under realistic missingness"
[MultiBench / low-quality-fusion survey]. So both sides of the flip exist in real benchmarks.

## Ranked shortlist
| Rank | Benchmark | Fit to named condition | Detectability | Conditions (n) | Acquisition cost | Risk |
|---|---|---|---|---|---|---|
| **1** | **MultiCorrupt** (LiDAR-camera, 3D det.) | **Excellent** — fusion clearly flips help↔harm across 10 corruptions; ReliFusion shows reliability is the signal | High (per-modality confidence drops under corruption) | 10 corruptions × 3 severities × many scenes = **hundreds** | **High** (nuScenes ~300GB+, run 3-5 detectors, corruption pipeline) | Metric is mAP/NDS, not per-sample AUROC → must adapt certificate to a detection metric |
| **2** | **MultiBench / MultiBench++** (missing/noisy-modality classification) | Good — controllable per-condition modality noise; documented fusion-hurts cases | Medium-High (reliability from per-modality conf.) | Many datasets × noise levels (**tunable, large**) | **Medium** (precomputed features; CPU-feasible) | Must find the dataset where flip is present AND detectable AND fusion gains |
| 3 | **EHR + Chest-X-ray "when does fusion help"** (healthcare) | Good — literally a help-vs-hurt fusion benchmark; per-patient labels | Medium | Patients × missingness regimes | Medium | Modality imbalance may make it freeze-dominated (like our nulls) |
| 4 | **nuScenes-C / Robo3D** | Excellent (same family as #1) | High | 20 corruptions | High | Same detection-metric issue as #1 |

## Recommended top pick: MultiBench++ first (tractable), MultiCorrupt as the high-ceiling option
- **MultiBench++** is the cheapest honest test: precomputed features, per-sample classification (certificate-native), controllable modality corruption to *manufacture* the flip, and many conditions for significance. Best ratio of "answers the question" to "weeks of compute." If KGA significantly beats both here, that's a real result.
- **MultiCorrupt** is the highest-ceiling, most *paper-compelling* venue (real AV sensor failure, the LiDAR-in-rain story the paper already invokes) — but it's the heavy lift (large data + detectors + a certificate adapted to detection metrics).

## Pre-registerable experiment design (apply to whichever is chosen)
- **Candidates:** fusion vs each unimodal (camera-only / LiDAR-only, or modality-A / modality-B).
- **Freeze default:** best val-chosen unimodal. **Adapt:** the fusion model.
- **Label-free evidence Z:** per-modality confidence / reliability + cross-modal disagreement (the
  signal ReliFusion uses) — fixed in advance.
- **Conditions:** the benchmark's full corruption × severity grid (the rich n that 3D-ADAM lacked).
- **Certificate:** leave-one-condition-out conformal; decide adapt/freeze/abstain at α=0.10.
- **Primary endpoint (pre-stated):** KGA mean metric **significantly** beats BOTH always-fuse and
  always-best-unimodal via paired bootstrap over conditions (both P>0.95) AND false-adapt ≤ α.
  STRONG if significant beats-both; STANDS if beats always-fuse + ties unimodal; else honest negative.
- **Forbidden:** dataset/condition selection by outcome; tuning Z/τ* after seeing test; reporting
  favorable conditions only. (Same discipline as D25 — seal it before running.)

## Honest odds & effort
- **MultiBench++ path:** ~1-2 weeks; moderate odds of a *significant* beats-both, because we can choose
  a regime where fusion genuinely gains under clean conditions and hurts under corruption, with many
  conditions for power. Risk: it still ties the unimodal baseline (the recurring pattern).
- **MultiCorrupt path:** ~3-6 weeks; higher paper impact if it lands, but heavier and the
  detection-metric adaptation adds methodological risk.
- Even a clean win here moves the paper to ~**80** (strong, real-shift beats-both), not 90; 90 is
  adoption-earned.

## Sources
- MultiCorrupt — arXiv 2402.11677
- Benchmarking Robustness of LiDAR-Camera Fusion (CVPRW 2023) — openaccess.thecvf.com
- Reliability-Driven LiDAR-Camera Fusion (ReliFusion) — arXiv 2502.01856
- MultiBench — arXiv 2107.07502; MultiBench++ — arXiv 2511.06452
- Multimodal Fusion on Low-quality Data: a Survey — arXiv 2404.18947
- When Does Multimodal Learning Help in Healthcare? (EHR+CXR) — arXiv 2602.23614
