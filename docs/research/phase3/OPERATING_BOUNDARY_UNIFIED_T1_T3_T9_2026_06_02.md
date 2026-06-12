# The Operating Boundary of Reliability-Gated Fusion (Unified T1 / T3 / T9)

Date: 2026-06-02
Status: theory synthesis — no new claim; unifies three existing registry theorems
(T1, T3, T9) into one operating envelope and shows the 2026-06-02 experiments
confirm all three regimes.

## One-paragraph thesis

Reliability-gated multimodal fusion is neither universally better nor a gimmick:
its value is a **function of the reliability regime**, and that function is now
characterized on both ends and at the boundary. When every modality is reliable
and near the discrimination ceiling, gating provably cannot beat the
confidence-weighted mean (T9). When modalities are corrupted, quality-blind
fusion — CW included — is provably dominated, so gating has a positive lower
bound (T1). The transition between the two is a closed-form threshold in how
many modalities have degraded (T3). Together these bound the gate from above,
from below, and locate the crossover — a complete, falsifiable operating
boundary rather than a "our method wins" claim.

## The three theorems as one envelope

| Regime | Theorem | Statement | What it bounds |
|---|---|---|---|
| **Clean / near-ceiling** | **T9** | For *any* fusion class G, `Δ*(G) ≤ ε_subopt = A* − A(CW)`. On clean near-separable transfer `ε_subopt → 0`, so CW is unbeatable. | **Upper bound** — gating's max achievable gain is the optimality gap of CW, which vanishes when CW is already near the Neyman–Pearson ceiling. |
| **Corrupted / stress** | **T1** | Quality-blind fusion (sparse-linear T1a; coherent all-rules T1b) is dominated under corruption by a constructive adversary; a reliability-aware rule strictly improves. | **Lower bound** — under genuine reliability asymmetry, gating has positive recoverable advantage. |
| **Partial failure** | **T3** | Mean-gate miss probability `P(miss) = Φ((μ̄ − τ)/σ̄)`; deterministic boundary `k* = D(μ_h − τ)/(μ_h − μ_c)`. | **Crossover** — how many of D modalities must degrade before mean/CW fusion misses, i.e. where the gate starts to matter. |

The picture: T9 caps the clean end at zero, T1 lifts the corrupted end above zero,
and T3 says where along the degradation axis the curve crosses. The gate's value
lives in the band that T1/T3 open and T9 closes.

## Why this is the right framing (not a retreat)

A "universal SOTA" claim is both false (T9 forbids it on clean data) and weaker
than what is provable. A *characterization with proofs on both ends* is a
stronger contribution: it tells a practitioner exactly when to deploy reliability
gating (stress/degradation, multiple reliable-but-asymmetric modalities) and when
not to (clean, near-ceiling, or single-informative-modality). The T9 certificate
(`gate_e_unpassable_certificate`) is itself a drop-in test any fusion study can
run before claiming a clean-data win.

## The 2026-06-02 experiments confirm all three regimes

**Clean end (T9), one-class leaderboard protocol.** On MVTec 3D-AD, canonical
one-class, category-averaged (`mvtec3d_one_class_degradation_results.json`):

| rule | clean image-AUROC |
|---|---|
| CW | 0.7910 |
| RGA | 0.7918 |
| gated-CW | 0.7910 |

`Δ(gated-CW − CW) = 0.000`, not significant — gating ties CW on clean data,
exactly as T9 predicts. (Leaderboard position: 0.791 vs published 0.865–0.945;
ELARA is score-fusion-only, no cross-modal patch head — see
`docs/research/tables/mvtec3d_sota_demarcation.tex`.)

**Stress end (T1/T3), same categories under degradation.** As one modality is
corrupted (α = 0.25 → 1.0), gated-CW beats CW by **+0.18 to +0.25** image-AUROC,
significant at every level — the positive lower bound T1 guarantees, crossing
over from the clean tie as T3 describes.

**The boundary, sharpened by the D18 guard analysis.** The degenerate-channel
guard (`src/elara/evaluation/degenerate_channel_guard.py`) showed that on messy
real Real-IAD-D3 data, *most* of CW's apparent deficit was trusting inverted /
saturated channels, not a true gating advantage: guarding both sides shrinks the
gated-vs-CW margin from +0.045 to +0.005, surviving only where ≥2 channels are
genuinely reliable. This is the T9/T1 boundary made operational — when a category
exposes one good channel, CW = gated (nothing to arbitrate, T9); the gate earns
its keep only when multiple reliable-but-asymmetric channels coexist (T1).

## Falsifiable predictions (for review defensibility)

1. **T9:** on any clean near-ceiling benchmark, no fusion rule (gate included)
   beats CW beyond `ε_subopt`; an unconstrained oracle on the same scores will
   tie CW.
2. **T1/T3:** as reliability asymmetry grows past `k*`, the gate's advantage
   over CW becomes positive and grows monotonically with the asymmetry.
3. **Boundary:** improving detectors so more categories expose ≥2 reliable
   channels should widen the guarded gated-vs-CW margin (this is the hypothesis
   the Lever-1c development experiment tests directly).

## Artifacts

- T1: `src/elara/theory/t1_impossibility.py`, `experiments/fusion/t1_impossibility_validation.json`
- T3: `src/elara/family_b/corruption.py`, `experiments/fusion/craf_real_k_domain_results.json`
- T9: `src/elara/theory/t9_clean_transfer_ceiling.py`, `experiments/fusion/t9_clean_transfer_ceiling_validation.json`
- One-class confirmation: `experiments/fusion/mvtec3d_one_class_degradation_results.json`
- Boundary / guard: `src/elara/evaluation/degenerate_channel_guard.py`,
  `experiments/fusion/guarded_channel_dev_analysis.json`
- Registry: `THEOREM_REGISTRY["T1"|"T3"|"T9"]`
