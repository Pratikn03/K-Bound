# D25 — PPI Target-Label-Light Micro-Probe (corrected harness): HONEST NEGATIVE

*Pre-registered: `research_lock/TARGET_LABEL_LIGHT_PPI_PROTOCOL_D25_v1.yaml` (sealed 2026-06-15 before any result). Run on 85 real multimodal categories (Real-IAD-D3, Real-IAD-NatDeg, MVTec-3D, 3D-ADAM, MulSen-AD). α=0.10.*

## Why this run exists
The earlier D24 harness (`target_label_light_probe_v1`) was **mis-specified**: its k=0 baseline
used the *full* test-label benefit vector and k>0 merely subsampled it, so the "k-sweep" was a
sample-size ablation with **no label-free baseline** — it could neither confirm nor refute the
micro-probe hypothesis. D25 fixes this with a genuine label-free, *biased* baseline (a val-trained
GradientBoosting predictor of the per-sample Brier benefit, applied to test with no test labels) and
**PPI debiasing** from k labeled test examples.

## Result
| k | commit-rate | adapt / freeze / abstain | false-adapt | sign-acc (committed) | mean regret |
|---|---|---|---|---|---|
| 0 (label-free) | 0.224 | 7 / 12 / 66 | 0.286 | 0.895 | 0.0290 |
| 8  | 0.024 | 0 / 2 / 83 | n/a | 1.000 | 0.0459 |
| 16 | 0.024 | 0 / 2 / 83 | n/a | 1.000 | 0.0459 |
| 32 | 0.024 | 0 / 2 / 83 | n/a | 1.000 | 0.0467 |
| 64 | 0.035 | 1 / 2 / 82 | 0.000 | 1.000 | 0.0444 |

Pre-stated criteria: **S1 commit↑ = FALSE**, **S2 false-adapt≤α = FALSE** (k=0 label-free itself is
0.286), **S3 regret↓ = FALSE**, S4 sign-acc≥1−α = True (trivially — it commits very rarely).
**VERDICT: HONEST_NEGATIVE.**

## Interpretation (honest)
- The micro-probe at k≤64 does **not** reproduce the simulated beats-both. Adding the probe makes the
  certificate **more conservative**: the empirical-Bernstein radius on the k-sample PPI *rectifier*
  (per-sample Brier-benefit residuals, which are high-variance) dominates, so commit-rate falls from
  22% (label-free) to 2–3% and mean regret rises (0.029 → 0.044–0.046).
- Bias removal is genuine (the point estimate is debiased), but at k≤64 the **variance cost exceeds
  the bias benefit** on this real data. The simulation was over-optimistic because it modeled the
  probe as near-noiseless bias removal.
- The label-free k=0 baseline commits more (22%) but at a poor false-adapt rate (0.286, small-n),
  consistent with the bias-limited natural-shift regime.

## What would be needed (not done here; would require a new sealed protocol)
Larger k, a lower-variance rectifier (a stronger label-free predictor so residuals are small), or a
tighter-than-empirical-Bernstein interval. None of these may be changed post-hoc for D25 — that is
forbidden by the protocol. This is logged as a genuine boundary.

## Paper consequence
Per the protocol's `paper_use` clause (update only if STANDS/STRONG), the verdict is HONEST_NEGATIVE,
so the simulation-based "beats-both" claim is **scoped down**, not upgraded: a correct real-label
harness does not validate it at k≤64.
