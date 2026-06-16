# Camelyon17 — Analysis of All Runs (K-Bound)

*Generated 2026-06-15. Honest synthesis across every Camelyon17 run on disk. Protocol F was still running (20/90 conditions) at time of writing; its final verdict is pending.*

## TL;DR

Camelyon17 is a **consistent, rigorously characterized boundary / honest negative** for label-free KGA.
With a legitimate frozen τ\*, the certificate is conservative (heavy abstain, 0% false-adapt) but does
**not** beat always-adapt. The pre-registered full-scale run (Protocol B) **fails its own success
criteria**: false-adapt stays ~2–3× α and the conformal radius does not shrink with sample size. A
bias–variance diagnostic pins the cause: the radius is **bias-limited** (2.55× the measurement floor;
only ~16% measurement noise) — the bottleneck is calibration-drift bias, not sample size, so **more
unlabeled data cannot close it.** This is the strongest empirical motivation for the micro-probe /
target-label-light direction.

## The four runs

### 1. Original frozen-τ\* run — `results/wilds/wilds_camelyon17_kga.json`
- `beats_both = False`, coverage 0.75. Baseline: with the default source-calibrated τ\*, KGA does not beat both trivial policies.

### 2. τ\* recalibration diagnostic — `theory_validation/frontier_decisive/camelyon_recal/`
- n = 432, n_harmful = 124, **base-rate-harmful = 0.287**, best-feature **harm-AUC = 0.776** (the paper's "0.29 / 0.78").
- **Frozen τ\*=0.52:** false-adapt **0.0**, coverage **0.215**, regret 0.0313 (adapt 79 / freeze 14 / abstain 339) — very conservative.
- **Recalibrated τ\* (refit on Camelyon):** false-adapt 0.097 (≈α), coverage **0.569**, regret **0.0104** — this beats always-adapt (0.013) and always-freeze (0.052).
- **But:** the recalibrated "win" requires refitting τ\* on the external panel itself, which forfeits its held-out validation role. It is a **diagnostic, not a claimable result**.

### 3. Protocol B — full-scale, 5 seeds, n_eval=1024 — `results/camelyon17_fullscale_B_v1/`  →  VERDICT: **FAILS** (per pre-registered criteria)
| run | ε | false-adapt | commit | radius ratio (vs 0.5 predicted) |
|-----|-----|-------------|--------|----------------------------------|
| n=256 (grid) | 0.030 | 0.139–0.185 | 0.58–0.65 | — |
| n=1024 tent | 0.029 | **0.333** | 0.60 | **0.99** |
| n=1024 eata | 0.038 | **0.333** | 0.60 | **1.29** |
- regret (n=1024 tent): KGA 0.020 vs always-adapt 0.0063 vs always-freeze 0.027 → **beats freeze only, not always-adapt**.
- False-adapt did **not** fall to ≤α (0.33 ≫ 0.10); radius did **not** halve (ratio ~1.0, not 0.5).
- Honest caveat: the n=1024 run serialized only per-seed aggregates (5 points/method), so the grid-granularity 1/√n prediction is **largely untested** at its stated granularity.

### 3b. Bias–variance diagnostic — `camelyon17_fullscale_B_v1/bias_variance_diag/`  →  VERDICT: **BIAS-LIMITED**
- ε₂₅₆ observed = **0.1124** (0.9-quantile of |B̂(Z)−B|, leave-one-seed cross-fit).
- Measurement-noise floor ε_meas(256) = **0.0441** → observed/floor = **2.55×**.
- Fraction of residual variance that is measurement noise = **0.16** (signed) / 0.39 (abs) → residual dominated by **model error**, not eval-sample noise.
- n=1024 cross-check: B-std ratio 0.71 (not the 0.50 a variance-limited halving needs) — consistent.
- **Implication:** the radius is dominated by B̂(Z) calibration-drift bias, which does not shrink with eval n. A fair n=1024 re-run touches only the ~16–39% variance slice → cannot close the certificate. **Reducing estimator bias (i.e. labels) is the lever.**

### 4. Protocol F — rich-evidence, running (20/90 at writing) — `results/camelyon17_richZ_F_v1/`
- Tests whether a **richer overdetermined evidence vector Z (M=7 rank-one channels)**, with a ppi-debias estimator + Mondrian conformal (dev seeds 0–1 / test seeds 2–4), shrinks the abstain radius.
- So far (20/90): **~18 ABSTAIN, ~2 ADAPT.** Abstentions fire because τ (overdetermination residual / margin) exceeds τ\*=0.52 → "Def-5 (risk-alignment) certified violated."
- **Early read:** richer label-free evidence is **not yet** flipping Camelyon out of the abstain regime — consistent with the bias-limited finding. Final verdict pending the full 90 conditions + `analyze_F.py`.

## What it all means

1. **Camelyon17 is the paper's cleanest "unknowable on natural shift" case, honestly reported as a non-win.** Every legitimate (held-out) configuration abstains or beats freeze only.
2. **The bias-limited diagnostic is the scientific payoff:** it reframes the open problem from "need more data" to "need to reduce estimator bias." Labels (a micro-probe) attack bias directly; richer label-free evidence (Protocol F) is being tested and is, so far, still abstaining.
3. **Consistent with the other natural shifts** — iWildCam (weak detectability, false-adapt 0.50) and ImageNet-R (full abstain) — all sit in the label-free-unidentifiable regime the impossibility theorem predicts.

## Honest caveats
- Protocol B's n=1024 collapsed the 72×6 grid → the 1/√n prediction is largely untested at its stated granularity; the bias-limited reading rests on the n=256 grid + cross-fit diagnostic.
- Protocol F is incomplete (20/90); no verdict yet.
- The recalibrated "beats-both" is not claimable (forfeits the validation role of τ\*).
