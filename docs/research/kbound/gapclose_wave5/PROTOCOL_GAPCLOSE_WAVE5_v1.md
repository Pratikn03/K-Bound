# PROTOCOL — Gap-Closure Wave 5 (v1)

**Pre-registered: 2026-07-02, BEFORE any Wave-5 retro-result was computed.**
Author: Pratik + Claude (method implementation session).
Companion code: `gapclose_wave5/*.py`. Results will be written to `RESULTS_WAVE5.md`
only after this file is frozen. An honest negative on any criterion is a valid outcome
and will be reported as FAILS.

Input data (locked, sha8):

- `5d286065` `experiments/kbound/results/wilds_kbound_debug_mps/result_73add410.json` (Camelyon17 n256 grid, 432 records, seeds 0–3)
- `a2a138c7` `experiments/kbound/results/camelyon17_richZ_F_v1/result_884129ba.json` (rich-Z 54-cell eata_online)
- `0f5c75fd`(+29 files) `experiments/kbound/results/imagenetr_protocol_d_multiseed_v1/per_condition_*.json` (10 backbones × 3 seeds, c_ij stored)

Baseline constants (from `camelyon17_fullscale_B_v1/bias_variance_diag/diag_results.json`, already published):
ε_256_observed = 0.11266 (symmetric |resid| Q90, leave-one-seed cross-fit);
mean σ_meas = 0.026816; ε_meas(Q90 of |noise|) = 1.645·σ = 0.04411; ratio = 2.554.

---

## Gap A — de-bias the conformal radius (`radius_v2.py`)

**Diagnosis being attacked:** radius is bias-limited — symmetric absolute-residual quantile
pays the systematic model error |b₀| in BOTH directions; signed quantiles absorb it once as
a recentering. Plus: global radius pays worst-case bias everywhere (fix: Mondrian bins);
exchangeability under drift (fix: weighted conformal); validity (finite-sample rank-corrected
quantiles; CV+/jackknife+ option).

**Method (frozen):** leave-one-seed cross-fitted GBR (EXACT estimator config from
`eps_recal_camelyon.py`: n_est 250, depth 2, lr 0.05, subsample 0.8, rs 0) →
signed out-of-fold residuals r = B − B̂. Commit rules at α = 0.10 per direction
(matches Thm cert): ADAPT iff B̂ + q̂_α(r) > 0; FREEZE iff B̂ + q̂_{1−α}(r) < 0.
Rank-corrected empirical quantiles (⌈(n+1)α⌉ index). Variants, all reported:
V0 baseline symmetric |r| Q90 (reproduces published pathology);
V1 signed asymmetric (primary);
V2 = V1 + cross-fitted ridge residual orthogonalization on Z (DML-style second stage);
V3 = V2 + Mondrian (terciles of B̂, per-bin quantiles, min-bin 30);
V4 = V1 + likelihood-ratio weighted quantiles (logistic density ratio cal→eval in Z space).

**Frozen metrics:**
- effective central-80% half-width w = (q̂_{0.90} − q̂_{0.10})/2, cross-fitted;
- matched noise floor w_meas = 1.2816·mean σ_meas (same central-80% convention);
  legacy floor 1.645·σ also reported for continuity with the diag.
- FA_emp: fraction of held-out cells with decision ADAPT and B ≤ 0.
- per-direction coverage: P(B ≥ B̂ + q̂_α) and P(B ≤ B̂ + q̂_{1−α}), cross-fitted.

**Acceptance (frozen):** on `result_73add410.json`, best variant achieves
**w / w_meas < 1.5** AND **FA_emp ≤ 0.10 + 2·MC-se** AND per-direction coverage ≥ 0.88.
Synthetic validator `val_gapA_radius.py` must pass first (exit 0): replicates the
bias-inflation pathology, shows V1–V4 recover ratio < 1.5 at level, and shows the
weighted variant restores coverage under injected covariate drift where unweighted breaks.

## Gap B — transferable τ gate (`tau_selfnorm.py`)

**Diagnosis being attacked:** fixed τ\* = 0.52 calibrated on synthetic panels does not
transfer (Camelyon τ̄ = 0.89, ImageNet-R τ̄ = 2.60) because the rank-one residual's null
scale depends on panel size K, sample size m, and |b| magnitudes.

**Method (frozen):** self-normalization by a label-free parametric-bootstrap null under
fitted H. From observed agreement matrix C and m: fit |b̂| by median triple products
(signs anchored to candidate 0, global flip irrelevant); π̂ from predicted-class balance
(0.5 if unavailable); simulate S = 400 panels of m draws from H(b̂, π̂); τ̂ = normalized
off-diagonal Frobenius residual ‖C_off − b̂b̂ᵀ_off‖_F / ‖C_off‖_F, identical formula for
observed and simulated. **τ′ = τ̂_obs / Q_{1−α}(τ̂_null); reject H iff τ′ > 1** (α = 0.05).

**Acceptance (frozen), `val_gapB_tau.py` exit 0:**
- Level: H-true panels across K ∈ {3,6,10}, m ∈ {200, 2000, 20000}, b-scale ∈ {0.2, 0.5, 0.8},
  π ∈ {0.3, 0.5}: rejection rate ≤ α + 2·MC-se in every cell (200 reps/cell).
- Power: co-adapted panels (latent common-noise flip, ρ = 0.25) at K = 6, m = 2000: rejection ≥ 0.9.
- Transfer failure of the fixed threshold: exists a family cell where fixed τ\* = 0.52 has
  level > 0.5 (spurious rejection) or power < 0.2, while τ′ holds both.
Retro-run `retro_gapB_imagenetr.py` (10-backbone panel, 36 conditions): **diagnostic only** —
reports per-condition τ′ vs the frozen-0.52 decision; no ground-truth CEI on real data, so no
win/loss verdict is claimed from the retro; the decisive natural-shift use stays gated on a
future pre-registered GPU protocol.

## Gap C — richer evidence map (`evidence_v2.py`)

**Method (frozen):** logits-only label-free features: MaNo (softrun-normalized logit matrix
L₄ norm), normalized softmax nuclear norm, logits-only GdScore proxy (mean ‖p − e_ŷ‖₁),
plus entropy/MSP baselines. ProjNorm documented as GPU-side (probe training) — not claimed here.

**Acceptance (frozen), `val_gapC_evidence.py` exit 0:** in a synthetic drift family with
confidence miscalibration (temperature drift, so entropy/MSP mislead) and mixed
helpful/harmful adaptation: harm-detection AUC of {baselines + new features} exceeds
{baselines} by ≥ +0.05, and the uplift is monotone-stable across the severity sweep
(no cell where it degrades by > 0.02). This validates mechanism only; the real-data claim
requires wiring into `kga/evidence.py` and a GPU re-run (spec in `GPU_WIRING.md`).

## Explicitly NOT claimed by Wave 5
A natural-shift beats-both. Wave 5 delivers instruments; the decisive run remains a
future pre-registered protocol (NATURAL_WIN_PROTOCOL_v1) executed on GPU with these
instruments frozen in.
