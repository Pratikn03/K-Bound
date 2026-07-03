# Wave-5 Gap-Closure — Results (2026-07-02)

Executed per `PROTOCOL_GAPCLOSE_WAVE5_v1.md` (frozen before any result).
Everything below was actually run in this session (CPU); artifacts:
`val_gapA_radius_results.json`, `val_gapB_tau_results.json`,
`val_gapC_evidence_results.json`, `retro_gapA_results.json`.

## Verdict table

| Gap | Synthetic validator | Real-data retro | Net |
|---|---|---|---|
| A radius | **PASS** (4/4 checks) | **FAILS frozen 1.5× bar** — but radius ↓34%, commits ×5–18 at FA ≤ α | Method valid; Camelyon radius is **evidence-limited**, not quantile-limited |
| B τ gate | **PASS** (27/27 level cells, power 1.00) | Not retro-computable (c_ij never serialized) | Ship to GPU runner (one-line serialization) |
| C evidence | **PASS** (uplift +0.064 ≥ +0.05, stable) | GPU-side (needs logits) | Wire per GPU_WIRING.md |

## Gap A — de-biased radius

Validator (synthetic, all frozen checks green): published symmetric radius inflates
(ratio80 2.03); signed + orthogonalized + metadata-augmented variant recovers 1.105 at
level; proper weighted conformal (per-test-point weight, Tibshirani correction) restores
coverage under moderate-window drift (severe-drift support collapse documented — the
conformal analogue of the γ-frontier); the irreducible-latent control stays honestly wide
(no cheating).

Retro on `result_73add410.json` (432 real records, leave-one-seed, EXACT published GBR):

| Variant | w_eff | ratio80 | FA | commit rate |
|---|---|---|---|---|
| V0 published | 0.1118 (replicates 0.1127 ✓) | 3.57 | 0.000 | 5% |
| V1 signed | 0.0819 | 2.62 | 0.035 | 27% |
| V3 signed+Mondrian | **0.0737** | **2.35** | 0.039 | 41% |
| V4 weighted | 0.1395 | 4.46 | 0.039 | — |

**Honest negative on the frozen acceptance (2.35 ≥ 1.5).** Mechanism, established:
mean bias is only +0.006 — the pathology is unlearnable residual **variance** (σ_signed
≈ 2.5× σ_meas), not directional drift; 16 dims of observable condition metadata explain
almost none of it (V1_base 2.62 → V1_aug 2.57); seed-to-seed reweighting only loses ESS
(V4 worse). Conclusion: **the radius is evidence-limited — the closing lever is richer Z
(Gap C), exactly the γ-frontier statement of the paper.** Deployment-relevant regardless:
V1/V3 cut the radius ~30% and lift commit rate 5%→27–41% at FA ≤ α, worth wiring in.

Deviations (documented): V2/V3 defined on metadata-augmented Z (set pre-retro);
A3 gated on "any drift level exhibiting pathology → restoration" (frozen wording had no
scale); coverage on real data 0.87/0.83 vs nominal 0.90 — residuals not exchangeable
across seeds (same evidence-limitation).

## Gap B — self-normalized τ (transferable CEI gate)

Final method: τ′ = τ̂ / Q₀.₉₅(τ̂_null), null = smoothed parametric bootstrap under
fitted H (b̂ perturbed by its own triple-product SE — added after the plain plug-in
null proved anti-conservative at weak-agreement scales: level 0.085 → 0.058 at 480 reps).

Validator, complete 27-cell grid (K ∈ {3,6,10} × m ∈ {200, 2k, 20k} × b ∈ {0.2,0.5,0.8}):
- **Level holds everywhere** (worst 0.083 within its cell bound). Fixed τ*=0.52: spurious
  rejection up to **100%** on H-true cells (small-m/low-b) — the transfer failure, reproduced.
- **Power 1.00** on twin-agreement co-adaptation (co-trained candidates agreeing on
  mistakes — the violation actually diagnosed on your panels). Fixed 0.52: power **0.025**.
- Two side-findings with theory content, now encoded in the validator: (i) joint flips of
  ALL candidates leave every pairwise agreement invariant (empirical analogue of the swap
  involution — undetectable in principle); (ii) uniform-strength co-adaptation collapses
  to near-rank-1 (δ ∝ b), weakly detectable; heterogeneous/twin structure is the
  detectable case.
- π dropped from the grid (provably vacuous under symmetric H); reps/n_sim per m
  documented in results JSON.

Retro: **not possible** — no logged ImageNet-R artifact contains agreement matrices
(multiseed per-condition files carry the keys but null; light-grid runs never had them).
GPU runner needs the one-line `c_ij`/`n_D` serialization (GPU_WIRING.md §2b), then τ′
drops in for the frozen 0.52.

## Gap C — richer evidence (logits-only)

`evidence_v2.py`: MaNo (softrun + L4 norm), normalized nuclear norm, GdScore proxy;
entropy/MSP baselines. Validator: drift family with confidence miscalibration
(temperature drift makes entropy/MSP mislead): harm-AUC 0.875 → **0.939 (uplift +0.064 ≥
+0.05 frozen)**, stable across severity bands. Note: nuclear norm is a reversed signal
(single AUC 0.06 ⇒ 0.94 flipped) — combiner handles orientation. Real-data payoff needs
logits → GPU re-run (GPU_WIRING.md §1–2); this is also the identified lever for Gap A's
evidence-limited radius.

## What this wave changes about the program

1. The bias-limited-radius open problem is now **mechanistically located**: not quantile
   mechanics (30% recoverable, done), not metadata, not seed-reweighting — missing
   evidence dimensions. Gap C's features are the specific, validated candidate.
2. The τ-transfer future-work item is **solved as a method** (validated at level and
   power); it needs only serialization + the pre-registered GPU protocol.
3. **NATURAL_WIN v1 executed (2026-07-03):** both arms **FAIL** under frozen bar —
   FA_u ≤ α on Camelyon17 (4.9%) and ImageNet-R (3.7%), but regret exceeds
   always-adapt in helpful-dominated regimes (59% / 46% abstain). Verdict JSONs:
   `experiments/kbound/results/natural_win_v1_*/NATURAL_WIN_v1_*.json`. Honest
   negative closes instrumented natural beats-both for this protocol; paper headline
   remains uniform natural no-harm under valid OOF radius.

## Reproduce

```bash
cd docs/research/kbound/gapclose_wave5
PART=core python3 val_gapA_radius.py && PART=moderate python3 val_gapA_radius.py && \
PART=mid python3 val_gapA_radius.py && PART=severe python3 val_gapA_radius.py && \
PART=latent python3 val_gapA_radius.py && PART=assemble python3 val_gapA_radius.py
# Gap B (chunked): PART=K{3,6,10}_m{200,2000,20000} then PART=power, then assemble
python3 val_gapC_evidence.py
python3 retro_gapA_camelyon.py
```
