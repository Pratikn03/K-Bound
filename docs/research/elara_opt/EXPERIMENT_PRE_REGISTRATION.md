# ELARA-Opt — Experiment Pre-Registration

**Machine-readable lock:** [`research_lock/elara_opt_protocol_v1.yaml`](../../../research_lock/elara_opt_protocol_v1.yaml) (authoritative; this doc is its readable companion).
**Status:** LOCKED. Built + smoke-tested only. **The held-out evaluation is NOT run** in this phase — it runs after Pratik reviews this pre-registration.
**Branch:** `feature/elara-opt`. **No K-Bound/KGA result, manifest, theorem, or lock is modified.** KGA's `false-adapt ≤ α` certificate is a property of the *gate*; ELARA-Opt is only a new *candidate* that gate certifies.

> ⚠️ This document asserts **no** claim of novelty, superiority, safety, or publishability. It fixes *what will be measured and how* before any test data is touched, so the later result — win, no-harm, or null — is honest by construction.

---

## 1. What ELARA-Opt is

A **label-free test-time parameter-update optimizer** (not a prediction router, not a renamed KGA threshold, distinct from the existing ELARA score-fusion router). On an unlabeled batch it updates only the **BN/LN-affine** parameters (the same surface Tent/EATA/SAR use) with:

1. a **reliability-gated mixture** of unlabeled objectives — entropy, reliability-filtered entropy, augmentation-consistency;
2. a **frozen-model KL stability anchor** `λ·KL(p₀‖p_t)` (function-space trust);
3. a **trust-region** constraint per step whose **radius grows with reliability**.

A gate `g_phi` maps label-free reliability features → nonnegative objective weights. Three modes: `elara_uniform` (uniform), `elara_rule` (deterministic rule, fully specified in the lock), `elara_meta` (tiny MLP trained **only** on source/dev shift tasks; checkpoint + training-data IDs saved, strict no-overlap with eval). It reuses the repo's validated primitives (`_clone_for_tta`, `_bn_affine_params`, `_entropy`, `evidence_vector`) and **references** the existing faithful `sar_adapt` (SAM) rather than re-deriving one.

## 2. Comparison set (10 arms)

Frozen · Tent · EATA · SAR (where faithful) · ELARA-Uniform · ELARA-Rule · ELARA-Meta · **KGA(base)** (the published gate over Tent/EATA/SAR) · **KGA(ELARA-Opt)** · **naive-uniform-no-gate-no-TR** (sanity floor: uniform weights, no gate, no trust region, no anchor ≈ Tent-like).

## 3. Nine-dataset panel — regime + frozen bar

Each dataset keeps the regime the reconciled panel already locked (`KBOUND_6_DATASET_PANEL_v2.yaml`). Frozen bars are **cited**, never re-estimated here.

| Dataset | num_cls | Locked regime | Frozen bar | Runner / protocol |
|---|---|---|---|---|
| CIFAR-10-C | 10 | controlled-corruption beats-both (Tent/EATA) | protocol artifact (stress_grid_multiseed_v1) | `cifar10c_suite.py` / STRESS_GRID_A_v1 |
| ImageNet-C | 10* | scale/breadth (honest Imagenette proxy; full set unavailable from host) | runner output | `imagenette_c_suite.py` |
| Office-Home | 65 | robust beats-both, CI excludes 0 | regret_freeze 0.0158 | `run_officehome_kbound.py` / M_v2 |
| iWildCam | 182 | no-harm Pareto (dominate adapt, tie freeze) | regret_freeze 0.0041 | `run_iwildcam_kbound.py` / H_v2 |
| Camelyon17 | 2 | robust beats-both, CI excludes 0 | regret_freeze 0.0749 | `run_camelyon17_kbound.py` / G_v1 |
| RxRx1 | 1139 | breadth, not headline | protocol artifact | `run_rxrx1_kbound.py` / J_v1 |
| ImageNet-R | 200 | **forced-abstention frontier** (undetectable harm AUC≈0.66) | protocol artifact | `run_imagenetr_kbound.py` / D_v1 |
| CIFAR-10.1 | 10 | low-margin boundary (not cross-seed beats-both) | protocol artifact | Protocol-K pipeline |
| fMoW | 62 | **honest null** (base false-adapt 0.375) | protocol artifact | `run_geoshift_kbound.py` / L_v1 |

\* Imagenette proxy; full ImageNet-C = 1000 classes is unavailable from this host (documented in the runner).

**Capstone (mixed stream):** `run_mixed_headtohead.py` over a pooled heterogeneous deployment; KGA(ELARA-Opt) vs always-adapt and always-freeze, paired bootstrap 95% CI, same bar as the base capstone (`KBOUND_MIXED_STREAM_v1.json`).

## 4. Metrics

α = 0.1; estimator **gbr**; **global** conformal; paired-bootstrap 95% CIs; Holm-Bonferroni across the panel — identical to the published K-Bound bar. **Primary:** mean regret-to-oracle and **false-adapt rate (must be ≤ α)**. `beats_both` ≡ regret-gap CI excludes zero vs **both** trivial policies **and** false-adapt ≤ α (the exact `analysis.policy_metrics.beats_both`). **Secondary:** mean balanced acc, coverage, abstention rate, worst-cell acc, decision counts. **Telemetry endpoints** (label-free): gate weights, trust radius, reliability features, gradient conflict, update norm, candidate hash.

## 5. Ablations

A1 no-gate (uniform) · A2 no-trust-region · A3 no-frozen-anchor (λ=0) · A4 single-objective (each of the three alone) · A5 gate family (uniform vs rule vs meta) · A6 naive floor (no gate + no TR + no anchor).

## 6. Success criteria — headline mode chosen on DEV only

The headline mode (`uniform` | `rule` | `meta`) is selected on **DEV** shift tasks and **locked before any test**.

- **Optimizer-value:** on DEV, KGA(ELARA-Opt headline) regret-to-oracle ≤ KGA(base) within CI **and** the ELARA-Opt candidate is not Pareto-dominated by {Tent,EATA,SAR}. *Null outcome is an allowed, reportable result.*
- **Safety-value:** under the KGA gate, false-adapt ≤ α on **every** panel including the null (fMoW) and the frontier (ImageNet-R → mostly ABSTAIN), with no-harm on iWildCam.
- **Novelty-evidence:** ≥ 1 ablation delta has a 95% CI excluding zero on a DEV regime; **otherwise reported as a null** (no novelty overclaim).
- **Equal visibility:** nulls, no-harm regimes, and negative ablations are reported as prominently as any win. Negative results are never deleted or hidden.

## 7. Leakage controls (L1–L8)

Update objective unlabeled only (L1); reliability features label-free, source-labels-only where used (L2); gate rule fully offline, meta trained on dev/source tasks only with saved IDs/seeds/ckpt (L3); steps/lr/radius/mode frozen here before eval, headline mode on DEV only (L4); split-conformal ε from dev/calibration residuals only (L5); held-out scored once — **this task does not run eval** (L6); telemetry carries no labels, guarded and scanned (L7); KGA benefit `B` from dev/calibration labels only, test side contributes label-free `Z` only (L8).

## 8. Stop condition

Deterministic seeds, environment metadata, and machine-readable label-free telemetry are in place. **STOP after smoke + pre-registration.** The locked held-out evaluation is **not** executed here — it awaits review and sign-off of this pre-registration.
