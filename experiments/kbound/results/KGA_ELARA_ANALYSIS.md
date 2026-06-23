# KGA ↔ ELARA: Connection Map + Dataset Inventory

**Analysis date:** 2026-06-20  **Mode:** READ-ONLY (this file is the only artifact written; nothing in the repo was modified).
**Repo:** repository root
**Scope verified against disk:** `kga/`, `experiments/kbound/`, `docs/research/kbound/`, `experiments/elara_u/`, `experiments/fusion/`, `src/elara/`, `data/raw/`.
**Purpose:** grounded input for paper finalization — what data each project actually has on disk, how the KGA router and the ELARA reliability gate are connected in code and in theory, and why they are correctly two papers.

> **Reading note on counts.** "Has results on disk" means a results/analysis JSON (not just a launch log or raw data) exists. Each row cites the path. Where the verdict contradicts the rough expectations in the task brief, it is flagged **⚠ DISCREPANCY**.

---

## 1. DATASET INVENTORY

### 1a. K-Bound — TTA / distribution-shift benchmark panel (the "KGA router" experiments)

KGA = label-free **ADAPT / FREEZE / ABSTAIN** routing of a test-time-adaptation (TTA) adapter, gated by the benefit-sign certificate. Regime vocabulary: **beats-both** (KGA strictly beats both always-adapt and always-freeze after Holm) / **helpful-leaning** (adapting is safe, KGA ties always-adapt) / **harmful-dominated** (adapting is catastrophic, KGA ties always-freeze) / **mixed-undetectable** (harm present but signal too weak to certify) / **null**.

| # | Dataset | Domain / shift type | Raw data on disk | Results manifest/JSON on disk | Regime / verdict | Seeds |
|---|---|---|---|---|---|---|
| 1 | **CIFAR-10-C** (stress grid) | Synthetic corruption (Hendrycks) | `experiments/kbound/data/` (corruptions applied to CIFAR-10 test) | `results/stress_grid_multiseed_v1/LOCKED_ANALYSIS_RESULTS.json` (+ `LOCKED_ANALYSIS_FINDINGS.md`) | **beats-both WIN** — tent & eata beat both trivials after Holm (tent vs freeze −0.1225, vs adapt −0.0064; eata similar); SAR ties always-adapt (harmful rate ≤ p*≈0.1). **p\* law confirmed: single-threshold separable.** | 5 (0–4), 432 conditions |
| 2 | **CIFAR-10-C** (65-cell suite) | Synthetic corruption | `experiments/kbound/data/` | `results/cifar10c_suite_results.json` (+ `cifar_tent_results.json`, `cifar_tent_online_results.json`) | **helpful (synthetic win)** — KGA tracks the best adapter: regret-to-oracle KGA **0.0032** vs frozen **0.217** (resnet18, 65 cells, N=800) | 1 config |
| 3 | **ImageNet-C** | Synthetic corruption | `experiments/kbound/data/imagenet-c/` | `results/imagenetc_official_sar_E_v1_s{0,1,2}/`, `imagenetc_noise*`, `imagenetc_noiseblur/`, `imagenetc_noise_vit/` | **helpful (synthetic win)** — official SAR protocol-E panel run; corruption regime mirrors CIFAR-10-C (KGA tracks best adapter) | 3 (s0–s2) + variant runs |
| 4 | **CIFAR-10.1** | Natural reproduction shift | via WILDS/torchvision cache | `results/cifar101_multiseed_v1/pooled_summary.json` | **~null / mixed** — beats-both count tent 0/5, eata 1/5, sar 0/5; regrets tiny, KGA ≈ ties. Harmful rate tent 0.675 / eata 0.50 / sar 0.05 | 5 (0–4), 24 conditions |
| 5 | **ImageNet-R** | Rendition shift | `experiments/kbound/data/imagenet-r/` | `results/imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json` | **helpful-leaning, no beats-both** — `beats_both_by_candidate` = **all false** across 10 architectures; harmful base rate ≈ 0 → adapting is safe, so nothing to gate (KGA ties always-adapt) | 3 (0–2) × 10 archs |
| 6 | **Camelyon17** (WILDS) | Histopathology hospital shift | `experiments/kbound/data/wilds/` | `results/camelyon17_fullscale_B_v2/MULTISEED_ANALYSIS_RESULTS.json`; `camelyon17_fullscale_B_v1/LOCKED_B_FINDINGS.md` | **mixed — FAILS pre-registered beats-both** — `beats_both` tent/eata/sar all false (pooled); KGA cuts freeze-regret massively but adapt usually best; conformal ε unstable (CV 0.73). B_v1 verdict: "**FAILS** per success_criteria_stated_in_advance" | 3 (0–2); B_v1 had 5 model seeds |
| 7 | **RxRx1** (WILDS) | Cellular microscopy batch shift | `experiments/kbound/data/wilds/` | `results/rxrx1_protocol_J_v1/VERIFIED_FINDINGS.md` + `analyze_F_results.json`; source `rxrx1_protocol_c_9plus_modelseed{0,1,2}/` | **harmful-dominated** — harmful base rate **100%**, SAR catastrophic (adapt regret 0.25); KGA commits FREEZE everywhere → regret 0.0000 = freeze. "FREEZE-ORACLE AUDIT PASS, **not** a beats-both win" (1,139-class scale) | model seeds 0–2; dev 0–4 / test 5–9 |
| 8 | **iWildCam** (WILDS) | Camera-trap location shift | `experiments/kbound/data/wilds/` | `results/iwildcam_protocol_H_v2/VERIFIED_FINDINGS.json` + `protocol_result.json` | **beats-both WIN** — `beats_both: true` (tent_episodic): KGA regret **0.0037** vs adapt **0.103** vs freeze **0.0041**; false-adapt 0.0 | dev-screen + held-out (protocol H) |
| 9 | **Office-Home** | Domain adaptation (4 domains) | `experiments/kbound/data/office_home/` | `results/officehome_protocol_M_v2/VERIFIED_FINDINGS.json` + `protocol_result.json` (many `officehome_*` runs) | **beats-both WIN** — `beats_both: true` (sar_online_aggressive): KGA regret **0.0022** vs adapt **0.047** vs freeze **0.0158**; false-adapt 0.0 | dev-screen + held-out (protocol M) |
| 10 | **fMoW** (WILDS) | Satellite, temporal/region shift | `experiments/kbound/data/wilds/` | `results/fmow_protocol_L_v1/`, `fmow_protocol_L_val_analyze/analyze_F_results.json`, `fmow_protocol_L_{val,test,dev}/` | **~null / not headline** — test regret KGA 0.0095 ≈ freeze 0.0094 < adapt 0.0157; commit 0.25; false-adapt high on the few adapts. *(Extra beyond brief's expected list.)* | dev 0–1 / test 2–4 (5 present) |
| 11 | **PovertyMap** (WILDS) | Satellite regression (poverty) | `experiments/kbound/data/wilds/` | `results/poverty_protocol_L_dev/VERIFIED_FINDINGS.json` (+ `_partial.json`, `result_9044c252.json`) | **mixed-undetectable (null, screened out)** — verdict `dev-screen-stop`; classification literally **"mixed+undetectable"**; harm_AUC 0.637 < 0.65 gate → held-out **not run** per pre-registration. *(Extra beyond brief's expected list.)* | dev screen, 108 records |
| — | **ACDC** | Segmentation, weather/night shift | `experiments/kbound/data/{acdc/, acdc_zips/}` (raw present) | **NONE** — only harness code: `experiments/kbound/acdc/{run_acdc_kbound.py, seg_certificate.py, seg_tta_methods.py, acdc_data.py}` | **present-but-UNRUN** — raw data + code on disk, no results JSON | — |

**K-Bound TTA-panel count: 11 datasets in scope — 10 with results on disk, 1 (ACDC) present-but-unrun.**
Of the brief's 9 expected: 8 have results (CIFAR-10-C, ImageNet-C, CIFAR-10.1, ImageNet-R, Camelyon17, RxRx1, iWildCam, Office-Home); **ACDC is code-only/unrun**; **fMoW and PovertyMap are two extras** found on disk that were not in the brief.

**Beats-both winners on disk:** CIFAR-10-C (synthetic), **iWildCam, Office-Home** (real). Synthetic-corruption wins: CIFAR-10-C, ImageNet-C. Everything else is helpful-leaning / harmful-dominated / mixed-undetectable / null — i.e. honest non-wins that map the frontier.

#### K-Bound controlled "knowability" suite (the paper's actual headline evidence — separate from the TTA panel)

The K-Bound paper's headline claims C1–C7 are **not** built on the WILDS panel above; they are built on a controlled/synthetic certificate suite plus the 123-task ADBench frontier (`docs/research/kbound/manifests/claim_result_map.csv`, `used_in_paper.csv`):

| Claim | Experiment | Result file (under `experiments/kbound/results/`) | Status |
|---|---|---|---|
| C1 certificate safe on clean suite | clean_suite | `knowability_results.json` | used |
| C2 KGA refuses harmful fusion (~11× regret cut) | harmful_fusion | `kbound_harmful_results.json` | used |
| C3 mixed regime: beats freeze, ties adapt | mixed_regime | `mixed_regime_results.json` | used |
| C4 8-seed paired t-tests | rigor | `rigor_multiseed.json` | used |
| C5 non-identifiability witness → 100% abstain | witness | `witness_clean.json` | used |
| C6 regression covariate shift | regression | `regression_covariate.json` | used |
| C7 disagreement = top evidence feature | ablations | `ablations.json` | used |
| C8 deep-TTA CIFAR/Tent "beats both" | cifar_tent | `cifar_tent_results.json` | verify_before_claim |
| C9 TTA collapse probe | tta_collapse | `tta_collapse_results.json` | verify_before_claim |
| C10 **multimodal instantiation (MVTec-3D +0.21)** | multimodal | `../../elara_u/multimodal_reliability_results_mvtec3d.json` | **corroborating — imported from ELARA** |

> **⚠ DISCREPANCY (manifest gap).** `docs/research/kbound/manifests/data_inventory.csv` lists only the `data/raw/*` tabular/anomaly "knowability" datasets (adbench×5, baf, cyber, fraud, realiad_d3, mvtec3d, …). It does **not** list the TTA vision benchmarks (CIFAR-10-C, ImageNet-C/R, CIFAR-10.1, the WILDS sets, Office-Home, ACDC), which live under `experiments/kbound/data/` and `data/wilds/`. The data manifest is therefore an incomplete inventory of what the repo actually ran.

### 1b. ELARA — reliability-gate panel (the "which-modality-to-trust" experiments)

ELARA's reliability gate = label-free **modality-reliability routing** for multimodal anomaly detection (protocol **D23**). Baselines: `equal_weight`, `stale_auto_select`, `no_reliability`. Metric: mean AUROC, per-category bootstrap 95% CI. "Validated" = the gate's gain CI excludes 0 in the failure regime.

| # | Dataset | Domain / modalities | Raw data on disk | Results JSON on disk (`experiments/elara_u/`) | Regime / verdict | Categories / stats |
|---|---|---|---|---|---|---|
| 1 | **Real-IAD-D3** (injected) | Industrial AD: RGB + pseudo-3D + point cloud | `data/raw/realiad_d3/` (259 GB) | `multimodal_reliability_results.json` | **helps under injected failure** — gate 0.767 vs no_reliability 0.519 (**+0.248**, CI [0.168, 0.323], pass); clean regime ≈ neutral | 15 cats; `reliability_validated: true` |
| 2 | **Real-IAD-D3-NatDeg** (natural) | Same, **natural** degradation | `data/raw/realiad_d3/` | `multimodal_reliability_results_realiad_natdeg.json` | **NULL — the decisive negative** — gate vs no_reliability **+0.031, n.s.** (CI [−0.003, 0.089]); all hypotheses fail | 16 cats; `reliability_validated: **false**` |
| 3 | **MVTec-3D** (injected) | Industrial AD: RGB + depth | `data/raw/mvtec3d/` (26 GB) | `multimodal_reliability_results_mvtec3d.json` | **helps under injected failure** — gate 0.792 vs no_reliability 0.581 (**+0.211**, CI [0.180, 0.244]). *This is the "+0.21" cited as K-Bound C10.* | 7 cats; validated true |
| 4 | **3D-ADAM** (injected) | Industrial AD: RGB + depth | `data/raw/3d_adam_anomalib/` (6.5 GB) | `multimodal_reliability_results_3d_adam.json` | **helps under injected failure** — gate 0.748 vs no_reliability 0.630 (**+0.118**, CI [0.038, 0.213]) | 12 cats; validated true |
| 5 | **MulSen-AD** (injected) | Multi-sensor AD: RGB + infrared + point cloud | `data/raw/mulsen_ad/` (19 GB) | `multimodal_reliability_results_mulsen.json` | **helps under injected failure (+0.163), but clean regime HURTS** — clean gate 0.883 vs no_reliability 0.931 (−0.048) | 15 cats; validated true (failure regime) |
| 6 | **UNSW-NB15 (cyber) + credit-card** | Network-intrusion + fraud event views (tabular) | `data/raw/cyber/`, `data/raw/fraud/` | `natural_shift_results.json` (D22); fusion side `experiments/fusion/unsw_*` | **NULL on natural shift** — drift_stack does **not** beat plain_stack (−0.050, CI excludes 0 in the *wrong* direction); also ELARA-U "Cyber" main-suite family | 7 tasks (6 UNSW attacks + creditcard) |
| 7 | **OpenOOD** | Image OOD benchmark | **NONE** | **NONE** — only `src/scripts/elara_u/ingest_openood.py` | **PLANNED / unrun** — explicitly "remain planned" in `ELARA_U_PAPER_v0.tex` (Table `tab:manifest`, l.290) | — |
| 8 | **MVTec-AD-2** | Industrial AD (RGB + pseudo-depth) | **NONE** | **NONE** — only `src/scripts/elara_u/ingest_mvtec_ad_2.py` | **PLANNED / unrun** — "Planned" in manifest table; "Scaffolded only" in `DATASET_USE_MATRIX.md` l.63 | — |

**ELARA reliability-gate panel count: 8 datasets in scope — 6 with results on disk (Real-IAD-D3 in two regimes, MVTec-3D, 3D-ADAM, MulSen-AD, UNSW/credit-card), 2 planned/unrun (OpenOOD, MVTec-AD-2).**
All four brief-expected multimodal sets (Real-IAD-D3, MVTec-3D, 3D-ADAM, MulSen-AD) **confirmed run**; UNSW-cyber **confirmed run**; OpenOOD + MVTec-AD-2 **confirmed planned, not run** — exactly as the brief anticipated.

> **Broader ELARA-U context (not the gate, for honesty):** the ELARA-U *paper* is a 123-task label-free **stacking / model-selection** result across 5 main families (tabular, image-OOD, text, cyber, fraud) + boundary families (time-series NAB+SMD, multimodal/3D D23, sealed D24, independent D27). The reliability gate above is one *family* (D23) inside it. The gate **also has an honest negative on i.i.d. data** (`statistical_audit.json` → `family_B_reliability_ablation`: Δ=−0.037, `reliability_hurts=true`). Manifest: `experiments/elara_u/manifest.json` (10 files, sha256-locked, commit `916fa2d6`).

---

## 2. KGA ↔ ELARA CONNECTION (grounded in code + theory)

### 2a. Shared infrastructure — they share the **certificate and the theory registry**, by literal vendoring

The single source of truth is documented in `docs/research/kbound/ELARA_KGA_MERGE_PLAN.md`:

- **Certificate code is shared and ELARA-derived.** `kga/certificate.py` (the canonical, typed, torch-free package: `certificate.py`, `evidence.py`, `kga.py`, `policy.py`, `cli.py`, `py.typed`) states in its own docstring that it was *"vendored_from_elara/certification/switching_certificate.py."* The shared math is the **Maurer–Pontil (2009) empirical-Bernstein LCB**, present byte-for-byte in three trees:
  - `kga/certificate.py::empirical_bernstein`
  - `src/elara/certification/switching_certificate.py::empirical_bernstein_lcb`
  - `docs/research/kbound/kbound_pkg/kbound/certificate.py::empirical_bernstein_lcb`
  Verified numerically identical over 3000 randomized inputs (max abs diff **1.8e-15**; `ELARA_KGA_MERGE_PLAN.md` §4). As of the 2026-06-20 merge, `src/elara/certification/switching_certificate.py` **delegates** its normal-path LCB to `kga.certificate.empirical_bernstein` — so the formula now lives in one place and ELARA imports K-Bound's copy.
- **The theory registry is shared and ELARA-origin.** K-Bound carries a verbatim snapshot at `experiments/kbound/vendored_from_elara/theory/` = `theorem_registry.py` + `t1_impossibility.py`, `t2_mixture_entropy.py`, `t3_mean_gate_miss.py`, `t6_sequential_detection.py`, `t8_certified_heterogeneous_fusion.py`, `t9_clean_transfer_ceiling.py`, `gdr_minimax.py`, `novel_theorem_bounds.py`. The live versions are ELARA's (`src/elara/theory/`, `src/elara/family_b/`). K-Bound's impossibility (Thm 1/2) and ELARA's T1 impossibility are the same lineage.
- **Drift detectors shared:** `experiments/kbound/vendored_from_elara/drift/{drift_vision,drift_tabular,drift_time_series,drift_nlp}.py` — ELARA's drift feature extractors, vendored into K-Bound.

**They do NOT share the TTA method implementations.** Tent / EATA / SAR / TTT (and the AETTA + POEM label-free *gates* used as KGA's competitors) live **only on the K-Bound side**: `packaging/kbound-tta/src/kbound_tta/_tta.py`, `src/scripts/kbound/{cifar_tent_mps.py, cifar10c_suite.py, imagenette_c_suite.py, tta_collapse_experiment.py}`, `docs/research/kbound/scripts/{run_wilds_camelyon17.py, cifar_tent_*}`, `experiments/kbound/poem_aetta/baselines.py` (faithful POEM/AETTA ports), and `experiments/kbound/acdc/seg_tta_methods.py`. `src/elara/` contains **no** TTA code. This is the role split: **K-Bound *studies* TTA** (TTA adapters are the object being gated; KGA decides when to run them); **ELARA *uses* adapters/detectors as baselines** and gates *modalities*, not TTA updates.

**They do NOT share the "Family A–D" labels.** "Family A–D" is **ELARA-specific** and appears in two distinct ELARA senses — (i) *data-regime families* (`DATASET_USE_MATRIX.md`, `RESEARCH_OVERVIEW_AND_RATING.md`, `docs/research/phase3/GATE_DECISION_RULE.md`: Family A = derived-view vision, Family B = label-aligned/coherent-collapse stress, Family C = exploratory, Family D = naturally-paired RGB-D clean transfer) and (ii) *evidence-tier families* in `docs/research/audit/STATISTICAL_ANALYSIS_POLICY.md` (Family A audited reanalysis … Family D locked confirmatory). **No K-Bound file uses Family A–D**; K-Bound uses regime words (helpful / harmful / mixed) and candidate names (tent/eata/sar). So this label set is *not* shared.

### 2b. Conceptual link — ELARA's gate is the **multimodal instance of K-Bound's benefit-sign frontier**

Both systems are **label-free routing gated by a certificate**, and both are bounded by the *same* limit: routing helps only when there is an **independent, detectable, transferable signal of failure**.

**K-Bound side — the limit, stated as theorem.** The non-identifiability core (`kbound_fulltext.txt` l.240–266, Thm 1):
> "… with (i) Law(Z | P_T1) = Law(Z | P_T2) and (ii) sign Δ1 = − sign Δ2 ≠ 0. For any such pair, Law(g) is identical across the two worlds; hence no label-free rule commits to the benefit-maximizing action in both, and the action minimizing the worst-case committal regret … is **abstain**."

Made quantitative as the **knowability frontier** (Thm 18, l.1422–1431), with the per-instance margin `κ_α(z) := |Δ(z)| − 2 ε_α(z)` ("the true benefit clears twice the certified uncertainty"):
> "(i) (Achievability.) … FA ≤ α, FF ≤ α, C ≥ P[κ_α(Z) > 0] − α. (ii) (Converse.) … Any label-free rule whose wrong-commit probability is at most α … commits on a (δ,t)-ambiguous pair with probability at most **2α + t**. … for observationally equivalent pairs (t = 0), ambiguous evidence can be covered with probability at most 2α."

where `FA = P[adapt ∧ Δ ≤ 0]`, `FF = P[freeze ∧ Δ ≥ 0]` — i.e. the **benefit-sign frontier**. The phase diagram (Fig. 2, l.257–263): KGA commits "only outside the central **unknowable wedge**, where the certificate radius exceeds the estimated benefit (ε > |Δ|)." Crucially, on the helpful-dominated 123-task archive "|Δ| is small relative to 2ε_α, so the margin lower bound is loose (trivial) there" (l.1451) — **the certificate cannot certify a win when the signal is below the noise.**

**ELARA side — the same limit, found empirically.** `ELARA_U_PAPER_v0.tex` (l.48–106) states the gate's three-part finding verbatim:
> "**(i) It helps when an independent modality carries a *detectable* drift [signal]** … when the deployment-best modality fails at test … **(ii)** [on i.i.d.] the reliability gate does not improve a plain per-task stack … **(iii) It does not yet hold under natural degradation.** Our one real-degradation test is a negative (**+0.031, n.s., versus +0.248 injected**) … a reliability win confined to **independent and detectable** modality failure, and an honest, unresolved **injected-to-natural gap**."

And the mechanism (`docs/research/phase3/GATE_DECISION_RULE.md`): the gate helps under "coherent collapse" because "the reliability signal is **tightly clustered**," and is null/hurts under "legitimate heterogeneity" because "a category mixture produces a **dispersed** reliability signal that the global threshold misreads as drift."

**The mapping is exact:**

| K-Bound (TTA benefit-sign) | ELARA (multimodal reliability) |
|---|---|
| Certificate margin `κ_α = |Δ| − 2ε_α > 0` (knowable) | Modality drift signal **detectable** above noise |
| Inside the "unknowable wedge" (ε > \|Δ\|) → **ABSTAIN** | Signal **dispersed/entangled** → gate adds nothing |
| Converse: observationally-equivalent worlds (t=0) cap coverage at 2α | **Natural degradation** can't be told from clean → gate null (+0.031 n.s.) |
| Injected/synthetic corruption = high evidence separability → certifiable win (CIFAR-10-C beats-both) | **Injected** independent-modality failure = high separability → gate win (+0.21…+0.25) |

So ELARA's empirical "injected works, natural fails" is precisely the multimodal realization of K-Bound's frontier: injected failure manufactures the *independent, detectable, transferable* signal the certificate needs; natural degradation does not, and lands inside the unknowable wedge. The shared theoretical residue is **Conjecture 1**.

**Conjecture 1 — and a naming caution.** The paper's Conjecture 1 (`kbound_fulltext.txt` l.501–505) is **label-free bracketing**:
> "What remains open is the label-free bracketing of that comparison (p_a vs p_0 for K ≥ 3 … or the sign of E[(f0 − f_a)Y | D] for regression) without target labels — **a reliability-model assumption, as in the binary case.**"

Its **conditional resolution** is exactly your "margin-monotone class + one-bit supplement": `experiments/kbound/conj1_validator.py` backs the *Definition* of the **margin-monotone relative-calibration class `C_mono`** and the *Theorem* that "`C_mono` is, under general position, the **weakest falsifiable class admitting a ONE-bit benefit-sign certificate**" (`results_conj1_validator.json` → `all_passed: true`, `check_D.one_bit_recovery_rate: 1.0`, `check_C.unsound: true` for the non-identifiability witness). The general-position case is genuinely **OPEN** (`COMPLETION_STATUS_2026-06-19.md`; `val_conj1_genpos.py` pins the counterexample). ELARA's "needs an independent, detectable signal" is the multimodal reading of "needs the one-bit reliability supplement."
> **⚠ DISCREPANCY (label collision).** `CLAIMS_CALIBRATION.md` calls a *different* statement "Conjecture 1" — the **empirical p\*-law** ("a single harmful-fraction threshold separates beats-both from not"), confirmed only on the CIFAR grid. That is the regime-separability conjecture from `wilds/multiseed_paired_ci.py`, **not** the paper's label-free-bracketing Conjecture 1. Two different open problems share the number "1" across the theory paper vs the experiment ledger — worth disambiguating before submission.

### 2c. Where they diverge — and why they are correctly two papers

| Axis | K-Bound (KGA) | ELARA (reliability gate) |
|---|---|---|
| Problem | *When* to apply a TTA update (ADAPT/FREEZE/ABSTAIN) | *Which modality* to trust in multimodal anomaly fusion |
| Decision object | One model, sequential TTA adapter (tent/eata/sar) | Several modality detectors, one fused score |
| Metric | Committal **regret** vs oracle policy; FA/FF ≤ α | **AUROC** uplift vs equal-weight / auto-select / no-reliability |
| Headline evidence | Controlled knowability suite + 123-task frontier + CIFAR-10-C beats-both | 123-task stacking (+0.036/+0.075; sealed +0.016; independent +0.111) + D23 gate |
| Contribution shape | A **theorem** (frontier + impossibility + finite-n Le Cam) with a deployed certificate | An **empirical taxonomy** of when fusion-reliability routing helps vs hurts |
| Shared spine | empirical-Bernstein certificate (`kga/`), theory registry T1–T9, drift detectors — **same code, different application** | |

They are the same *idea* (certificate-gated label-free routing) applied to two different decision problems with different objects, metrics, and evidence. Folding them together would blur a clean theorem (K-Bound) into an empirical multimodal study (ELARA) and inflate the dataset count without strengthening either claim.

---

## 3. FINALIZATION IMPLICATIONS

**Cleanest honest K-Bound panel.** Lead with the controlled certificate suite + the frontier theorem (C1–C7, all `used`), which is where the math actually bites, then show the TTA panel as a **dichotomy demonstration**, not a leaderboard:

1. **Synthetic-corruption win (knowable regime):** CIFAR-10-C stress grid — the locked, multi-seed **beats-both** result (`stress_grid_multiseed_v1`), p\*-law confirmed; ImageNet-C as the second corruption point.
2. **Two real beats-both wins:** iWildCam (tent_episodic) and Office-Home (sar_online) — these are the strongest natural-data evidence and should be foregrounded.
3. **The frontier's other corners, reported as honest non-wins** (this is the contribution, not a weakness): RxRx1 = harmful-dominated (damage-prevention/insurance); Camelyon17 = mixed, **fails** its own pre-registered beats-both bar; ImageNet-R/CIFAR-10.1 = adapt-is-safe / null; PovertyMap = mixed-undetectable (screened out, "STOP"); fMoW = null.
4. **Drop or footnote:** ACDC (no results — present-but-unrun; either run it or cut it); C8/C9 are still `verify_before_claim` in the manifest; the two GPU runs (ImageNet-R, Camelyon17) that `CLAIMS_CALIBRATION.md`/`COMPLETION_STATUS` (2026-06-19) listed as PENDING-GPU are **now complete on disk** (their `MULTISEED_ANALYSIS_RESULTS.json` exist as of 2026-06-20) — update those ledgers, and note both came back as **honest non-wins**, which is fine and on-narrative.

**ELARA should stay a separate paper — do not fold it in to pad the count.** ELARA's reliability gate is a different problem (multimodal anomaly reliability), different metric (AUROC), and its honest center of gravity is an *empirical* finding with a *negative* (injected works, natural fails). It enters K-Bound correctly and minimally as **one corroborating appendix row (C10, MVTec-3D +0.21)** — the multimodal instantiation of the frontier — and nothing more. Importing ELARA's full D23/stacking suite into K-Bound would (a) mix a theorem-driven paper with an empirically-driven one, (b) double-count the shared certificate as if it were two methods, and (c) attach ELARA's unresolved injected-to-natural gap to K-Bound's headline. Keep the shared spine (the `kga/` certificate + T1–T9 registry) cited across both; keep the claims separate.

---

### Discrepancies vs prior notes (summary)
1. **TTA datasets absent from the data manifest.** `manifests/data_inventory.csv` omits every TTA/WILDS vision benchmark that actually ran (they live under `experiments/kbound/data/`, `data/wilds/`).
2. **"Conjecture 1" names two different open problems** — label-free bracketing (paper) vs the empirical p\*-law (experiment ledger / `CLAIMS_CALIBRATION.md`).
3. **PENDING-GPU runs are now complete.** ImageNet-R (`imagenetr_protocol_d_multiseed_v1`) and Camelyon17 (`camelyon17_fullscale_B_v2`) have `MULTISEED_ANALYSIS_RESULTS.json` on disk; the 2026-06-19 status docs predate them. Both are honest non-wins (no beats-both).
4. **Two K-Bound datasets beyond the brief's list:** fMoW and PovertyMap (both null/mixed). **ACDC is code-only / unrun**, not a results-bearing dataset.
5. **Brief's ELARA expectations all confirmed:** Real-IAD-D3, MVTec-3D, 3D-ADAM, MulSen-AD, UNSW-cyber run; OpenOOD + MVTec-AD-2 planned/unrun.
