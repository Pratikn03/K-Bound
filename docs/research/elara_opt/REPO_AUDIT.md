# ELARA-Opt — PHASE 0 Repo Audit

**Branch:** `feature/elara-opt` (off `main` @ `a17d744`)
**Host:** macOS (Pratik-Ns-MacBook-Pro), Python `~/.venv_wilds/bin/python` = 3.12.13, torch 2.5.1 (CPU/MPS, **CUDA False**), numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0, wilds 2.0.0, torchvision 0.20.1, `kga` 0.1.0, `kbound_tta` 0.1.0 — all import cleanly.
**Scope:** locate existing TTA adapters, KGA/K-Bound, the 9 dataset runners, calibration-split logic, `research_lock` pre-registration, result manifests, theorem validators; specify the adapter API + dataset-protocol API; enumerate every label-leakage risk; give the integration plan and smoke commands. **No existing K-Bound/KGA artifact is read-modified by this task.**

> ⚠️ Working tree was already dirty when this task started (pre-existing, **unstaged**: 102 `D`, 44 `M`, 126 `??`, nothing in the index, no stash). Those changes are **not mine** and are left untouched; ELARA-Opt commits will `git add` only its own new files by explicit path.

---

## 1. Existing TTA adapters (the API ELARA-Opt slots behind)

### 1a. Faithful vision TTA (the relevant one)
- **Runner-facing module:** `experiments/kbound/wilds/tta_methods.py` (372 lines)
- **Packaged mirror (installed, importable):** `packaging/kbound-tta/src/kbound_tta/_tta.py` (393 lines), exported via `packaging/kbound-tta/src/kbound_tta/__init__.py` (`import kbound_tta as kb`).

Method bodies are "ported VERBATIM from `docs/research/kbound/scripts/cifar_tent_mps_v2.py`" (the validated reference harness). **A faithful SAR/SAM already exists** — `sar_adapt` implements Niu et al. ICLR-2023 SAM first/second step + reliable-sample selection (entropy `< E_0`) + entropy-EMA collapse reset. ELARA-Opt will **reuse** these primitives and **not** fake a SAR/SAM.

**Adapter contract (all candidates share this):**
```
tent_adapt(base, stream, steps, lr)                       -> (adapted_model, update_norm)
eata_adapt(base, stream, steps, lr, num_classes, ...)     -> (adapted_model, update_norm)
sar_adapt (base, stream, steps, lr, num_classes, ...)     -> (adapted_model, update_norm)   # faithful SAM
shot_adapt(base, stream, steps, lr)                       -> (adapted_model, update_norm)   # package only
```
- `base` = frozen `f0` (nn.Module); `stream` = list of normalized tensor batches on device; updates **only BN/LN affine params** (`_bn_affine_params`, `_clone_for_tta`); `update_norm` = L2 of the param delta (`_upd_norm`).
- Registry: `kbound_tta.TTA_METHODS = {"tent","eata","sar","shot"}`; dispatch `_adapt(method, base, stream, steps, lr, num_classes)`.

**Reusable label-free helpers (ELARA-Opt depends on these):**
`_entropy(p)`, `evidence_vector(f0, fa, x, num_classes, upd_norm)` → 11-dim `Z` (`EVIDENCE_NAMES` = pre/post entropy·conf·pbal, pbal_drop, entropy_drop, frac_highconf, marginal_KL, update_norm), `rich_evidence_vector(...)` → 6-dim (`disagreement, ent_gap, energy_shift, bn_kl, atc, conf_drop`; ATC uses **source** labels only), `bn_running_stats`, `bn_batch_stats`, `bn_stat_kl_drift`, `predict_logits`, `balanced_acc`, `run_candidate(method, mode, f0, adapt_stream, eval_x, eval_y_np, num_classes, steps, lr, ...)` (online/episodic; returns `aa_balanced, Z, upd_norm, preds, pa_pos`), `eval_frozen`.

### 1b. Score-head TTA variants (different track — multimodal fusion, NOT used by ELARA-Opt)
`src/uais/fusion/attention/baselines.py`: `TentScoreAdapter`, `EATAScoreAdapter`, `SARScoreAdapter`, `TTTPseudoLabelAdapter`; `reliability_boosted_fusion.py: _Candidate`. Noted to avoid confusion; ELARA-Opt targets the vision-TTA candidate API in §1a.

---

## 2. KGA / K-Bound implementation (what "certifies" the candidate)

- **`kga/` package** (pure numpy/scipy, torch-free) — the clean certify API:
  - `kga/kga.py: class KGA(alpha=0.1, method="ebern")` with `.evidence(calib, test)`, `.certify(scores=benefits, …)` / `.certify(adapt_risk=, freeze_risk=, calib_residuals=)`, `.decide()`, `.explain()`.
  - `kga/certificate.py`: `empirical_bernstein` (Maurer–Pontil LCB, Thm 3), `hoeffding`, `conformal_split`, `evalue_anytime` (Ville/Thm 3b); `class Certificate(.lower/.upper)`.
  - `kga/policy.py`: `class Decision(ADAPT/FREEZE/ABSTAIN)`, `decide(certificate, alpha)` (trichotomy, strict boundaries).
  - `kga/evidence.py`: `class Evidence`, `compute_evidence(...)` (KS drift, disagree, ess_frac…).
  - Contract: **ADAPT iff `Δ̂ − ε > 0`**, FREEZE iff `Δ̂ + ε < 0`, else ABSTAIN; false-adapt ≤ α (Thm 3). `README` + `tests/test_kga_package.py` carry the behavioural contract (empirical false-adapt ≤ α).
- **Pipeline analysis layer** (`sklearn`-based, used by the runners):
  - `experiments/kbound/wilds/analysis.py` and packaged `kbound_tta/_analysis.py`: `decide_kga(Z, B, alpha=0.10, …)` (LOO **GradientBoostingRegressor** B̂(Z) + split-conformal radius ε → ADAPT/FREEZE/ABSTAIN), `policy_metrics(dec, a0, aa, B, alpha)` (`beats_both` requires `false_adapt ≤ α`, not regret alone), `label_regime`, `detectability_analysis`, `agreement_matrix`, `multicandidate_route` (Thm 1A τ-residual, **label-free**, reuses `val_multicandidate_residual.py`), `smooth_drift_route` (Thm 1B; marked diagnostic/stub).

**How KGA "consumes a candidate":** the candidate produces `fa`. Per cell: (i) **label-free** `Z = evidence_vector(f0, fa, x_test, …)`; (ii) a **dev/calibration-labeled** benefit `B = balanced_acc(fa) − balanced_acc(f0)`; (iii) `decide_kga(Z, B)` (or `KGA.certify` on per-sample benefits) → ADAPT/FREEZE/ABSTAIN. **Test labels never enter the certificate** — only `Z` (label-free) does on the test side.

---

## 3. The 9-dataset runners + regimes (locked verdicts I must not invalidate)

Reconciled panel: `research_lock/KBOUND_6_DATASET_PANEL_v2.yaml`; verdicts: `research_lock/KBOUND_HEADLINE_FINDINGS.json`; CIs: `KBOUND_WIN_BOOTSTRAP_CIS.json`; capstone: `KBOUND_MIXED_STREAM_v1.json`/`v2.json`. Locked bar: estimator **gbr**, conformal **global**, **α=0.1**, success = dev-screen pass (or fixed adapter) ∧ held-out scored once ∧ **false-adapt ≤ α** ∧ beats both trivial policies.

| # | Dataset | Runner (real path) | Protocol lock | num_classes | Locked regime / verdict |
|---|---|---|---|---|---|
| 1 | CIFAR-10-C | `src/scripts/kbound/cifar10c_suite.py` + stress grid `experiments/kbound/results/stress_grid_multiseed_v1/` | `STRESS_GRID_MULTISEED_PROTOCOL_A_v1.yaml` | 10 | controlled corruption; multiseed **beats-both** (Tent/EATA) |
| 2 | ImageNet-C | `src/scripts/kbound/imagenette_c_suite.py` (**honest proxy:** Imagenette + official `imagecorruptions`; full ImageNet-C unavailable from host) | (in paper, not core six) | 10 (proxy) / 1000 (full) | scale/breadth; kept in paper, CIFAR-10-C is the core corruption bench |
| 3 | Office-Home | `experiments/kbound/officehome/run_officehome_kbound.py` (+`oh_data.py`,`oh_candidates.py`,`train_f0_officehome.py`) | `OFFICEHOME_PROTOCOL_M_v2.yaml` | 65 | **robust beats-both**, CI excludes 0 (primary clean win) |
| 4 | iWildCam | `experiments/kbound/wilds/run_iwildcam_kbound.py` (+`train_iwildcam_f0.py`,`analyze_iwildcam_kbound.py`) | `IWILDCAM_PROTOCOL_H_v2.yaml` | 182 | **no-harm Pareto**: dominates adapt, ties freeze |
| 5 | Camelyon17 | `experiments/kbound/wilds/run_camelyon17_kbound.py` (+`analyze_camelyon_kbound.py`) | `CAMELYON17_PROTOCOL_G_v1.yaml` | 2 | **robust beats-both**, CI excludes 0 (biomedical shift) |
| 6 | RxRx1 | `experiments/kbound/wilds/run_rxrx1_kbound.py` | `RXRX1_PROTOCOL_J_v1.yaml` | 1139 | demoted: single-seed win not robust; breadth only |
| 7 | ImageNet-R | `experiments/kbound/wilds/run_imagenetr_kbound.py` (`MaskedImageNetModel`, select indices) | `IMAGENETR_DIVERSE_PANEL_PROTOCOL_D_v1.yaml` | ~200 | **forced-abstention frontier** (undetectable harm AUC≈0.66) — NOT a win |
| 8 | CIFAR-10.1 | Protocol-K pipeline; data `data/raw/adbench_cv/CIFAR10_1.npz` + `experiments/kbound/cifar/resnet18_cifar.pt`; results `experiments/kbound/results/cifar101_protocol_K_v1/` | `CIFAR101_PROTOCOL_K_v1.yaml` | 10 | low-margin boundary case; NOT cross-seed beats-both |
| 9 | fMoW | `experiments/kbound/wilds/run_geoshift_kbound.py` (+`fmow_data.py`, `NUM_CLASSES=62`) | `FMOW_PROTOCOL_L_v1.yaml` | 62 | **honest null** (false-adapt 0.375); breadth |

Mixed-stream **capstone:** `experiments/kbound/poem_aetta/run_mixed_headtohead.py` (+`verify_headtohead.py`), manifest `research_lock/KBOUND_MIXED_STREAM_v1/v2.json`.

**Dataset-protocol API (shared shape across runners):** `make_model/load_f0 → f0`; build per-cell `(adapt_stream, eval_x, eval_y, source/calib)`; `run_candidate(...)` → `(aa, Z, upd_norm, preds, pa_pos)`; assemble `records=[{Z, B, a0, aa, preds,…}]`; `aggregate_single_candidate` / `aggregate_multicandidate` → `decide_kga` / `multicandidate_route` → `policy_metrics`; `build_manifest(...)` writes `experiments/kbound/results/<cell>/...json` (atomic dump, content-hash filenames).

---

## 4. Calibration-split logic + research_lock + manifests + validators

- **Calibration / dev / test split:** certificates are **source/dev-calibrated** (e.g. Camelyon "source train present patches for source-calibrated rich evidence"). `B` and split-conformal residual `ε` come from **dev/calibration labels only**; the **held-out test contributes only label-free `Z`**, scored **once**. Governing locks: `research_lock/primary_endpoints_v1.yaml` (`threshold_selection: validation_only`, `no_test_driven_tuning: true`, Holm-Bonferroni, effect+bootstrap-CI), `research_lock/statistical_policy_v1.md`, `research_lock/strongest_baseline_frozen_v1.json`, `research_lock/frozen_test_sets_v*.yaml` (note: v3/v4 + `dataset_registry_v*` govern the **separate** anomaly/multimodal track, not the KGA TTA panel).
- **research_lock pre-registration corpus:** per-dataset `*_PROTOCOL_*_v*.yaml` (CAMELYON17 G, IWILDCAM H, OFFICEHOME M, CIFAR101 K, RXRX1 J, FMOW L, IMAGENETR D, STRESS_GRID A, OFFICIAL_SAR_SCHEDULE E…), panel manifests (`KBOUND_6_DATASET_PANEL_v2`, `KBOUND_HEADLINE_FINDINGS.json`, `KBOUND_WIN_BOOTSTRAP_CIS.json`, `KBOUND_MIXED_STREAM_v*.json`), `protocol_registry_v1.yaml`, `claim_matrix_v1.csv`, `DECISIONS_v1.md`, `BASELINE_STATE_v1.md`. Existing ELARA locks (the **router**, not the optimizer): `ELARA_U_PROTOCOL_v1.yaml`, `ELARA_CHF_v1.yaml`, `ELARA_DEPLOY_v1/2/3.yaml`.
- **Result manifests:** `experiments/kbound/wilds/build_results_manifest.py`, `multiseed_paired_ci.py`, `per_condition_serialize.py`; locked JSON under `experiments/kbound/results/<dataset>_protocol_*/...`.
- **Theorem validators:** `experiments/kbound/theory_validation/val_*.py` — `val_thm1_lecam`, `val_thm2_lecam_finite_n`, `val_thm2_regret`, `val_thm3_evalue`, `val_thm5_multiclass`, `val_thm9prime_drift`, `val_knowability_dichotomy`, `val_knowability_capacity[_general]`, `val_multicandidate_residual`, `val_reach_unification`, `val_rademacher_router`, `val_benefit_frontier`, `val_agl`; plus `experiments/kbound/conj1_validator.py` + `results_conj1_validator.json`; `val_smooth_drift.py` is **absent** (Thm 1B route is a documented stub). KGA contract test: `tests/test_kga_package.py`. **None are touched by ELARA-Opt.**

---

## 5. Existing "ELARA" ≠ ELARA-Opt

`src/elara/` and `experiments/kbound/kga_elara_demo.py` define **ELARA = reliability-gated multimodal score *fusion*** (a parameter-free prediction/score router: `cw_fuse`, `relgate_fuse`). **ELARA-Opt is different by construction:** a genuine **parameter-update optimizer** (BN/LN affine updates with a reliability-gated objective mixture + frozen-KL anchor + trust region). It is **not** a prediction router and **not** a renamed KGA threshold. Naming kept distinct (`elara_opt`, modes `elara_uniform/elara_rule/elara_meta`) to avoid collision with the existing ELARA-U router locks.

---

## 6. Label-leakage risk register (every surface) + closure

| # | Surface | Risk | Closure |
|---|---|---|---|
| L1 | Update objective | using `y_test` in the loss | adapter takes only `stream` tensors; objectives = entropy / reliability-filtered entropy / aug-consistency / frozen-KL — all label-free. Unit test asserts no `y` arg path. |
| L2 | Reliability features / Z | test labels in a feature | features from `f0,fa,x` only; reuse label-free `evidence_vector`; `rich_evidence_vector`/ATC use **source** labels only (allowed), never test. |
| L3 | Gate g_phi | gate fit/selected on test benefit | `elara_rule` fully specified offline in lock; `elara_meta` trained **only on source/dev shift tasks** with saved data-IDs/seeds/checkpoint, strict no-overlap with eval conditions. |
| L4 | Hyperparam / candidate / headline-mode selection | tuning on held-out test | all of {steps, lr, radius, mode} frozen in `research_lock/elara_opt_protocol_v1.yaml` **before** eval; headline mode chosen on **dev** only (mirrors H_v2/M_v2 dev-lock). |
| L5 | Calibration ε | residual from test | split-conformal residual from **dev/calibration** benefits only (`decide_kga` calib path). |
| L6 | Reported metrics | peeking / multiple looks | held-out scored **once**; **this task STOPS before eval**. |
| L7 | Telemetry | dumping labels | telemetry schema excludes labels; `test_no_label_leakage.py` scans emitted telemetry keys/values for any label tensor. |
| L8 | KGA benefit `B` | computing `B` on test | `B` from dev/calibration labels only; test side contributes only label-free `Z` (existing pipeline invariant). |

---

## 7. Integration plan (additive — zero edits to validated files)

New, self-contained package **`experiments/kbound/elara_opt/`** importing validated primitives from `kbound_tta` (and `kga`), never modifying them:
- `elara_opt.py` — `ELARAOptAdapter` class + `elara_opt_adapt(base, stream, steps, lr, num_classes, *, mode, seed, telemetry_sink, ...) -> (adapted_model, update_norm, telemetry)`. Genuine update: per step combine objectives `{H, H_filtered, aug_consistency}` with gate weights `w=g_phi(reliability)` (softmax/temperature, nonneg) + `λ·KL(f0‖f_t)` frozen anchor; **trust-region** clip of the affine-param step to radius `r(reliability)`. Reuses `_clone_for_tta/_bn_affine_params/_entropy/_upd_norm`. Optional faithful SAM step delegated to the existing `sar_adapt` building blocks (referenced, never re-derived).
- `reliability.py` — label-free reliability features (entropy mean/var, conf stats, pred-class balance+drift, aug disagreement, frozen-vs-current divergence, BN/feature-stat drift, update norm, **inter-objective gradient cosine/conflict**, plus KGA's `evidence_vector` block). Named vector.
- `gate.py` — `g_phi`: `uniform` | `rule` (deterministic, lock-specified) | `meta` (tiny torch MLP; deterministic load/save).
- `modes.py` — `ELARA_MODES={"elara_uniform","elara_rule","elara_meta"}`; `EXTENDED_TTA_METHODS = {**kbound_tta.TTA_METHODS, ...elara...}` (shows the one-line registration without mutating the frozen package).
- `run_elara_candidate.py` — mirrors `run_candidate` for elara so KGA consumes the candidate (`evidence_vector`+`decide_kga`+`KGA.certify`); `run_smoke.py --dataset … --n … --seed …`.
- `telemetry.py` — JSONL schema/writer; deterministic candidate hash; label-free guard.
- `meta/` — saved meta-gate checkpoint + `training_data_ids.json` + seeds (dev/source tasks only).

Tests in **`tests/elara_opt/`**: `test_objectives.py`, `test_no_label_leakage.py`, `test_integration_batch.py`, `test_smoke_datasets.py` (parametrized over the 9 configs).

**Locked-run integration (documented, NOT done now):** add `elara_uniform/elara_rule/elara_meta` to a runner's candidate list / `TTA_METHODS`; one line, frozen in the lock.

---

## 8. Smoke-test commands (host)
```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
VENV=~/.venv_wilds/bin/python
PP=.:packaging/kbound-tta/src

# unit + integration + no-leakage
PYTHONPATH=$PP $VENV -m pytest tests/elara_opt -q

# per-dataset deterministic smoke (tiny n, CPU); KGA must emit a Decision + telemetry
for D in cifar10c imagenet_c officehome iwildcam camelyon17 rxrx1 imagenet_r cifar101 fmow; do
  PYTHONPATH=$PP $VENV experiments/kbound/elara_opt/run_smoke.py --dataset $D --n 16 --seed 0 --modes elara_uniform,elara_rule,elara_meta
done
```
Per-dataset smoke uses a small deterministic BN-CNN stand-in sized to each runner's `(num_classes, 3×H×W)` (real `resnet18_cifar.pt`+tiny cell optionally for CIFAR), reusing the faithful adapter/evidence primitives — **integration mechanics only, explicitly not a performance result**.

**STOP after smoke + pre-registration. The locked held-out evaluation is NOT run in this task.**
