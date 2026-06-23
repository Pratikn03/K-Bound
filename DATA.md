# DATA — pipeline, provenance, integrity, and the hermetic smoke path

This document describes the K-Bound data pipeline: how raw datasets flow into the
score archives that the K-Bound / KGA experiments consume, where each dataset
comes from, how to verify input integrity, and how to run a **zero-dependency,
CPU-only smoke** that reproduces the algorithm with **no external data at all**.

K-Bound studies *when label-free test-time adaptation is knowable*. Its decision
algorithm is **KGA** (Knowability-Guided Adaptation), implemented in the
importable [`kga/`](kga/) package: from label-free evidence `Z` and a
finite-sample certificate `Δ̂ ± ε` it returns **ADAPT / FREEZE / ABSTAIN** with a
false-adapt probability bounded by `α` (Theorem 3).

---

## 1. Pipeline DAG

```
                         ┌─────────────────────────────────────────────┐
                         │  RAW datasets  (data/raw/…)                   │
                         │  ADBench tabular, credit-card fraud, UNSW-NB15│
                         │  cyber, Enron (NLP), CIFAR-10/-C, Imagenette, │
                         │  HAR, NAB, SMD, TSB-AD, MVTec/RealIAD (images)│
                         └───────────────┬─────────────────────────────┘
                                         │  loaders: src/uais/data/*.py
                                         │  (load_fraud_data, load_cyber_data,
                                         │   load_behavior_data, load_datasets,
                                         │   download_nlp_vision, …)
                                         ▼
                         ┌─────────────────────────────────────────────┐
                         │  INTERIM  (cleaned / split frames)            │
                         │  train/val/test via                          │
                         │  src/uais/data/split_data.py                 │
                         └───────────────┬─────────────────────────────┘
                                         │  fit base detectors per task
                                         │  (ECOD/COPOD/IForest/LOF/KNN/OCSVM
                                         │   for tabular; deep features for images)
                                         ▼
                         ┌─────────────────────────────────────────────┐
                         │  PROCESSED  (per-detector anomaly scores)     │
                         └───────────────┬─────────────────────────────┘
                                         │  serialize one .npz per task
                                         ▼
            ┌────────────────────────────────────────────────────────────────┐
            │  SCORE ARCHIVE  experiments/elara_u/score_archive/*.npz          │
            │  123 tasks, each: Sval/yval/Stest/ytest/det_names/val_auc/domain │
            │  (POINTER artifact — see manifests/score_cache_manifest.csv;     │
            │   local-only, not committed)                                     │
            └───────────────┬────────────────────────────────────────────────┘
                            │  consumed (label-free Z + benefit B) by:
                            ▼
            ┌────────────────────────────────────────────────────────────────┐
            │  K-BOUND EXPERIMENTS  src/scripts/kbound/                        │
            │   • knowability_experiment.py   (trichotomy on 123 real tasks)   │
            │   • mixed_regime_experiment.py  (clean/detectable/covert regimes)│
            │   • kbound_full_experiments.py, kbound_harmful_regime.py, …      │
            │  → results in experiments/kbound/results/ ; figures under        │
            │    docs/research/kbound/figures/                                 │
            └────────────────────────────────────────────────────────────────┘

            ┌────────────────────────────────────────────────────────────────┐
            │  HERMETIC SMOKE  (no raw data, no archive, no torch)            │
            │   make_synth_archive.py  →  experiments/kbound/_smoke/…          │
            │      writes a tiny SYNTHETIC archive with the SAME schema        │
            │   smoke_trichotomy.py    →  runs the REAL kga package over it    │
            │      asserts helpful→ADAPT, harmful→FREEZE, unknowable→ABSTAIN   │
            │   scripts/smoke_kbound.sh  (one-command wrapper, <60s, CPU)      │
            └────────────────────────────────────────────────────────────────┘
```

The hermetic smoke is a *drop-in substitute* for the score-archive stage: it
emits the identical `.npz` schema so the same `kga` evidence → certificate →
decision pipeline runs unchanged, but every byte is synthetic and seeded.

---

## 2. Score-archive schema (the K-Bound contract)

Each task is one `.npz` file under `experiments/elara_u/score_archive/`, uniform
across all 123 tasks (verified against the loaders in
`src/scripts/kbound/knowability_experiment.py` and `mixed_regime_experiment.py`):

| key         | shape            | dtype   | meaning                                            |
|-------------|------------------|---------|----------------------------------------------------|
| `Sval`      | `(n_val, n_det)` | float64 | validation-set anomaly scores (one column/detector)|
| `yval`      | `(n_val,)`       | int64   | validation labels (0 = normal, 1 = anomaly)        |
| `Stest`     | `(n_test, n_det)`| float64 | test-set anomaly scores (unlabelled at decision)   |
| `ytest`     | `(n_test,)`      | int64   | test labels — used **only** for oracle evaluation  |
| `det_names` | `(n_det,)`       | str     | detector names, e.g. ECOD/COPOD/IForest/LOF/KNN/OCSVM |
| `val_auc`   | `(n_det,)`       | float64 | per-detector validation AUC (selects `f0`)         |
| `domain`    | scalar           | str     | dataset/domain tag (e.g. `adbench`, `fraud`)       |

The decision setup: `f0` (FREEZE candidate) is the best-validation-AUC detector,
`f0 = argmax(val_auc)`; `fa` (ADAPT candidate) is a rank-normalised ensemble/stack
over the detectors. The **true benefit** `B = AUC_test(fa) − AUC_test(f0)`
(oracle; labels used for evaluation only). The gate must decide using only the
label-free evidence `Z` computed from `Sval`, `yval`, and `Stest` — never
`ytest`.

The synthetic generator (`src/scripts/kbound/make_synth_archive.py`) emits
**exactly** these keys/dtypes, so it is schema-compatible by construction (this is
asserted in `tests/test_smoke_trichotomy.py::test_synth_archive_schema_matches_real`).

---

## 3. Dataset provenance

Sources are loaded by the modules in `src/uais/data/`. All loaders have a
synthetic fallback for smoke tests but use the real files when present under
`data/raw/`.

| dataset / group        | raw location (`data/raw/…`)        | loader                                   | provenance |
|------------------------|------------------------------------|------------------------------------------|------------|
| ADBench tabular (47)   | `adbench/*.npz`                    | refit via sklearn detectors              | ADBench benchmark (Han et al., 2022) |
| Credit-card fraud      | `fraud/`                          | `load_fraud_data.py`                     | Kaggle Credit Card Fraud (PCA features V1–V28, `Class`) |
| PaySim / BAF fraud     | `fraud/`                          | `load_paysim.py`, `load_fraud_data.py`   | PaySim / Bank-Account-Fraud panels |
| UNSW-NB15 (cyber)      | `cyber/`                          | `load_cyber_data.py`                     | UNSW-NB15 network-traffic (49 features) |
| Behaviour / HAR        | `behavior/`                       | `load_behavior_data.py`                  | Human Activity Recognition (561 features) |
| Enron emails (NLP)     | `nlp/enron_emails.csv`            | `load_datasets.load_enron_emails`        | Enron corpus; fetched by `download_nlp_vision.py` |
| CIFAR-10 (vision)      | `vision/cifar-10-python/`         | `load_datasets.load_cifar10`             | CIFAR-10 (Krizhevsky); `download_nlp_vision.py` from cs.toronto.edu |
| CIFAR-10-C             | `experiments/kbound/cifar/CIFAR-10-C/` | GPU TTA scripts                     | CIFAR-10-C corruptions (Hendrycks & Dietterich) — **torch-required** |
| Imagenette-320         | `vision/imagenette/`              | GPU TTA scripts                          | Imagenette subset of ImageNet — **torch-required** |
| NAB / SMD / TSB-AD     | `nab/`, `smd/`, `tsb_ad/`         | tabular time-series loaders              | NAB, Server-Machine-Dataset, TSB-AD benchmarks |
| MVTec-AD / -3D / RealIAD / 3D-ADAM / MulSen | `mvtec*/`, `realiad*/`, … | image/3D detectors            | industrial AD benchmarks — **torch-required** |

A fuller inventory with per-task counts lives in
`docs/research/kbound/DATASET_INVENTORY.md`. The pointer registry for the
local-only score caches is
`docs/research/kbound/manifests/score_cache_manifest.csv`.

> **Torch-free vs torch-required.** The tabular/score-cached tasks (ADBench,
> fraud, cyber, NLP, time-series) drive the CPU-only K-Bound experiments. The
> image/industrial benchmarks need a forward pass through a CNN/ViT/PatchCore and
> are excluded from the CPU path. The hermetic smoke below needs **none** of
> these.

---

## 3b. Run TTA benchmark suite (vision / WILDS) — corrected `beats_both` verdicts

The provenance table in §3 covers the tabular / score-cached tasks but **omitted the
vision and WILDS benchmarks that were actually run** for the paper. They are listed
here, each with its **re-derived** decision verdict. Per the pre-registered protocol a
benchmark counts as a *win* only if the K-Bound router strictly beats **both** trivial
policies on regret **and** stays within the false-adapt budget:

> `beats_both := (router_regret < always-freeze_regret) AND (router_regret < best-fixed-adapt_regret) AND (false_adapt_rate ≤ α)  [α = 0.1]`, stable across seeds.

These verdicts were recomputed in the 2026-06-20 integrity pass directly from the
underlying numbers in each artifact (the historical stored `beats_both` flag was
regret-only and did **not** enforce `false_adapt ≤ α`). Audit trail and the
machine-extracted table: `audits/integrity_2026-06-20/` (`benchmark_verdicts.json`).

| Benchmark | What ran | router vs freeze | router vs best-adapt | false-adapt | corrected verdict | canonical artifact |
|-----------|----------|------------------|----------------------|-------------|-------------------|--------------------|
| **CIFAR-10-C** | ResNet-18, 432-cell harmful/helpful mix (tent & eata) | 0.0016 < 0.1232 ✓ | 0.0016 < 0.0086 ✓ | 0.000 ✓ | **WIN (True)** — verified | `decisive_tta_results.json::benchmarks.cifar10c` |
| **ImageNet-C** | ResNet-50 noise corruptions, 36 cells (sar) | 0.0086 < 0.0277 ✓ | 0.0086 < 0.0606 ✓ | 0.000 ✓ | **WIN (True)** — verified | `imagenetc_noise/decisive_tta_results.json` |
| **Camelyon17** | Protocol G, eata_online, 5-seed dev→test held-out | 3.6e-5 < 0.0749 ✓ | 3.6e-5 < 0.0013 ✓ | 0.026 ✓ | **WIN (True)** — verified | `camelyon17_protocol_G_v1/analyze_F_results.json` |
| **CIFAR-10.1** | 5-seed quick protocol (tent/eata/sar) | — | — | — | **False** — best adapter beats both in only **1/5** seeds (not stable) | `cifar101_multiseed_v1/pooled_summary.json` |
| **ImageNet-R** | multiseed panel, 10 architectures | — | — | — | **False** — **0/10** candidates beat both at a realistic harmful fraction (only ≥0.83) | `imagenetr_protocol_d_multiseed_v1/MULTISEED_ANALYSIS_RESULTS.json` |
| **RxRx1** | Protocol J, sar_online, 10-seed held-out | 0.0 = 0.0 (tie) | — | 0.000 | **False** — KGA correctly freezes (no win, no harm) | `rxrx1_protocol_J_v1/analyze_F_results.json` |
| **iWildCam** | full-val route-b multicandidate; 5th-null headline | 0.0307 ≈ 0.0310 (tie) | 0.0307 < 0.0546 | **0.500 ✗** | **False** — false-adapt 0.5 ≫ α (stored flag was True; **patched**) | `iwildcam_full_val/result_f08e751c.json` |
| **Office-Home** | val, route-a deployed (eata_online_mild) | 0.0281 = 0.0281 (tie) | — | 0.000 | **False (val)** — KGA freezes | `officehome_full_FINAL/VERDICT_val.json` |
| **fMoW** | Protocol L, sar_online, 5-seed held-out | 0.0129 = 0.0129 (tie) | 0.0129 > 0.0092 | **0.375 ✗** | **False** — false-adapt 0.375 ≫ α | `fmow_protocol_L_v1/analyze_F_results.json` |
| **PovertyMap** | Protocol L dev screen | — | — | — | **N/A** — stopped at dev screen (no held-out verdict) | honest null (headline manifest) |
| **ACDC** | segmentation TTA pipeline coded | — | — | — | **N/A** — code-only, **not run** | `experiments/kbound/acdc/` (code present) |

**Notes.** (1) iWildCam and Office-Home each also have a *dev-locked six-arm panel*
sub-protocol (`IWILDCAM_PROTOCOL_H_v2`, `OFFICEHOME_PROTOCOL_M_v2`) whose **held-out
test** row is a properly **gated** win (`verdict_win=True`, false-adapt 0.0). Those
protocol wins are honest and survive the gate; the **dataset-level / val** verdicts in
the table above remain **null** (no win on the main evaluation), which is the
conservative figure recorded here. (2) The three confirmed wins (CIFAR-10-C, ImageNet-C,
Camelyon17) were re-verified, not assumed: each satisfies `false_adapt ≤ α` and both
regret bars. (3) See `INTEGRITY_FIXES.md` (2026-06-20 entry) for the full audit.

## 4. Verifying input integrity (the manifest)

`data/MANIFEST.json` records the SHA-256, byte size, and mtime of every file
under the key data roots (default: `experiments/elara_u/score_archive`). It is
produced and checked by the stdlib-only module `src/uais/data/manifest.py`.

Build (or rebuild) the manifest with current hashes:

```bash
python -m uais.data.manifest build
# optionally choose roots / output:
python -m uais.data.manifest build --roots experiments/elara_u/score_archive --output data/MANIFEST.json
```

Verify on-disk files against the manifest:

```bash
python -m uais.data.manifest verify
# exit 0 = all match; exit 1 = drift (missing / changed); prints a JSON report
```

Notes:

- The score archive is a **local-only pointer artifact**; if it is absent on a
  given machine, `build` writes a valid manifest with `entries: []` (version
  stamped) and `verify` reports `empty: true, ok: true`. Re-run `build`
  centrally once the real archive is present to populate the hashes.
- `verify` catches three failure modes: a **missing** recorded file, a **size
  mismatch**, and a **content (sha256) change** even at identical size. These are
  exercised in `tests/test_data_manifest.py`.

---

## 5. Hermetic smoke path (zero external data, CPU-only, <60 s)

This is the fastest way to confirm the K-Bound algorithm actually runs and makes
the right decisions, on any machine, with nothing downloaded:

```bash
bash scripts/smoke_kbound.sh          # honours $PYTHON, defaults to python3
```

What it does:

1. `src/scripts/kbound/make_synth_archive.py` writes a tiny **deterministic
   synthetic** score archive (fixed seed) to
   `experiments/kbound/_smoke/score_archive/`, using the exact schema in §2, with
   three tasks spanning the trichotomy:
   - `synth_helpful` — the validation-selected detector `f0` has degraded on test
     while the ensemble `fa` is far more correct → benefit clearly **positive**;
   - `synth_harmful` — `f0` stays excellent while `fa` is corrupted → benefit
     clearly **negative**;
   - `synth_unknowable` — `f0` and `fa` are statistically tied → benefit ≈ **0**.
2. `src/scripts/kbound/smoke_trichotomy.py` runs the **real** `kga` package over
   the archive — `compute_evidence` (label-free `Z`), `KGA.certify` (the
   empirical-Bernstein certificate `Δ̂ ± ε`), and `KGA.decide` (the trichotomy) —
   and **asserts** `synth_helpful → ADAPT`, `synth_harmful → FREEZE`,
   `synth_unknowable → ABSTAIN`. The separations are large relative to the
   certificate radius, so the result is robust, not luck.
3. It writes `experiments/kbound/_smoke/smoke_result.json` and prints
   `SMOKE PASS` with a one-line metrics summary; it exits non-zero on any
   mismatch.

Run the same logic under pytest:

```bash
pytest tests/test_smoke_trichotomy.py tests/test_data_manifest.py -q
```

Both suites are pure `numpy` + `kga` (no torch, no network, no real datasets) and
complete in a few seconds.
