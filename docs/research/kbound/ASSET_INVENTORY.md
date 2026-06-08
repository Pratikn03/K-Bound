# K-Bound — Asset Inventory & Mapping to Paper Needs

Full sweep of `AutoML_Flagship_V8/` (≈402 GB, of which 387 GB is `data/raw`),
mapped to what the K-Bound paper actually requires. Honest verdict at the bottom:
you have a **very strong asset base for the anomaly-routing track**, the theory
scaffolding is partly built, and **two specific things the headline claims need are
missing**.

---

## 1. Datasets — `data/raw/` (387 GB)

| Dataset | Size | Type | Role for K-Bound |
|---|---:|---|---|
| realiad_d3 | 259 GB | RGB + pseudo-3D + point cloud, multimodal | **Core**: mixed clean/failure regime (the discrimination case) |
| mvtec3d | 26 GB | RGB + depth | **Core**: the cleanest validated mixed regime (+0.21 under failure) |
| eyecandies | 26 GB | synthetic RGB-D-normals | Negative/development evidence only |
| mulsen_ad | 19 GB | RGB + IR + point cloud | Multi-sensor failure regime |
| mvtec_loco | 12 GB | RGB + edge proxy | Secondary diagnostic |
| real3d | 11 GB | point cloud | Exploratory |
| tsb_ad | 9.4 GB | time-series AD | Temporal-shift regime |
| 3d_adam_anomalib | 6.5 GB | RGB + depth | External transfer |
| mvtec_ad, visa | 6.3 + 4.3 GB | RGB industrial | Industrial family |
| realiad | 4.1 GB | RGB | Natural-transfer |
| baf, cyber, fraud | 1.3 GB + 606 MB + 144 MB | tabular/flow | Tabular knowability suite |
| smd, nab, har | 479 + 22 MB + 65 MB | time-series / sensor | Temporal + sensor shift |
| adbench (+cv/nlp/sealed/industrial/classical) | ~1.3 GB total | tabular/image-OOD/text features | **Core**: the 123-task score archive source |
| healthcare | 354 MB | clinical time-series | Development only |
| indep_external, openml_indep | 241 + <1 MB | mixed | Independent external validation |

**What this buys the paper:** the entire anomaly-routing track (safety,
failure-recovery, mixed-regime, multimodal) is fully supported by real, on-disk data.

**Dataset GAPS (not present):**
- ❌ **CIFAR-10-C / ImageNet-C / corruption benchmarks** — needed for the
  *catastrophic-harm deep-TTA* experiment that would let K-Bound beat **both**
  trivial baselines. Must be downloaded (~3–6 GB) and needs GPU.
- ❌ **Standard regression-shift dataset** (e.g. housing, bike, superconductivity) —
  needed to validate Theorem 4's covariate-shift positive case. Synthesizable, or a
  small download.
- ❌ **Office-Home / DomainNet** — only if going to the broad cross-domain claim.

---

## 2. Code directly reusable for K-Bound — `src/`

| Module | What it is | K-Bound reuse |
|---|---|---|
| `src/elara/certification/switching_certificate.py` | paired-bootstrap **+ empirical-Bernstein LCB** switch certificate | **This IS Theorem 3's certificate.** Reuse directly as KGA's `(Δ̂, ε)` decision core |
| `src/elara/certification/risk_dominance.py` | risk-dominance LCB, prevalence sweep | Supports the freeze/abstain bound |
| `src/elara/theory/t1_impossibility.py` | quality-blind fusion impossibility (T1a/T1b) | **Related but NOT the same** as K-Bound Thm 1 (label-free *decision* impossibility). Useful supporting result, not a substitute |
| `src/elara/theory/gdr_minimax.py` | minimax decision-rule argument | Supports the "abstain is minimax-safe" claim in Thm 1 |
| `src/elara/theory/novel_theorem_bounds.py`, `theorem_registry.py` | theorem code registry + validators | Reuse the registry pattern to register K-Bound theorems with machine checks |
| `src/uais/drift/{tabular,vision,nlp,time_series}.py` | drift detectors (KS/MMD etc.) | The **observable evidence Z** extractor — drop-in for KGA `phi()` |
| `src/uais/elara_u/{router.py, contract.py}` | routing + claim contract | Router scaffolding for KGA wrapper |
| `src/uais/fusion/` + `attention/` | fusion + RGA gating | The candidate adaptation `f_a` (gated fusion) |

## 3. Experiment scripts that already feed K-Bound — `src/scripts/elara_u/`

Directly relevant (≈10 of ~40): `build_score_archive.py` (makes the 123-task
archive), `multimodal_reliability_experiment.py` + `multimodal_failure_matrix.py`
(the mixed-regime D23 evidence), `shift_stress_ablation.py` (the knowability
crossover), `per_sample_routing.py`, `heterogeneous_degradation_ablation.py`,
`statistical_audit.py`, `sealed_external_eval.py` / `openml_indep_eval.py` (held-out
checks). These are the honest backbone — reuse, don't rewrite.

## 4. Existing result artifacts — `experiments/elara_u/*.json`

The 123 `.npz` score archive + ~40 result JSONs (multimodal, stress, failure-matrix,
sealed/independent). All already mined by the K-Bound experiments in `kbound_paper/`.

## 5. Models / pipeline / notebooks — mostly NOT needed for the paper

- `models/` (4.4 MB), embedding caches: minor; the score archive is the real input.
- `deploy/api/` (FastAPI + auth + monitoring): a **production engineering track**,
  irrelevant to the theory paper. Don't spend paper time here.
- `notebooks/` (24): ELARA-era EDA (fraud/cyber/vision/nlp). Tangential; useful only
  for figures or sanity, not core results.

---

## 6. What you HAVE vs what the paper NEEDS

**Have (real, on disk, reusable):**
- ✅ All data + scores for the anomaly-routing track (safety, failure-recovery, mixed regime, multimodal).
- ✅ The finite-sample certificate (Thm 3) already coded (`switching_certificate.py`).
- ✅ Drift/evidence extractors for `Z`; fusion/gating for `f_a`; a theorem registry pattern.
- ✅ Honest statistical tooling (bootstrap, Holm, sealed holdouts).

**Need (gaps):**
- ❌ **Deep-TTA corruption data + GPU run** (CIFAR-10-C/ImageNet-C + Tent/EATA/SAR) —
  the only path to the "beats both trivial baselines" headline.
- ❌ **Regression-shift dataset** — to validate Theorem 4's proved covariate case.
- ❌ **A clean Theorem-1 witness** (constructed pair with provably equal Z-law) — paper, not data.
- ❌ **Theorem 4 general proof** (sign-of-difference identifiability, delta over Steinhardt-Liang) — the real open theory work.

---

## 7. Bottom line

For the **scoped anomaly-routing paper**, you are *asset-complete*: data, code,
certificate, and honest tooling all exist and are already wired into `kbound_paper/`.
The gaps are precisely the two things that separate a solid scoped paper from the
broad claim — a deep-TTA catastrophic-harm benchmark (data + GPU) and the Theorem-4
positive proof. Neither is faked; both are concrete, named next steps. The 387 GB of
multimodal data is an asset *most* TTA-theory authors don't have — it is the reason
the failure-recovery and mixed-regime evidence is real rather than toy.
