# Senior-Engineer Audit — AutoML_Flagship_V8 (ELARA / RGA)

**Pass type:** full-repo, honest, not flattering.
**Scope:** code organization, deployment surface, MLOps maturity, research
artifact track, paper potential. Cross-checked against the prior 2026-05-14
audit in [FULL_RESEARCH_AUDIT_2026-05-14.md](FULL_RESEARCH_AUDIT_2026-05-14.md).
**Date:** 2026-05-15.

---

## 0. One-paragraph verdict

This is a **substantial, mostly real, partially over-built research codebase
with one genuinely interesting scientific finding** (the contrastive
gate-helps-vs-hurts result) wrapped in a meaningful amount of dead weight
(two parallel Python packages, two parallel config trees, two parallel
FastAPI files, several promised-but-unimplemented modules, and an MLOps
surface that's installed but turned off). The asset pipeline and the
manuscripts are reproducible end-to-end in this working tree; the headline
numbers verify against the JSON artifacts; the tests are green (216 pass).
The credibility-blocking issue is **not** code quality — it's *benchmark
split discipline*, already self-flagged in the prior audit but only
partially remediated. A senior engineer looking at this for an interview, a
review, or a hire-decision would see: solid systems engineering, honest
research instincts, and a too-large surface that needs ~1 week of
deliberate pruning before it stops sending mixed signals.

**Overall grade as a research codebase:** **6.5 / 10.**
**Overall grade as a production system:** **3 / 10** (intentionally not
the goal, but the deployment scaffolding implies otherwise).
**Overall grade as a portfolio / hireable artifact:** **7 / 10.**
**Overall grade as a submittable paper:** **5 / 10** (workshop-grade with
cleanup; not yet conference-defensible).

---

## 1. Repository surface — what's actually here

### 1.1 Top-level layout

| Dir | Purpose | Honest status |
|---|---|---|
| `src/uais/` | Primary research package, 127 files, 18 submodules | ✅ Real |
| `src/uais_v/` | Aspirational "v2" rewrite, 39 files, 4 submodules | ⚠️ Abandoned mid-build |
| `src/scripts/` | 30 experiment runners + paper-asset generators | ✅ Active |
| `src/orchestration/` | "Prefect" domain flow wrappers | ❌ Not real Prefect |
| `src/pipeline/` | Older data ingest + feature build | ⚠️ Pre-uais layer |
| `configs/` | 17 YAML — canonical | ✅ Active |
| `config/` | 6 YAML — legacy | ❌ Dead, unused |
| `data/raw/` | mvtec3d, fraud, cyber, behavior, nlp datasets | ✅ Real |
| `experiments/fusion/` | JSON results + CSVs + metadata for the paper | ✅ Real |
| `docs/research/` | The two manuscripts, tables, figures, review folder | ✅ Active |
| `docs/` | API + reproducibility + technical-brief markdowns | ⚠️ Stale |
| `notebooks/` | 25 notebooks numbered 00–100 | ⚠️ Overlap with `src/scripts/` |
| `dashboard/` | Streamlit app reading from `experiments/<domain>/metrics/` | ⚠️ Demo-grade, no live inference |
| `deploy/api/` | Two FastAPI files (`main.py` + `main_enhanced.py`) | ❌ Pick one |
| `scripts/` | 14 shell wrappers + `ci_smoke.py` + `rebuild_paper.sh` | ✅ Active |
| `tests/` | 57 test files, 216 passing | ✅ Healthy |
| `reports/` | `metrics_*.csv` placeholder + 0-byte docx (deleted) | ⚠️ Labeled legacy |
| `output/pdf/` | Compiled PDFs of paper + thesis | ✅ Current |
| `mlflow_config.yaml` | MLflow tracking-uri config | ❌ MLflow off in every run config |
| `Dockerfile` + `docker-compose.yml` | Container scaffolding | ❌ Scaffolding, not production |
| `.github/workflows/ci.yml` | Single CI job: ruff + pytest + smoke | ⚠️ Minimal |

**Diagnosis:** the repo carries ~30% deadweight in surface area. None of it is
catastrophic on its own, but together it creates the impression of a system
that does more than it does. A senior engineer reading this for the first
time will spend their first 20 minutes asking "is `uais_v` the new one or
the old one?" — and that's bad first impression.

### 1.2 The duplicate-surface problem (top 5 cleanups)

| Cost to fix | Item | Action |
|---|---|---|
| 30 min | `config/` (legacy) vs `configs/` (canonical) | Delete `config/`, redirect any stale references |
| 1 hr | `deploy/api/main.py` vs `deploy/api/main_enhanced.py` | Pick `main_enhanced.py`; rename to `main.py`; delete old |
| 2 hr | `src/uais_v/` (39 files, 3 NotImplementedError stubs) | Keep only `uais_v/models/multi_sequence_30_torch.py` + `multi_sequence_30_tf.py` (the only modules that real tests depend on); delete the rest; merge what remains into `src/uais/sequence/` |
| 30 min | `src/orchestration/*_flow.py` (Prefect-flavored function wrappers) | Either add real `@flow` decorators or delete and call the experiment scripts directly |
| 1 hr | Stale top-level docs (`UAISV_Final_Project_Summary.md`, `PHASE_2_RESEARCH_PLAN.md`, `IMPROVEMENTS.md`, `data/README.md`) | Either delete or rewrite to match ELARA reality |

Total: ~5 hours of focused pruning to drop ~30% of the perceived surface
without losing any working functionality. This is the single highest-value
cleanup in the entire audit.

---

## 2. Code quality (the `src/uais/` real package)

### 2.1 What works

- **18 submodules with clean separation** — `anomaly/`, `data/`, `features/`,
  `fusion/`, `nlp/`, `vision/`, `sequence/`, `supervised/`, `utils/`,
  `explainability/`, etc. Each carries its own purpose; minimal cross-coupling.
- **`src/uais/fusion/attention/`** is the cleanest part of the codebase:
  `cross_modal_attention.py`, `reliability_estimator.py`,
  `counterfactual_explainer.py`, `baselines.py`, `learned_gate.py`,
  `attention_utils.py` — all reasonably scoped, all tested.
- **216 tests, 3 skipped, 0 failing.** That's a real number; the test
  suite isn't decorative.
- **Asset pipeline (`generate_craf_paper_assets.py` +
  `emit_mvtec3d_assets.py`)** turns JSON into deterministic LaTeX tables
  and figures. The double-renderer pattern (LA via direct generator;
  MVTec via prefix-rewriting shim) is mildly clever and avoids touching
  the working generator.

### 2.2 What doesn't

- **Dead modules that survive in the package:**
  - [src/uais/drift/drift_vision.py](../../../src/uais/drift/drift_vision.py)
    returns `{"embedding_shift": 0.0}`
  - [src/uais/drift/drift_nlp.py](../../../src/uais/drift/drift_nlp.py)
    returns `{"js_divergence": 0.0}`
  - [src/uais/recommender/rules.py](../../../src/uais/recommender/rules.py)
    — `assign_action_from_scores` is never called from any script or test
  - [src/uais/generative/](../../../src/uais/generative/) — `train_vae.py`
    and `train_wgan.py` exported but only referenced from `notebooks/90_*.ipynb`
- **uais_v sprawl** — 39 files, three of which (`train_cyber_supervised.py`,
  `train_fraud_supervised.py`, `train_nlp_supervised.py`) raise
  `NotImplementedError` immediately. The only real code in `uais_v/` is
  the 30-sequence dataset builder and the model definitions I had to
  create during the test-fix pass.
- **`src/pipeline/`** — `ingest.py`, `build_features.py`, `train_models.py`
  pre-date the modular `uais/` package and aren't called by current
  experiments. Effectively museum pieces.
- **`src/orchestration/*_flow.py`** — six "Prefect flows" that each define
  a single function that imports and calls an experiment script. No
  `@task`, no `@flow`, no Prefect runtime hooks. Prefect is a 50 MB
  dependency that buys nothing.

**Diagnosis:** the *real* code quality is good. The *perceived* code
quality is dragged down by the dead modules occupying namespace.

---

## 3. Test suite

- **57 test files, 216 passing.** Good baseline.
- **Real coverage** of: attention masking, reliability estimator (multiple
  files), counterfactual explainer, baselines, learned gate, TTA adapters
  (Tent / TTT), benchmark builders, asset metadata, multi-seed
  aggregation, leakage guards.
- **Weak coverage**: NLP transformer (smoke only), vision (one ResNet
  test), generative modules (none), drift detectors (none).
- **Test names that overpromise**: `test_nlp_tiny` (shape-only),
  `test_api_payloads` (schema-only), `test_unsupervised_baselines` (does
  some real comparisons but not many).

Verdict: **healthy**, with calibration error mostly in NLP/vision/generative
corners that the paper doesn't depend on anyway.

---

## 4. Deployment surface

### 4.1 FastAPI

| Item | Status |
|---|---|
| `deploy/api/main.py` | 7 endpoints, no auth, no monitoring, model artifacts loaded at startup with graceful 503 if missing |
| `deploy/api/main_enhanced.py` | 10 endpoints, optional API-key auth (defaults to no-op in dev), Prometheus metrics, health-check tree |
| Auth | Wired in `main_enhanced.py` only; defaults to no-op when `UAIS_API_KEYS` env var is unset |
| Monitoring | Wired in `main_enhanced.py` only |
| Container | `Dockerfile` exists but `CMD bash` — not a service entrypoint |
| Compose | Wires API + Streamlit + MLflow with bind-mount volumes; no healthchecks, no restart policies |

**Diagnosis:** demo-grade. The two-file split (`main` + `main_enhanced`) is
classic mid-project drift. Pick one, delete the other, set `CMD` to
`uvicorn deploy.api.main:app --host 0.0.0.0 --port 8000`, add a health
endpoint with real liveness probes, add restart policies in compose.
Half a day of work for production-grade.

### 4.2 Streamlit dashboard

- `app_streamlit.py` is runnable; reads
  `experiments/<domain>/metrics/metrics.csv` files; graceful warnings on
  missing artifacts. No live inference path.
- Useful as an internal demo dashboard or a portfolio piece. Not a
  customer-facing surface.

### 4.3 MLflow + Prefect

- `requirements.txt` pins `mlflow>=2.14.0` and `prefect>=2.14.0`.
- **MLflow** — `mlflow.enabled: false` in every fusion config and every
  domain baseline config. Effectively off. Installing it costs ~50 MB
  for nothing.
- **Prefect** — 0 `@task` / `@flow` decorators in the entire codebase
  (`grep -r "@task\|@flow" src/` returns nothing). The "Prefect flows"
  are just functions named `*_pipeline()`.

**Diagnosis:** both MLOps tools are *theater*. They're listed in the
README as part of the system architecture, they're installed, they show
up in `pip list` — but they don't actually do anything. Two options: (a)
turn them on properly and use them, (b) remove them from requirements
and the README and the architecture description.

### 4.4 CI

- Single workflow ([.github/workflows/ci.yml](../../../.github/workflows/ci.yml)).
- Ruff syntax/undefined-name lint only (no `--select=E,F,W,B,I` etc.).
- pytest with coverage, Codecov upload, 50% coverage threshold.
- Smoke tests of orchestration flows that are best-effort and don't fail
  the build.
- No matrix testing, no pip cache, no Docker build, no artifact upload, no
  secret scanning, no SBOM.

**Diagnosis:** minimal but functional. Two-hour upgrade: add pip cache,
matrix on Python 3.11/3.12/3.14, fail on lint warnings, build Docker as
a sanity check, upload coverage HTML as artifact.

---

## 5. Research-artifact track (the paper + thesis)

### 5.1 Numerical claims verify

Three spot checks vs the on-disk JSON artifacts:

| Claim | Source | Verified? |
|---|---|---|
| 8 MVTec categories | `mvtec3d_fusion_metadata.json` categories array | ✅ |
| 3,226 paired samples | `mvtec3d_fusion_metadata.json` samples field | ✅ |
| RF clean ROC-AUC 0.959 | `mvtec3d_results.json` → `clean_metric_summary.random_forest.roc_auc.mean = 0.9592` | ✅ |
| RGA −0.060 clean delta | Difference of static (0.728) − RGA (0.668) in the same JSON | ✅ |

**Asset pipeline is honest.** All numbers in the paper flow through
parametric rendering (`_fmt_pm_ci`, `_bold_best`) from the JSON. No
hardcoded values in tables.

### 5.2 But the benchmarks have split-discipline issues

This is the most damaging finding and it's already self-documented in
[FULL_RESEARCH_AUDIT_2026-05-14.md](FULL_RESEARCH_AUDIT_2026-05-14.md):

- **C1 (RealFusion-LA):** Original audit found source rows reused across
  fusion train/test splits. Audit notes claim this is fixed in code; need
  to verify by re-running and inspecting overlap audit.
- **C2 (MVTec 3D-AD):** Score normalization is now fit on train/good
  only (verified in
  [prepare_mvtec3d_fusion_benchmark.py:190-198](../../../src/scripts/prepare_mvtec3d_fusion_benchmark.py#L190-L198))
  — the scorer itself respects MVTec's split. But the fusion runner does
  its own `train_test_split` on the entire dataset
  ([run_breakthrough_experiment.py:123-138](../../../src/scripts/run_breakthrough_experiment.py#L123-L138))
  — so fusion training sees re-shuffled rows from MVTec's original
  train/val/test, defeating split discipline at the fusion level.
- **C3:** Generated assets not tracked in git. Anyone cloning the repo
  can't build the PDF without regenerating JSON, which requires the full
  Python env + raw MVTec data.

**Diagnosis:** the scorer side is now clean. The fusion-split side is
still a methodological soft spot. A reviewer will accept the negative-result
finding *only if* the fusion runner respects MVTec's original splits
(use `_split` with `split` column as a stratification key, not a fresh
random split).

### 5.3 Manuscript honesty

- Both manuscripts now report the negative MVTec result in the abstract.
- Both have a "Cross-Benchmark Contrast" section that frames the asymmetry
  as the central finding.
- Both reference the existing threats-to-validity discussion.
- Reviewer-q&a preempt list lives in
  [REVIEWER_RATING_AND_PHASE_PLAN.md](REVIEWER_RATING_AND_PHASE_PLAN.md).

This is genuinely above average for an independent research codebase. Most
authors don't write their own negative-result audit.

### 5.4 Companion docs are stale

| Doc | Status |
|---|---|
| `README.md` (root) | Generic "UAIS-V" framing; no ELARA mention |
| `UAISV_Final_Project_Summary.md` | "UAIS-V" framing; "Expected grade 95-100"; describes CERT/DistilBERT/ResNet pipelines that don't exist |
| `PHASE_2_RESEARCH_PLAN.md` | Proposes CMAF (Cross-Modal Attention Fusion); never built; superseded by ELARA/RGA |
| `IMPROVEMENTS.md` | Lists pre-commit hooks and test improvements; unrelated to research story |
| `REPRODUCIBILITY.md` | Covers attention-fusion training only; missing the four other steps |
| `data/README.md` | Still mentions "bagel smoke-run"; reality is 8-category benchmark |

**Diagnosis:** the marketing docs lie about what's in the repo, the
research docs are several iterations behind, the reproducibility doc covers
1/5 of the actual pipeline. A reader skims these first and forms an
inaccurate picture before they ever open the paper. Half a day to fix.

---

## 6. The scientific finding — what's actually in the paper

After all the audit, what does this paper *say*?

> *On naturally paired RGB+3D anomaly data (MVTec 3D-AD, 8 categories,
> 3,226 samples), a reliability-gated attention module with a
> validation-derived KS-drift signal degrades both clean and adversarial
> ROC-AUC relative to static attention. On a label-aligned synthetic
> benchmark, the same module improves coherent all-domain attack
> robustness. The asymmetry is explained by component ablation: the
> KS-drift component, which delivers all the label-aligned stress gain,
> misfires on legitimate inter-category variation in naturally paired
> data.*

This is **a real, publishable finding** — modest, scoped, contrastive,
and methodologically interesting. It's also **defensible only after the
fusion-runner split-discipline is fixed**, because right now a reviewer
can argue the MVTec negative-result is partly an artifact of fusion-level
re-splitting.

### What it *isn't*

- It is not a new fusion architecture (the attention block is standard).
- It is not a new reliability signal (ECE + KS + sharpness are all
  textbook).
- It is not a new explanation method (masking-based attribution is
  decades old).
- It is not a deployment system (despite the FastAPI/Streamlit/Docker
  surface).

The work is **integration + ablation + contrastive measurement**, and
the contribution is the *measurement* — the experimental observation
that validation-derived drift gates can flip sign across pairing regimes.

---

## 7. Potential — what this can become

I see four realistic paths, in increasing order of effort.

### Path A — arXiv preprint + portfolio (~1 week)

Effort: prune dead code (§1.2), fix companion docs (§5.4), regenerate
benchmarks with split-disciplined fusion-runner split, recompile, post
arXiv.

Outcome: a credible technical report you can link to from a resume or
job application. Demonstrates research instincts + engineering
competence.

### Path B — Workshop submission (~3 weeks)

Effort: above + run learned-gate ablation as a row in the τ-sweep,
add per-domain subset adversarial attack table, write reviewer-Q&A
preempt paragraph, target NeurIPS-W or ICML-W or an IEEE security
workshop.

Outcome: defensible workshop paper. Demonstrates ability to extend
work in response to anticipated review.

### Path C — Mid-tier conference (~2–3 months)

Effort: above + replace the lightweight image-statistic MVTec scorer
with M3DM-style ResNet/PatchCore features (this is the single biggest
empirical upgrade available), re-run all MVTec experiments, expand the
contrast story with whichever direction the new numbers point. Target
CIKM, ICDM, IEEE BigData, or IEEE TKDE.

Outcome: real conference paper. Demonstrates ability to take research
through to publication.

### Path D — Production research platform (~6+ months)

Effort: collapse the duplicate surfaces, turn on MLflow/Prefect for
real, implement real per-customer auth, multi-tenant model versioning,
canary-deploy rollout, online drift detection that actually fires, model
registry with promotion gates. This is real engineering work that would
need either a team or a much longer runway.

Outcome: a system that could underpin an anomaly-detection product or
internal platform. Realistic only if there's a downstream user/customer
forcing the priorities — building it speculatively is the wrong move.

**Recommended path: A, then B if you have the time.** C requires real
data-engineering effort (M3DM features). D requires customers.

---

## 8. Top 10 actionable items (priority-ordered)

1. **Fix fusion-runner split discipline** so MVTec's original
   train/val/test boundaries are respected when building the fusion
   train/test fold. This is the single most important credibility fix.
2. **Delete `src/uais_v/` except for the two `models/` files**, fold the
   30-seq dataset builder into `src/uais/sequence/`.
3. **Delete `config/`** (legacy).
4. **Delete `deploy/api/main.py`**, rename `main_enhanced.py` to `main.py`.
5. **Decide on MLflow + Prefect:** either turn them on
   (`mlflow.enabled: true` in all configs, real `@flow` decorators), or
   delete them from `requirements.txt` and the README.
6. **Rewrite root `README.md`** to describe ELARA accurately (15 min) and
   delete or rewrite `UAISV_Final_Project_Summary.md`,
   `PHASE_2_RESEARCH_PLAN.md`, `IMPROVEMENTS.md`.
7. **Commit generated assets** (`docs/research/tables/*.tex`,
   `docs/research/figures/*.png`) so the PDF builds from a clean clone.
8. **Add CI matrix testing** (Python 3.11 / 3.12 / 3.14), pip cache,
   Docker build job.
9. **Delete dead modules** ([drift_vision.py](../../../src/uais/drift/drift_vision.py),
   [drift_nlp.py](../../../src/uais/drift/drift_nlp.py),
   [recommender/rules.py](../../../src/uais/recommender/rules.py)) or
   implement them.
10. **Regenerate all paper assets after fix #1**, recompile both PDFs,
    update the `docs/research/manuscript_review/REVIEWER_RATING_AND_PHASE_PLAN.md`
    rating from 5/10 to something defensible.

Total estimated effort: **3 focused engineering days** for items 2–10.
Item 1 needs the underlying experiments to re-run, so it's a few hours
of code + however long the experiment takes.

---

## 9. What I would tell you in a 1-on-1

You've built a real codebase. It does what the paper says it does, the
numbers verify, the test suite is healthy, the asset pipeline is
reproducible, and you self-flagged your own benchmark problems before I
or anyone else asked. That's unusually good research hygiene for an
independent project.

The work isn't done yet. The surface area is too big — there's a second
package no one uses, a second config tree no one uses, a second API file
no one uses, two MLOps tools no one uses, and three or four marketing
documents that no longer describe the system. None of this is *broken*,
but together it makes the codebase look bigger and less considered than
it really is. A reviewer or a hiring manager who reads the README first
will leave with a worse impression than what the code actually deserves.

The single best thing you can do this week is **shrink the surface** —
delete 30% of the files, fix four documents, and rerun the experiments
with proper split discipline. After that one week, your repo says
exactly what you mean it to say, the paper says exactly what the
evidence supports, and the next reviewer's first impression matches the
quality of what you've actually built.

Then go submit it to a workshop. The negative-result finding is real and
publishable. It just needs a clean stage.
