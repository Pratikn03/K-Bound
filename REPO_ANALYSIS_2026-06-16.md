# Repository Analysis — K-Bound / ELARA-U (`AutoML_Flagship_V8`)

*Generated 2026-06-16. Scope: whole-folder analysis across four axes — high-level overview, code quality & architecture, research integrity & reproducibility, and disk/cleanup. Read-only; nothing in the repo was modified.*

---

## 1. Executive summary

This is a **research monorepo** for two related contributions: **K-Bound** (a theory + importable certificate, *KGA*, for deciding without target labels whether to **adapt / freeze / abstain** under distribution shift) and **ELARA-U** (a cross-domain anomaly *meta-routing* benchmark whose headline result is "stacking beats selection"). It is the work of a single independent researcher, on `main`, with a clean working tree.

| Axis | Verdict | One-line |
|---|---|---|
| Research integrity | **Strong** | Headline numbers match on-disk artifacts to the digit; the repo ships its own adversarial self-audit and honest negatives. Under-claims more than it over-claims. |
| Engineering of the core | **Strong** | `kga/` is clean, typed, dependency-light, well-tested; reproducibility/anti-leakage test culture (732 test functions) is well above average for research code. |
| Engineering of the periphery | **Mixed** | ~160 one-off scripts, a 3,016-line experiment monolith, triple-vendored certificate logic, and a batch of **non-existent dependency versions** that silently break the "production" Docker/CI path. |
| Disk hygiene | **Needs attention** | Drive is **95% full (108 GB free)**. `.git` is **1.1 GB**, ~79 MB of stray `wget-log*` junk, committed `src/mlruns/`, and hundreds of macOS `._` sidecars. |

**The single most important concrete defect** is not in the research — it is the set of fictional/unresolvable dependency pins (e.g. `httpx2`, `pandas==3.0.3`, `pytest==9.0.3`, `starlette==1.2.0`, `certifi==2026.5.20`) in `requirements-api.txt` and inside `ci.yml`. The polished Docker image and main CI pipeline **cannot install as written**. The hermetic `kbound-ci.yml` lane (unpinned minimal deps) does work.

---

## 2. What this project is (high-level overview)

The repo is the union of an older domain-ML system and a newer, paper-grade research program. Three names matter:

- **K-Bound** — *the current paper.* "When Is Label-Free Adaptation Knowable?" A theory (5 core theorems) plus a decision rule that says when label-free test-time adaptation is safe. The decision rule is productized as the pure-numpy **`kga/`** package (`from kga import KGA`, `python -m kga decide`, served at `POST /decide`). Paper (20 pp) and a 166-pp manuscript live under `docs/research/kbound/`.
- **ELARA-U** — the *superseded foundation* and benchmark. Studies anomaly detection as a meta-routing problem over a detector "zoo": decide when to select/fuse/stack/fall back without test labels. Result: a rank-normalized logistic **stack beats validation auto-selection (+0.036 AUROC)** and the best fixed detector (+0.075), and reliability/drift routing helps **only** where modalities can fail independently (two real multimodal datasets).
- **ELARA / RGA** — the *legacy source system* (Reliability-Gated Attention), retained as foundation and honest-negative evidence. Code under `src/uais/fusion/attention/`.

The framing throughout is deliberately hedged: "Honest scope," "we do **not** claim universal/per-dataset SOTA," and explicit "honest negatives." That posture is backed by the evidence (Section 5).

---

## 3. Architecture & layout map

Five Python "packages" exist, but only two are canonical:

| Package | `.py` files | Role | Verdict |
|---|---|---|---|
| `src/uais/` | ~149 | Main library (anomaly, fusion, supervised, sequence, nlp, vision, drift, ensembles, training, validation, registry, utils) | **Canonical / real** |
| `kga/` (top-level) | 8 | Pure-numpy K-Bound decision core; typed (`py.typed`), torch-free, has CLI | **Canonical, production-grade** — the cleanest code in the repo |
| `src/elara/` | 23 | Theorem implementations (T1–T9), retrospective certification, evaluation | Active **research**, not production |
| `infer_rga/` (top-level) | 2 | Minimal deployment inference surface; *reuses* `uais.fusion.attention` (no copy) | Active, intentionally minimal |
| `src/uais_v/` | **0 (empty)** | — | **Dead directory** still declared in stale `egg-info` |

The codebase's own auto-generated `CODEBASE_MAP.md` self-classifies the ~72.7k LOC / 1,662 tracked files as: **UAIS-legacy** 246 files / 40.9k LOC, **tests-legacy** 142 / 11.7k, **RGA/provenance** 42 / 10.8k (the live fusion engine), **ELARA-U** 55 / 7.6k, **production** 9 / 1.7k (`deploy/api/` + `dashboard/`). The "legacy" label on the largest slice is the repo's own admission.

**Main architectural smell — the K-Bound certificate exists in three lineages:** `src/elara/certification/switching_certificate.py` (origin) → `kga/certificate.py` (vendored, productized) → `docs/research/kbound/kbound_pkg/kbound/` (a **complete 7-module second copy checked into `docs/`**). Intentional vendoring, but a genuine drift risk.

There is also a small **C++ subproject** (`research_dashboard/`, with its own CI workflow) that builds a JSON aggregator.

---

## 4. Code quality & engineering

**Strengths**
- `kga/` is the extractable jewel: clean, typed, dependency-light, the only module under strict mypy + full ruff in CI.
- **Test culture is a real asset.** 154 test files / **732 `def test_` functions**; zero disabled tests (no `@pytest.mark.skip`/`xfail`; the 54 `pytest.skip()` calls are runtime guards for missing optional datasets). Many tests assert research invariants directly — no test-set labels used for selection, manuscript claims match artifacts (e.g. `test_no_test_selected_rga_plus.py`, `test_manuscript_claim_consistency.py`).
- **Tooling is real and thorough:** mature 123-line `.gitignore`, `.pre-commit-config.yaml` (ruff pinned, large-file guard, mypy on `kga/`), centralized ruff config, and a genuinely hermetic `kbound-ci.yml` (`<60 s` smoke, per-theorem validators must exit 0, "imports work without prefect" contract).
- Very low in-code debt: only **2 TODOs**, zero FIXME/HACK/XXX across `src/`. No `__pycache__`/`.pyc`/egg-info are *tracked*.

**Weaknesses / tech debt** (priority order)
1. **Fictional dependency pins break the production path.** `requirements-api.txt` lists packages/versions that do not exist (`httpx2==2.2.0`, `httpcore2`, `annotated-doc`, `starlette==1.2.0`, `cryptography==48.0.0`, `numpy==2.4.6`, `scikit-learn==1.8.0`, `certifi==2026.5.20`). `ci.yml` installs more (`pandas==3.0.3`, `pytest==9.0.3`, `pyarrow==24.0.0`, `torch==2.12.0`, `tifffile==2026.3.3`). The Dockerfile and main `ci.yml` therefore fail at install. *(The Dockerfile/compose definitions themselves are high quality — pinned digests, non-root, read-only, healthchecks — only the requirements they install are broken.)*
2. **`httpx2` typo** propagated into `requirements.txt`, `requirements-api.txt`, `requirements.lock.txt`, and three CI install lines (real package is `httpx`).
3. **Script sprawl.** ~160 single-purpose modules in `src/scripts/` (dozens of near-identical `emit_*` table generators), plus 116 `.py` under `experiments/`. Worst offender: `src/scripts/run_breakthrough_experiment.py` at **3,016 LOC**. Other large files: `healthcare_gap_closure.py` (1,050), `deploy/api/main.py` (948), `evaluate_attention_harness.py` (909).
4. **Triple-duplicated certificate logic** (Section 3) — intentional but drift-prone.
5. **Stale packaging metadata:** empty `src/uais_v/` still declared in `egg-info`, which also omits `kga`; two competing pytest configs (`pyproject.toml` vs `setup.cfg`, plus a stale flake8 block); Python version referenced as 3.9 / 3.10 / 3.11 / 3.12 across configs and bytecode.
6. **`src/mlruns/` is committed** (51 MLflow metadata files) — should be gitignored.
7. **Heavy dependency surface:** ships *both* PyTorch and TensorFlow, plus xgboost/lightgbm/catboost, shap/lime, mlflow, prefect, streamlit, fastapi. Main `requirements.txt` is lower-bound (`>=`) only, not locked.

---

## 5. Research integrity & reproducibility

**Verdict: honest and appropriately hedged — among the better-disciplined research repos.** Every headline number checked traces to a real artifact and matches.

**Claims vs evidence (independently opened the JSONs):**

| README claim | On-disk value | File | Match |
|---|---|---|---|
| Stack > auto-select **+0.036**, CI [0.023, 0.050]; 86/123 wins | 0.03551, CI [0.02333, 0.04952], win_rate 0.699 (=86/123) | `honest_benchmark.json` / `statistical_audit.json` | ✅ |
| Stack > best fixed **+0.075**, CI [0.052, 0.101] | 0.07501, CI [0.05201, 0.10109] | `honest_benchmark.json` | ✅ |
| Sealed external **+0.016**, CI [0.004, 0.032], 74 tasks | 0.0161, CI [0.0039, 0.0319], 74 | `sealed_external_results.json` | ✅ |
| Independent external **+0.111**, CI [0.052, 0.176] | 0.1106, CI [0.0519, 0.1755] | `indep_external_results.json` | ✅ |
| Reliability gate *hurts* stack, Holm p<10⁻⁸ | −0.0374, p_holm 4.96e-09 | `statistical_audit.json` | ✅ |

**Self-audit is genuine.** `GAP_AUDIT.md` (2026-06-14) is an adversarial static audit whose own summary is *"No fabrication was found,"* with **zero CRITICAL** items, while still flagging real MAJOR gaps (a Thm-3 number reported at the wrong α; missing Camelyon17 SAR baseline; an undelivered "certified" natural-shift win; single-seed natural-shift tracks; "ImageNet-C" is actually 36 noise-only cells; two named baselines are surrogates). `INTEGRITY_FIXES.md` claims fixes — and they are **physically present on disk** (e.g. `results_thm3_evalue_alpha005.json` now shows worst-case false-adapt 0.0316 ≤ 0.05). `audits/training_truth_audit/` self-reports two **Critical** training bugs (best val weights tracked but never restored before test eval) and their fix — a self-incriminating log, not a whitewash.

**Reproducibility surface is solid:** `rebuild_kbound.sh`, `rebuild_paper.sh`, `smoke_kbound.sh`, `download_all_datasets.sh` all exist; `kga` CLI is wired (`kga/__main__.py`); `experiments/elara_u/sha256sums.txt` (38 hashes) + `manifest.json` give artifact integrity; **`research_lock/`** (~45 files) is a genuine pre-registration with an enforced immutability rule, a `forbidden:` action list, and a banned-phrase list ("universal SOTA", "always robust"). Raw datasets are actually present in `data/raw/` (33 dataset dirs), so this checkout is closer to fully reproducible than the "~88 GB download required" caveat implies.

**Honest negatives are first-class, not buried:** reliability gating hurts the stack; drift routing is non-significant on single-input / natural-shift data; naive averaging doesn't beat best-fixed; BAF fraud and time-series are explicitly marginal. Healthcare gap-closures are repeatedly captioned "local replay readiness, not prospective clinical deployment."

**Discrepancies found — all minor, none fabrication:**
- README says "**39/39 theorem tests pass**," but every internal doc (`THEOREM_CODE_STATUS.md`, `REPO_HEALTH.md`, the `kbound.tex` source) says "**33/33**." Unreconciled status-line drift, not a result claim.
- README/`DATA.md` frame raw data as an 88 GB download, yet `data/raw/` is fully populated here (a harmless under-claim).
- Acknowledged-but-open in the project's own ledger: Camelyon17 SAR omission, uncertified natural-shift win, single-seed natural-shift coverage, surrogate baselines.

---

## 6. Disk usage & cleanup

The drive is **95% full — 1.8 TB used of 1.9 TB, 108 GB free** — so reclaiming space is worthwhile. Sizes below are from `du`; `.venv`, `data/`, and `experiments/` were too large/slow to size within the tool's time budget on this exFAT volume (`data/` is the legitimate multi-GB raw-dataset store and should be kept).

**Safe to delete now (regenerable junk, all gitignored):**
- `wget-log`, `wget-log.1/.2/.3` — **≈79 MB** (78,851,420 bytes) of stray download logs at repo root.
- `.ruff_cache/` (6.8 MB), `.pytest_cache/` (1.3 MB), `.virtual_documents/` (384 KB), `.tex_build_universal/` (2.0 MB) — tool caches.
- macOS `._*` sidecar files — 84 found within two levels alone (hundreds across the tree); an exFAT/macOS resource-fork artifact, safe to strip.

**Review before acting:**
- **`.git/` is 1.1 GB** — large for a code repo, indicating big blobs in history. A `git gc --aggressive` will help; permanently shrinking it requires history rewriting (`git filter-repo`), which is destructive and optional.
- **`src/mlruns/` is committed** — should be added to `.gitignore` and removed from tracking.
- **`.venv/`** — fully regenerable from `requirements*.txt`; with torch + tensorflow it is typically several GB. Safe to delete and rebuild, but only when you don't need the current environment.

**Keep:** `data/` (raw datasets), `experiments/` (result JSONs + score caches feeding the paper), `docs/` (175 MB — manuscripts/PDFs/figures), `src/` (192 MB), `tests/` (81 MB), `notebooks/` (12 MB).

One-shot cleanup of the clearly-safe items (≈90 MB, no history rewrite):
```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
rm -f wget-log wget-log.1 wget-log.2 wget-log.3
rm -rf .ruff_cache .pytest_cache .tex_build_universal .virtual_documents
find . -name '._*' -type f -delete          # strip macOS sidecars
git gc --aggressive --prune=now             # compact .git (non-destructive)
```

---

## 7. Prioritized recommendations

**P0 — correctness of the build/CI path**
1. Fix the fictional dependency pins in `requirements-api.txt` and `ci.yml` (and the `httpx2` → `httpx` typo everywhere). Until then the Docker image and main CI are non-functional; `kbound-ci.yml` is the only trustworthy lane.

**P1 — hygiene that affects others' trust**
2. Reconcile the "39/39" vs "33/33" theorem-test count between `README.md` and the internal status docs (add it to the existing claim-consistency CI check).
3. `.gitignore` + untrack `src/mlruns/`; delete the empty `src/uais_v/` and refresh `egg-info`; collapse the duplicate pytest config (`pyproject.toml` vs `setup.cfg`).
4. Run the safe disk cleanup in Section 6 (~90 MB) and `git gc`.

**P2 — structural debt (larger effort)**
5. Break up `run_breakthrough_experiment.py` (3,016 LOC) and consolidate the ~160 `src/scripts/` one-offs.
6. Pick one home for the K-Bound certificate (keep `kga/`, generate the `docs/.../kbound_pkg` copy at build time instead of committing it).
7. Pin a single supported Python version across configs/CI.

---

## 8. How this was analyzed

Read directly: `README.md`, `CODEBASE_MAP.md`, plus `du`/`find`/`git` probes on the volume. Two independent sub-audits cross-checked the code (module layout, tests, dependencies, CI, large files, debt counts) and the research (claims-vs-artifacts on the experiment JSONs, the self-audit docs, theorem/proof status, `research_lock` pre-registration). Findings above cite specific files and the actual on-disk values. Nothing was modified; experiments and `pytest` were not executed (out of scope and impractical on this drive), so test-pass counts are reported as the repo states them, not independently re-run.
