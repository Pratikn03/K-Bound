# K-Bound — Level-80 Repo Finalization, Architecture & Security Hardening Plan

*Generated 2026-06-26 (multi-agent: architecture + security + git/CI). **PLAN ONLY — no code changed.** Execution is gated on your confirmation per phase (§6). Pair this with `REPO_CLEANUP_PLAN.md` (the disk/data tiers).*

---

## 0. Principles (and one honest correction)

You asked for a repo that is "OOP/DSA complex, harder to hack, safety full." One correction, because it matters: **a level-80 repo is not made stronger by being complex.** Complexity is the enemy of both security and maintainability. A top-tier research repo is:

- **Simple, clean OOP** with one source of truth — not clever, not layered for its own sake.
- **Reproducible** — anyone can `pip install`, download data via scripts, and regenerate every number.
- **Specifically hardened** — pinned/audited dependencies, safe deserialization, validated API, scanned secrets, CI gates. Security is a property of *controls*, not of obfuscation.

Good news from the audit: your **core (`kga/`) is already clean, typed, and well-tested**, and your **inference API is already well-hardened**. "Level 80" here means *finishing the edges* — dedupe, dependency fix, history slimming, CI gates, release — not rewriting the core.

**Current grade (from `REPO_ANALYSIS_2026-06-16.md`):** research integrity **Strong**, core engineering **Strong**, periphery **Mixed** (broken dep pins, script sprawl, triple-vendored certificate, 800 GB bloat). Closing the periphery gaps is the path to a clean ~80 repo.

---

## 1. Architecture, OOP & Algorithms

**1. Single source of truth for the triple-vendored certificate.** Keep `kga/certificate.py` as canonical: it is the only typed, `py.typed`, mypy-strict, fully-tested copy, and it already carries all four estimators. The other two lineages should be *thin, non-authoritative* satellites — and the wiring is mostly done:
- `src/elara/certification/switching_certificate.py` already delegates `empirical_bernstein_lcb` to `kga.certificate`. Finish the job: delete any remaining re-derived Maurer–Pontil arithmetic there; keep only the elara-specific bootstrap/`SwitchingCertificate` gate machinery, so zero math is duplicated.
- `docs/research/kbound/kbound_pkg/kbound/` is a frozen reproduction snapshot. Stop committing it: generate it at paper-build time from `kga/` via a `rebuild_kbound.sh`, stamped with the `kga.__version__` it was vendored from. Add a CI guard that imports both and asserts numeric identity on a fixed input, so drift fails the build.

**2. Clean OOP for the kga core — already right; do not over-build.** Preserve the current textbook decomposition:
- `Evidence` (frozen dataclass) + `compute_evidence()` — extractor.
- `Certificate` (frozen dataclass, `.lower`/`.upper`) + four free estimator functions — the Δ̂±ε layer.
- `Decision` (str-Enum) + `decide()` — the trichotomy.
- `KGA` — a thin stateless facade (`evidence → certify → decide → explain`).

Resist adding an `Estimator` ABC/registry, a plugin system, config objects, or a base `Certifier` class — the `_BATCH_ESTIMATORS` dict already gives polymorphism without inheritance, and free functions keep estimators independently testable. **The biggest risk to this package is gratuitous abstraction.** Keep it free-function-first, numpy/scipy-only, `from __future__ import annotations` everywhere. Small typing nit: use `TypedDict`/`Mapping` for `explain()` returns.

**3. DSA/efficiency that actually matters.** `conformal_split` is O(n log n) (quantile sort) — fine; only use `np.partition` (O(n)) if n is huge. The real hotspot is `evalue_anytime`: a Python loop over the stream (bets must stay F_{i-1}-measurable). The genuine win is a true **streaming/online certificate** — incremental Welford mean/variance so empirical-Bernstein updates O(1) per sample and the gate fires mid-stream. Vectorize only the embarrassingly-parallel parts (KS per detector, pairwise rank-corr). Don't over-engineer the batch path.

**4. Split the 3,016-line monolith** `src/scripts/run_breakthrough_experiment.py` into a small package: `data_loading.py`, `splits.py` (leakage-safe), `scoring.py`, `fusion.py`, `certify.py` (calls `kga`), `metrics.py`, `reporting.py`, slim `__main__.py`. Lift the duplicated `emit_*` table generators from the ~160 one-off scripts into `uais/reporting/`.

**5. Keep as-is.** `kga/` is the cleanest code in the repo — treat its public API as frozen and refactor *around* it.

---

## 2. Security & Hardening

**Threat model.** Two assets: a research artifact (training/eval code, model checkpoints) and a small authenticated inference service exposing `POST /decide`. Realistic adversaries: (1) a **malicious model/pickle file** that executes code on load; (2) **dependency confusion / typosquatting** via unresolvable or attacker-registrable package names; (3) **unvalidated API input** (oversized/NaN/crafted payloads) causing DoS or error leakage; (4) **secret leakage**. Security is a property of explicit controls, not code intricacy.

**Already strong — keep:** `deploy/api/main.py` uses `weights_only=True`, checksum-gated loads (`TRUSTED_MODEL_ARTIFACTS` off by default), pydantic validation with finite/range checks and size caps, fixed-window rate limiting, request timeout, generic error responses (no stack traces), constant-time `hmac.compare_digest` key check, env-only secrets, fail-closed production startup. The Dockerfile is least-privilege (non-root, pinned base **digest**, deps purged, `HEALTHCHECK`, localhost bind).

**Prioritized fixes (checklist):**

- [ ] **(a) Dependencies — highest priority.** Replace the fictional pins in `requirements-api.txt` and `ci.yml` (`httpx2`, `httpcore2`, `pandas==3.0.3`, `certifi==2026.5.20`, `scikit-learn==1.8.0`, `starlette==1.2.0`, `torch==2.12.0`, `pyarrow==24.0.0`) with **real, resolvable versions**; pin the loose ranges in `requirements.txt`. Generate a hash-locked file (`pip-compile --generate-hashes` / `uv lock`), install with `--require-hashes`. Add **pip-audit** as a CI gate and **Dependabot** for pip + Actions.
- [ ] **(b) Safe deserialization beyond the API.** Extend the API's discipline to `src/`: add `weights_only=True` to every `torch.load` in `src/scripts/kbound/*` (10+ scripts); replace bare `pickle.load` (`frozen_calibrators.py`, `load_datasets.py`) and `joblib.load` with checksum/allowlist-gated loaders; always `yaml.safe_load`.
- [ ] **(c) Secret scanning.** Add **gitleaks**/**trufflehog** to pre-commit **and** CI, scan history. Confirmed currently clean (secrets read from env only).
- [ ] **(d) Static analysis in CI.** Expand `bandit` from `deploy/` to `src/`; add **ruff** security rules (`S`/flake8-bandit); run **trivy**/**grype** on the built image.
- [ ] **(e) API edge.** Document the upstream reverse-proxy body-size limit so size enforcement isn't validator-only.

Security comes from these controls — pinned/audited deps, gated deserialization, validated input, scanned secrets — **not** from making the code complex.

---

## 3. Git, CI/CD & Reproducibility

**1. Shrink `.git` (3.2 GB → expected <300 MB).** Bloat is committed `src/mlruns` artifacts + large PDFs baked into history. Purge with `git-filter-repo` (preferred) or BFG: `git filter-repo --path src/mlruns --path-glob '*.pdf' --invert-paths`, then `git reflog expire --expire=now --all && git gc --prune=now --aggressive`. **This is a HISTORY REWRITE:** make a mirror backup (`git clone --mirror`) first, force-push all refs/tags, and have any collaborators re-clone (old clones reintroduce blobs). Serve non-canonical PDFs as release assets.

**2. `.gitignore` completeness.** Already strong. Add explicit `mlruns/` + `src/mlruns/`; generalize PDF ignores to `**/*.pdf` with `!` re-includes for the 2 canonical files; confirm the ~800 GB datasets stay covered (`data/{raw,interim,processed}/`, `experiments/**/data/`).

**3. Packaging hygiene.** Resolve the **two competing pytest configs** — delete `[tool:pytest]` from `setup.cfg`, keep `[tool.pytest.ini_options]` in `pyproject.toml`. Remove the stale `[flake8]`/`[isort]` blocks (Ruff is source of truth) and stale `*.egg-info` (it omits `kga`, declares empty `src/uais_v`). Keep `kga`'s `py.typed`. One Python policy: `requires-python = ">=3.11"`, aligned across ruff/black/mypy.

**4. CI gates.** Fix the broken `ci.yml` install path (use `requirements.lock.txt`). Make the hermetic smoke + every `val_thm*.py` validator **required** checks. Add a **py3.11/3.12 matrix**, gate `ruff check` + `mypy` (on `kga/`). Enable **branch protection** on `main` (required review + required checks) and require **signed commits/tags**.

**5. Release & reproducibility.** Treat `requirements.lock.txt` as the verified repro path (CI installs + asserts it resolves). Promote `DATA.md` + `scripts/download_all_datasets.sh --verify-only` (SHA256 anchors) as the data-repro contract. Publish `kga` to PyPI (`pip install kbound`). Cut a **tagged release** + mint a **Zenodo DOI** as the citable artifact.

---

## 4. What's already level-80 (don't touch)

- `kga/` core: clean, typed, `py.typed`, 700+ tests, mypy-strict, CLI + API.
- `deploy/api/main.py` + Dockerfile: already hardened (see §2).
- `research_lock/` pre-registration + honest negatives: the integrity backbone.
- The cached result JSONs in `experiments/kbound/results/`: the paper's reproducible numbers.

---

## 5. Unified phased roadmap

| Phase | Scope | Risk | Reversible? | Status |
|---|---|---|---|---|
| **A. Tidy** | Tier 1 cleanup (junk, dupes, mlruns, backups) | none | git | ✅ **done** (`8f4090d`) |
| **B. Dependencies + security** | fix fictional pins, lockfile+hashes, pip-audit/Dependabot, weights_only/yaml.safe_load in src/, gitleaks, bandit→src/, trivy | low | yes (PRs) | pending |
| **C. Architecture dedupe** | finish certificate single-source + CI drift guard; split the 3016-line monolith; TypedDict typing | low–med | yes | pending |
| **D. Git + CI** | one pytest config, egg-info cleanup, CI matrix + required gates, branch protection, signed tags | low (except history rewrite) | mostly | pending |
| **E. Data reclaim** | Tier 2/3 from `REPO_CLEANUP_PLAN.md` (~700 GB), then `.git` history rewrite | med | snapshot + re-download | pending |
| **F. Release** | lockfile-verified CI green, `pip install kbound`, tagged release + Zenodo DOI | none | n/a | pending |

Recommended order: **B → C → D → E → F** (security/quality first; the big destructive/history steps last, after a snapshot).

---

## 6. Confirmation gate

Nothing in Phases B–F runs without your go-ahead, phase by phase. Each phase: branch off, make changes as a reviewable diff, run the test suite + hermetic smoke + PDF build, report, then merge on your OK. The history rewrite (Phase E) additionally takes a full mirror backup first.

**Reply with the next phase to execute** (e.g. *"do Phase B"*). I recommend **Phase B** next — it's the highest-value, lowest-risk step and unblocks honest CI.
