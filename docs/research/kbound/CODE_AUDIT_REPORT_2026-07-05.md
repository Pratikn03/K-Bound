# K-Bound — Code Quality & Security Audit

**Date:** 2026-07-05
**Scope:** the K-Bound / KGA repository (`AutoML_Flagship_V8`), excluding `archive/` and `vendored_from_elara/`.
**Method:** static analysis — structural inventory (`git ls-files`), pattern scans (ripgrep) for unsafe
deserialization, dynamic execution, shell injection, hardcoded secrets, and network exposure; targeted
reads of the typed core (`kga/`), the deploy API (`deploy/api/`), and CI/lint/type config.
**Not covered:** running the full test suite, `ruff`/`mypy`, or dynamic (DAST) testing of the API — the
sandbox lacks `torch`/`scipy`/`sklearn` and a GPU. This is a source-level audit, not a runtime pentest.

---

## 1. Verdict

- **AI slop? No.** This is a mature, disciplined research codebase: ~47.6k lines of first-party Python
  across 314 files, a **typed core package** gated by `mypy`, **104 test files**, two CI workflows, and
  the project's own automated code / formal / assumption-audit tooling. That is the opposite of slop.
- **Security posture: strong.** A static scan surfaced **no high-severity issues** — no `eval`/`exec`,
  no `os.system`, no `shell=True`, no unsafe `yaml.load`, and **no hardcoded secrets**. The deploy API
  has a properly hardened auth + CORS layer.
- **One low-severity finding:** six research runners call `torch.load(..., weights_only=False)`
  (pickle deserialization of checkpoints). Real-world risk is low (own checkpoints), and a **production
  gate already enforces safe loading** — but the research runners should be brought in line.

---

## 2. Is the code AI slop? — Evidence

| Signal | Finding | Reading |
|---|---|---|
| Size / structure | 314 py files, ~47,609 LOC (ex-archive); typed core `kga/` = 2,185 LOC | Substantial, organized |
| Tests | 104 test files, incl. integrity tests that *block placeholders* (`test_family_d_v2_no_placeholders_before_freeze`, `..._manifest_no_placeholders`, `test_family_d_v1_never_executable`) | Real, enforcing suite |
| CI | `.github/workflows/ci.yml` + `kbound-ci.yml`: repo-wide critical-error ruff gate, **ruff + mypy on the typed `kga` core**, pytest on the certificate drift guard | Enforced quality gates |
| Types / lint | `pyproject.toml` with `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest]`, `[tool.coverage]`, `[tool.black]`, `[tool.isort]`; `py.typed` marker; pre-commit config | Professionally configured |
| Documentation | Core modules carry substantial docstrings tying code to specific paper theorems and validator scripts | Not boilerplate |
| Provenance | `kga/certificate.py` *openly attributes* the shared Maurer–Pontil certificate lineage with the companion ELARA work rather than hiding it | Honesty signal |
| Self-audit | `code_audit_uav.py`, `formal_audit.py`, `assumption_audit/`, `audit_gate_p_production.py`, `patch_bugs.py`/`verify_after_patch.py` | Unusual engineering rigor |
| Stubs/TODOs | 36 hits across 19 files — several are the *anti-placeholder tests themselves*; core `kga/` has none | Very low; core is clean |

The typed core (`kga/certificate.py`) is representative: a math-grounded module docstring, `from __future__
import annotations`, four clearly-named estimators (empirical-Bernstein, Hoeffding, split-conformal,
anytime e-process), each cross-referenced to the paper theorem and validator it mirrors, "pure numpy/math,
deterministic, torch-free." This reads like careful human research engineering.

---

## 3. Security findings

| ID | Severity | Finding | Status |
|---|---|---|---|
| S1 | **Low** | `torch.load(..., weights_only=False)` in 6 research runners (pickle → arbitrary code execution if a `.pt` is malicious) | Open (research paths) |
| S2 | Pass | No `eval` / `exec` / `os.system` / `shell=True` / unsafe `yaml.load` anywhere in first-party code | — |
| S3 | Pass | No hardcoded secrets; auth reads `UAIS_SECRET_KEY` / `UAIS_API_KEYS` from env and **fails safe** if unset | — |
| S4 | Pass | `subprocess` calls use list-form args (e.g. `["git","rev-parse","HEAD"]`) — no shell-injection surface | — |
| S5 | Pass | Deploy API: API-key + JWT auth, passlib password hashing; **CORS rejects wildcard in production** and requires explicit origins | — |
| S6 | Info | Confirm `UAIS_CORS_ORIGINS`, `UAIS_API_KEYS`, `UAIS_SECRET_KEY` are actually set in any real deployment (the code enforces this only when `PRODUCTION_MODE`) | Ops checklist |

### S1 detail (the `torch.load` / FutureWarning you saw)
The warning printed during training —
`You are using torch.load with weights_only=False … can execute arbitrary code during unpickling` —
comes from loading model checkpoints via pickle. Files affected (first-party):
`run_iwildcam_kbound.py`, `run_iwildcam_aetta.py`, `run_iwildcam_streaming_pilot.py`,
`run_geoshift_kbound.py`, `run_rxrx1_kbound.py`, `_validate_f0.py`.

- **Actual risk: low.** These load checkpoints the project generated itself. The danger only materializes
  if a checkpoint is fetched from an untrusted source and tampered with (relevant for downloaded WILDS f0
  weights).
- **Already mitigated in production:** `src/scripts/audit_gate_p_production.py` gate **P4** requires
  `weights_only=True` **plus** a SHA-256 checksum (`_sha256`) and `_artifact_is_trusted` on the serving
  path. So the deploy path is hardened; only the research runners lag.
- **Fix:** set `weights_only=True` (state-dict checkpoints support it), or
  `torch.serialization.add_safe_globals([...])` for objects that need it, and checksum any downloaded
  weights before load. Removes the warning and closes S1.

---

## 4. What was NOT verified (honesty)

- The full `pytest` suite, `ruff`, and `mypy` were **not executed here** (no `torch`/`scipy`/`sklearn`/GPU
  in the audit sandbox). CI is configured to run them; confirm CI is green on the current commit.
- No dynamic/DAST testing of the FastAPI service (auth bypass, rate-limit, fuzzing).
- Not every one of 314 files was read line-by-line; this is a pattern-driven scan plus targeted reads of
  the highest-risk surfaces (deserialization, auth, subprocess, secrets, core logic).

---

## 5. Recommendations (prioritized)

1. **Close S1:** switch the six research runners to `weights_only=True` (or `add_safe_globals`), and
   checksum any externally-downloaded checkpoints. One-line-per-file change; silences the FutureWarning.
2. **Confirm CI green** on the current commit (ruff critical-error gate + `mypy kga` + the certificate
   test) and, ideally, expand the pytest job beyond the single drift-guard test in `ci.yml`.
3. **Deployment checklist:** ensure `UAIS_SECRET_KEY`, `UAIS_API_KEYS`, `UAIS_CORS_ORIGINS` are set and
   `PRODUCTION_MODE` is on wherever the API is exposed (the code enforces safe values only in that mode).
4. **Optional:** add `pip-audit`/`safety` (dependency CVE scan) and `bandit` (Python security linter) to
   CI for continuous coverage of S1-class issues.

---

## Appendix — raw signal counts

```
first-party python files (ex-archive/vendored) : 314
first-party LOC                                : 47,609
test files                                     : 104
typed core kga/ LOC                            : 2,185
CI workflows                                   : 2  (ci.yml, kbound-ci.yml)
eval/exec/os.system/shell=True/yaml.load       : 0
hardcoded secrets                              : 0  (all via os.getenv)
torch.load(weights_only=False), first-party    : 6 files  (S1)
Lean core: theorems/lemmas / sorry / admit / axiom : 82 / 0 / 0 / 0
```
