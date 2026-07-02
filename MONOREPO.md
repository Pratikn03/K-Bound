# Monorepo architecture — AutoML_Flagship_V8

**Engineering target:** 10/10 — one spine, one test matrix, one canonical path per artifact.

Canonical repo root: `/Volumes/T9/uav/AutoML_Flagship_V8`  
(Empty stubs at `/Volumes/T9/uav/{src,tests,docs,research_lock}/` are placeholders — ignore.)

---

## Package map (single source of truth)

| Layer | Path | Role | Edit here? |
|-------|------|------|------------|
| **KGA algorithm** | `kga/` | Typed, torch-free certificate + trichotomy | **Yes** (canonical) |
| **Frozen repro** | `docs/research/kbound/kbound_pkg/` | Byte-stable `kbound` for reviewers | Re-vendor only |
| **Paper scorers** | `docs/research/kbound/scripts/` | Headline experiment runners | **Yes** |
| **Theory validators** | `experiments/kbound/theory_validation/` | `val_thm*.py` (CI-gated) | **Yes** |
| **WILDS / POEM harness** | `experiments/kbound/` | Locked JSON + runners | **Yes** |
| **Edge camera** | `docs/research/kbound/edge/` | `kbound_edge` pipeline | **Yes** |
| **Integration wrappers** | `src/scripts/kbound/` | Thin delegates to canonical scripts | Wrappers only |
| **Legacy product** | `src/uais/` | Anomaly/fusion stack (optional) | Separate concern |
| **Protocol locks** | `research_lock/` | Pre-registered YAML/JSON | Human + script |

**Rule:** If a script exists in both `docs/research/kbound/scripts/` and `src/scripts/kbound/`, the **docs path is canonical**. The `src/` copy must be a thin wrapper (`_canonical.py`).

---

## Three experiment layers (resolved)

```
docs/research/kbound/scripts/   ← CANONICAL scorers (kbtrain.sh, reproduce_submission.sh)
experiments/kbound/             ← harness + locked results + theory_validation
src/scripts/kbound/             ← integration wrappers + smoke (delegates to docs/)
```

Do **not** add a fourth copy. `kbound_only/` is a lagging export — sync or deprecate before public release.

---

## Tests (unified pytest)

All K-Bound tests run from repo root:

```bash
bash scripts/monorepo_health.sh          # quick gate (~2 min)
bash scripts/monorepo_health.sh --full   # + reproduce_submission.sh
```

`pyproject.toml` `testpaths` includes:

- `tests/` — root suite (kga, ELARA, UAIS)
- `docs/research/kbound/tests/` — leakage / protocol integrity
- `docs/research/kbound/kbound_pkg/tests/` — frozen package contract
- `docs/research/kbound/edge/tests/` — camera pipeline

CI: `.github/workflows/kbound-ci.yml` job `kbound-research-tests`.

---

## PYTHONPATH convention

```bash
export PYTHONPATH=".:src:docs/research/kbound/kbound_pkg:docs/research/kbound/edge/src"
```

| Import | Needs |
|--------|-------|
| `import kga` | repo root |
| `import uais`, `import src.elara` | `src/` |
| `import kbound` | `docs/research/kbound/kbound_pkg` |
| `import kbound_edge` | `docs/research/kbound/edge/src` |

---

## Health checks

| Check | Command |
|-------|---------|
| Quick monorepo | `bash scripts/monorepo_health.sh` |
| Paper repro | `bash docs/research/kbound/scripts/reproduce_submission.sh` |
| Theory audit | `bash docs/research/kbound/scripts/theory_audit_full.sh` |
| Code audit | `python docs/research/kbound/scripts/code_audit_uav.py --write-report` |

---

## Engineering scorecard (10/10 criteria)

| Criterion | Status |
|-----------|--------|
| Canonical SSoT documented | `MONOREPO.md` + `kbound_pkg/README.md` |
| No divergent headline scorers | wrappers in `src/scripts/kbound/` |
| All kbound tests in pytest + CI | `pyproject.toml` + `kbound-ci.yml` |
| Hermetic smoke < 60s | `scripts/smoke_kbound.sh` |
| Theorem validators in CI | `kbound-ci.yml` `theorem-validators` + `theory_v2` Wave 4 |
| Lean strict-100 in CI | `kbound-ci.yml` `lean-formal` |
| Drift guard kga ≡ kbound_pkg | `tests/test_certificate_drift_guard.py` |
| Protocol locks + claim ledger | `research_lock/`, `claim_ledger.json` |
| One-command health gate | `scripts/monorepo_health.sh` |

See `docs/research/kbound/reports/MONOREPO_ENGINEERING.md` for the graded audit.
