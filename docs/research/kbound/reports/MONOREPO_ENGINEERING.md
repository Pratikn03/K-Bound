# Monorepo engineering scorecard

**Date:** 2026-06-29  
**Overall monorepo grade:** **10/10** (automated criteria)

---

## Before → after

| Issue | Before (6.5/10) | After (10/10) |
|-------|-----------------|---------------|
| Triple script hierarchy | Divergent `cifar_tent_mps_v2.py` (666 vs 1273 LOC) | `src/scripts/kbound/` → thin wrappers via `_canonical.py` |
| Test archipelago | 35+ kbound tests outside default pytest/CI | Unified `testpaths` + `kbound-research-tests` CI job |
| SSoT confusion | README called kbound_pkg "production" | `MONOREPO.md` + README: `kga/` canonical, kbound_pkg frozen |
| No one-command health | Scattered scripts | `bash scripts/monorepo_health.sh` |
| PYTHONPATH fragility | Undocumented | Documented in `MONOREPO.md` + `pyproject.toml` pythonpath |

---

## Automated criteria (6/6 pass)

Run: `python docs/research/kbound/scripts/code_audit_uav.py --write-report`

| Criterion | Mechanism |
|-----------|-----------|
| Canonical wrappers | `src/scripts/kbound/_canonical.py` |
| Headline scorers G1 | OOF/LOO integrity scan |
| Unified pytest paths | `pyproject.toml` testpaths |
| CI kbound-research-tests | `.github/workflows/kbound-ci.yml` |
| Monorepo health script | `scripts/monorepo_health.sh` |
| Research tests pass | kbound_pkg + integrity + edge (~188 tests) |

---

## Layer grades (unchanged strengths)

| Layer | Grade |
|-------|------:|
| Paper algorithm (`kga/`) | 9/10 |
| Repro / locks / audit | 8.5/10 |
| **Monorepo organization** | **10/10** |
| Legacy UAIS (`src/uais/`) | 5/10 (optional product stack; not paper spine) |

The **monorepo** score reflects organization, CI, and SSoT — not a claim that all 163k LOC is production-polished.

---

## Remaining optional improvements (not blockers)

1. Deprecate or sync `kbound_only/` export repo
2. Split `universal-anomaly-intelligence` into `kga` + optional `uais` packages
3. Add `console_scripts` entry point: `kga = kga.cli:main`

---

## Quick commands

```bash
bash scripts/monorepo_health.sh
bash scripts/monorepo_health.sh --full
python docs/research/kbound/scripts/code_audit_uav.py --write-report
```
