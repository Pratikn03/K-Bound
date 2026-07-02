# UAV / K-Bound Code Audit (automated)

Generated: 2026-06-30T01:13:29.669722+00:00

Full line-by-line human review of ~161k LOC is not feasible in one pass; this audit targets integrity + mirrors + hygiene.

## Scale

- **AutoML_Flagship_V8**: 996 Python files, ~160,049 lines
- **kbound_only**: 254 Python files, ~42,285 lines
- **uav_root_src**: 1 Python files, ~0 lines

## Stale / duplicate layout

- /Volumes/T9/uav/tests/ is empty placeholder — use AutoML_Flagship_V8/
- /Volumes/T9/uav/docs/ is empty placeholder — use AutoML_Flagship_V8/
- /Volumes/T9/uav/research_lock/ is empty placeholder — use AutoML_Flagship_V8/
- Root kbound.log/aux are stale LaTeX artifacts (Jun 15); canonical papers in AutoML_Flagship_V8/docs/research/kbound/
- **kga**: identical {'automl': 11, 'kbound_only': 11}
- **src/scripts/kbound**: diverged {'automl': 15, 'kbound_only': 14}
- **kbound_pkg**: diverged {'automl': 14, 'kbound_only': 13}
- **poem_aetta**: skip 

## Headline scorer integrity (G1)

- ✓ `docs/research/kbound/scripts/cifar_tent_mps_v2.py`
- ✓ `docs/research/kbound/scripts/score_kbound_holdout.py`
- ✓ `docs/research/kbound/scripts/mixed_stream_kbound.py`
- ✓ `docs/research/kbound/scripts/pacs_vlcs_runner.py`
- ✓ `docs/research/kbound/kbound_pkg/kbound/certificate.py`
- ✓ `experiments/kbound/poem_aetta/run_mixed_headtohead.py`

**Scorers OK:** True

## Canonical wrappers (src → docs)

- ✓ `src/scripts/kbound/cifar_tent_mps_v2.py`
- ✓ `src/scripts/kbound/cifar_tent_online.py`

**Wrappers OK:** True

## Monorepo engineering

- ✓ canonical_wrappers
- ✓ headline_scorers_g1
- ✓ unified_pytest_paths
- ✓ ci_kbound_research_tests
- ✓ monorepo_health_script
- ✓ research_tests_pass

**Monorepo grade:** 10.0/10 (6/6 criteria)

## Tests
- K-Bound research pytest: -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html

## Secret pattern scan (sample)
- No hardcoded secret patterns in K-Bound paths (sample scan).

## What a full line-by-line audit would still need

- Human review of `kga/` vs `kbound_pkg/` drift (documented in kbound_pkg/README)
- `kbound_only/` sync or deprecate before public dual-repo confusion
- Audit-only scripts under `audits/` and `theory_v2/realdata/eps_recal/` (in-sample OK for diagnostics)
- ELARA/UAIS code under `src/uais/` (legacy product stack, not paper spine)
