# K-Bound reproducibility & artifact-authority hardening

Point-in-time engineering report. The **machine-checkable** authority is the
tooling under `kbound_repro/` + `claim_ledger.json`; this document is a summary
and may lag the code — trust the tests, not this prose.

Generated: 2026-07-21. Branch `main` → upstream `origin/flagship-history`.

## Outcome

Engineering weaknesses were addressed **without changing any scientific
conclusion, rerunning completed experiments, or touching active training**. A
new torch-independent toolkit (`docs/research/kbound/kbound_repro/`) now provides
the canonical metrics, schemas, device/runtime selection, path resolution,
dependency handling, storage policy, and the claim-authority chain, all covered
by 74 passing unit tests.

**Blocker (unchanged by this task):** the PACS seeds 1–2 and ImageNet-R
Protocol D seed 3 runs were **active and incomplete** during this work
(`experiments/kbound/results/per_cell/`, `.../imagenetr_protocol_d_seed3_v1/`).
The canonical numerical manifest that depends on those aggregates is therefore
**not yet generated**; the release gate fails closed on its absence (by design).

## Defects fixed (engineering weaknesses 1–9)

1. **Absolute paths** — `kbound_repro/paths.py` gives repo-root discovery +
   `KBOUND_DATA_ROOT` / `KBOUND_RESULTS_ROOT` / `KBOUND_IMAGENETR_ROOT` /
   `KBOUND_PACS_ROOT` env resolution with actionable errors. The new current
   command (`runbooks/release_candidate.sh`) is fully portable and verified by a
   relocation test. **70 legacy/executable scripts still hard-code
   `/Volumes/T9` or `/Users/pratik_n`** and are listed as a remediation backlog;
   several are the *active-training* runners and were deliberately not edited.
2. **Device/runtime** — `kbound_repro/runtime.py`: single selector, order
   requested→CUDA→MPS→CPU, **fails clearly** instead of silently switching, records
   the resolved device for manifests, lazy torch import. No `set_device` calls.
3. **Silent imports** — `kbound_repro/deps.py` (`require`/`optional`) raises named
   `ImportError`s with install hints. `scripts/cifar_tent_mps_v2.py` refactored so
   test **collection no longer defines `Dataset` subclasses through `torch=None`**
   (sentinel `_DatasetBase`), and `_require_torch()` now raises `ImportError`
   (not `sys.exit`). `kbound_pkg/kbound/optimizer.py` already used the correct
   guarded pattern (left as-is).
4. **Duplicated metrics** — `kbound_repro/metrics.py` is the single canonical,
   torch-independent implementation (Δ/B, regrets, FA_u/FA_c, coverage, Wilson,
   paired bootstrap, Holm). Boundary pinned: `FA_u = mean(ADAPT ∧ Δ ≤ 0)` (a tie
   is unsafe); `FA_c` has a separate name/denominator and is descriptive only.
5. **Incompatible schemas** — `kbound_repro/schema.py`: versioned validators for
   per-condition / per-seed / multiseed / ledger / manifest / empirical-metrics,
   plus historical migration that **preserves the original** artifact.
6. **Counts from rounded rates** — schema forbids it: rate-only history migrates to
   `count: null, status: not_retained` (never `round(rate·n)`).
7. **Generated vs manual docs** — `kbound_repro/authority.py` generates the claim
   matrix from ledger + manifest and detects disagreement across ledger /
   manifest / manuscript / TODO; the release build fails on disagreement.
8. **Dirty worktree** — changes are additive and surgical (2 tracked files
   modified). Exact file list in the task report; staging is explicit-path only.
9. **Storage separation** — `kbound_repro/storage.py` + `check_repo.py` guard,
   `.gitignore` extended for `AETTA/` data/caches/checkpoints, and
   `STORAGE_MANIFEST.json` describes tracked vs external artifacts.
   ~1010 files were **historically committed** into external classes (behind the
   6.7 GB `.git`); untracking those from history is a separate explicit operation.

## Scientific guardrails verified

- KB-CLAIM-022 (Camelyon17 pooled) stays **withdrawn**; Camelyon = reconciled OOD
  no-harm only. KB-CLAIM-004 (FA_c≤α), 012 (jackknife+), 023 (13×/24× mixed), 050
  (universal improvement) stay withdrawn.
- Protocol D33 stays controlled mechanism confirmation (KB-CLAIM-027), not a
  natural benchmark.
- Empirical coverage kept separate from theoretical; risk alignment stays an
  assumption; fitting the benefit estimator is not equated with risk alignment.
- The forbidden-wording gate (negation-aware, semantic) scanned the 5 promoted
  manuscript files and found **zero** asserted withdrawn-claim wording — the
  paper correctly *disclaims* each withdrawn claim.

## How to reproduce

```
bash docs/research/kbound/runbooks/release_candidate.sh preflight        # env + datasets
bash docs/research/kbound/runbooks/release_candidate.sh test             # tests + authority gate
bash docs/research/kbound/runbooks/release_candidate.sh all              # full chain (fails closed w/o manifest)
```

Clean environment (do not commit the venv):
`python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.lock.txt`.
