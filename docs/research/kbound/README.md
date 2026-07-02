# K-Bound / KGA — Repository Guide

**Start here if you are new.** This repo couples a theory paper, a Python certificate
implementation, locked experiment artifacts, and a small notebook curriculum.

| If you want… | Open this |
|--------------|-----------|
| **Doc map (what to read)** | [`DOCS_INDEX.md`](DOCS_INDEX.md) |
| **Complete picture (theory → proof → results)** | [`THEORY_TO_CODE_MAP.md`](THEORY_TO_CODE_MAP.md) |
| **Full theory audit** | `bash docs/research/kbound/scripts/theory_audit_full.sh` |
| **Re-run all 25 validators (~6 min)** | `bash docs/research/kbound/scripts/theory_audit_full.sh --run-validators` |
| **Run the tour in Jupyter** | [`../../notebooks/00_KBound_Master_Guide.ipynb`](../../notebooks/00_KBound_Master_Guide.ipynb) |
| **Canonical project status** | [`PROJECT_STATUS_AND_OPEN_PROBLEMS.md`](PROJECT_STATUS_AND_OPEN_PROBLEMS.md) |
| **Theory core closure gate** | [`THEORY_100_PERCENT_CLOSURE_PLAN.md`](THEORY_100_PERCENT_CLOSURE_PLAN.md) + `cd formal && python3 formal_audit.py --strict-core` |
| **Wave 4 validators** | `cd theory_v2 && .venv/bin/python val_*.py` (see `THEORY_100_PERCENT_CLOSURE_PLAN.md`) |
| **Reviewer reproduction** | [`REVIEWER_REPRO_PACKET.md`](REVIEWER_REPRO_PACKET.md) |
| **Short paper PDF** | [`kbound_short.pdf`](kbound_short.pdf) |
| **Long paper PDF** | [`kbound.pdf`](kbound.pdf) |
| **One-command integrity check** | `bash docs/research/kbound/scripts/reproduce_submission.sh` |
| **Monorepo health (all kbound tests)** | `bash scripts/monorepo_health.sh` |
| **Architecture / SSoT map** | [`../../../MONOREPO.md`](../../../MONOREPO.md) |

---

## What this project is (one paragraph)

Label-free test-time adaptation can help or hurt. **K-Bound** proves when the **sign of
adaptation benefit** is identifiable from evidence alone, proves an **impossibility** when
it is not (minimal supplement = one bit), and implements a **certificate** that controls
**false adaptation** (FA_u ≤ α). Empirically: **beats-both** on synthetic CIFAR stress
grids and vs **POEM/AETTA** on a pre-registered mixed benchmark; **no-harm** on five
natural shifts (Office-Home, iWildCam, Camelyon, RxRx1, PACS).

---

## Directory map

```
docs/research/kbound/
  kbound_short.tex / kbound.tex     # papers (source of truth for claims)
  claim_ledger.json                 # every claim → artifact → allowed wording
  PROJECT_STATUS_AND_OPEN_PROBLEMS.md
  THEORY_TO_CODE_MAP.md             # theory ↔ code ↔ results (this guide's deep dive)
  kbound_pkg/kbound/                # frozen repro snapshot (edit kga/, re-vendor here)
  scripts/                          # CANONICAL scorers, kbtrain.sh, reproduce_submission.sh
  theory_v2/                        # Wave 4 closures + validators (strict core)
  formal/                           # Lean 4 mechanization + formal_audit.py
  edge/                             # physical camera protocol (R2 pending)

experiments/kbound/
  results/                          # locked JSON results (headline numbers)
  theory_validation/                # numeric theorem checks (val_thm*.py)
  poem_aetta/                       # POEM/AETTA head-to-head harness

notebooks/
  00_KBound_Master_Guide.ipynb      # START HERE (2026-06, current)
  01–09_*.ipynb                     # topic notebooks (some predate POEM/AETTA integration)
```

---

## Notebook curriculum (11 files)

| # | Notebook | Role |
|---|----------|------|
| **00** | `00_KBound_Master_Guide.ipynb` | **Master map** — theory spine, claim ledger, headline numbers |
| 00b | `00_KBound_Reproduction.ipynb` | Older 123-task ELARA reproduction (partially stale) |
| 01 | `01_Problem_and_Theory.ipynb` | Theorem validators + Le Cam / regret JSON |
| 02 | `02_Knowability_Trichotomy.ipynb` | Adapt / freeze / abstain demos |
| 03 | `03_Harmful_Mixed_Rigor.ipynb` | Harmful + mixed regimes |
| 04 | `04_Regression_and_Witness.ipynb` | Witness / regression constructions |
| 05 | `05_TTA_CIFAR_and_Online.ipynb` | CIFAR TTA stress grid |
| 06 | `06_Evidence_and_Drift.ipynb` | Natural-shift evidence |
| 07 | `07_Certificate_and_Calibration.ipynb` | Certificate + conformal |
| 08 | `08_ELARA_Multimodal_Instantiation.ipynb` | Multimodal D33 |
| 09 | `09_Conclusions_and_Reproducibility.ipynb` | Artifact inventory |

Legacy `notebooks/legacy_elara/` is the superseded ELARA fraud/cyber EDA stack — not K-Bound.

---

## Theory: what is proven vs open

**Proven (in paper + Wave 4):** frontier identifiability, impossibility / one-bit dichotomy,
certificate FA_u control, unconditional weakest one-bit class, `conj:gen` resolved
negatively, anytime + multicandidate extensions, tight constants, minimax optimality,
multiclass capacity impossibility, margin-computability dichotomy, regression bracketing closure.

**Lean scope:** `formal/` kernel-checks the algebraic theorem spine and finite-sample bridge
lemmas. It does not claim a full foundational Mathlib probability development of exchangeability,
optional stopping, product KL/TV, or martingale rates.

**Open (not blocking submission):** physical camera R2 (KB-CLAIM-030), external reviewer sign-off.

See [`PROJECT_STATUS_AND_OPEN_PROBLEMS.md`](PROJECT_STATUS_AND_OPEN_PROBLEMS.md) §1.

---

## Quick reproduce

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8

# Integrity + tables + artifact checks (~2 min, CPU)
bash docs/research/kbound/scripts/reproduce_submission.sh

# Human-readable complete map (macOS: use bash wrapper or python3 — not bare `python`)
bash docs/research/kbound/scripts/kbound_tour.sh

# Headline empirics (cached, seconds)
PY=.venv/bin/python bash experiments/kbound/poem_aetta/run_all_headtohead.sh
.venv/bin/python docs/research/kbound/scripts/mixed_stream_kbound.py
```

Optional GPU refresh (9 datasets):

```bash
# ~0.5% smoke, single seed (separate output dir; ~1h)
bash docs/research/kbound/scripts/kbtrain.sh smoke-all

# ~1% multiseed smoke + pipeline report (recommended pre-flight; ~2–3h)
KB_SMOKE_SEEDS="0 1" KB_DEVICE=mps \
  bash docs/research/kbound/scripts/run_smoke_showcase.sh

# Full multi-seed rerun + paper rebuild (many hours)
KB_SEEDS="0 1 2 3 4" KB_DEVICE=mps \
  caffeinate -is bash docs/research/kbound/scripts/run_final_showcase.sh \
    --device mps --seeds "0 1 2 3 4"
```
