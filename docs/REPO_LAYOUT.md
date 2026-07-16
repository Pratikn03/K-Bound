# Repository layout (K-Bound only)

This repo follows the same shape as clean ML research codebases (PyTorch examples,
Hugging Face `transformers`, papers-with-code releases): **one installable package,
one experiment tree, one paper tree, one lock folder, one test suite.**

## Packages — do not confuse these

| Path | Role | Edit? |
|------|------|-------|
| [`kga/`](../kga/) | **Source of truth** — certificate / evidence / policy / routing | **Yes** |
| [`docs/research/kbound/kbound_pkg/`](../docs/research/kbound/kbound_pkg/) | **Frozen** paper-reproduction snapshot (`import kbound`) | **No** (re-vendor only on deliberate release) |

## Top level (what you should see on GitHub)

| Path | Role |
|------|------|
| [`kga/`](../kga/) | Live certificate core (numpy/scipy, typed) |
| [`docs/research/kbound/`](../docs/research/kbound/) | Paper, Lean formalization, frozen `kbound_pkg/`, edge pilot |
| [`experiments/kbound/`](../experiments/kbound/) | **Canonical** runners, cached `results/`, theorem validators |
| [`docs/research/kbound/experiments/`](../docs/research/kbound/experiments/) | Lightweight run_config stubs only — see its README |
| [`research_lock/`](../research_lock/) | Pre-registered K-Bound protocols + frozen decision log |
| [`scripts/`](../scripts/) | Thin launchers (`smoke_kbound.sh`, `run_natural_win_v1.sh`, …) |
| [`src/scripts/kbound/`](../src/scripts/kbound/) | Canonical experiment scripts (imported by runners) |
| [`tests/`](../tests/) | K-Bound / KGA hermetic tests (ELARA Family tests archived) |
| [`deploy/api/`](../deploy/api/) | KGA FastAPI (`POST /decide`, `/kga/health`); env `KGA_*` |
| [`AETTA/`](../AETTA/) | Vendored CVPR 2024 baseline (plain files; see `VENDOR.md`) |
| [`.github/workflows/`](../.github/workflows/) | `kbound-ci.yml` (primary) + slim `ci.yml` |
| [`archive/legacy_elara/`](../archive/legacy_elara/) | **Read-only** ELARA/UAIS-V artifacts + archived Family tests |

Root files: `README.md`, `LICENSE`, `CITATION.cff`, `DATA.md`, `pyproject.toml`, `requirements*.txt`.

## What we removed from the default view

ELARA multimodal anomaly work (fraud/cyber/fusion dashboard, Family A/B/D locks)
lives under `archive/legacy_elara/` for provenance only. It is **not** part of
the K-Bound scientific claim or CI.

## Nested map (K-Bound)

```
kga/                          # LIVE certificate, evidence, routing
docs/research/kbound/
  kbound_pkg/                 # FROZEN pip reproduction snapshot
  formal/                     # Lean 4 sources (+ local .lake cache, gitignored)
  edge/                       # real-camera pilot
  experiments/                # run_config stubs ONLY
  archive/paper_drafts_*/     # old .bak / Word / 2col drafts
experiments/kbound/           # CANONICAL runners + results + data
research_lock/                # K-Bound protocol YAML + headline locks
```

## Clone expectations

```bash
git clone https://github.com/Pratikn03/K-Bound.git
cd K-Bound   # or AutoML_Flagship_V8 until local folder is renamed
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/smoke_kbound.sh
```
