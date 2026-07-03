# Repository layout (K-Bound only)

This repo follows the same shape as clean ML research codebases (PyTorch examples,
Hugging Face `transformers`, papers-with-code releases): **one installable package,
one experiment tree, one paper tree, one lock folder, one test suite.**

## Top level (what you should see on GitHub)

| Path | Role |
|------|------|
| [`kga/`](../kga/) | **Source of truth** — certificate core (numpy/scipy, typed) |
| [`docs/research/kbound/`](../docs/research/kbound/) | Paper, Lean formalization, pip package (`kbound_pkg/`), edge pilot |
| [`experiments/kbound/`](../experiments/kbound/) | Runners, cached `results/`, theorem validators |
| [`research_lock/`](../research_lock/) | Pre-registered protocols + frozen decision log |
| [`scripts/`](../scripts/) | Thin launchers (`smoke_kbound.sh`, `run_natural_win_v1.sh`, …) |
| [`src/scripts/kbound/`](../src/scripts/kbound/) | Canonical experiment scripts (imported by runners) |
| [`tests/`](../tests/) | Hermetic tests + anti-leakage / drift guards |
| [`deploy/api/`](../deploy/api/) | KGA-only FastAPI (`POST /decide`, `/kga/health`) |
| [`.github/workflows/`](../.github/workflows/) | `kbound-ci.yml` (primary) + slim `ci.yml` |
| [`archive/legacy_elara/`](../archive/legacy_elara/) | **Read-only** ELARA/UAIS-V artifacts (not maintained) |

Root files: `README.md`, `LICENSE`, `CITATION.cff`, `DATA.md`, `pyproject.toml`, `requirements*.txt`.

## What we removed from the default view

ELARA multimodal anomaly work (fraud/cyber/fusion dashboard, 40 attention YAMLs,
union research orchestrator, Streamlit UAIS dashboard) lives under
`archive/legacy_elara/` for provenance only. It is **not** part of the K-Bound
scientific claim or CI.

## Benchmark (why this layout)

Patterns taken from well-maintained repos (~top research / OSS norms):

1. **Package at repo root** (`kga/`, not buried under `src/`) — like `torchvision/`, `sklearn/`.
2. **Experiments namespaced** (`experiments/kbound/`) — like `examples/` in PyTorch.
3. **Paper + formal + package co-located** under `docs/research/kbound/` — single citation surface.
4. **Pre-registration outside code** (`research_lock/`) — immutable protocol YAML/JSON.
5. **No generated junk at root** — logs, coverage, scratch JSON gitignored.
6. **Archive, don't delete** — history for audits without polluting the tree.

## Nested map (K-Bound)

```
kga/                          # certificate, evidence, routing
docs/research/kbound/
  kbound_pkg/                 # pip install target
  formal/                     # Lean 4 + formal_audit.py
  gapclose_wave5/             # Wave 5 validators + NATURAL_WIN analysis
  edge/                       # real-camera pilot (optional)
  notebooks/                  # K-Bound Jupyter curriculum
experiments/kbound/
  results/                    # committed JSON evidence
  wilds/                      # Camelyon / ImageNet-R runners
  theory_validation/          # val_thm*.py
research_lock/                # protocol YAML + headline locks
```

## Clone expectations

```bash
git clone https://github.com/Pratikn03/AutoML_Flagship_V8.git
cd AutoML_Flagship_V8
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/smoke_kbound.sh
```

Optional rename: GitHub → Settings → rename repository to `kbound` (URLs redirect).
