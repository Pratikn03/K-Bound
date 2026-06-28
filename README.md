# K-Bound / KGA — Knowability-Guided Adaptation

[![kbound-ci](https://github.com/Pratikn03/AutoML_Flagship_V8/actions/workflows/kbound-ci.yml/badge.svg)](https://github.com/Pratikn03/AutoML_Flagship_V8/actions/workflows/kbound-ci.yml)
[![CI](https://github.com/Pratikn03/AutoML_Flagship_V8/actions/workflows/ci.yml/badge.svg)](https://github.com/Pratikn03/AutoML_Flagship_V8/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Should a model adapt to new data it has no labels for — or would adapting make it worse?**
K-Bound answers that *before* adapting. It formalizes label-free test-time adaptation (TTA)
as an **adapt / freeze / abstain** decision governed by the sign of the adaptation benefit,
and ships a finite-sample certificate, **KGA** (Knowability-Guided Adaptation), that controls
the false-adapt rate at a chosen level α.

It is a **safety layer for TTA**, not a new adaptation method: KGA wraps existing adapters
(Tent, EATA, SAR) and decides *whether* to trust them on this batch.

## Why it exists

The same unlabeled objective that recovers accuracy under one shift can silently destroy it
under another, and with no target labels the system can't tell which it's facing. K-Bound
proves this is partly **fundamental**: when two target worlds produce identical label-free
evidence but opposite adaptation benefit, no label-free rule can be right in both —
abstention is information-theoretically necessary. It also gives the exact frontier: the
benefit sign is recoverable **iff** an observable margin exceeds the calibration-drift budget.

## Install

```bash
# Today (lightweight, numpy + scikit-learn only):
pip install "git+https://github.com/Pratikn03/AutoML_Flagship_V8.git#subdirectory=docs/research/kbound/kbound_pkg"
# After the PyPI release:  pip install kbound
```

## 30-second quickstart

```python
import numpy as np
from kbound.certificate import conformal_radius, decide

# calibration residuals r_i = |Δ̂_i − Δ_i| from a held-out split
residuals = np.abs(np.random.default_rng(0).standard_normal(200)) * 0.05
eps = conformal_radius(residuals, alpha=0.10)        # finite-sample radius

decide(Bhat=0.12,  eps=eps)   # -> 'adapt'    (benefit certified positive)
decide(Bhat=-0.12, eps=eps)   # -> 'freeze'   (benefit certified negative)
decide(Bhat=0.01,  eps=eps)   # -> 'abstain'  (sign not identifiable)
```

## Results — honest scope

KGA's value is **regime-specific**: it wins where harmful adaptation is frequent, detectable,
and costly, and it *ties* (does no harm) where adapting is already the right call. It is **not**
a universal accuracy booster. Every number below is from a pre-registered protocol
(`research_lock/`) scored once on held-out test, with bootstrap confidence intervals.

| Setting | Result | Reading |
|---|---|---|
| **CIFAR-10-C / ImageNet-C** (collapse-prone) | **beats-both, CI-robust** (Tent/EATA; SAR-collapse cells) | the headline win — harmful adaptation is frequent *and* detectable |
| **Office-Home** (Protocol M v2) | **no-harm** — beats always-adapt (CI excl. 0), ties always-freeze; false-adapt 0% | damage-prevention: blocks harm, keeps useful adaptation |
| **iWildCam** (Protocol H v2) | **no-harm** — beats always-adapt, ties always-freeze; false-adapt 0% | damage-prevention on a natural shift |
| **Camelyon17 / RxRx1** | **no-harm** — matches the better fixed policy | one-sided shifts; nothing to beat |
| **PACS** (leave-one-domain-out) | **no-harm on 3/4 domains**; safe partial-adapt on the 4th | domain-generalization breadth check |
| **CIFAR-10.1 / ImageNet-R** | honest nulls | evidence-poor / *unknowable* regime the theory predicts |
| Mixed-deployment stream | *withdrawn* — pending a corrected out-of-fold re-run | not currently claimed |

> Under a valid out-of-fold conformal radius, the only **beats-both** rows are the synthetic
> corruption grids (CIFAR-10-C, ImageNet-C SAR); **every natural shift is no-harm**, not a win.
> An earlier in-sample-radius Office-Home/iWildCam "beats-both" was a calibration bug and is corrected here.

> A previously reported Camelyon17 "beats-both" was traced to pooling in-distribution
> validation cells into the held-out set and **withdrawn** — see the paper's natural-shift section.

## Reproduce

```bash
bash scripts/smoke_kbound.sh                              # hermetic, no data, no torch, <60s
python -m pytest tests/test_certificate_drift_guard.py    # paper's certificate ≡ kga core
python -m pytest tests/test_kga_package.py -q             # the importable certificate
```
Raw datasets are not committed; re-download via the scripts referenced in [`DATA.md`](DATA.md).
Headline numbers reproduce from the cached result JSONs in `experiments/kbound/results/`.

## Papers

- Conference paper: [`docs/research/kbound/kbound.pdf`](docs/research/kbound/kbound.pdf)
- Short version: [`docs/research/kbound/kbound_short.pdf`](docs/research/kbound/kbound_short.pdf)
- Long-form thesis: [`docs/research/kbound/manuscript/`](docs/research/kbound/manuscript/)

## Repository layout

| Path | What |
|---|---|
| [`kga/`](kga/) | The KGA certificate core — pure-numpy, typed, the maintained source of truth |
| [`docs/research/kbound/kbound_pkg/`](docs/research/kbound/kbound_pkg/) | The pip-installable `kbound` package (frozen copy, drift-guarded) |
| [`experiments/kbound/`](experiments/kbound/) | Experiment drivers + cached result JSONs (`results/`) + theorem validators |
| [`research_lock/`](research_lock/) | Pre-registered protocols + decision log (the integrity backbone) |
| [`tests/`](tests/) | Test suite incl. anti-leakage + manuscript-claim-consistency guards |
| [`deploy/api/`](deploy/api/) | Hardened FastAPI service exposing `POST /decide` |
| [`docs/dev-notes/`](docs/dev-notes/) | Archived development notes / audit trail |

`src/uais/` and the ELARA-U anomaly meta-routing benchmark (a worked instantiation K-Bound
generalizes) remain in the tree as foundation and breadth evidence.

## Cite

See [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository" button). A Zenodo
DOI is minted at release — see [`docs/research/kbound/RELEASE_CHECKLIST.md`](docs/research/kbound/RELEASE_CHECKLIST.md).

## License

MIT — see [`LICENSE`](LICENSE). Author: **Pratik Niroula**.
