# K-Bound v0.1.0

First tagged release of **K-Bound / KGA** — a theory and finite-sample certificate
for deciding, **without target labels**, whether to adapt, freeze, or abstain under
distribution shift.

## What's included
- **`kga/`** — pure-numpy certificate core (`pip`-installable as `kbound`): evidence
  → certificate (Δ̂ ± ε) → adapt/freeze/abstain decision, with the false-adapt rate
  controlled at level α.
- **Theory** — impossibility result (the benefit sign is unidentifiable under matched
  label-free evidence), the exact benefit-sign frontier, and the finite-sample
  certificate (per-theorem validators included).
- **Papers** — conference paper (`kbound.pdf`) and short version (`kbound_short.pdf`).

## Verified empirical results (honest scope)
- **Office-Home (Protocol M v2)** — CI-robust beats-both real natural shift
  (protects against harmful adaptation *and* preserves useful adaptation; FA 0%).
- **iWildCam (Protocol H v2)** — damage-prevention on a harmful-dominated shift.
- **CIFAR-10-C / ImageNet-C** — beats both trivial policies in collapse-prone regimes.
- **62-task anomaly breadth** — correct decisions with 0 false-adapts.
- Honest nulls recorded (fMoW, PovertyMap, ImageNet-R as the theory-predicted
  *unknowable* regime). The method is **safety insurance**: it ties always-adapt on
  helpful-dominated shifts (the no-regret guarantee holding) and wins where
  catastrophic, detectable harm is present.

## Engineering hardening in this release
- Fixed dependency hazards (typosquat `httpx2`/`httpcore2` → `httpx`/`httpcore`;
  non-existent version pins corrected); resolvability verified.
- Security gates: `pip-audit`, bandit on `src/`, gitleaks pre-commit, Dependabot.
- Certificate **single-source-of-truth** drift guard (CI fails if the vendored copy
  diverges from `kga`).
- CI matrix (py3.11/3.12) + hermetic smoke + theorem validators as required gates.
- Repository slimmed (~700 GB of re-downloadable raw data removed; download scripts
  retained; results reproduce from cached artifacts).

## Reproduce
```bash
bash scripts/smoke_kbound.sh          # hermetic, no data, < 60s
pytest tests/test_certificate_drift_guard.py -q
```
Raw datasets are re-downloadable via the scripts referenced in `DATA.md`.
