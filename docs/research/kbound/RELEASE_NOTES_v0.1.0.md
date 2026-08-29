# K-Bound v0.1.0 — historical release record

> **SUPERSEDED FOR SCIENTIFIC CLAIMS (2026-08-29).** This file originally accompanied an early
> engineering release. Its Office-Home, iWildCam, broad CIFAR/ImageNet, and 62-task summaries were
> written before the source-hashed reconciliation and must not be cited as current evidence. The
> original conclusions are preserved below as a correction ledger rather than silently erased.
> Current claim authority is `KBOUND_SHORT_RESULT_AUDIT.md`, `KBOUND_SHORT_CLAIM_MANIFEST.md`,
> `claim_ledger.json`, the canonical reconciled panel, and the separately receipt-linked CCT-20 and
> So2Sat authorities.

First tagged release of **K-Bound / KGA** — a theory and finite-sample certificate
for deciding, **without target labels**, whether to adapt, freeze, or abstain under
distribution shift.

## Historical package contents
- **`kga/`** — pure-numpy certificate core (`pip`-installable as `kbound`): evidence
  → certificate (Δ̂ ± ε) → adapt/freeze/abstain decision, with the false-adapt rate
  controlled at level α.
- **Theory** — impossibility result (the benefit sign is unidentifiable under matched
  label-free evidence), the exact benefit-sign frontier, and the finite-sample
  certificate (per-theorem validators included).
- **Papers** — the historical compatibility PDFs `kbound.pdf` and `kbound_short.pdf`. The maintained
  artifacts are now `kbound_short_final_draft.pdf`, `kbound_tmlr.pdf`, and
  `kbound_short_final_draft.docx`.

## Scientific correction ledger

- **Office-Home:** the original release described a CI-robust natural beats-both result. The current
  exact-rank primary replay instead ties always-freeze with zero ADAPT decisions. A separate
  test-stream replication has only a tiny point edge and its interval includes zero.
- **iWildCam:** the original release promoted damage prevention. Its archived scorer does not match
  the official WILDS label-present macro-F1 contract, so the numerical and action row is withheld
  until a pinned official-metric, population-sealed rerun exists.
- **CIFAR-10-C:** Tent and EATA have controlled-grid point-estimate gains. Tent has positive
  ordinary, unadjusted six-family intervals, but the retrospective Holm adjustment over the six
  prospectively named contrasts gives `p=0.09375` against both fixed policies. This inference is
  non-confirmatory; no confirmatory or cluster-robust win is promoted. SAR is negative.
- **ImageNet-C:** results are candidate dependent: SAR has a pooled point edge without a promoted
  robust interval; Tent ties freeze; EATA trails adapt. The early broad beats-both wording is
  retracted.
- **62-task anomaly breadth:** this historical simulated diagnostic is not part of the maintained
  paper's release-level empirical claims and carries no current significance claim.
- **CCT-20 (later prospective study):** the receipt-linked target result is
  `SAFE_UTILITY_ONLY`: 44 FREEZE, zero ADAPT, one ABSTAIN; it ties always-freeze and protects against
  harmful always-adapt, but does not establish strong bidirectional routing.
- **So2Sat-LCZ42 (later development study):** neither locked candidate was feasible. Execution
  stopped before gate calibration, with zero target pixel or label reads and no target score.
- **Natural-shift headline:** no current single-dataset result is a CI-robust beats-both routing win.

## Historical engineering changes
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

This command is a historical smoke check only. Publication requires the current clean-checkout
release workflow documented in `REPRODUCE.md` and `runbooks/release_candidate.sh`; a dated PASS in
this file does not certify later artifacts.
