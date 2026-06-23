# Camelyon17 diagnostics — resolved (2026-06-16)

## Status after fixes

| Track | Fix applied | Outcome | Role |
|-------|-------------|---------|------|
| **Protocol F GPU data** | Complete serialization (rich 17-dim Z) | 540 records | **Data layer** for G/H |
| **Protocol G headline** | Canonical KGA, eata_online | regret=0.0000, FA=2.56%, beats_both=yes | **Headline win** |
| **Multicandidate frozen τ*=0.52** | Re-scored from stored conditions | beats_both=False | **Diagnostic failure** (scale mismatch) |
| **Multicandidate source-cal τ*** | id_val → test calibration | tau*=1.84, 0% FA on adapts, beats_both=False | **Route fixed** (commits); still loses to best-fixed-adapt |
| **Route-a domain-split (deployed)** | Source-fit ε on id_val | beats_both=False | Appendix audit |
| **ε-recal sparse Z (debug)** | In-domain by seed | PRECISE_NEGATIVE | **Calibration diagnostic** |
| **ε-recal rich Z (eata_online)** | Dev {0,1} / test {2,3,4} | **WIN** (FA 2.56%, beats both) | Confirms G operating point |
| **Protocol B sparse n=1024** | Wrong runner (aggregates only) | Integrity FAIL | **Needs GPU B-v2** (see launch script) |

## Interpretation

- **F is not a failure** — it is the GPU record source. Route audits on those records show which estimators/routes work.
- **Multicandidate is fixed** in the sense of source-calibrated τ* (no test peeking); it still does not clear beats-both on Camelyon test.
- **Sparse-Z / sample-size path is closed** — bias-limited ε; rich Z + in-domain calibration (G) is the fix.
- **Protocol B v2** requires `run_camelyon17_kbound.py` full grid re-run (GPU).

Artifacts: `experiments/kbound/results/camelyon17_diagnostics_resolved_v1/`