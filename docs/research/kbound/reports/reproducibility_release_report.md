# Reproducibility Release Report

> **SUPERSEDED SNAPSHOT — DO NOT USE AS A CURRENT RELEASE PASS (2026-08-29).** This report records
> commit `46eae5f...` on 2026-07-27. It predates the Phase-1 reconciliation, CCT-20 target result,
> So2Sat development stop, current manuscript artifacts, and final code cleanup. The step log below
> is retained unchanged as execution history. Publication requires a new clean-checkout report and
> regenerated release checksums after all maintained files are frozen.

Generated: 2026-07-27T07:18:46Z
Commit: 46eae5f570f66e513ec8d9bf0bdf40b3428cef88
Historical overall at the recorded commit: **PASS** (0 required step(s) failed); current status:
**SUPERSEDED / NOT EVALUATED BY THIS REPORT**.

Every step below was attempted. PASS = ran and exited 0. FAIL = a required step
did not (this sets Overall to FAIL). SKIP = an optional step could not run,
usually because an artifact is not in the release; the reason is in Note.

| Step | Status | Note |
|---|---|---|
| 1 core unit tests (leakage + claim semantics + conformal + policy + routing) | PASS |  |
| 1b env-dependent tests (edge capture artifacts + torch) | SKIP | needs torch and experiments/kbound/results/edge_real_phone_v1/{calibration_summary,split_audit}.json, which are not in the release |
| 2 theory validator artifacts present | PASS |  |
| 2b full theory audit (artifacts + claim ledger cross-check) | PASS |  |
| 2c wave-4 theory_v2 validators + routing selftest | PASS |  |
| 3 gate baseline CPU selftest | PASS |  |
| 4a refresh results_source (locked) | PASS |  |
| 4b regenerate paper table macros | PASS |  |
| 5 unified result verdict audit | PASS |  |
| 6 validate claim ledger | PASS |  |
| 7 mixed head-to-head (POEM/AETTA) result present | PASS |  |
| 8 cached headline artifacts | PASS |  |

Not covered by this script: the full GPU protocol re-runs (stress grid,
ImageNet-C, WILDS) — see RELEASE_10X_TRACK.md — and the physical-camera R2
capture, which is still pending.
