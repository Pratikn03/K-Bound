# Reproducibility Release Report

Generated: 2026-07-26T03:19:43Z
Commit: unknown
Overall: **PASS** (0 required step(s) failed)

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
