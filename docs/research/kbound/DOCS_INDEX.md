# K-Bound documentation index

**Last reconciled:** 2026-07-26 (post-external-review revision).
Previous reconciliation: 2026-07-01 (Wave 4 strict-100 + doc cleanup).

Use this file instead of dated status notes. Stale process MDs from June 2026 were removed;
history remains in git.

---

## Start here

| Doc | Role |
|-----|------|
| [`SUBMISSION_LEDGER.md`](SUBMISSION_LEDGER.md) | **CANONICAL. Overrides every other document.** Venue, freeze status, nine-track evidence tiers, absent artifacts, open items. |
| [`README.md`](README.md) | Repo tour, current-state banner, evidence tiers, quick reproduce |
| [`../../../DATA.md`](../../../DATA.md) | **NEW 2026-07-26.** Per-dataset version, split, DOI/URL, licence, acquisition, and which table depends on it |
| [`PLACEHOLDER_INVENTORY.md`](PLACEHOLDER_INVENTORY.md) | **NEW 2026-07-26.** The 143 unreadable iCloud placeholders, what depends on them, recovery command, release-guard spec |
| [`COMPARISON_FAMILY.md`](COMPARISON_FAMILY.md) | **NEW 2026-07-26.** Prospective comparison family, full pre-registered arm inventory, multiplicity correction |
| [`PHASE6_LEAKAGE_AUDIT.md`](PHASE6_LEAKAGE_AUDIT.md) | **Corrected 2026-07-26** — its 2026-07-21 "PASS (clean)" verdict is retracted at the top of the file |
| [`PIPELINE_VS_PDF_AUDIT.md`](PIPELINE_VS_PDF_AUDIT.md) | What `run_final_showcase.sh` updates vs PDF tables |
| [`PROJECT_STATUS_AND_OPEN_PROBLEMS.md`](PROJECT_STATUS_AND_OPEN_PROBLEMS.md) | Theory ledger, empirical ledger, freeze gate (superseded by `SUBMISSION_LEDGER.md` where they disagree) |
| [`THEORY_100_PERCENT_CLOSURE_PLAN.md`](THEORY_100_PERCENT_CLOSURE_PLAN.md) | Wave 4 closure gate (`formal_audit.py --strict-100`) |
| [`THEORY_TO_CODE_MAP.md`](THEORY_TO_CODE_MAP.md) | Theorem → proof → validator → code → JSON |
| [`claim_ledger.json`](claim_ledger.json) | Every claim ID → artifact → allowed wording |

### Superseded — stamped in place, retained as history

`GAP_AUDIT.md`, `INTEGRITY_FIXES.md` (both repo root), `EVIDENCE_MATRIX.md`,
`PHASE7_INTEGRATION_AUDIT.md`, and `REVIEWER_REPRO_PACKET.md` (partially). Each carries a header
stating what it still gets wrong. Registry: `SUBMISSION_LEDGER.md §11`.

---

## Papers (source of truth for claims)

| Artifact | Pages | Use |
|----------|------:|-----|
| [`kbound_short.tex`](kbound_short.tex) / [`kbound_short.pdf`](kbound_short.pdf) | ~21 | Venue main submission |
| [`kbound.tex`](kbound.tex) / [`kbound.pdf`](kbound.pdf) | ~57 | Full version + Wave 4 appendix |

---

## Reproduce & train

| Command / doc | Purpose |
|---------------|---------|
| `bash docs/research/kbound/scripts/reproduce_submission.sh` | CPU integrity (~4 min): tests, validators, tables, ledger |
| `bash docs/research/kbound/scripts/kbtrain.sh theory-v2` | Wave 4 validators + routing selftest |
| `bash docs/research/kbound/scripts/kbtrain.sh smoke-all` | ~0.5% smoke, single seed, all 9 datasets |
| `bash docs/research/kbound/scripts/kbtrain.sh smoke-all-v2` | ~1% **multiseed** smoke (Protocol-A CIFAR, theory preflight) |
| `bash docs/research/kbound/scripts/run_smoke_showcase.sh` | **Multiseed smoke + locked-analysis mini-run + pipeline report** |
| `bash docs/research/kbound/scripts/kbtrain.sh final-all-v2` | Theory preflight + full 9-dataset GPU refresh |
| `bash docs/research/kbound/scripts/run_85plus_readiness.sh` | **85+ scorecard** — theory, smoke, edge, RxRx1 blockers |
| `bash docs/research/kbound/edge/scripts/run_edge_source_gate.sh` | Physical R2 phase 1: S01–S02 + 0.80 gate |
| `bash docs/research/kbound/scripts/prepare_rxrx1_data.sh` | RxRx1 download instructions / check |
| `bash docs/research/kbound/scripts/run_final_showcase.sh` | **Full end-to-end:** v2 train → tables → figures → PDFs |
| [`REVIEWER_REPRO_PACKET.md`](REVIEWER_REPRO_PACKET.md) | External reproducer checklist |
| [`RUN_FINAL_SHOWCASE.md`](RUN_FINAL_SHOWCASE.md) | Pre-registered showcase wrapper (calls `final-all-v2`) |

---

## Theory & formal

| Path | Role |
|------|------|
| [`formal/README.md`](formal/README.md) | Lean 4 package + `formal_audit.py` |
| [`theory_v2/`](theory_v2/) | Wave 4 `.tex` fragments + `val_*.py` |
| [`theory_v2/UNCONDITIONAL_WEAKEST_CLASS_ATTEMPT.md`](theory_v2/UNCONDITIONAL_WEAKEST_CLASS_ATTEMPT.md) | `thm:uncond-weakest` derivation notes |
| [`THEORY_AUDIT_senior_review.md`](THEORY_AUDIT_senior_review.md) | External-style theory review |
| `bash docs/research/kbound/scripts/theory_audit_full.sh` | Full theory audit report |

---

## Protocols & empirics

| Doc | Role |
|-----|------|
| [`MIXED_BENCHMARK_PROTOCOL.md`](MIXED_BENCHMARK_PROTOCOL.md) | Mixed harmful+helpful benchmark |
| [`gate_comparison.md`](gate_comparison.md) | Decision-gate vs certificate |
| [`realshift_win/PROTOCOL_realshift_win.md`](realshift_win/PROTOCOL_realshift_win.md) | Real-shift win protocol |
| [`RELEASE_10X_TRACK.md`](RELEASE_10X_TRACK.md) | Release artifact manifest |

---

## Edge / camera (open empirical)

| Doc | Role |
|-----|------|
| [`edge/README.md`](edge/README.md) | Physical deployment package |
| [`edge/PHYSICAL_STUDY_RUNBOOK.md`](edge/PHYSICAL_STUDY_RUNBOOK.md) | Capture S01–S10 |
| [`edge/R2_SESSION_TRACKER.md`](edge/R2_SESSION_TRACKER.md) | Session log (KB-CLAIM-030 pending) |

---

## Generated reports (`reports/`)

Auto-written or point-in-time audits — **not** canonical status:

- `reports/THEORY_AUDIT_FULL.md` — from `theory_audit_full.py`
- `reports/reproducibility_release_report.md` — from `reproduce_submission.sh`
- `reports/KBOUND_10X_FINAL_GATE.md` — June 2026 gate snapshot

---

## Deprecated (do not use)

| Path | Note |
|------|------|
| [`manuscript/`](manuscript/README.md) | Parallel book; stale `conj:gen` wording |
| `kbound_submission.tex` | Frozen snapshot — use live `kbound*.tex` |

---

## Optional ops plans (not theory closure)

| Doc | Role |
|-----|------|
| [`REPO_CLEANUP_PLAN.md`](REPO_CLEANUP_PLAN.md) | Disk / dataset tiering (~800 GB) |
| [`REPO_LEVEL80_PLAN.md`](REPO_LEVEL80_PLAN.md) | Architecture & CI hardening plan |
| [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) | Zenodo / PyPI publish steps |

---

## What is still open (not doc gaps)

1. Physical camera R2 captures (KB-CLAIM-030)
2. External reviewer sign-off (`REVIEWER_REPRO_PACKET.md`)
3. Optional fresh `final-all` GPU rerun for new JSON manifests

Everything in the **theory closure plan** (Section A + B) is **done**.
