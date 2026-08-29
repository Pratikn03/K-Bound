# K-Bound documentation index

**Last reconciled:** 2026-08-29 (Phase-1 provenance, CCT-20 target authority, stopped So2Sat development study, and maintained artifacts).
Previous reconciliation: 2026-07-01 (Wave 4 strict-100 + doc cleanup).

Use this file instead of dated status notes. Stale process MDs from June 2026 were removed;
history remains in git.

---

## Start here

| Doc | Role |
|-----|------|
| [`KBOUND_PHASE1_PROVENANCE_AUDIT_2026-08-27.md`](KBOUND_PHASE1_PROVENANCE_AUDIT_2026-08-27.md) | Current configuration, dataset, checkpoint, and code-hash coverage, with unrecoverable historical identities kept explicit. |
| [`audits/empirical_data_quality_2026_08_27/audit_summary.json`](audits/empirical_data_quality_2026_08_27/audit_summary.json) | Machine-readable 2026-08-27 forensic snapshot; its 14/14 checksum result refers to pre-Phase-1 bytes, and its unopened-target statement predates CCT-20/So2Sat. |
| [`KBOUND_EMPIRICAL_AND_RELEASE_CLOSURE_PLAN.md`](KBOUND_EMPIRICAL_AND_RELEASE_CLOSURE_PLAN.md) | **ACTIVE.** Executable plan for natural evidence, uniform protocols, official baselines, camera validation, and release CI. |
| [`KBOUND_SHORT_RESULT_AUDIT.md`](KBOUND_SHORT_RESULT_AUDIT.md) | Current source-hashed empirical verdicts and protocol scope. |
| [`KBOUND_SHORT_CLAIM_MANIFEST.md`](KBOUND_SHORT_CLAIM_MANIFEST.md) | Current claim-to-artifact authority. |
| [`paper/generated/cct20_release_manifest.json`](paper/generated/cct20_release_manifest.json) | Separate receipt-linked authority for the prospective CCT-20 target result (`SAFE_UTILITY_ONLY`, not strong routing success). |
| [`../../../experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/README.md`](../../../experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/README.md) | Separate authority for the So2Sat negative development-gate stop; no target access and no target score. |
| [`KBOUND_RELEASE_SHA256SUMS.txt`](KBOUND_RELEASE_SHA256SUMS.txt) | Release byte seal. Treat it as authoritative only after `runbooks/release_candidate.sh checksums` is run on the final frozen artifacts and every entry verifies. |
| [`SUBMISSION_LEDGER.md`](SUBMISSION_LEDGER.md) | Historical July/August freeze ledger; superseded for current verdicts and paths. |
| [`KBOUND_RELEASE_CLEANUP_REPORT_2026-08-27.md`](KBOUND_RELEASE_CLEANUP_REPORT_2026-08-27.md) | Historical pre-Phase-1 cleanup/checksum snapshot; operational cleanup ledger only. |
| [`COMPARISON_FAMILY.md`](COMPARISON_FAMILY.md) | Superseded 2026-07-26 search census and proposed Holm family; retained as history, not current multiplicity evidence. |
| [`README.md`](README.md) | Repo tour, current-state banner, evidence tiers, quick reproduce |
| [`../../../DATA.md`](../../../DATA.md) | **NEW 2026-07-26.** Per-dataset version, split, DOI/URL, licence, acquisition, and which table depends on it |
| [`PLACEHOLDER_INVENTORY.md`](PLACEHOLDER_INVENTORY.md) | **NEW 2026-07-26.** The 143 unreadable iCloud placeholders, what depends on them, recovery command, release-guard spec |
| [`PHASE6_LEAKAGE_AUDIT.md`](PHASE6_LEAKAGE_AUDIT.md) | **Corrected 2026-07-26** — its 2026-07-21 "PASS (clean)" verdict is retracted at the top of the file |
| [`PIPELINE_VS_PDF_AUDIT.md`](PIPELINE_VS_PDF_AUDIT.md) | Superseded July showcase/PDF map; retained as history, not the current generation chain |
| [`PROJECT_STATUS_AND_OPEN_PROBLEMS.md`](PROJECT_STATUS_AND_OPEN_PROBLEMS.md) | Theory ledger, empirical ledger, freeze gate (superseded by `SUBMISSION_LEDGER.md` where they disagree) |
| [`THEORY_100_PERCENT_CLOSURE_PLAN.md`](THEORY_100_PERCENT_CLOSURE_PLAN.md) | Wave 4 closure gate (`formal_audit.py --strict-100`) |
| [`THEORY_TO_CODE_MAP.md`](THEORY_TO_CODE_MAP.md) | Theorem → proof → validator → code → JSON |
| [`claim_ledger.json`](claim_ledger.json) | Every claim ID → artifact → allowed wording |

### Superseded — stamped in place, retained as history

`GAP_AUDIT.md`, `INTEGRITY_FIXES.md` (both repo root), `EVIDENCE_MATRIX.md`,
`PHASE7_INTEGRATION_AUDIT.md`, `KBOUND_RESULT_AUDIT.md`,
`KBOUND_EMPIRICAL_RECOVERY_AUDIT_2026-08-13.md`,
`KBOUND_TABLE4_NATURAL_SHIFT_RECONCILIATION_2026-08-27.md`, `COMPARISON_FAMILY.md`, and
`REVIEWER_REPRO_PACKET.md` (partially). Each carries a header stating what it still gets wrong or
is listed here as retained process history. Registry: `SUBMISSION_LEDGER.md §11`.
The July `audits/MAIN_PAPER_*` files and dated generated audit snapshots are routed through
[`audits/README.md`](audits/README.md); they do not override the current claim ledger.

---

## Papers (source of truth for claims)

| Artifact | Pages | Use |
|----------|------:|-----|
| [`kbound_submission.tex`](kbound_submission.tex) / [`kbound_short_final_draft.pdf`](kbound_short_final_draft.pdf) / [`kbound_short_final_draft.docx`](kbound_short_final_draft.docx) | 30 | Primary compact Phase-1 submission and synchronized Word export |
| [`kbound_tmlr.tex`](kbound_tmlr.tex) / [`kbound_tmlr.pdf`](kbound_tmlr.pdf) | 34 | Maintained anonymous official-style TMLR driver synchronized through `kbound_submission_body.tex` |

---

## Reproduce & train

| Command / doc | Purpose |
|---------------|---------|
| `bash docs/research/kbound/runbooks/release_candidate.sh all` | Current clean-checkout publication gate: authorities, generation, tests, Lean, both PDFs, required compact DOCX, PDF rendering, checksums |
| `bash docs/research/kbound/scripts/reproduce_submission.sh` | Historical compatibility verifier; not sufficient for a current release PASS |
| `BUILD_LONG_TMLR=1 BUILD_DOCX=1 bash docs/research/kbound/scripts/build_pdfs.sh` | Direct manuscript build of the compact PDF, synchronized long companion, and required compact DOCX |
| `bash docs/research/kbound/scripts/kbtrain.sh theory-v2` | Wave 4 validators + routing selftest |
| `bash docs/research/kbound/scripts/kbtrain.sh smoke-all` | ~0.5% smoke, single seed, all 9 datasets |
| `bash docs/research/kbound/scripts/kbtrain.sh smoke-all-v2` | ~1% **multiseed** smoke (Protocol-A CIFAR, theory preflight) |
| `bash docs/research/kbound/scripts/run_smoke_showcase.sh` | **Multiseed smoke + locked-analysis mini-run + pipeline report** |
| `bash docs/research/kbound/scripts/kbtrain.sh final-all-v2` | Theory preflight + full 9-dataset GPU refresh |
| `bash docs/research/kbound/scripts/run_85plus_readiness.sh` | **85+ scorecard** — theory, smoke, edge, RxRx1 blockers |
| `bash docs/research/kbound/edge/scripts/run_edge_source_gate.sh` | Physical R2 phase 1: S01–S02 + 0.80 gate |
| `bash docs/research/kbound/scripts/prepare_rxrx1_data.sh` | RxRx1 download instructions / check |
| `bash docs/research/kbound/scripts/run_final_showcase.sh` | Legacy train-and-showcase workflow; not the maintained publication generator |
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

The July reproducibility report and `RELEASE_MANIFEST.json` are historical PASS snapshots, not a
PASS for the current checkout. They remain stamped in place as history; the current certification
surface is the successful final clean-checkout gate output together with a freshly generated and
verified `KBOUND_RELEASE_SHA256SUMS.txt`.

---

## Deprecated (do not use)

| Path | Note |
|------|------|
| [`manuscript/`](manuscript/README.md) | Parallel book; stale `conj:gen` wording |
| `kbound_short.tex`, `kbound_short_body.tex`, `kbound_short_appendix.tex` | Superseded empirical archive; excluded from maintained drivers |
| `kbound.pdf`, `kbound_edited.pdf`, `kbound_long_companion.pdf`, `kbound_short.pdf`, `kbound_short_edited.pdf`, `kbound_short_companion.pdf`, `kbound_short.docx` | Historical compatibility snapshots; not refreshed or delivered |

---

## Optional ops plans (not theory closure)

| Doc | Role |
|-----|------|
| [`REPO_CLEANUP_PLAN.md`](REPO_CLEANUP_PLAN.md) | Disk / dataset tiering (~800 GB) |
| [`REPO_LEVEL80_PLAN.md`](REPO_LEVEL80_PLAN.md) | Architecture & CI hardening plan |
| [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) | Zenodo / PyPI publish steps |

---

## What is still open (not doc gaps)

1. No natural result currently establishes CI-robust beats-both routing. CCT-20 completed as
   `SAFE_UTILITY_ONLY`; So2Sat stopped at development with no feasible candidate and no target access.
2. Independent Office-Home and official-metric, population-sealed iWildCam reruns remain optional
   extensions if stronger natural evidence is sought.
3. PACS per-cell replay and official POEM/AETTA runs remain optional evidence upgrades.
4. Physical camera R2 captures remain pending; they are not required for the theory-led claim set.
5. Final venue/anonymity selection, clean-checkout release gate, and source-hashed release freeze.

Everything in the **theory closure plan** (Section A + B) is **done**.
