# K-Bound documentation index

## Current routing qualification — 2026-09-20

Use [ACTIVE_NATURAL_STUDY_PLAN.md](ACTIVE_NATURAL_STUDY_PLAN.md) as the living
execution ledger. The older status descriptions below are dated history, not a
request to rerun completed SAR or Task3 collection/scoring. Release verification
remains separate from completed experiments, including the separate painting attempt.

Thirty additional retired active duplicates were moved recoverably after exact
archive/Git byte comparison. Archived reproduction notebooks and packets are
history; current reproduction starts with `runbooks/release_candidate.sh all`.
The stale-publication-builds inventory
(`archive/stale_publication_builds_2026-09-02/`) now preserves ten exact blobs
recovered from pinned historical Git on 2026-09-20. Its old three-artifact names
are not the current release contract. No current PDF was changed or promoted;
source/runtime, artifact-contract and package verification remain separate gates.

Seven additional retired local files are preserved byte-for-byte in
[`preserved_worktree_2026-09-20`](archive/preserved_worktree_2026-09-20/README.md).
Its manifest retains the six variants that differ from older archived copies;
neither set is current scientific authority. Do not execute archived launchers.

### Historical qualification recorded 2026-09-12

The [superseded empirical authority archive](archive/superseded_empirical_authorities_2026-09-02/MANIFEST.json)
contains authenticated historical bytes, **not current authority**. Its 21 original-path
duplicates were moved into a separate recoverable backup after comparison with both
the archive manifest and historical Git bytes; the archive itself remains unchanged.
Older completion statements below are historical scope statements, not closure of Tasks 1–4.
Use `experiments/kbound/results/task_closure_20260911/STATUS.json` for the living
verification ledger; prospective natural/population evidence, official synchronized
baselines, corrected SAR execution, and repository-wide release verification remain open.

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
| [`ACTIVE_NATURAL_STUDY_PLAN.md`](ACTIVE_NATURAL_STUDY_PLAN.md) | Living execution and verification ledger; earlier closure plans are historical. |
| [`KBOUND_SHORT_RESULT_AUDIT.md`](KBOUND_SHORT_RESULT_AUDIT.md) | Current source-hashed empirical verdicts and protocol scope. |
| [`KBOUND_SHORT_CLAIM_MANIFEST.md`](KBOUND_SHORT_CLAIM_MANIFEST.md) | Current claim-to-artifact authority. |
| [`paper/generated/cct20_release_manifest.json`](paper/generated/cct20_release_manifest.json) | Separate receipt-linked authority for the prospective CCT-20 target result (`SAFE_UTILITY_ONLY`, not strong routing success). |
| [`../../../experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/README.md`](../../../experiments/kbound/results/so2sat_lcz42_prospective_v1/development_mps_bn_fix_v1/README.md) | Separate authority for the So2Sat negative development-gate stop; no target access and no target score. |
| [`KBOUND_RELEASE_SHA256SUMS.txt`](KBOUND_RELEASE_SHA256SUMS.txt) | Release byte seal. Treat it as authoritative only after `runbooks/release_candidate.sh checksums` is run on the final frozen artifacts and every entry verifies. |
| [`SUBMISSION_LEDGER.md`](archive/legacy_publication_surfaces_2026-09-02/retired_tree/docs/research/kbound/SUBMISSION_LEDGER.md) | Historical July/August freeze ledger; superseded for current verdicts and paths. |
| [`KBOUND_RELEASE_CLEANUP_REPORT_2026-08-27.md`](archive/legacy_publication_surfaces_2026-09-02/retired_tree/docs/research/kbound/KBOUND_RELEASE_CLEANUP_REPORT_2026-08-27.md) | Historical pre-Phase-1 cleanup/checksum snapshot; operational cleanup ledger only. |
| [`COMPARISON_FAMILY.md`](archive/legacy_publication_surfaces_2026-09-02/retired_tree/docs/research/kbound/COMPARISON_FAMILY.md) | Superseded 2026-07-26 search census and proposed Holm family; retained as history, not current multiplicity evidence. |
| [`README.md`](README.md) | Repo tour, current-state banner, evidence tiers, quick reproduce |
| [`../../../DATA.md`](../../../DATA.md) | **NEW 2026-07-26.** Per-dataset version, split, DOI/URL, licence, acquisition, and which table depends on it |
| [`PLACEHOLDER_INVENTORY.md`](PLACEHOLDER_INVENTORY.md) | **NEW 2026-07-26.** The 143 unreadable iCloud placeholders, what depends on them, recovery command, release-guard spec |
| [`PHASE6_LEAKAGE_AUDIT.md`](PHASE6_LEAKAGE_AUDIT.md) | **Corrected 2026-07-26** — its 2026-07-21 "PASS (clean)" verdict is retracted at the top of the file |
| [`PIPELINE_VS_PDF_AUDIT.md`](archive/legacy_publication_surfaces_2026-09-02/retired_tree/docs/research/kbound/PIPELINE_VS_PDF_AUDIT.md) | Superseded July showcase/PDF map; retained as history, not the current generation chain |
| [`PROJECT_STATUS_AND_OPEN_PROBLEMS.md`](archive/preserved_worktree_2026-09-20/tree/docs/research/kbound/PROJECT_STATUS_AND_OPEN_PROBLEMS.md) | Preserved historical local status; superseded by the living ledger and current claim/evidence authorities |
| [`THEORY_100_PERCENT_CLOSURE_PLAN.md`](archive/preserved_worktree_2026-09-20/tree/docs/research/kbound/THEORY_100_PERCENT_CLOSURE_PLAN.md) | Preserved superseded theory plan, not current formal acceptance |
| [`THEORY_TO_CODE_MAP.md`](archive/legacy_publication_surfaces_2026-09-02/retired_tree/docs/research/kbound/THEORY_TO_CODE_MAP.md) | Theorem → proof → validator → code → JSON |
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

| Artifact | Page count | Use |
|----------|------:|-----|
| [`kbound_submission.tex`](kbound_submission.tex) / [`kbound_short_final_draft.pdf`](kbound_short_final_draft.pdf) / [`kbound_short_final_draft.docx`](kbound_short_final_draft.docx) | Verify exact delivered file; not the old fixed count | Maintained compact driver and Word-export route; file presence is not release sealing |
| [`kbound_tmlr.tex`](kbound_tmlr.tex) / [`kbound_tmlr.pdf`](kbound_tmlr.pdf) | Verify exact delivered file; not the old fixed count | Maintained long driver using the shared submission source |

---

## Reproduce & train

| Command / doc | Purpose |
|---------------|---------|
| `bash docs/research/kbound/runbooks/release_candidate.sh all` | Current clean-checkout publication gate: authorities, generation, tests, Lean, both PDFs, required compact DOCX, PDF rendering, checksums |
| [Archived compatibility verifier](archive/preserved_worktree_2026-09-20/README.md) | Read-only history; do not execute as a release gate |
| `BUILD_LONG_TMLR=1 BUILD_DOCX=1 bash docs/research/kbound/scripts/build_pdfs.sh` | Direct manuscript build of the compact PDF, synchronized long companion, and required compact DOCX |
| `bash docs/research/kbound/scripts/kbtrain.sh theory-v2` | Wave 4 validators + routing selftest |
| `bash docs/research/kbound/scripts/kbtrain.sh smoke-all` | ~0.5% smoke, single seed, all 9 datasets |
| `bash docs/research/kbound/scripts/kbtrain.sh smoke-all-v2` | ~1% **multiseed** smoke (Protocol-A CIFAR, theory preflight) |
| `bash docs/research/kbound/scripts/run_smoke_showcase.sh` | **Multiseed smoke + locked-analysis mini-run + pipeline report** |
| `bash docs/research/kbound/scripts/kbtrain.sh final-all-v2` | Theory preflight + full 9-dataset GPU refresh |
| `bash docs/research/kbound/scripts/run_85plus_readiness.sh` | **85+ scorecard** — theory, smoke, edge, RxRx1 blockers |
| `bash docs/research/kbound/edge/scripts/run_edge_source_gate.sh` | Physical R2 phase 1: S01–S02 + 0.80 gate |
| `bash docs/research/kbound/scripts/prepare_rxrx1_data.sh` | RxRx1 download instructions / check |
| [`RUN_FINAL_SHOWCASE.md`](RUN_FINAL_SHOWCASE.md) | Retired workflow notice; does not start training or establish release completion |
| [`REVIEWER_REPRO_PACKET.md`](archive/superseded_empirical_authorities_2026-09-02/retired_tree/docs/research/kbound/REVIEWER_REPRO_PACKET.md) | External reproducer checklist |
| [`runbooks/release_candidate.sh`](runbooks/release_candidate.sh) | Gated release verification; never blanket training authorization |

---

## Theory & formal

| Path | Role |
|------|------|
| [`formal/README.md`](formal/README.md) | Lean 4 package + `formal_audit.py` |
| [`theory_v2/`](theory_v2/) | Wave 4 `.tex` fragments + `val_*.py` |
| [`theory_v2/UNCONDITIONAL_WEAKEST_CLASS_ATTEMPT.md`](theory_v2/UNCONDITIONAL_WEAKEST_CLASS_ATTEMPT.md) | `thm:uncond-weakest` derivation notes |
| [`THEORY_AUDIT_senior_review.md`](archive/legacy_publication_surfaces_2026-09-02/retired_tree/docs/research/kbound/THEORY_AUDIT_senior_review.md) | External-style theory review |
| `bash docs/research/kbound/scripts/theory_audit_full.sh` | Full theory audit report |

---

## Protocols & empirics

| Doc | Role |
|-----|------|
| [`MIXED_BENCHMARK_PROTOCOL.md`](MIXED_BENCHMARK_PROTOCOL.md) | Mixed harmful+helpful benchmark |
| [`gate_comparison.md`](gate_comparison.md) | Decision-gate vs certificate |
| [`realshift_win/PROTOCOL_realshift_win.md`](realshift_win/PROTOCOL_realshift_win.md) | Real-shift win protocol |
| [`RELEASE_10X_TRACK.md`](archive/legacy_publication_surfaces_2026-09-02/retired_tree/docs/research/kbound/RELEASE_10X_TRACK.md) | Release artifact manifest |

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
| [Preserved parallel-exposition guide](archive/preserved_worktree_2026-09-20/tree/docs/research/kbound/manuscript/README.md) | Not the maintained driver; unique `manuscript/theory_spine/` fragments remain live |
| `kbound_short.tex`, `kbound_short_body.tex`, `kbound_short_appendix.tex` | Superseded empirical archive; excluded from maintained drivers |
| `kbound.pdf`, `kbound_edited.pdf`, `kbound_long_companion.pdf`, `kbound_short.pdf`, `kbound_short_edited.pdf`, `kbound_short_companion.pdf`, `kbound_short.docx` | Historical compatibility snapshots; not refreshed or delivered |

---

## Optional ops plans (not theory closure)

| Doc | Role |
|-----|------|
| [`REPO_CLEANUP_PLAN.md`](REPO_CLEANUP_PLAN.md) | Disk / dataset tiering (~800 GB) |
| [`REPO_LEVEL80_PLAN.md`](archive/legacy_publication_surfaces_2026-09-02/retired_tree/docs/research/kbound/REPO_LEVEL80_PLAN.md) | Architecture & CI hardening plan |
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

The earlier blanket theory-closure statement is superseded. Formal declarations,
external assumptions, experiment evidence and current release verification have
separate completion conditions in the living ledger.
