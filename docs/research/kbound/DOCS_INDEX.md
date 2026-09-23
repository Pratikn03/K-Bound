# K-Bound Documentation Index

This index organizes the public research documentation, manuscript drivers, protocols, and reproducibility guides for the **K-Bound** project.

---

## 📄 Manuscripts & Submission Drivers

| Document | Format | Description |
|:---|:---:|:---|
| [`kbound_submission.tex`](kbound_submission.tex) | LaTeX | Main maintained manuscript driver (body in `kbound_submission_body.tex`, appendix in `kbound_submission_supplement.tex`). |
| [`kbound_tmlr.tex`](kbound_tmlr.tex) | LaTeX | Synchronized long-format / TMLR driver. |
| [`kbound_short_main.tex`](kbound_short_main.tex) | LaTeX | Compact manuscript driver. |
| [`kbound_tmlr.pdf`](kbound_tmlr.pdf) | PDF | Precompiled full-length manuscript PDF. |
| [`kbound_short_final_draft.pdf`](kbound_short_final_draft.pdf) | PDF | Precompiled compact draft PDF. |

---

## 🚀 Reproducibility & Environment Setup

| Guide | Description |
|:---|:---|
| [`README.md`](README.md) | Technical overview of the K-Bound formal framework, empirical findings, and quickstart usage. |
| [`REPRODUCE.md`](REPRODUCE.md) | Step-by-step instructions for reproducing experimental results and benchmarks. |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Formal reproducibility statement and verification standards. |
| [`RUN_ON_MAC.md`](RUN_ON_MAC.md) | Environment setup, dependencies, and execution notes for Apple Silicon (MPS) and Linux. |
| [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) | Final release verification steps and artifact sanity checks. |
| [`KBOUND_RELEASE_SHA256SUMS.txt`](KBOUND_RELEASE_SHA256SUMS.txt) | Release SHA-256 cryptographic verification checksums. |

---

## 🔬 Benchmark Protocols & Empirical Methodology

| Protocol / Spec | Description |
|:---|:---|
| [`MIXED_BENCHMARK_PROTOCOL.md`](MIXED_BENCHMARK_PROTOCOL.md) | Specification for the mixed helpful/harmful domain-shift benchmark. |
| [`MIXED_BENCHMARK_EXT_PROTOCOL.md`](MIXED_BENCHMARK_EXT_PROTOCOL.md) | Extended benchmark protocols and stress-test evaluation parameters. |
| [`INDEPENDENT_REPLICATION_PROTOCOL.md`](INDEPENDENT_REPLICATION_PROTOCOL.md) | Independent replication protocol for external researchers. |
| [`PREREG_HARM_PREVENTION_v1.md`](PREREG_HARM_PREVENTION_v1.md) | Pre-registered harm prevention contract and error control criteria. |
| [`RELATED_WORK_POSITIONING.md`](RELATED_WORK_POSITIONING.md) | Literature review and academic positioning relative to prior TTA/conformal works. |
| [`EXTERNAL_STORAGE_POLICY.md`](EXTERNAL_STORAGE_POLICY.md) | Policy and paths for external datasets, model weights, and heavy arrays. |
| [`KBOUND_SHORT_CLAIM_MANIFEST.md`](KBOUND_SHORT_CLAIM_MANIFEST.md) | Claim-to-artifact mapping and numerical evidence ledger. |

---

## 📐 Formal Verification & Theory

| Directory / File | Description |
|:---|:---|
| [`formal/README.md`](formal/README.md) | Lean 4 formalization package and declaration index (142 scoped declarations). |
| [`theory_v2/`](theory_v2/) | Theoretical derivation notes, validators, and formal proof fragments. |
| [`claim_ledger.json`](claim_ledger.json) | Machine-readable ledger mapping each claim ID to authoritative evidence files. |

---

## 🗄️ Historical Development & Archives

Internal sprint closeouts, working runsheets, intermediate forensic audits, and review response drafts are cataloged under [`archive/`](archive/):

* **[`archive/internal_dev_notes/`](archive/internal_dev_notes/)**: Sprint closeout reports (`TASK1_4_FINAL_CLOSEOUT.md`, `ROOT_FINAL_TASK1_4_CLOSEOUT.md`), execution runsheets, working logs, and editorial changelogs.
* **[`archive/audits/`](archive/audits/)**: Historical forensic and provenance audits (`KBOUND_PHASE1_PROVENANCE_AUDIT_2026-08-27.md`, `PHASE6_LEAKAGE_AUDIT.md`, etc.).
* **[`archive/reviews_and_rebuttals/`](archive/reviews_and_rebuttals/)**: Historical reviewer response notes and internal mock review packets.
* **[`archive/backups/`](archive/backups/)**: Pre-editorial snapshots and legacy manuscript versions.
