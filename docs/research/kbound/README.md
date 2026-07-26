# K-Bound Research Guide

This directory is the maintained research surface for K-Bound and KGA. It contains the manuscript,
historical extended manuscript, formalization, canonical result manifest, dashboard, and
physical-camera validation package.

> ## State of the project as of 2026-07-26 — read this before anything else
>
> An external five-specialist review panel audited the manuscript and the release on 2026-07-25.
> The project's own audit documents were found to certify things the source contradicted. The
> 2026-07-26 revision fixes that. What a reader arriving today needs to know:
>
> - **Target venue is TMLR**, not an IEEE conference. Single-column; the length problem was a
>   two-column artifact and no results were cut for it.
> - **The manuscript is NOT FROZEN.** The pinned commit and PDF sha256 in the old ledger were
>   stale by construction — `EDIT_NOTES_2026-07-23.md` records 12 edits made the day after the
>   freeze, two of which change the compiled output. A dated re-freeze procedure replaces the stale
>   hashes: `SUBMISSION_LEDGER.md §0`.
> - **The conformal radius was calibrated in sample** on five shipped scripts and seven `decide_kga`
>   forks. Fixed. The CIFAR-10-C flagship result is **completely unaffected** (0 of 9 504 decisions
>   change); ImageNet-C SAR moves from a CI-supported beats-both to a **point-estimate no-harm**;
>   Camelyon17 Table VIII gets slightly worse. `SUBMISSION_LEDGER.md §9`.
> - **`PHASE6_LEAKAGE_AUDIT.md`'s 2026-07-21 "PASS (clean)" verdict is retracted** at the top of
>   that file, with the original text preserved so the correction can be diffed.
> - **Three tracks are demoted**: ImageNet-C SAR (beats-both -> point-estimate no-harm),
>   Camelyon17 OOD (locked -> **sealed but not recomputable from release**), and Office-Home /
>   iWildCam (locked rows whose source record files are absent). `SUBMISSION_LEDGER.md §3`, `§8`.
> - **143 committed text artifacts are NUL-filled iCloud placeholders**, including the whole
>   Office-Home runner and every ablation JSON. Census, one-command recovery, and the release-guard
>   spec: [PLACEHOLDER_INVENTORY.md](PLACEHOLDER_INVENTORY.md).
> - **Multi-seed runs were not produced under one environment** — three Python/torch stacks across
>   five CIFAR seeds, and no manifest records a scikit-learn version. Their spread is not seed
>   variance. `REPRODUCE.md §0a`.
> - **The multiplicity family was declared post hoc** (the three comparisons that won, out of 1 427
>   recorded `beats_both` determinations). Prospective declaration and full arm inventory:
>   [COMPARISON_FAMILY.md](COMPARISON_FAMILY.md).
>
> The single-sentence version: the CIFAR-10-C stress-grid safety result is real, well-powered and
> survives every check; most of the rest of the panel supports a narrower claim than the one
> originally written, and the documents now say so.

## Start Here

| Goal | Entry point |
|---|---|
| **Understand the current state and every open item** | **[SUBMISSION_LEDGER.md](SUBMISSION_LEDGER.md)** — canonical; overrides every other document |
| Read the manuscript | [kbound_short.tex](kbound_short.tex) and [kbound_short_appendix.tex](kbound_short_appendix.tex) |
| Inspect every promoted number | [paper/generated/kbound_result_manifest.json](paper/generated/kbound_result_manifest.json) |
| Audit claim-to-artifact links | [KBOUND_SHORT_CLAIM_MANIFEST.md](KBOUND_SHORT_CLAIM_MANIFEST.md) |
| Obtain the datasets | [../../../DATA.md](../../../DATA.md) — per-dataset version, split, licence, acquisition |
| Reproduce the submission | [REPRODUCE.md](REPRODUCE.md), then [REVIEWER_REPRO_PACKET.md](REVIEWER_REPRO_PACKET.md) (partially superseded) |
| Run an independent replication | [INDEPENDENT_REPLICATION_PROTOCOL.md](INDEPENDENT_REPLICATION_PROTOCOL.md) |
| See what is unreadable and why | [PLACEHOLDER_INVENTORY.md](PLACEHOLDER_INVENTORY.md) |
| See the comparison family and arm inventory | [COMPARISON_FAMILY.md](COMPARISON_FAMILY.md) |
| Read the corrected leakage audit | [PHASE6_LEAKAGE_AUDIT.md](PHASE6_LEAKAGE_AUDIT.md) |
| Understand tracked vs external artifacts | [EXTERNAL_STORAGE_POLICY.md](EXTERNAL_STORAGE_POLICY.md) / [STORAGE_MANIFEST.json](STORAGE_MANIFEST.json) |
| Inspect the withheld CIFAR SAR arm | [CIFAR10C_SAR_QUARANTINE.md](CIFAR10C_SAR_QUARANTINE.md) |
| Inspect theory-to-code mapping | [THEORY_TO_CODE_MAP.md](THEORY_TO_CODE_MAP.md) |
| Build the research dashboard | [dashboard/README.md](dashboard/README.md) |
| Start the physical study | [edge/PHYSICAL_STUDY_RUNBOOK.md](edge/PHYSICAL_STUDY_RUNBOOK.md) |
| Verify Lean files | [formal/README.md](formal/README.md) |

## Fixed Terminology

- **K-Bound**: population theory and the adapt/freeze/abstain framework.
- **KGA**: finite-sample empirical wrapper around a candidate adapter.
- **Population frontier**: M, gamma, and beta.
- **Empirical certificate**: Delta_hat and epsilon.
- **Abstain**: do not commit the update; continue prediction with the frozen fallback.

The population frontier and empirical certificate are related but distinct.
Real-data KGA does not receive beta, and empirical abstention does not by itself
prove structural non-identifiability.

## Evidence Tiers (revised 2026-07-26)

Authoritative version with numbers: `SUBMISSION_LEDGER.md §3`.

### CI-supported beats-both — one track, two candidates

- **CIFAR-10-C Tent**: five seeds, 432 cells per seed, CI beats-both. Unaffected by the radius fix
  (0 of 9 504 decisions change). Operating point is 6 of the 15 corruptions at severities {1,5} —
  say so wherever the track is reported (`DATA.md §2`).
- **CIFAR-10-C EATA**: same, with one caveat now disclosed — the adapt-gap CI excludes zero at 432
  i.i.d. cells but **not** when clustered by corruption family.
- The three-source OOF stream also clears the bar, but it is a **researcher-constructed routing
  mixture**, not unseen-domain transfer, and is labelled so.

### Point-estimate result — no CI claim

- **ImageNet-C SAR**: 27 cells per seed, five seeds. Under the declared leave-one-out-of-pool
  radius the point estimate still beats always-freeze (0.0289 vs 0.0319), but the freeze-gap CI at
  the seed-averaged unit includes zero. **Demoted from beats-both 2026-07-26.**

### One-sided no-harm — with source problems

Office-Home M v2, iWildCam H v2, RxRx1 J, Camelyon17 OOD. None is a natural beats-both win, and
three of the four have missing sources:

- **Camelyon17 OOD**: **sealed but not recomputable from release** — the promoted triple exists in
  one sealed YAML and in no computable artifact; the promoted FA_u = 0 is recorded nowhere.
- **Office-Home**: both source record files absent, and the entire runner directory is unreadable.
- **iWildCam**: source record file absent; 1 ADAPT decision, so the guarantee is untested.
- **RxRx1**: 0 ADAPT decisions, so the guarantee is untested.

### Diagnostic, negative or withheld

- **CIFAR-10.1** fails the declared transfer bar (FA_u 0.167, FA_c 0.444) — a pre-declared negative
  that came out worse than declared.
- **ImageNet-R**: four of four seeds; a null, and a worse one than the mean row shows — KGA is
  worse than always-adapt on 7 of 10 backbones and 4 of 10 have a 0% harmful base rate.
- **PACS**: three of three seeds; a null. Cannot be re-scored from the release.
- **CIFAR-10-C SAR** is withheld after a replay mismatch and contributes no empirical claim — note
  that the non-reproducing seed is also the seed on a different Python, torch and commit
  (`SUBMISSION_LEDGER.md §10`).

## Canonical Build

~~~bash
bash scripts/reproduce_submission.sh
bash scripts/build_dashboard.sh
~~~

Paper-only build (PDF **and** Word):

~~~bash
bash scripts/build_pdfs.sh
~~~

Outputs: `kbound_short.pdf`, `kbound_short.docx` (and matching `*_final_draft.*` copies).
Optional long paper: `BUILD_LONG=1 bash scripts/build_pdfs.sh`.

The generated result manifest is authoritative for repeated headline values.
Historical notes and archived runs are provenance, not automatic evidence.

**Caveat added 2026-07-26.** `bash scripts/reproduce_submission.sh` uses `set -euo pipefail`, so a
failure in step 1 silently prevents steps 2-9 from running; and several of its checks reference
files that are absent or unreadable. Read `REPRODUCE.md §1`'s "Known failures" box before treating
a green run as a clean bill.

## Dashboard

~~~bash
bash scripts/build_dashboard.sh
python3 -m http.server 8765 --directory .
~~~

Open http://127.0.0.1:8765/kbound_dashboard.html.

The dashboard reads the canonical paper manifest and the active
experiments/kbound/results/edge_real_phone_v1 tree. It never reads
archive/legacy_elara.

## Physical Validation

The edge code is a maintained, tested module rather than an informal demo. The
publication workflow is:

1. Prepare the protocol lock and deterministic checklists.
2. Capture S01-S02 and pass the source-model quality gate.
3. Capture S03-S06 and seal development plus conformal calibration.
4. Open S07-S08 once for held-out Phone A evaluation.
5. Capture S09-S10 on Phone B for replication.
6. Run the strict anti-leakage and publication gates.
7. Export camera tables and refresh the dashboard.

Start with:

~~~bash
python edge/scripts/preflight_r2.py
~~~

Browser preview, simulation, pilot data, and mock captures are connectivity or
software tests only. They cannot satisfy the publication gate.

## Formalization

~~~bash
cd formal
bash build.sh
~~~

The theorem map reports exactly which Lean declarations correspond to paper
statements. Do not describe the repository as a full foundational Mathlib
development: several measure-theoretic and deployment assumptions remain
external.

## Manuscript Policy

**Target venue: TMLR, single-column.** The submission core is `kbound_short.tex` +
`kbound_short_appendix.tex`. The 59-page `kbound.tex` predates the current claim corrections and is
not submission-ready; use it only as a source inventory for proofs, diagnostics, and background.
(Known stale row: `kbound.tex` Table 1 still carries the superseded single-seed ImageNet-C values
0.0108/0.0625/0.0319 — see `EDIT_NOTES_2026-07-23.md`.)

Because the venue is TMLR rather than a two-column conference, **no result is cut for length**. The
eight meta-tables the review flagged (`tab:regime-summary`, `tab:data-access`,
`tab:assumptions-role`, `tab:notation-main`, `tab:evidence-map`, `tab:failure-modes`,
`tab:claim-status`, `tab:baseline-faithfulness`) may be merged for readability, not for page count.

A balanced version should retain:

- problem and validity boundary;
- three core theory results plus the multiclass bridge;
- KGA architecture and calibration protocol;
- controlled beats-both evidence;
- natural no-harm and negative evidence;
- concise limitations and reproducibility.

Keep extended minimax, one-bit, martingale, historical ELARA, and large
diagnostic ladders in the supplement unless a venue explicitly allows them.
The detailed keep/move policy is in [KBOUND_MANUSCRIPT_STRATEGY.md](KBOUND_MANUSCRIPT_STRATEGY.md).
