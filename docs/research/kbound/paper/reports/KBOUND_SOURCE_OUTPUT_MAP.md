# K-Bound current source-to-output map

Baseline captured on September 3, 2026, before the release-sync edits requested in
`KBOUND-2026-09-03-R1`; final outputs were sealed on September 4, 2026. The source checkout is on
`fix/kbound-current-release-sync-2026-09-03`. Because the primary iCloud-backed checkout contains
unrelated and partly dataless files, no reset, stash, clean, or broad checkout was used. Source
edits were committed on that branch, and the final build and audits ran from an isolated sparse
mirror at `/tmp/kbound-r1-final.SZUjxm/worktree` containing the exact source closure. The source
snapshot commit is `efecbafd74e192c1314ea81a0aa8868a8e784a94`.

## Current role map

| Current role | Published PDF | Root TeX | Shared source | Generated macros | Build selector | Anonymity | Baseline pages | Final pages | Final SHA-256 | PDF tracked in Git |
|---|---|---|---|---|---|---|---:|---:|---|---|
| Named compact main paper | `docs/research/kbound/release/current/kbound_short_main.pdf` | `docs/research/kbound/kbound_short_main.tex` | abstract core, submission body, theory modules, references | primary tables, interval diagnostics, CCT reporting macros, release identity | `BUILD_SHORT_MAIN=1` | named visible author block and named metadata | 22 | 20 | `8a60e37a3212236b2c2fce044e35eeeaab2e788c4fbbd2a3cc92463920b29dae` | yes, current release artifact |
| Named standalone supplement | `docs/research/kbound/release/current/kbound_short_supplement.pdf` | `docs/research/kbound/kbound_short_supplement.tex` | submission supplement, figure fallback, references | auxiliary tables, family sensitivity, CCT tables, release identity | `BUILD_SHORT_SUPPLEMENT=1` | named visible author block and named metadata | 25 | 26 | `7549e8370ebb2035a5a6f08143d281f28d1f453d6d5818c20c5916dda27eea0a` | yes, current release artifact |
| Anonymous integrated TMLR review manuscript | `docs/research/kbound/release/current/kbound_tmlr.pdf` | `docs/research/kbound/kbound_tmlr.tex` | main-paper closure plus appendices and float parameters | all main and appendix generated tables plus anonymous release identity | `BUILD_LONG_TMLR=1` | `\anontrue`; anonymous author block; empty Author metadata; no source commit printed | 46 | 43 | `a33dbbb403eedeb4df532bec1c697a99b660e8ef7c7e1e0f5fff1c303951ea62` | yes, current release artifact |
| Named full technical report | `docs/research/kbound/release/current/kbound_full_report.pdf` | `docs/research/kbound/kbound_full_report.tex` | main-paper closure, supplement, report extensions, three full-report atlas modules | all current generated tables plus named release identity | `BUILD_FULL_REPORT=1` | named visible author block and named metadata | 100 | 97 | `9de2fac1405bb5f89077d2ae8fc9ccdbf636b068f9f964684f9edcc2a5b1c6d1` | yes, current release artifact |

The count-free release names are intentional. The final layout has 20, 26, 43, and 97 pages;
the obsolete count-bearing names in the incoming checklist are not treated as page-count
assertions. The pre-existing named combined main-plus-supplement build
`kbound_short_final_draft.pdf` is a comparison artifact, not one of the four current release roles.
The integrated TMLR manuscript finishes at 43 pages because avoidable float gaps and repeated prose
were removed while required supplement tables were restored and wide tables were kept after their
introducing prose; no blank material or forced page
break was added merely to reach an exact page count within the blueprint's approximate
42--45-page range.

## Build command

All four roles are built together so release publication cannot mix source generations:

```bash
SOURCE_SNAPSHOT_COMMIT=efecbafd74e1 \
PYTHON=/tmp/kbound-release-venv2/bin/python \
BUILD_LONG_TMLR=1 BUILD_SHORT_MAIN=1 \
BUILD_SHORT_SUPPLEMENT=1 BUILD_FULL_REPORT=1 \
bash docs/research/kbound/scripts/build_pdfs.sh
```

The command was run twice. References, contents, and counters settled, and all four PDF SHA-256
digests matched across the two builds.
`docs/research/kbound/scripts/build_pdfs.sh` compiles the named combined comparison driver first,
then the four role builds selected above. A normal manuscript build leaves the protected
So2Sat development authority untouched; `KBOUND_AUTHORIZE_PROTECTED_SO2SAT` must remain unset for
this release-sync task.

## Shared source closure

All four documents share the following current sources:

- `docs/research/kbound/paper/generated/kbound_numbers.tex`
- `docs/research/kbound/paper/generated/cct20_numbers.tex`
- `docs/research/kbound/paper/figure_fallback.tex`
- `docs/research/kbound/paper/references_kbound_expanded.tex`

The main paper, integrated TMLR manuscript, and full report additionally share:

- `docs/research/kbound/kbound_abstract.tex`
- `docs/research/kbound/kbound_abstract_core.tex`
- `docs/research/kbound/kbound_submission_body.tex`
- `docs/research/kbound/paper/sections/theory_core_main.tex`
- `docs/research/kbound/paper/sections/theory_certificate.tex`
- `docs/research/kbound/paper/generated/kbound_primary_accuracy_table.tex`
- `docs/research/kbound/paper/generated/current_policy_interval_diagnostics.tex`
- `docs/research/kbound/paper/generated/cct20_safe_utility_display.tex`

The supplement and integrated TMLR manuscript share:

- `docs/research/kbound/kbound_submission_supplement.tex`
- `docs/research/kbound/paper/generated/kbound_auxiliary_accuracy_table.tex`
- `docs/research/kbound/paper/generated/kbound_auxiliary_balanced_accuracy_table.tex`
- `docs/research/kbound/paper/generated/current_policy_family_sensitivity.tex`
- `docs/research/kbound/paper/generated/current_policy_interval_diagnostics_groups.tex`
- `docs/research/kbound/paper/generated/cct20_primary_table_display.tex`
- `docs/research/kbound/paper/generated/cct20_location_effects_display.tex`

The full report also includes `kbound_full_report_extensions.tex`, the three files under
`paper/full_report/`, the shared supplement, and the protected presentation macro
`paper/generated/so2sat_numbers.tex`. The integrated TMLR driver additionally includes
`paper/float_params.tex`.

## Numerical authorities and release boundary

- Canonical CIFAR panel:
  `experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`
  (`35d4c165843de1ece3cb35ffb4ac50dbfcbfa33b646754e2b6d133fbdfa78c6e`).
- Canonical source manifest:
  `experiments/kbound/results/reconciled_panels_v1/source_manifest.json`
  (`03b1d2b1e9e5ed1cf835126f871ed83eb8497a1ac81727b3955af0d69eb89742`).
- CCT-20 release authority and receipt:
  `docs/research/kbound/paper/generated/cct20_release_manifest.json` and
  `docs/research/kbound/paper/generated/cct20_release_manifest.json.receipt.json`.
- Generated TeX is presentation output and must agree with these authorities; it is not an
  independent numerical source.

## Superseded roots

The historical outputs `kbound_short.pdf`, `kbound.pdf`, and `kbound_submission.pdf` are not current
roots. Their former source surfaces are deleted from the active working tree or already retained in
the dated legacy/stale publication archives. They must not be reconstructed or used as prose
templates. A dedicated `archive/superseded_do_not_cite/` index will record their status without
promoting them into a release target.
