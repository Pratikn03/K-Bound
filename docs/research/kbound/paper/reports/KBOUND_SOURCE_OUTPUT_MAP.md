# K-Bound current source-to-output map

Baseline captured on September 3, 2026, before the release-sync edits requested in
`KBOUND-2026-09-03-R1`. The checkout is on
`fix/kbound-current-release-sync-2026-09-03`. A separate linked worktree was not safe because the
current manuscript roots and full-report modules include untracked working-tree files that are not
present in `HEAD`; the existing publication branch is therefore the isolation boundary. No user
changes were reset, stashed, cleaned, or discarded.

## Current role map

| Current role | Source build | Root TeX | Build selector | Anonymity | Baseline pages | Baseline SHA-256 | PDF tracked in Git |
|---|---|---|---|---|---:|---|---|
| Named compact main paper | `docs/research/kbound/kbound_short_main.pdf` | `docs/research/kbound/kbound_short_main.tex` | `BUILD_SHORT_MAIN=1` | named author block and named PDF metadata | 22 | `5e66b290472ac26f1ee79590e680e3d9234789a237e6a00f45bc2178f1fcc2bd` | no |
| Named standalone supplement | `docs/research/kbound/kbound_short_supplement.pdf` | `docs/research/kbound/kbound_short_supplement.tex` | `BUILD_SHORT_SUPPLEMENT=1` | named author block and named PDF metadata | 25 | `8861e564671004f525cdb9b199aeefe62fa4497c555cb6140cf9c499d43b5ce6` | no |
| Anonymous integrated TMLR review manuscript | `docs/research/kbound/kbound_tmlr.pdf` | `docs/research/kbound/kbound_tmlr.tex` | `BUILD_LONG_TMLR=1` | `\anontrue`; anonymous visible author block and scrubbed metadata | 46 | `ef461adbec9c25dd274aee9a1b6f91f70dc32df3d39290b587c700738a80ec7b` | yes |
| Named full technical report | `docs/research/kbound/kbound_full_report.pdf` | `docs/research/kbound/kbound_full_report.tex` | `BUILD_FULL_REPORT=1` | named author block and named PDF metadata | 100 | `6c6f9f7d1d151b8ef9648ff93003fd196915129bc72432e585c51f453f68960a` | no |

The count-free release names are intentional. The current layout has 22, 25, 46, and 100 pages;
the obsolete count-bearing names in the incoming checklist are not treated as page-count
assertions. The pre-existing named combined main-plus-supplement build
`kbound_short_final_draft.pdf` is a comparison artifact, not one of the four current release roles.

## Build command

All four roles are built together so release publication cannot mix source generations:

```bash
BUILD_LONG_TMLR=1 \
BUILD_SHORT_MAIN=1 \
BUILD_SHORT_SUPPLEMENT=1 \
BUILD_FULL_REPORT=1 \
bash docs/research/kbound/scripts/build_pdfs.sh
```

`docs/research/kbound/scripts/build_pdfs.sh` always compiles the named combined comparison driver
first, then the four role builds selected above. A normal manuscript build leaves the protected
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
