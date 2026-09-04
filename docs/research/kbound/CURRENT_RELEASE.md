# Current K-Bound manuscript release

Release ID: `KBOUND-2026-09-03-R1`  
Source closure: September 3, 2026  
Source snapshot commit: `pending-source-only-commit`  
Canonical panel SHA-256: `35d4c165843de1ece3cb35ffb4ac50dbfcbfa33b646754e2b6d133fbdfa78c6e`  
Source-manifest SHA-256: `03b1d2b1e9e5ed1cf835126f871ed83eb8497a1ac81727b3955af0d69eb89742`

## Current outputs and upload roles

| Output | Role | Review-system use |
|---|---|---|
| `release/current/kbound_short_main.pdf` | Named compact main paper | Identified reading copy; do not upload to a double-blind review system unless the venue explicitly permits identified material. |
| `release/current/kbound_short_supplement.pdf` | Named standalone supplement | Identified reading copy; do not upload to a double-blind review system unless the venue explicitly permits identified material. |
| `release/current/kbound_tmlr.pdf` | Anonymous integrated TMLR review manuscript | This is the only PDF in this release intended for the TMLR double-blind review upload. |
| `release/current/kbound_full_report.pdf` | Named full technical report | Identified archival report; do not upload to a double-blind review system unless the venue explicitly permits identified material. |

The release uses count-free filenames because pagination is a verified output property, not a claim
embedded in a filename. Actual page counts and hashes are recorded in the release-repair report and
checksum manifest after every build.

## Superseded filenames

`kbound_short.pdf`, `kbound.pdf`, and `kbound_submission.pdf` are superseded. They are preserved only
under `archive/superseded_do_not_cite/` for provenance. Do not cite, submit, or use them as the source
of current results. The combined `kbound_short_final_draft.pdf` is a nonrelease comparison build and
is not part of the four-document current release.

## Exact manuscript build

From the repository root, after choosing the reviewed 12-character source snapshot commit:

```bash
SOURCE_SNAPSHOT_COMMIT=<12-hex-source-commit> \
PYTHON=<python-3.12-executable> \
BUILD_LONG_TMLR=1 BUILD_SHORT_MAIN=1 BUILD_SHORT_SUPPLEMENT=1 BUILD_FULL_REPORT=1 \
bash docs/research/kbound/scripts/build_pdfs.sh
```

The all-four invocation publishes the four PDFs and `KBOUND_CURRENT_SHA256SUMS.txt` to both
`docs/research/kbound/release/current/` and the repository-level `output/pdf/`. Partial manuscript
builds do not publish a mixed-age release set. No manuscript build authorizes protected So2Sat
authority refresh, experiment execution, model training, action changes, or recalibration.

## Verification

```bash
python docs/research/kbound/scripts/audit_current_kbound_release.py \
  --pdf-dir docs/research/kbound/release/current --require-exact-inventory
python docs/research/kbound/scripts/render_pdf_pages.py \
  --pdf-root docs/research/kbound/release/current \
  --output-root docs/research/kbound/paper/reports/final_renders
```

See `paper/RELEASE_TABLE_CROSSWALK.md` for stable semantic table identities across layouts and
`paper/reports/TMLR_ANONYMITY_AUDIT.md` for the review-PDF privacy check.
