# Superseded K-Bound drafts

SUPERSEDED DRAFTS — NUMBERS, PROTOCOLS, THEORY SCOPE, AND CLAIMS DO NOT MATCH THE CURRENT K-BOUND SUBMISSION. DO NOT CITE, SUBMIT, OR USE THESE FILES AS THE SOURCE OF CURRENT RESULTS.

This directory exists only to preserve provenance. The byte-for-byte historical
files are isolated under `originals/`. The adjacent `*_SUPERSEDED.pdf` files add
a visible first-page warning without rasterizing the underlying PDF pages. Neither
set is part of the current release.

The historical root drivers `docs/research/kbound/kbound.tex` and
`docs/research/kbound/kbound_short.tex` are absent from the active working tree. Their original
bytes remain recoverable from Git history; they are not reconstructed here because doing so would
restore obsolete build entry points. This README is the current, explicit supersession marker for
those historical roots as well as for their PDF outputs.

## Current replacements

| Superseded output | Current count-free replacement |
|---|---|
| `kbound_short.pdf` | `kbound_short_main.pdf` together with `kbound_short_supplement.pdf` |
| `kbound.pdf` | `kbound_full_report.pdf` |
| `kbound_submission.pdf` | `kbound_tmlr.pdf` |

Only the PDFs published by the current release command are authoritative. Page
counts embedded in older release names are not version identifiers.

## Preserved originals

| File | Pages | SHA-256 |
|---|---:|---|
| `originals/kbound_short.pdf` | 58 | `c87cbf8042e160cd8217bc8fa698b7ff3fb77ff6b0f0469c97d2bcb4d2c1834a` |
| `originals/kbound.pdf` | 52 | `eb83c197be9a06ef6f9167fe6872daf608836d7ee8de8cd4b58b6f86b677bf42` |
| `originals/kbound_submission.pdf` | 35 | `acd938247b08b5f8665dd3b0d7a307d5b157343c8eb841aadb6195e54736d6d4` |

## Reproduction and verification

From the repository root, run:

```sh
docs/research/kbound/archive/superseded_do_not_cite/build_warning_copies.sh
python3 docs/research/kbound/archive/superseded_do_not_cite/verify_quarantine.py
```

The builder refuses to replace an existing preserved original unless it is
byte-identical to the historical source. The verifier checks the fixed hashes,
page counts, searchable watermark text, retained searchable manuscript text, and
the absence of the old filenames from default public-release surfaces.
