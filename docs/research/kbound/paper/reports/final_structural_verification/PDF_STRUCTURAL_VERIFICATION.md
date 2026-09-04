# K-Bound PDF structural verification

Each current release role was checked with `pdfinfo`, `pdffonts`, `pdftotext -layout`, and a Ghostscript null-device parse.

| Release role | Status | Pages | Page size | Fonts embedded | Extracted characters | Ghostscript | SHA-256 |
|---|---|---:|---|---|---:|---|---|
| kbound_short_main | PASS | 20 | 612 x 792 pts (letter) | yes (30) | 97450 | clean | `8a60e37a3212236b2c2fce044e35eeeaab2e788c4fbbd2a3cc92463920b29dae` |
| kbound_short_supplement | PASS | 26 | 612 x 792 pts (letter) | yes (25) | 102960 | clean | `7549e8370ebb2035a5a6f08143d281f28d1f453d6d5818c20c5916dda27eea0a` |
| kbound_tmlr | PASS | 43 | 612 x 792 pts (letter) | yes (35) | 172698 | clean | `a33dbbb403eedeb4df532bec1c697a99b660e8ef7c7e1e0f5fff1c303951ea62` |
| kbound_full_report | PASS | 97 | 612 x 792 pts (letter) | yes (31) | 311099 | clean | `9de2fac1405bb5f89077d2ae8fc9ccdbf636b068f9f964684f9edcc2a5b1c6d1` |

## Exact tool paths

- `gs`: `/opt/homebrew/bin/gs`
- `pdffonts`: `/opt/homebrew/bin/pdffonts`
- `pdfinfo`: `/opt/homebrew/bin/pdfinfo`
- `pdftotext`: `/opt/homebrew/bin/pdftotext`

## Exact commands

### `kbound_short_main`

- `/opt/homebrew/bin/pdfinfo /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_short_main.pdf`
- `/opt/homebrew/bin/pdffonts /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_short_main.pdf`
- `/opt/homebrew/bin/pdftotext -layout /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_short_main.pdf /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/paper/reports/final_structural_verification/text/kbound_short_main.txt`
- `/opt/homebrew/bin/gs -q -dSAFER -dNOPAUSE -dBATCH -sDEVICE=nullpage /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_short_main.pdf`

### `kbound_short_supplement`

- `/opt/homebrew/bin/pdfinfo /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_short_supplement.pdf`
- `/opt/homebrew/bin/pdffonts /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_short_supplement.pdf`
- `/opt/homebrew/bin/pdftotext -layout /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_short_supplement.pdf /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/paper/reports/final_structural_verification/text/kbound_short_supplement.txt`
- `/opt/homebrew/bin/gs -q -dSAFER -dNOPAUSE -dBATCH -sDEVICE=nullpage /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_short_supplement.pdf`

### `kbound_tmlr`

- `/opt/homebrew/bin/pdfinfo /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_tmlr.pdf`
- `/opt/homebrew/bin/pdffonts /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_tmlr.pdf`
- `/opt/homebrew/bin/pdftotext -layout /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_tmlr.pdf /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/paper/reports/final_structural_verification/text/kbound_tmlr.txt`
- `/opt/homebrew/bin/gs -q -dSAFER -dNOPAUSE -dBATCH -sDEVICE=nullpage /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_tmlr.pdf`

### `kbound_full_report`

- `/opt/homebrew/bin/pdfinfo /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_full_report.pdf`
- `/opt/homebrew/bin/pdffonts /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_full_report.pdf`
- `/opt/homebrew/bin/pdftotext -layout /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_full_report.pdf /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/paper/reports/final_structural_verification/text/kbound_full_report.txt`
- `/opt/homebrew/bin/gs -q -dSAFER -dNOPAUSE -dBATCH -sDEVICE=nullpage /private/tmp/kbound-r1-final.SZUjxm/worktree/docs/research/kbound/release/current/kbound_full_report.pdf`


Detailed Poppler output, extracted text, and Ghostscript output are stored beside this report.
