# K-Bound release table crosswalk

This crosswalk binds scientifically important tables to stable semantic LaTeX labels. Rendered
numbers differ legitimately because the compact main and standalone supplement compile
independently, while the TMLR manuscript integrates both and the full report inserts additional
technical-report tables. External material should cite a title or semantic label rather than a bare
table number.

Numbers below are from release `KBOUND-2026-09-03-R1`. They are regenerated and checked after the
four-role build; “not included” means the table is outside that document's role.

| Semantic label | Table title | Compact main | Standalone supplement | Integrated TMLR | Full report |
|---|---|---:|---:|---:|---:|
| `tab:method-target-comparison` | Conceptual comparison of methods by inferential target | 1 | not included | 1 | 1 |
| `tab:notation-levels` | Notation by inferential level | 2 | not included | 2 | 2 |
| `tab:claim-level-validity` | Validity obligations for the three claim levels | 3 | not included | 3 | 3 |
| `tab:evidence-status` | Evidence status and admissible claim by study | 4 | not included | 4 | 4 |
| `tab:cifar-primary` | Opened, dependent, constructed CIFAR-10-C diagnostic | 5 | not included | 5 | 5 |
| `tab:cifar-interval-diagnostics` | Retrospective CIFAR-10-C interval diagnostics | 6 | not included | 6 | 6 |
| `tab:cct-safe-utility` | Locked CCT-20 safe-utility endpoint | 7 | not included | 7 | 7 |
| `tab:cifar-family-sensitivity` | Retrospective CIFAR-10-C six-family sensitivity | not included | 6 | 13 | 20 |
| `tab:release-status` | Release-status inventory | not included | 8 | 15 | 22 |
| `tab:cct-strong-audit` | Locked CCT-20 strong-success audit | not included | 9 | 16 | 23 |
| `tab:deployment-checklist` | Non-empirical deployment checklist | not included | 12 | 19 | 26 |
| `tab:future-confirmation-protocol` | Future confirmatory natural-shift protocol | not included | 13 | 20 | 27 |

The source labels are defined in `kbound_submission_body.tex` and
`kbound_submission_supplement.tex`. The post-build reference audit reads the corresponding `.aux`
files and rejects unresolved references; no attempt is made to force identical numbering across
document roles.
