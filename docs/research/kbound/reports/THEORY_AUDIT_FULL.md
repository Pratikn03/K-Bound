# K-Bound Full Theory Audit

Generated: 2026-06-29T17:06:53.019344+00:00
Verdict: **PASS**

## Closed spine (paper claims)

| Theorem | Status | Validators | Artifacts | Claims |
|---------|--------|------------|-----------|--------|
| ✓ `thm:headline` | closed | 2/2 | 2/2 | KB-CLAIM-001 |
| ✓ `thm:disagree` | closed | 2/2 | 2/2 | KB-CLAIM-001 |
| ✓ `thm:imp` | closed | 2/2 | 3/3 | KB-CLAIM-002 |
| ✓ `thm:cert` | closed | 2/2 | 1/1 | KB-CLAIM-003 |
| ✓ `thm:conj1-dichotomy` | closed | 3/3 | 3/3 | KB-CLAIM-025 |
| ✓ `thm:cmono-weakest` | closed | 1/1 | 1/1 | KB-CLAIM-028 |
| ✓ `thm:uncond-weakest` | closed | 1/1 | 1/1 | KB-CLAIM-029 |
| ✓ `thm:anytime` | extension | 1/1 | 1/1 | KB-CLAIM-031 |
| ✓ `thm:multicand` | extension | 1/1 | 1/1 | KB-CLAIM-032 |
| ✓ `thm:ev-rate` | extension | 1/1 | 1/1 | — |

## Open frontier (not claimed closed)

- `conj:dich-compute`: Constructive measurability; partial via val_knowability_dichotomy only.
- `thm:reg-iff`: Regression bracketing open; val_thm9prime_drift is partial probe.
- `conj:gen-capacity`: Exploratory validators; full removal of R1/R2 not closed.
- `thm:minimax-opt`: Wave 2 draft; not claimed solved in PROJECT_STATUS.

## Documentation drift (not failures)

- **docs/research/kbound/manuscript/**: Parallel book manuscript still marks conj:gen as open (resolved negatively in live kbound.tex). → Do not cite manuscript/ for submission status; use kbound_short.tex + PROJECT_STATUS.
- **docs/research/kbound/COMPLETION_STATUS_2026-06-19.md**: Superseded; banner added but file retained. → Use PROJECT_STATUS_AND_OPEN_PROBLEMS.md only.
