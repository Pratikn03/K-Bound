# Deprecated — do not cite for submission status

The parallel book manuscript under `manuscript/` is **not** the source of truth for
K-Bound claims after Wave 4 (2026-07-01).

Use instead:

- **Compact paper:** `../kbound_submission.tex` / `../kbound_short_final_draft.pdf`
- **Maintained full paper:** `../kbound_tmlr.tex` / `../kbound_tmlr.pdf`
- **Current documentation index:** `../DOCS_INDEX.md`
- **Closure gate:** `THEORY_100_PERCENT_CLOSURE_PLAN.md` + `formal/formal_audit.py --strict-100`

Several `manuscript/` chapters still contain superseded theorem and empirical claims, including
iWildCam values that are withheld in the current release. The tracked PDFs in this directory are
historical snapshots. Do not compile, cite, or use this tree as release evidence; use the maintained
full paper and current claim ledger instead.

## Preserved theory source: `theory_spine/`

`theory_spine/theory_beta_impossible.tex` and `theory_spine/theory_beta_estimable.tex` are reusable
theory-source records rather than empirical evidence. The superseded `../kbound.tex` archive inputs
them directly:

```
\input{manuscript/theory_spine/theory_beta_impossible}   % Sec. VI  (label app:beta-impossible)
\input{manuscript/theory_spine/theory_beta_estimable}    % Sec. VIII (label app:episode-beta)
```

They are derived from `kb_fixes/theory_beta_{impossible,estimable}.tex` with three edits:
`\providecommand` macro guards (`\Var`, `\E`, ...), `\ref{sec:experiments}` repointed to
`\ref{sec:exp}`, `\ref{rem:fa-marginal}` repointed to `\ref{thm:cert}` (that label is not in the
archival build), the CIFAR-10-C episode-coverage table promoted to a `table*` float
(`tab:episode-coverage`), several displays broken for the two-column measure, and one added
forward pointer (`rem:seam`). Their presence does not promote the deprecated monograph's empirical claims.
