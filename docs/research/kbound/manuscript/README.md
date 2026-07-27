# Deprecated — do not cite for submission status

The parallel book manuscript under `manuscript/` is **not** the source of truth for
K-Bound claims after Wave 4 (2026-07-01).

Use instead:

- **Short paper:** `kbound_short.tex` / `kbound_short.pdf`
- **Long paper:** `kbound.tex` / `kbound.pdf`
- **Status ledger:** `PROJECT_STATUS_AND_OPEN_PROBLEMS.md`
- **Closure gate:** `THEORY_100_PERCENT_CLOSURE_PLAN.md` + `formal/formal_audit.py --strict-100`

Several `manuscript/` chapters still mark `conj:gen` and related items as open; those rows
were closed negatively or as dichotomies in the live paper (see `main_theory_5.tex` and
Wave 4 appendix in `kbound.tex`).

## Exception: `theory_spine/` is live

`theory_spine/theory_beta_impossible.tex` and `theory_spine/theory_beta_estimable.tex` are **not**
part of the deprecated book edition. They are live main-body sections of `../kbound.tex`
(the impossibility spine, acts 2 and 4) and are `\input` from it directly:

```
\input{manuscript/theory_spine/theory_beta_impossible}   % Sec. VI  (label app:beta-impossible)
\input{manuscript/theory_spine/theory_beta_estimable}    % Sec. VIII (label app:episode-beta)
```

They are derived from `kb_fixes/theory_beta_{impossible,estimable}.tex` with three edits:
`\providecommand` macro guards (`\Var`, `\E`, ...), `\ref{sec:experiments}` repointed to
`\ref{sec:exp}`, `\ref{rem:fa-marginal}` repointed to `\ref{thm:cert}` (that label is not in the
long build), the CIFAR-10-C episode-coverage table promoted to a `table*` float
(`tab:episode-coverage`), several displays broken for the two-column measure, and one added
forward pointer (`rem:seam`). Edit these copies, not the `kb_fixes/` originals, for the long build.
