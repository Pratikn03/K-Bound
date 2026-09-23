# Task 1 report: theory, comparison, and formal scope

Implemented the bounded theory task without changing manuscript sources,
experiment inputs, historical receipts, Lean declarations, or dependency pins.
No substantive theorem correction was needed after checking the active proof
chain and paired-transport supplement. The stale formal README was corrected.

## Delivered files

- `theory_audit.md`: edge cases, assumptions, theorem scope, numerical boundaries,
  and one-to-one formalization exclusions.
- `theory_comparison.json`: eight claim-level primary-source comparisons with
  inherited content, explicit extra content, assumptions, and forbidden
  overclaims. “Added” is a local mathematical comparison, not novelty priority.
- `theory_source_snapshot_journal_2026_09_22.json`: source hashes and pre-integration
  HEAD. Includes every project Lean source, audit script, dependency pins,
  reviewed active TeX sources, and old strict receipt.
- `../formal/README.md`: corrected 150-declaration inventory and current
  probability/construction scope.
- Newly named strict/full audit JSON receipts and logs under `../formal/`;
  historical `formal_strict_core_receipt_2026_09_22.json` remains unchanged.

## Exact reviewed anchors

| Source | Reviewed content |
|---|---|
| `kbound_submission_body.tex` | `ass:deploy`, `rem:gamma-residual`, `def:risk-align`, `def:strict-sound`, `lem:fibre`, `thm:compact-beta-minimax`, `eq:compact-beta-radius`, `eq:compact-rank`, `sec:nextphase-paired` |
| `paper/sections/theory_core_main.tex` | `lem:reduction`, `lem:nonid`, `cor:matched-abstain`, `prop:closed-band`, `thm:frontier`, clipped identified-set paragraph |
| `paper/sections/theory_certificate.tex` | `thm:certificate`, `rem:unit-scaling`, `thm:population-transfer`, sampled-frontier remark, `rem:fa-marginal`, fallback |
| `kbound_submission_supplement.tex` | `app:population-transfer`, protocols P1–P5, `thm:app-pop-cert`, `prop:sampled-frontier`, `app:compact-formal`, `app:nextphase-paired-proof`, `prop:nextphase-transport`, `eq:app-paired-polytope`, `eq:app-paired-dual` |

The root agent concurrently edited introduction, caption/prose, and empirical
integration. The snapshot records the body reviewed before those edits; the
root reports no theorem/proof changes. Final release verification must bind the
final source commit, and any substantive proof edit requires a new audit. No
unused `theory_setup.tex` statement substitutes for these active anchors.

## Verification command scope

Working directory: `docs/research/kbound/formal` in the journal worktree.
Interpreter: `/Users/pratik_n/.cache/kbound-release-verify.MsziU5/venv/bin/python`.

```text
<interpreter> formal_audit.py --build --strict-core --json-out formal_strict_core_receipt_journal_2026_09_22.json
<interpreter> formal_audit.py --build --strict-core --full-foundations --json-out formal_full_scope_receipt_journal_2026_09_22.json
```

Each command captures stdout/stderr in its corresponding journal log. The
existing tooling invokes `lake build KBound`, checks declaration registration,
scans forbidden tokens/compiler warnings, and inspects registered kernel axiom
dependencies. `lake update` was not run. A build may reuse pinned compiled
artifacts; the receipt is a fresh build/check invocation, not a clean-room
rebuild of all Mathlib dependencies.

## Limitations retained

No claim of a new general partial-identification, testing, LP-confidence, or
conformal theorem is supported by this comparison. The distinctive mathematical
content is the declared binary paired-benefit frontier/audit floor and the
conditional paired transport contrast. Scientific novelty across all literature
and empirical superiority are unestablished. The latter construction has no
complete Lean/solver/Clopper–Pearson inverse-CDF proof chain.

The historical orbit-only one-bit sufficiency assertion is refuted. It should
remain excluded rather than be “completed.” Corrected fibre-consistent decoding
and conditional H-rate budget propagation are narrower verified statements;
prospective evidence and a separately justified nondegenerate ratio-rate model
remain outside the receipt. Real-world exchangeability, transport, deployment
richness and fresh evaluation provenance are not established by any build.

## Observed check results

Strict-core: **PASS**, exit 0; 150/150 registered declarations, successful
build and kernel-axiom audit. Full scope: **FAIL_EXPECTED**, exit 1; the same
150/150 declarations pass, with exactly the disclosed historical one-bit/H
extension blocker. `full_foundations_scope_complete=false` in both receipts.
Formal source hashes and the historical receipt match the initial snapshot;
only the concurrently integrated main body differs. All new theory JSON files
parse successfully. Log files are ignored by the repository defaults and must
be explicitly included in the final evidence package (or force-added if desired).
