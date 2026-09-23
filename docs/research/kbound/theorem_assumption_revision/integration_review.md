# Integration review of the theorem and assumption revision

Review date: 2026-09-23. Base commit:
`69548cf44bcbede86053ea3ef6572ef57b2a1615`.

No unresolved blocking regression was identified in the reviewed non-formal
changes. This conclusion is limited to the paths and probes below; it is not a
claim that every theorem or empirical assumption has been independently proved.

## Scope and independence

This review inspected the current diffs for `kga/assumptions.py`, the legacy
`assumption_audit` package, its correction/replay generator, the new
theorem-assumption correspondence checker and tests, and the changed active
manuscript paragraphs. It also inspected the theorem/assumption registers,
the statistical review, and the corrected legacy replay.

The reviewer authored the separate random-radius Lean module. That module and
its formal proof work were deliberately excluded from this independent
integration review. They have separate kernel checks and require the other
reviewer's correspondence assessment. This report is not an external peer review.

## Findings and disposition

1. **Missing Lean-reference binding guard — repaired and retested.** Initially,
   deleting `MeasureTarget.lean` from the register's `scope.source_files` left
   the correspondence checker at PASS, even though registered statements still
   referenced its declarations. The existing real register was bound; the
   defect was failure to detect a future omitted binding. The checker now
   requires bindings recursively for Lean references, including supporting
   claims and historical exclusions. Repeating the deletion probe now produces
   FAIL with explicit `unbound Lean reference` errors.
2. **Stale statement/proof location metadata — repaired and retested.** Initially,
   setting a proof's line number to `999999` still passed while the proof hash
   remained correct. The checker now compares statement and proof start/end
   lines to the active-source inventory. Repeating the probe now produces FAIL
   identifying the mismatched proof location.
3. **Legacy replay numbers versus preserved history — clarified.** The v2 replay
   retains the v1 five-no-warning/one-warning categories but does not reproduce
   every historical diagnostic value. For example, the benign feature-range
   score changes from approximately `0.111818` in v1 to `0.105455` in the
   current-input replay; the shifted score changes from `0.14` to approximately
   `0.133636`. The statistical review now explicitly states this distinction.
   The old artifacts remain preserved; v2 is a posthoc correction/replay, not a
   recovered original execution or a new prospective study.

No additional material runtime defect or scientific overclaim was found in
the changed paths inspected. The following changes are appropriately scoped:

- Passing the legacy support/drift diagnostics no longer authorizes ADAPT or
  asserts that coverage applies. Unmeasured residual drift is null, not zero.
- The maintained conformal helper rejects malformed/nonfinite residuals and
  invalid alpha; empty calibration produces an infinite-radius diagnostic
  fallback. The finite-rank lower bound is distinguished from true coverage.
- Group-label counts are explicitly declared counts, not established numbers
  of independent observations. Coverage-basis metadata validation is explicitly
  distinguished from byte authentication and distributional validity.
- The active manuscript supplies the integer sample-size restrictions and
  independent fresh-sample condition needed for the sampled-frontier argument.
  Conditioning on each positive disagreement count, then averaging, supports
  the stated marginal bound; count zero abstains.
- The manuscript's new formal-scope wording does not turn coverage, deployment
  exchangeability, label-blind collection, or numerical LP/quantile correctness
  into consequences of the formal audit.

## Checks performed

The following focused tests passed:

```sh
PYTHONPATH=. python -m pytest tests/test_assumptions.py tests/test_legacy_assumption_audit_scope.py tests/test_theorem_assumption_revision.py -q
```

After the binding/location fixes, the correspondence test module was rerun and
all 19 tests passed. Both concrete mutation probes above changed from PASS to
the expected FAIL. The unmodified current register passed with 13 named
statements and 20 assumptions. The checker explicitly emits
`mathematical_correctness_established=false` and
`empirical_assumptions_established=false`.

The legacy replay regression also checks that original v1 JSON/Markdown bytes
are unchanged and a second run cannot overwrite an existing correction.
`git diff --check` passed during review.

## Limits

The source inventory is a bounded parser for this repository's literal TeX
inputs, not a complete TeX interpreter. Declaration-name lookup is a reference
existence check; kernel validity and exact formal source snapshots remain
separate checks. The scientific statuses in the registers depend on mathematical
and methodological review, not on their JSON consistency test.

This review did not rerun model training, reopen stopped target evaluations,
authenticate missing historical producers, independently validate the new
Lean proof, inspect rebuilt PDF/Word pages, or verify every public release
asset. Those activities must not be inferred from this integration result.
