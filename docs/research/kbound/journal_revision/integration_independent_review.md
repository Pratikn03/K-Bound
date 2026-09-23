# Independent review of manuscript integration

**Verdict: PASS in the reviewed scope. No unresolved actionable discrepancy.**
Reviewed 2026-09-22 local date (2026-09-23 UTC). This is an internal review of
claim correctness and integration, not an external endorsement or a statement
that prospective empirical limitations have been solved.

## Scope and resolved finding

Reviewed `journal_synthesis.tex`, the body/supplement differences against
`output/journal_revision_20260922/baseline`, the abstract change against Git,
all 30 claim-register entries, `validate_journal_revision.py`, and its integrity
tests. The editorial baseline has no separate abstract copy; the abstract
comparison therefore uses the maintained Git version explicitly. Frozen
authorities and previously reviewed theory/empirical outputs were consulted
without changing them. No experiments, Lean build or test suite were rerun.

**IIR-1, resolved:** the first version of claim T5 listed conditional independence
and bounded losses without explicitly binding their common mean to the target
benefit. The cited proposition uses fresh conditionally i.i.d. evaluation draws
with paired correctness differences in [-1,1]. Root corrected the register to
state the target law, common conditional mean Delta, positive sample count,
fixed predictor pair and independent fitting/adaptation history. I reread the
corrected JSON and confirmed these assumptions match
`paper/sections/theory_certificate.tex`, lines 44–68. No manuscript theorem change
was necessary.

## Checks supporting the verdict

- The added opening and conclusion distinguish an information obstruction from
  empirical calibration transfer. They do not turn the empirical gate into an
  implementation of the population frontier. The abstract now reports lower
  observed Tent loss without suggesting a general calibration advantage.
- Figure 4's new caption identifies the preserved historical Tent seed-0
  mixture illustration, separates it from the current Protocol B aggregate,
  and avoids calling mixture reweighting independent deployments. Its evidence
  entry points to the saved seed-0 decisive-results record. The producer's
  mixture routine constructs the displayed harmful-fraction sweep from saved
  conditions rather than independently sampled deployments.
- New scientific positioning agrees with the reviewed primary comparisons.
  It limits necessity to the declared rich class, treats paired transport as a
  different construction, distinguishes shared-feasible-set cancellation from
  dominance over another method, and preserves numerical endpoint exclusions.
  The corrected Barber statement explicitly includes original conformal
  procedures. No global priority claim is introduced.
- The new accuracy differences and table values agree with the versioned
  empirical synthesis. Accuracy uses equal cell weights; regret and 100L5 use
  percentage-point scaling. Tent, EATA and SAR retain gains of +0.6183,
  +0.1673 and -0.1326 percentage points relative to always-adapt. The six-cluster
  sensitivity table retains negative and positive bootstrap endpoints, the
  JPEG deletion's reduced point advantage and reversed margin comparison,
  and its explicitly retrospective status. It does not claim population
  significance from the favorable matched-exposure descriptive interval.
- Cell and all-cells-in-group inclusion have separate columns and denominators.
  The text identifies 2,160 cells and 216 complete semantic groups per CIFAR
  candidate while retaining only six corruption types. These are not called
  independent deployments. CCT's 52.2936% versus 34.1034% comparison remains
  retention with zero ADAPT, and its secondary 16-indicator per-cell macro-F1
  is not promoted to pooled or confirmatory macro-F1. Invalid iWildCam routing
  remains withheld. Failed So2Sat screening and the absence of an eligible fresh
  natural-shift study remain explicit.
- All four new appendix/table labels have exactly one definition in the active
  inspected sources and a matching reference. The four literature citation keys
  used by the synthesis exist in the maintained expanded bibliography.
- Independently checked every current claim location anchor and evidence
  SHA-256: all 30 claims have matching anchors and bytes. The statuses distinguish
  conditional theory, descriptive studies, supported implementation statements,
  and unestablished deployment claims. Hash validity is not used as scientific
  authentication. The validator and tests enforce references, byte changes,
  scope-safe paths, duplicate IDs and permitted scientific statuses; they do
  not purport to establish the truth or completeness of the claims.
- Independently checked `integration_receipt.json`: the reviewed-source ZIP
  hash matches; its two archived source members match the earlier reviewed
  hashes; both current body/supplement hashes match the integration receipt.
  The added supplement text preserves the explicit limits of one-to-one Lean
  correspondence. The source snapshot remains historically scoped rather than
  being silently relabeled as final-source verification.

## Remaining release boundary

Final source-bound gates, exported PDF/Word visual inspection, packaging and
downloaded-asset verification are outside this review and still require their
own completion records. This review does not infer scientific quality from
validator counts or close the blocked prospective study.
