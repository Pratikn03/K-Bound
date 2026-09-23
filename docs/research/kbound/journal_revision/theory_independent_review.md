# Independent review of journal Task 1

Reviewed 2026-09-22 local date (2026-09-23 UTC). Scope: Task 1 in
`IMPLEMENTATION_PLAN.md`, `theory_task_report.md`, `theory_audit.md`,
`theory_comparison.json`, and the change to `../formal/README.md`, checked
against active TeX, named Lean declarations, the new receipts and their logs,
and the cited primary literature. This review did not execute Lean, experiments,
or validators, and did not modify the implementation under review.

**Verdict: one minor literature-attribution correction; no substantive theory,
formal-scope, or Task 1 implementation discrepancy found in the inspected scope.**
This is not a proof of novelty, empirical assumptions, or final release integrity.

## Actionable finding

**TIR-1 — minor: avoid saying nonexchangeable coverage requires a modified
procedure.** In `theory_comparison.json`, row `coverage_transfer`,
`primary_claim` (line 88 at review), the wording says that coverage beyond
exchangeability “requires modified constructions.” Barber et al. also bound
coverage loss for the original split/full/jackknife+ procedures as special cases
of their framework; see the paragraph “Robustness results for the original
algorithms” on printed pages 18–19 of the
[primary paper](https://www.stat.cmu.edu/~ryantibs/papers/nexcp.pdf).
Modified constructions can improve robustness, but are not necessary merely
to state a qualified coverage bound under nonexchangeability.

Suggested replacement:

> Quantifies coverage loss from distribution drift and fitting asymmetry;
> modified weighted and randomized constructions can improve robustness.

The associated warning that KGA has no automatic nominal coverage under shift
is correct and should remain. No theorem, experiment, or manuscript rewrite is
needed for this correction.

## Evidence supporting the scoped verdict

- The active binary frontier inputs state fixed measurable predictors,
  positive disagreement mass, feasible margin and the full declared correctness
  class. The clipped interval, zero-budget case, strict zero-benefit boundary,
  and large-budget limits agree with `theory_audit.md` and the capstones in
  `Probability/MeasureTarget.lean` and `Probability/MeasureFrontier.lean`.
  The audit does not silently extend necessity to an arbitrary restricted class
  or to unrestricted multiclass labels.
- The coverage-to-action and population bridge distinguish marginal coverage
  from conditional acceptance error; the range-two Hoeffding radius applies to
  fresh paired accuracy observations rather than macro-F1 or transductive batch
  adaptation. The random extended-real manuscript radius is explicitly broader
  than the fixed-real certificate corollaries. The actual exchangeable random
  residual result in `Probability/MeasureConformal.lean` is correctly recognized.
- The paired-transport appendix retains the external transport budget, all bins,
  missing-source-class boxes, zero-mass qualification for sharp realization,
  empty-set abstention, fixed-predictor sampling, and the numerical inverse-beta
  limitation. Direct algebra checks support the displayed [.2,.8] and [.1,.4]
  examples, the 1/6 threshold, and the separate-subtraction interval [-.1,.6].
  The violated-transport counterexample is not presented as satisfying transport.
- Both new receipts have exactly the 150 names reconstructed from the current
  audit registry: 70 legacy names plus 80 foundation/counterexample/conditional
  names, with no duplicate, missing or extra entries. Each contains 150 kernel
  dependency records and no forbidden dependencies. Strict-core reports PASS;
  full-foundations reports FAIL with exactly the retained unrestricted historical
  one-bit/H-extension blocker. Their interpretation in the task report and README
  matches those statuses; 150 is not presented as 150 independent paper results.
- The source-snapshot SHA-256 and both log SHA-256 values in each new receipt
  independently match the files. Of 50 source-snapshot inputs, 48 still match;
  the two differences are the concurrently integrated body and supplement.
  All recorded Lean sources, formal audit code, dependency pins and the old
  dated receipt remain identical to the snapshot. The README and theory report
  explicitly restrict the receipts to that snapshot. The supplement's added
  exclusions at lines 1966–1970 agree with the review: the fibre-supremum audit,
  paired LP and numerical solver/CDF implementation are not additional one-to-one
  Lean encodings.
- The inspected Jhawar–Wang identification statements, Li–Han–Ma matrix-constraint
  results, MORPHEUS method description, split-conformal construction, and original
  Clopper–Pearson paper support the comparisons' inherited-versus-added framing.
  Qiu–Stoye is expressly used only at abstract/conceptual level. The comparison
  does not claim global priority or empirical superiority. TIR-1 is the sole
  source-attribution discrepancy found.

## Review limits and remaining integration obligation

No clean rebuild or independent proof-assistant replay was performed in this
review; verification relies on inspecting the recorded runs and matching their
named inputs. The original Hoeffding publisher full text remained inaccessible;
the range-two scaling and the actual Lean statement were checked, and the
implementation report already discloses the literature-access limit.

The two concurrent manuscript changes mean the Task 1 snapshot must not be
described as a seal of the final integrated manuscript. The plan's Task 5 already
requires fresh source-bound final verification and a new release seal. This is
a remaining integration obligation, not an undisclosed failure of the Task 1
receipt. No further actionable discrepancy was identified.
