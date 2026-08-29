# K-Bound Remaining Work

Date: 2026-08-24

The manuscript, canonical result pipeline, KGA implementation tests, and current Lean development
are complete enough to circulate as a research draft. The following items require new evidence or
submission packaging; they cannot be solved by rewriting existing numbers.

The executable work breakdown, acceptance gates, required schemas, and run order are maintained in
[`KBOUND_EMPIRICAL_AND_RELEASE_CLOSURE_PLAN.md`](KBOUND_EMPIRICAL_AND_RELEASE_CLOSURE_PLAN.md).

## Empirical Evidence

- [ ] Run a prospectively locked natural-shift experiment with mixed helpful and harmful conditions,
  separate estimator fitting and residual calibration, untouched target environments, and inference
  at the environment or independently trained checkpoint unit.
- [ ] Complete the preregistered physical-camera sessions and populate deployment, accuracy,
  macro-F1, regret, FA_u, FA_c, action exposure, and end-to-end latency tables from fresh logs.
- [ ] Add independent model seeds for Office-Home and iWildCam if those tracks are to support more
  than descriptive endpoint results.
- [ ] Export PACS per-cell `Delta`, `Delta_hat`, residual pools, and decisions for a complete replay.
- [ ] Use official neighboring-method code when claiming an official POEM, AETTA, or other baseline;
  retain `protocol-matched port` until then.

## Submission Packaging

- [ ] Replace the local article shim with the official venue style and rerun visual QA.
- [ ] Select named versus anonymous submission and remove repository/author metadata accordingly.
- [ ] Freeze the canonical JSON, source manifest, PDFs, config hashes, and commit hash in one release.
- [ ] Archive or clearly stamp older July audit documents whose historical wording conflicts with
  the current README. They remain provenance but are not current status.
- [ ] Run the release gate from a clean checkout and freeze the resulting checksums. The maintained
  full collection, tests, Lean build, manuscript checks, and PDF rendering are green under the
  supported Python 3.12 environment; Python 3.14 is not a release target.

## Claims That Must Remain Out

- [ ] Do not claim universal accuracy improvement or universal natural-shift no-harm.
- [ ] Do not call EATA corruption-cluster robust, ImageNet-C SAR CI-robust, or SAR CIFAR-10-C a win.
- [ ] Do not present zero-adapt natural rows as powered false-adapt evidence.
- [ ] Do not equate empirical abstention with structural non-identifiability.
- [ ] Do not equate `epsilon` with `beta`, or state that real-data KGA evaluates `|M| > beta`.
- [ ] Do not present the prospective camera protocol or historical constructed mixture as completed
  held-out natural evidence.
