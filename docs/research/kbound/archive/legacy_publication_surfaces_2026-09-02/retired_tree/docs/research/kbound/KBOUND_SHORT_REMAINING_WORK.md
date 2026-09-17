# K-Bound Remaining Work

Date: 2026-08-29

The manuscript, canonical result pipeline, KGA implementation tests, and current Lean development
are complete enough to circulate as a research draft. The following items require new evidence or
submission packaging; they cannot be solved by rewriting existing numbers.

The executable work breakdown, acceptance gates, required schemas, and run order are maintained in
[`KBOUND_EMPIRICAL_AND_RELEASE_CLOSURE_PLAN.md`](KBOUND_EMPIRICAL_AND_RELEASE_CLOSURE_PLAN.md).

## Empirical Evidence

- [x] Execute and retain a prospectively governed natural target result. CCT-20 completed as
  `SAFE_UTILITY_ONLY`: it tied always-freeze and protected against harmful always-adapt, but made
  zero ADAPT decisions and therefore did not establish strong mixed-regime routing.
- [x] Retain the So2Sat development outcome without target peeking. Neither candidate was feasible,
  so the protocol stopped before gate calibration; no target input, pixel, label, or score was read.
- [ ] If a natural beats-both claim is desired as a future extension, design a new versioned study
  with untouched evidence. The opened So2Sat gate-fit panel cannot be retuned and reused as fresh
  confirmation.
- [ ] Complete the preregistered physical-camera sessions and populate deployment, accuracy,
  macro-F1, regret, FA_u, FA_c, action exposure, and end-to-end latency tables from fresh logs.
- [ ] Add independent model seeds for Office-Home and iWildCam if those tracks are to support more
  than descriptive endpoint results.
- [ ] Export PACS per-cell `Delta`, `Delta_hat`, residual pools, and decisions for a complete replay.
- [ ] Use official neighboring-method code when claiming an official POEM, AETTA, or other baseline;
  retain `protocol-matched port` until then.

## Submission Packaging

- [x] Replace the local article shim with the vendored official TMLR style; the maintained long
  driver now uses it directly.
- [x] Select named versus anonymous submission mode: the maintained TMLR driver is anonymous and
  suppresses repository/author metadata.
- [ ] Freeze the canonical JSON, source manifest, PDFs, config hashes, and commit hash in one release.
- [x] Stamp the July reproducibility PASS, v0.1.0 notes, legacy CSV manifests, and conflicting
  process documents as historical without deleting their provenance.
- [ ] Run the release gate from a clean checkout and freeze the resulting checksums. The maintained
  full collection, tests, Lean build, manuscript checks, and PDF rendering are green under the
  supported Python 3.12 environment; Python 3.14 is not a release target.

## Claims That Must Remain Out

- [ ] Do not claim universal accuracy improvement or universal natural-shift no-harm.
- [ ] Do not call EATA corruption-cluster robust, ImageNet-C SAR CI-robust, or SAR CIFAR-10-C a win.
- [ ] Do not present zero-adapt natural rows as powered false-adapt evidence.
- [ ] Do not equate empirical abstention with structural non-identifiability.
- [ ] Do not equate `epsilon` with `beta`, or state that real-data KGA evaluates `|M| > beta`.
- [ ] Do not present the prospective physical-camera protocol or historical constructed mixture as
  completed held-out natural evidence. CCT-20 is completed but must remain `SAFE_UTILITY_ONLY`.
