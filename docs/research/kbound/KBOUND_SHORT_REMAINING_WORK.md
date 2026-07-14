# K-Bound Short Paper Remaining Work

The maintained software, compact claim evidence, dashboard, paper build, Lean
strict-core verification, and multiseed completion implementation are
release-complete. Expensive measurements and external validation remain. They
are not silently represented as finished results.

## Completed implementation

- Added one fail-closed launcher and a versioned protocol lock for the remaining
  multiseed matrix.
- Migrated the required raw-data runners into the clean repository with no
  runtime dependency on the historical AutoML tree.
- Replaced interpolated certificate residual quantiles with exact observed order
  statistics and labeled stress-grid calibration as empirical cross-fitting.
- Corrected PACS to use disjoint source-train images, development domains,
  residual-calibration domain, and untouched target domain.
- Added one outer held-out-seed scorer for all supported datasets with schema
  validation, target-label-isolated routing, hierarchical gain intervals, and a
  hierarchical FA_u interval.
- Added calibration-size, batch-regime, and architecture sensitivity analysis.
- Added full candidate/controller timing code without double-counting the model
  copy already performed by the adapter.
- Added an iWildCam stream gate that requires separate held-out evidence and
  outcome files and seals live decisions before opening outcomes.
- Added regression tests for calibration, split isolation, artifact completeness,
  imported hashes, and live-decision sealing.

## Measurements required before submission

- Run clean CIFAR-10-C SAR seeds 0-4.
- Run ImageNet-C SAR seeds 1-4 and combine them with the hash-locked seed 0.
- Run PACS Tent/EATA/SAR seeds 0-2.
- Run ImageNet-R Protocol D seed 3 and combine it with seeds 0-2 for all ten
  backbones.
- Run the uniform outer-seed analysis, then the calibration-size, batch-size,
  and architecture ablations on compatible completed panels.
- Execute the end-to-end runtime profiler and retain its raw timing rows.
- Generate fresh iWildCam held-out evidence-only logs and separate offline
  outcome logs before making a streaming claim.

## Strengthening work

- Reproduce official POEM and AETTA implementations if they remain central baselines.
- Obtain external independent replication of the natural-shift summaries.
- Run the preregistered physical-camera sessions with fresh held-out days/objects and release raw logs.

## Submission risks that remain

- No clean single-dataset natural-shift CI-robust beats-both result.
- Controlled-grid wins depend on archived aggregates whose raw replay lineage must be documented carefully.
- ImageNet-C beats-both is currently a single-seed operating point.
- Full foundational probability mechanization remains incomplete.
- The physical study, measured full runtime profile, and fresh label-free
  sequential evaluation remain explicit pending work rather than placeholder results.

## Current execution blocker

The most recent automated preflight could not see `/Volumes/T9` and reported no
MPS or CUDA device inside the current execution environment. The queue therefore
did not start. This is the intended fail-closed behavior; rerun preflight from a
terminal session that sees both T9 and the accelerator.
