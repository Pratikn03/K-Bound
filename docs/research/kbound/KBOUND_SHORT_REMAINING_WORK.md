# K-Bound Short Paper Remaining Work

The maintained software, compact claim evidence, dashboard, paper build, and
Lean strict-core verification are release-complete. The items below are new
experiments or external validation; they are not silently represented as
finished implementation.

## Required before submission

- Rebuild CIFAR-10-C SAR from a clean immutable five-seed tree or keep it withheld.
- Add calibration-size, batch-size, and architecture ablations. The seed-0
  alpha/evidence/estimator/adapter study is now source-backed and rerunnable.
- Produce a full end-to-end component runtime profile with raw logs. The current
  controller-only microbenchmark does not measure candidate adaptation or total latency.
- Replace the label-informed iWildCam streaming diagnostic with a genuinely
  label-free sequential evaluation before making any streaming deployment claim.

## Strengthening work

- Add multiple ImageNet-C seeds.
- Complete PACS seeds 2 and 3 and ImageNet-R seed 3.
- Reproduce official POEM and AETTA implementations if they remain central baselines.
- Obtain external independent replication of the natural-shift summaries.
- Run the preregistered physical-camera sessions with fresh held-out days/objects and release raw logs.

## Submission risks that remain

- No clean single-dataset natural-shift CI-robust beats-both result.
- Controlled-grid wins depend on archived aggregates whose raw replay lineage must be documented carefully.
- ImageNet-C beats-both is currently a single-seed operating point.
- Full foundational probability mechanization remains incomplete.
- The physical study, full runtime profile, and genuinely label-free sequential evaluation remain explicit pending work rather than placeholder results.
