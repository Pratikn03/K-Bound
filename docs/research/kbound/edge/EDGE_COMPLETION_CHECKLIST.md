# Physical Study Completion Checklist

Use [PHYSICAL_STUDY_RUNBOOK.md](PHYSICAL_STUDY_RUNBOOK.md) as the operational
source of truth.

## Before Data

- [ ] Protocol source and SHA-256 lock saved.
- [ ] Ten physical objects P01-P10 assigned once.
- [ ] Phone A and Phone B identities recorded.
- [ ] Python environment passes edge tests.
- [ ] Browser/OpenCV preview confirms framing.
- [ ] No pilot or mock file is inside an S01-S10 directory.

## Development

- [ ] S01 contains 120 physical clips.
- [ ] S02 contains 40 physical clips.
- [ ] Source model passes S02 balanced accuracy >= 0.80.
- [ ] Source model passes S02 macro-F1 >= 0.80.
- [ ] Training command does not use --bypass-gate.
- [ ] S03-S06 contain 224 physical clips in total.
- [ ] Calibration-conformal split is sealed before any S07-S10 capture.

## Held-Out and Replication

- [ ] S07-S08 contain 112 Phone A physical clips.
- [ ] S09-S10 contain 112 Phone B physical clips.
- [ ] P09-P10 were never used in development.
- [ ] No tuning occurred after held-out access.

## Publication

- [ ] Full pipeline exits successfully.
- [ ] All eight anti-leakage checks pass.
- [ ] publication_gate.json reports passed: true.
- [ ] Phone A and Phone B metrics are reported separately.
- [ ] FA_u and FA_c are labeled separately.
- [ ] Runtime includes candidate adaptation and end-to-end latency.
- [ ] Raw clips pass privacy review before any release.
- [ ] Paper language matches the observed result, including a negative result.

The publication gate validates integrity, not scientific success. A no-harm,
abstention-heavy, or negative result can be admissible.

