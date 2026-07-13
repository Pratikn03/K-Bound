# K-Bound Short Paper Empirical Consistency Audit

## Canonical source

`paper/generated/kbound_result_manifest.json` is the authoritative promoted-number index. It records policy order, metric definitions, seed counts, source artifacts, quantile provenance, verdicts, and caveats.

## Promoted evidence

| Track | KGA / adapt / freeze regret | Evidence scope | Final claim |
|---|---:|---|---|
| CIFAR-10-C Tent | 0.0016 / 0.0079 / 0.1241 | archived 5 x 432 aggregate | CI beats-both; `FA_u=0` |
| CIFAR-10-C EATA | 0.0013 / 0.0033 / 0.1314 | archived 5 x 432 aggregate | CI beats-both; `FA_u=0` |
| ImageNet-C SAR | 0.0108 / 0.0625 / 0.0319 | 27 cells, seed 0 | paired-bootstrap beats-both; `FA_u=0` |
| Camelyon17 OOD | 0.0000 / 0.0000 / 0.1381 | genuine OOD test, n=18 | reconciled no-harm |
| iWildCam H v2 | 0.0041 / 0.1028 / 0.0041 | OOF lock, n=72 | exact tie with freeze; no-harm |
| Office-Home M v2 | 0.0157 / 0.0468 / 0.0158 | OOF lock, n=35 | no-harm; tiny point edge only |
| RxRx1 J | 0.0000 / 0.2531 / 0.0000 | locked test, n=60 | tie with freeze; no-harm |
| Three-source OOF | 0.0059 / 0.0632 / 0.0342 | constructed n=143 stream | CI beats-both; not transfer |

## Diagnostic or incomplete evidence

- CIFAR-10.1: `FA_u=0.167`, `FA_c=0.444`; transfer bar fails.
- ImageNet-R: 3/4 planned seeds; no stable CI-robust beats-both.
- PACS: 1/3 planned seeds; breadth diagnostic only.
- CIFAR-10-C SAR: withheld because current raw seed-0 does not replay the archived aggregate.

## Calibration provenance

- Current clean-split implementation: exact split-conformal rank.
- Archived benchmark artifacts: earlier interpolated empirical quantile.
- Stress grids: leave-one-condition-out cross-fitted empirical residual calibration; approximate nominal empirical coverage, not exact split-conformal validity.
- Sensitivity ablations: recomputed from three hashed seed-0, 432-cell files with
  NumPy 2.4.6 and scikit-learn 1.9.0. The fresh artifact records minor numerical
  drift from the archived reference (maximum scalar change 0.007) without changing
  the empirical safety conclusions.

## Baseline fidelity

- KGA is the paper's method.
- POEM-style and AETTA-style rows are protocol-matched ports, not official implementations.
- Higher observed false-adapt in these ports is described as consistent with the lack of an explicit marginal certificate, not caused solely by it.

## Consistency checks

- ImageNet-C 27-cell values are not mixed with the superseded 36-cell configuration.
- Natural-shift point wins are not promoted as CI-robust beats-both.
- Always-adapt/freeze are treated as fully decisive policies; adapt rate and decision coverage are not conflated.
- `FA_u` and `FA_c` are separately named.
- No blank camera table is used as evidence.
- The alpha, estimator, evidence-dropout, and adapter-transfer tables trace to
  `experiments/kbound/results/ablation_exactrank/ablation_exactrank.json`.
