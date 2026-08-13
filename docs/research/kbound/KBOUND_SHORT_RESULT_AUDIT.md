# K-Bound Short Paper Result Audit

Date: 2026-08-11

## Canonical source

Authoritative artifact:
`experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`

Provenance:

- 72 compact source files recorded in `source_manifest.json` with SHA-256 hashes.
- Generator hash and runtime versions recorded in the canonical JSON.
- Runtime: Python 3.14.3, NumPy 2.4.4, scikit-learn 1.8.0.
- The release stores sufficient records to replay ImageNet-C, ImageNet-R, Office-Home, and iWildCam.
- PACS aggregate arithmetic is cross-validated, but its gate cannot be replayed from the archived
  files because `b_hat` and calibration residual records are absent.

## Authoritative panel

| Track | n | KGA | Adapt | Freeze | AD/FZ/AB | FA_u | Defensible verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Office-Home primary | 35 | 0.0158 | 0.0468 | 0.0158 | 0/11/24 | 0 | Ties freeze; descriptive no-harm |
| iWildCam | 72 | 0.0041 | 0.1028 | 0.0041 | 0/21/51 | 0 | Ties freeze; descriptive no-harm |
| ImageNet-C SAR | 135 | 0.0289 | 0.0529 | 0.0319 | 13/15/107 | 0.0074 | Pooled point beats-both only |
| ImageNet-R | 480 | 0.0150 | 0.0064 | 0.0325 | 165/29/286 | 0 | Negative; worse than adapt on 8/10 backbones |
| PACS | 12 domain-seed units | 0.0431 | 0.0176 | 0.0446 | aggregate only | 0.0093 | Diagnostic null; replay incomplete |

Regrets are ordered KGA / always-adapt / always-freeze. AD/FZ/AB is adapt/freeze/abstain.

## Configuration controls

- ImageNet-C authority is the 27-cell x 5-seed panel, not older 36-cell configurations.
- ImageNet-C and ImageNet-R use exact-rank, leave-one-condition-out replay at the declared operating
  point.
- Office-Home and iWildCam use separate calibration/test transfer scoring. Their A7 full-fit versus
  leave-one-out stability premise was not predeclared, so theorem-level transfer coverage is not
  promoted.
- Always-adapt and always-freeze have decision coverage 1 by definition; the manuscript keeps
  decision coverage separate from adapt rate.
- Point beats-both, seed-robust beats-both, and CI-robust beats-both are not conflated.

## Corrected stale summaries

- Removed the old Office-Home one-adapt headline from the primary panel.
- Removed the old iWildCam freeze-count summary and replaced it with 21 freezes under the locked
  runtime replay.
- Replaced the old ImageNet-R ratio/backbone summary with 2.35x worse than adapt and 8/10 backbones.
- Demoted the historical constructed mixture until it is replayed from reconciled components.
- Marked superseded permutation and power-probe diagnostics as draft TODOs instead of reusing them.

## Remaining empirical gaps

- No clean held-out natural single-dataset CI-robust beats-both result is established.
- ImageNet-C SAR does not have seed-robust or CI-robust beats-both support.
- Natural transfer tracks need independent seeds and a declared stability/coverage design.
- PACS needs per-cell benefit predictions and calibration residuals for a full gate replay.
- Physical-camera tables remain preregistration templates, not result evidence.

