# K-Bound Short Final Draft Changelog

Date: 2026-08-11

## Submission driver

- Authoritative driver: `kbound_tmlr.tex`.
- Shared scientific sources: `kbound_abstract.tex`, `kbound_short_body.tex`, and
  `kbound_short_appendix.tex`.
- Compatibility driver: `kbound_short.tex` (legacy IEEE two-column build).
- Delivered PDF: `kbound_short_final_draft.pdf`.

## Result reconciliation

- Added `scripts/reconcile_result_panels.py` and imported 72 compact, source-hashed records into
  `experiments/kbound/results/reconciled_panels_v1/`.
- Added `scripts/sync_reconciled_panels.py` so the canonical panel updates the generated result
  manifest, claim ledger, and decision-value artifact from one source.
- Replayed ImageNet-C and ImageNet-R with the declared exact-rank leave-one-condition-out rule.
- Replayed Office-Home and iWildCam transfer scoring under the locked repository runtime:
  Python 3.14.3, NumPy 2.4.4, and scikit-learn 1.8.0.
- Cross-validated the PACS three-seed aggregate. PACS per-cell gate replay remains unavailable
  because the archived files omit `b_hat` and calibration residual records.

## Corrected empirical statements

- Office-Home primary: KGA/adapt/freeze regret `0.0158/0.0468/0.0158`, with 0 ADAPT, 11 FREEZE,
  and 24 ABSTAIN decisions. The result ties always-freeze and is descriptive no-harm only.
- iWildCam: `0.0041/0.1028/0.0041`, with 0 ADAPT, 21 FREEZE, and 51 ABSTAIN decisions. The result
  ties always-freeze and is descriptive no-harm only.
- ImageNet-C SAR: `0.0289/0.0529/0.0319`, 13 ADAPT, 15 FREEZE, 107 ABSTAIN, and one false adapt
  over 135 cells. This is a pooled point beats-both result, not a seed-robust or CI-robust win.
- ImageNet-R: `0.0150/0.0064/0.0325`, 165 ADAPT, 29 FREEZE, 286 ABSTAIN, and zero false adapts
  over 480 cells. KGA is worse than always-adapt on 8 of 10 backbones.
- PACS remains a diagnostic null: `0.0431/0.0176/0.0446` on the pooled domain-seed mean.
- The historical three-source mixed-routing win was demoted pending replay because its component
  Office-Home and iWildCam decisions changed.

## Tables and figures

- Regenerated `paper/generated/kbound_numbers.tex` and the compatible result source.
- Regenerated the decision-value frontier and yield-ceiling figures from the canonical manifest.
- Updated the uniform nine-track panel, decision accounting, ImageNet-C, ImageNet-R, Office-Home,
  iWildCam, PACS, kappa-sweep, and claim-accounting text.
- Split the unbreakable 35-row guarantee table into three continued blocks. This removed the
  previous 700-point vertical overflow.
- Replaced unbreakable artifact identifiers with breakable paths and adjusted table widths.

## Theory and claim scope

- Retained the population/empirical distinction: the population frontier uses `M`, `gamma`, and
  `beta`; real-data KGA uses `Delta_hat +/- epsilon` and does not numerically receive `beta`.
- Preserved the strict-commitment wording and the distinction between marginal `FA_u` and
  descriptive conditional `FA_c`.
- Weakened natural-shift claims to descriptive no-harm where the gate ties always-freeze.
- Kept ImageNet-C SAR as point-estimate evidence only and ImageNet-R/PACS as negative results.

## Verification

- `tests/test_reconciled_panels.py`: 3 passed.
- Combined K-Bound package, canonical-rule, certificate-drift, routing, and reconciled-panel set:
  109 passed.
- Three unrelated ELARA checks still fail because two legacy ELARA files are absent; the existing
  K-Bound closure report already classifies those checks as outside K-Bound scope.
- Authoritative and legacy LaTeX drivers compile successfully.
- Final authoritative pass: no undefined references, undefined citations, duplicate labels,
  missing figures, fatal errors, or overfull boxes.
- Visual inspection covered all 91 rendered pages, including the title, main result tables,
  regenerated figures, appendix claim table, and references.

## Remaining draft items

- Rerun the corrected ImageNet-R per-commitment permutation diagnostic.
- Rerun the corrected ImageNet-C Tent counterfactual power probe.
- Rebuild the constructed heterogeneous mixture from reconciled component records.
- Complete the preregistered real-camera study before claiming physical validation.
- Establish or replace the undeclared A7 stability premise for transfer coverage claims.
