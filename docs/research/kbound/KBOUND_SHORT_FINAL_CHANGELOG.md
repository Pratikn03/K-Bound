# K-Bound Short Paper Final-Draft Changelog

## Authoritative source

- Main entrypoint: `kbound_short.tex`.
- Baseline: the source corresponding to `kbound_short_draft(2)(1).pdf`, snapshotted before editing for the retained source diff.
- The older `kbound_short(17).pdf` was not used to restore claims.

## Theory statements corrected

- Standardized benefit as `Delta = R_T(f_0) - R_T(f_a)`.
- Corrected pointwise candidate correctness to `eta_a(x)=P_T(f_a(X)=Y | X=x)`.
- Expanded the disagreement algebra explicitly through `M + gamma = E[eta_a | D] - 1/2`.
- Split interior opposite-sign impossibility from boundary zero-versus-strict ambiguity.
- Replaced broad sign-identification headlines with the exact strict-commitment frontier.
- Corrected the `beta=0` interpretation and separated `beta` from `epsilon` and `M` from `Delta_hat`.
- Restated Theorem 3 as coverage-implies-marginal-error control; exchangeability supports coverage and risk alignment is not a direct premise once coverage is assumed.
- Added the multiclass bridge and retained the target-class richness caveat for converse claims.

## Empirical corrections

- Added `paper/generated/kbound_result_manifest.json` as the canonical promoted-number manifest.
- Corrected iWildCam to an exact tie with always-freeze under the OOF lock.
- Corrected CIFAR-10.1 from conditional false-adapt `0.444` mislabeled as unconditional to `FA_u=0.167`, `FA_c=0.444`.
- Restricted CIFAR-10-C promotion to archived Tent/EATA results; SAR is withheld after the raw seed-0 replay mismatch.
- Standardized ImageNet-C to the authoritative 27-cell, seed-0 SAR result and removed the unsupported Holm claim for that isolated table.
- Marked PACS as 1/3 planned seeds and ImageNet-R as 3/4 planned seeds.
- Kept Office-Home, iWildCam, Camelyon17, and RxRx1 as no-harm/safety results rather than natural beats-both claims.
- Kept the three-source OOF result as a researcher-constructed routing aggregate, not transfer.

## Calibration and code

- The maintained implementation uses the exact clean-split rank `k=min(n,ceil((n+1)(1-alpha)))`.
- Stress-grid artifacts remain labeled leave-one-condition-out cross-fitted empirical residual calibration, not exact split conformal or jackknife+.
- Archived benchmark JSONs are explicitly disclosed as using the earlier interpolated empirical quantile.
- Recomputed the seed-0 alpha/evidence/estimator/adapter-transfer ablations from
  three hashed 432-cell inputs. The artifact records software versions and minor
  drift from the archived reference; stress-grid validity remains empirical.

## Figures regenerated

- `figures/fig_frontier_schematic.png`: population-only strict-commitment frontier using `M` and `beta`.
- `figures/fig_phase_diagram.png`: conceptual regime geometry with no measured-looking coordinates.
- `figures/fig_natural_forest.png`: corrected OOF Office-Home/iWildCam no-harm intervals.

## Tables regenerated or reconciled

- Generated headline macros in `paper/generated/kbound_numbers.tex` from the canonical manifest.
- Rebuilt the nine-track empirical panel, primary numeric table, assumptions table, and five-column claim-to-support map.
- Removed SAR macros from the promoted CIFAR table path.

## Appendix and claim cleanup

- Retained core proofs, boundary case, multiclass/regression derivations, evidence/configuration details, protocol reconciliation, runtime status, formalization inventory, and claim-to-artifact map.
- Deferred one-bit, minimax, extensive rate, and weakest-class material from the short build.
- Removed disabled legacy blocks containing withdrawn universal-gate, seven-source, and superseded ablation claims from the public source.
- Removed the stale verdict-migration figure and unsupported runtime tables;
  retained only sensitivity tables backed by the fresh exact-rank recomputation.
- Weakened formalization and prior-work language; no priority claim remains.
- Pruned bibliography entries not cited by the compiled short paper.

## Pending evidence stated in prose

- The historical streaming script is disclosed as label-informed and excluded from label-free evidence.
- A full component-level runtime profile plus calibration-size, batch-size, and
  architecture ablations remain pending.
- Fresh held-out real-camera evidence remains pending; blank templates are not treated as results.

## Research tooling

- Rebuilt the TypeScript dashboard from the canonical result manifest and removed legacy ELARA as a data source.
- Added a fail-closed physical-study publication gate covering provenance, exact session inventory, source-model quality, calibration sealing, held-out chronology, replication, and leakage checks.
- Synchronized the Word editing draft, resolved raw Pandoc cross-reference
  tokens from the final LaTeX map, and verified its 38-page rendering.
- Restricted wheel packaging to the public `kga` package; historical integration
  files are absent from the verified wheel.

## Build outcome

- Final PDF: 21 pages, letter size.
- Synchronized Word rendering: 38 single-column pages.
- Fatal errors: 0.
- Undefined citations: 0.
- Undefined references: 0.
- Missing figures: 0.
- Duplicate labels: 0 detected.
- Remaining overfull messages are from pre-scaled table construction or boxes
  under 6.4 points; raster review of the affected pages shows no clipping.

## Verification

- Focused Python suite: 223 passed and 2 intentional skips across 225 collected package, edge, and dashboard tests.
- `analyze_F.py --self-test`: passed for global, Mondrian, and CQR paths.
- Last recorded Lean/Mathlib build: 2,544 jobs; formal audit passed with 43 indexed
  theorem checks and a clean proof-hole scan. The clean release CI reruns this gate.
