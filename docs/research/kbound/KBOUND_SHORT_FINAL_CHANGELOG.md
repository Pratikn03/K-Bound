# K-Bound Final Manuscript Changelog

Date: 2026-08-28

## Maintained Papers

- Compact submission source: `kbound_submission.tex` and `kbound_submission_body.tex`.
- Compact outputs: `kbound_short_final_draft.pdf` (28 pages) and
  `kbound_short_final_draft.docx` (28 rendered pages).
- Long companion source: `kbound_tmlr.tex` with the shared body and appendices.
- Long companion PDF: `kbound_tmlr.pdf` (37 pages).
- Older compatibility PDFs remain historical snapshots and are not refreshed or delivered.
- The legacy full-source IEEE render is diagnostic only and is no longer built by default.

## Mathematical Corrections

- Standardized benefit as `Delta = R_T(f_0) - R_T(f_a)` throughout.
- Replaced broad sign-identifiability language with the exact strict-commitment statement.
- Separated the interior opposite-sign construction from boundary zero-benefit ambiguity and the
  closed-band abstention conclusion.
- Defined pointwise candidate correctness as `P(f_a(X)=Y | X=x)` and retained the explicit
  `M + gamma` algebra.
- Corrected `beta = 0` to mean the strongest zero-drift transfer assumption, not a conservative
  default under unknown drift.
- Separated population quantities `(M, gamma, beta)` from empirical KGA quantities
  `(Delta_hat, epsilon)` in the abstract, method, figures, tables, and conclusion.
- Stated interval coverage as the direct premise of the marginal false-adapt implication. Risk
  alignment supports population identifiability and estimator transfer, not the implication after
  coverage is assumed.
- Retained the multiclass identity `Delta = P_T(D)(p_a-p_0)` without using a binary complement
  outside binary classification.

## Theory-to-KGA Link

- Added a direct proposition and displayed chain connecting the common target functional
  `Delta = 2 mu_T(D)(M+gamma)` to `Delta_hat = h_theta(Z)`.
- Explicitly separated uniform population quantifiers from marginal finite-sample coverage.
- Documented the external obligations: declared class, risk alignment on the operational class,
  calibration transfer, estimator support, and inference unit.
- Stated that KGA does not estimate or numerically receive `M`, `gamma`, or `beta` on the real-data
  panels.

## Empirical Corrections

- Rebuilt the canonical panel from 106 compact source artifacts with recorded SHA-256 hashes,
  including the superseded iWildCam H--v2 result used for explicit historical reconciliation.
- Kept candidate rows separate for CIFAR-10-C and ImageNet-C.
- CIFAR-10-C: Tent has positive ordinary intervals in a retrospective six-family sensitivity, but
  the preregistered six-comparison Holm gate fails; EATA is point-estimate beats-both with an
  adapt-side family interval containing zero; SAR is a completed negative arm.
- ImageNet-C: the authoritative configuration is 27 conditions per seed times five seeds. SAR has
  a pooled point edge without a promoted CI-robust claim; Tent ties freeze; EATA trails adapt.
- Natural tracks are scoped as no-harm, endpoint reproduction, point-only replication, or negative
  diagnostics. No clean natural single-dataset CI-robust beats-both claim is made.
- Locked canonical regeneration to `numpy==2.4.4` and `scikit-learn==1.8.0`; reconciliation now
  fails closed on runtime drift. Under that lock, Office-Home primary ties freeze with zero adapt
  exposure, while its separate replication retains only a tiny point edge.
- Removed stale CIFAR action counts, the old 66.8 percent aggregate, the 5x regret ratio, the old
  CIFAR-10.1 false-adapt story, and the historical constructed-mixture promotion.
- Replaced the stress-grid split-conformal claim with leave-one-condition-out cross-fitted empirical
  residual calibration. Jackknife+ is not claimed.

## Tables And Figures

- Regenerated `paper/generated/kbound_numbers.tex` and
  `canonical_panel_table.tex` from the same canonical JSON.
- Added the code-bound `current_policy_cluster_inference.json` artifact and generated
  `current_policy_family_sensitivity.tex` table, with post-hoc and preregistered Holm families kept
  distinct.
- Regenerated `fig_decision_value_frontier.png` directly from canonical kappa sweeps.
- Regenerated `fig_phase_diagram.png` as a conceptual, no-tick regime diagram rather than using
  pseudo-measured coordinates.
- Restored the population-only frontier figure and kept the empirical-only certificate figure.
- Updated natural repetition, action exposure, cluster inference, alpha, estimator, transfer, and
  claim-accounting tables.
- Removed the stale yield-ceiling and counterfactual-power block from the live manuscript. Its old
  sources remain historical provenance only.

## Appendix Scope

- The compact paper contains the core proofs, calibration/evaluation contracts, action exposure,
  inference discipline, provenance, implementation contract, evidence inventory, prospective
  natural validation protocol, formalization scope, and claim ledger.
- Extended budget, episode, minimax, historical reconciliation, and superseded diagnostics remain
  outside both maintained drivers rather than being presented as current submission evidence.
- Blank physical-camera templates are not presented as results.

## Verification

- Compact and full LaTeX builds: zero fatal errors, undefined citations, undefined references,
  duplicate labels, missing figures, or overfull boxes.
- Repository collection gate: 513 tests collect under Python 3.12. The maintained K-Bound,
  reconciliation, closure, edge, and reproducibility suites pass with declared environment-dependent
  skips and no failures.
- Manuscript claim gate: PASS.
- Canonical reconciliation tests: PASS.
- Lean/Mathlib: 2,554 jobs completed; formal audit and proof-hole scan PASS; 65 declarations mapped.
- Visual inspection: all 28 compact PDF pages, all 37 long-companion pages, and all 28 rendered
  DOCX pages pass full-page inspection with no clipping, overlap, broken tables, blank figures, or
  unreadable glyphs.

## Claims Retained And Weakened

- Retained: exact conditional population frontier, coverage-based marginal certificate, deployable
  shadow-state KGA wrapper, controlled mixed-regime Tent result, and negative diagnostics.
- Weakened: EATA to point-only at the corruption-family unit; ImageNet-C SAR to point-only;
  natural rows to descriptive no-harm or negative evidence; POEM/AETTA to protocol-matched ports.
- Pending: fresh held-out natural routing confirmation, physical-camera validation, official
  neighboring-method implementations, and a complete PACS per-cell gate replay.

## Closure Engineering

- Added fail-closed audits for unopened natural targets and official AETTA/POEM provenance.
- Added a deterministic, disjoint exact split-conformal confirmation manifest and leakage-checked
  decision/evaluation pipeline; the manifest remains deliberately unsealed and unexecuted.
- Added a controlled population-frontier/KGA API bridge without treating it as a benchmark or a
  real-data estimate of `beta`.
- Added Office-Home/iWildCam independent-checkpoint provenance checks and separated model seeds from
  test-stream seeds in code, manifests, and prose.
- Added PACS v2 per-cell serialization and a one-domain MPS smoke replay; the smoke audit is explicitly
  not the full five-seed panel.
- Unified canonical reconciliation, structured-manifest synchronization, compatibility-view
  generation, figures, tests, Lean, and both PDFs under `runbooks/release_candidate.sh`.
- Wrote `KBOUND_RELEASE_SHA256SUMS.txt` for the PDFs, canonical manifests, closure locks, and
  fail-closed audit artifacts. The file identifies the current working-tree release; a clean-checkout
  commit freeze remains separate pending work.
