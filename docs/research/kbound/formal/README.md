# K-Bound Lean 4 formalization

The maintained strict audit checks **238 registered declarations** (70 legacy
finite/algebraic/deterministic checks and 168 measurable-foundation,
counterexample, and conditional-extension checks). The count is a registry
count, not a count of paper claims or independent scientific results.
The frozen 2026-09-23 release retains its original 150-declaration receipt. This
separate theorem-assumption revision adds scoped probability, finite-audit and
transport capstones; it does not retroactively change that frozen receipt.

Use the existing pinned dependency checkout. Do not run `lake update` for a
release verification. Lean and Mathlib are pinned to v4.29.1 by `lean-toolchain`,
`lakefile.lean`, and `lake-manifest.json`.

```bash
cd docs/research/kbound/formal
python3 formal_audit.py --build --strict-core --json-out NEW_STRICT_RECEIPT.json
python3 formal_audit.py --build --strict-core --full-foundations --json-out NEW_FULL_RECEIPT.json
```

The first command requires a successful `lake build KBound`, checks registered
names, rejects forbidden proof holes and compiler proof-hole warnings, and
inspects kernel dependencies against `Classical.choice`, `Quot.sound`, and
`propext`. A static scan alone is not kernel verification. The second command
is **expected to fail** because the unrestricted historical orbit-only one-bit
sufficiency assertion is refuted; the full historical H/ratio-rate extension
has not been established. Do not remove that blocker to make this gate green.

The random-radius revision audit is recorded in
`../theorem_assumption_revision/formal_random_radius_strict_receipt.json`.
Its scope includes the new module and all previous registered declarations.
The earlier journal audit is recorded in
`formal_strict_core_receipt_journal_2026_09_22.json` and
`formal_strict_core_journal_2026_09_22.log`;
the full-scope receipt and log use `formal_full_scope_receipt_journal_2026_09_22.json`
and `formal_full_scope_journal_2026_09_22.log`. Command scope, hashes, and
limitations are in `../journal_revision/theory_task_report.md` and
`../journal_revision/theory_source_snapshot_journal_2026_09_22.json`.
The older dated receipt is preserved unchanged. These receipts bind to the
recorded source snapshot, not to later manuscript edits or a later commit.

The completion-phase receipt is
`../theorem_assumption_completion/formal_completion_receipt.json`. It records
238 declarations, a 3,534-job Lean build, and the allowed kernel axiom set.
The completion directory also contains module-level receipts and source-bound
reports for randomized matched-evidence abstention, the arbitrary fibre
supremum, conditional episode coverage, raw iid random-count sampling, the
finite counterexample, paired transport and compact extrema, and fresh fixed
pair evaluation. Binomial confidence-box coverage, deployment exchangeability,
external residual/transport bounds, correct labels, and random-source integrity
remain explicit premises; a registry count cannot establish them.

## Paper correspondence and actual scope

| Paper claim | Mechanized layer | Boundary |
|---|---|---|
| `lem:reduction`, `lem:fibre` | `Probability/MeasureTarget.lean`: `measurable_label_kernel_freedom`, `measurable_target_benefit_reduction`, `measurable_correctness_identified_interval`, `measurable_target_frontier_attainment` | Genuine measurable joint laws; labels on disagreement supported on the two predictions. Binary classification satisfies this support condition. |
| `thm:frontier`, `lem:nonid`, `prop:closed-band` | `Probability/MeasureFrontier.lean`: `measurable_frontier_adapt_iff`, `measurable_frontier_freeze_iff`, `measurable_closed_band_zero_target`, `measurable_open_band_opposite_targets` | Full correctness-field class, measurable predictors/kernels, feasible margin, positive disagreement mass. No `RichAt` premise in these capstones. |
| `cor:matched-abstain` | `Impossibility.lean`: `matched_opposite_worlds_force_abstain` and action-probability arithmetic | Conditional on matched laws and both directional error bounds. |
| `thm:certificate` | `Certificate.lean`, `Probability/MeasureCertificate.lean`, `Probability/RandomRadiusCertificate.lean` | The random-radius module directly covers finite real benefit/estimate maps and radii in `[0,∞]`. Each directional error and their union are bounded by the same marginal coverage budget. Infinite radii and exact-zero interval endpoints abstain. Coverage remains an assumption; independence of radius and estimate is not required. |
| Split-conformal coverage | `Probability/MeasureConformal.lean`: `exchangeable_residual_coverage_ge`, `exchangeable_residual_either_error_le` | Measurable exchangeable residuals, including ties; calibration threshold and augmented rank construction. Earlier `ConformalExchangeability.lean` alone is only finite-rank algebra. |
| Population transfer | `Population.lean`, `Probability/Concentration.lean`: `paired_benefit_hoeffding_coverage` | Deterministic compound interval and bounded independent sampling layer. Benchmark sampling assumptions are external. |
| Anytime/Ville | `Probability/FilteredVille.lean`: `filtered_ville`, `filtered_optional_stopping_le`, `bounded_predictable_betting_anytime` | Filtered nonnegative supermartingales, bounded optional stopping, countable-time crossing, explicit conditional null/integrability. |
| Le Cam/KL/TV | `Probability/GeneralLeCam.lean` | Arbitrary probability measures, measurable randomized tests, TV identity, Bretagnolle–Huber and finite iid products; not only the old two-point implementation. |
| Concentration | `Probability/Concentration.lean` | Independent bounded Hoeffding and adapted martingale-difference results; not automatic concentration for correlated benchmark cells. |
| Historical one-bit/H extension | `Probability/ChannelCounterexample.lean`, `MeasureSwap.lean`, `HistoricalExtension.lean` | Orbit-only sufficiency refuted. Corrected fibre-consistent decoder and conditional H-budget propagation mechanized. |
| `thm:compact-beta-minimax`, `prop:nextphase-transport` | `Probability/AuditFloor.lean`, `Probability/PairedTransport.lean`, `Probability/PairedTransportCompactness.lean` | The fibre supremum and finite true-table/extrema layers are mechanized. Binomial confidence coverage and Python numerical evaluation remain separate premises and receipts. |

`KBound/TheoremMap.lean` and `formal_audit.py` identify the complete registry.
A kernel pass does not verify deployment-class restrictions, empirical
exchangeability, calibration transfer, target-law transport, preprocessing,
prospective provenance, numerical LP/inverse-CDF correctness, or scientific
novelty. The unrestricted historical extension remains excluded even when all
238 registered declarations pass.
