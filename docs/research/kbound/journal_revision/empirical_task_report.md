# Empirical synthesis implementation report

Task 2 is implemented locally. No historical/frozen inputs, maintained manuscript source, protected target, original runner, model, or training data was changed. No commit or publication was performed by this worker.

## Delivered files

- `docs/research/kbound/scripts/build_empirical_synthesis.py`: self-contained builder with independently implemented metric calculations, pairing, exact input commitments, create-only output defaults, and byte-for-byte `--check` mode. It imports no historical scoring function and has no dependency on the earlier external scratch scripts.
- `tests/test_kbound_empirical_synthesis.py`: 17 tests for units, weighted loss versus regret/accuracy, equal-cell aggregation, zero exposure, nonpositive versus negative events, finite-cell versus simultaneous-group inclusion, fold/group identity, matched exposure, pairing mismatches, nonfinite/duplicate cells, secondary macro-F1 from counts, input tampering, and output preservation.
- `empirical_synthesis_v1.json`: all 160 calibration cohort/arm rows, nine principal CIFAR policy rows, three CCT policy rows, uncertainty, influence and metric contracts.
- `empirical_rows_v1.csv`: the 172 main machine-readable rows. The literal `null` denotes unavailable; the full JSON supplies reasons and nested results.
- `empirical_cct_secondary_cells_v1.json`: 45 archived CCT cells, each with verified macro-F1 for three policies. Class F1 is reconstructed from preserved TP/FP/FN counts before macro averaging. No aggregate macro-F1 or new inference claim is introduced.
- `empirical_metric_contracts_v1.json`: definitions, units, aggregation, denominators and unavailable-value semantics.
- `empirical_input_manifest_v1.json`: 70 consulted input files with SHA-256 and size, plus the two consulted CCT ZIP members, builder identity and runtime versions.
- `empirical_synthesis_receipt_v1.json`: output checksums and the explicitly bounded verification result.
- `empirical_synthesis_v1.md`: readable report.
- `empirical_tables.tex`: three generated supplement tables: cell versus simultaneous-group inclusion, principal CIFAR accuracy/regret/exposure, and Tent cluster/influence sensitivity.

## Verification completed

The initial 11 contract tests failed because the implementation was missing. Three subsequent tests failed for missing authority/F1/output helpers. The output-symlink preservation test subsequently exposed a real unchecked-write case; the writer now rejects symlinks before any mutation. All these tests pass after implementation.

Using the pinned research interpreter:

```sh
python -m pytest -q tests/test_kbound_empirical_synthesis.py tests/test_calibration_value.py
python -m ruff check docs/research/kbound/scripts/build_empirical_synthesis.py tests/test_kbound_empirical_synthesis.py
python -m ruff format --check docs/research/kbound/scripts/build_empirical_synthesis.py tests/test_kbound_empirical_synthesis.py
python docs/research/kbound/scripts/build_empirical_synthesis.py
python docs/research/kbound/scripts/build_empirical_synthesis.py --check
```

Results: **33 targeted tests pass** (17 new, 16 existing); Ruff lint and formatting pass; all eight generated outputs pass exact-byte reproduction. The final output regeneration used explicit `--replace` only for this worker's newly generated `empirical_` files after adding the LaTeX tables. Existing historical result files were never replacement targets.

The builder independently validates all 32 original calibration input commitments, 6,993 unique compact records and 7,101 scored fold-cell links. It covers 57 selected folds, 142,020 scored arm decisions, all 160 final arm rows, all 160 held-out group rows, 684 matched-exposure rows and 513 constant-radius selection equivalences. It performs 31,989 calibration scalar comparisons, 6,534 principal CIFAR comparisons and 230 CCT accuracy/regret/exposure comparisons, in addition to checking CCT secondary per-class F1 and macro means. A separate post-build pass confirmed that all 70 consulted input files still matched their recorded sizes and SHA-256 values.

The full repository suite, source seal, document builds, rendered-page inspection and publication remain root's Task 5 work; this report does not claim those later checks passed.

## Integration recommendations

Include `journal_revision/empirical_tables.tex` in the new supplement. Its labels are `tab:journal-inclusion-units`, `tab:journal-principal-means` and `tab:journal-tent-sensitivity`. Existing principal manuscript tables, including the shared-predictor Table 11, remain unchanged.

Use `empirical_synthesis_v1.json` as the analysis authority and the input/output manifests as its byte lineage. The public evidence ZIP must be extracted at the repository root before reproduction because frozen `output/next_phase/...` records are intentionally not recreated by the builder. The CCT reader uses the existing portable public evidence bundle; it never depends on the historical `/Volumes/T9` source paths and reads only the manifest and archived score object.

The principal CIFAR reconstructed accuracies are **recorded-cell means**, not pooled image accuracy and not a new authentication of original image-level execution. They recover EATA 80.0666%, SAR 80.9741% and Tent 79.3076% for KGA, with SAR below always-adapt. The separate shared-predictor study's weighted losses are not classifier accuracies. CCT accuracy has its declared set-membership top-1 contract; its 45 cells are five checkpoints crossed with nine locations, not 45 independent deployments.

For coverage, distinguish cell inclusion under group-max-calibrated intervals from simultaneous inclusion of all cells in a semantic group. The latter is 56.94%, 45.83% and 61.57% for EATA, SAR and Tent, respectively. Constant-cell simultaneous-group inclusion is 42.13%, 32.41% and 42.13%. All these are observed retrospective rates under nominal 90% intervals, not new coverage guarantees.

For Tent, report the all-six-cluster estimate as primary. Scaled-minus-point loss is -0.143843 percentage points with a descriptive bootstrap range [-0.428565, 0.138755]; versus the tuned margin it is -0.083495 with range [-0.416991, 0.188282]. At matched exposure the difference is only -0.018148. Removing JPEG is an exploratory influence calculation, not an edited result: it reduces the point comparison to -0.004889 and reverses the margin comparison to +0.090889. All cohort/design and arm rows remain in the machine-readable release, including the unfavorable ones.

CCT retains zero ADAPT exposure and misses its only helpful candidate. Its conditional acceptance error is null, not zero. The retention endpoint and stronger routing criterion remain separate. Archived CCT cell-level 16-indicator macro-F1 stays secondary and descriptive, while aggregate macro-F1 remains null with its explicit scope reason.

Bootstrap ranges and leave-one-cluster-out diagnostics remain descriptive sensitivity over related historical units. They do not establish population significance, new-environment calibration, independent deployment replication, general superiority or production safety.
