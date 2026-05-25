# Polarity Diagnostic Report (Phase 1.F)

**Status:** locked.
**Policy:** **no polarity flipping in primary metrics for any cell (Family A / B / C / D).**
**Diagnostic artifact:** [`experiments/audit/polarity_diagnostic_log.csv`](../../../experiments/audit/polarity_diagnostic_log.csv).

---

## 1. Old behaviour (before Phase 1.F)

Prior to Phase 1.F, the runner applied a synthetic-anomaly validation probe at evaluation time and, when the probe AUROC fell below 0.5, flipped the test predictions of `static_attention` and `craf_attention` (`run_breakthrough_experiment.py:2078-2082`). The flip was:

- Applied only per seed (some seeds flipped, others did not).
- Applied only to static + RGA (not to `rga_boosted_fusion`, `rga_meta_router`, Tent, TTT, EATA, SAR, RF, MLP, LFE, Conf-mean).
- Described in the paper as a "deployment-grade sanity check".

## 2. Why this was invalid for primary comparisons

1. **Asymmetric across methods.** RGA and static were flipped; the same-fold-trained score-adapter baselines (Tent, TTT, EATA, SAR) inherit the static model's logits and therefore share its orientation, but the runner did not flip them. Reported RGA-vs-Tent comparisons therefore compare a flipped RGA against an unflipped Tent — incoherent.
2. **Noise-dominated at the threshold.** On canonical one-class MVTec cells the probe AUROC sat in `[0.45, 0.55]` for the majority of seeds. The flip decision became coin-flip-noise; cross-seed means mixed flipped and unflipped predictions.
3. **Mis-described as "deployment-grade".** The probe is a validation-time diagnostic; the term "deployment-grade" overstates what it does.

## 3. New behaviour (Phase 1.F)

The Phase 1.F patch removes the four `1.0 - *_probs` lines from `run_breakthrough_experiment.py:2078-2082` and replaces them with a multi-line comment explaining the lock:

```
# Phase 1.F lock (Locked Audited-Reanalysis Policy):
# the polarity probe is a validation-only score-orientation
# diagnostic; it MUST NOT alter primary predictions. The flip
# decision is logged via polarity_info but the prediction
# arrays remain unchanged. Any orientation-corrected variant
# must be evaluated as a separately named method.
```

The probe call (`_calibrate_polarity`) and the JSON logging are retained. Future runs of the runner will:
- continue to log `polarity_calibration.flip_required` and `polarity_calibration.calibration_auroc` per seed;
- **not** modify `static_*_probs` or `craf_*_probs`.

The audited reanalysis of existing JSONs uses the unflipped per-seed predictions as the primary metric source. The `experiments/audit/polarity_diagnostic_log.csv` artifact records the historical flip decisions (so a reviewer can see which seeds would have been flipped under the old logic) but `primary_metrics_use_flip` is `False` for every row.

## 4. Number of historically flipped seeds by benchmark

From the Phase 1.A audit:

| Benchmark / protocol | n_seeds | n_seeds_with_flip_required | n_borderline_seeds (probe AUROC in [0.45, 0.55]) |
|---|---|---|---|
| MVTec 3D-AD PatchCore canonical | 5 | 3 | 4 (probes ranged 0.49–0.53) |
| MVTec LOCO-AD PatchCore canonical | 5 | (similar borderline pattern) | (similar) |
| VisA RGB+edge canonical | 5 | (similar borderline pattern) | (similar) |
| MVTec 3D-AD PatchCore supervised-paired | 30 | 0 | 0 (probes 0.56–0.59, comfortably above 0.5) |
| Supervised-paired cells generally | — | 0 | 0 |

The flip historically affected canonical cells only. Removing it from the primary path has zero numerical effect on supervised-paired Family A confirmatory cells (A2, A3, A5, A7, A8), and shifts the canonical-cell ROC-AUCs slightly without affecting the canonical cells' classification as protocol-diagnostic (the canonical PR-AUC / ECE / Brier are blocked regardless of flip per Phase 1.A).

## 5. Does removal change any primary audited metric?

- **Family A confirmatory cells (A2, A3, A5, A7, A8):** no change. No seeds were historically flipped in supervised-paired cells.
- **Family A protocol-diagnostic cells (A1, A4, A6):** the static / RGA ROC-AUC numbers shift slightly, but those cells are not subject to superiority claims under Phase 0.6's policy lock. Canonical PR-AUC / ECE / Brier remain blocked.
- **Family B / C cells:** no change. The flip never affected them.

The audited inference results (Phase 1.D) and the master comparison table (Phase 1.G) are therefore unchanged by Phase 1.F.

## 6. Manuscript prose updates required (Phase 1.G)

The Phase 1.G manuscript update must:
1. Replace any "deployment-grade sanity check" phrase with "validation-only score-orientation diagnostic".
2. Add a short paragraph (or footnote) stating that the polarity probe is logged for diagnostic purposes only and does not alter primary metrics.
3. Remove the previous flip-rule paragraph from §`sec:cross-benchmark-master` polarity-calibration footnote (the rule no longer applies).

## 7. Tests covering the lock

- `tests/test_primary_metrics_do_not_apply_polarity_flip.py` — asserts the runner source contains the Phase 1.F lock comment and no longer applies the flip in the primary path.
- `tests/test_polarity_probe_diagnostic_only.py` — asserts the diagnostic CSV exists and every row has `primary_metrics_use_flip = False`.
- `tests/test_canonical_label_semantics.py::test_polarity_log_primary_metrics_do_not_use_flip` — companion assertion in the canonical-cell test.
