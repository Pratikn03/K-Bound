# Deep-Grid Audit / gamma-Meter Re-Analysis (Task E)

CPU-only - repo venv (py3.14.3, numpy2.4.4). Numbers in `deepgrid_audit_results.json`.
Rule: Thm T-II(a) bit-robust budget audit (beta=0.05, alpha=0.05, bootstrap radius) +
Cor 0.3 tau rank-1 falsifier. Three zones FALSIFIED / CERTIFIED / BLIND, scored vs the
LOGGED ground truth (per-candidate accuracy `aa_all`, frozen `a0`). Labels score only;
the label-free estimator (`route.b_hat`, `route.tau`) is never refit. Failures reported as-is.

## Which audit inputs are serialized (per grid)
- **ImageNet-R light** (`imagenetr_kbound_light_mps_internal/_partial.json`, 33/48 cells logged):
  per-condition `route.b_hat` (product-ratio advantage vector), `route.tau`, `tau_star`,
  `margin`, `n_D`, plus ground-truth `aa_all`/`a0`. FULL audit possible.
- **ImageNet-R 1pct** (`result_604f04ba.json`, 6 cells): same schema. FULL audit.
- **CIFAR-10-C 65-cell CSV**: per-cell ACCURACY only (frozen/tent/eata/sar/kga/oracle).
  No pairwise agreements c_ij / b_hat / tau -> only a REDUCED realized-benefit-sign accounting.

## Results (auditable grids)
| Grid | N | FALSIFIED | CERTIFIED | BLIND | false-cert | true-harm safe |
|---|---|---|---|---|---|---|
| ImageNet-R light | 33 | 33 | 0 | 0 | 0 | n/a (0 harm cells) |
| ImageNet-R 1pct  |  6 |  6 | 0 | 0 | 0 | 1/1 (100%) |
| CIFAR-10-C CSV (reduced) | 65 | 0 | 64 | 1 | 0 | 1/1 (100%) |

- **Zero false-certifications across all 104 auditable conditions.** Every true-harm
  condition (2 total) landed in a safe zone (FALSIFIED/BLIND). Soundness of T-II(a) holds.
- **ImageNet-R: every cell is FALSIFIED.** The tau rank-1 falsifier fires on 33/33 (resp.
  6/6) cells (tau in 0.98-2.15 >> tau*=0.52), and the budget audit independently rejects
  29/33 (resp. 5/6). H (CEI + per-class symmetry) is universally violated by correlated
  TTA candidates on the 200-class label space - exactly the rank-2 departure Cor 0.3 predicts.
  The certificate therefore correctly ABSTAINS everywhere rather than ever certifying a sign.
- **Sign channel quality (informative even under abstain):** recovered sign(b_hat_a-b_hat_0)
  matches the true sign on 27/33 = 82% (light) and 5/6 = 83% (1pct) of cells - consistent with
  the 78-85% from the earlier anomaly-bank P1 run.
- **CIFAR-10-C CSV** has no label-free inputs, so its "CERTIFIED" column is just the realized
  oracle benefit sign (adapt beats frozen on 64/65 cells, 1 tie); it cannot exercise the
  gamma/tau audit and is reported as REDUCED, not a true certificate test.

## NOT SERIALIZED (audit not possible)
- `decisive_tta_results.json` (CIFAR-10-C, **432** conditions): aggregate metrics only
  (decision_counts, coverage, base_rate_harmful, mean_true_B, pareto); the `conditions`
  field is a list of condition-NAME strings. Per-condition B / tau / b_hat / agreements absent.
- `imagenetc_1pct/decisive_tta_results.json` (ImageNet-C, 36): aggregate only - same gap.
- `imagenetc_noise/decisive_tta_results.json` (ImageNet-C, 36): aggregate only - same gap.
- `cifar101/decisive_tta_results.json` (CIFAR-10.1, 36): aggregate only - same gap.
- Per-sample prediction arrays / pairwise agreements c_ij: not serialized in ANY grid
  (only the derived `route.b_hat`/`tau` on ImageNet-R), so the worst-case Hoeffding radius
  could not be recomputed; the bootstrap radius uses the logged `n_D`.
