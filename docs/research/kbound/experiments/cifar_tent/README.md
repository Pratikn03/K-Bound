# CIFAR-10-C + Tent (deep TTA)

- **Generating script:** `src/scripts/kbound/cifar_tent_mps_v2.py (decisive) / cifar_tent_mps.py`
- **Result file(s):** `../../results/tta/cifar_tent_results.json (+online)`
- **Claim status:** **verify_before_claim**
- **Decision rule:** KGA leave-one-out gradient-boosted benefit estimator $\hat B(Z)$ + split-conformal radius $\varepsilon$ ($\alpha=0.10$); ADAPT if $\hat B-\varepsilon>0$, FREEZE if $\hat B+\varepsilon<0$, else ABSTAIN.

Metrics live in the result JSON above. Per-task `decisions.csv` / `oracle.csv` and
per-seed folders require re-running the script with row-level dumping enabled (see
`../../scripts/02_verify_results.py`). This folder currently holds the run config and
this card; the canonical numbers are in `results/`.
