# TTA collapse probe (2-D logistic)

- **Generating script:** `src/scripts/kbound/tta_collapse_experiment.py`
- **Result file(s):** `../../results/tta/tta_collapse_results.json`
- **Claim status:** **verify_before_claim**
- **Decision rule:** KGA leave-one-out gradient-boosted benefit estimator $\hat B(Z)$ + split-conformal radius $\varepsilon$ ($\alpha=0.10$); ADAPT if $\hat B-\varepsilon>0$, FREEZE if $\hat B+\varepsilon<0$, else ABSTAIN.

Metrics live in the result JSON above. Per-task `decisions.csv` / `oracle.csv` and
per-seed folders require re-running the script with row-level dumping enabled (see
`../../scripts/02_verify_results.py`). This folder currently holds the run config and
this card; the canonical numbers are in `results/`.
