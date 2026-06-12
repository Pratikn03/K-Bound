# Provenance — decision_baselines/

`decision_baselines.json` and `decision_baselines_table.md` were generated on
2026-06-10 (~04:00) by `docs/research/kbound/scripts/run_decision_baselines.py`
from `experiments/kbound/results/imagenetc_noise/checkpoint.json` (36
cells/method, per-condition rows with evidence Z, a0, aa).

**That source checkpoint was deleted later the same morning** (run-completion
cleanup), so these outputs can no longer be re-derived from it. They remain
valid analyses of the logged rows, and the built-in cross-check confirmed the
reproduced KGA regrets matched `decisive_tta_results.json` exactly
(`kga_crosscheck` block inside `decision_baselines.json`).

## Regeneration path

The reference-parity rerun `imagenetc_noise_sarfix/` regenerates the same
per-condition rows with the corrected SAR implementation. When it reaches
36/36 cells, refresh the analysis with:

```bash
cd /Volumes/T9/uav/AutoML_Flagship_V8
~/.venv_wilds/bin/python3 docs/research/kbound/scripts/run_decision_baselines.py \
  --checkpoint experiments/kbound/results/imagenetc_noise_sarfix/checkpoint.json \
  --kga-results experiments/kbound/results/imagenetc_noise_sarfix/decisive_tta_results.json \
  --out-dir experiments/kbound/results/decision_baselines_sarfix
```

**Immediately archive the checkpoint when the run completes** (before any
cleanup step can remove it):

```bash
cp experiments/kbound/results/imagenetc_noise_sarfix/checkpoint.json \
   experiments/kbound/results/decision_baselines_sarfix/source_checkpoint_archived.json
```

If the sarfix numbers differ materially from the current Table
(tab:decision-baselines in kbound.tex), update the table from
`decision_baselines_sarfix/decision_baselines_table.md` and note the SAR
implementation change in the caption.
