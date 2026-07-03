# kbound-tta

**K-Bound / KGA — a label-free certificate for test-time adaptation.**

Test-time adaptation (TTA) can recover accuracy under one distribution shift and
silently destroy it under another — and without labels you can't tell which is
happening. `pykbound` reframes TTA as a *decision*: given a frozen model and a
candidate adapted model on an **unlabeled** batch, it decides **adapt / freeze /
abstain** with finite-sample control on the false-adapt and false-freeze rates
(the Knowability-Guided Adaptation gate).

## Install

```bash
pip install kbound-tta
```

## Quickstart

```python
import kbound_tta as kb

# 1) adapt a frozen model with a label-free TTA candidate
adapted, update_norm = kb.tent_adapt(f0, stream, steps=1, lr=1e-3)
#   candidates: tent_adapt, eata_adapt, sar_adapt, shot_adapt  (kb.TTA_METHODS)

# 2) extract label-free evidence
Z = kb.evidence_vector(f0, adapted, batch, num_classes, update_norm)

# 3) let the gate decide (trained leave-one-task-out on (Z, true benefit))
decision = kb.decide_kga(Z_train, B_train)   # -> 'adapt' / 'freeze' / 'abstain'
```

## What's inside

- **Candidates** — `tent_adapt`, `eata_adapt`, `sar_adapt` (faithful SAM+recovery),
  `shot_adapt` (SHOT-IM); all adapt BN/LN-affine params under a shared budget.
- **Evidence** — `evidence_vector`, `rich_evidence_vector` (ATC, energy, BN-KL drift).
- **Gate & routing** — `decide_kga` (single-candidate certificate),
  `multicandidate_route`, `smooth_drift_route`, plus `policy_metrics`,
  `detectability_analysis`.

## Citation

If you use this, please cite the K-Bound paper:
*"When Is Label-Free Adaptation Knowable? Helpful, Harmful, and Unknowable
Regimes Under Distribution Shift"* (Niroula).
Code & paper: <https://github.com/Pratikn03/AutoML_Flagship_V8> (`docs/research/kbound`).

## License

MIT — see [LICENSE](LICENSE).
