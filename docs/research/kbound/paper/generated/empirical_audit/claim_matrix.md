# Empirical claim matrix

| Claim | Dataset | Seeds | Authoritative artifact | Paper consumer | Status |
|---|---|---|---|---|---|
| Three-way action behavior | nine-track panel | track-specific | decision_metrics.json | regime/action tables | closed: rates and Wilson intervals reported |
| False-adapt uncertainty | nine-track panel | track-specific | decision_metrics.json | decision metrics table | closed where denominator exists |
| Coverage uncertainty | nine-track panel | track-specific | decision_metrics.json | coverage audit | closed where interval-hit records exist; historical natural logs marked not retained |
| CIFAR-SAR reconciliation | CIFAR-10-C | 0--4 | stress_grid_multiseed_v1/seed*/per_condition_cifar10c_sar_seed*.json | uniform panel | closed: locked analysis rebuilt; radius CV 0.390 disclosed |
| PACS planned seeds | PACS | 0 complete; 1--2 pending | experiments/kbound/results/pacs_seed*.json | PACS row | blocked: PyTorch import failure in Python 3.14 runtime |
| ImageNet-R planned seeds | ImageNet-R | 0--2 complete; 3 pending | imagenetr_protocol_d_multiseed_v1 | ImageNet-R row | pending seed 3 |
