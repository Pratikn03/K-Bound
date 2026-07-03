# Final consolidated manifest (retest)

Mean +/- std across all seeds/records per dataset (all reported, never the best).

> **`beats-both (pt-est)` is a point-estimate count, NOT the verdict.** A hairline gap (e.g. 0.0157 vs 0.0158) is a **tie / no-harm**, not a win. The CI-robust win/no-harm verdict for the natural shifts comes only from the condition-bootstrap (`bootstrap_win_cis.py`, `verify_realshift_win.py`).

| Dataset | n | regret KGA | regret adapt | regret freeze | beats-both (pt-est) |
|---|---|---|---|---|---|
| cifar101 | 13 | 0.1189+/-0.2745 | 0.1327+/-0.2539 | 0.1189+/-0.2745 | 0/13 |
| cifar10c | 3 | 0.5234+/-0.3693 | 0.5231+/-0.3697 | 0.5234+/-0.3431 | 0/3 |
| imagenetc | 13 | 0.0468+/-0.1253 | 0.0703+/-0.1053 | 0.0965+/-0.0908 | 9/13 |
| imagenetr | 30 | 0.3667+/-0.2503 | 0.3900+/-0.2713 | 0.3650+/-0.2445 | 0/30 |
| iwildcam | 7 | 0.0201+/-0.0079 | 0.0979+/-0.0044 | 0.0201+/-0.0079 | 0/7 |
| officehome | 7 | 0.0156+/-0.0049 | 0.0168+/-0.0235 | 0.0213+/-0.0066 | 1/7 |
| pacs | 1 | 0.1151+/-0.0000 | 0.0026+/-0.0000 | 0.1151+/-0.0000 | 0/1 |
