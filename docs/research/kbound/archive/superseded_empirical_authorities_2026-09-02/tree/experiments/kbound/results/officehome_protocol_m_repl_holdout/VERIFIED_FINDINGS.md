# OfficeHome Protocol M Replication: VERIFIED

Date: 2026-06-18

Fresh GPU replication of the dev-selected OfficeHome Protocol M configuration.

Locked configuration:
- dataset: OfficeHome
- candidate: `sar_online_aggressive`
- estimator: `gbr`
- conformal: `global`
- alpha: `0.10`
- calibration: fresh target-val records, seeds `2,3,4`
- held-out evaluation: fresh target-test records, seeds `2,3,4`

Held-out target-test result:
- KGA regret: `0.0004273504`
- always-adapt regret: `0.0457977208`
- always-freeze regret: `0.0217236467`
- coverage: `0.9814814815`
- adapt rate: `0.6296296296`
- false-adapt: `0.0294117647`
- beats both: `true`
- false-adapt <= alpha: `true`
- verdict win: `true`
- margin over best fixed baseline: `0.0212962963`

Artifacts:
- calibration records: `experiments/kbound/results/officehome_protocol_m_repl_targetval/result_target_val_eb504dd6.json`
- held-out records: `experiments/kbound/results/officehome_protocol_m_repl_targettest/result_target_test_f761540b.json`
- score JSON: `experiments/kbound/results/officehome_protocol_m_repl_holdout/holdout_score.json`
- run log: `experiments/kbound/results/officehome_protocol_m_repl.log`

Interpretation:
This is an independent replication of the OfficeHome beats-both result on new seeds.
It supports promoting Protocol M from a lead to a paper-relevant real-data win,
subject to updating the manuscript text and tables carefully.
