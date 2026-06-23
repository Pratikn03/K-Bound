# K-Bound Cross-Dataset Win Loop

Two-stage report: dev/val-selected configs scored on separate held-out artifacts where available.

| dataset | paper-ready? | dev win? | heldout win? | candidate | est | conf | margin | KGA | adapt | freeze | FA | cov | n |
|---|:---:|:---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| officehome_protocol_m | yes | YES | YES | `sar_online_aggressive` | gbr | global | 0.0136264 | 0.0021978 | 0.0468132 | 0.0158242 | 0 | 0.971 | 35 |
| officehome_protocol_m | yes | YES | YES | `sar_online_aggressive` | gbr | mondrian | 0.0136264 | 0.0021978 | 0.0468132 | 0.0158242 | 0 | 0.971 | 35 |
| officehome_protocol_m | yes | YES | YES | `sar_online_aggressive` | ppi_debias | global | 0.0136264 | 0.0021978 | 0.0468132 | 0.0158242 | 0 | 0.971 | 35 |
| officehome_protocol_m | yes | YES | YES | `sar_online_aggressive` | ppi_debias | mondrian | 0.0136264 | 0.0021978 | 0.0468132 | 0.0158242 | 0 | 0.971 | 35 |
| fmow_val_to_test | diag |  |  | `tent_online` | gbr | mondrian | 0 | 0.0071771 | 0.0263995 | 0.0071771 | 0 | 0.43 | 300 |
| fmow_val_to_test | diag |  |  | `tent_online` | gbr | global | 0 | 0.0071771 | 0.0263995 | 0.0071771 | 0 | 0.423 | 300 |
| fmow_val_to_test | diag |  |  | `tent_online` | ppi_debias | mondrian | -5.55556e-05 | 0.00723265 | 0.0263995 | 0.0071771 | 1 | 0.513 | 300 |
| fmow_val_to_test | diag |  |  | `tent_online` | ppi_debias | global | -5.55556e-05 | 0.00723265 | 0.0263995 | 0.0071771 | 1 | 0.507 | 300 |
| iwildcam_full_idval_to_test | diag | YES |  | `tent_episodic` | gbr | mondrian | -0.0059038 | 0.0113964 | 0.102562 | 0.00549262 | 1 | 0.931 | 144 |
| iwildcam_full_idval_to_test | diag | YES |  | `tent_episodic` | gbr | global | -0.0059038 | 0.0113964 | 0.102562 | 0.00549262 | 1 | 0.924 | 144 |
| iwildcam_full_idval_to_test | diag |  |  | `tent_online` | gbr | global | -0.00682941 | 0.0132417 | 0.107532 | 0.00641227 | 1 | 0.938 | 144 |
| iwildcam_full_idval_to_test | diag |  |  | `tent_online` | ppi_debias | mondrian | -0.00836126 | 0.0147735 | 0.107532 | 0.00641227 | 1 | 0.944 | 144 |
