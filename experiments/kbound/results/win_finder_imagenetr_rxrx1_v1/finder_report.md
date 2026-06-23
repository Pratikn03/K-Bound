# K-Bound Dev Win Finder

Exploratory/dev-screen report. A row is paper-eligible only after a fresh locked held-out evaluation.

| rank | win? | label | est | conf | margin | KGA | adapt | freeze | FA | cov | seeds |
|---:|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | YES | `cifar10c_stress:tent` | ppi_debias | global | 0.00683333 | 0.000943287 | 0.00777662 | 0.124417 | 0.00209 | 0.744 | `[0, 1, 2]->[3, 4]` |
| 2 | YES | `cifar10c_stress:tent` | ppi_debias | mondrian | 0.00683333 | 0.000943287 | 0.00777662 | 0.124417 | 0.00209 | 0.744 | `[0, 1, 2]->[3, 4]` |
| 3 | YES | `cifar10c_stress:tent` | gbr | global | 0.00646238 | 0.00131424 | 0.00777662 | 0.124417 | 0.00219 | 0.715 | `[0, 1, 2]->[3, 4]` |
| 4 | YES | `cifar10c_stress:tent` | gbr | mondrian | 0.00646238 | 0.00131424 | 0.00777662 | 0.124417 | 0.00219 | 0.715 | `[0, 1, 2]->[3, 4]` |
| 5 | YES | `cifar10c_stress:tent` | gbr | cqr | 0.00609433 | 0.00168229 | 0.00777662 | 0.124417 | 0 | 0.694 | `[0, 1, 2]->[3, 4]` |
| 6 | YES | `cifar10c_stress:tent` | ppi_debias | cqr | 0.00609433 | 0.00168229 | 0.00777662 | 0.124417 | 0 | 0.694 | `[0, 1, 2]->[3, 4]` |
| 7 | YES | `cifar10c_stress:eata` | ppi_debias | global | 0.00224826 | 0.000740741 | 0.002989 | 0.131793 | 0 | 0.707 | `[0, 1, 2]->[3, 4]` |
| 8 | YES | `cifar10c_stress:eata` | ppi_debias | mondrian | 0.00224826 | 0.000740741 | 0.002989 | 0.131793 | 0 | 0.707 | `[0, 1, 2]->[3, 4]` |
| 9 | YES | `cifar10c_stress:eata` | gbr | global | 0.00215451 | 0.000834491 | 0.002989 | 0.131793 | 0 | 0.689 | `[0, 1, 2]->[3, 4]` |
| 10 | YES | `cifar10c_stress:eata` | gbr | mondrian | 0.00215451 | 0.000834491 | 0.002989 | 0.131793 | 0 | 0.689 | `[0, 1, 2]->[3, 4]` |
| 11 | YES | `cifar10c_stress:eata` | gbr | cqr | 0.00173669 | 0.00125232 | 0.002989 | 0.131793 | 0.00196 | 0.677 | `[0, 1, 2]->[3, 4]` |
| 12 | YES | `cifar10c_stress:eata` | ppi_debias | cqr | 0.00173669 | 0.00125232 | 0.002989 | 0.131793 | 0.00196 | 0.677 | `[0, 1, 2]->[3, 4]` |
| 13 | YES | `experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json:tent_online` | gbr | cqr | 0.00104167 | 0.00416667 | 0.00520833 | 0.0180208 | 0.0714 | 0.583 | `[0, 1]->[2, 3]` |
| 14 | YES | `experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json:tent_online` | ppi_debias | cqr | 0.00104167 | 0.00416667 | 0.00520833 | 0.0180208 | 0.0714 | 0.583 | `[0, 1]->[2, 3]` |
| 15 | YES | `experiments/kbound/results/rxrx1_kbound_light_mps_internal/result_f6b268c7.json:tent_online` | gbr | cqr | 0.000488281 | 0.000488281 | 0.0664062 | 0.000976562 | 0 | 0.708 | `[0, 1]->[2, 3]` |
| 16 | YES | `experiments/kbound/results/rxrx1_kbound_light_mps_internal/result_f6b268c7.json:tent_online` | ppi_debias | cqr | 0.000488281 | 0.000488281 | 0.0664062 | 0.000976562 | 0 | 0.708 | `[0, 1]->[2, 3]` |
| 17 | YES | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json:tent_online` | ppi_debias | mondrian | 0.000406901 | 0.00101725 | 0.0502116 | 0.00142415 | 0 | 0.833 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 18 | YES | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json:tent_online` | ppi_debias | global | 0.000406901 | 0.00101725 | 0.0502116 | 0.00142415 | 0 | 0.812 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 19 | YES | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json:tent_online` | gbr | global | 0.000406901 | 0.00101725 | 0.0502116 | 0.00142415 | 0 | 0.688 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 20 | YES | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json:tent_online` | gbr | mondrian | 0.000406901 | 0.00101725 | 0.0502116 | 0.00142415 | 0 | 0.688 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 21 | YES | `cifar101_multiseed:eata` | gbr | cqr | 0.000312501 | 0.00210417 | 0.005 | 0.00241667 | 0 | 0.271 | `[0, 1, 2]->[3, 4]` |
| 22 | YES | `cifar101_multiseed:eata` | ppi_debias | cqr | 0.000312501 | 0.00210417 | 0.005 | 0.00241667 | 0 | 0.271 | `[0, 1, 2]->[3, 4]` |
| 23 |  | `experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json:sar_online` | gbr | global | 0.01625 | 0.00177083 | 0.0446875 | 0.0180208 | 0.167 | 1 | `[0, 1]->[2, 3]` |
| 24 |  | `experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json:sar_online` | gbr | mondrian | 0.01625 | 0.00177083 | 0.0446875 | 0.0180208 | 0.167 | 1 | `[0, 1]->[2, 3]` |
| 25 |  | `experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json:sar_online` | ppi_debias | global | 0.01625 | 0.00177083 | 0.0446875 | 0.0180208 | 0.167 | 1 | `[0, 1]->[2, 3]` |
| 26 |  | `experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json:sar_online` | ppi_debias | mondrian | 0.01625 | 0.00177083 | 0.0446875 | 0.0180208 | 0.167 | 1 | `[0, 1]->[2, 3]` |
| 27 |  | `experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json:sar_online` | gbr | cqr | 0.0103125 | 0.00770833 | 0.0446875 | 0.0180208 | 0.214 | 0.792 | `[0, 1]->[2, 3]` |
| 28 |  | `experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json:sar_online` | ppi_debias | cqr | 0.0103125 | 0.00770833 | 0.0446875 | 0.0180208 | 0.214 | 0.792 | `[0, 1]->[2, 3]` |
| 29 |  | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json:eata_online` | ppi_debias | global | 0.00158691 | 0.0041097 | 0.0106608 | 0.00569661 | 0.353 | 0.854 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 30 |  | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json:eata_online` | gbr | global | 0.00150553 | 0.00419108 | 0.0106608 | 0.00569661 | 0.308 | 0.729 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 31 |  | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json:eata_online` | ppi_debias | mondrian | 0.00142415 | 0.00427246 | 0.0106608 | 0.00569661 | 0.375 | 0.833 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 32 |  | `experiments/kbound/results/rxrx1_kbound_light_mps_internal/result_f6b268c7.json:eata_online` | gbr | cqr | 0.00130208 | 0.00488281 | 0.016276 | 0.0061849 | 0.375 | 0.542 | `[0, 1]->[2, 3]` |
| 33 |  | `experiments/kbound/results/rxrx1_kbound_light_mps_internal/result_f6b268c7.json:eata_online` | ppi_debias | cqr | 0.00130208 | 0.00488281 | 0.016276 | 0.0061849 | 0.375 | 0.542 | `[0, 1]->[2, 3]` |
| 34 |  | `cifar101_multiseed:eata` | gbr | global | 0.001125 | 0.00129167 | 0.005 | 0.00241667 | 0.304 | 0.833 | `[0, 1, 2]->[3, 4]` |
| 35 |  | `cifar101_multiseed:eata` | gbr | mondrian | 0.001125 | 0.00129167 | 0.005 | 0.00241667 | 0.304 | 0.833 | `[0, 1, 2]->[3, 4]` |
| 36 |  | `cifar101_multiseed:eata` | ppi_debias | global | 0.0010625 | 0.00135417 | 0.005 | 0.00241667 | 0.333 | 0.875 | `[0, 1, 2]->[3, 4]` |
| 37 |  | `cifar101_multiseed:eata` | ppi_debias | mondrian | 0.0010625 | 0.00135417 | 0.005 | 0.00241667 | 0.333 | 0.875 | `[0, 1, 2]->[3, 4]` |
| 38 |  | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json:eata_online` | gbr | mondrian | 0.000895182 | 0.00480143 | 0.0106608 | 0.00569661 | 0.333 | 0.708 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 39 |  | `experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json:tent_online` | gbr | global | 0.000833333 | 0.004375 | 0.00520833 | 0.0180208 | 0.182 | 1 | `[0, 1]->[2, 3]` |
| 40 |  | `experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json:tent_online` | gbr | mondrian | 0.000833333 | 0.004375 | 0.00520833 | 0.0180208 | 0.182 | 1 | `[0, 1]->[2, 3]` |
| 41 |  | `experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json:tent_online` | ppi_debias | global | 0.000833333 | 0.004375 | 0.00520833 | 0.0180208 | 0.182 | 1 | `[0, 1]->[2, 3]` |
| 42 |  | `experiments/kbound/results/imagenetr_kbound_light_mps_internal/result_f4a1293b.json:tent_online` | ppi_debias | mondrian | 0.000833333 | 0.004375 | 0.00520833 | 0.0180208 | 0.182 | 1 | `[0, 1]->[2, 3]` |
| 43 |  | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed2/result_6585f5b7.json:eata_online` | ppi_debias | global | 0.000732422 | 0.00260417 | 0.0111898 | 0.00333659 | 0.5 | 1 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 44 |  | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed2/result_6585f5b7.json:eata_online` | ppi_debias | mondrian | 0.000732422 | 0.00260417 | 0.0111898 | 0.00333659 | 0.5 | 1 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 45 |  | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed2/result_6585f5b7.json:tent_episodic` | ppi_debias | global | 0.000610352 | 0.00309245 | 0.00939941 | 0.0037028 | 0.333 | 0.896 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 46 |  | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed2/result_6585f5b7.json:tent_episodic` | ppi_debias | mondrian | 0.000610352 | 0.00309245 | 0.00939941 | 0.0037028 | 0.333 | 0.896 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 47 |  | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed2/result_6585f5b7.json:tent_episodic` | gbr | global | 0.000366211 | 0.00333659 | 0.00939941 | 0.0037028 | 0.5 | 0.812 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 48 |  | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed2/result_6585f5b7.json:tent_episodic` | gbr | mondrian | 0.000366211 | 0.00333659 | 0.00939941 | 0.0037028 | 0.5 | 0.792 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 49 |  | `cifar101_multiseed:tent` | gbr | cqr | 0.000270835 | 0.0014375 | 0.0190208 | 0.00170834 | 0.25 | 0.583 | `[0, 1, 2]->[3, 4]` |
| 50 |  | `cifar101_multiseed:tent` | ppi_debias | cqr | 0.000270835 | 0.0014375 | 0.0190208 | 0.00170834 | 0.25 | 0.583 | `[0, 1, 2]->[3, 4]` |
| 51 |  | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json:eata_episodic` | ppi_debias | mondrian | 0.00012207 | 0.00431315 | 0.00590007 | 0.00443522 | 0.333 | 0.708 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 52 |  | `experiments/kbound/results/rxrx1_protocol_c_9plus_modelseed0/result_3f579e72.json:eata_episodic` | ppi_debias | global | 0.00012207 | 0.00431315 | 0.00590007 | 0.00443522 | 0.333 | 0.688 | `[0, 1, 2, 3, 4, 5]->[6, 7, 8, 9]` |
| 53 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:convnext_base` | gbr | global | 0 | 0 | 0 | 0.0697917 | 0 | 1 | `[0, 1]->[2, 3]` |
| 54 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:convnext_base` | gbr | mondrian | 0 | 0 | 0 | 0.0697917 | 0 | 1 | `[0, 1]->[2, 3]` |
| 55 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:convnext_base` | gbr | cqr | 0 | 0 | 0 | 0.0697917 | 0 | 1 | `[0, 1]->[2, 3]` |
| 56 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:convnext_base` | ppi_debias | global | 0 | 0 | 0 | 0.0697917 | 0 | 1 | `[0, 1]->[2, 3]` |
| 57 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:convnext_base` | ppi_debias | mondrian | 0 | 0 | 0 | 0.0697917 | 0 | 1 | `[0, 1]->[2, 3]` |
| 58 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:convnext_base` | ppi_debias | cqr | 0 | 0 | 0 | 0.0697917 | 0 | 1 | `[0, 1]->[2, 3]` |
| 59 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:efficientnet_b0` | gbr | global | 0 | 0 | 0.0509375 | 0 | 0 | 1 | `[0, 1]->[2, 3]` |
| 60 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:efficientnet_b0` | gbr | mondrian | 0 | 0 | 0.0509375 | 0 | 0 | 1 | `[0, 1]->[2, 3]` |
| 61 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:efficientnet_b0` | ppi_debias | global | 0 | 0 | 0.0509375 | 0 | 0 | 1 | `[0, 1]->[2, 3]` |
| 62 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:efficientnet_b0` | ppi_debias | mondrian | 0 | 0 | 0.0509375 | 0 | 0 | 1 | `[0, 1]->[2, 3]` |
| 63 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:efficientnet_b3` | gbr | global | 0 | 0 | 0 | 0.0452083 | 0 | 1 | `[0, 1]->[2, 3]` |
| 64 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:efficientnet_b3` | gbr | mondrian | 0 | 0 | 0 | 0.0452083 | 0 | 1 | `[0, 1]->[2, 3]` |
| 65 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:efficientnet_b3` | ppi_debias | global | 0 | 0 | 0 | 0.0452083 | 0 | 1 | `[0, 1]->[2, 3]` |
| 66 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:efficientnet_b3` | ppi_debias | mondrian | 0 | 0 | 0 | 0.0452083 | 0 | 1 | `[0, 1]->[2, 3]` |
| 67 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:resnet152` | gbr | global | 0 | 0 | 0 | 0.0390625 | 0 | 1 | `[0, 1]->[2, 3]` |
| 68 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:resnet152` | gbr | mondrian | 0 | 0 | 0 | 0.0390625 | 0 | 1 | `[0, 1]->[2, 3]` |
| 69 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:resnet152` | gbr | cqr | 0 | 0 | 0 | 0.0390625 | 0 | 1 | `[0, 1]->[2, 3]` |
| 70 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:resnet152` | ppi_debias | global | 0 | 0 | 0 | 0.0390625 | 0 | 1 | `[0, 1]->[2, 3]` |
| 71 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:resnet152` | ppi_debias | mondrian | 0 | 0 | 0 | 0.0390625 | 0 | 1 | `[0, 1]->[2, 3]` |
| 72 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:resnet152` | ppi_debias | cqr | 0 | 0 | 0 | 0.0390625 | 0 | 1 | `[0, 1]->[2, 3]` |
| 73 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:resnext101_32x8d` | gbr | global | 0 | 0 | 0 | 0.0516667 | 0 | 1 | `[0, 1]->[2, 3]` |
| 74 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:resnext101_32x8d` | gbr | mondrian | 0 | 0 | 0 | 0.0516667 | 0 | 1 | `[0, 1]->[2, 3]` |
| 75 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:resnext101_32x8d` | gbr | cqr | 0 | 0 | 0 | 0.0516667 | 0 | 1 | `[0, 1]->[2, 3]` |
| 76 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:resnext101_32x8d` | ppi_debias | global | 0 | 0 | 0 | 0.0516667 | 0 | 1 | `[0, 1]->[2, 3]` |
| 77 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:resnext101_32x8d` | ppi_debias | mondrian | 0 | 0 | 0 | 0.0516667 | 0 | 1 | `[0, 1]->[2, 3]` |
| 78 |  | `experiments/kbound/results/imagenetr_protocol_d_size_diverse_panel_v2/result_fb5afb1e.json:resnext101_32x8d` | ppi_debias | cqr | 0 | 0 | 0 | 0.0516667 | 0 | 1 | `[0, 1]->[2, 3]` |
| 79 |  | `experiments/kbound/results/rxrx1_kbound_light_mps_internal/result_f6b268c7.json:sar_online` | gbr | global | 0 | 0 | 0.260579 | 0 | 0 | 1 | `[0, 1]->[2, 3]` |
| 80 |  | `experiments/kbound/results/rxrx1_kbound_light_mps_internal/result_f6b268c7.json:sar_online` | gbr | mondrian | 0 | 0 | 0.260579 | 0 | 0 | 1 | `[0, 1]->[2, 3]` |
