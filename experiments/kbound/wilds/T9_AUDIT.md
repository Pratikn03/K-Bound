# T9 drive audit — K-Bound relevance + space-reclaim map (2026-06-09)

Drive: /Volumes/T9, 1862 GB total, **~74 GB free (96% full)**. Sizes are TRUE on-disk
allocation (st_blocks, incl. exFAT 128 KB cluster slack). "PARTIAL≥" = time-capped scan
(lower bound); "measured" = complete; reads only, nothing deleted/moved. Camelyon job
(PID 86973) + downloads untouched.

## Categories
- **A = NEEDED for K-Bound** (paper, code/harness, theory validators, ACTIVE datasets, result manifests)
- **B = NOT K-Bound** (other projects — Pratik's, out of scope for K-Bound)
- **C = RECLAIMABLE** (redundant / bloated / duplicate)

## K-Bound — datasets (experiments/kbound/data/)
| Folder | Cat | Size | Needed | Note |
|---|---|---|---|---|
| imagenet-c | A | ≥47 GB (est ~70–115, partial) | yes | active SAR sweep data (paused run reads this) |
| wilds/camelyon17_v1.0 | **C** | ~75 GB st_blocks (du earlier ~110) | redundant | complete T9 copy incl 9.7 GB archive.tar.gz; **active run uses internal ~/kbound_cam, NOT this** → #1 reclaimable |
| wilds/rxrx1_v1.0 | A | 33 GB measured | yes | just downloaded; load-bug to patch before use |
| imagenet-r | A | 8.5 GB measured | yes | just downloaded |
| cifar | A | 3.3 GB measured | yes | CIFAR-10/100-C TTA data |
| imagenette | A | 3.9 GB measured | yes (minor) | imagenette TTA experiments |

## K-Bound — code / paper / results
| Folder | Cat | Size | Needed | Note |
|---|---|---|---|---|
| experiments/kbound/results (all run dirs + JSONs) | A | 0.36 GB | yes | manifests incl. active wilds_kbound_debug_mps + paused imagenetc_noise_sarfix |
| docs (docs/research/kbound paper+sections+status) | A | 0.1 GB | yes | paper sources |
| .git | A | 1.5 GB | yes | repo history |
| src / kga / scripts / configs / models / output / notebooks | A | <0.2 GB total | yes | code + harness |
| .venv (repo env) | A* | 9.3 GB | maybe | runs use ~/.venv_wilds (internal); **reclaimable IF this repo venv is unused — verify first** |
| uav/torch_cache | A | 0.14 GB | yes | small model cache |

## NOT K-Bound — other projects (Category B; the drive's real space hogs)
| Folder | Cat | Size | Note |
|---|---|---|---|
| gridpulse | B | **≥620 GB (partial; largest on drive)** | orius/gridpulse project |
| gridpulse_local_production_quarantine_2026…Z | B/C | **156 GB measured** | quarantined gridpulse backup — reclaimable, but Pratik's call |
| pratik_n_offload | B | 36 GB | offloaded archive |
| Pro_v8_AllInOne | B | ≥11 GB (partial) | other project |
| universal-anomaly-intelligence-v2 (ELARA?) | B | ≥10 GB (partial) | other project |
| trading-agent (sentifargo?) | B | 2.0 GB | other project |
| gridpulse-rq95 | B | 1.35 GB | other project |
| orius_external_data | B | 0.13 GB | other project |
| huggingface_cache / elite_engineering_900 / gridpulse_local_quarantine_* / .Trashes | B | ~0 GB (empty) | empty/near-empty |

## RECLAIMABLE — ranked (to free room for iWildCam)
iWildCam exFAT footprint ≈ **30–55 GB** (calibrated from rxrx1 = 33 GB / 251k files); peak +12 GB archive during download. Current 74 GB free is borderline for the ≥20 GB margin.

| Rank | Item | Frees | Risk | Domain |
|---|---|---|---|---|
| 1 | **camelyon17_v1.0 T9 copy** (whole dir) | **~75–110 GB** | low–med: active run uses internal copy; but this is the only 100% copy (internal is 91%) → re-download for full set later | K-Bound (Claude can act) |
| 1a | └ just archive.tar.gz inside it | 9.7 GB | **zero** (re-extractable compressed source) | K-Bound |
| 2 | .venv repo env | 9.3 GB | low IF unused (verify ~/.venv_wilds covers it) | K-Bound |
| 3 | gridpulse_local_production_quarantine | 156 GB | Pratik's call (not K-Bound) | other project |
| 4 | pratik_n_offload | 36 GB | Pratik's call (not K-Bound) | other project |

**Recommendation:** Reclaiming **#1 (camelyon17_v1.0 T9 copy, ~75–110 GB)** alone takes free space to ~150–185 GB — iWildCam then fits with large margin, touching only redundant K-Bound data. Nothing deleted here; Pratik decides.
