# K-Bound Empirical Recovery Audit

Date: 2026-08-13

## Scope and decision rule

This audit searched the internal working repository, the T9 archive, the nine-track lock,
generated paper tables, per-condition records, run manifests, protocol locks, and the current
91-page final-draft PDF. It addresses one question: are defensible results present on disk but
missing from the paper, and do any of them change the empirical conclusion?

Regret is regret to the per-condition oracle, so lower is better. The policy triple used below is
`KGA / always-adapt / always-freeze`. A result is called beats-both only when KGA is strictly below
both fixed policies under the declared inference unit. A point result is not called CI-robust when
either paired interval includes zero.

## Executive verdict

1. The repository does not contain a hidden clean, single-dataset natural-shift, multi-seed,
   CI-robust beats-both result.
2. The strongest defensible empirical result remains CIFAR-10-C Tent on a controlled mixed stress
   grid. It is cluster-robust when observed corruption families are resampled, but the utility win
   does not establish transfer to a new corruption family.
3. Several newer runs are absent or stale in the paper. They strengthen reproducibility and the
   no-harm story, not the natural-domain win story.
4. Several older files still say beats-both because they use interpolated or in-pool radii,
   incompatible seed lineages, pooled in-distribution and OOD cells, post-hoc split searches, or
   averages across separately calibrated candidates. These are not promotable.
5. The current empirical release is not frozen. Canonical reconciliation outputs and natural-seed
   robustness outputs are untracked, while manuscript and generated-manifest files are modified.
   A fresh checkout does not reproduce the current PDF's evidence state.

## Storage and repository inventory

The current code and manuscript authority is:

`/Users/pratik_n/Documents/AutoML_Flagship_V8`

The T9 path is not dataset-only. It also contains an older repository, two K-Bound working copies,
bare Git history/rewrite repositories, result trees, and caches under:

`/Volumes/T9/uav`

The internal repository contains approximately 2,145 result files: 828 JSON, 980 NPZ, 118 logs,
81 Markdown files, and 17 CSV files. Comparing result metadata against the older T9 repository found
246 files present only in the internal copy, none present only in the T9 copy, and two differing
files. The internal repository should therefore be treated as current code/result authority and T9
as raw-data plus archival storage.

Approximate raw-data inventory observed on T9:

| Track | T9 location | Observed state |
|---|---|---|
| CIFAR-10-C | `experiments/kbound/cifar/CIFAR-10-C` | Present; 19 corruption arrays plus labels, about 2.7 GB |
| CIFAR-10.1 | `experiments/kbound/cifar/CIFAR-10.1` | Present; about 6.4 MB |
| ImageNet-C | `experiments/kbound/data/imagenet-c` | Partially extracted; 13 corruption directories observed; `pixelate` and `jpeg_compression` absent; `gaussian_noise` has only severity 4 |
| ImageNet-R | `experiments/kbound/data/imagenet-r` | Present; about 8.0 GB; 60,202 files including macOS metadata sidecars |
| Office-Home | `experiments/kbound/data/office_home` | Present; all four domains, about 5.2 GB |
| PACS | `experiments/kbound/domainbed/PACS` | Present; all four domains, about 2.4 GB |
| iWildCam | `experiments/kbound/data/wilds/iwildcam_v2.0` | Present; about 38.2 GB |
| RxRx1 | `experiments/kbound/data/wilds/rxrx1_v1.0` | Present; about 30.7 GB |
| Camelyon17 | `experiments/kbound/data/wilds/camelyon17_v1.0` | Present; about 111.5 GB |

ImageNet-C requires special care. The saved 27-cell result is auditable from its serialized records,
but the full locked Protocol E run is not currently rerunnable from the observed extracted layout.
Protocol E requires all 19 corruptions at severities 4 and 5 and official SAR settings. The current
headline result is a different three-noise, five-seed stress configuration.

Additional raw datasets exist, including fMoW and CIFAR-100-C. They are not part of the locked
nine-track primary panel. No K-Bound CIFAR-100-C result artifact was found, and no ACDC K-Bound
runner/result artifact was found.

## Authoritative nine-track table

| Track | Unit and seeds | KGA / adapt / freeze regret | FA_u | Defensible verdict | Current paper status |
|---|---:|---:|---:|---|---|
| CIFAR-10-C Tent | 5 x 432 cells | 0.001626 / 0.007976 / 0.123937 | 0 | Controlled beats-both; robust to observed corruption-family resampling | Main result; headline pool still mixes in quarantined SAR |
| CIFAR-10-C EATA | 5 x 432 cells | 0.001313 / 0.003276 / 0.131337 | 0 | Point beats-both; adapt-side CI includes zero at corruption-cluster units | Mostly correct after caveat |
| CIFAR-10-C SAR rebuild | 5 x 432 cells | 0.001547 / 0.000335 / 0.140471 | 0 | Safe but loses to always-adapt on every seed | Incorrectly still described as withheld/incomplete |
| ImageNet-C Tent | 5 x 27 cells | 0.014465 / 0.019061 / 0.014465 | 0 | Ties freeze; no adapt decisions | Reported |
| ImageNet-C EATA | 5 x 27 cells | 0.000670 / 0.000096 / 0.034189 | 0.0074 | Helpful-dominated; loses to adapt | Reported |
| ImageNet-C SAR | 5 x 27 cells | 0.028893 / 0.052933 / 0.031894 | 0.0074 | Point beats-both only; freeze-side seed CI includes zero | Reported with substantial caveat; protocol label still ambiguous |
| ImageNet-R D | 4 x 10 x 12 = 480 cells | 0.014969 / 0.006359 / 0.032526 | 0 | Improves over freeze but significantly loses to adapt | Reported |
| Camelyon17 OOD | dev seeds 0-1; test seeds 2-4; n=18 | 0 / 0 / 0.138129 | 0 | Ties always-adapt; every test cell helpful | Reported; older multiseed subsection is stale |
| iWildCam H-v2 | cal seed 0; test seed 1; n=72 | 0.004102 / 0.102830 / 0.004102 | 0 | Ties freeze; 0/21/51 decisions; A7 open | Decision table incorrectly prints 0/24/48 |
| Office-Home M-v2 primary | seeds 0-1; n=35 | 0.015824 / 0.046813 / 0.015824 | 0 | Ties freeze; A7 open | Reported as single-run despite saved replication |
| RxRx1 J | dev seeds 0-4; test seeds 5-9; n=60 | 0 / 0.253060 / 0 | 0 | Always-freeze behavior; no-harm but guarantee untested | Reported as single-run despite three model seeds |
| PACS | 3 seeds x 4 domains x 18 cells | 0.043136 / 0.017637 / 0.044606 | 0.00926 | Null; KGA is 2.45x worse than adapt | Reported; gate replay cannot be reconstructed |
| CIFAR-10.1 K | dev seeds 0-2; test seeds 3-4; n=48 | 0.002063 / 0.019021 / 0.001708 | 0.1667 | Fails safety and loses to freeze | Reported as diagnostic |

The table has more than nine rows because CIFAR-10-C and ImageNet-C contain separately calibrated
candidate adapters. They must not be averaged and presented as one deployable policy.

## Omitted evidence that should be folded in

### 1. Completed CIFAR-10-C SAR negative control

The five-seed rebuild under `cifar10c_sar_rebuild_v2` is complete and passes the validator. All five
manifests use runner commit `675ebfcb7a56854123b13250e01843f69007589b`, the frozen runner SHA-256
`f1687904...e62dbc`, and the same checkpoint SHA-256 `43333456...da7b`. It contains 2,160 complete
per-cell records and 1,448 ADAPT plus 712 ABSTAIN decisions, with zero false adaptations.

Direct replay gives KGA minus always-adapt regret `+0.001212`, so KGA is worse. An i.i.d. cell
bootstrap gives approximately `[+0.00104,+0.00138]`. Every seed separately loses to always-adapt.
This should replace the current “withheld SAR” description with a completed, reproducible negative
control. It does not enter the positive CIFAR headline.

The `0.001547` value in this audit is the mean produced by the rebuilt run's serialized decisions.
A later exact leave-one-record-out sensitivity rescore is approximately `0.00156`. The two values
must not be presented as the same estimator pass, but both are decisively above always-adapt's
`0.000335` regret and therefore have the same scientific verdict.

### 2. Office-Home independent stream-seed replication

Fresh stream seeds 2-4 give `0.021510 / 0.045798 / 0.021724`, with decisions `1/14/39`. The point
margin over freeze is only `0.000214`; the seed-bootstrap interval is `[0,0.000641]`, so it is not
CI-robust. This is eligible as secondary replication evidence, explicitly labeled A7-open and not a
natural beats-both result.

### 3. RxRx1 model-seed stability

Three independent model checkpoints give KGA/freeze regret zero while always-adapt gives about
`0.2531`, `0.2638`, and `0.2583`. Across 48 decision-seed replays, no-harm remains unchanged. This is
valuable appendix evidence that the freeze result is not a single checkpoint accident. It remains a
zero-adapt result, so it does not exercise the false-adapt guarantee.

### 4. Camelyon17 Protocol B-v2 SAR

The three-seed, 36-condition-per-seed cross-fitted stress panel gives
`0.000633 / 0.001637 / 0.100125`. All three seed point estimates beat both, but the paired adapt-side
CI is `[-0.002586,+0.000706]` and the Holm-adjusted test does not pass. This is a useful exploratory
stress result, not a held-out natural-domain win.

### 5. iWildCam five-seed stability diagnostics

Five-seed Tent online and episodic files show stable no-harm in within-seed LOO stress grids. Their
own metadata states that they are not Protocol H-v2 OOF replays. They may support a diagnostic
stability appendix but cannot be used as extra held-out iWildCam test seeds.

### 6. fMoW and stopped tracks

fMoW gives `0.012938 / 0.009168 / 0.012938` over 180 cells: a freeze tie that loses to adapt. The
stored `false_adapt=.375` is conditional FA_c, not theorem-controlled FA_u. PovertyMap stopped before
held-out testing because development harm-AUC `0.6373` was below the preregistered `0.65` gate.
Both are useful transparent negative evidence, not headline results.

## Strong-looking artifacts that must not be promoted

| Artifact/result | Why it looks strong | Why it is not paper-eligible |
|---|---|---|
| Old CIFAR-10-C SAR aggregate | 0.00155 / 0.01119 / 0.12864 | Mixed an anomalous seed-0 lineage with later low-harm seeds; clean rebuild gives adapt regret 0.000335 and no win |
| ImageNet-C three-candidate aggregate | 0.01468 / 0.02403 / 0.02685 with seed CI beats-both | Average of three separately calibrated candidate policies; not one deployable router |
| Old ImageNet-C SAR in-pool radius | CI-supported beats-both | Radius included the scored cell's residual; exact LOO changes the promoted conclusion |
| Office-Home old `VERIFIED_FINDINGS.json` | 0.00220 KGA and beats-both | Interpolated/older scoring; exact-rank replay ties freeze |
| iWildCam old `VERIFIED_FINDINGS.json` | 0.00367 KGA and beats-both | Superseded scorer; exact-rank replay ties freeze |
| Camelyon Protocol G pooled n=54 | Strong pooled beats-both | Combined OOD test, OOD validation, and in-distribution cells; explicitly withdrawn |
| Historical three-source OOF mixture | Strong CI beats-both | Researcher-constructed mixture of separately calibrated gates; components have changed and replay is pending |
| ImageNet-R selected split or backbone wins | Some split/seed point wins | Split/backbone search; pooled four-seed ten-backbone panel loses to adapt |
| CIFAR-10.1 isolated EATA seed | One seed point beats-both | Aggregate locked cross-seed result fails FA_u and loses to freeze |
| iWildCam LAME router | Macro-F1 CI above freeze | Old radius, partial staged stream, camera split mismatch, one checkpoint, incomplete exact replay |
| Protocol D official baseline arm | Favorable provisional two-seed comparison | Only seeds 0-1 of five completed; baseline implementation/reproduction scope incomplete |
| 3D-ADAM named-condition routing | Point AUROC 0.8026 vs freeze 0.7942 and adapt 0.7561 | KGA-minus-freeze bootstrap touches zero; selected from five caches; diagnostic only |

D33 remains useful controlled multimodal mechanism evidence. Like the constructed heterogeneous
mixture, it does not convert into a single-dataset natural-shift claim.

## Why empirical performance is genuinely limited

### One-sided natural shifts

Camelyon17 OOD is entirely helpful, so always-adapt is already oracle. RxRx1 and iWildCam are strongly
harmful, so always-freeze is already oracle-like. A router cannot strictly beat the best fixed policy
unless helpful and harmful conditions coexist in meaningful proportions.

### Signal is smaller than uncertainty

On ImageNet-R, mean epsilon is `0.04429` while mean absolute benefit is `0.03889`, a radius-to-signal
ratio of `1.139`. Abstention is expected. Reducing the radius without new calibration evidence raises
false-adapt risk; it does not create information.

### Small effective sample size

Hundreds of cells are not hundreds of independent datasets. CIFAR twin repeats are highly correlated,
ImageNet-C has only three corruption families, Camelyon B-v2 has three independent model/data seeds,
and several natural protocols have one held-out seed. Cell-wise bootstrap intervals can therefore be
much narrower than scientifically appropriate cluster- or seed-level intervals.

### Estimator transfer is not established

Office-Home and iWildCam have no predeclared uniform full-fit-versus-LOO stability bound. The observed
maximum gaps are approximately `0.0491` and `0.2583`. This does not invalidate their descriptive
replays, but it prevents a strong coverage claim under the current transfer argument.

### Adapter configuration changes the regime

The 27-cell ImageNet-C SAR result uses an aggressive stress operating point. The locked full Protocol
E asks whether the behavior survives official SAR at learning rate `2.5e-4`, a frozen final block,
all 19 corruptions, severities 4-5, and three seeds. That experiment is not the saved 27-cell result.
Calling both “Protocol E” hides a material adapter/configuration difference.

### Historical quantile and pooling choices inflated wins

Interpolated quantiles, in-pool residual radii, post-hoc split searches, and pooling across domains or
separately calibrated candidates explain most strong historical natural results. Exact rank and clean
holdout replay usually widen the radius or remove the strict advantage.

## Current manuscript/result inconsistencies

1. The CIFAR headline cites 6,480 cells, which is three candidates x five seeds x 432 cells, while the
   promoted decision counts and positive claim are Tent/EATA and SAR is explicitly quarantined. The
   headline must be regenerated on an eligible population or labeled an all-candidate diagnostic.
2. The alpha, estimator, radius-free, and cross-adapter ablation prose does not match the named
   `ablation_exactrank.json` artifact. For example, radius-free Tent FA_u is 0.0324 in the artifact,
   not 0.051; SAR-to-Tent FA_u is 0.0949, not 0.25.
3. CIFAR-10-C SAR is complete but the paper still says withheld/incomplete.
4. iWildCam counts are `0/21/51`, not `0/24/48`.
5. The paper says iWildCam, Office-Home, and RxRx1 are single-run. Office has a three-seed replication,
   RxRx1 has three model seeds, and iWildCam has five diagnostic stress seeds. Their protocol scope
   must be stated instead of calling all of them single-run.
6. The old Camelyon “four seeds x nine conditions” subsection is not the current OOD result and is
   superseded by the n=18 OOD reconciliation plus the separate B-v2 stress panel.
7. `VERIFIED_FINDINGS.json` for Office-Home and iWildCam still says beats-both, contradicting canonical
   exact-rank replay.
8. `results_source.json` and the generated result manifest still contain a stale Camelyon
   non-reproducibility statement even though the tracked source and reconciliation now exist.
9. The “9,504 committed cells” total sums overlapping artifact trees and must not be presented as an
   independent experimental sample count.
10. A demoted three-source heterogeneous result is called locked in the main text while the appendix
    correctly says replay is pending.
11. The current `kbound_short_final_draft.pdf` is byte-identical to `kbound_tmlr.pdf`, not
    `kbound_short.pdf`. It is 91 pages and depends on untracked reconciliation state.

## Required reconciliation order

1. Extend `scripts/reconcile_result_panels.py` from five panels to all nine tracks and all promoted
   adapter rows, including CIFAR-10-C SAR rebuild, Camelyon OOD/B-v2, RxRx1, and CIFAR-10.1.
2. Make one generated verdict JSON the only numerical source for LaTeX macros, tables, figures, the
   claim ledger, and result manifest.
3. Mark old Office/iWild `VERIFIED_FINDINGS.json`, the old CIFAR SAR aggregate, and pooled Camelyon G
   as superseded in machine-readable metadata. Do not delete raw evidence.
4. Replace the paper's stale counts and ablation numbers from the canonical generated source.
5. Separate primary held-out results, controlled stress results, and diagnostics in different tables.
6. Rebuild all figures from the same canonical source and remove the 6,480-cell mixed-population
   headline unless it is explicitly diagnostic.
7. Commit the reconciler, compact source records, canonical outputs, tests, and updated manuscript;
   regenerate the nine-track seal from a clean worktree.
8. Run the full locked Protocol E only after the complete ImageNet-C data contract is restored. Its
   outcome must not be inferred from the aggressive 27-cell panel.

## Source integrity

Primary canonical panel:

`experiments/kbound/results/reconciled_panels_v1/canonical_panel_results.json`

SHA-256: `fb2e4b989a55c1cb9525f18dc9d26c93bc8a2e04879f43e0e9fdb864b2c48a30`

Camelyon B-v2 multiseed:

`experiments/kbound/results/camelyon17_fullscale_B_v2/MULTISEED_ANALYSIS_RESULTS.json`

SHA-256: `0e337d9de2867ef6288ee49d9988a5c1fd3908163695ab4e488da129bcaad158`

Natural seed robustness:

`experiments/kbound/results/natural_seed_robustness_v1/natural_seed_robustness_v1.json`

SHA-256: `bc47f799cf896a46df7e3a24e086ef515ff6c37e16f327d498f2119054f6a1d7`

Full ImageNet-C Protocol E lock:

`research_lock/imagenetc_protocol_E_v1.yaml`

SHA-256: `d22f929d0e30cb1eebdb92be6023ffea9231e8052598888e9df23c1677515197`

Verification performed during this audit:

- `tests/test_reconciled_panels.py`: PASS, 3 tests.
- CIFAR-10-C SAR rebuild validator: PASS, complete seeds 0-4.
- Current final PDF: 91 pages; SHA-256
  `b698c1e1dc8533d53a109b7e97f9066d26126f3d732ca5e22c1eb404beaa5d6f`.

## Bottom line

There is valuable unwritten evidence, but it is mostly replication, stability, and negative-control
evidence. Folding it in will make the paper more credible and less vulnerable to a reproducibility
review. It will not honestly create a clean natural-dataset beats-both claim. The path to a stronger
empirical paper is a frozen all-track pipeline plus a prospectively locked mixed-sign natural stream,
not recovery of a hidden positive number from the current artifact tree.
