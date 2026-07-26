# K-Bound Short Paper Claim-to-Artifact Manifest

> Revised 2026-07-26. Three empirical rows re-scoped; artifact pointers corrected. Canonical:
> `SUBMISSION_LEDGER.md §3`, `§5`, `§8`, `§9`.

| Claim | Type | Exact requirement | Artifact | Caveat |
|---|---|---|---|---|
| Interior matched-evidence impossibility | theorem | `beta>0`, `|M|<beta`, rich declared class | `paper/sections/theory_core_main.tex` | opposite nonzero signs are interior only |
| Closed-band abstention | theorem | strict dual-error semantics | Proposition 2 in main source | boundary is zero-versus-strict ambiguity |
| Strict-commitment frontier | theorem | declared drift class and risk alignment | Theorem 2 in main source | not universal sign recovery |
| Marginal `FA_u` certificate | theorem | marginal interval coverage | Theorem 3; Lean inventory | no `FA_c` guarantee |
| CIFAR mixed-regime win | empirical | Tent/EATA head-to-head 5-seed aggregate and CIs | `mixed_headtohead_v1/HEADTOHEAD_RESULTS_cifar10c_{tent_primary,eata_secondary}.json` (**not** `LOCKED_ANALYSIS_RESULTS.json`, which is the stress-grid aggregate — see `SUBMISSION_LEDGER.md §5`) | SAR withheld after replay mismatch; EATA's adapt-gap CI does not survive corruption-family clustering |
| ImageNet-C SAR | empirical | 27 cells x 5 seeds = 135, paired bootstrap seed-averaged to 27 conditions | `win_hunt_v5_imagenetc_ms/pooled_5seed/` | **point-estimate no-harm only** under the declared LOO radius; freeze-gap CI includes zero. Not a beats-both. |
| Natural no-harm | empirical | reconciled OOF/locked held-out artifacts | OOF lock, RxRx1 lock; **Camelyon reconciliation directory is ABSENT** | not natural beats-both; Office-Home and iWildCam source records absent; RxRx1 (0 adapts) and iWildCam (1 adapt) leave the guarantee untested |
| POEM/AETTA comparison | empirical | same logged stress cells | mixed head-to-head artifact | protocol-matched ports only |
| Constructed mixed routing | empirical | fixed researcher-constructed stream | `mixed_protocol_oof_v2_result.json` | not unseen-shift transfer |
| Universal improvement | diagnostic | none | limitations/conclusion | explicitly not claimed |
| Single-dataset natural beats-both | diagnostic | held-out CIs against both fixed policies | natural tables | explicitly not claimed |
| Real-camera result | pending | fresh held-out physical sessions | preregistration templates | templates are not evidence |

Machine-readable exact values and source paths are in `paper/generated/kbound_result_manifest.json`.
