# WIN_HUNT v4 completion + v5 build — fold-in status (2026-07-05)

CPU-session fold-in of the WIN_HUNT_v4 arms and the WIN_HUNT_v5 aggressive-regime build.
Everything below is report-all and honest about what is GPU-gated. No frozen bar was changed;
no prior verdict (v1–v3, or v4 A/B/C) was edited.

## WIN_HUNT_v5 (aggressive-regime natural wave) — BUILT, awaiting GPU
- **config_lock FROZEN** in `research_lock/WIN_HUNT_v5_PROTOCOL_SHELL.yaml` (parses as valid YAML):
  one aggressive operating point per dataset = ONLINE adapter (no episodic reset), adapt-lr
  **0.004** (= 4× the uniform 1e-3 shared baseline), batch **16**, **aggressive** step budget
  (50; iWildCam/Office-Home 30), α=0.10. Interpretation of the shell's three axes is documented
  inline in the yaml (lr = absolute 0.004 override; "continual" = the runners' online mode;
  batch = small/tiny=16).
- **7 runners patched, opt-in, byte-identical by default** (new flags: `--adapt-lr`; CIFAR & PACS
  also `--batch-regimes`/`--aggressiveness`; camelyon/rxrx1/imagenet-r also `--online-only`). The
  new flags enter each runner's config-hash so v5 runs never collide with prior partials.
  Verified: all 7 `py_compile` clean; CIFAR & PACS `--help` expose the flags; selection + lr-override
  logic unit-checked against the real module constants; the wilds `--online-only` filter yields
  exactly the three `*_online` candidates.
- **RUNSHEET_WAVE7** at `docs/research/kbound/RUNSHEET_WAVE7.md` (bash-syntax clean): exact GPU/MPS
  commands for all 9 datasets, grouped by interpreter, with the real data-roots/ckpts/paths verified
  on disk (ImageNet-C, CIFAR-C, PACS under experiments/kbound/domainbed, iWildCam f0).
- **Pending: the user's GPU runs**, then score-once + pooled headline (Phase 4 of the runsheet).

## WIN_HUNT_v4 arm D (official per-sample POEM + dropout-AETTA) — PROVISIONAL WIN (2 of 5 seeds)
- Update 2026-07-05 PM: the user completed **seeds 0 and 1** (all 6 quick corruptions, 432 conditions
  each, `per_condition` JSONs present). Scored via the unmodified `score_official_headtohead.score()`
  (POEM per-sample decision, 22s/seed, staged per-seed to fit the runner budget; determinism guard
  on condition order passed). Result: `research_lock/WIN_HUNT_v4_ARM_D_result.json`.

  | policy | mean regret | false-adapt |
  |---|---|---|
  | **KGA** | **0.00154** | **0.000** |
  | official POEM | 0.01894 | 0.095 |
  | dropout-AETTA | 0.00233 | 0.144 |

  KGA beats POEM (diff −0.0174, CI [−0.0239, −0.0115], p_holm 2e-4) and AETTA (diff −0.0008,
  CI [−0.00153, **−0.00008**], p_holm 0.031) — **VERDICT: WIN, replacement_eligible** at FA=0 ≤ α.
- **PROVISIONAL — 2 of the protocol's 5 seeds.** seed2 is partial (gaussian_noise only, no
  `per_condition`); seeds 3-4 absent. The AETTA CI upper bound (−0.00008) sits essentially at zero, so
  the WIN vs AETTA could become a **TIE** at the full 5 seeds. **Do not fold as a headline replacement
  until seeds 0-4 are scored once.** Finalize: run seeds 2-4 then re-score `--seeds 0 1 2 3 4`
  (commands in RUNSHEET_WAVE7 APPENDIX). Full history in `WIN_HUNT_v4_ARM_D_STATUS.json`.

## WIN_HUNT_v4 arm E (stress-grid seeds 5-9) — PENDING GPU
- Seeds 5-9 not present under `stress_grid_multiseed_v1/`. GPU command in RUNSHEET_WAVE7 (APPENDIX).
  Pooled 10-seed CIs are produced at scoring time once seeds 5-9 land.

## WIN_HUNT_v4 arm F (KGA-v2 composite: jk+ radius + estimator v2) — WIRED + SCORED
Scorer: `docs/research/kbound/gapclose_wave5/rerun_F_composite_logged.py` (CPU, logged data only).
The composite is a SINGLE validity-preserving gate = the estimator-v2 3-config GBR ensemble (arm B)
used as the base learner inside the jackknife+/CV+ interval (arm A). Reuses arm A's frozen loader,
incumbent (radius_v2 crossfit_oof + Mondrian), and 10^4 paired bootstrap verbatim.

Verdicts (scored once; written to research_lock/):
| source | FA_u ≤ α | composite regret | incumbent regret | CI-robust improvement? | verdict |
|---|---|---|---|---|---|
| camelyon17 (Protocol G) | yes (0.000) | 0.0509 | 0.0308 | **no** (Δ=−0.0200, CI [−0.0244,−0.0161]) | no-change; composite abstains 95.6%, valid but more conservative |
| imagenet-r (panel) | yes (0.000) | 0.0000 | 0.0000 | no (tie at oracle) | no-change; helpful regime, ties always-adapt at [0,0] |
| pooled (2 avail. sources) | yes (0.051) | 0.0488 | 0.0320 | no; regret 0.0488 > 0.0183 existing → universal-gate WIN=false | no-change |

**Replacement determination (per the frozen v4 headline-replacement policy):** arm F produces NO
CI-robust improvement on any split, so it **replaces no headline** — it is reported alongside the
existing KGA rows, labeled **KGA-v2**. FA_u ≤ α holds everywhere (the composite is synthetically
valid), consistent with the standing v4 conclusion: better-calibrated instruments are valid but do
not cross the frontier — Camelyon is evidence-limited.

### arm F scope boundary (honest)
- Scored on the two natural protocols whose logs are in the arm-A `per_condition/per_panel` schema
  (**Camelyon G, ImageNet-R**). **iWildCam (H)** and **Office-Home (M)** store records inside their
  result JSONs, not `per_condition_*` files, so scoring them needs a small loader shim (follow-up)
  or a per_condition re-serialize of those runs.
- The pooled result is a **2-source partial** (evidence truncated to the shared base width), NOT the
  full arm-E base-11 nine-source universal gate; that requires the v5 GPU wave + arm E seeds.

## What remains (next, scoring session — per the shell handoff)
1. Run RUNSHEET_WAVE7 Phases 1–3 (9 datasets) + the APPENDIX (arm D completion, arm E 5-9).
2. Score v5 once (per-dataset + pooled 9-source headline via win_hunt_E_universal7.py); score arm D.
3. Fold-in + regenerate tables and both PDFs. Arm F rows fold in now as KGA-v2 (no replacement).
