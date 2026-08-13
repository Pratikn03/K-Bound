# NATURAL_MULTISEED_REPLAY_v1 — DRAFT pre-registration (NOT YET LOCKED)

**Status: DRAFT.** Per repository discipline, this protocol must be reviewed and locked
(committed to `research_lock/` with a lock date) BEFORE any new run it governs is scored.
Analyses of already-serialized records (Part A) are diagnostics and may run now, but no
tier upgrade is promoted until this document is locked and the criteria below are met.

## Purpose

Raise the three single-run natural-shift tracks — iWildCam H v2, Office-Home M v2,
RxRx1 J — from "single-run" evidence tier to a multi-seed tier, answering the
"one lucky run" objection. The claim under test is **no-harm stability**, NOT beats-both:
per seed, KGA must (i) keep FA_u ≤ α = 0.10 and (ii) match the better fixed policy
(regret_KGA ≤ min(regret_adapt, regret_freeze) + 0.005 tolerance). Beats-both point
estimates, if any, are reported but NOT promoted (consistent with the existing panel:
Office-Home LOO beats-both remains unpromoted).

## Seed taxonomy (state which dimension each part varies)

- **Decision seeds**: GBR `random_state` in the benefit estimator (repo pins 0).
- **Stream seeds**: the `seed` field in serialized records (adaptation-run randomness).
- **Model seeds**: independently trained source checkpoints (RxRx1 has 3 on disk).

A full "5-seed" parity claim with CIFAR-10-C/ImageNet-C requires stream- or model-seed
depth; decision-seed sweeps are supporting robustness only.

## Part A — from existing locked records (no GPU; runnable today)

1. **RxRx1 J, 3 model seeds.** Re-score `rxrx1_protocol_c_9plus_modelseed{0,1,2}`
   records under the locked J config (sar_online, gbr/global, dev {0–4}, test {5–9});
   verify byte-replay against the saved `analyze_F_results.json` files; aggregate
   mean±std; FA max. Also run the 16-point decision-seed sweep per model seed and the
   split-rotation diagnostic (dev {5–9} → test {0–4}, diagnostic only).
2. **Office-Home M v2, stream seeds.** Primary lock = stream seeds {0,1}
   (`officehome_full_targetval/targettest`); replication lock = fresh stream seeds
   {2,3,4} (`officehome_protocol_m_repl_*`). Score per-stream-seed (cal seed s →
   test seed s) plus pooled; decision-seed sweep on the pooled repl.
3. **iWildCam H v2.** Replay the locked dev screen (idval, cal 0 → eval 1) and its
   decision-seed sweep now; the held-out per-seed scoring (cal 0 → test 1 on
   `iwildcam_full_test/result_e40faf29.json`) runs as soon as the file is downloaded
   from iCloud.

**Aggregation format:** the Camelyon multi-seed table (paper `tab:multiseed`):
regret mean±std across seeds; FA_u = per-seed maximum; verdict column.

## Part B — new runs (GPU/MPS on this Mac; optional but what earns "5-seed" parity)

4. **iWildCam stream seeds {2,3,4}**: re-run the record-generating evaluation that
   produced `iwildcam_full_test` (same checkpoint, same 72 test conditions, same
   6-adapter panel, stream seeds 2–4), then score per seed under the locked H v2
   config. Runtime: comparable to the original full_test run on this Mac — check
   `experiments/kbound/results/iwildcam_full_test/run_test.log` timestamps before
   scheduling.
5. **Office-Home stream seeds {5,6,7}** (optional; {0..4} already exist across
   primary+repl): same generator as `officehome_protocol_m_repl_*`.
6. **RxRx1 model seeds {3,4}** (optional; requires training two more source
   checkpoints — the expensive arm; defer unless 5-model-seed parity is wanted).

## Success criteria (fixed in advance; no goalpost movement)

- Per track: no-harm (as defined above) holds on EVERY seed evaluated, and
  FA_u ≤ α on every seed. If any seed fails, the track's tier stays as-is and the
  failure is reported in the panel notes (no quiet exclusion).
- Tier label changes on pass, exactly:
  - RxRx1 J: "locked (real ckpt; single-run)" → "locked (3 model seeds, re-scored locked records)"
  - Office-Home M v2: "locked (OOF no-harm only; ...)" → append "(5 stream seeds: 0–1 primary + 2–4 replication)"
  - iWildCam H v2: "locked (OOF bootstrap; single-run)" → append "(stream seeds 0–1 + {2,3,4} if Part B runs)"
- The paper's promoted numeric row stays the primary-lock value; the multi-seed table
  is added alongside (as with Camelyon), not substituted.

## Artifacts

- `experiments/kbound/results/natural_seed_robustness_v1/natural_seed_robustness_v1.json`
  (Part A output, this analysis)
- `docs/research/kbound/scripts/natural_multiseed_aggregate.py` (aggregator; reruns Part A on this Mac)
- `docs/research/kbound/scripts/run_iwildcam_heldout_seed_sweep.py` (held-out sweep, pending iCloud download)
- `docs/research/kbound/RUNSHEET_NATURAL_MULTISEED.md` (commands)
