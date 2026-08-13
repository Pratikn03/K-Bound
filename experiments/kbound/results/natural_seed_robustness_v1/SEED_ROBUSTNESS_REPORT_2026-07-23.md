# Natural-shift seed-robustness — results (Part A, 2026-07-23)

Diagnostic robustness analysis on locked, already-serialized records only. No new
adaptation runs; no headline promotion (pending lock of NATURAL_MULTISEED_REPLAY_v1).
Pipeline: the repo's own `analyze_F.py` (gbr + global exact-rank conformal, α=0.10,
LOO calibration residuals), with only the GBR `random_state` exposed for the sweep.

## RxRx1 Protocol J — now a 3-model-seed result ✅

| model seed | KGA | always-adapt | always-freeze | FA_u | replay = locked artifact | no-harm |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | 0.2531 | 0.0000 | 0.000 | **exact** | yes |
| 1 | 0.0000 | 0.2638 | 0.0000 | 0.000 | **exact** | yes |
| 2 | 0.0000 | 0.2583 | 0.0000 | 0.000 | **exact** | yes |

Aggregate: KGA 0.0000 ± 0.0000; adapt 0.2584 ± 0.0044; freeze 0.0000 ± 0.0000; FA max 0.000.
Decision-seed sweep (16 GBR seeds × 3 model seeds): no-harm on **48/48**; FA max 0.000;
KGA regret range [0.0000, 0.0000]. Split-rotation diagnostic (dev 5–9 → test 0–4): clean.

**Meaning:** the promoted single-run RxRx1 row is confirmed by two additional
independently trained checkpoints already on disk, byte-replaying their locked
artifacts. Supports tier change to "locked (3 model seeds, re-scored locked records)"
once the protocol is locked. This closes the "one lucky run" objection for RxRx1
without any new GPU time. (Doc nit: `rxrx1_protocol_J_v1/VERIFIED_FINDINGS.md`'s
robustness table lists always-adapt 0.2531 for all three model seeds; the underlying
JSONs say 0.2531/0.2638/0.2583 — KGA/freeze columns are unaffected.)

## Office-Home M v2 — three FRESH stream seeds, all no-harm ✅ (with one honest flag)

The replication-lock records (`officehome_protocol_m_repl_*`) are fresh stream seeds
{2,3,4} — the primary lock used {0,1}. Scored per seed (cal seed s → test seed s,
corrected LOO radius, sar_online_aggressive):

| stream seed | KGA | always-adapt | always-freeze | FA_u | no-harm |
|---|---|---|---|---|---|
| 2 | 0.0291 | 0.0382 | 0.0291 | 0.000 | yes (ties freeze) |
| 3 | 0.0154 | 0.0470 | 0.0154 | 0.000 | yes (ties freeze) |
| 4 | 0.0207 | 0.0521 | 0.0207 | 0.000 | yes (ties freeze) |

Decision-seed sweep (16 seeds, pooled 2–4): no-harm on **16/16**, FA max 0.000,
KGA regret range [0.0196, 0.0217]. Combined with the promoted primary run (seeds
0–1: 0.0157 vs freeze 0.0158, FA 0), Office-Home no-harm now replicates across
**5 stream seeds**.

**Honest flag (report, don't hide):** the archived `officehome_protocol_m_repl_holdout/
holdout_score.json` (KGA 0.00043, adapt-rate 0.63, eps 0.0047, "beats_both: true")
does NOT reproduce under the corrected out-of-fold radius — consistent with it
predating the 2026-06-26 in-sample-ε correction, the same class of artifact your
audit already withdrew elsewhere. Under the corrected rule the repl seeds give
no-harm-ties-freeze, matching the corrected primary verdict. Recommend marking that
holdout_score.json SUPERSEDED (in-sample-ε era) in the ledger, exactly as was done
for the withdrawn OH/iWildCam wins. This analysis also supplies what the Phase 4–5
evidence matrix asked for on this track: a traced, raw-record OOF artifact.

## iWildCam H v2 — dev-screen stability confirmed; held-out blocked by iCloud ⏸

Dev screen (idval records, tent_episodic, cal seed 0 → eval seed 1, corrected LOO
radius): KGA 0.0208 = ties freeze 0.0208, beats always-adapt 0.0988, FA 0.000.
Decision-seed sweep: no-harm on **16/16**, FA max 0.000, regret flat at 0.0208.
(The stored `dev_rows` in `protocol_result.json` differ — they carry the
"reporting-only copy" of the in-sample bug your status ledger already documents for
`run_protocol_dev_lock`; the corrected replay behavior is the no-harm one.)

The held-out sweep (cal 0 → test 1 on `iwildcam_full_test/result_e40faf29.json`)
is ready to run (`run_iwildcam_heldout_seed_sweep.py`) but that file is an iCloud
placeholder on this Mac — **Finder → Download Now**, then run the script.
Full stream-seed parity (seeds 2–4) needs the optional GPU arm in the runsheet.

## Bottom line

- **RxRx1: seed-depth objection closed** (3 model seeds, exact replays, all no-harm).
- **Office-Home: seed-depth objection substantially closed** (5 stream seeds across
  primary + replication, all no-harm at FA 0), with one superseded-era artifact
  flagged for the ledger.
- **iWildCam: decision-layer stability shown; held-out one download away; stream-seed
  parity needs one GPU run** (runsheet Step 3).
- Nothing here is promoted yet: lock `NATURAL_MULTISEED_REPLAY_v1` first (its
  criteria and exact tier-label changes are pre-stated), then apply.
