# Runsheet — natural-shift multi-seed replays (NATURAL_MULTISEED_REPLAY_v1 DRAFT)

Goal: retire the "single-run" tier on iWildCam H v2, Office-Home M v2, RxRx1 J.
Everything in Step 1–2 is CPU-only re-scoring of existing locked records (minutes).
Step 3 is the optional GPU arm that earns full stream-seed parity for iWildCam.

## Step 0 — one-time iCloud downloads (Finder)

Four record files are iCloud placeholders on this Mac. In Finder, right-click each →
**Download Now** (or download the whole `experiments/kbound/results` folder):

- `experiments/kbound/results/officehome_full_targetval/result_target_val_361a1e8c.json`
- `experiments/kbound/results/officehome_full_targettest/result_target_test_6605675d.json`
- `experiments/kbound/results/iwildcam_full_test/result_e40faf29.json`
- `experiments/kbound/results/officehome_protocol_m_repl.log` (optional, provenance only)

Without Step 0 the aggregator still runs — it just skips Office-Home seeds 0–1 and the
iWildCam held-out sweep, and says so in its output.

## Step 1 — lock the protocol

Review `research_lock/NATURAL_MULTISEED_REPLAY_v1_DRAFT.md`, adjust if needed, then
commit it with a lock date (rename `_DRAFT` away). Success criteria and tier-label
changes are pre-stated there; no result may adjust them afterward.

## Step 2 — run Part A (CPU, ~5–15 min)

```bash
cd <repo-root>
python3 docs/research/kbound/scripts/natural_multiseed_aggregate.py \
    --out experiments/kbound/results/natural_seed_robustness_v1
```

Outputs: `natural_seed_robustness_v1.{json,md}` — per-seed no-harm table (Camelyon
`tab:multiseed` format), decision-seed sweeps, replay-vs-locked checks.

What Part A alone already earns (if criteria pass):
- RxRx1 J tier → "locked (3 model seeds, re-scored locked records)"
- Office-Home M v2 tier → "(5 stream seeds: 0–1 primary + 2–4 replication)"
  (requires Step 0 for seeds 0–1; else "3 fresh stream seeds (replication lock)")
- iWildCam H v2 → held-out seed-split confirmed + decision-seed stability
  (still one stream-seed pair until Step 3; label stays honest about that)

## Step 3 — optional GPU arm (earns iWildCam stream-seed parity)

Re-generate iWildCam test records for stream seeds 2–4 with the SAME checkpoint,
conditions, and adapter panel as `iwildcam_full_test` (see that run's own
`run_test.log` for the original command line and runtime on this Mac), e.g. one seed
at a time. Then score each seed under the locked H v2 config with the aggregator
(it picks up new seeds automatically from the records' `seed` field).

RxRx1 model seeds 3–4 (two more source checkpoints) are the expensive optional arm —
defer unless 5-model-seed parity is explicitly wanted.

## Step 4 — paper updates (only after criteria pass, per the locked protocol)

- Add a `tab:multiseed-natural` table next to the Camelyon multi-seed table with the
  per-seed rows from `natural_seed_robustness_v1.md`.
- Update the three evidence-tier cells in `tab:uniform-panel` exactly as pre-stated
  in the protocol. The promoted numeric rows do NOT change.
- Update `SUBMISSION_LEDGER.md` §3 tier column + `claim_ledger.json` accordingly.
