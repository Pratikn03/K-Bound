# Lever 4 — Reproduction & Prospective/Temporal Eval: Honest Scoping

Date: 2026-06-02
Status: **scoping only.** Two of the three Lever-4 components require external
resources (a prospective temporal dataset; an independent reproducer / APFS
volume) that cannot be manufactured in-session. This document states exactly
what is done, what is blocked, and on what.

## Component status

| Component | Status | Blocker |
|---|---|---|
| Dependency lockfile + one-command rebuild | ✅ present | `requirements.lock.txt` (3.6 KB), pinned; `pip check` clean per gap audit |
| Test suite | ✅ present | ~765 collected per prior audits; targeted research-integrity + API suites pass |
| Clean-checkout / APFS reproduction | ❌ blocked | repo lives on an **exfat** external volume (confirmed via `mount`), the operational risk flagged in `RESEARCH_OVERVIEW_AND_RATING.md`; needs an APFS clone + clean-clone test run |
| Prospective / temporal (M4) validation | ❌ blocked | `research_lock/M4_TEMPORAL_STREAM_PROTOCOL_v1.yaml` is `PROTOCOL_SCAFFOLD`, stream `NOT_SELECTED`; the prospective industrial RGB+depth line-scan is **not acquired** |
| Independent reproduction | ❌ blocked | requires an external party to rerun; out of scope for a single working session |

## Why these cannot be faked

- **M4 prospective stream.** The protocol explicitly lists the only candidates as
  (a) `elara_bench_la` — a label-aligned mechanism stream marked *development only
  until natural temporal pairing is acquired*, and (b) a `prospective_industrial_stream`
  that is *not acquired*. There is no real temporal RGB+depth deployment stream on
  disk. A prospective/temporal claim therefore cannot be made without acquiring
  one; asserting it would be exactly the overclaim the research lock forbids.
- **Independent reproduction & APFS clone.** Both are environment/process actions,
  not computations. The honest deliverable here is the *procedure* and the
  *risk statement*, not a fabricated "reproduced" stamp.

## What IS actionable now (and the procedure)

1. **APFS clean clone (operational de-risk).** Copy the repo to an APFS volume,
   `git clean`-equivalent purge of `._*` AppleDouble sidecars and caches
   (`find . -name '._*' -not -path './.git/*' -delete`), then `pip install -r
   requirements.lock.txt` and run the full suite from the clean checkout. This is
   the single highest-value reproduction step and removes the exfat instability
   risk. It needs a target APFS volume to be designated.
2. **Reproduction manifest.** Record exact commands, lockfile hash, and dataset
   hashes (`research_lock/frozen_test_sets_v*.yaml`, generated split hashes) so a
   third party can rebuild. Most pieces exist; they should be collected into one
   `REPRODUCE.md`.

## Honest verdict for Lever 4

Lever 4 is **not completable in this session.** The reproducible-build skeleton
(lockfile, tests, hashes) is in place, but the two evidence-bearing parts —
prospective/temporal validation and independent reproduction — are gated on
acquiring a temporal dataset and on an external environment, respectively. These
are real, named gaps, not effort gaps. Recommended next concrete action: nominate
an APFS target volume so the clean-clone reproduction can be executed and a
`REPRODUCE.md` manifest cut.
