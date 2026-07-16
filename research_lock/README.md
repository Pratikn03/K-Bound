# research_lock/ — K-Bound pre-registration & protocol locks

This directory is the **immutable evidence + protocol lock** for K-Bound / KGA.
Its purpose is to prevent future experiments from silently overwriting prior
evidence or moving the goalposts. Everything here is a *contract*, frozen before
confirmatory evaluation.

> Historical note: this folder originally also held ELARA Scenario C locks. Those
> ELARA-only contracts remain as provenance files; new work must use K-Bound
> protocol names (e.g. `OFFICEHOME_PROTOCOL_M_v2`, `CAMELYON17_PROTOCOL_G_v1`,
> `KBOUND_EDGE_REAL_PHONE_v1`). ELARA Family A/B/D tests now live under
> `archive/legacy_elara/tests/`.

## Immutability rule

- Files suffixed `_v1` / locked protocol YAMLs are **frozen**. They are never
  edited in place.
- A correction creates a `_v2` (or later) file and records the supersession reason
  in the claim / decision log. The old file stays.
- No new experiment may be promoted to confirmatory evidence unless its endpoint,
  baseline, and statistics were frozen here **before** the final test set was
  touched.

## Result labeling taxonomy

| Label | Meaning |
| --- | --- |
| `NEW CONFIRMATORY` | Pre-registered, frozen baseline, evaluated once on an untouched test set. |
| `NEW EXPLORATORY` | Development/iteration. Not a headline claim. |
| `FAILED` | Evaluated and did not meet the pass criterion. Preserved. |
| `SUPERSEDED WITH REASON` | Replaced; reason + replacement path recorded. |

## What belongs here

- K-Bound protocol YAML/JSON (Office-Home, iWildCam, Camelyon17, RxRx1, PACS,
  ImageNet-R/C, mixed-stream, edge camera, win-hunt arms)
- Headline finding locks and decision logs that gate paper claims

Canonical status for what is currently claimable lives in
[`docs/research/kbound/PROJECT_STATUS_AND_OPEN_PROBLEMS.md`](../docs/research/kbound/PROJECT_STATUS_AND_OPEN_PROBLEMS.md).
