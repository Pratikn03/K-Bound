# research_lock/ — ELARA Scenario C Research Contract (v1)

This directory is the **immutable evidence + protocol lock** for the ELARA
Scenario C program. Its purpose is to prevent future experiments from silently
overwriting prior evidence or moving the goalposts. Everything here is a
*contract*, frozen before any new confirmatory experiment is evaluated.

> Scope note: this lock **consolidates and references** existing authoritative
> artifacts already in the repo (`docs/research/phase2/`, `docs/research/phase3/`,
> `docs/research/audit/`, `experiments/phase2/`). It does not re-derive or
> restate any number that is not already audited there. Where a value appears,
> its authoritative source path is cited.

## Immutability rule

- Files suffixed `_v1` are **frozen**. They are never edited in place.
- A correction or update creates a `_v2` file and adds a `SUPERSEDED WITH
  REASON` row in `claim_matrix_v1.csv` (or its successor), pointing to the new
  file. The old file stays.
- No new experiment may be promoted to confirmatory evidence unless its
  endpoint, dataset role, baseline, and statistics were frozen here **before**
  the final test set was touched.

## Result labeling taxonomy (mandatory for every new result)

Every new result added to the program MUST carry exactly one label:

| Label | Meaning |
| --- | --- |
| `NEW CONFIRMATORY` | Pre-registered endpoint, frozen baseline, evaluated once on an untouched test set, passed the frozen statistical policy. |
| `NEW EXPLORATORY` | Development/iteration result. Informative, but NOT admissible as a headline claim. |
| `FAILED` | Endpoint evaluated and did not meet the pre-registered pass criterion. Preserved, never deleted. |
| `SUPERSEDED WITH REASON` | Replaced by a later result; the reason and replacement path are recorded. The original remains. |

## Base RGA vs RGA+ separation rule (non-negotiable)

No figure, table, sentence, or claim may attribute an RGA+ (supervised
reliability-feature) gain to **base RGA** (the reliability gate mechanism), or
vice versa. Each experiment maps to exactly one component:

| Component | What it must prove |
| --- | --- |
| Base RGA | The reliability gate improves behavior under degradation and stays quiet on clean evidence. |
| RGA+ | Reliability-derived features improve supervised fusion performance. |
| Router | Validation-selected model selection improves outcomes without test leakage. |
| Monitor / Certificate | Deployment-style switching is justified, or the system should abstain. |

## Index

| File | Role |
| --- | --- |
| `SCENARIO_C_CLAIM_CONTRACT.md` | Definition of Done: six claim pillars, three readiness tiers, forbidden language, central-claim slot. |
| `BASELINE_STATE_v1.md` | Frozen snapshot of current Family A / B / D evidence with labels. |
| `claim_matrix_v1.csv` | Every current claim mapped to pillar + component + status + authoritative artifact + label. |
| `dataset_registry_v1.yaml` | Datasets classified by pairing strength and M0–M4 role (development / confirmation / final-audit). |
| `protocol_registry_v1.yaml` | Evaluation protocols and their authoritative specs. |
| `model_registry_v1.yaml` | Components, gates (G0–G3), and promotion status. |
| `primary_endpoints_v1.yaml` | Pre-registered primary metric for each benchmark family. |
| `frozen_test_sets_v1.yaml` | Which test sets are sealed, development, or final-unseen audit. |
| `statistical_policy_v1.md` | Bootstrap / DeLong / Holm policy and the freeze rule (references existing policy docs). |
| `family_d_failure_record.md` | Sealed record of the Eyecandies held-out transfer failure. |

## Open decisions (block confirmatory work; see contract §Decisions)

1. **Eyecandies policy**: keep as sealed FAILED external test (Policy A) vs.
   reclassify as development with a new untouched final transfer dataset
   (Policy B). Until decided, Eyecandies is treated as **sealed FAILED**.
2. **Central claim wording** (one sentence) — slot left in the contract.
3. **New untouched RGB+depth transfer dataset** (T2) — not yet selected.
4. **Non-vision naturally co-observed domain** (M3) — not yet selected.
