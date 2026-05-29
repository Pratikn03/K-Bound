# DECISIONS_v1.md (append-only ratification log)

Records ratified Scenario C decisions. Append-only: never edit prior entries;
add new dated entries. Supersedes the "default until ratified" placeholders in
the `_v1` registries where noted.

---

## 2026-05-28 — D1 ratified: Policy B (reclassify Eyecandies to development)

**Decision (user):** "fix it if you can" → adopt **Policy B**.

- Eyecandies is **reclassified from sealed-FAILED to DEVELOPMENT** so the
  calibration-transfer failure can be studied and an improved transfer method
  developed on it.
- **Mandatory consequences (binding):**
  1. A **new, untouched** naturally paired RGB+depth dataset MUST be acquired
     and sealed as the **final** M2 transfer audit (decision D3). The final
     held-out transfer claim (P4) may ONLY be made on that new dataset, never on
     Eyecandies.
  2. The manuscript MUST **explicitly disclose** that Eyecandies changed from
     confirmation to development *after* the initial failed result.
  3. The original failure remains permanently recorded in
     `family_d_failure_record.md` (still `FAILED`, never deleted).
- **Supersedes:** the Policy-A default in `dataset_registry_v1.yaml`
  (eyecandies.role) and `frozen_test_sets_v1.yaml` (eyecandies_v3_heldout).
  New effective role: `development`.
- **Status of P4 (held-out transfer):** remains `NOT_CONFIRMED` until the new
  untouched M2 dataset passes the frozen protocol.

## 2026-05-28 — D2 ratified: full Scenario C central claim is the target

**Decision (user):** target the **full Scenario C claim**:

> *ELARA is a validated reliability-aware multimodal anomaly-fusion framework
> that improves over strong frozen baselines under clean, degraded, and
> transferred conditions across naturally co-observed domains, with formal
> switching conditions and deployment-style monitoring.*

- This is the **target**, NOT a currently supported claim. It becomes usable
  only when pillars P1–P6 pass under `NEW CONFIRMATORY` results.
- Until then, the only admissible claim is the current-manuscript claim in
  `BASELINE_STATE_v1.md` ("bounded reliability-stress gains; held-out transfer
  unresolved").

## Still OPEN (block confirmatory work)

- **D3** — select + seal the new untouched M2 RGB+depth transfer dataset
  (now REQUIRED by D1/Policy B).
- **D4** — select the non-vision naturally co-observed domain (M3).
- **D5** — freeze the strongest-baseline family (Phase 5).
