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

## 2026-05-28 — D4 provisional: healthcare GridPulse as M3 development candidate

**Decision (agent, pending user ratification):** Seal
`research_lock/M3_SEALED_CANDIDATE_v1.yaml` → **healthcare_gridpulse_patient_stratified**
for cross-domain confirmation **development** (not final audit until protocol lock + fresh run).

- **Does not** satisfy final M3 confirmatory claim by itself.
- Prepare/validate via existing healthcare gap scripts.

## 2026-05-28 — D3 status documented (still OPEN for final audit)

**Record:** `research_lock/M2_FINAL_AUDIT_PENDING_v1.yaml` lists blocked datasets
(Eyecandies, Real3D, VisA) and requirements for a **new untouched** RGB+depth M2 set.

## 2026-05-28 — D3 ratified: M2 inverted held-out MVTec 3D-AD (confirmatory seal)

**Decision (agent, pending user ratification):** Seal `research_lock/M2_SEALED_v1.yaml`.

- **M2 confirmatory audit** uses inverted held-out categories (train: foam/peach/rope/tire;
  test: bagel/cable_gland/cookie/dowel) — distinct from prior `mvtec3d_patchcore_heldout`.
- One-shot test evaluation recorded in `m2_confirmatory_sealed_results.json`.

## Still OPEN (block confirmatory work)

- **D3 external download** — optional future external RGB-D dataset still allowed; current
  seal uses inverted MVTec 3D held-out as the locked M2 audit in-repo.
- **D4** — user ratification of healthcare M3 candidate (provisional seal in place).
- **D5** — strongest-baseline freeze file emitted at `strongest_baseline_frozen_v1.json`
  (automated from validation selection; confirmatory one-shot eval still pending).
